"""GRPO training entry point (single- and multi-agent).

    python -m train.run_train train/configs/multi.yaml
    python -m train.run_train train/configs/single_step_matched.yaml --set lr=2e-5
    python -m train.run_train train/configs/multi.yaml --smoke        # 2 tiny steps
    python -m train.run_train train/configs/multi.yaml --max_steps 5 --output_root /drive/runs \
        --micro_batch_size 4 --allow_fp16 --set eval_every=0 --set save_every=0   # timing

Each step: B problems from train_stream (in order, step k uses positions [k*B, (k+1)*B)),
rollouts with vLLM, rewards and group-normalized advantages, one optimizer update, then
the adapter(s) are saved to runs/<run_name>/tmp/step_<k>/<adapter> and re-registered in
vLLM under new ids. Outputs in runs/<run_name>/:
    config.json        resolved config, git commit, GPU, library versions
    train_log.jsonl    one line per step (+ a step-0 line with the dev accuracy before training)
    dev_eval/          dev predictions at each evaluation (run_eval JSONL format)
    checkpoints/step_<k>/<adapter>   every save_every steps and at the end
eval_every: 0 / save_every: 0 turn evaluation / checkpoints off (e.g. for timing runs).

Per-step cost in train_log.jsonl: step_seconds (whole step, excluding dev eval and
checkpoints), step_gen_seconds (rollouts), step_train_seconds (log-probs, backward,
optimizer), step_sync_seconds (saving adapters for vLLM), step_gen_tokens,
step_prefill_tokens, peak_mem_gb (torch.cuda.max_memory_allocated of this process, reset
at the start of every step; vLLM's engine process is not included) and device_mem_used_gb
(whole GPU at the end of the step, vLLM included).

vLLM is created before the training model so its memory share (gpu_memory_utilization)
is reserved first. bf16 needs an L4/A100-class GPU; on a T4 pass --allow_fp16 to fall
back to float16 (fine for timing, not recommended for real training runs).
"""
import argparse
import copy
import json
import os
import platform
import random
import shutil
import subprocess
import time
from dataclasses import asdict

import yaml

from common.cost import Cost, Timer
from common.data import records, train_stream
from train.grpo import assign_rewards, compute_advantages, make_optimizer, step_stats, train_step
from train.rollout import VLLMEngine, evaluate_dev, rollout_multi, rollout_single

DEFAULTS = {
    "run_name": None,
    "mode": "single",                 # single | multi
    "model": "Qwen/Qwen2.5-1.5B-Instruct",
    "dtype": "bfloat16",
    "seed": 0,
    "steps": 200,
    "batch_problems": 16,             # B
    "group_size": 8,                  # G
    "n_rounds": 1,                    # multi: solver -> reviser rounds
    "shared_adapter": False,          # multi ablation: reviser uses the solver adapter
    "reward_mode": "team",            # team | per_agent
    "reward_require_boxed": True,
    "beta": 0.0,                      # KL (k3) coefficient against the base model
    "loss_agg": "token_mean",
    "lr": 1.0e-5,
    "warmup_steps": 10,
    "weight_decay": 0.0,
    "max_grad_norm": 1.0,
    "micro_batch_tokens": 4096,       # padded tokens per micro-batch (gradient accumulation)
    "micro_batch_size": None,         # optional cap on sequences per micro-batch
    "allow_fp16": False,              # fall back to float16 on GPUs without bf16 (T4)
    "gradient_checkpointing": True,
    "lora": {"r": 16, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]},
    "temperature": 1.0,
    "top_p": 1.0,
    "max_new_tokens": 1024,
    "gpu_memory_utilization": 0.35,
    "max_model_len": 4096,
    "eval_every": 25,
    "eval_dev_size": 200,
    "save_every": 50,
    "output_dir": "runs",
}

SMOKE = {"steps": 2, "batch_problems": 2, "group_size": 2, "max_new_tokens": 128,
         "eval_every": 1, "eval_dev_size": 4, "save_every": 1, "warmup_steps": 1}


def parse_value(text):
    """YAML value of a --set override; '2e-5' is a float here (PyYAML reads it as a string)."""
    value = yaml.safe_load(text)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def load_config(path, overrides=(), smoke=False):
    cfg = copy.deepcopy(DEFAULTS)
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cli = [(k, parse_value(v)) for k, v in (kv.split("=", 1) for kv in overrides)]
    for key, value in list(user.items()) + cli:
        if key.startswith("lora."):
            cfg["lora"][key[5:]] = value
        elif key == "lora":
            cfg["lora"].update(value)
        elif key in cfg:
            cfg[key] = value
        else:
            raise KeyError(f"unknown config key {key!r}")
    if smoke:
        cfg.update(SMOKE)
        cfg["run_name"] = f"{cfg['run_name']}_smoke"
    if cfg["mode"] not in ("single", "multi"):
        raise ValueError(f"mode must be single or multi, not {cfg['mode']!r}")
    if cfg["mode"] == "single" and cfg["n_rounds"] != 1:
        raise ValueError("n_rounds applies to multi mode only")
    if not cfg["run_name"]:
        raise ValueError("run_name is required")
    return cfg


def resolve_dtype(cfg):
    """bfloat16 needs compute capability >= 8.0 (L4, A100). torch.cuda.is_bf16_supported()
    can report True on a T4 through emulation, so check the capability directly."""
    import torch

    if cfg["dtype"] != "bfloat16" or not torch.cuda.is_available():
        return cfg["dtype"]
    if torch.cuda.get_device_capability(0)[0] >= 8:
        return "bfloat16"
    gpu = torch.cuda.get_device_name(0)
    if not cfg["allow_fp16"]:
        raise SystemExit(f"{gpu} does not support bfloat16; use an L4/A100 or pass --allow_fp16")
    print(f"WARNING: {gpu} does not support bfloat16; using float16 (--allow_fp16)")
    return "float16"


def gpu_memory():
    """(peak allocated by this process since the last reset, used on the whole GPU), in GB."""
    import torch

    if not torch.cuda.is_available():
        return None, None
    free, total = torch.cuda.mem_get_info()
    return torch.cuda.max_memory_allocated() / 1e9, (total - free) / 1e9


def adapters_for(cfg):
    """Adapter names to train, and which adapter each role uses."""
    if cfg["mode"] == "multi" and not cfg["shared_adapter"]:
        return ["solver", "reviser"], {"solver": "solver", "reviser": "reviser"}
    return ["solver"], {"solver": "solver", "reviser": "solver"}


def build_policy(cfg, device):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=getattr(torch, cfg["dtype"]))
    lc = LoraConfig(r=cfg["lora"]["r"], lora_alpha=cfg["lora"]["alpha"],
                    lora_dropout=cfg["lora"]["dropout"],
                    target_modules=list(cfg["lora"]["target_modules"]), task_type="CAUSAL_LM")
    names, _ = adapters_for(cfg)
    model = get_peft_model(model, lc, adapter_name=names[0])
    for name in names[1:]:
        model.add_adapter(name, lc)
    for n, p in model.named_parameters():   # adapters train in fp32 on the bf16 base
        if "lora_" in n:
            p.data = p.data.float()
            p.requires_grad_(True)
    if cfg["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    return model.to(device)


def role_setter(model, role_to_adapter):
    """Activate the role's adapter. PEFT's set_adapter freezes the other adapters; turn
    requires_grad back on so both keep the gradients they accumulate this step."""
    state = {"active": None}

    def set_role(role):
        name = role_to_adapter[role]
        if state["active"] != name:
            model.set_adapter(name)
            for n, p in model.named_parameters():
                if "lora_" in n:
                    p.requires_grad_(True)
            state["active"] = name
    return set_role


def save_adapters(model, names, directory):
    """Save to directory/<adapter>/ (PEFT's layout for named adapters); return name -> path."""
    model.save_pretrained(directory, selected_adapters=names)
    paths = {n: os.path.join(directory, n) for n in names}
    for n, path in paths.items():
        if not os.path.exists(os.path.join(path, "adapter_config.json")):
            raise FileNotFoundError(f"adapter {n!r} not saved at {path}")
    return paths


def environment():
    import torch

    info = {"python": platform.python_version(), "gpu": None}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    for lib in ("torch", "transformers", "peft", "vllm"):
        try:
            info[lib] = __import__(lib).__version__
        except Exception:
            info[lib] = None
    try:
        info["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                                     stderr=subprocess.DEVNULL).strip()
    except Exception:
        info["git_commit"] = None
    return info


def write_dev_eval(run_dir, step, rows):
    os.makedirs(os.path.join(run_dir, "dev_eval"), exist_ok=True)
    with open(os.path.join(run_dir, "dev_eval", f"step_{step}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config key (lora.r=8 for LoRA keys)")
    p.add_argument("--smoke", action="store_true", help="2 steps, B=2, G=2, short generations")
    p.add_argument("--max_steps", type=int, help="override steps")
    p.add_argument("--output_root", help="override output_dir (runs go to <output_root>/<run_name>)")
    p.add_argument("--micro_batch_size", type=int,
                   help="max sequences per micro-batch (gradient accumulation; same optimizer steps)")
    p.add_argument("--allow_fp16", action="store_true", help="use float16 on GPUs without bf16")
    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set, args.smoke)
    for key, value in (("steps", args.max_steps), ("output_dir", args.output_root),
                       ("micro_batch_size", args.micro_batch_size)):
        if value is not None:
            cfg[key] = value
    cfg["allow_fp16"] = cfg["allow_fp16"] or args.allow_fp16

    import torch
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg["dtype"] = resolve_dtype(cfg)
    run_dir = os.path.join(cfg["output_dir"], cfg["run_name"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**cfg, **environment()}, f, indent=2)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    engine = VLLMEngine(cfg, tok)            # first: reserves its share of GPU memory
    model = build_policy(cfg, device)
    names, role_to_adapter = adapters_for(cfg)
    set_role = role_setter(model, role_to_adapter)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    optimizer, scheduler = make_optimizer(params, cfg["lr"], cfg["weight_decay"], cfg["warmup_steps"])

    tmp_dir = os.path.join(run_dir, "tmp")

    def sync(step):
        """Save the current adapters and point vLLM at them under new ids."""
        paths = save_adapters(model, names, os.path.join(tmp_dir, f"step_{step}"))
        engine.set_adapters({role: paths[a] for role, a in role_to_adapter.items()})
        old = os.path.join(tmp_dir, f"step_{step - 2}")   # keep the last two in case vLLM is loading
        if os.path.isdir(old):
            shutil.rmtree(old)

    def make_params(n, step):
        return SamplingParams(n=n, temperature=cfg["temperature"], top_p=cfg["top_p"],
                              max_tokens=cfg["max_new_tokens"], seed=cfg["seed"] * 1_000_003 + step)

    dev = records("dev")[:cfg["eval_dev_size"]]
    cost = Cost()                             # cumulative TRAINING cost (eval not included)
    log_path = os.path.join(run_dir, "train_log.jsonl")
    open(log_path, "w").close()

    def log(record):
        with open(log_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(json.dumps({k: v for k, v in record.items() if k not in ("qids", "pos")}))

    def dev_eval(step):
        model.eval()
        rows, eval_cost = evaluate_dev(engine, dev, cfg["mode"], cfg["n_rounds"],
                                       run_id=f"{cfg['run_name']}_dev_step{step}")
        model.train()
        write_dev_eval(run_dir, step, rows)
        return {"dev_acc": sum(r["final_correct"] for r in rows) / len(rows), "dev_n": len(rows),
                "dev_at_gen_tokens": cost.gen_tokens, "dev_eval_gen_tokens": eval_cost.gen_tokens}

    sync(0)
    if cfg["eval_every"]:
        log({"step": 0, **dev_eval(0), **asdict(cost)})
    model.train()

    for step in range(1, cfg["steps"] + 1):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_step, before = time.perf_counter(), asdict(cost)
        problems = train_stream(cfg["batch_problems"], offset=(step - 1) * cfg["batch_problems"])
        with Timer(cost, "gen_seconds"):
            if cfg["mode"] == "single":
                groups = rollout_single(engine, problems, cfg["group_size"],
                                        lambda n: make_params(n, step), cost)
            else:
                groups = rollout_multi(engine, problems, cfg["group_size"],
                                       lambda n: make_params(n, step), cost, cfg["n_rounds"])
        assign_rewards(groups, cfg["reward_mode"], cfg["reward_require_boxed"])
        compute_advantages(groups)
        turns = [t for g in groups for ro in g for t in ro.turns]
        with Timer(cost, "bwd_seconds"):
            out = train_step(model, optimizer, scheduler, turns, pad_id=tok.pad_token_id,
                             set_role=set_role, beta=cfg["beta"], loss_agg=cfg["loss_agg"],
                             micro_batch_tokens=cfg["micro_batch_tokens"],
                             micro_batch_size=cfg["micro_batch_size"],
                             max_grad_norm=cfg["max_grad_norm"], device=device)
        cost.bwd_tokens += out["bwd_tokens"]
        cost.optimizer_steps += 1
        t_sync = time.perf_counter()
        sync(step)
        now = time.perf_counter()
        peak_mem, device_mem = gpu_memory()

        record = {"step": step, **step_stats(groups), "loss": out["loss"], "kl": out["kl"],
                  "grad_norm": out["grad_norm"], "lr": scheduler.get_last_lr()[0],
                  "step_seconds": now - t_step,
                  "step_gen_seconds": cost.gen_seconds - before["gen_seconds"],
                  "step_train_seconds": cost.bwd_seconds - before["bwd_seconds"],
                  "step_sync_seconds": now - t_sync,
                  "step_gen_tokens": cost.gen_tokens - before["gen_tokens"],
                  "step_prefill_tokens": cost.prefill_tokens - before["prefill_tokens"],
                  "peak_mem_gb": peak_mem, "device_mem_used_gb": device_mem, "dtype": cfg["dtype"],
                  **asdict(cost), "qids": [p["qid"] for p in problems],
                  "pos": [p["pos"] for p in problems]}
        if cfg["eval_every"] and (step % cfg["eval_every"] == 0 or step == cfg["steps"]):
            record.update(dev_eval(step))
        log(record)
        if cfg["save_every"] and (step % cfg["save_every"] == 0 or step == cfg["steps"]):
            save_adapters(model, names, os.path.join(run_dir, "checkpoints", f"step_{step}"))

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
