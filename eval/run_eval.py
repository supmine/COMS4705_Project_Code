"""Run a training-free baseline (or a trained adapter) on GSM8K with vLLM.

Generation is staged: all solver outputs first, then all reviser prompts in one
more batch. Writes outputs/<run_id>.jsonl (one line per problem) and
outputs/<run_id>.json (config + total cost).

    python -m eval.run_eval --system cot --split test
    python -m eval.run_eval --system sc --k 5
    python -m eval.run_eval --config configs/solver_reviser_frozen.yaml
"""
import argparse
import json
import os
import subprocess
from collections import Counter
from dataclasses import asdict

from common.answer import grade, normalize
from common.cost import Cost, Timer
from common.data import records
from common.prompts import reviser_messages, solver_messages

SYSTEMS = ["cot", "sc", "solver_reviser", "self_revision"]


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML file; its keys set defaults, CLI flags override")
    p.add_argument("--system", choices=SYSTEMS)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")  # keep -Instruct
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    p.add_argument("--k", type=int, default=5, help="samples for sc")
    p.add_argument("--temperature", type=float, default=0.7, help="sc sampling")
    p.add_argument("--top_p", type=float, default=0.95, help="sc sampling")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--solver_lora", help="adapter path for the solver (and self_revision)")
    p.add_argument("--reviser_lora", help="adapter path for the reviser (solver_reviser)")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_id")
    p.add_argument("--out_dir", default="outputs")
    args = p.parse_args(argv)
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
        unknown = set(cfg) - set(vars(args))
        if unknown:
            p.error(f"unknown keys in {args.config}: {sorted(unknown)}")
        p.set_defaults(**cfg)
        args = p.parse_args(argv)
    if args.system is None:
        p.error("--system is required (flag or config)")
    if args.run_id is None:
        args.run_id = default_run_id(args)
    return args


def default_run_id(args):
    size = args.model.split("-")[-2].lower() if "-" in args.model else "model"
    name = {"sc": f"sc{args.k}"}.get(args.system, args.system)
    trained = "trained" if (args.solver_lora or args.reviser_lora) else "frozen"
    return f"{name}_{trained}_{size}_gsm8k_{args.split}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def majority_vote(preds):
    """Most common non-None prediction; ties go to the earliest sample."""
    counts = Counter(p for p in preds if p is not None)
    if not counts:
        return None
    best = max(counts.values())
    return next(p for p in preds if counts.get(p) == best)


class Runner:
    def __init__(self, args):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.args = args
        self.tok = AutoTokenizer.from_pretrained(args.model)
        use_lora = bool(args.solver_lora or args.reviser_lora)
        kw = dict(enable_lora=True, max_lora_rank=args.max_lora_rank) if use_lora else {}
        self.llm = LLM(model=args.model, seed=args.seed,
                       gpu_memory_utilization=args.gpu_memory_utilization, **kw)
        self.solver_lora = LoRARequest("solver", 1, args.solver_lora) if args.solver_lora else None
        self.reviser_lora = LoRARequest("reviser", 2, args.reviser_lora) if args.reviser_lora else None

    def to_prompt(self, messages):
        return self.tok.apply_chat_template(messages, tokenize=False,
                                            add_generation_prompt=True)

    def generate(self, messages_list, params, lora=None):
        prompts = [self.to_prompt(m) for m in messages_list]
        return self.llm.generate(prompts, params, lora_request=lora)


def make_turn(agent, out, completion, gold, prefill):
    g = grade(completion.text, gold)
    return {"agent": agent, "text": completion.text, "pred": g["pred"],
            "method": g["method"], "correct": g["correct"],
            "finish_reason": completion.finish_reason,
            "prefill_tokens": prefill, "gen_tokens": len(completion.token_ids)}


def run(args, problems, runner):
    from vllm import SamplingParams

    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    cost = Cost()
    questions = [pr["question"] for pr in problems]
    rows = [{"run_id": args.run_id, "system": args.system, "qid": pr["qid"],
             "gold": normalize(pr["gold"]), "turns": []} for pr in problems]

    if args.system == "sc":
        params = SamplingParams(n=args.k, temperature=args.temperature,
                                top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
        with Timer(cost, "gen_seconds"):
            outs = runner.generate([solver_messages(q) for q in questions], params,
                                   runner.solver_lora)
        for row, pr, out in zip(rows, problems, outs):
            cost.add_vllm_output(out)
            # The k samples share one prompt: its prefill is charged to the first sample.
            for i, c in enumerate(out.outputs):
                prefill = len(out.prompt_token_ids) if i == 0 else 0
                row["turns"].append(make_turn("sample", out, c, pr["gold"], prefill))
            row["final_pred"] = majority_vote([t["pred"] for t in row["turns"]])
            row["final_correct"] = row["final_pred"] is not None and row["final_pred"] == row["gold"]
    else:
        with Timer(cost, "gen_seconds"):
            s_outs = runner.generate([solver_messages(q) for q in questions], greedy,
                                     runner.solver_lora)
        for row, pr, out in zip(rows, problems, s_outs):
            cost.add_vllm_output(out)
            row["turns"].append(make_turn("solver", out, out.outputs[0], pr["gold"],
                                          len(out.prompt_token_ids)))

        if args.system in ("solver_reviser", "self_revision"):
            # self_revision: same reviser prompt, solver's own adapter.
            lora = runner.solver_lora if args.system == "self_revision" else runner.reviser_lora
            with Timer(cost, "gen_seconds"):
                r_outs = runner.generate([reviser_messages(q, o.outputs[0].text)
                                          for q, o in zip(questions, s_outs)], greedy, lora)
            for row, pr, out in zip(rows, problems, r_outs):
                cost.add_vllm_output(out)
                row["turns"].append(make_turn("reviser", out, out.outputs[0], pr["gold"],
                                              len(out.prompt_token_ids)))

        for row in rows:
            last = row["turns"][-1]
            row["final_pred"] = last["pred"]
            row["final_correct"] = last["correct"]

    for row in rows:
        row["total_gen_tokens"] = sum(t["gen_tokens"] for t in row["turns"])
        row["total_prefill_tokens"] = sum(t["prefill_tokens"] for t in row["turns"])
    return rows, cost


def main(argv=None):
    args = parse_args(argv)
    problems = records(args.split)
    if args.limit:
        problems = problems[:args.limit]

    runner = Runner(args)
    rows, cost = run(args, problems, runner)

    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.join(args.out_dir, args.run_id)
    with open(base + ".jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    acc = sum(r["final_correct"] for r in rows) / len(rows)
    meta = {"config": vars(args), "git_commit": git_commit(), "n": len(rows),
            "accuracy": acc, "cost": asdict(cost)}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{args.run_id}: accuracy {acc:.4f} on {len(rows)} problems -> {base}.jsonl")


if __name__ == "__main__":
    main()
