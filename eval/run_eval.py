"""Shared GSM8K / MATH (levels 3-5) evaluation for the four baselines, with vLLM.

| # | name                         | weights                      | pipeline         |
|---|------------------------------|------------------------------|------------------|
| 1 | zero-shot CoT                | base model                   | solver           |
| 2 | frozen solver-reviser        | base model                   | solver -> reviser|
| 3 | single-agent GRPO            | --adapter                    | solver           |
| 4 | single-agent GRPO + self-rev | --adapter (both steps)       | solver -> reviser|

Generation is staged: all solver outputs first, then all reviser prompts in one
more batch. Writes the run directory outputs/<run_id>/ with predictions.jsonl
(one line per problem) and config.json (config, system, group, total cost).

    python -m eval.run_eval --baseline 1 --split dev --limit 20     # smoke test
    python -m eval.run_eval --baseline 2
    python -m eval.run_eval --baseline 4 --adapter checkpoints/single_grpo
    python -m eval.run_eval --config configs/baseline1.yaml --dtype half   # T4
    python -m eval.run_eval --baseline 1 --dataset math --split dev --limit 20

Max new tokens are fixed per dataset (GSM8K 512, MATH 2048) so every baseline
on a dataset gets the same budget; --max_tokens overrides with a warning.
"""
import argparse
import json
import os
import subprocess
from dataclasses import asdict

from common.answer import grade, normalize
from common.cost import Cost, Timer
from common.data import DATASET_INFO, level_counts, math_records, records
from common.prompts import render, reviser_messages, solver_messages
from eval.metrics import level_summary

BASELINES = {
    1: {"name": "cot", "group": "frozen", "adapter": False, "revise": False},
    2: {"name": "solver_reviser", "group": "frozen", "adapter": False, "revise": True},
    3: {"name": "grpo", "group": "single_agent", "adapter": True, "revise": False},
    4: {"name": "grpo_self_revision", "group": "single_agent", "adapter": True, "revise": True},
}
MAX_TOKENS = {"gsm8k": 512, "math": 2048}  # GSM8K 512 matches runs already made


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", help="YAML file; its keys set defaults, CLI flags override")
    p.add_argument("--baseline", type=int, choices=sorted(BASELINES))
    p.add_argument("--adapter", help="LoRA path; required for baselines 3 and 4")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")  # keep -Instruct
    p.add_argument("--dataset", default="gsm8k", choices=sorted(MAX_TOKENS))
    p.add_argument("--levels", default="3,4,5", help="MATH levels, comma-separated (MATH only)")
    p.add_argument("--split", default="test", choices=["train", "dev", "test"])
    p.add_argument("--limit", type=int, help="first N problems only (smoke tests)")
    p.add_argument("--max_tokens", type=int,
                   help="default: 512 for GSM8K, 2048 for MATH; keep the default for reported runs")
    p.add_argument("--max_lora_rank", type=int, default=64)
    p.add_argument("--dtype", default="auto", help="'half' on GPUs without bf16 (T4)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0,
                   help="for trained adapters, pass the training seed so plots can average seeds")
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
    if args.baseline is None:
        p.error("--baseline is required (flag or config)")
    spec = BASELINES[args.baseline]
    if spec["adapter"] and not args.adapter:
        p.error(f"baseline {args.baseline} requires --adapter")
    if not spec["adapter"] and args.adapter:
        p.error(f"baseline {args.baseline} runs the base model; drop --adapter")
    args.levels = sorted(int(x) for x in str(args.levels).replace(" ", "").strip("[]").split(","))
    if args.max_tokens is None:
        args.max_tokens = MAX_TOKENS[args.dataset]
    elif args.max_tokens != MAX_TOKENS[args.dataset]:
        print(f"WARNING: max_tokens={args.max_tokens} differs from the {args.dataset} default "
              f"{MAX_TOKENS[args.dataset]}; results are not comparable with other baselines.")
    if args.run_id is None:
        args.run_id = default_run_id(args)
    return args


def default_run_id(args):
    data = "" if args.dataset == "gsm8k" else f"{args.dataset}_"
    run_id = f"b{args.baseline}_{data}{args.split}"
    return run_id if args.seed == 0 else f"{run_id}_seed{args.seed}"


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


class Runner:
    def __init__(self, args):
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.tok = AutoTokenizer.from_pretrained(args.model)
        kw = dict(enable_lora=True, max_lora_rank=args.max_lora_rank) if args.adapter else {}
        self.llm = LLM(model=args.model, seed=args.seed, dtype=args.dtype,
                       gpu_memory_utilization=args.gpu_memory_utilization, **kw)
        # Baseline 4 uses this same adapter for the reviser step.
        self.lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None

    def to_prompt(self, messages):
        return render(self.tok, messages)

    def generate(self, messages_list, params):
        prompts = [self.to_prompt(m) for m in messages_list]
        return self.llm.generate(prompts, params, lora_request=self.lora)


def make_turn(agent, out, gold, dataset):
    c = out.outputs[0]
    g = grade(c.text, gold, dataset=dataset)
    turn = {"agent": agent, "text": c.text, "pred": g["pred"],
            "method": g["method"], "correct": g["correct"],
            "finish_reason": c.finish_reason,
            "prefill_tokens": len(out.prompt_token_ids), "gen_tokens": len(c.token_ids)}
    if dataset == "math":  # raw extracted answer + which path (string/symbolic/...) decided it
        turn.update(pred_raw=g["pred_raw"], decided_by=g["decided_by"])
    return turn


def run(args, problems, runner):
    from vllm import SamplingParams

    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    spec = BASELINES[args.baseline]
    cost = Cost()
    questions = [pr["question"] for pr in problems]
    rows = [{"run_id": args.run_id, "baseline": args.baseline, "system": spec["name"],
             "dataset": args.dataset, "qid": pr["qid"], "level": pr.get("level"),
             "subject": pr.get("subject"),
             "gold": normalize(pr["gold"]) if args.dataset == "gsm8k" else pr["gold"],
             "turns": []} for pr in problems]

    with Timer(cost, "gen_seconds"):
        s_outs = runner.generate([solver_messages(q) for q in questions], greedy)
    for row, pr, out in zip(rows, problems, s_outs):
        cost.add_vllm_output(out)
        row["turns"].append(make_turn("solver", out, pr["gold"], args.dataset))

    if spec["revise"]:
        with Timer(cost, "gen_seconds"):
            r_outs = runner.generate([reviser_messages(q, o.outputs[0].text)
                                      for q, o in zip(questions, s_outs)], greedy)
        for row, pr, out in zip(rows, problems, r_outs):
            cost.add_vllm_output(out)
            row["turns"].append(make_turn("reviser", out, pr["gold"], args.dataset))

    for row in rows:
        last = row["turns"][-1]
        row["final_pred"] = last["pred"]
        row["final_correct"] = last["correct"]
        row["total_gen_tokens"] = sum(t["gen_tokens"] for t in row["turns"])
        row["total_prefill_tokens"] = sum(t["prefill_tokens"] for t in row["turns"])
    return rows, cost


def load_problems(args):
    if args.dataset == "gsm8k":
        return records(args.split)
    for split in ("dev", "test"):
        print(f"MATH {split}: {level_counts(math_records(split, args.levels))} problems per level")
    return math_records(args.split, args.levels)


def print_level_summary(rows):
    for level, s in level_summary(rows).items():
        print(f"  level {level}: accuracy {s['acc']:.4f} "
              f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] (n={s['n']})")


def main(argv=None):
    args = parse_args(argv)
    problems = load_problems(args)
    if args.limit:
        problems = problems[:args.limit]

    runner = Runner(args)
    print("=== rendered solver prompt (problem 0) ===")
    print(runner.to_prompt(solver_messages(problems[0]["question"])))
    print("==========================================")

    rows, cost = run(args, problems, runner)

    run_dir = os.path.join(args.out_dir, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "predictions.jsonl"), "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    acc = sum(r["final_correct"] for r in rows) / len(rows)
    spec = BASELINES[args.baseline]
    meta = {**vars(args), **DATASET_INFO[args.dataset], "system": spec["name"],
            "group": spec["group"], "git_commit": git_commit(), "n": len(rows), "accuracy": acc,
            "cost": asdict(cost)}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{args.run_id}: accuracy {acc:.4f} on {len(rows)} problems -> {run_dir}/")
    print_level_summary(rows)


if __name__ == "__main__":
    main()
