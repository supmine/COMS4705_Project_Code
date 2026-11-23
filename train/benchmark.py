"""Per-step cost benchmark of the GRPO trainer (used by notebooks/colab_grpo_timing.ipynb).

Each config runs train/run_train.py for a few steps in a fresh process (so GPU memory from
one config cannot affect the next), with dev evaluation and checkpoints off. On out-of-
memory it is retried with smaller micro-batches (gradient accumulation; same optimizer
steps). Step 1 (model loading, vLLM warmup) is excluded; the summary is the median over
the remaining steps.

Files under <bench_dir>/:
    results/<config>.json   status, micro-batch size, per-step numbers (skip if present)
    logs/<config>_mb<N>.log  stdout/stderr of each attempt
    runs/bench_<config>/     the training run directory (train_log.jsonl, config.json)
    raw.jsonl, step_cost.csv, step_cost.tex
"""
import csv
import json
import math
import os
import statistics
import subprocess
import sys

CONFIGS = ["single_step_matched", "single_token_matched", "multi"]
DISPLAY = {"single_step_matched": "Single-agent, step-matched",
           "single_token_matched": "Single-agent, token-matched",
           "multi": "Multi-agent"}
REFERENCE = "single_step_matched"
# None = the config's default packing (micro_batch_tokens only), then smaller micro-batches.
MICRO_BATCH_RETRIES = [None, 4, 1]
OOM_MARKERS = ("out of memory", "outofmemoryerror", "no available memory for the cache blocks")
METRICS = ["step_seconds", "step_gen_seconds", "step_train_seconds", "step_sync_seconds",
           "step_gen_tokens", "step_prefill_tokens", "peak_mem_gb", "device_mem_used_gb"]


# ---------------------------------------------------------------- running

def is_oom(text):
    low = text.lower()
    return any(m in low for m in OOM_MARKERS)


def train_command(config, run_root, max_steps, micro_batch_size=None, allow_fp16=False):
    cmd = [sys.executable, "-m", "train.run_train", f"train/configs/{config}.yaml",
           "--max_steps", str(max_steps), "--output_root", run_root,
           "--set", f"run_name=bench_{config}", "--set", "eval_every=0", "--set", "save_every=0"]
    if micro_batch_size:
        cmd += ["--micro_batch_size", str(micro_batch_size)]
    if allow_fp16:
        cmd.append("--allow_fp16")
    return cmd


def tee_runner(cmd, log_path, cwd=None):
    """Run cmd, streaming its output to the notebook and to log_path. Returns (code, text)."""
    lines = []
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            log.write(line)
            lines.append(line)
            print(line, end="")
        proc.wait()
    return proc.returncode, "".join(lines)


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def run_config(config, bench_dir, max_steps=5, allow_fp16=False, force=False,
               runner=tee_runner, cwd=None):
    """Benchmark one config; reuse results/<config>.json unless force. Returns the result."""
    result_path = os.path.join(bench_dir, "results", f"{config}.json")
    if os.path.exists(result_path) and not force:
        print(f"{config}: using existing {result_path} (FORCE_RERUN=False)")
        with open(result_path) as f:
            return json.load(f)
    for sub in ("results", "logs", "runs"):
        os.makedirs(os.path.join(bench_dir, sub), exist_ok=True)
    run_root = os.path.join(bench_dir, "runs")
    run_dir = os.path.join(run_root, f"bench_{config}")
    attempts = []
    status, steps, run_cfg = "error", [], {}
    for mb in MICRO_BATCH_RETRIES:
        log_path = os.path.join(bench_dir, "logs", f"{config}_mb{mb or 'default'}.log")
        print(f"=== {config}: micro_batch_size={mb or 'default'}")
        code, output = runner(train_command(config, run_root, max_steps, mb, allow_fp16), log_path, cwd)
        steps = [r for r in load_jsonl(os.path.join(run_dir, "train_log.jsonl")) if "step_seconds" in r]
        attempts.append({"micro_batch_size": mb, "returncode": code, "log": log_path,
                         "oom": is_oom(output), "steps_logged": len(steps)})
        if code == 0 and len(steps) >= max_steps:
            status = "ok"
            break
        if not is_oom(output):
            status = "error"           # not a memory problem: smaller micro-batches won't help
            break
        status = "OOM"
    cfg_path = os.path.join(run_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            run_cfg = json.load(f)
    result = {"config": config, "status": status,
              "micro_batch_size": attempts[-1]["micro_batch_size"] if status == "ok" else None,
              "group_size": run_cfg.get("group_size"), "batch_problems": run_cfg.get("batch_problems"),
              "gpu": run_cfg.get("gpu"), "dtype": run_cfg.get("dtype"), "max_steps": max_steps,
              "attempts": attempts,
              "steps": [{k: r.get(k) for k in ["step"] + METRICS} for r in steps] if status == "ok" else []}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    return result


# ---------------------------------------------------------------- summarizing

def summarize(result, skip_first=1):
    """One row: medians over steps after the first `skip_first` (steps 2-5 by default)."""
    row = {"config": result["config"], "run": DISPLAY.get(result["config"], result["config"]),
           "status": result["status"], "G": result.get("group_size"),
           "B": result.get("batch_problems"), "gpu": result.get("gpu"), "dtype": result.get("dtype"),
           "micro_batch_size": (result.get("micro_batch_size") or "default") if result["status"] == "ok" else "-"}
    used = [s for s in result.get("steps", []) if s["step"] > skip_first]
    row["steps_used"] = ",".join(str(s["step"]) for s in used)
    for m in METRICS:
        vals = [s[m] for s in used if s.get(m) is not None]
        row[m] = statistics.median(vals) if vals else None
    return row


def write_raw(results, path):
    with open(path, "w") as f:
        for res in results:
            for s in res.get("steps", []):
                f.write(json.dumps({"config": res["config"], "G": res.get("group_size"),
                                    "B": res.get("batch_problems"), "gpu": res.get("gpu"),
                                    "dtype": res.get("dtype"),
                                    "micro_batch_size": res.get("micro_batch_size"),
                                    "excluded_warmup": s["step"] == 1, **s}) + "\n")


CSV_FIELDS = ["config", "run", "status", "G", "B", "gpu", "dtype", "micro_batch_size", "steps_used"] + METRICS


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def _tex_escape(text):
    return str(text).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def write_tex(rows, path, B, gpu, dtype, steps="2--5"):
    """booktabs table: Run, G, generated tokens/step, generation/training/total s per step, peak GB."""
    def num(v, fmt):
        return "--" if v is None else format(v, fmt)
    body = []
    for r in rows:
        if r["status"] != "ok":
            body.append(f"{_tex_escape(r['run'])} & {r['G'] or '--'} & "
                        f"\\multicolumn{{5}}{{c}}{{{_tex_escape(r['status'])}}} \\\\")
            continue
        body.append(" & ".join([_tex_escape(r["run"]), str(r["G"]),
                                num(r["step_gen_tokens"], ",.0f").replace(",", "{,}"),
                                num(r["step_gen_seconds"], ".1f"), num(r["step_train_seconds"], ".1f"),
                                num(r["step_seconds"], ".1f"), num(r["peak_mem_gb"], ".1f")]) + r" \\")
    tex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Run & $G$ & Gen.\ tokens/step & Gen.\ s/step & Train s/step & Total s/step & Peak mem.\ (GB) \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        (rf"\caption{{Per-step cost of GRPO training (median of steps {steps}), $B={B}$ problems "
         rf"per step, on one {_tex_escape(gpu)} in {_tex_escape(dtype)}. Training time covers "
         r"log-probs, backward and the optimizer step; peak memory is the training process "
         r"(\texttt{torch.cuda.max\_memory\_allocated}), excluding vLLM's engine.}"),
        r"\label{tab:step_cost}",
        r"\end{table}",
    ])
    with open(path, "w") as f:
        f.write(tex + "\n")
    return tex


def token_ratios(rows, reference=REFERENCE):
    """Generated tokens per step relative to the reference config (expected ~2 for the others)."""
    ref = next((r for r in rows if r["config"] == reference and r["status"] == "ok"), None)
    if ref is None or not ref["step_gen_tokens"]:
        return {}
    return {r["config"]: r["step_gen_tokens"] / ref["step_gen_tokens"]
            for r in rows if r["status"] == "ok" and r["step_gen_tokens"]}


def compute_estimate(rows, train_size, B, n_epochs=1, eval_overhead=0.1):
    """Steps per epoch = ceil(train_size / B); hours = steps * median s/step * (1 + overhead)."""
    steps_per_epoch = math.ceil(train_size / B)
    out = []
    for r in rows:
        hours = None
        if r["status"] == "ok" and r["step_seconds"] is not None:
            hours = steps_per_epoch * n_epochs * r["step_seconds"] * (1 + eval_overhead) / 3600
        out.append({"config": r["config"], "run": r["run"], "steps_per_epoch": steps_per_epoch,
                    "total_steps": steps_per_epoch * n_epochs, "seconds_per_step": r["step_seconds"],
                    "hours": hours})
    known = [e["hours"] for e in out if e["hours"] is not None]
    return out, (sum(known) if len(known) == len(out) else None)
