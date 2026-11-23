"""Tests for train/benchmark.py on fake training logs (no GPU).  Run:  pytest train/"""
import json
import os

import pytest

from train import benchmark as bm

# Fake per-step costs: step 1 is slow (warmup) and must be excluded from the medians.
FAKE = {  # config: (G, gen tokens/step, gen s, train s, peak GB)
    "single_step_matched": (8, 40_000, 30.0, 20.0, 9.0),
    "single_token_matched": (16, 81_000, 55.0, 41.0, 12.5),
    "multi": (8, 79_000, 58.0, 40.0, 11.0),
}


def fake_runner(oom_on=(), error_on=()):
    """Stands in for tee_runner: writes train_log.jsonl/config.json like run_train.py does."""
    calls = []

    def run(cmd, log_path, cwd=None):
        config = os.path.basename(cmd[cmd.index("-m") + 2]).removesuffix(".yaml")
        mb = int(cmd[cmd.index("--micro_batch_size") + 1]) if "--micro_batch_size" in cmd else None
        calls.append((config, mb, "--allow_fp16" in cmd))
        run_dir = os.path.join(cmd[cmd.index("--output_root") + 1], f"bench_{config}")
        os.makedirs(run_dir, exist_ok=True)
        steps = int(cmd[cmd.index("--max_steps") + 1])
        G, tok, gen, train, mem = FAKE[config]
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({"group_size": G, "batch_problems": 16, "gpu": "NVIDIA L4",
                       "dtype": "float16" if "--allow_fp16" in cmd else "bfloat16"}, f)
        if (config, mb) in error_on:
            open(os.path.join(run_dir, "train_log.jsonl"), "w").close()
            return 1, "Traceback ...\nKeyError: 'gold'\n"
        with open(os.path.join(run_dir, "train_log.jsonl"), "w") as f:
            n = 2 if (config, mb) in oom_on else steps
            for s in range(1, n + 1):
                warm = 3.0 if s == 1 else 1.0 + 0.01 * s
                f.write(json.dumps({"step": s, "reward_mean": 0.5, "step_seconds": (gen + train + 1) * warm,
                                    "step_gen_seconds": gen * warm, "step_train_seconds": train * warm,
                                    "step_sync_seconds": 1.0, "step_gen_tokens": tok + s,
                                    "step_prefill_tokens": tok // 4, "peak_mem_gb": mem,
                                    "device_mem_used_gb": mem + 8}) + "\n")
        if (config, mb) in oom_on:
            return 1, "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
        return 0, "done"
    run.calls = calls
    return run


def run_all(tmp_path, runner, **kw):
    return [bm.run_config(c, str(tmp_path), max_steps=5, runner=runner, **kw) for c in bm.CONFIGS]


def test_medians_exclude_step_one(tmp_path):
    results = run_all(tmp_path, fake_runner())
    rows = [bm.summarize(r) for r in results]
    r = rows[0]
    assert r["status"] == "ok" and r["steps_used"] == "2,3,4,5"
    assert r["step_gen_seconds"] == pytest.approx(30.0 * 1.035)     # median of steps 2-5, not step 1
    assert r["step_gen_tokens"] == pytest.approx(40_000 + 3.5)
    assert r["G"] == 8 and r["B"] == 16 and r["dtype"] == "bfloat16"


def test_oom_retries_with_smaller_micro_batches_then_records_oom(tmp_path):
    runner = fake_runner(oom_on={("single_token_matched", None), ("multi", None), ("multi", 4), ("multi", 1)})
    results = {r["config"]: r for r in run_all(tmp_path, runner)}
    assert [c for c in runner.calls if c[0] == "single_token_matched"] == [
        ("single_token_matched", None, False), ("single_token_matched", 4, False)]
    assert results["single_token_matched"]["status"] == "ok"
    assert results["single_token_matched"]["micro_batch_size"] == 4
    assert results["multi"]["status"] == "OOM" and results["multi"]["steps"] == []
    assert len(results["multi"]["attempts"]) == 3


def test_non_oom_failure_is_not_retried(tmp_path):
    runner = fake_runner(error_on={("multi", None)})
    res = bm.run_config("multi", str(tmp_path), runner=runner)
    assert res["status"] == "error" and len(runner.calls) == 1


def test_existing_result_is_reused_unless_forced(tmp_path):
    runner = fake_runner()
    bm.run_config("multi", str(tmp_path), runner=runner)
    bm.run_config("multi", str(tmp_path), runner=runner)
    assert len(runner.calls) == 1
    bm.run_config("multi", str(tmp_path), runner=runner, force=True)
    assert len(runner.calls) == 2


def test_allow_fp16_is_passed_and_recorded(tmp_path):
    runner = fake_runner()
    res = bm.run_config("multi", str(tmp_path), runner=runner, allow_fp16=True)
    assert runner.calls[0][2] is True and res["dtype"] == "float16"


def test_outputs_csv_raw_tex_ratios_and_estimate(tmp_path):
    runner = fake_runner(oom_on={("multi", None), ("multi", 4), ("multi", 1)})
    results = run_all(tmp_path, runner)
    rows = [bm.summarize(r) for r in results]
    bm.write_raw(results, tmp_path / "raw.jsonl")
    bm.write_csv(rows, tmp_path / "step_cost.csv")
    tex = bm.write_tex(rows, tmp_path / "step_cost.tex", B=16, gpu="NVIDIA L4", dtype="bfloat16")

    raw = [json.loads(line) for line in open(tmp_path / "raw.jsonl")]
    assert len(raw) == 10 and sum(r["excluded_warmup"] for r in raw) == 2
    csv_lines = open(tmp_path / "step_cost.csv").read().splitlines()
    assert csv_lines[0].startswith("config,run,status,G,B,gpu,dtype") and len(csv_lines) == 4
    assert r"\toprule" in tex and r"\bottomrule" in tex and r"$B=16$" in tex and "NVIDIA L4" in tex
    assert r"Multi-agent & 8 & \multicolumn{5}{c}{OOM} \\" in tex
    assert "40{,}004" in tex                                   # thousands separator, median tokens

    ratios = bm.token_ratios(rows)
    assert ratios["single_step_matched"] == 1.0
    assert ratios["single_token_matched"] == pytest.approx(2.02, abs=0.01)

    est, total = bm.compute_estimate(rows, train_size=6973, B=16, n_epochs=1, eval_overhead=0.1)
    assert est[0]["steps_per_epoch"] == 436          # ceil(6973 / 16)
    assert est[0]["hours"] == pytest.approx(436 * rows[0]["step_seconds"] * 1.1 / 3600)
    assert total is None                                       # one config OOM -> no total
    est, total = bm.compute_estimate(rows[:2], 6973, 16, n_epochs=2)
    assert total == pytest.approx(sum(e["hours"] for e in est)) and est[0]["total_steps"] == 872


def test_train_command():
    cmd = bm.train_command("multi", "/drive/runs", 5, micro_batch_size=4, allow_fp16=True)
    assert cmd[1:4] == ["-m", "train.run_train", "train/configs/multi.yaml"]
    for flag in ("--max_steps", "--output_root", "--micro_batch_size", "--allow_fp16"):
        assert flag in cmd
    assert "eval_every=0" in cmd and "save_every=0" in cmd and "run_name=bench_multi" in cmd
