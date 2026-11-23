"""Metrics over run_eval outputs. A run is a directory (predictions JSONL +
config.json) or a JSONL file. Figures live in eval/plot_results.py.

    python -m eval.metrics outputs/*/                            # summary table
    python -m eval.metrics outputs/*/ --compare outputs/b1_test outputs/b2_test
"""
import argparse
import glob
import json
import math
import os
import random


def run_jsonl(path):
    """The JSONL file of a run directory (or the path itself if it is a file)."""
    if not os.path.isdir(path):
        return path
    files = sorted(glob.glob(os.path.join(path, "*.jsonl")))
    if len(files) != 1:
        raise ValueError(f"{path}: expected one .jsonl file, found {len(files)}")
    return files[0]


def run_name(path):
    return os.path.basename(os.path.normpath(path)).removesuffix(".jsonl")


def load_run(path):
    with open(run_jsonl(path)) as f:
        return [json.loads(line) for line in f if line.strip()]


def accuracy(rows):
    return sum(r["final_correct"] for r in rows) / len(rows)


def bootstrap_ci(rows, n_boot=1000, alpha=0.05, seed=0):
    """Percentile bootstrap CI over problems for accuracy."""
    rng = random.Random(seed)
    xs = [int(r["final_correct"]) for r in rows]
    n = len(xs)
    stats = sorted(sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = stats[int(math.floor(alpha / 2 * n_boot))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return lo, hi


def _paired(rows_a, rows_b):
    a = {r["qid"]: int(r["final_correct"]) for r in rows_a}
    b = {r["qid"]: int(r["final_correct"]) for r in rows_b}
    qids = sorted(a.keys() & b.keys())
    if len(qids) != len(a) or len(qids) != len(b):
        print(f"warning: comparing on the {len(qids)} shared qids only")
    return [a[q] for q in qids], [b[q] for q in qids]


def mcnemar(rows_a, rows_b):
    """Exact two-sided McNemar test. Returns (b_only, a_only, p_value), where
    a_only = problems A got right and B got wrong."""
    xa, xb = _paired(rows_a, rows_b)
    a_only = sum(1 for x, y in zip(xa, xb) if x and not y)
    b_only = sum(1 for x, y in zip(xa, xb) if y and not x)
    n = a_only + b_only
    if n == 0:
        return b_only, a_only, 1.0
    k = min(a_only, b_only)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return b_only, a_only, min(1.0, p)


def paired_bootstrap(rows_a, rows_b, n_boot=1000, alpha=0.05, seed=0):
    """CI for acc(B) - acc(A), resampling the same problems for both."""
    xa, xb = _paired(rows_a, rows_b)
    d = [y - x for x, y in zip(xa, xb)]
    n = len(d)
    rng = random.Random(seed)
    stats = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    lo = stats[int(math.floor(alpha / 2 * n_boot))]
    hi = stats[min(n_boot - 1, int(math.ceil((1 - alpha / 2) * n_boot)) - 1)]
    return sum(d) / n, lo, hi


def mean_gen_tokens(rows):
    return sum(r["total_gen_tokens"] for r in rows) / len(rows)


def flip_rates(rows):
    """From solver vs reviser correctness. None for runs without both turns.
    no_change = reviser's normalized answer equals the solver's."""
    pairs = []
    for r in rows:
        ag = {t["agent"]: t for t in r["turns"]}
        if "solver" in ag and "reviser" in ag:
            pairs.append((ag["solver"], ag["reviser"]))
    if not pairs:
        return None
    n = len(pairs)
    wrong = [p for p in pairs if not p[0]["correct"]]
    right = [p for p in pairs if p[0]["correct"]]
    return {
        "n": n,
        "wrong_to_right": sum(r["correct"] for _, r in wrong) / len(wrong) if wrong else float("nan"),
        "right_to_wrong": sum(not r["correct"] for _, r in right) / len(right) if right else float("nan"),
        "no_change": sum(s["pred"] == r["pred"] for s, r in pairs) / n,
        "n_solver_wrong": len(wrong),
        "n_solver_right": len(right),
    }


def method_rates(rows):
    """Share of turns per extraction method, and share truncated by max_tokens."""
    turns = [t for r in rows for t in r["turns"]]
    counts = {}
    for t in turns:
        counts[t["method"]] = counts.get(t["method"], 0) + 1
    rates = {m: c / len(turns) for m, c in counts.items()}
    rates["truncated"] = sum(t.get("finish_reason") == "length" for t in turns) / len(turns)
    return rates


def summarize(path, n_boot=1000):
    rows = load_run(path)
    lo, hi = bootstrap_ci(rows, n_boot)
    out = {"run": run_name(path), "n": len(rows),
           "acc": accuracy(rows), "ci_lo": lo, "ci_hi": hi,
           "gen_tokens": mean_gen_tokens(rows), "methods": method_rates(rows)}
    flips = flip_rates(rows)
    if flips:
        out["flips"] = flips
    return out


def print_summary(summaries):
    print(f"{'run':40s} {'n':>5s} {'acc':>7s} {'95% CI':>17s} {'gen tok':>8s} "
          f"{'boxed':>6s} {'last#':>6s} {'none':>6s} {'trunc':>6s}")
    for s in summaries:
        m = s["methods"]
        print(f"{s['run']:40s} {s['n']:5d} {s['acc']:7.4f} "
              f"[{s['ci_lo']:.4f}, {s['ci_hi']:.4f}] {s['gen_tokens']:8.1f} "
              f"{m.get('boxed', 0):6.3f} {m.get('last_number', 0):6.3f} "
              f"{m.get('none', 0):6.3f} {m['truncated']:6.3f}")
    for s in summaries:
        if "flips" in s:
            f = s["flips"]
            print(f"{s['run']}: P(wrong->right)={f['wrong_to_right']:.3f} "
                  f"(of {f['n_solver_wrong']}), P(right->wrong)={f['right_to_wrong']:.3f} "
                  f"(of {f['n_solver_right']}), no_change={f['no_change']:.3f}")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories (or JSONL files) from run_eval")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="paired comparison of two runs")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args(argv)

    summaries = [summarize(r, args.n_boot) for r in args.runs]
    print_summary(summaries)
    if args.compare:
        a, b = (load_run(x) for x in args.compare)
        b_only, a_only, pval = mcnemar(a, b)
        diff, lo, hi = paired_bootstrap(a, b, args.n_boot)
        print(f"\nB - A = {diff:+.4f} [{lo:+.4f}, {hi:+.4f}] (paired bootstrap); "
              f"McNemar: B-only {b_only}, A-only {a_only}, p={pval:.4g}")


if __name__ == "__main__":
    main()
