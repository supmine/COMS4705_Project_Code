"""Summary table and figures from run directories (predictions JSONL + config.json).

    python eval/plot_results.py outputs/b1_test outputs/b2_test --output_dir figures/

Writes to --output_dir:
  summary.csv                 one row per run
  acc_vs_tokens.{png,pdf}     accuracy vs mean generated tokens per problem
  flip_rates.{png,pdf}        solver -> final answer outcomes for two-step runs

config.json keys read: system, group, baseline, adapter, seed, and optional
budget keys (budget, k, num_rounds, max_tokens). Runs that share a system and
budget but differ in seed are averaged (mean +/- 1 std). Multi-agent runs from
train/ should write the same config.json with group "multi_agent".
"""
import argparse
import csv
import json
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.metrics import bootstrap_ci, load_run, run_name  # noqa: E402

# Okabe-Ito-style categorical slots, validated colorblind-safe for the first four
# systems on all pairs. Systems keep their slot in every figure; new systems take
# the next free slot, and past four the direct labels and markers carry identity.
SYSTEM_COLORS = {
    "cot": "#2a78d6",
    "solver_reviser": "#eb6834",
    "grpo": "#1baf7a",
    "grpo_self_revision": "#4a3aa7",
}
EXTRA_COLORS = ["#e87ba4", "#008300", "#eda100", "#e34948"]
FALLBACK_COLOR = "#898781"

SYSTEM_LABELS = {
    "cot": "Zero-shot CoT",
    "solver_reviser": "Frozen solver–reviser",
    "grpo": "Single-agent GRPO",
    "grpo_self_revision": "GRPO + self-revision",
}
GROUP_MARKERS = {"frozen": "o", "single_agent": "s", "multi_agent": "^"}
GROUP_LABELS = {"frozen": "Frozen (1–2)", "single_agent": "Trained single-agent (3–4)",
                "multi_agent": "Trained multi-agent"}
BASELINE_GROUPS = {1: "frozen", 2: "frozen", 3: "single_agent", 4: "single_agent"}
BUDGET_KEYS = ("budget", "k", "num_rounds", "max_tokens")

# Flip outcomes: unchanged correctness in light tints, flips in strong blue/red.
FLIPS = [("right_right", "right→right", "#b7d3f6"),
         ("wrong_right", "wrong→right", "#256abf"),
         ("right_wrong", "right→wrong", "#e34948"),
         ("wrong_wrong", "wrong→wrong", "#e1e0d9")]

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


# ---------------------------------------------------------------- loading

def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        cfg = json.load(f)
    nested = cfg.get("config")  # older layout nested args under "config"
    return {**nested, **cfg} if isinstance(nested, dict) else cfg


def summarize_run(run_dir, n_boot):
    rows = load_run(run_dir)
    cfg = load_config(run_dir)
    baseline = cfg.get("baseline")
    lo, hi = bootstrap_ci(rows, n_boot)
    n = len(rows)
    two_step = [r for r in rows if len(r["turns"]) >= 2 and r["turns"][0]["agent"] != "sample"]
    flips = None
    if two_step:
        counts = defaultdict(int)
        for r in two_step:
            first = "right" if r["turns"][0]["correct"] else "wrong"
            last = "right" if r["final_correct"] else "wrong"
            counts[f"{first}_{last}"] += 1
        flips = {key: counts[key] / len(two_step) for key, _, _ in FLIPS}
    return {
        "run": run_name(run_dir),
        "system": cfg.get("system") or rows[0].get("system") or run_name(run_dir),
        "group": cfg.get("group") or BASELINE_GROUPS.get(baseline, "multi_agent"),
        "baseline": baseline,
        "adapter": cfg.get("adapter"),
        "seed": cfg.get("seed"),
        "budget": tuple(cfg.get(k) for k in BUDGET_KEYS),
        "budget_desc": {k: cfg.get(k) for k in BUDGET_KEYS if cfg.get(k) is not None},
        "n": n,
        "accuracy": sum(r["final_correct"] for r in rows) / n,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean_gen_tokens": sum(r["total_gen_tokens"] for r in rows) / n,
        "mean_prefill_tokens": sum(r.get("total_prefill_tokens",
                                         sum(t["prefill_tokens"] for t in r["turns"]))
                                   for r in rows) / n,
        # Per problem: the graded (final) answer had no \boxed{}.
        "frac_no_boxed": sum(r["turns"][-1]["method"] != "boxed" for r in rows) / n,
        # Per problem: any generation hit max_tokens.
        "frac_truncated": sum(any(t.get("finish_reason") == "length" for t in r["turns"])
                              for r in rows) / n,
        "flips": flips,
    }


CSV_FIELDS = ["run", "system", "baseline", "group", "adapter", "seed", "n", "accuracy",
              "ci_lo", "ci_hi", "mean_gen_tokens", "mean_prefill_tokens",
              "frac_no_boxed", "frac_truncated"]


def write_summary(runs, path):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in runs:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})


# ---------------------------------------------------------------- aggregation

def system_colors(runs):
    colors, extra = dict(SYSTEM_COLORS), iter(EXTRA_COLORS)
    for system in sorted({r["system"] for r in runs} - colors.keys()):
        colors[system] = next(extra, FALLBACK_COLOR)
    return colors


def aggregate_seeds(runs):
    """One point per (system, budget): mean +/- std over seeds, or the bootstrap
    CI when there is a single seed."""
    groups = defaultdict(list)
    for r in runs:
        groups[(r["system"], r["budget"])].append(r)
    points = []
    for (system, budget), rs in groups.items():
        accs = [r["accuracy"] for r in rs]
        toks = [r["mean_gen_tokens"] for r in rs]
        acc, tok = statistics.mean(accs), statistics.mean(toks)
        if len(rs) > 1:
            sd = statistics.stdev(accs)
            yerr, xerr, err_kind = (sd, sd), statistics.stdev(toks), "std"
        else:
            yerr, xerr, err_kind = (acc - rs[0]["ci_lo"], rs[0]["ci_hi"] - acc), 0.0, "ci"
        flips = None
        if all(r["flips"] for r in rs):
            flips = {k: [r["flips"][k] for r in rs] for k, _, _ in FLIPS}
        points.append({"system": system, "group": rs[0]["group"], "budget": budget,
                       "budget_desc": rs[0]["budget_desc"], "n_seeds": len(rs),
                       "accuracy": acc, "tokens": tok, "yerr": yerr, "xerr": xerr,
                       "err_kind": err_kind, "flips": flips})
    return points


def system_label(system):
    return SYSTEM_LABELS.get(system, system.replace("_", " "))


def point_label(p, varying_keys):
    name = system_label(p["system"])
    extra = [f"{k}={p['budget_desc'][k]}" for k in varying_keys if k in p["budget_desc"]]
    if p["n_seeds"] > 1:
        extra.append(f"{p['n_seeds']} seeds")
    return f"{name} ({', '.join(extra)})" if extra else name


def varying_budget_keys(points, system):
    descs = [p["budget_desc"] for p in points if p["system"] == system]
    keys = {k for d in descs for k in d}
    return [k for k in BUDGET_KEYS if k in keys and len({d.get(k) for d in descs}) > 1]


# ---------------------------------------------------------------- figures

def style_axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def place_labels(ax, points, texts):
    """Direct labels beside each marker, trying offsets until a label overlaps
    no earlier label, marker or error bar."""
    from matplotlib.transforms import Bbox

    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    obstacles = []
    for p in points:
        x, y = ax.transData.transform((p["tokens"], p["accuracy"]))
        _, y_lo = ax.transData.transform((p["tokens"], p["accuracy"] - p["yerr"][0]))
        _, y_hi = ax.transData.transform((p["tokens"], p["accuracy"] + p["yerr"][1]))
        obstacles.append(Bbox([[x - 8, y - 8], [x + 8, y + 8]]))        # marker
        obstacles.append(Bbox([[x - 5, y_lo - 2], [x + 5, y_hi + 2]]))  # error bar
    candidates = [((10, 3), "left", "bottom"), ((10, -3), "left", "top"),
                  ((-10, 3), "right", "bottom"), ((-10, -3), "right", "top"),
                  ((10, 16), "left", "bottom"), ((10, -16), "left", "top"),
                  ((-10, 16), "right", "bottom"), ((-10, -16), "right", "top")]
    for p, text in sorted(zip(points, texts), key=lambda pt: -pt[0]["accuracy"]):
        best = None
        for offset, ha, va in candidates:
            ann = ax.annotate(text, (p["tokens"], p["accuracy"]), xytext=offset,
                              textcoords="offset points", ha=ha, va=va,
                              fontsize=8, color=INK)
            box = ann.get_window_extent(renderer).padded(3)
            hits = sum(box.overlaps(b) for b in obstacles)
            if best is None or hits < best[0]:
                if best:
                    best[1].remove()
                best = (hits, ann)
            else:
                ann.remove()
            if hits == 0:
                break
        obstacles.append(best[1].get_window_extent(renderer))


def save(fig, output_dir, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    print(f"saved {os.path.join(output_dir, name)}.png/.pdf")


def plot_acc_vs_tokens(points, colors, output_dir):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter

    fig, ax = plt.subplots(figsize=(7, 4.5))
    style_axes(ax)

    # Same system, different budget: connect in token order.
    for system in sorted({p["system"] for p in points}):
        ps = sorted((p for p in points if p["system"] == system), key=lambda p: p["tokens"])
        if len(ps) > 1:
            ax.plot([p["tokens"] for p in ps], [p["accuracy"] for p in ps],
                    color=colors[system], linewidth=2, zorder=2)

    for p in points:
        ax.errorbar(p["tokens"], p["accuracy"],
                    yerr=[[p["yerr"][0]], [p["yerr"][1]]],
                    xerr=p["xerr"] or None,
                    fmt=GROUP_MARKERS.get(p["group"], "D"), markersize=8,
                    color=colors[p["system"]], ecolor=colors[p["system"]],
                    elinewidth=1.2, capsize=3, markeredgecolor="white",
                    markeredgewidth=1.5, zorder=3)

    ax.set_xlabel("Mean generated tokens per problem", color=INK_2)
    ax.set_ylabel("Accuracy", color=INK_2)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.margins(x=0.15, y=0.12)
    ax.set_xlim(left=0)

    systems = sorted({p["system"] for p in points}, key=lambda s: list(colors).index(s))
    groups = [g for g in GROUP_MARKERS if any(p["group"] == g for p in points)]
    handles = [Line2D([], [], color=colors[s], marker="o", linestyle="", markersize=7,
                      label=system_label(s)) for s in systems]
    handles += [Line2D([], [], color=MUTED, marker=GROUP_MARKERS[g], linestyle="",
                       markersize=7, label=GROUP_LABELS[g]) for g in groups]
    ax.legend(handles=handles, fontsize=8, frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1))
    kinds = {p["err_kind"] for p in points}
    note = {"ci": "95% bootstrap CI (one seed)", "std": "±1 std over seeds"}
    ax.set_title("Error bars: " + "; ".join(note[k] for k in sorted(kinds)),
                 fontsize=8, color=MUTED, loc="left")
    place_labels(ax, points, [point_label(p, varying_budget_keys(points, p["system"]))
                              for p in points])
    save(fig, output_dir, "acc_vs_tokens")
    plt.close(fig)


def plot_flip_rates(points, colors, output_dir):
    import matplotlib.pyplot as plt

    ps = [p for p in points if p["flips"]]
    if not ps:
        print("no two-step runs; skipping flip_rates")
        return
    ps.sort(key=lambda p: (list(colors).index(p["system"]), p["tokens"]))
    labels = [point_label(p, varying_budget_keys(ps, p["system"])) for p in ps]

    fig, ax = plt.subplots(figsize=(7.5, 0.85 * len(ps) + 1.4))
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9, left=False)

    ys = [1.6 * i for i in range(len(ps))][::-1]
    ax.set_xlim(0, 100)
    ax.set_ylim(-1.0, ys[0] + 1.0)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    left = [0.0] * len(ps)
    for key, name, color in FLIPS:
        widths = [100 * statistics.mean(p["flips"][key]) for p in ps]
        ax.barh(ys, widths, left=left, height=0.8, color=color, edgecolor="white",
                linewidth=1.5, label=name)
        if key in ("wrong_right", "right_wrong"):
            for y, l, w, p in zip(ys, left, widths, ps):
                vals = [100 * v for v in p["flips"][key]]
                text = f"{w:.1f}%" + (f" ±{statistics.stdev(vals):.1f}" if len(vals) > 1 else "")
                t = ax.text(l + w / 2, y, text, ha="center", va="center",
                            fontsize=8, color="white")
                seg = (ax.transData.transform((l + w, y))[0]
                       - ax.transData.transform((l, y))[0])
                if t.get_window_extent(renderer).width > seg - 4:
                    # Too narrow: wrong->right goes above the bar, right->wrong below.
                    above = key == "wrong_right"
                    t.set_position((l + w / 2, y + (0.45 if above else -0.45)))
                    t.set_va("bottom" if above else "top")
                    t.set_color(INK)
        left = [l + w for l, w in zip(left, widths)]

    ax.set_yticks(ys, labels)
    # Identity swatch in the system's color beside each bar.
    for y, p in zip(ys, ps):
        ax.plot(-1.5, y, marker="s", markersize=8, color=colors[p["system"]],
                clip_on=False, markeredgecolor="white")
    ax.tick_params(axis="y", pad=14)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Share of problems (solver answer → final answer)", color=INK_2)
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="lower left",
              bbox_to_anchor=(0, 1.02), handlelength=1.2)
    save(fig, output_dir, "flip_rates")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories from run_eval")
    p.add_argument("--output_dir", default="figures")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(args.output_dir, exist_ok=True)
    runs = [summarize_run(r, args.n_boot) for r in args.runs]
    write_summary(runs, os.path.join(args.output_dir, "summary.csv"))
    print(f"saved {os.path.join(args.output_dir, 'summary.csv')}")

    colors = system_colors(runs)
    points = aggregate_seeds(runs)
    plot_acc_vs_tokens(points, colors, args.output_dir)
    plot_flip_rates(points, colors, args.output_dir)


if __name__ == "__main__":
    main()
