"""Training curves from one or more runs/<run_name>/train_log.jsonl.

    python -m train.plot_curve runs/multi runs/single_step_matched runs/single_token_matched \
        --output figures/training_curve

Left: mean training reward per step (moving average). Right: dev accuracy against the
cumulative generated training tokens, the axis on which the step- and token-matched
single-agent runs are compared with the multi-agent run. Writes <output>.png and .pdf.
"""
import argparse
import json
import os

# Same validated colorblind-safe order as eval/plot_results.py (first four slots).
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4", "#008300"]
INK_2, MUTED, GRID, AXIS = "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load_log(run_dir):
    with open(os.path.join(run_dir, "train_log.jsonl")) as f:
        return [json.loads(line) for line in f if line.strip()]


def moving_average(xs, window):
    out, total = [], 0.0
    for i, x in enumerate(xs):
        total += x
        if i >= window:
            total -= xs[i - window]
        out.append(total / min(i + 1, window))
    return out


def run_label(run_dir):
    path = os.path.join(run_dir, "config.json")
    name = os.path.basename(os.path.normpath(run_dir))
    if os.path.exists(path):
        with open(path) as f:
            cfg = json.load(f)
        return f"{cfg.get('run_name', name)} ({cfg.get('mode')}, G={cfg.get('group_size')})"
    return name


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="run directories (runs/<run_name>)")
    p.add_argument("--output", default="figures/training_curve")
    p.add_argument("--window", type=int, default=10, help="moving-average window (steps)")
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import EngFormatter, MaxNLocator, PercentFormatter

    fig, (ax_r, ax_d) = plt.subplots(1, 2, figsize=(10, 4))
    for ax in (ax_r, ax_d):
        style(ax)
    for i, run in enumerate(args.runs):
        log = load_log(run)
        color, label = COLORS[i % len(COLORS)], run_label(run)
        train = [r for r in log if "reward_mean" in r]
        if train:
            ax_r.plot([r["step"] for r in train],
                      moving_average([r["reward_mean"] for r in train], args.window),
                      color=color, linewidth=2, label=label)
        dev = [r for r in log if "dev_acc" in r]
        if dev:
            ax_d.plot([r["dev_at_gen_tokens"] for r in dev], [r["dev_acc"] for r in dev],
                      color=color, linewidth=2, marker="o", markersize=6,
                      markeredgecolor="white", markeredgewidth=1.2, label=label)

    ax_r.set_xlabel("Optimizer step", color=INK_2)
    ax_r.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax_r.set_ylabel(f"Mean training reward ({args.window}-step average)", color=INK_2)
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax_d.set_xlabel("Cumulative generated training tokens", color=INK_2)
    ax_d.xaxis.set_major_formatter(EngFormatter(sep=""))
    ax_d.set_ylabel("Dev accuracy (greedy)", color=INK_2)
    ax_d.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    handles, labels = (ax_d if ax_d.lines else ax_r).get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, 1.0), ncol=min(len(labels), 3))
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.output}.{ext}", dpi=200, bbox_inches="tight")
    print(f"saved {args.output}.png/.pdf")


if __name__ == "__main__":
    main()
