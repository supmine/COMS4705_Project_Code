# Multi-agent RL on GSM8K — evaluation harness & baselines (COMS 4705)

```
common/   shared by the whole team: prompts, answer checker, data, cost
eval/     run_eval.py (vLLM baselines + trained adapters), metrics.py, plot_results.py
train/    single_grpo.py (Matthew), multi_grpo.py (Tiancheng)
tests/    test_answer.py
configs/  one YAML per run
outputs/  one directory per run (not committed)
colab_eval.ipynb  run one baseline on Colab, results saved to Drive
```

`train/` must import prompts and the answer checker from `common/`, so the
training reward and the eval accuracy are the same function.

## Setup

```
pip install -r requirements.txt
pytest tests/          # gold solutions grade 100%; rendered prompt matches Qwen's template
```

## Baselines (GSM8K, Qwen2.5-1.5B-Instruct)

One script, `eval/run_eval.py`, selected by `--baseline`:

| # | Name | Weights | Pipeline | Generations / problem | Run by |
|---|------|---------|----------|-----------------------|--------|
| 1 | Zero-shot CoT | base model | solver | 1 | Matthew |
| 2 | Frozen solver–reviser | base model | solver → reviser | 2 | Supanart |
| 3 | Single-agent GRPO | `--adapter` | solver | 1 | Andy |
| 4 | Single-agent GRPO + self-revision | `--adapter` (both steps) | solver → reviser | 2 | TBD |

Run from the repo root:

```
python -m eval.run_eval --baseline 1 --split dev --limit 20     # smoke test
python -m eval.run_eval --config configs/baseline2.yaml
python -m eval.run_eval --baseline 4 --adapter <path/to/single_grpo_lora>
```

On a T4 (no bfloat16) add `--dtype half`. The script prints one rendered
solver prompt at startup; it must start with Qwen's default system prompt.

Each run writes a directory `outputs/<run_id>/` (default run_id `b<N>_<split>`,
plus `_seed<S>` when `--seed` is not 0) holding `predictions.jsonl` (one line
per problem) and `config.json` (args, system, group, git commit, total `Cost`).
For trained adapters, pass the training seed as `--seed`.

On Colab, open `colab_eval.ipynb`: set `BASELINE` (and `ADAPTER` for 3–4)
in the settings cell and run top to bottom.

## Metrics and figures

```
python -m eval.metrics outputs/*/
python -m eval.metrics outputs/*/ --compare outputs/b1_test outputs/b2_test
python eval/plot_results.py outputs/*/ --output_dir figures/
```

`metrics.py` prints accuracy with a 95% bootstrap CI (1,000 resamples), mean
generated tokens, extraction-method and truncation rates, and flip rates for
baselines 2 and 4. `--compare` adds a paired bootstrap and exact McNemar test.

`plot_results.py` writes to `--output_dir`:
- `summary.csv`: one row per run (accuracy, CI, mean generated/prefill tokens,
  share of problems whose final answer has no `\boxed{}`, share with any
  generation truncated).
- `acc_vs_tokens.png/.pdf`: one point per system; marker = frozen / trained
  single-agent / multi-agent; runs of one system at different budgets
  (`k`, `num_rounds`, `budget`, `max_tokens` in `config.json`) are connected;
  several seeds are averaged with ±1 std bars.
- `flip_rates.png/.pdf`: solver answer → final answer outcomes for two-step runs.

Multi-agent runs appear automatically if they write the same run directory
layout with `"system"` and `"group": "multi_agent"` in `config.json`.

## Sanity checks before trusting numbers

1. `pytest tests/` passes (gold solutions score 100%).
2. Zero-shot CoT is close to the Qwen2.5 technical report; far lower usually
   means a wrong chat template or the base model instead of `-Instruct`.
3. `last_number` + `none` rate under ~5%.
4. Truncation (`finish_reason == "length"`) is low; otherwise raise `max_tokens`.
5. Read 20 random JSONL lines by eye before sharing numbers.
