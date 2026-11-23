# Multi-agent RL on GSM8K — evaluation harness & baselines (COMS 4705)

```
common/   shared by the whole team: prompts, answer checker, data, cost
eval/     run_eval.py (vLLM baselines + trained adapters), metrics.py
train/    single_grpo.py (Matthew), multi_grpo.py (Tiancheng)
tests/    test_answer.py
configs/  one YAML per run
outputs/  JSONL per run (not committed)
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
| 3 | Single-agent GRPO | `--adapter` | solver | 1 | Matthew |
| 4 | Single-agent GRPO + self-revision | `--adapter` (both steps) | solver → reviser | 2 | Matthew |

Run from the repo root:

```
python -m eval.run_eval --baseline 1 --split dev --limit 20     # smoke test
python -m eval.run_eval --config configs/baseline2.yaml
python -m eval.run_eval --baseline 4 --adapter <path/to/single_grpo_lora>
```

On a T4 (no bfloat16) add `--dtype half`. The script prints one rendered
solver prompt at startup; it must start with Qwen's default system prompt.

Each run writes `outputs/<run_id>.jsonl` (one line per problem) and
`outputs/<run_id>.json` (config, git commit, total `Cost`).

## Metrics

```
python -m eval.metrics outputs/*.jsonl --plot outputs/acc_vs_tokens.png
python -m eval.metrics outputs/*.jsonl \
    --compare outputs/b1_cot_1.5b_gsm8k_test.jsonl outputs/b2_solver_reviser_1.5b_gsm8k_test.jsonl
```

Prints accuracy with a 95% bootstrap CI (1,000 resamples), mean generated
tokens, extraction-method and truncation rates, and flip rates for
baselines 2 and 4. `--compare` adds a paired bootstrap and exact McNemar test.

## Sanity checks before trusting numbers

1. `pytest tests/` passes (gold solutions score 100%).
2. Zero-shot CoT is close to the Qwen2.5 technical report; far lower usually
   means a wrong chat template or the base model instead of `-Instruct`.
3. `last_number` + `none` rate under ~5%.
4. Truncation (`finish_reason == "length"`) is low; otherwise raise `max_tokens`.
5. Read 20 random JSONL lines by eye before sharing numbers.
