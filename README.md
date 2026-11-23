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
pytest tests/          # includes: every GSM8K gold solution grades 100%
```

## Baselines (GSM8K test, Qwen2.5-1.5B-Instruct)

Run from the repo root:

```
python -m eval.run_eval --config configs/cot.yaml
for k in 1 3 5 8; do python -m eval.run_eval --config configs/sc_k$k.yaml; done
python -m eval.run_eval --config configs/solver_reviser_frozen.yaml
python -m eval.run_eval --config configs/self_revision_frozen.yaml
python -m eval.run_eval --system cot --split dev --limit 20   # smoke test
```

| `--system`       | what it does                                   | generations / problem |
|------------------|------------------------------------------------|-----------------------|
| `cot`            | solver only, greedy                            | 1 |
| `sc`             | solver samples `--k`, majority vote            | k |
| `solver_reviser` | solver → reviser (`--solver_lora`, `--reviser_lora`) | 2 |
| `self_revision`  | solver → reviser prompt with the solver's adapter | 2 |

Each run writes `outputs/<run_id>.jsonl` (one line per problem) and
`outputs/<run_id>.json` (config, git commit, total `Cost`).

## Metrics

```
python -m eval.metrics outputs/*.jsonl --plot outputs/acc_vs_tokens.png
python -m eval.metrics outputs/*.jsonl \
    --compare outputs/cot_frozen_1.5b_gsm8k_test.jsonl outputs/solver_reviser_frozen_1.5b_gsm8k_test.jsonl
```

Prints accuracy with a 95% bootstrap CI (1,000 resamples), mean generated
tokens, extraction-method and truncation rates, and flip rates for
solver/reviser runs. `--compare` adds a paired bootstrap and exact McNemar test.

## Sanity checks before trusting numbers

1. `pytest tests/` passes (gold solutions score 100%).
2. Zero-shot CoT is close to the Qwen2.5 technical report; far lower usually
   means a wrong chat template or the base model instead of `-Instruct`.
3. `last_number` + `none` rate under ~5%; otherwise fix the prompt first.
4. Truncation (`finish_reason == "length"`) is low; otherwise raise `max_tokens`.
5. SC at k=1 is close to CoT, and accuracy rises with k.
6. Read 20 random JSONL lines by eye before sharing numbers.
