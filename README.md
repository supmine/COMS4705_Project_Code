# Multi-agent RL on GSM8K and MATH — evaluation harness & baselines (COMS 4705)

```
common/   shared by the whole team: prompts, answer checker, data + frozen split
          (gsm8k_split_v1.json, made by make_gsm8k_split.py), cost
eval/     run_eval.py (vLLM baselines + trained adapters), metrics.py, plot_results.py
train/    single_grpo.py (Matthew), multi_grpo.py (Tiancheng)
tests/    test_answer.py, test_math_answer.py, test_prompts.py, test_data_split.py
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

## Data split (GSM8K, frozen in `common/gsm8k_split_v1.json`)

GSM8K only has train (7,473) and test (1,319), so we carve a dev set out of train:

| split | size | what it is | allowed use |
|-------|------|------------|-------------|
| `train` | 6,973 | official train minus dev, in a **fixed order** | the only data any run trains on |
| `dev` | 500 | held out from official train | checkpoint selection, hyperparameter tuning, sanity checks |
| `test` | 1,319 | official test | **final numbers only**; never used to pick checkpoints or hyperparameters |

* Rule: `perm = numpy.random.default_rng(0).permutation(7473)`; `dev = perm[:500]`, `train = perm[500:]` (same
  membership and order as the original `train.shuffle(seed=0)`), frozen in the JSON so it cannot drift.
* `qid` = row index in the original HF `openai/gsm8k` split (train/dev share the train index space; test has
  its own). qids are stable across runs, so per-problem results can be joined (McNemar, flips, difficulty).
* Training runs consume `common.data.train_stream(n)` **in order, without reshuffling**. Two runs with the same
  number of training prompts see exactly the same problems; a longer run (e.g. the token-matched single-agent
  run) sees a strict superset. Rules for trainers are in `train/README.md`.
* No GSM8K test question appears verbatim in train (checked on whitespace/case-normalized text).

## Data: MATH, levels 3–5

Loaded from [`EleutherAI/hendrycks_math`](https://huggingface.co/datasets/EleutherAI/hendrycks_math)
at a pinned revision (`21a5633`), because the original `hendrycks/competition_math` copy is a loading
script that current `datasets` no longer runs. The mirror has the full 7,500 train / 5,000 test
problems, split by subject, with `level` and `type` fields. Only levels 3, 4 and 5 are kept.

| split | L3 | L4 | L5 | total | what it is |
|-------|----|----|----|-------|------------|
| `dev` | 136 | 165 | 199 | 500 | sampled from level 3–5 train problems (`random.Random(0)`) |
| `test` | 1,131 | 1,214 | 1,324 | 3,669 | every level 3–5 test problem |

* `qid` = `"<subject>/<row index in that subject's split>"`, stable across runs.
* Gold answer = the last `\boxed{}` or `\fbox{}` in `solution`.
* `python -m common.data` prints the per-level counts.

Answer checking (`common/answer.py`, `grade(..., dataset="math")`; the GSM8K checker is unchanged):
1. Extract the last `\boxed{}`/`\fbox{}` (`method = boxed`), else the last number (`last_number`, a
   weak fallback on MATH), else `none`.
2. Exact match after LaTeX normalization (strip `\left`/`\right`, `\!`, `\,`, spaces, `^\circ`,
   `\text{}` units, a trailing period, `x =`; `\dfrac`/`\tfrac` → `\frac`; `0.5` = `\frac{1}{2}`).
3. Otherwise symbolic equivalence with [`math-verify`](https://github.com/huggingface/Math-Verify)
   (SymPy) under a 5 s timeout. Each turn logs `decided_by`: `string`, `symbolic`, `unparsed`,
   `timeout` or `none`; `metrics.py` prints the share of each and lists every timeout.

## Baselines (Qwen2.5-1.5B-Instruct)

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
python -m eval.run_eval --baseline 1 --dataset math --split dev --limit 20   # MATH smoke test
```

`--dataset gsm8k|math` (default `gsm8k`) and `--levels 3,4,5` (MATH only). All four baselines use the
same prompts on both datasets. Max new tokens are fixed per dataset so every baseline gets the same
budget: **512 for GSM8K (same as runs already made), 2,048 for MATH** (recorded in `config.json`; `--max_tokens` overrides
with a warning).

On a T4 (no bfloat16) add `--dtype half`. The script prints one rendered
solver prompt at startup; it must start with Qwen's default system prompt.

Each run writes a directory `outputs/<run_id>/` (default run_id `b<N>_<split>` for GSM8K,
`b<N>_math_<split>` for MATH, plus `_seed<S>` when `--seed` is not 0) holding `predictions.jsonl`
(one line per problem, with `dataset`, `level`, `subject`) and `config.json` (args, dataset name
and revision, system, group, git commit, total `Cost`). MATH runs also print accuracy with a 95%
bootstrap CI for each level.
For trained adapters, pass the training seed as `--seed`.

On Colab, open `colab_eval.ipynb`: set `BASELINE`, `DATASET` (and `ADAPTER` for 3–4)
in the settings cell and run top to bottom.

## Metrics and figures

```
python -m eval.metrics outputs/*/
python -m eval.metrics outputs/*/ --compare outputs/b1_test outputs/b2_test
python eval/plot_results.py outputs/*/ --output_dir figures/
```

`metrics.py` prints accuracy with a 95% bootstrap CI (1,000 resamples), mean
generated tokens, extraction-method and truncation rates, and flip rates for
baselines 2 and 4. For MATH runs it adds per-level accuracy with CIs and the
checker-path shares and timeouts. `--compare` adds a paired bootstrap and exact
McNemar test.

`plot_results.py` writes to `--output_dir`:
- `summary.csv`: one row per run (dataset, accuracy, CI, mean generated/prefill
  tokens, share of problems whose final answer has no `\boxed{}`, share with any
  generation truncated; MATH runs add accuracy, CI and n per level).
- `acc_vs_tokens.png/.pdf`: one point per system; marker = frozen / trained
  single-agent / multi-agent; runs of one system at different budgets
  (`k`, `num_rounds`, `budget` in `config.json`) are connected;
  several seeds are averaged with ±1 std bars.
- `flip_rates.png/.pdf`: solver answer → final answer outcomes for two-step runs.
- Figures 1–2 are drawn per dataset; the MATH versions end in `_math`.
- `acc_by_level.png/.pdf` (when a MATH run is present): accuracy on GSM8K,
  MATH L3, L4, L5, one line per system.

Multi-agent runs appear automatically if they write the same run directory
layout with `"system"` and `"group": "multi_agent"` in `config.json`.

## Sanity checks before trusting numbers

1. `pytest tests/` passes (gold solutions score 100%).
2. Zero-shot CoT is close to the Qwen2.5 technical report; far lower usually
   means a wrong chat template or the base model instead of `-Instruct`.
3. `last_number` + `none` rate under ~5% (on MATH, `last_number` answers are rarely right).
4. Truncation (`finish_reason == "length"`) is low; otherwise raise `max_tokens`.
5. Read 20 random JSONL lines by eye before sharing numbers.
