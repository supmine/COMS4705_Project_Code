# GRPO training (single- and multi-agent)

One trainer, two modes, same loss / advantages / optimizer / logging:

```
train/grpo.py        rewards, group-normalized advantages, masked loss (+ k3 KL), one optimizer step
train/rollout.py     vLLM rollouts on the training GPU, LoRA weight sync, dev evaluation
train/run_train.py   entry point (YAML config)
train/plot_curve.py  training-curve figure
train/configs/       multi.yaml, single_step_matched.yaml, single_token_matched.yaml
train/test_grpo.py   CPU tests (tiny random Qwen2 + fake vLLM): pytest train/
```

```
python -m train.run_train train/configs/multi.yaml --smoke          # 2 tiny steps, check the pipeline
python -m train.run_train train/configs/multi.yaml
python -m train.run_train train/configs/single_step_matched.yaml --set lr=2e-5   # override any key
python -m train.plot_curve runs/multi runs/single_step_matched runs/single_token_matched \
    --output figures/training_curve
```

Needs a bf16 GPU (L4 / A100; not a T4). vLLM takes `gpu_memory_utilization` (0.35) of the GPU
and is created before the training model. If the backward pass runs out of memory, lower
`micro_batch_tokens` (padded tokens per micro-batch; one step is still one optimizer update).

| | `single` | `multi` |
|---|---|---|
| adapters | `solver` | `solver`, `reviser` (`shared_adapter: true` → both roles use `solver`) |
| rollout per problem | G solver samples | G solver samples, each revised once → G joint rollouts (`n_rounds: 2` → solver → reviser → solver → reviser) |
| loss | solver tokens | solver tokens with `solver` active + reviser tokens with `reviser` active, same advantage per joint rollout (`reward_mode: team`), one optimizer step over both |
| dev eval | run_eval baseline 3 pipeline | solver → reviser with the trained adapters |

Reward: `is_correct` on the final turn, `\boxed{}` only (`reward_require_boxed`), 0 if truncated.
Advantage per problem over its G rollouts: `(r - mean) / (std + 1e-4)`. Loss: `-A * log pi(token)`
(+ `beta` * k3 KL to the base model = adapters disabled), `token_mean` over every generated
token of the step. Only generated tokens are in the loss (the solver text inside the reviser
prompt is masked). μ = 1, so no old log-probs.

Outputs in `runs/<run_name>/`: `config.json` (resolved config, git commit, GPU, versions),
`train_log.jsonl` (one line per step; a step-0 line holds the dev accuracy before training),
`dev_eval/step_<k>.jsonl`, `checkpoints/step_<k>/<adapter>` (every `save_every` steps and at the end).

Evaluate a single-agent checkpoint on test with
`python -m eval.run_eval --baseline 3 --adapter runs/<run>/checkpoints/step_<k>/solver`
(baseline 4 adds self-revision with the same adapter).

Rule: never write a prompt or an answer checker in this folder. Import them:

```python
from common.prompts import solver_messages, reviser_messages, render  # render = apply_chat_template
from common.answer import grade            # reward = float(grade(text, gold)["correct"])
from common.cost import Cost, Timer, asdict
from common.data import train_stream, records   # frozen split, see below
```

## Training-data rules (all trainers, single- and multi-agent)

1. Train only on `train_stream(n)` (6,973 problems in a fixed order). Never on `dev` or `test`.
2. Consume prompts **in stream order** and do not reshuffle (in TRL: `shuffle_dataset=False`). A run that
   uses `S` optimizer steps × `P` prompts per step trains on exactly `train_stream(S * P)`.
3. Budget matching follows from (2) (`train/configs/`):
   * multi-agent run: `S` steps × `P` prompts, G = 8 → stream positions `[0, S*P)`.
   * single-agent, **step-matched**: same `S`, same `P`, G = 8 → the same problems.
   * single-agent, **token-matched**: same `S`, same `P`, **G = 16** (two generations per joint rollout in
     the multi-agent run ↔ twice the samples), so it sees the same problems and makes the same number of
     optimizer updates. Compare runs on dev accuracy vs. cumulative generated tokens (`plot_curve.py`).
4. A difficulty filter (e.g. drop problems the base model always / never solves) is optional, must be computed
   on `train` only, and must be applied identically to every arm via `train_stream(n, keep_qids=...)`;
   save the kept qids next to the run and record the file name in the run config.
5. Use `dev` (500) to choose checkpoints / learning rates; report `test` only for the final, pre-chosen
   checkpoints. Log the stream positions (`pos`) and qids each step so every run is auditable.
