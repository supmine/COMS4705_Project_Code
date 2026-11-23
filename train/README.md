`single_grpo.py` (Matthew) and `multi_grpo.py` (Tiancheng) go here.

Rule: never write a prompt or an answer checker in this folder. Import them:

```python
from common.prompts import solver_messages, reviser_messages, render  # render = apply_chat_template
from common.answer import grade            # reward = float(grade(text, gold)["correct"])
from common.cost import Cost, Timer, asdict
from common.data import train_stream, records   # frozen split, see below
```

Evaluate a trained adapter with `python -m eval.run_eval --baseline 3 --adapter <path>`
(baseline 4 adds self-revision with the same adapter).

## Training-data rules (all trainers, single- and multi-agent)

1. Train only on `train_stream(n)` (6,973 problems in a fixed order). Never on `dev` or `test`.
2. Consume prompts **in stream order** and do not reshuffle (in TRL: `shuffle_dataset=False`). A run that
   uses `S` optimizer steps × `P` prompts per step trains on exactly `train_stream(S * P)`.
3. Budget matching follows from (2):
   * multi-agent run: `S` steps × `P` prompts → stream positions `[0, S*P)`.
   * single-agent, **step-matched**: same `S`, same `P` → the same problems.
   * single-agent, **token-matched**: keep going in the same stream until its cumulative generated training
     tokens reach the multi-agent run's (≈ `2*S*P` prompts). It sees a superset of the problems.
4. A difficulty filter (e.g. drop problems the base model always / never solves) is optional, must be computed
   on `train` only, and must be applied identically to every arm via `train_stream(n, keep_qids=...)`;
   save the kept qids next to the run and record the file name in the run config.
5. Use `dev` (500) to choose checkpoints / learning rates; report `test` only for the final, pre-chosen
   checkpoints. Log the stream positions (`pos`) and qids each step so every run is auditable.
