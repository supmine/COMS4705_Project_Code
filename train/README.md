`single_grpo.py` (Matthew) and `multi_grpo.py` (Tiancheng) go here.

Rule: never write a prompt or an answer checker in this folder. Import them:

```python
from common.prompts import solver_messages, reviser_messages
from common.answer import grade            # reward = float(grade(text, gold)["correct"])
from common.cost import Cost, Timer, asdict
from common.data import load_gsm8k         # same seed as eval
```

Evaluate a trained adapter with `python -m eval.run_eval --system cot --solver_lora <path>`.
