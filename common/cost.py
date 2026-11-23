"""Cost accounting on every axis, shared by eval and the training loops.

In a training loop:
    with Timer(cost, "gen_seconds"):  ...rollout...
    with Timer(cost, "bwd_seconds"):  ...backward...
and log asdict(cost) every N steps. Count bwd_tokens only where the loss mask is 1.
"""
import time
from dataclasses import dataclass, asdict  # noqa: F401  (asdict re-exported)


@dataclass
class Cost:
    gen_tokens: int = 0       # tokens the model generated
    prefill_tokens: int = 0   # prompt tokens read (incl. solver output read by reviser)
    bwd_tokens: int = 0       # tokens that received gradient (train only)
    optimizer_steps: int = 0
    gen_seconds: float = 0.0
    bwd_seconds: float = 0.0

    def add_vllm_output(self, out):
        self.prefill_tokens += len(out.prompt_token_ids)
        self.gen_tokens += sum(len(o.token_ids) for o in out.outputs)

    def flops(self, n_params, c_bwd=4.0):
        return (2 * n_params * (self.gen_tokens + self.prefill_tokens)
                + c_bwd * n_params * self.bwd_tokens)


class Timer:
    def __init__(self, cost, field):
        self.cost, self.field = cost, field

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        setattr(self.cost, self.field,
                getattr(self.cost, self.field) + time.perf_counter() - self.t)
