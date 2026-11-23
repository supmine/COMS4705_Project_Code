"""GRPO core shared by the single- and multi-agent trainers: rewards, group-normalized
advantages, the masked policy-gradient loss (+ optional k3 KL to the base model), one
optimizer update per step, and per-step statistics.

One optimizer update per batch of rollouts (mu = 1), so the policy ratio is 1 and the
per-token loss is  -A * log pi(token)  (+ beta * KL). No old log-probs are stored.
Log-probs always come from the training model, never from vLLM.
"""
import math
from dataclasses import dataclass, field

import torch

from common.answer import grade

ADV_EPS = 1e-4


@dataclass
class Turn:
    role: str                  # "solver" | "reviser"
    prompt_ids: list           # exactly the prompt tokens vLLM generated from
    completion_ids: list       # tokens the agent generated (the only tokens in the loss)
    text: str
    finish_reason: str
    pred: object = None
    method: str = None
    correct: bool = False      # graded under the reward rule (boxed-only if required)
    reward: float = 0.0
    advantage: float = 0.0


@dataclass
class Rollout:
    """One (joint) rollout: a single solver turn, or solver -> reviser [-> solver -> reviser]."""
    problem: dict
    turns: list = field(default_factory=list)
    reward: float = 0.0        # reward of the final answer


# ---------------------------------------------------------------- rewards and advantages

def turn_reward(turn, gold, require_boxed):
    """1 if the turn's answer is correct, 0 otherwise; 0 if truncated by max tokens."""
    g = grade(turn.text, gold, allowed_methods={"boxed"} if require_boxed else None)
    turn.pred, turn.method, turn.correct = g["pred"], g["method"], g["correct"]
    return float(g["correct"] and turn.finish_reason != "length")


def assign_rewards(groups, reward_mode="team", require_boxed=True):
    """team: every turn of a joint rollout gets the final answer's reward.
    per_agent: solver turns get their own answer's reward; reviser turns the final one."""
    if reward_mode not in ("team", "per_agent"):
        raise ValueError(f"unknown reward_mode {reward_mode!r}")
    for group in groups:
        for ro in group:
            own = [turn_reward(t, ro.problem["gold"], require_boxed) for t in ro.turns]
            ro.reward = own[-1]
            for t, r in zip(ro.turns, own):
                t.reward = r if (reward_mode == "per_agent" and t.role == "solver") else ro.reward


def compute_advantages(groups):
    """Per problem, over its G (joint) rollouts and separately for each turn position:
    A_i = (r_i - mean(r)) / (std(r) + 1e-4), std = population std over the group.
    In team mode all turns of a rollout share one advantage."""
    for group in groups:
        for j in range(len(group[0].turns)):
            rs = [ro.turns[j].reward for ro in group]
            mean = sum(rs) / len(rs)
            std = math.sqrt(sum((r - mean) ** 2 for r in rs) / len(rs))
            for ro, r in zip(group, rs):
                ro.turns[j].advantage = (r - mean) / (std + ADV_EPS)


# ---------------------------------------------------------------- batching and log-probs

@dataclass
class Batch:
    role: str
    input_ids: torch.Tensor        # (B, L), left-padded prompt + completion
    attention_mask: torch.Tensor   # (B, L)
    position_ids: torch.Tensor     # (B, L)
    targets: torch.Tensor          # (B, K) the last K tokens (completions are right-aligned)
    mask: torch.Tensor             # (B, K) 1 on generated tokens only
    advantages: torch.Tensor       # (B,)

    @property
    def n_tokens(self):
        return int(self.mask.sum().item())


def collate(turns, pad_id, device="cpu"):
    """Left-pad prompt+completion so every completion ends at the last position; then the
    last K = max completion length positions hold all generated tokens."""
    K = max(len(t.completion_ids) for t in turns)
    L = max(len(t.prompt_ids) + len(t.completion_ids) for t in turns)
    ids = torch.full((len(turns), L), pad_id, dtype=torch.long)
    attn = torch.zeros((len(turns), L), dtype=torch.long)
    mask = torch.zeros((len(turns), K), dtype=torch.float32)
    for i, t in enumerate(turns):
        seq = list(t.prompt_ids) + list(t.completion_ids)
        ids[i, L - len(seq):] = torch.tensor(seq, dtype=torch.long)
        attn[i, L - len(seq):] = 1
        mask[i, K - len(t.completion_ids):] = 1.0
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    adv = torch.tensor([t.advantage for t in turns], dtype=torch.float32)
    return Batch(turns[0].role, ids.to(device), attn.to(device), pos.to(device),
                 ids[:, L - K:].to(device), mask.to(device), adv.to(device))


def selective_log_softmax(logits, targets):
    """log p(target) per position, one row at a time in fp32 to bound memory."""
    out = []
    for row_logits, row_targets in zip(logits, targets):
        row = row_logits.float()
        out.append(row.gather(-1, row_targets[:, None]).squeeze(-1) - torch.logsumexp(row, -1))
    return torch.stack(out)


def completion_logprobs(model, batch):
    """(B, K) log-probs of the last K tokens under `model` (only K+1 positions of logits)."""
    K = batch.targets.shape[1]
    out = model(input_ids=batch.input_ids, attention_mask=batch.attention_mask,
                position_ids=batch.position_ids, logits_to_keep=K + 1, use_cache=False)
    return selective_log_softmax(out.logits[:, :-1], batch.targets)


def micro_batches(turns, max_tokens, max_seqs=None):
    """Group turns by role, sort by length, and pack so padded size (rows * longest)
    stays within max_tokens and rows within max_seqs (if set). A single over-long
    sequence gets its own micro-batch."""
    batches = []
    for role in sorted({t.role for t in turns}):
        cur, cur_len = [], 0
        for t in sorted((t for t in turns if t.role == role),
                        key=lambda t: len(t.prompt_ids) + len(t.completion_ids)):
            n = len(t.prompt_ids) + len(t.completion_ids)
            if cur and ((len(cur) + 1) * max(cur_len, n) > max_tokens
                        or (max_seqs and len(cur) >= max_seqs)):
                batches.append(cur)
                cur, cur_len = [], 0
            cur.append(t)
            cur_len = max(cur_len, n)
        if cur:
            batches.append(cur)
    return batches


# ---------------------------------------------------------------- one optimizer update

def kl_k3(logp, ref_logp):
    """k3 estimator of KL(pi || ref) per token: exp(ref - logp) - (ref - logp) - 1 >= 0."""
    d = ref_logp - logp
    return torch.exp(d) - d - 1


def train_step(model, optimizer, scheduler, turns, *, pad_id, set_role, beta=0.0,
               loss_agg="token_mean", micro_batch_tokens=4096, micro_batch_size=None,
               max_grad_norm=1.0, device="cpu"):
    """One optimizer update over every generated turn of the step.

    loss = sum over generated tokens of (-A * log pi + beta * k3) / (number of generated
    tokens in the batch), accumulated over micro-batches. `set_role(role)` activates the
    adapter for that role before its micro-batches. With beta = 0, turns with A = 0 add
    nothing to the loss and are skipped (they still count in the denominator).
    Returns dict(loss, kl, grad_norm, bwd_tokens).
    """
    if loss_agg != "token_mean":
        raise ValueError(f"unsupported loss_agg {loss_agg!r}")
    n_total = sum(len(t.completion_ids) for t in turns)
    active = [t for t in turns if t.completion_ids and (beta > 0 or t.advantage != 0)]
    params = [p for g in optimizer.param_groups for p in g["params"]]
    optimizer.zero_grad(set_to_none=True)
    loss_sum, kl_sum, bwd_tokens = 0.0, 0.0, 0
    for mb in micro_batches(active, micro_batch_tokens, micro_batch_size):
        set_role(mb[0].role)
        batch = collate(mb, pad_id, device)
        logp = completion_logprobs(model, batch)
        tok_loss = -batch.advantages[:, None] * logp
        if beta > 0:
            with torch.no_grad(), model.disable_adapter():
                ref = completion_logprobs(model, batch)
            kl = kl_k3(logp, ref)
            tok_loss = tok_loss + beta * kl
            kl_sum += float((kl.detach() * batch.mask).sum())
        loss = (tok_loss * batch.mask).sum() / n_total
        loss.backward()
        loss_sum += float(loss.detach())
        bwd_tokens += batch.n_tokens
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, max_grad_norm)) if bwd_tokens else 0.0
    optimizer.step()
    scheduler.step()
    return {"loss": loss_sum, "kl": kl_sum / max(n_total, 1), "grad_norm": grad_norm,
            "bwd_tokens": bwd_tokens}


def make_optimizer(params, lr=1e-5, weight_decay=0.0, warmup_steps=10):
    """AdamW with linear warmup to `lr`, then constant."""
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: min(1.0, (s + 1) / warmup_steps) if warmup_steps > 0 else 1.0)
    return optimizer, scheduler


# ---------------------------------------------------------------- statistics

def step_stats(groups):
    """Per-step training statistics (see train_log.jsonl)."""
    rollouts = [ro for g in groups for ro in g]
    turns = [t for ro in rollouts for t in ro.turns]
    rewards = [ro.reward for ro in rollouts]
    mean = sum(rewards) / len(rewards)
    stats = {
        "reward_mean": mean,
        "reward_std": math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards)),
        "frac_zero_std_groups": sum(len({ro.reward for ro in g}) == 1 for g in groups) / len(groups),
        "frac_truncated": sum(t.finish_reason == "length" for t in turns) / len(turns),
        "frac_boxed": sum(t.method == "boxed" for t in turns) / len(turns),
    }
    for role in sorted({t.role for t in turns}):
        lens = [len(t.completion_ids) for t in turns if t.role == role]
        stats[f"gen_len_{role}"] = sum(lens) / len(lens)
    if any(len(ro.turns) > 1 for ro in rollouts):
        # First solver turn vs final answer, and whether the reviser kept the solver's answer.
        wrong = [ro for ro in rollouts if not ro.turns[0].correct]
        right = [ro for ro in rollouts if ro.turns[0].correct]
        stats["p_wrong_to_right"] = (sum(ro.turns[-1].correct for ro in wrong) / len(wrong)
                                     if wrong else None)
        stats["p_right_to_wrong"] = (sum(not ro.turns[-1].correct for ro in right) / len(right)
                                     if right else None)
        stats["frac_reviser_keeps"] = sum(ro.turns[1].pred == ro.turns[0].pred
                                          for ro in rollouts) / len(rollouts)
    return stats
