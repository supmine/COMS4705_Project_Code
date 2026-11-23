"""CPU tests for the GRPO trainer: a tiny random Qwen2 model stands in for Qwen2.5-1.5B and a
fake vLLM stands in for generation.  Run:  pytest train/"""
import sys
import types
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")

from train import grpo  # noqa: E402
from train.grpo import Rollout, Turn  # noqa: E402

PAD = 0


# ---------------------------------------------------------------- helpers

def tiny_policy(adapters=("solver",), seed=0):
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    lc = LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                    "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(Qwen2ForCausalLM(cfg), lc, adapter_name=adapters[0])
    for a in adapters[1:]:
        model.add_adapter(a, lc)
    for n, p in model.named_parameters():
        if "lora_" in n:
            p.requires_grad_(True)
            if "lora_B" in n:   # non-zero B so adapters actually change the outputs
                torch.nn.init.normal_(p, std=0.5)
    return model


def set_role_for(model, mapping):
    from train.run_train import role_setter
    return role_setter(model, mapping)


def turn(role, prompt, completion, adv=1.0, finish="stop", text=""):
    return Turn(role, list(prompt), list(completion), text, finish, advantage=adv)


def random_turns(n, role="solver", seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        lp, lc = int(torch.randint(3, 9, (1,), generator=g)), int(torch.randint(1, 7, (1,), generator=g))
        out.append(turn(role, torch.randint(1, 64, (lp,), generator=g).tolist(),
                        torch.randint(1, 64, (lc,), generator=g).tolist(),
                        adv=float(torch.randn(1, generator=g))))
    return out


def lora_grads(model, adapter):
    return [p.grad for n, p in model.named_parameters() if f".{adapter}." in n and "lora_" in n]


# ---------------------------------------------------------------- rewards and advantages

def test_advantages_group_normalized():
    group = [Rollout({"gold": "1"}, [turn("solver", [1], [2])]) for _ in range(4)]
    for ro, r in zip(group, [1.0, 0.0, 0.0, 0.0]):
        ro.turns[0].reward = r
    grpo.compute_advantages([group])
    adv = [ro.turns[0].advantage for ro in group]
    mean, std = 0.25, (0.25 * 0.75) ** 0.5
    assert adv[0] == pytest.approx((1 - mean) / (std + 1e-4))
    assert adv[1] == pytest.approx((0 - mean) / (std + 1e-4))
    assert sum(adv) == pytest.approx(0, abs=1e-6)


def test_zero_std_group_has_zero_advantage():
    group = [Rollout({"gold": "1"}, [turn("solver", [1], [2])]) for _ in range(3)]
    for ro in group:
        ro.turns[0].reward = 1.0
    grpo.compute_advantages([group])
    assert all(ro.turns[0].advantage == 0 for ro in group)


def test_rewards_truncation_boxed_and_modes():
    gold = "5"
    def ro():
        return Rollout({"gold": gold}, [
            Turn("solver", [1], [2], r"so \boxed{4}", "stop"),       # solver wrong
            Turn("reviser", [1], [2], r"fixed: \boxed{5}", "stop")])  # final right
    team, per_agent = [ro()], [ro()]
    grpo.assign_rewards([team], "team", True)
    grpo.assign_rewards([per_agent], "per_agent", True)
    assert [t.reward for t in team[0].turns] == [1.0, 1.0]
    assert [t.reward for t in per_agent[0].turns] == [0.0, 1.0]

    truncated = [Rollout({"gold": gold}, [Turn("solver", [1], [2], r"\boxed{5}", "length")])]
    unboxed = [Rollout({"gold": gold}, [Turn("solver", [1], [2], "the answer is 5", "stop")])]
    grpo.assign_rewards([truncated], "team", True)
    grpo.assign_rewards([unboxed], "team", True)
    assert truncated[0].reward == 0.0 and unboxed[0].reward == 0.0
    grpo.assign_rewards([unboxed], "team", False)
    assert unboxed[0].reward == 1.0


# ---------------------------------------------------------------- masking and log-probs

def test_collate_masks_only_generated_tokens():
    b = grpo.collate([turn("reviser", [5, 6, 7, 8], [9, 10]), turn("reviser", [5, 6], [9, 10, 11])], PAD)
    assert b.input_ids.tolist() == [[5, 6, 7, 8, 9, 10], [0, 5, 6, 9, 10, 11]]
    assert b.mask.tolist() == [[0, 1, 1], [1, 1, 1]]          # prompt (incl. solver text) masked
    assert b.targets.tolist() == [[8, 9, 10], [9, 10, 11]]
    assert b.n_tokens == 5


def test_logprobs_match_unpadded_reference():
    """Left padding + position_ids + logits_to_keep give the same log-probs as running each
    sequence alone with full logits."""
    model = tiny_policy()
    model.eval()
    turns = random_turns(5)
    batch = grpo.collate(turns, PAD)
    with torch.no_grad():
        logp = grpo.completion_logprobs(model, batch)
        for i, t in enumerate(turns):
            ids = torch.tensor([t.prompt_ids + t.completion_ids])
            ref = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
            c = len(t.completion_ids)
            want = ref[len(t.prompt_ids) - 1:-1].gather(-1, torch.tensor(t.completion_ids)[:, None])[:, 0]
            got = logp[i, batch.mask[i].bool()]
            assert torch.allclose(got, want, atol=1e-5), (i, got, want)
            assert got.shape[0] == c


def _step_grads(micro_batch_tokens, beta=0.0):
    model = tiny_policy()
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt, sched = grpo.make_optimizer(params, lr=1e-3, warmup_steps=0)
    turns = random_turns(6, seed=1)
    out = grpo.train_step(model, opt, sched, turns, pad_id=PAD, set_role=lambda r: None, beta=beta,
                          micro_batch_tokens=micro_batch_tokens, max_grad_norm=1e9)
    return out, [p.grad.clone() for p in params]


def test_token_mean_is_independent_of_micro_batching():
    full, g_full = _step_grads(micro_batch_tokens=10_000)
    small, g_small = _step_grads(micro_batch_tokens=12)      # many micro-batches
    assert full["loss"] == pytest.approx(small["loss"], rel=1e-5)
    assert full["bwd_tokens"] == small["bwd_tokens"]
    for a, b in zip(g_full, g_small):
        assert torch.allclose(a, b, atol=1e-6)


def test_loss_is_token_mean_of_minus_adv_logp():
    model = tiny_policy()
    turns = random_turns(4, seed=2)
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt, sched = grpo.make_optimizer(params, lr=0.0, warmup_steps=0)
    with torch.no_grad():
        b = grpo.collate(turns, PAD)
        logp = grpo.completion_logprobs(model, b)
        want = float((-b.advantages[:, None] * logp * b.mask).sum() / b.mask.sum())
    out = grpo.train_step(model, opt, sched, turns, pad_id=PAD, set_role=lambda r: None)
    assert out["loss"] == pytest.approx(want, rel=1e-5)


def test_zero_advantage_turns_count_in_denominator_only():
    turns = random_turns(4, seed=3)
    for t in turns[2:]:
        t.advantage = 0.0
    model = tiny_policy()
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt, sched = grpo.make_optimizer(params, lr=0.0, warmup_steps=0)
    out = grpo.train_step(model, opt, sched, turns, pad_id=PAD, set_role=lambda r: None)
    assert out["bwd_tokens"] == sum(len(t.completion_ids) for t in turns[:2])
    with torch.no_grad():
        b = grpo.collate(turns[:2], PAD)
        logp = grpo.completion_logprobs(model, b)
        want = float((-b.advantages[:, None] * logp * b.mask).sum()) / sum(len(t.completion_ids) for t in turns)
    assert out["loss"] == pytest.approx(want, rel=1e-5)


# ---------------------------------------------------------------- KL

def test_kl_zero_when_adapter_is_identity_and_positive_otherwise():
    model = tiny_policy()
    turns = random_turns(3, seed=4)
    b = grpo.collate(turns, PAD)
    with torch.no_grad():
        logp = grpo.completion_logprobs(model, b)
        with model.disable_adapter():
            ref = grpo.completion_logprobs(model, b)
        assert float((grpo.kl_k3(logp, ref) * b.mask).sum()) > 0
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.zero_()
        logp0 = grpo.completion_logprobs(model, b)
        assert torch.allclose(grpo.kl_k3(logp0, ref) * b.mask, torch.zeros_like(b.mask), atol=1e-6)


def test_beta_adds_kl_term():
    out0, _ = _step_grads(10_000, beta=0.0)
    out1, _ = _step_grads(10_000, beta=0.5)
    assert out1["kl"] > 0
    assert out1["loss"] == pytest.approx(out0["loss"] + 0.5 * out1["kl"], rel=1e-4)


# ---------------------------------------------------------------- multi-agent adapters

def test_multi_each_role_trains_only_its_adapter_and_one_step_updates_both():
    model = tiny_policy(("solver", "reviser"))
    set_role = set_role_for(model, {"solver": "solver", "reviser": "reviser"})
    params = [p for n, p in model.named_parameters() if "lora_" in n]
    opt, sched = grpo.make_optimizer(params, lr=1e-2, warmup_steps=0)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}

    solver_only = random_turns(3, "solver", seed=5)
    grpo.train_step(model, opt, sched, solver_only, pad_id=PAD, set_role=set_role)
    assert all(g is not None and g.abs().sum() > 0 for g in lora_grads(model, "solver"))
    assert all(g is None for g in lora_grads(model, "reviser"))

    both = random_turns(3, "solver", seed=6) + random_turns(3, "reviser", seed=7)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    grpo.train_step(model, opt, sched, both, pad_id=PAD, set_role=set_role)
    changed = {a: any(not torch.equal(before[n], p) for n, p in model.named_parameters()
                      if f".{a}." in n and "lora_" in n) for a in ("solver", "reviser")}
    assert changed == {"solver": True, "reviser": True}


def test_reviser_logprobs_use_reviser_adapter():
    model = tiny_policy(("solver", "reviser"))
    set_role = set_role_for(model, {"solver": "solver", "reviser": "reviser"})
    b = grpo.collate(random_turns(2, "reviser", seed=8), PAD)
    with torch.no_grad():
        set_role("solver")
        as_solver = grpo.completion_logprobs(model, b)
        set_role("reviser")
        as_reviser = grpo.completion_logprobs(model, b)
    assert not torch.allclose(as_solver, as_reviser)


# ---------------------------------------------------------------- rollouts (fake vLLM)

class FakeEngine:
    """Records every call; solver says \\boxed{1}, reviser \\boxed{2}."""
    def __init__(self):
        self.calls = []

    def generate(self, messages_list, params, role="solver"):
        self.calls.append((role, params, messages_list))
        outs = []
        for i, m in enumerate(messages_list):
            n = params
            comps = [NS(text=f"{role} {i}.{k} \\boxed{{{1 if role == 'solver' else 2}}}",
                        token_ids=[10 + k, 11], finish_reason="stop") for k in range(n)]
            outs.append(NS(prompt_token_ids=[1, 2, 3, i], outputs=comps))
        return outs

    def runner(self, role="solver"):
        from train.rollout import VLLMEngine
        return VLLMEngine.runner(self, role)


def test_rollout_single_shapes():
    from common.cost import Cost
    from train.rollout import rollout_single
    eng, cost = FakeEngine(), Cost()
    groups = rollout_single(eng, [{"question": "q1", "gold": "1"}, {"question": "q2", "gold": "2"}],
                            4, lambda n: n, cost)
    assert [len(g) for g in groups] == [4, 4]
    assert all(len(ro.turns) == 1 and ro.turns[0].role == "solver" for g in groups for ro in g)
    assert cost.gen_tokens == 2 * 4 * 2 and cost.prefill_tokens == 2 * 4
    assert groups[0][0].turns[0].prompt_ids == [1, 2, 3, 0]   # vLLM's own prompt tokens


@pytest.mark.parametrize("n_rounds,roles", [(1, ["solver", "reviser"]),
                                            (2, ["solver", "reviser", "solver", "reviser"])])
def test_rollout_multi_joint_rollouts(n_rounds, roles):
    from common.cost import Cost
    from train.rollout import rollout_multi
    eng = FakeEngine()
    groups = rollout_multi(eng, [{"question": "q1", "gold": "2"}, {"question": "q2", "gold": "2"}],
                           3, lambda n: n, Cost(), n_rounds)
    assert [len(g) for g in groups] == [3, 3]
    assert all([t.role for t in ro.turns] == roles for g in groups for ro in g)
    assert [c[0] for c in eng.calls] == roles
    assert [c[1] for c in eng.calls] == [3] + [1] * (len(roles) - 1)   # G solver samples, then 1 each
    # Each reviser prompt contains the solver output of its own joint rollout.
    reviser_prompt = eng.calls[1][2][4][0]["content"]
    assert groups[1][1].turns[0].text in reviser_prompt and "q2" in reviser_prompt
    grpo.assign_rewards(groups, "team", True)
    grpo.compute_advantages(groups)
    stats = grpo.step_stats(groups)
    assert stats["reward_mean"] == 1.0 and stats["frac_zero_std_groups"] == 1.0
    assert stats["p_wrong_to_right"] == 1.0 and stats["frac_reviser_keeps"] == 0.0


def test_evaluate_dev_multi_uses_both_roles(monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda temperature, max_tokens: 1
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    from train.rollout import evaluate_dev
    eng = FakeEngine()
    problems = [{"qid": 0, "question": "q", "gold": "2"}, {"qid": 1, "question": "r", "gold": "1"}]
    rows, _ = evaluate_dev(eng, problems, "multi")
    assert [c[0] for c in eng.calls] == ["solver", "reviser"]
    assert [r["final_correct"] for r in rows] == [True, False]
    assert [t["agent"] for t in rows[0]["turns"]] == ["solver", "reviser"]

    eng = FakeEngine()
    rows, _ = evaluate_dev(eng, problems, "single")
    assert [c[0] for c in eng.calls] == ["solver"]


# ---------------------------------------------------------------- configs

def test_configs_match_except_mode_and_group_size():
    from train.run_train import load_config
    m = load_config("train/configs/multi.yaml")
    s = load_config("train/configs/single_step_matched.yaml")
    t = load_config("train/configs/single_token_matched.yaml")
    diff = lambda a, b: {k for k in a if a[k] != b[k]}
    assert diff(m, s) == {"run_name", "mode"}
    assert diff(s, t) == {"run_name", "group_size"}
    assert (m["batch_problems"], m["group_size"], t["group_size"]) == (16, 8, 16)
    smoke = load_config("train/configs/multi.yaml", ["lr=2e-5", "lora.r=8"], smoke=True)
    assert smoke["lr"] == 2e-5 and smoke["lora"]["r"] == 8 and smoke["steps"] == 2
