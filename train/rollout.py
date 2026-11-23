"""Rollouts with vLLM on the training GPU, LoRA weight sync, and the dev evaluation.

single: G solver samples per problem.
multi:  G solver samples per problem; each gets one reviser sample whose {solution} is
        that solver output -> G joint rollouts per problem. With n_rounds = 2 the reviser
        output is fed back as a new solver turn and revised again
        (solver -> reviser -> solver -> reviser). The second solver turn uses the reviser
        prompt (the only prompt that takes a solution) with the solver's adapter.

Training uses vLLM's own prompt_token_ids, so the log-probs are computed on exactly the
tokens vLLM generated from.
"""
from types import SimpleNamespace

from common.prompts import render, reviser_messages, solver_messages
from train.grpo import Rollout, Turn


class VLLMEngine:
    """vLLM with LoRA enabled. After every optimizer step `set_adapters` registers the newly
    saved adapter directories under NEW integer ids: vLLM caches adapters by id, so reusing
    an id would keep serving the old weights."""

    def __init__(self, cfg, tokenizer):
        from vllm import LLM

        self.tok = tokenizer
        self.llm = LLM(model=cfg["model"], dtype=cfg["dtype"], seed=cfg["seed"],
                       enable_lora=True, max_lora_rank=cfg["lora"]["r"], max_loras=2,
                       gpu_memory_utilization=cfg["gpu_memory_utilization"],
                       max_model_len=cfg["max_model_len"], enable_prefix_caching=True)
        self.loras = {}
        self._next_id = 1

    def set_adapters(self, paths):
        """paths: role -> adapter directory. Roles sharing a directory share one request."""
        from vllm.lora.request import LoRARequest

        by_path = {}
        for role, path in paths.items():
            if path not in by_path:
                by_path[path] = LoRARequest(f"{role}_{self._next_id}", self._next_id, path)
                self._next_id += 1
            self.loras[role] = by_path[path]

    def prompt(self, messages):
        return render(self.tok, messages)

    def generate(self, messages_list, params, role="solver"):
        return self.llm.generate([self.prompt(m) for m in messages_list], params,
                                 lora_request=self.loras.get(role), use_tqdm=False)

    def runner(self, role="solver"):
        """Object with run_eval's Runner interface: generate(messages_list, params)."""
        return SimpleNamespace(generate=lambda msgs, params: self.generate(msgs, params, role))


def _turn(role, out, k):
    c = out.outputs[k]
    return Turn(role, list(out.prompt_token_ids), list(c.token_ids), c.text, c.finish_reason)


def rollout_single(engine, problems, G, make_params, cost):
    outs = engine.generate([solver_messages(p["question"]) for p in problems], make_params(G), "solver")
    groups = []
    for p, out in zip(problems, outs):
        cost.add_vllm_output(out)
        groups.append([Rollout(p, [_turn("solver", out, k)]) for k in range(G)])
    return groups


def rollout_multi(engine, problems, G, make_params, cost, n_rounds=1):
    outs = engine.generate([solver_messages(p["question"]) for p in problems], make_params(G), "solver")
    groups = []
    for p, out in zip(problems, outs):
        cost.add_vllm_output(out)
        groups.append([Rollout(p, [_turn("solver", out, k)]) for k in range(G)])
    flat = [ro for g in groups for ro in g]
    for role in ["reviser"] + ["solver", "reviser"] * (n_rounds - 1):
        msgs = [reviser_messages(ro.problem["question"], ro.turns[-1].text) for ro in flat]
        for ro, out in zip(flat, engine.generate(msgs, make_params(1), role)):
            cost.add_vllm_output(out)
            ro.turns.append(_turn(role, out, 0))
    return groups


# ---------------------------------------------------------------- dev evaluation

def evaluate_dev(engine, problems, mode, n_rounds=1, run_id="train_dev_eval", dataset="gsm8k"):
    """Greedy dev evaluation of the adapters currently loaded in `engine`.
    single: eval/run_eval.py's baseline 3 pipeline. multi: solver -> reviser with the trained
    adapters (run_eval supports one adapter for both steps, so the turns are built here with
    run_eval.make_turn and the same greedy settings). Returns (rows, eval Cost)."""
    from eval import run_eval

    args = SimpleNamespace(baseline=3 if mode == "single" else 4, run_id=run_id, dataset=dataset,
                           max_tokens=run_eval.MAX_TOKENS[dataset])
    if mode == "single":
        return run_eval.run(args, problems, engine.runner("solver"))

    from vllm import SamplingParams

    from common.answer import normalize
    from common.cost import Cost, Timer

    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    cost = Cost()
    questions = [p["question"] for p in problems]
    rows = [{"run_id": run_id, "system": "multi_agent_grpo", "dataset": dataset, "qid": p["qid"],
             "gold": normalize(p["gold"]), "turns": []} for p in problems]
    prev = None
    for i, role in enumerate(["solver", "reviser"] * n_rounds):
        msgs = ([solver_messages(q) for q in questions] if i == 0
                else [reviser_messages(q, t) for q, t in zip(questions, prev)])
        with Timer(cost, "gen_seconds"):
            outs = engine.generate(msgs, greedy, role)
        for row, p, out in zip(rows, problems, outs):
            cost.add_vllm_output(out)
            row["turns"].append(run_eval.make_turn(role, out, p["gold"], dataset))
        prev = [o.outputs[0].text for o in outs]
    for row in rows:
        row["final_pred"] = row["turns"][-1]["pred"]
        row["final_correct"] = row["turns"][-1]["correct"]
        row["total_gen_tokens"] = sum(t["gen_tokens"] for t in row["turns"])
        row["total_prefill_tokens"] = sum(t["prefill_tokens"] for t in row["turns"])
    return rows, cost
