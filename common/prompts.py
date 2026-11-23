"""Every prompt the team uses lives here. train/ must import from this file.

Prompts are identical for baselines and training; do not change them.
No custom system message: Qwen's chat template adds its default system prompt.
Baseline 4 (self-revision) uses reviser_messages() with the solver's adapter.
"""

SOLVER_TEMPLATE = (
    "{question}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)

REVISER_TEMPLATE = (
    "{question}\n\n"
    "Here is a proposed solution, which may or may not be correct:\n"
    "{solution}\n\n"
    "Check the solution step by step. If it is correct, restate its final "
    "answer. If it contains a mistake, correct it.\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}."
)


def solver_messages(question):
    return [{"role": "user", "content": SOLVER_TEMPLATE.format(question=question)}]


def reviser_messages(question, solution):
    content = REVISER_TEMPLATE.format(question=question, solution=solution)
    return [{"role": "user", "content": content}]


def render(tokenizer, messages):
    """Chat-formatted prompt string, as fed to vLLM and to training rollouts."""
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
