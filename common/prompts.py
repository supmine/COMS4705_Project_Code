"""Every prompt the team uses lives here. train/ must import from this file.

The self-revision baseline uses reviser_messages() verbatim; only the adapter
changes. Otherwise a gap between solver_reviser and self_revision could come
from the prompt instead of from the second agent.
"""

SOLVER_SYSTEM = ("Solve the math problem step by step. "
                 "Put the final answer in \\boxed{}.")

REVISER_SYSTEM = ("You are given a math problem and a proposed solution. "
                  "Check each step. If the solution is correct, restate its "
                  "final answer; otherwise, fix it. "
                  "Put the final answer in \\boxed{}.")


def solver_messages(question):
    return [{"role": "system", "content": SOLVER_SYSTEM},
            {"role": "user", "content": question}]


def reviser_messages(question, solution):
    user = f"Problem:\n{question}\n\nProposed solution:\n{solution}"
    return [{"role": "system", "content": REVISER_SYSTEM},
            {"role": "user", "content": user}]


# Self-revision is the same prompt, run with the solver's own adapter.
self_revision_messages = reviser_messages
