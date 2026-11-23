import pytest

from common.prompts import render, reviser_messages, solver_messages

EXPECTED_SOLVER = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "What is 2+2?\n"
    "Please reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def test_no_custom_system_message():
    assert [m["role"] for m in solver_messages("q")] == ["user"]
    assert [m["role"] for m in reviser_messages("q", "s")] == ["user"]


def test_reviser_content():
    c = reviser_messages("Q?", "SOL")[0]["content"]
    assert c.startswith("Q?\n\nHere is a proposed solution, which may or may not be correct:\nSOL\n\n")
    assert c.endswith("put your final answer within \\boxed{}.")


def test_rendered_solver_prompt():
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")
    assert render(tok, solver_messages("What is 2+2?")) == EXPECTED_SOLVER
