import pytest

from common.answer import extract_answer, grade, is_correct, normalize


def test_boxed_last_wins():
    assert extract_answer(r"first \boxed{3} then \boxed{18}") == ("18", "boxed")


def test_boxed_nested_braces():
    assert extract_answer(r"\boxed{\text{\$}1{,}000}") == (r"\text{\$}1{,}000", "boxed")
    assert is_correct(r"\boxed{\text{\$}1{,}000}", "1000")
    assert is_correct(r"\boxed{1\,000}", "1000")


def test_unclosed_boxed_falls_back():
    assert extract_answer(r"answer \boxed{12") == ("12", "last_number")


def test_last_number_fallback():
    assert extract_answer("work...\n#### 72") == ("72", "last_number")
    assert extract_answer("so she has 5 apples, then 7") == ("7", "last_number")
    assert extract_answer("no digits here") == (None, "none")


@pytest.mark.parametrize("raw,norm", [
    ("1,000", "1000"), ("$18", "18"), ("18.00", "18"), ("50\\%", "50"),
    ("-3", "-3"), ("0.5", "0.5"), (r"18 \text{ dollars}", "18"), (None, None),
])
def test_normalize(raw, norm):
    assert normalize(raw) == norm


def test_numeric_comparison():
    assert is_correct(r"\boxed{18.0}", "18")
    assert is_correct(r"\boxed{\$1,000.00}", "1000")
    assert is_correct(r"\boxed{50\%}", "50")


def test_is_correct():
    assert is_correct(r"... \boxed{1,000}", "1000")
    assert not is_correct(r"... \boxed{999}", "1000")
    assert not is_correct("nothing", "5")


def test_strict_reward_rejects_last_number():
    g = grade("the answer is 5", "5", allowed_methods={"boxed"})
    assert g["method"] == "last_number" and not g["correct"]


def test_gold_solutions_score_100():
    """Every GSM8K reference solution must grade as correct against its own gold."""
    datasets = pytest.importorskip("datasets")
    from common.data import gold_from_gsm8k
    try:
        ds = datasets.load_dataset("openai/gsm8k", "main")
    except Exception as e:
        pytest.skip(f"GSM8K unavailable: {e}")
    for split in ("train", "test"):
        bad = [ex for ex in ds[split]
               if not is_correct(ex["answer"], gold_from_gsm8k(ex["answer"]))]
        assert not bad, f"{len(bad)} {split} solutions fail, e.g. {bad[0]['answer'][-80:]!r}"
