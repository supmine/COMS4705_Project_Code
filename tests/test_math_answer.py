import pytest

from common.answer import (extract_math_answer, grade, is_correct, math_equal, math_gold,
                           normalize_math)

pytest.importorskip("math_verify")


@pytest.mark.parametrize("pred,gold", [
    (r"\frac{1}{2}", "0.5"),
    (r"\dfrac{3}{4}", r"\frac34"),
    (r"2\sqrt{5}", r"2 \sqrt 5"),
    (r"10^\circ", "10"),
    (r"\left( 1, 2 \right)", "(1,2)"),
    (r"-\frac{33}{2}", r"k = \frac{-33}{2}"),
    (r"\frac{\sqrt{3}}{2}", r"\dfrac{\sqrt3}{2}"),
    (r"\text{(B)}", "(B)"),
])
def test_equal(pred, gold):
    assert math_equal(pred, gold)[0]
    assert math_equal(gold, pred)[0]


@pytest.mark.parametrize("pred,gold", [
    (r"\frac{1}{2}", r"\frac{1}{3}"),
    ("(1,2)", "[1,2]"),
    ("x+1", "x-1"),
])
def test_different(pred, gold):
    assert not math_equal(pred, gold)[0]
    assert not math_equal(gold, pred)[0]


def test_normalization_alone_handles_simple_forms():
    assert normalize_math("0.5") == normalize_math(r"\frac{1}{2}") == normalize_math(r"\tfrac12")
    assert normalize_math(r"\left( 1, 2 \right)") == normalize_math("(1,2)")
    assert normalize_math(r"5\text{ cm}.") == "5"
    assert normalize_math(r"\text{(C)}") == "(C)"
    assert normalize_math("x = 3") == "3"


def test_symbolic_path_is_logged():
    assert math_equal(r"x^2+2x+1", r"(x+1)^2") == (True, "symbolic")
    assert math_equal(r"\frac{1}{2}", r"\frac{1}{2}") == (True, "string")
    assert math_equal(None, "1") == (False, "none")


def test_timeout_is_reported():
    # A tiny budget forces the symbolic path to be cut off.
    assert math_equal(r"x^2+2x+1", r"(x+1)^2", timeout=1e-6) == (False, "timeout")


def test_extraction():
    assert math_gold(r"so \boxed{\frac{1}{2}} and \fbox{3}") == "3"
    assert math_gold(r"\boxed{\{1, 2\}}") == r"\{1, 2\}"
    assert math_gold(r"answer is \boxed 7$.") == "7"
    assert extract_math_answer("roughly 4 or 5") == ("5", "last_number")
    assert extract_math_answer("no answer") == (None, "none")


def test_grade_dispatch():
    g = grade(r"... \boxed{\dfrac{3}{4}}", r"\frac34", dataset="math")
    assert g["correct"] and g["method"] == "boxed" and g["decided_by"] == "string"
    assert not is_correct("the answer is 0.75", r"\frac34", allowed_methods={"boxed"}, dataset="math")
    assert is_correct(r"\boxed{1,000}", "1000")  # GSM8K default unchanged


def test_gold_solutions_score_100():
    """Every level 3-5 MATH solution grades correct against its own extracted gold."""
    pytest.importorskip("datasets")
    from common.data import math_records
    try:
        rows = math_records("test") + math_records("dev") + math_records("train")
    except Exception as e:
        pytest.skip(f"MATH unavailable: {e}")
    missing = [r["qid"] for r in rows if r["gold"] is None]
    assert not missing, f"{len(missing)} solutions without \\boxed/\\fbox, e.g. {missing[:5]}"
    bad = [r["qid"] for r in rows if not is_correct(r["solution"], r["gold"], dataset="math")]
    assert not bad, f"{len(bad)} gold solutions fail, e.g. {bad[:5]}"
