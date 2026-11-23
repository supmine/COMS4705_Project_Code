"""Answer extraction and grading. Used as the training reward and the eval metric.

grade(..., dataset="gsm8k" | "math") picks the checker.

GSM8K: last \\boxed{...} (nested braces handled); otherwise the last number.
Normalization: drop commas, $, %, \\text{...}; answers are compared as numbers.

MATH: see the MATH section below (normalized string match, then symbolic).
"""
import logging
import math
import re
import signal
import threading
from contextlib import contextmanager
from fractions import Fraction

NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    j = k = i + len("\\boxed{")
    depth = 1
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j:k - 1] if depth == 0 else None


def extract_answer(text):
    """Return (answer_string, method); method is boxed | last_number | none."""
    if text is None:
        return None, "none"
    s = _last_boxed(text)
    if s is not None:
        return s, "boxed"
    nums = NUM.findall(text)
    return (nums[-1], "last_number") if nums else (None, "none")


def to_number(s):
    """Normalized numeric value of an answer string, or None."""
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("{,}", ",").replace("\\,", "").replace("\\!", "")  # LaTeX 1{,}000, 1\,000
    s = s.replace(",", "").replace("\\$", "").replace("$", "").replace("\\%", "").replace("%", "")
    m = NUM.search(s)
    return float(m.group(0)) if m else None


def normalize(s):
    """Canonical string for logs: '18', '0.5', or None if no number."""
    v = to_number(s)
    if v is None:
        return None
    return str(int(v)) if v.is_integer() else str(round(v, 6))


def grade(pred_text, gold, allowed_methods=None, dataset="gsm8k"):
    """Return dict(pred, method, correct) (+ pred_raw, decided_by on MATH).
    allowed_methods restricts which extraction methods can count as correct
    (e.g. {"boxed"} for a strict reward)."""
    if dataset == "math":
        return grade_math(pred_text, gold, allowed_methods)
    pred, method = extract_answer(pred_text)
    p, g = to_number(pred), to_number(gold)
    correct = p is not None and g is not None and math.isclose(p, g, rel_tol=1e-9, abs_tol=1e-6)
    if allowed_methods is not None and method not in allowed_methods:
        correct = False
    return {"pred": normalize(pred), "method": method, "correct": correct}


def is_correct(pred_text, gold, allowed_methods=None, dataset="gsm8k"):
    return grade(pred_text, gold, allowed_methods, dataset)["correct"]


# ---------------------------------------------------------------- MATH
# Answers can be fractions, roots, expressions, intervals or text. Check:
# 1) exact match after LaTeX normalization ("string"), else
# 2) symbolic equivalence via math-verify/SymPy under a timeout ("symbolic").
# Each grade records which path decided it in "decided_by":
# string | symbolic | unparsed (math-verify could not parse a side) | timeout | none (no answer).

MATH_TIMEOUT_SECONDS = 5.0


def _last_box_command(text):
    """Content of the last \\boxed{...} or \\fbox{...} (nested braces handled),
    or of the brace-less form `\\boxed 5`."""
    i = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if i < 0:
        return None
    j = i + (len("\\boxed") if text.startswith("\\boxed", i) else len("\\fbox"))
    if text.startswith(" ", j):
        m = re.match(r"\s*([^\s$]+)", text[j:])
        return m.group(1) if m else None
    if not text.startswith("{", j):
        return None
    k = j + 1
    depth = 1
    while k < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[k], 0)
        k += 1
    return text[j + 1:k - 1] if depth == 0 else None


def math_gold(solution):
    """Gold answer of a MATH problem: the last \\boxed{} / \\fbox{} in its solution."""
    return _last_box_command(solution)


def extract_math_answer(text):
    """Return (answer_string, method); method is boxed | last_number | none.
    On MATH, last_number is a weak fallback."""
    if text is None:
        return None, "none"
    s = _last_box_command(text)
    if s is not None:
        return s, "boxed"
    nums = NUM.findall(text)
    return (nums[-1], "last_number") if nums else (None, "none")


def _fix_fracs(s):
    """\\frac34 -> \\frac{3}{4}, \\frac3{4} -> \\frac{3}{4}, \\frac{3}4 -> \\frac{3}{4}."""
    s = re.sub(r"\\frac(\d)(\d)", r"\\frac{\1}{\2}", s)
    s = re.sub(r"\\frac(\d)\{", r"\\frac{\1}{", s)
    return re.sub(r"\\frac\{([^{}]*)\}(\d)", r"\\frac{\1}{\2}", s)


def _numeric_canonical(s):
    """'0.5', '\\frac{1}{2}', '1/2' -> '1/2'; '1,000' -> '1000'; non-numbers -> None."""
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+", s):
        s = s.replace(",", "")
    try:
        if re.fullmatch(r"-?\d+(\.\d+)?", s):
            return str(Fraction(s))
        m = re.fullmatch(r"(-?)\\frac\{(-?\d+)\}\{(-?\d+)\}", s) or re.fullmatch(r"(-?)(\d+)/(\d+)", s)
        if m:
            v = Fraction(int(m.group(2)), int(m.group(3)))
            return str(-v if m.group(1) else v)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def normalize_math(s):
    """Canonical LaTeX string used for the exact-match path."""
    if s is None:
        return None
    s = s.strip().replace("\n", "")
    m = re.fullmatch(r"\\(?:text|textbf|mbox|mathrm)\{(.*)\}", s)
    if m:  # the whole answer is text, e.g. \text{(C)}
        s = m.group(1)
    else:  # units after a value, e.g. 5\text{ cm}
        stripped = re.sub(r"\\(?:text|mbox|mathrm)\{[^{}]*\}", "", s)
        s = stripped if stripped.strip() else s
    for tok in ("\\!", "\\,", "\;", "\\:", "\\displaystyle", "\\$", "$", "\\%", "%"):
        s = s.replace(tok, "")
    s = re.sub(r"\\(?:left|right)(?![a-zA-Z])", "", s)
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace("^{\\circ}", "").replace("^\\circ", "")
    s = re.sub(r"\s+", "", s)
    s = s.rstrip(".")
    s = re.sub(r"\\sqrt(\w)", r"\\sqrt{\1}", s)
    s = _fix_fracs(s)
    m = re.fullmatch(r"[a-zA-Z]{1,2}=([^=]+)", s)  # x=5 -> 5
    if m:
        s = m.group(1)
    if s.startswith("."):
        s = "0" + s
    s = s.replace("{.", "{0.")
    return _numeric_canonical(s) or s


class SymbolicTimeout(BaseException):
    """BaseException so math-verify/SymPy `except Exception` blocks don't swallow it."""


@contextmanager
def _time_limit(seconds):
    # SIGALRM works only in the main thread on Unix; elsewhere run without a limit.
    if (not seconds or not hasattr(signal, "setitimer")
            or threading.current_thread() is not threading.main_thread()):
        yield
        return

    def handler(signum, frame):
        raise SymbolicTimeout()

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def math_equal(pred, gold, timeout=MATH_TIMEOUT_SECONDS):
    """Return (equal, decided_by) for two extracted MATH answer strings."""
    if pred is None or gold is None:
        return False, "none"
    if normalize_math(pred) == normalize_math(gold):
        return True, "string"
    from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify

    # Our own _time_limit replaces math-verify's timeouts; silence its warning about that.
    logging.getLogger("math_verify").setLevel(logging.ERROR)
    cfg = [LatexExtractionConfig(boxed_match_priority=0), ExprExtractionConfig()]
    try:
        with _time_limit(timeout):
            g = parse(f"${gold}$", extraction_config=cfg, parsing_timeout=None)
            p = parse(f"${pred}$", extraction_config=cfg, parsing_timeout=None)
            if not g or not p:
                return False, "unparsed"
            return bool(verify(g, p, timeout_seconds=None)), "symbolic"
    except SymbolicTimeout:
        return False, "timeout"


def grade_math(pred_text, gold, allowed_methods=None, timeout=MATH_TIMEOUT_SECONDS):
    """Return dict(pred, pred_raw, method, correct, decided_by) for a MATH response."""
    pred, method = extract_math_answer(pred_text)
    correct, decided_by = math_equal(pred, gold, timeout)
    if allowed_methods is not None and method not in allowed_methods:
        correct = False
    return {"pred": normalize_math(pred), "pred_raw": pred, "method": method,
            "correct": correct, "decided_by": decided_by}
