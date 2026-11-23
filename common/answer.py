"""Answer extraction and grading. Used as the training reward and the eval metric."""
import re

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
    """Return (answer_string, method); method is boxed | hash | last_number | none."""
    if text is None:
        return None, "none"
    s = _last_boxed(text)
    if s is not None:
        return s, "boxed"
    m = re.search(r"####\s*(.+)", text)
    if m:
        return m.group(1), "hash"
    nums = NUM.findall(text)
    return (nums[-1], "last_number") if nums else (None, "none")


def normalize(s):
    if s is None:
        return None
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = s.replace("{,}", ",").replace("\\,", "").replace("\\!", "")  # LaTeX 1{,}000, 1\,000
    s = s.replace(",", "").replace("$", "").replace("\\%", "").replace("%", "")
    m = NUM.search(s)
    if not m:
        return s.strip()
    v = float(m.group(0))
    return str(int(v)) if v.is_integer() else str(round(v, 6))


def grade(pred_text, gold, allowed_methods=None):
    """Return dict(pred, method, correct). allowed_methods restricts which
    extraction methods can count as correct (e.g. {"boxed"} for a strict reward)."""
    pred, method = extract_answer(pred_text)
    p = normalize(pred)
    correct = p is not None and p == normalize(gold)
    if allowed_methods is not None and method not in allowed_methods:
        correct = False
    return {"pred": p, "method": method, "correct": correct}


def is_correct(pred_text, gold, allowed_methods=None):
    return grade(pred_text, gold, allowed_methods)["correct"]
