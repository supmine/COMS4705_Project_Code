"""GSM8K loading with ONE frozen split, shared by eval and every training run.

The split lives in common/gsm8k_split_v1.json (committed; regenerate only with common/make_gsm8k_split.py):

    dev   = 500 problems carved out of the official train split   -> checkpoint selection / tuning only
    train = the other 6,973 train problems, in a FIXED order       -> the only data any run trains on
    test  = the official 1,319-problem test split                  -> final numbers only, never for selection

It is the same membership and order as the original `train.shuffle(seed=0)` split, but frozen in a file so it
cannot drift with library versions, and qids are now the ORIGINAL GSM8K row indices (stable across runs):
    train/dev qid = row index in HF `openai/gsm8k` train split;  test qid = row index in the test split.

Training-data rule (see train/README.md): trainers consume `train_stream(n)` in order and never reshuffle.
A run that uses n prompts sees exactly the first n of the fixed order, so runs with matched optimizer steps see
the same problems, and a longer run (e.g. the token-matched single-agent run) sees a strict superset.
"""
import hashlib
import json
import os
import warnings

SPLIT_NAME = "gsm8k_v1"
SPLIT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_split_v1.json")
DEV_SIZE = 500   # kept for reference; the split file is the source of truth
SEED = 0

_SPLIT = None
_RAW = {}


def gold_from_gsm8k(answer_field):
    return answer_field.split("####")[-1].strip()


def split_spec():
    """The frozen split: dict with dev_ids, train_order, sizes, questions_sha256."""
    global _SPLIT
    if _SPLIT is None:
        with open(SPLIT_FILE) as f:
            _SPLIT = json.load(f)
    return _SPLIT


def _raw(hf_split):
    """Original HF rows (list of dicts) for 'train' or 'test', with a fingerprint check."""
    if hf_split not in _RAW:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=hf_split)
        rows = [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
        spec = split_spec()
        expected_n = spec["sizes"]["train"] + spec["sizes"]["dev"] if hf_split == "train" else spec["sizes"]["test"]
        if len(rows) != expected_n:
            raise RuntimeError(f"GSM8K {hf_split} has {len(rows)} rows, split file expects {expected_n}")
        want = spec.get("questions_sha256", {}).get(hf_split)
        got = hashlib.sha256("\n".join(r["question"] for r in rows).encode("utf-8")).hexdigest()
        if want and got != want:
            warnings.warn(f"GSM8K {hf_split} question fingerprint differs from gsm8k_split_v1.json; "
                          "the dataset revision may have changed. Tell the team before trusting new numbers.")
        _RAW[hf_split] = rows
    return _RAW[hf_split]


def _record(rows, i, split):
    ex = rows[i]
    return {"qid": i, "split": split, "question": ex["question"],
            "gold": gold_from_gsm8k(ex["answer"]), "solution": ex["answer"]}


def records(split):
    """List of {"qid", "split", "question", "gold", "solution"} for 'train' | 'dev' | 'test'.
    train is returned in the fixed training order; dev in ascending qid; test in original order."""
    spec = split_spec()
    if split == "test":
        rows = _raw("test")
        return [_record(rows, i, "test") for i in range(len(rows))]
    rows = _raw("train")
    if split == "dev":
        return [_record(rows, i, "dev") for i in spec["dev_ids"]]
    if split == "train":
        return [_record(rows, i, "train") for i in spec["train_order"]]
    raise ValueError(f"unknown split {split!r}")


def train_stream(n, offset=0, keep_qids=None):
    """The first n training records (starting at position `offset`) in the fixed order, cycling into a new
    epoch (same order) if n exceeds the training set. `keep_qids` optionally restricts to a subset (e.g. a
    difficulty filter computed on TRAIN ONLY); the order of the remaining problems is unchanged.
    Each record gets "pos": its position in the stream (0-based), so runs can log exactly what they saw."""
    base = records("train")
    if keep_qids is not None:
        keep = set(keep_qids)
        base = [r for r in base if r["qid"] in keep]
    if not base:
        raise ValueError("empty training set")
    out = []
    for pos in range(offset, offset + n):
        r = dict(base[pos % len(base)])
        r["pos"], r["epoch"] = pos, pos // len(base)
        out.append(r)
    return out


def load_gsm8k(dev_size=DEV_SIZE, seed=SEED):
    """Backward-compatible: (train, dev, test) as HF datasets following the frozen split."""
    if dev_size != DEV_SIZE or seed != SEED:
        raise ValueError("the split is frozen in common/gsm8k_split_v1.json; dev_size/seed cannot be changed")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    spec = split_spec()
    return ds["train"].select(spec["train_order"]), ds["train"].select(spec["dev_ids"]), ds["test"]



# ---------------------------------------------------------------- MATH (levels 3-5)
# The original hendrycks/competition_math copy is script-based and no longer loads with current
# `datasets`; this mirror has the full 7,500 train / 5,000 test problems, split by subject, with
# `level` and `type` fields. The revision is pinned so membership cannot drift.
#   qid  = "<subject>/<row index in that subject's HF split>"  (stable)
#   dev  = 500 problems sampled (random.Random(0), independent of `datasets` versions) from the
#          level 3-5 train problems; train = the remaining level 3-5 train problems
#   test = every level 3-5 problem of the test split
MATH_NAME = "EleutherAI/hendrycks_math"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]
MATH_LEVELS = (3, 4, 5)

DATASET_INFO = {
    "gsm8k": {"dataset_name": "openai/gsm8k", "dataset_config": "main",
              "dataset_revision": None, "split_name": SPLIT_NAME},
    "math": {"dataset_name": MATH_NAME, "dataset_config": ",".join(MATH_SUBJECTS),
             "dataset_revision": MATH_REVISION, "split_name": "math_l345_seed0"},
}

_MATH_RAW = {}


def math_level(level_field):
    """'Level 4' -> 4; 'Level ?' -> None."""
    tail = level_field.split()[-1]
    return int(tail) if tail.isdigit() else None


def _math_raw(hf_split):
    """All MATH rows of 'train' or 'test' as records (every level), in subject then row order."""
    if hf_split not in _MATH_RAW:
        from datasets import load_dataset
        from common.answer import math_gold

        rows = []
        for subject in MATH_SUBJECTS:
            ds = load_dataset(MATH_NAME, subject, split=hf_split, revision=MATH_REVISION)
            for i, ex in enumerate(ds):
                rows.append({"qid": f"{subject}/{i}", "question": ex["problem"],
                             "gold": math_gold(ex["solution"]), "solution": ex["solution"],
                             "level": math_level(ex["level"]), "subject": ex["type"]})
        _MATH_RAW[hf_split] = rows
    return _MATH_RAW[hf_split]


def math_records(split, levels=MATH_LEVELS, dev_size=DEV_SIZE, seed=SEED):
    """List of {"qid", "split", "question", "gold", "solution", "level", "subject"}
    for 'train' | 'dev' | 'test', restricted to `levels`."""
    import random

    keep = set(levels)
    if split == "test":
        rows = [r for r in _math_raw("test") if r["level"] in keep]
    elif split in ("dev", "train"):
        pool = [r for r in _math_raw("train") if r["level"] in keep]
        dev_ids = set(random.Random(seed).sample(range(len(pool)), dev_size))
        rows = [r for i, r in enumerate(pool) if (i in dev_ids) == (split == "dev")]
    else:
        raise ValueError(f"unknown split {split!r}")
    return [{**r, "split": split} for r in rows]


def level_counts(rows):
    counts = {}
    for r in rows:
        counts[r["level"]] = counts.get(r["level"], 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    # python -m common.data  -> MATH problems per level in dev and test
    for split in ("dev", "test"):
        rows = math_records(split)
        print(f"MATH {split}: {len(rows)} problems, per level {level_counts(rows)}")
