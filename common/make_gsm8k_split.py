"""Regenerate common/gsm8k_split_v1.json (the frozen GSM8K split). You should never need to run this:
the JSON is committed and is the source of truth. It exists to document exactly how the split was made.

    python -m common.make_gsm8k_split            # writes common/gsm8k_split_v1.json

Rule (identical to the original common/data.py, which used HF `train.shuffle(seed=0)`; HF implements that
as `np.random.default_rng(0).permutation(len(train))`):
    perm      = np.random.default_rng(0).permutation(7473)   # over original GSM8K train row indices
    dev_ids   = perm[:500]                                   # held out: checkpoint selection / tuning only
    train_ids = perm[500:]                                   # 6,973 problems, in this exact order
    test      = the official 1,319-problem test split, original order (final numbers only)
"""
import hashlib
import json
import os
import sys

import numpy as np

SEED, DEV_SIZE = 0, 500
N_TRAIN, N_TEST = 7473, 1319
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gsm8k_split_v1.json")


def questions_sha256(questions):
    return hashlib.sha256("\n".join(questions).encode("utf-8")).hexdigest()


def main():
    perm = np.random.default_rng(SEED).permutation(N_TRAIN).tolist()
    split = {
        "name": "gsm8k_v1",
        "source": "openai/gsm8k (config 'main'); qids are row indices of the original HF train/test splits",
        "rule": "perm = numpy.random.default_rng(0).permutation(7473); dev = perm[:500]; train = perm[500:]",
        "seed": SEED,
        "sizes": {"train": N_TRAIN - DEV_SIZE, "dev": DEV_SIZE, "test": N_TEST},
        "dev_ids": sorted(perm[:DEV_SIZE]),
        "train_order": perm[DEV_SIZE:],
    }
    # Optional fingerprint of the dataset contents, so a changed dataset revision is detected.
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main")
        assert len(ds["train"]) == N_TRAIN and len(ds["test"]) == N_TEST
        split["questions_sha256"] = {s: questions_sha256(ds[s]["question"]) for s in ("train", "test")}
    except Exception as e:  # offline: fingerprint can be added later
        print("warning: could not fingerprint the dataset:", e, file=sys.stderr)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(split, f)
    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
