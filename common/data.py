"""GSM8K loading. Use the same seed across the team so the dev split matches."""

DEV_SIZE = 500
SEED = 0


def gold_from_gsm8k(answer_field):
    return answer_field.split("####")[-1].strip()


def load_gsm8k(dev_size=DEV_SIZE, seed=SEED):
    """Return (train, dev, test) HF datasets. dev is carved out of train;
    test (1,319 problems) is for final numbers only."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    train = ds["train"].shuffle(seed=seed)
    dev = train.select(range(dev_size))
    train = train.select(range(dev_size, len(train)))
    return train, dev, ds["test"]


def records(split, dev_size=DEV_SIZE, seed=SEED):
    """Return a list of {"qid", "question", "gold", "solution"} for train | dev | test.
    qid is the row index within that split."""
    train, dev, test = load_gsm8k(dev_size, seed)
    ds = {"train": train, "dev": dev, "test": test}[split]
    return [{"qid": i, "question": ex["question"],
             "gold": gold_from_gsm8k(ex["answer"]), "solution": ex["answer"]}
            for i, ex in enumerate(ds)]
