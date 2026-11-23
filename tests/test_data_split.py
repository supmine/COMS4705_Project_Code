import pytest

from common.data import split_spec


def test_split_is_a_partition():
    s = split_spec()
    dev, train = s["dev_ids"], s["train_order"]
    assert len(dev) == s["sizes"]["dev"] == 500
    assert len(train) == s["sizes"]["train"] == 6973
    assert s["sizes"]["test"] == 1319
    assert len(set(dev)) == len(dev) and len(set(train)) == len(train)
    assert not set(dev) & set(train)
    assert set(dev) | set(train) == set(range(7473))


def test_split_matches_documented_rule():
    """Same membership and order as the original HF `train.shuffle(seed=0)` split."""
    np = pytest.importorskip("numpy")
    perm = np.random.default_rng(0).permutation(7473).tolist()
    s = split_spec()
    assert s["train_order"] == perm[500:]
    assert s["dev_ids"] == sorted(perm[:500])


def test_train_stream_prefix_property():
    """A longer run sees a strict superset (prefix) of a shorter run's problems, in the same order."""
    pytest.importorskip("datasets")
    from common.data import train_stream
    try:
        short, long_ = train_stream(64), train_stream(128)
    except Exception as e:
        pytest.skip(f"GSM8K unavailable: {e}")
    assert [r["qid"] for r in long_[:64]] == [r["qid"] for r in short]
    assert [r["pos"] for r in long_] == list(range(128))


def test_records_follow_split():
    pytest.importorskip("datasets")
    from common.data import records
    try:
        tr, dev, te = records("train"), records("dev"), records("test")
    except Exception as e:
        pytest.skip(f"GSM8K unavailable: {e}")
    s = split_spec()
    assert [r["qid"] for r in tr] == s["train_order"]
    assert [r["qid"] for r in dev] == s["dev_ids"]
    assert [r["qid"] for r in te] == list(range(1319))
    assert not {r["question"] for r in dev} & {r["question"] for r in tr}
