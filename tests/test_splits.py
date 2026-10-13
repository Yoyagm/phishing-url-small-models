"""The split must keep every registrable domain on one side, and must be the original's.

Fast tests use a synthetic frame; the ``slow`` tests read the cached real splits
and are skipped when the data has not been fetched.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import GroupShuffleSplit

from phishsmall import splits
from phishsmall.config import PROCESSED, SPLIT_SEED, TEST_SIZE


@pytest.fixture
def frame() -> pd.DataFrame:
    """2,000 hosts under 300 domains of very uneven size, like real hosting platforms."""
    rng = np.random.default_rng(0)
    sizes = np.maximum(1, rng.zipf(1.6, 300) % 120)
    domains = np.repeat([f"site{i}.com" for i in range(300)], sizes)[:2000]
    hosts = [f"h{i}.{domain}" for i, domain in enumerate(domains)]
    return pd.DataFrame({"host": hosts, "y": rng.integers(0, 2, len(hosts)), "registrable_domain": domains})


def test_no_registrable_domain_crosses_train_val_or_test(frame) -> None:
    frame["split"] = splits.assign_splits(frame)
    assert set(frame["split"]) == {"train", "val", "test"}
    assert all(count == 0 for count in splits.leaked_groups(frame).values())


def test_test_split_is_exactly_the_original_group_shuffle_split(frame) -> None:
    """Same splitter, same seed, same groups: the hold-out must be the original's hold-out."""
    original = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SPLIT_SEED)
    _, test_index = next(original.split(frame, groups=frame["registrable_domain"]))

    labels = splits.assign_splits(frame)
    assert np.array_equal(np.flatnonzero(labels.to_numpy() == "test"), test_index)


def test_assignment_is_deterministic(frame) -> None:
    assert splits.assign_splits(frame).equals(splits.assign_splits(frame))


def test_leak_detector_catches_a_planted_leak(frame) -> None:
    """A guard that cannot fail is worthless: plant a shared domain and expect it counted."""
    frame["split"] = splits.assign_splits(frame)
    victim = frame.index[frame["split"] == "test"][0]
    train_domain = frame.loc[frame["split"] == "train", "registrable_domain"].iloc[0]
    frame.loc[victim, "registrable_domain"] = train_domain
    assert splits.leaked_groups(frame)[("test", "train")] == 1


def test_fingerprint_ignores_order_but_not_labels() -> None:
    hosts = pd.Series(["a.com", "b.com", "c.com"])
    labels = pd.Series([0, 1, 0])
    reordered = splits.fingerprint(hosts[::-1], labels[::-1])
    assert splits.fingerprint(hosts, labels) == reordered
    assert splits.fingerprint(hosts, pd.Series([1, 1, 0])) != reordered


real_data = pytest.mark.skipif(
    not (PROCESSED / "hosts.csv.gz").exists(), reason="real splits not built (make data splits)"
)


@pytest.fixture(scope="module")
def real() -> splits.Splits:
    return splits.load()


@pytest.mark.slow
@real_data
def test_real_splits_keep_domains_apart_and_match_the_published_sizes(real) -> None:
    assert all(count == 0 for count in splits.leaked_groups(real.hosts).values())
    test = real.rows("test")
    # reports/metrics.json of phishing-url-detector: n and n_phishing of both test sets.
    assert (len(test), int(test["y"].sum())) == (50_332, 23_424)
    assert (len(real.ood), int(real.ood["y"].sum())) == (131_396, 31_396)


@pytest.mark.slow
@real_data
def test_no_training_or_validation_host_reaches_a_test_set(real) -> None:
    seen = set(real.rows("train", "val")["host"])
    assert not seen & set(real.rows("test")["host"])
    assert not set(real.hosts["host"]) & set(real.ood["host"])
    assert not set(real.hosts["host"]) & set(real.openphish["host"])


@pytest.mark.slow
@real_data
def test_hosts_that_look_like_missing_values_survive_the_cache(real) -> None:
    """``nan``/``null`` are valid host strings; pandas would read them as missing by default."""
    for frame in (real.hosts, real.ood, real.ood_unfiltered, real.openphish):
        assert frame["host"].notna().all()
        assert (frame["host"].str.len() > 0).all()
