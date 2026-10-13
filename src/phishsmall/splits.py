"""Every evaluation set in this study, built with the original repository's own code.

Nothing about *which* hosts land where is re-implemented. The training frame and
the out-of-distribution set come from the vendored :mod:`phishurl.datasets`, and
the hold-out is drawn with the same ``GroupShuffleSplit`` and the same seed, so
the test sets are the published ones host for host.

The only addition is a validation split for early stopping, carved out of the
original *training* portion and grouped by registrable domain in the same way.
Neural models need it; the published LightGBM never had one.

Built once and cached under ``data/processed/`` (never committed: it is derived
from third-party data).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from phishsmall.config import PROCESSED, SPLIT_SEED, TEST_SIZE, VAL_SIZE

SPLIT_FILES = ("hosts", "ood", "ood_unfiltered", "openphish")

# Host names are free text from phishing feeds. Without these options pandas reads
# a host literally named "nan" or "null" back as a missing value.
_CSV_READ = {"dtype": str, "keep_default_na": False, "na_filter": False}


@dataclass(frozen=True)
class Splits:
    """The study's data: one frame per evaluation regime, all keyed by canonical host.

    ``hosts`` is PhiUSIIL reduced to unique hosts with a ``split`` column taking
    ``train`` / ``val`` / ``test``; the original model trained on ``train`` + ``val``.
    """

    hosts: pd.DataFrame
    ood: pd.DataFrame
    ood_unfiltered: pd.DataFrame
    openphish: pd.DataFrame
    stats: dict

    def rows(self, *names: str) -> pd.DataFrame:
        return self.hosts[self.hosts["split"].isin(names)]

    def evaluation_universe(self) -> list[str]:
        """Every host any model must score, deduplicated and sorted.

        Scoring this union once per model, rather than each set separately, means
        a host shared between two sets cannot receive two different scores.
        """
        hosts = set(self.rows("val", "test")["host"])
        for frame in (self.ood, self.ood_unfiltered, self.openphish):
            hosts.update(frame["host"])
        return sorted(hosts)


def assign_splits(frame: pd.DataFrame, seed: int = SPLIT_SEED) -> pd.Series:
    """Label each row ``train`` / ``val`` / ``test``, grouped by ``registrable_domain``.

    The test draw is exactly the original's: ``GroupShuffleSplit`` with the same
    ``test_size`` and ``random_state`` over the same groups in the same order. The
    validation draw then splits the remaining groups again.
    """
    groups = frame["registrable_domain"].to_numpy()
    positions = np.arange(len(frame))

    outer = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
    train_index, test_index = next(outer.split(positions, groups=groups))

    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=seed)
    fit_rel, val_rel = next(inner.split(train_index, groups=groups[train_index]))

    labels = np.empty(len(frame), dtype=object)
    labels[test_index] = "test"
    labels[train_index[fit_rel]] = "train"
    labels[train_index[val_rel]] = "val"
    return pd.Series(labels, index=frame.index, name="split")


def leaked_groups(frame: pd.DataFrame) -> dict[tuple[str, str], int]:
    """Registrable domains appearing in more than one split, counted per pair of splits."""
    by_split = {name: set(part["registrable_domain"]) for name, part in frame.groupby("split")}
    names = sorted(by_split)
    leaks: dict[tuple[str, str], int] = {}
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            leaks[(left, right)] = len(by_split[left] & by_split[right])
    return leaks


def fingerprint(hosts: pd.Series, labels: pd.Series) -> str:
    """Order-independent SHA-256 of a set of labelled hosts.

    Recorded in ``reports/results.json`` so anyone re-running the pipeline can tell
    whether they evaluated on the same hosts, not merely the same number of them.
    """
    lines = sorted(f"{host}\t{label}" for host, label in zip(hosts, labels, strict=True))
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def build() -> Splits:
    """Assemble every set from the raw data. Takes about a minute."""
    from phishurl.datasets import build_ood_frame, build_training_frame

    frame = build_training_frame()
    conflicts = int(frame.attrs.get("n_conflicting_hosts", 0))
    frame = frame[["host", "y", "registrable_domain"]].copy()
    frame["split"] = assign_splits(frame)

    leaks = {pair: count for pair, count in leaked_groups(frame).items() if count}
    if leaks:
        raise AssertionError(f"registrable domains leaked across splits: {leaks}")

    # The original passes every PhiUSIIL host, train and test alike, as "seen".
    seen = set(frame["host"])
    filtered = build_ood_frame(seen)
    # An empty "seen" set reproduces the naive evaluation that skips the filter;
    # the ablation measures how much each model would gain from it.
    unfiltered = build_ood_frame(set())

    # Kept one row per URL, duplicates included, because that is how the original
    # computed its published OpenPhish rate; evaluation reports per-host as well.
    openphish = filtered.openphish[["host"]].assign(y=1)
    stats = {
        "dropped_conflicting_hosts": conflicts,
        "ood_composition": filtered.stats,
        "ood_unfiltered_composition": unfiltered.stats,
    }
    return Splits(
        hosts=frame.reset_index(drop=True),
        ood=filtered.frame[["host", "y", "registrable_domain"]].reset_index(drop=True),
        ood_unfiltered=unfiltered.frame[["host", "y", "registrable_domain"]].reset_index(drop=True),
        openphish=openphish.reset_index(drop=True),
        stats=stats,
    )


def save(splits: Splits) -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_FILES:
        getattr(splits, name).to_csv(PROCESSED / f"{name}.csv.gz", index=False)
    (PROCESSED / "split_stats.json").write_text(json.dumps(splits.stats, indent=2))


def _read(name: str) -> pd.DataFrame:
    frame = pd.read_csv(PROCESSED / f"{name}.csv.gz", **_CSV_READ)
    frame["y"] = frame["y"].astype(int)
    return frame


def load(rebuild: bool = False) -> Splits:
    """Load the cached splits, building them first if they are missing."""
    cached = all((PROCESSED / f"{name}.csv.gz").exists() for name in SPLIT_FILES)
    if rebuild or not cached:
        splits = build()
        save(splits)
        return splits
    return Splits(
        **{name: _read(name) for name in SPLIT_FILES},
        stats=json.loads((PROCESSED / "split_stats.json").read_text()),
    )


def main() -> None:
    splits = load(rebuild=True)
    counts = splits.hosts.groupby("split")["y"].agg(["size", "sum"])
    print("PhiUSIIL hosts by split (size, phishing):")
    print(counts.to_string())
    print(f"OOD + temporal:   {len(splits.ood):,} hosts ({int(splits.ood['y'].sum()):,} phishing)")
    print(f"OOD unfiltered:   {len(splits.ood_unfiltered):,} hosts")
    print(f"OpenPhish probe:  {len(splits.openphish):,} hosts")
    print(f"Evaluation universe: {len(splits.evaluation_universe()):,} distinct hosts")


if __name__ == "__main__":
    main()
