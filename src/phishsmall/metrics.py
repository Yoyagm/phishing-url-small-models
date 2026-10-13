"""Ranking metrics that can be bootstrapped a thousand times without re-sorting.

The point estimates in this study come from the vendored
:func:`phishurl.evaluate.compute`, unchanged, so they are the original's numbers
by construction. Confidence intervals need the same two metrics — ROC-AUC and
recall at a fixed false-positive rate — recomputed on ~1,000 resamples of up to
131k hosts for every model, and calling scikit-learn for each would re-sort the
scores every time.

:class:`RankedScores` sorts once. A bootstrap replicate is then a vector of
non-negative integer weights, and the ROC curve under those weights is a pair of
cumulative sums — O(n), no sort. The curve is built to be *identical* to
``sklearn.metrics.roc_curve`` with ``sample_weight`` (zero-weight samples dropped,
collinear points dropped), which ``tests/test_metrics.py`` checks point for point.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Curve:
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


class RankedScores:
    """Scores sorted once, with the boundaries of every run of tied scores."""

    def __init__(self, y_true: np.ndarray, scores: np.ndarray) -> None:
        y_true = np.asarray(y_true)
        scores = np.asarray(scores, dtype=np.float64)
        if y_true.shape != scores.shape:
            raise ValueError(f"y_true {y_true.shape} and scores {scores.shape} differ in shape")
        if not np.isfinite(scores).all():
            raise ValueError("scores contain NaN or infinity; the model output is broken")

        self.order = np.argsort(-scores, kind="stable")
        sorted_scores = scores[self.order]
        self.y = y_true[self.order].astype(np.float64)
        self.group_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
        self.group_scores = sorted_scores[self.group_ends]
        self.size = len(scores)

    def curve(self, weights: np.ndarray | None = None) -> Curve:
        """ROC curve, optionally under integer sample weights (a bootstrap replicate)."""
        w = np.ones(self.size) if weights is None else np.asarray(weights, dtype=np.float64)[self.order]
        tps = np.cumsum(self.y * w)[self.group_ends]
        fps = np.cumsum((1.0 - self.y) * w)[self.group_ends]
        thresholds = self.group_scores

        # scikit-learn drops zero-weight samples before building the curve; a run of
        # tied scores that received no weight must therefore vanish, not repeat a point.
        carried = np.diff(np.r_[0.0, tps + fps]) > 0
        tps, fps, thresholds = tps[carried], fps[carried], thresholds[carried]

        if len(fps) > 2:
            corner = np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True]
            tps, fps, thresholds = tps[corner], fps[corner], thresholds[corner]

        tps = np.r_[0.0, tps]
        fps = np.r_[0.0, fps]
        thresholds = np.r_[np.inf, thresholds]
        return Curve(fpr=fps / fps[-1], tpr=tps / tps[-1], thresholds=thresholds)


def recall_at_fpr(curve: Curve, target_fpr: float) -> tuple[float, float]:
    """Highest recall with FPR not above ``target_fpr``, and the threshold reaching it.

    Same definition as :func:`phishurl.evaluate.recall_at_fpr`: the last ROC point
    whose false-positive rate is within the target.
    """
    index = int(np.searchsorted(curve.fpr, target_fpr, side="right")) - 1
    return float(curve.tpr[index]), float(curve.thresholds[index])


def roc_auc(curve: Curve) -> float:
    return float(np.trapezoid(curve.tpr, curve.fpr))


def cluster_bootstrap_weights(groups: np.ndarray, reps: int, seed: int) -> Iterator[np.ndarray]:
    """Yield ``reps`` integer weight vectors from resampling *groups* with replacement.

    Hosts sharing a registrable domain are not independent observations — the
    split itself is grouped for that reason — so the bootstrap resamples whole
    domains, and every host of a drawn domain enters with that domain's count.
    Yielded one at a time: 1,000 replicates of 131k hosts would be half a gigabyte.
    """
    codes, n_groups = _group_codes(groups)
    rng = np.random.default_rng(seed)
    for _ in range(reps):
        drawn = np.bincount(rng.integers(0, n_groups, size=n_groups), minlength=n_groups)
        yield drawn[codes]


def _group_codes(groups: np.ndarray) -> tuple[np.ndarray, int]:
    uniques, codes = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    return codes, len(uniques)


def interval(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of a bootstrap distribution."""
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(samples, [tail, 1.0 - tail])
    return float(low), float(high)
