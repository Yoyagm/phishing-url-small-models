"""The two metrics every claim rests on: recall at a fixed FPR, and ROC-AUC.

Checked three ways: against a case small enough to count by hand, against
scikit-learn under bootstrap weights (ties and zero weights included), and
against the vendored function that produced the published numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score, roc_curve

from phishsmall.metrics import (
    RankedScores,
    cluster_bootstrap_weights,
    interval,
    recall_at_fpr,
    roc_auc,
)
from phishurl.evaluate import recall_at_fpr as published_recall_at_fpr


@pytest.fixture
def counted_case() -> tuple[np.ndarray, np.ndarray]:
    """100 benign hosts scored 0.00-0.99 and 10 phishing hosts placed around them.

    5 phishing score 0.995 (above every benign host), 3 score 0.985 (above all but
    the benign host at 0.99), 2 score 0.5. Allowing one false positive in 100 —
    FPR 1 % — the threshold drops to 0.985 and catches 8 of 10. Allowing none,
    it stops at 0.995 and catches 5.
    """
    benign = np.arange(100) / 100
    phishing = np.array([0.995] * 5 + [0.985] * 3 + [0.5] * 2)
    scores = np.concatenate([benign, phishing])
    labels = np.concatenate([np.zeros(100, dtype=int), np.ones(10, dtype=int)])
    return labels, scores


def test_recall_at_one_percent_fpr_on_a_hand_counted_case(counted_case) -> None:
    labels, scores = counted_case
    curve = RankedScores(labels, scores).curve()

    assert recall_at_fpr(curve, 0.01) == (0.8, 0.985)
    assert recall_at_fpr(curve, 0.0) == (0.5, 0.995)
    assert published_recall_at_fpr(labels, scores, 0.01) == (0.8, 0.985)


def test_auc_on_the_hand_counted_case(counted_case) -> None:
    labels, scores = counted_case
    # Each of the 5 top phishing beats all 100 benign; the 3 at 0.985 beat 99;
    # the 2 at 0.5 beat 50 and tie with one (half credit each).
    expected = (5 * 100 + 3 * 99 + 2 * 50.5) / (10 * 100)
    assert roc_auc(RankedScores(labels, scores).curve()) == pytest.approx(expected, abs=1e-12)


def _sklearn_reference(labels, scores, weights, target):
    fpr, tpr, thresholds = roc_curve(labels, scores, sample_weight=weights)
    index = int(np.max(np.where(fpr <= target)[0]))
    return float(tpr[index]), float(thresholds[index]), roc_auc_score(labels, scores, sample_weight=weights)


@pytest.mark.parametrize("seed", range(25))
def test_weighted_curve_matches_scikit_learn_with_ties_and_zero_weights(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 400
    labels = rng.integers(0, 2, n)
    # Few distinct values, so ties are everywhere — the hard case for curve building.
    scores = np.round(rng.random(n) + 0.3 * labels, 1)
    weights = rng.integers(0, 3, n)
    # Keep both classes present after zero weights.
    weights[np.flatnonzero(labels == 0)[0]] = 1
    weights[np.flatnonzero(labels == 1)[0]] = 1

    ranked = RankedScores(labels, scores)
    for target in (0.0, 0.01, 0.05, 0.2):
        expected = _sklearn_reference(labels, scores, weights, target)
        expected_recall, expected_threshold, expected_auc = expected
        curve = ranked.curve(weights)
        assert recall_at_fpr(curve, target) == (expected_recall, expected_threshold)
        assert roc_auc(curve) == pytest.approx(expected_auc, abs=1e-12)


def test_unweighted_curve_matches_the_vendored_published_function() -> None:
    rng = np.random.default_rng(7)
    labels = rng.integers(0, 2, 5_000)
    scores = rng.random(5_000) + 0.5 * labels
    curve = RankedScores(labels, scores).curve()
    for target in (0.001, 0.01, 0.1):
        assert recall_at_fpr(curve, target) == published_recall_at_fpr(labels, scores, target)


def test_non_finite_scores_are_rejected() -> None:
    with pytest.raises(ValueError, match="NaN"):
        RankedScores(np.array([0, 1]), np.array([0.1, np.nan]))


def test_cluster_bootstrap_gives_every_host_of_a_domain_the_same_weight() -> None:
    groups = np.array(["a.com", "a.com", "b.com", "c.com", "c.com", "c.com"])
    for weights in cluster_bootstrap_weights(groups, reps=50, seed=1):
        assert weights[0] == weights[1]
        assert weights[3] == weights[4] == weights[5]
        # Three domains are drawn per replicate, so domain counts sum to three.
        assert weights[0] + weights[2] + weights[3] == 3


def test_cluster_bootstrap_is_reproducible_for_a_seed() -> None:
    groups = np.array([f"d{i % 7}" for i in range(40)])
    first = np.stack(list(cluster_bootstrap_weights(groups, reps=10, seed=3)))
    second = np.stack(list(cluster_bootstrap_weights(groups, reps=10, seed=3)))
    assert np.array_equal(first, second)


def test_percentile_interval() -> None:
    assert interval(np.arange(1001) / 1000) == pytest.approx((0.025, 0.975))
