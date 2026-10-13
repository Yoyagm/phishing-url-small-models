"""Run storage and the evaluation's guards, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from phishsmall import evaluation, runs
from phishsmall.config import PUBLISHED


@pytest.fixture
def scores_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(runs, "SCORES", tmp_path)
    return tmp_path


def _record(model: str = "char_cnn", seed: int = 0) -> runs.RunRecord:
    return runs.RunRecord(model=model, seed=seed, train_rows=10, device="cpu",
                          n_parameters=5, train_seconds=1.0)


def test_scores_round_trip_against_the_same_universe(scores_dir) -> None:
    universe = ["a.com", "b.com", "c.com"]
    runs.save(_record(), universe, np.array([0.1, 0.9, 0.5]))
    assert runs.load_scores("char_cnn", 0, universe).tolist() == [0.1, 0.9, 0.5]
    assert runs.available("char_cnn", (0, 1, 2)) == [0]


def test_scores_written_for_another_universe_are_refused(scores_dir) -> None:
    runs.save(_record(), ["a.com", "b.com"], np.array([0.1, 0.9]))
    with pytest.raises(ValueError, match="different evaluation universe"):
        runs.load_scores("char_cnn", 0, ["a.com", "z.com"])


def test_non_finite_scores_are_never_written(scores_dir) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        runs.save(_record(), ["a.com", "b.com"], np.array([0.1, np.inf]))


def test_reproduction_check_passes_on_the_published_numbers() -> None:
    report = evaluation.check_reproduction({name: dict(values) for name, values in PUBLISHED.items()})
    assert all(entry["match"] for metrics in report.values() for entry in metrics.values())


def test_reproduction_check_refuses_a_baseline_that_drifted() -> None:
    drifted = {name: dict(values) for name, values in PUBLISHED.items()}
    drifted["ood"]["recall_at_fpr_1pct"] += 1e-6
    with pytest.raises(evaluation.ReproductionError, match="ood recall_at_fpr_1pct"):
        evaluation.check_reproduction(drifted)


def test_phishing_concentration_counts_the_largest_domains() -> None:
    frame = pd.DataFrame({
        "registrable_domain": ["web.app"] * 6 + ["repl.co"] * 3 + ["x.com", "y.com"] + ["ok.com"] * 5,
        "y": [1] * 11 + [0] * 5,
    })
    concentration = evaluation.phishing_concentration(frame)
    assert concentration["top_domains"] == ["web.app", "repl.co"]
    assert concentration["share_of_phishing"] == pytest.approx(9 / 11)


def test_threshold_transfer_measures_the_fpr_actually_incurred() -> None:
    validation = pd.DataFrame({"host": [f"v{i}" for i in range(200)], "y": [0] * 100 + [1] * 100})
    ood = pd.DataFrame({"host": [f"o{i}" for i in range(200)], "y": [0] * 100 + [1] * 100})
    scores = {f"v{i}": i / 100 for i in range(100)}             # validation benign: 0.00-0.99
    scores.update({f"v{i}": 2.0 for i in range(100, 200)})      # validation phishing: all above
    scores.update({f"o{i}": 2.0 for i in range(10)})            # 10 OOD benign drift up to it
    scores.update({f"o{i}": 0.0 for i in range(10, 100)})
    scores.update({f"o{i}": 2.0 for i in range(100, 200)})

    def lookup(model, seed, hosts):
        return np.array([scores[host] for host in hosts])

    report = evaluation.threshold_transfer(validation, ood, lookup, {"char_cnn": [0]})
    entry = report["char_cnn"]["per_seed"][0]
    # Fixed on validation at FPR 1 %, the threshold (2.0) flags the 10 drifted benign hosts.
    assert entry["threshold"] == 2.0
    assert entry["ood_fp"] == 10
    assert entry["ood_fpr"] == pytest.approx(0.10)
    assert entry["ood_recall"] == 1.0
