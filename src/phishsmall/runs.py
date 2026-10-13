"""Storage for one training run: its scores over the evaluation universe, and its log.

A run never stores metrics, only raw scores. Metrics are computed in one place
(:mod:`phishsmall.evaluation`) for every model alike, so no model can be scored
by a slightly different function than another.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

import numpy as np

from phishsmall.config import SCORES


@dataclass
class RunRecord:
    """What happened during training, kept for the README's cost column and for audit."""

    model: str
    seed: int
    train_rows: int
    device: str
    n_parameters: int
    train_seconds: float
    score_seconds: float = 0.0
    best_epoch: int | None = None
    epochs_run: int | None = None
    history: list[dict] = field(default_factory=list)
    settings: dict = field(default_factory=dict)


def run_name(model: str, seed: int) -> str:
    return f"{model}-seed{seed}"


def universe_digest(universe: list[str]) -> str:
    return hashlib.sha256("\n".join(universe).encode("utf-8")).hexdigest()


def save(record: RunRecord, universe: list[str], scores: np.ndarray) -> None:
    if len(scores) != len(universe):
        raise ValueError(f"{len(scores)} scores for {len(universe)} hosts")
    if not np.isfinite(scores).all():
        raise ValueError(f"{run_name(record.model, record.seed)} produced non-finite scores")
    SCORES.mkdir(parents=True, exist_ok=True)
    name = run_name(record.model, record.seed)
    np.savez_compressed(
        SCORES / f"{name}.npz",
        scores=np.asarray(scores, dtype=np.float64),
        universe_sha256=np.array(universe_digest(universe)),
    )
    (SCORES / f"{name}.json").write_text(json.dumps(asdict(record), indent=2))


def load_scores(model: str, seed: int, universe: list[str]) -> np.ndarray:
    """Scores aligned to ``universe``, refusing files written against another one."""
    name = run_name(model, seed)
    with np.load(SCORES / f"{name}.npz") as stored:
        digest = str(stored["universe_sha256"])
        scores = stored["scores"]
    if digest != universe_digest(universe):
        raise ValueError(
            f"{name} was scored against a different evaluation universe; "
            "the splits changed since it was trained. Re-run: make train"
        )
    return scores


def load_record(model: str, seed: int) -> dict:
    return json.loads((SCORES / f"{run_name(model, seed)}.json").read_text())


def available(model: str, seeds: tuple[int, ...]) -> list[int]:
    return [seed for seed in seeds if (SCORES / f"{run_name(model, seed)}.npz").exists()]
