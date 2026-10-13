"""The LightGBM baseline, trained with the original's exact features and parameters.

Imported, not copied: ``PARAMS``, ``NUM_ROUNDS`` and ``build_matrix`` come from the
vendored :mod:`phishurl.train`, so the only things this module decides are which
rows to train on and which seed to use.
"""

from __future__ import annotations

import lightgbm as lgb
import numpy as np

from phishurl.features import FEATURE_NAMES
from phishurl.train import NUM_ROUNDS, PARAMS, build_matrix


def fit(hosts: list[str], labels: np.ndarray, seed: int) -> lgb.Booster:
    dataset = lgb.Dataset(build_matrix(hosts), label=labels, feature_name=list(FEATURE_NAMES))
    return lgb.train({**PARAMS, "seed": seed}, dataset, num_boost_round=NUM_ROUNDS)


def score(model: lgb.Booster, hosts: list[str]) -> np.ndarray:
    return model.predict(build_matrix(hosts))


def settings() -> dict:
    return {"params": PARAMS, "num_boost_round": NUM_ROUNDS, "n_features": len(FEATURE_NAMES)}
