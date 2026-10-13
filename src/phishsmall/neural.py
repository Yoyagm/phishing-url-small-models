"""Neural model families: their architectures, inputs and fixed training settings.

Imported only by processes that train or score a neural model. PyTorch and
LightGBM ship different OpenMP runtimes, and on macOS loading both into one
process either segfaults or deadlocks (measured, both import orders), so
:mod:`phishsmall.train` never imports this module and :mod:`phishsmall.baseline`
together.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from phishsmall import runs, splits
from phishsmall.config import CHECKPOINTS, NEURAL_MODELS
from phishsmall.encoding import Encoded, encode_chars, encode_tokens
from phishsmall.models import BiGRU, CharCNN, PretrainedClassifier, count_parameters
from phishsmall.training import FitConfig, fit, pick_device, predict, seed_everything

PRETRAINED_ID = "distilbert-base-uncased"
# Pinned so a later push to the Hub cannot silently change the starting weights.
PRETRAINED_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
MAX_TOKENS = 64


@lru_cache(maxsize=1)
def _tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(PRETRAINED_ID, revision=PRETRAINED_REVISION)


def _pretrained() -> nn.Module:
    from transformers import AutoModelForSequenceClassification

    backbone = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_ID, revision=PRETRAINED_REVISION, num_labels=1
    )
    return PretrainedClassifier(backbone)


def _encode_subwords(hosts: list[str]) -> Encoded:
    return encode_tokens(hosts, _tokenizer(), MAX_TOKENS)


@dataclass(frozen=True)
class NeuralSpec:
    build: Callable[[], nn.Module]
    encode: Callable[[list[str]], Encoded]
    config: FitConfig
    extra_settings: dict


# Hyperparameters were fixed before any test-set result existed and are the same
# for every seed. Nothing here was tuned against either test set.
SPECS: dict[str, NeuralSpec] = {
    "char_cnn": NeuralSpec(
        build=CharCNN,
        encode=encode_chars,
        config=FitConfig(max_epochs=20, patience=3, batch_size=512, learning_rate=2e-3),
        extra_settings={"input": "characters"},
    ),
    "bigru": NeuralSpec(
        build=BiGRU,
        encode=encode_chars,
        config=FitConfig(max_epochs=20, patience=3, batch_size=512, learning_rate=2e-3),
        extra_settings={"input": "characters"},
    ),
    "distilbert": NeuralSpec(
        build=_pretrained,
        encode=_encode_subwords,
        config=FitConfig(max_epochs=3, patience=1, batch_size=128, learning_rate=5e-5,
                         eval_batch_size=512),
        extra_settings={"input": "WordPiece subwords", "pretrained": PRETRAINED_ID,
                        "revision": PRETRAINED_REVISION, "max_tokens": MAX_TOKENS},
    ),
}
if tuple(SPECS) != NEURAL_MODELS:  # pragma: no cover - a registry edit missed config.py
    raise RuntimeError(f"neural registry {tuple(SPECS)} disagrees with config {NEURAL_MODELS}")


def log(message: str) -> None:
    print(message, flush=True)


def train(name: str, seed: int, data: splits.Splits, universe: list[str]) -> None:
    """Fit on the ``train`` split, stop early on ``val``, then score the whole universe."""
    spec = SPECS[name]
    device = pick_device()
    train_rows, validation_rows = data.rows("train"), data.rows("val")

    seed_everything(seed)
    model = spec.build()
    log(f"  {count_parameters(model):,} trainable parameters on {device}")

    started = time.perf_counter()
    model, history, best_epoch = fit(
        model,
        spec.encode(train_rows["host"].tolist()), train_rows["y"].to_numpy(),
        spec.encode(validation_rows["host"].tolist()), validation_rows["y"].to_numpy(),
        spec.config, device, seed, log,
    )
    trained = time.perf_counter() - started
    scores = predict(model, spec.encode(universe), device, spec.config.eval_batch_size)

    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CHECKPOINTS / f"{runs.run_name(name, seed)}.pt")
    record = runs.RunRecord(
        model=name, seed=seed, train_rows=len(train_rows), device=str(device),
        n_parameters=count_parameters(model), train_seconds=round(trained, 1),
        score_seconds=round(time.perf_counter() - started - trained, 1),
        best_epoch=best_epoch, epochs_run=len(history), history=history,
        settings={**spec.config.to_dict(), **spec.extra_settings, "score": "logit"},
    )
    runs.save(record, universe, scores)
