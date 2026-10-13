"""One training loop for every neural model, with grouped-validation early stopping.

The loop knows nothing about the architecture: it sees ``model(ids, lengths) ->
logits`` and an :class:`~phishsmall.encoding.Encoded` input. Early stopping reads
ROC-AUC on the *validation* split — grouped by registrable domain and disjoint
from both test sets — and the weights restored at the end are those of the best
validation epoch. The test sets are never looked at during training.
"""

from __future__ import annotations

import copy
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from phishsmall.encoding import Encoded


@dataclass(frozen=True)
class FitConfig:
    max_epochs: int
    patience: int
    batch_size: int
    learning_rate: float
    weight_decay: float = 0.01
    warmup_fraction: float = 0.06
    gradient_clip: float = 1.0
    eval_batch_size: int = 1024
    # Abandon training if one epoch would push the run past this budget.
    time_budget_seconds: float = 40 * 60

    def to_dict(self) -> dict:
        return asdict(self)


def pick_device() -> torch.device:
    """Apple-silicon GPU when present, otherwise CPU. ``PHISHSMALL_DEVICE`` overrides."""
    requested = os.environ.get("PHISHSMALL_DEVICE")
    if requested:
        return torch.device(requested)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def length_pooled_batches(lengths: torch.Tensor, batch_size: int, generator: torch.Generator,
                          pool_batches: int = 50) -> list[torch.Tensor]:
    """Shuffled batches whose members have similar lengths.

    Host names are short (median 16 characters, 11 subword tokens) but a batch of
    random rows almost always contains one long outlier, and the whole batch is
    padded to it. Sorting within pools of 50 batches, then shuffling the batch
    order, cut the measured transformer step time from 354 ms to 150 ms while every
    row is still seen exactly once per epoch in a seed-determined order.
    """
    permutation = torch.randperm(len(lengths), generator=generator)
    batches: list[torch.Tensor] = []
    pool = batch_size * pool_batches
    for start in range(0, len(permutation), pool):
        chunk = permutation[start : start + pool]
        chunk = chunk[torch.argsort(lengths[chunk], stable=True)]
        batches.extend(chunk.split(batch_size))
    order = torch.randperm(len(batches), generator=generator)
    return [batches[i] for i in order]


@torch.no_grad()
def predict(model: nn.Module, inputs: Encoded, device: torch.device, batch_size: int) -> np.ndarray:
    """Logits for every row of ``inputs``, in input order.

    Rows are processed shortest-first so each batch carries little padding, then
    put back in their original order.
    """
    model.eval()
    order = torch.argsort(inputs.lengths, stable=True)
    logits = torch.empty(len(inputs), dtype=torch.float32)
    for start in range(0, len(inputs), batch_size):
        index = order[start : start + batch_size]
        ids, lengths = inputs.batch(index)
        logits[index] = model(ids.to(device), lengths.to(device)).float().cpu()
    return logits.double().numpy()


def fit(
    model: nn.Module,
    train: Encoded,
    train_labels: np.ndarray,
    validation: Encoded,
    validation_labels: np.ndarray,
    config: FitConfig,
    device: torch.device,
    seed: int,
    log: Callable[[str], None] = print,
) -> tuple[nn.Module, list[dict], int]:
    """Train with AdamW and a warm-up/linear-decay schedule; return the best-validation model.

    Returns ``(model, history, best_epoch)``. ``history`` holds one entry per
    epoch: training loss, validation loss and validation ROC-AUC, and seconds.
    """
    model.to(device)
    # torch.tensor copies: pandas hands out read-only arrays under copy-on-write.
    labels = torch.tensor(train_labels, dtype=torch.float32)
    steps_per_epoch = len(length_pooled_batches(train.lengths, config.batch_size, torch.Generator()))
    total_steps = steps_per_epoch * config.max_epochs
    warmup_steps = max(1, int(config.warmup_fraction * total_steps))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps,
                         max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))),
    )
    loss_function = nn.BCEWithLogitsLoss()
    shuffler = torch.Generator().manual_seed(seed)

    history: list[dict] = []
    best_auc, best_epoch, best_state = -1.0, 0, None
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        running, seen = 0.0, 0
        for index in length_pooled_batches(train.lengths, config.batch_size, shuffler):
            ids, lengths = train.batch(index)
            target = labels[index].to(device)
            logits = model(ids.to(device), lengths.to(device))
            loss = loss_function(logits, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            schedule.step()

            running += float(loss.detach()) * len(index)
            seen += len(index)

        validation_logits = predict(model, validation, device, config.eval_batch_size)
        validation_auc = float(roc_auc_score(validation_labels, validation_logits))
        validation_loss = float(loss_function(
            torch.as_tensor(validation_logits), torch.as_tensor(validation_labels, dtype=torch.float64)
        ))
        seconds = time.perf_counter() - epoch_started
        history.append({
            "epoch": epoch,
            "train_loss": running / seen,
            "val_loss": validation_loss,
            "val_roc_auc": validation_auc,
            "seconds": round(seconds, 1),
        })
        log(f"  epoch {epoch:2d}  train loss {running / seen:.4f}  "
            f"val loss {validation_loss:.4f}  val AUC {validation_auc:.4f}  ({seconds:.0f}s)")

        if validation_auc > best_auc:
            best_auc, best_epoch = validation_auc, epoch
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        elif epoch - best_epoch >= config.patience:
            log(f"  early stop: no validation gain for {config.patience} epochs")
            break

        elapsed = time.perf_counter() - started
        if elapsed + seconds > config.time_budget_seconds:
            budget = config.time_budget_seconds / 60
            log(f"  stopping: another epoch would exceed the {budget:.0f}-min budget")
            break

    if best_state is None:  # pragma: no cover - max_epochs >= 1 always sets it
        raise RuntimeError("training finished without a single evaluated epoch")
    model.load_state_dict(best_state)
    return model, history, best_epoch
