"""Train one model family for one or more seeds, and score the evaluation universe.

Every run trains on PhiUSIIL hosts only and writes raw scores for every host any
evaluation set contains; metrics are computed later, identically for all models,
by :mod:`phishsmall.evaluation`. Runs whose scores already exist are skipped, so an
interrupted ``make train`` resumes where it stopped.

LightGBM and the neural models are imported lazily and never into the same
process (see :mod:`phishsmall.neural` for why).

Run::

    uv run python -m phishsmall.train --model char_cnn --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import time

from phishsmall import runs, splits
from phishsmall.config import (
    LIGHTGBM_MODELS,
    MODEL_NAMES,
    NEURAL_MODELS,
    PUBLISHED_BASELINE_SEED,
    TRAIN_SEEDS,
)

# The published baseline trains on the original 80 % (train + val) with seed 42.
# Re-seeding it on the same rows measures how much of its number is the seed; the
# control trains on exactly the rows the neural models see.
LIGHTGBM_TRAINING_SPLITS: dict[str, tuple[str, ...]] = {
    "lightgbm_published": ("train", "val"),
    "lightgbm_reseeded": ("train", "val"),
    "lightgbm": ("train",),
}


def log(message: str) -> None:
    print(message, flush=True)


def train_lightgbm(name: str, seed: int, data: splits.Splits, universe: list[str]) -> None:
    from phishsmall import baseline

    split_names = LIGHTGBM_TRAINING_SPLITS[name]
    rows = data.rows(*split_names)
    started = time.perf_counter()
    model = baseline.fit(rows["host"].tolist(), rows["y"].to_numpy(), seed)
    trained = time.perf_counter() - started
    scores = baseline.score(model, universe)
    record = runs.RunRecord(
        model=name, seed=seed, train_rows=len(rows), device="cpu",
        n_parameters=int(model.num_trees()), train_seconds=round(trained, 1),
        score_seconds=round(time.perf_counter() - started - trained, 1),
        settings={**baseline.settings(), "seed": seed, "trained_on": list(split_names),
                  "n_parameters_counts": "trees", "score": "probability"},
    )
    runs.save(record, universe, scores)


def default_seeds(name: str) -> tuple[int, ...]:
    return (PUBLISHED_BASELINE_SEED,) if name == "lightgbm_published" else TRAIN_SEEDS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--seeds", type=int, nargs="+", help="defaults to the study's seeds")
    parser.add_argument("--force", action="store_true", help="retrain even if scores exist")
    args = parser.parse_args(argv)

    seeds = tuple(args.seeds) if args.seeds else default_seeds(args.model)
    if args.model == "lightgbm_published" and seeds != (PUBLISHED_BASELINE_SEED,):
        parser.error(f"the published baseline is defined by seed {PUBLISHED_BASELINE_SEED}")

    data = splits.load()
    universe = data.evaluation_universe()
    for seed in seeds:
        name = runs.run_name(args.model, seed)
        if not args.force and runs.available(args.model, (seed,)):
            log(f"{name}: scores already exist, skipping (use --force to retrain)")
            continue
        log(f"{name}: training")
        if args.model in LIGHTGBM_MODELS:
            train_lightgbm(args.model, seed, data, universe)
        elif args.model in NEURAL_MODELS:
            from phishsmall import neural

            neural.train(args.model, seed, data, universe)
        log(f"{name}: done")


if __name__ == "__main__":
    main()
