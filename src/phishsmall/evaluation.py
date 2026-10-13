"""Score every trained run on every evaluation set and write ``reports/results.json``.

Point estimates come from the vendored :func:`phishurl.evaluate.compute` — the
function that produced the published numbers — applied identically to every
model. Uncertainty comes from a bootstrap over registrable domains (the unit the
split is grouped by), with the *same* resamples for every model, so differences
between models are paired.

Before anything is written, the reproduced LightGBM must match the published
metrics to 1e-9. If it does not, the comparison is not against the published
model and the script refuses to continue.

Run::

    uv run python -m phishsmall.evaluation
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from phishsmall import runs, splits
from phishsmall.config import (
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    DATA,
    MODEL_NAMES,
    PUBLISHED,
    PUBLISHED_BASELINE_SEED,
    PUBLISHED_OPENPHISH_RATE_ROWS,
    REPRODUCTION_TOLERANCE,
    RESULTS_PATH,
    ROOT,
    STRICT_FPR,
    TARGET_FPR,
    TRAIN_SEEDS,
    TRAINED_ON_VALIDATION,
)
from phishsmall.defang import defang
from phishsmall.metrics import (
    RankedScores,
    cluster_bootstrap_weights,
    interval,
    recall_at_fpr,
    roc_auc,
)
from phishurl.evaluate import compute
from phishurl.evaluate import recall_at_fpr as published_recall_at_fpr

TEST_SETS = ("internal", "ood")
REFERENCE_MODEL = "lightgbm_published"
# Two baselines to measure against. The published run is the number the research
# question names; the control is LightGBM trained on exactly the neural models'
# rows and averaged over the same seeds, which separates "a different model" from
# "fewer rows" and "a lucky or unlucky seed".
DELTA_REFERENCES = (REFERENCE_MODEL, "lightgbm")


class ReproductionError(RuntimeError):
    """The reproduced baseline does not match the published numbers."""


def _model_seeds(model: str) -> tuple[int, ...]:
    return (PUBLISHED_BASELINE_SEED,) if model == REFERENCE_MODEL else TRAIN_SEEDS


def available_models() -> dict[str, list[int]]:
    """Models with every one of their seeds trained. Partial families are left out."""
    found: dict[str, list[int]] = {}
    for model in MODEL_NAMES:
        seeds = _model_seeds(model)
        present = runs.available(model, seeds)
        if len(present) == len(seeds):
            found[model] = present
        elif present:
            print(f"  skipping {model}: {len(present)}/{len(seeds)} seeds trained")
    if REFERENCE_MODEL not in found:
        raise SystemExit("The reproduced baseline has not been trained. Run: make train")
    return found


def _set_summary(frame: pd.DataFrame) -> dict:
    summary = {
        "n": int(len(frame)),
        "n_phishing": int(frame["y"].sum()),
        "sha256": splits.fingerprint(frame["host"], frame["y"]),
    }
    if "registrable_domain" in frame:
        summary["registrable_domains"] = int(frame["registrable_domain"].nunique())
        summary["phishing_concentration"] = phishing_concentration(frame)
    return summary


def phishing_concentration(frame: pd.DataFrame, top: int = 2) -> dict:
    """How much of a set's phishing sits under its ``top`` largest registrable domains.

    A set whose phishing is mostly two hosting platforms measures, in effect, two
    decisions; this is what makes a domain-level confidence interval wide. Dedicated
    phishing domains are defanged before they are written anywhere.
    """
    phishing = frame[frame["y"] == 1]
    sizes = phishing["registrable_domain"].value_counts()
    return {
        "top_domains": [defang(domain) for domain in sizes.index[:top]],
        "top_domain_hosts": [int(size) for size in sizes.iloc[:top]],
        "share_of_phishing": float(sizes.iloc[:top].sum() / max(1, len(phishing))),
    }


def _metrics(y: np.ndarray, scores: np.ndarray) -> dict:
    measured = compute("", y, scores)
    return {
        "roc_auc": measured.roc_auc,
        "pr_auc": measured.pr_auc,
        "recall_at_fpr_1pct": measured.recall_at_fpr_1pct,
        "threshold_at_fpr_1pct": measured.threshold_at_fpr_1pct,
        "recall_at_fpr_01pct": measured.recall_at_fpr_01pct,
        "precision_at_operating_point": measured.precision_at_operating_point,
        "confusion": measured.confusion,
    }


def check_reproduction(reference: dict[str, dict]) -> dict:
    """Compare the reproduced baseline's metrics per test set with the published ones.

    Raises :class:`ReproductionError` on any gap above the tolerance.
    """
    report: dict[str, dict] = {}
    for test_set, published in PUBLISHED.items():
        reproduced = reference[test_set]
        report[test_set] = {}
        for metric, expected in published.items():
            got = reproduced[metric]
            matches = abs(got - expected) <= REPRODUCTION_TOLERANCE
            report[test_set][metric] = {"published": expected, "reproduced": got, "match": matches}
            if not matches:
                raise ReproductionError(
                    f"{test_set} {metric}: reproduced {got!r} vs published {expected!r}. "
                    "The baseline is not the published model; no comparison is valid."
                )
    return report


def bootstrap(frame: pd.DataFrame, per_model: dict[str, list[np.ndarray]], reps: int,
              seed: int) -> dict[str, dict[str, np.ndarray]]:
    """Bootstrap ROC-AUC and recall@1 % FPR of every model's seed-mean, on shared resamples."""
    y = frame["y"].to_numpy()
    ranked = {model: [RankedScores(y, scores) for scores in seeds] for model, seeds in per_model.items()}
    samples = {model: {"roc_auc": np.empty(reps), "recall_at_fpr_1pct": np.empty(reps)}
               for model in per_model}
    weights = cluster_bootstrap_weights(frame["registrable_domain"].to_numpy(), reps, seed)
    for rep, w in enumerate(weights):
        for model, seeds in ranked.items():
            curves = [entry.curve(w) for entry in seeds]
            samples[model]["roc_auc"][rep] = np.mean([roc_auc(curve) for curve in curves])
            samples[model]["recall_at_fpr_1pct"][rep] = np.mean(
                [recall_at_fpr(curve, TARGET_FPR)[0] for curve in curves]
            )
    return samples


def _mean_and_sd(per_seed: list[dict], key: str) -> tuple[float, float]:
    values = np.array([entry[key] for entry in per_seed], dtype=float)
    spread = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return float(values.mean()), spread


def evaluate_test_set(frame: pd.DataFrame, per_model: dict[str, list[np.ndarray]],
                      seeds: dict[str, list[int]], reps: int) -> dict:
    y = frame["y"].to_numpy()
    samples = bootstrap(frame, per_model, reps, BOOTSTRAP_SEED)
    results: dict[str, dict] = {}
    for model, all_scores in per_model.items():
        per_seed = [{"seed": seed, **_metrics(y, scores)}
                    for seed, scores in zip(seeds[model], all_scores, strict=True)]
        summary: dict = {"per_seed": per_seed, "mean": {}, "sd": {}, "ci95": {}, "delta_vs": {}}
        for key in ("roc_auc", "pr_auc", "recall_at_fpr_1pct", "recall_at_fpr_01pct",
                    "precision_at_operating_point"):
            summary["mean"][key], summary["sd"][key] = _mean_and_sd(per_seed, key)
        for key in ("tp", "fp", "fn", "tn"):
            summary["mean"][key] = float(np.mean([entry["confusion"][key] for entry in per_seed]))
        for key in ("roc_auc", "recall_at_fpr_1pct"):
            summary["ci95"][key] = interval(samples[model][key])
        results[model] = summary

    for model, summary in results.items():
        for reference in DELTA_REFERENCES:
            if model == reference or reference not in results:
                continue
            summary["delta_vs"][reference] = {
                key: {
                    "point": summary["mean"][key] - results[reference]["mean"][key],
                    "ci95": interval(samples[model][key] - samples[reference][key]),
                    "share_of_resamples_above_zero": float(
                        np.mean(samples[model][key] > samples[reference][key])
                    ),
                }
                for key in ("roc_auc", "recall_at_fpr_1pct")
            }
    return results


def openphish_rates(openphish: pd.DataFrame, ood: pd.DataFrame, lookup, models: dict) -> dict:
    """Share of OpenPhish detected at each model's OOD operating point.

    Reported per URL row (how the original computed its 58.5 %) and per unique host.
    """
    rows = {}
    unique = openphish.drop_duplicates("host")
    for model, seeds in models.items():
        per_seed = []
        for seed in seeds:
            ood_scores = lookup(model, seed, ood["host"])
            _, threshold = published_recall_at_fpr(ood["y"].to_numpy(), ood_scores, TARGET_FPR)
            per_seed.append({
                "seed": seed,
                "threshold": threshold,
                "rate_rows": float((lookup(model, seed, openphish["host"]) >= threshold).mean()),
                "rate_hosts": float((lookup(model, seed, unique["host"]) >= threshold).mean()),
            })
        rows[model] = {
            "per_seed": per_seed,
            "mean_rate_hosts": float(np.mean([entry["rate_hosts"] for entry in per_seed])),
            "mean_rate_rows": float(np.mean([entry["rate_rows"] for entry in per_seed])),
        }
    return rows


def threshold_transfer(validation: pd.DataFrame, ood: pd.DataFrame, lookup, models: dict) -> dict:
    """Ablation: pick the 1 %-FPR threshold where a deployer could — on validation data.

    Recall at a fixed FPR, as reported in the main table, reads the threshold off
    the test set's own ROC curve. A deployed model has to fix it in advance; here
    it is fixed on the grouped validation split and carried to the OOD set
    unchanged, and the false-positive rate it actually produces there is measured.
    Models trained on the validation hosts are excluded: for them it is not held out.
    """
    report = {}
    y_ood = ood["y"].to_numpy()
    negatives, positives = int((y_ood == 0).sum()), int(y_ood.sum())
    for model, seeds in models.items():
        if model in TRAINED_ON_VALIDATION:
            continue
        per_seed = []
        for seed in seeds:
            _, threshold = published_recall_at_fpr(
                validation["y"].to_numpy(), lookup(model, seed, validation["host"]), TARGET_FPR
            )
            flagged = lookup(model, seed, ood["host"]) >= threshold
            fp = int((flagged & (y_ood == 0)).sum())
            tp = int((flagged & (y_ood == 1)).sum())
            per_seed.append({"seed": seed, "threshold": threshold, "ood_fp": fp, "ood_tp": tp,
                             "ood_fpr": fp / negatives, "ood_recall": tp / positives})
        report[model] = {
            "per_seed": per_seed,
            "mean": {key: float(np.mean([entry[key] for entry in per_seed]))
                     for key in ("ood_fpr", "ood_recall", "ood_fp", "ood_tp")},
        }
    return report


def host_traits(hosts: pd.Series) -> dict:
    """Surface traits of a set of host names: the vocabulary for describing false positives."""
    return {
        "n": int(len(hosts)),
        "share_with_digit": float(hosts.str.contains(r"\d", regex=True).mean()),
        "share_punycode": float(hosts.str.contains("xn--", regex=False).mean()),
        "share_with_hyphen": float(hosts.str.contains("-", regex=False).mean()),
        "mean_length": float(hosts.str.len().mean()),
    }


def false_positive_profile(ood: pd.DataFrame, lookup, models: dict) -> dict:
    """What the benign hosts each model flags at its 1 %-FPR operating point look like.

    Compared with the benign OOD hosts as a whole, this shows which surface traits a
    model has learned to treat as phishing. Averaged over seeds.
    """
    benign = ood[ood["y"] == 0]
    profile = {"all_benign": host_traits(benign["host"])}
    for model, seeds in models.items():
        traits = []
        for seed in seeds:
            ood_scores = lookup(model, seed, ood["host"])
            _, threshold = published_recall_at_fpr(ood["y"].to_numpy(), ood_scores, TARGET_FPR)
            flagged = benign[lookup(model, seed, benign["host"]) >= threshold]
            traits.append(host_traits(flagged["host"]))
        profile[model] = {key: float(np.mean([entry[key] for entry in traits])) for key in traits[0]}
    return profile


def unfiltered_ood(filtered: pd.DataFrame, unfiltered: pd.DataFrame, lookup, models: dict) -> dict:
    """Ablation: what each model would have reported had seen hosts not been removed."""
    report = {}
    for model, seeds in models.items():
        per_seed = []
        for seed in seeds:
            clean = _metrics(filtered["y"].to_numpy(), lookup(model, seed, filtered["host"]))
            naive = _metrics(unfiltered["y"].to_numpy(), lookup(model, seed, unfiltered["host"]))
            per_seed.append({
                "seed": seed,
                "filtered": {key: clean[key] for key in ("roc_auc", "recall_at_fpr_1pct")},
                "unfiltered": {key: naive[key] for key in ("roc_auc", "recall_at_fpr_1pct")},
            })
        report[model] = {
            "per_seed": per_seed,
            "mean_inflation": {
                key: float(np.mean([entry["unfiltered"][key] - entry["filtered"][key] for entry in per_seed]))
                for key in ("roc_auc", "recall_at_fpr_1pct")
            },
        }
    return report


def raw_data_hashes() -> dict[str, str]:
    """SHA-256 of each raw input, so a reader can tell whether they hold the same capture."""
    hashes = {}
    for path in sorted((DATA / "raw").rglob("*")):
        if path.is_file() and not path.name.startswith("."):
            hashes[str(path.relative_to(DATA))] = _sha256(path)
    return hashes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def model_summaries(models: dict[str, list[int]]) -> dict:
    summaries = {}
    for model, seeds in models.items():
        records = [runs.load_record(model, seed) for seed in seeds]
        summaries[model] = {
            "seeds": seeds,
            "train_rows": records[0]["train_rows"],
            "n_parameters": records[0]["n_parameters"],
            "device": records[0]["device"],
            "train_minutes_mean": float(np.mean([r["train_seconds"] for r in records]) / 60),
            "best_epochs": [r["best_epoch"] for r in records],
            "epochs_run": [r["epochs_run"] for r in records],
            "best_val_roc_auc": [
                max(entry["val_roc_auc"] for entry in r["history"]) if r["history"] else None
                for r in records
            ],
            "settings": records[0]["settings"],
            "history_per_seed": [r["history"] for r in records],
        }
    return summaries


@dataclass(frozen=True)
class CurveInputs:
    """What the figures need beyond the JSON: the test frames and per-run scores."""

    frames: dict[str, pd.DataFrame]
    lookup: Callable[[str, int, pd.Series], np.ndarray]
    models: dict[str, list[int]]


def run(reps: int = BOOTSTRAP_REPS) -> tuple[dict, CurveInputs]:
    data = splits.load()
    universe = data.evaluation_universe()
    models = available_models()
    position = pd.Series(np.arange(len(universe)), index=universe)
    cache: dict[tuple[str, int], np.ndarray] = {}

    def lookup(model: str, seed: int, hosts: pd.Series) -> np.ndarray:
        """Scores of ``hosts`` from one run, in the order of ``hosts``."""
        if (model, seed) not in cache:
            cache[(model, seed)] = runs.load_scores(model, seed, universe)
        return cache[(model, seed)][position.loc[hosts.to_numpy()].to_numpy()]

    frames = {"internal": data.rows("test"), "ood": data.ood}
    reproduction = check_reproduction({
        test_set: _metrics(frame["y"].to_numpy(),
                           lookup(REFERENCE_MODEL, PUBLISHED_BASELINE_SEED, frame["host"]))
        for test_set, frame in frames.items()
    })
    print("  reproduced baseline matches the published metrics")

    results: dict[str, dict] = {model: {} for model in models}
    for test_set in TEST_SETS:
        frame = frames[test_set]
        per_model = {model: [lookup(model, seed, frame["host"]) for seed in seeds]
                     for model, seeds in models.items()}
        print(f"  {test_set}: {len(frame):,} hosts, {reps} domain-level bootstrap resamples")
        for model, summary in evaluate_test_set(frame, per_model, models, reps).items():
            results[model][test_set] = summary

    validation = data.rows("val")
    for model, seeds in models.items():
        if model not in TRAINED_ON_VALIDATION:
            results[model]["validation"] = {"roc_auc_per_seed": [
                float(roc_auc_score(validation["y"], lookup(model, seed, validation["host"])))
                for seed in seeds
            ]}

    openphish = openphish_rates(data.openphish, data.ood, lookup, models)
    published_openphish = openphish[REFERENCE_MODEL]["per_seed"][0]["rate_rows"]
    if abs(published_openphish - PUBLISHED_OPENPHISH_RATE_ROWS) > REPRODUCTION_TOLERANCE:
        raise ReproductionError(
            f"OpenPhish rate reproduced as {published_openphish!r}, "
            f"published {PUBLISHED_OPENPHISH_RATE_ROWS!r}"
        )

    report = {
        "meta": {
            "generated": date.today().isoformat(),
            "target_fpr": TARGET_FPR,
            "strict_fpr": STRICT_FPR,
            "bootstrap": {"reps": reps, "seed": BOOTSTRAP_SEED, "unit": "registrable domain",
                          "level": 0.95, "statistic": "mean over seeds"},
            "reference_model": REFERENCE_MODEL,
            "reproduction": {**reproduction, "openphish_rate_rows": published_openphish},
            "raw_data_sha256": raw_data_hashes(),
            "sets": {
                "train": _set_summary(data.rows("train")),
                "val": _set_summary(data.rows("val")),
                "internal": _set_summary(data.rows("test")),
                "ood": _set_summary(data.ood),
                "ood_unfiltered": _set_summary(data.ood_unfiltered),
                "openphish_rows": _set_summary(data.openphish),
                "openphish_hosts": _set_summary(data.openphish.drop_duplicates("host")),
            },
            "ood_composition": data.stats["ood_composition"],
            "dropped_conflicting_hosts": data.stats["dropped_conflicting_hosts"],
        },
        "models": model_summaries(models),
        "results": results,
        "openphish": openphish,
        "ablations": {
            "threshold_transfer": threshold_transfer(data.rows("val"), data.ood, lookup, models),
            "unfiltered_ood": unfiltered_ood(data.ood, data.ood_unfiltered, lookup, models),
        },
        "false_positive_profile": false_positive_profile(data.ood, lookup, models),
    }
    return report, CurveInputs(frames=frames, lookup=lookup, models=models)


def main() -> None:
    from phishsmall import figures

    print("Evaluating every trained run:")
    report, curves = run()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"  wrote {RESULTS_PATH.relative_to(ROOT)}")
    for path in figures.render(report, curves):
        print(f"  wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
