"""Regenerate every number in the README from ``reports/results.json``.

The README and the model card hold marker pairs — ``<!-- BEGIN:name -->`` … ``<!-- END:name -->`` —
and everything between them is overwritten here. No result in the README is
typed by hand: change the pipeline, re-run ``make eval tables``, and the README
changes with it. A missing marker is an error, not a silent skip, so a block
cannot quietly stop updating.

Run::

    uv run python -m phishsmall.tables
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path

from phishsmall.config import MODEL_LABELS, MODEL_NAMES, RESULTS_PATH, ROOT
from phishsmall.defang import defang

README = ROOT / "README.md"
MODEL_CARD = ROOT / "MODEL_CARD.md"
# The model card repeats the blocks a model user needs; the README carries all of them.
MODEL_CARD_BLOCKS = ("headline", "results", "threshold", "false_positives")
TEST_SET_LABELS = {"internal": "Internal hold-out", "ood": "OOD + temporal"}
NEURAL_DELTA_REFERENCES = {
    "lightgbm_published": "vs published LightGBM",
    "lightgbm": "vs LightGBM control",
}


class MissingMarkerError(ValueError):
    """The README lacks a block this module is supposed to fill."""


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f} %"


def signed_points(value: float) -> str:
    return f"{100 * value:+.1f}"


def interval_pct(bounds: list[float]) -> str:
    return f"[{100 * bounds[0]:.1f}, {100 * bounds[1]:.1f}]"


def interval_auc(bounds: list[float]) -> str:
    return f"[{bounds[0]:.3f}, {bounds[1]:.3f}]"


def _models(report: dict) -> list[str]:
    return [model for model in MODEL_NAMES if model in report["results"]]


def _seed_note(report: dict, model: str) -> str:
    seeds = report["models"][model]["seeds"]
    return f"{len(seeds)} ({', '.join(str(seed) for seed in seeds)})"


def _per_seed(entry: dict, key: str) -> str:
    """Each seed's value: with three seeds a standard deviation would hide more than it shows."""
    values = [entry_seed[key] for entry_seed in entry["per_seed"]]
    return "—" if len(values) == 1 else " / ".join(f"{100 * value:.1f}" for value in values)


def results_table(report: dict) -> str:
    lines = [
        "| Model | Seeds | Test set | ROC-AUC [95 % CI] | Recall @ FPR 1 % [95 % CI] | Recall per seed "
        "| TP | FP | Recall @ FPR 0.1 % |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for test_set in ("ood", "internal"):
        for model in _models(report):
            entry = report["results"][model][test_set]
            mean = entry["mean"]
            lines.append(
                f"| {MODEL_LABELS[model]} | {_seed_note(report, model)} | {TEST_SET_LABELS[test_set]} "
                f"| {mean['roc_auc']:.4f} {interval_auc(entry['ci95']['roc_auc'])} "
                f"| **{pct(mean['recall_at_fpr_1pct'])}** "
                f"{interval_pct(entry['ci95']['recall_at_fpr_1pct'])} "
                f"| {_per_seed(entry, 'recall_at_fpr_1pct')} "
                f"| {mean['tp']:,.0f} | {mean['fp']:,.0f} | {pct(mean['recall_at_fpr_01pct'])} |"
            )
    return "\n".join(lines)


def delta_table(report: dict) -> str:
    lines = [
        "| Model | Test set | Compared with | Δ recall @ FPR 1 % (pts) [95 % CI] "
        "| Δ ROC-AUC [95 % CI] | Resamples where the model is ahead |",
        "|---|---|---|---:|---:|---:|",
    ]
    for test_set in ("ood", "internal"):
        for model in _models(report):
            for reference, label in NEURAL_DELTA_REFERENCES.items():
                delta = report["results"][model][test_set]["delta_vs"].get(reference)
                if delta is None or model.startswith("lightgbm"):
                    continue
                recall, auc = delta["recall_at_fpr_1pct"], delta["roc_auc"]
                lines.append(
                    f"| {MODEL_LABELS[model]} | {TEST_SET_LABELS[test_set]} | {label} "
                    f"| {signed_points(recall['point'])} "
                    f"[{signed_points(recall['ci95'][0])}, {signed_points(recall['ci95'][1])}] "
                    f"| {auc['point']:+.4f} [{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] "
                    f"| {pct(recall['share_of_resamples_above_zero'], 0)} |"
                )
    # LightGBM against itself: how far the published number moves with nothing but
    # the seed, or 10 % fewer training domains. Context for every row above.
    for model in _models(report):
        if not model.startswith("lightgbm") or model == "lightgbm_published":
            continue
        delta = report["results"][model]["ood"]["delta_vs"]["lightgbm_published"]
        recall, auc = delta["recall_at_fpr_1pct"], delta["roc_auc"]
        lines.append(
            f"| *{MODEL_LABELS[model]}* | {TEST_SET_LABELS['ood']} | vs published LightGBM "
            f"| {signed_points(recall['point'])} "
            f"[{signed_points(recall['ci95'][0])}, {signed_points(recall['ci95'][1])}] "
            f"| {auc['point']:+.4f} [{auc['ci95'][0]:+.4f}, {auc['ci95'][1]:+.4f}] "
            f"| {pct(recall['share_of_resamples_above_zero'], 0)} |"
        )
    return "\n".join(lines)


def threshold_table(report: dict) -> str:
    transfer = report["ablations"]["threshold_transfer"]
    lines = [
        "| Model | OOD recall, threshold read off the OOD curve | OOD recall, threshold fixed on validation "
        "| OOD FPR actually incurred | False positives per 100,000 benign |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in _models(report):
        if model not in transfer:
            continue
        mean = transfer[model]["mean"]
        oracle = report["results"][model]["ood"]["mean"]["recall_at_fpr_1pct"]
        negatives = report["meta"]["sets"]["ood"]["n"] - report["meta"]["sets"]["ood"]["n_phishing"]
        lines.append(
            f"| {MODEL_LABELS[model]} | {pct(oracle)} | {pct(mean['ood_recall'])} "
            f"| **{pct(mean['ood_fpr'], 2)}** | {mean['ood_fp'] * 100_000 / negatives:,.0f} |"
        )
    return "\n".join(lines)


def unfiltered_table(report: dict) -> str:
    ablation = report["ablations"]["unfiltered_ood"]
    lines = [
        "| Model | Recall @ FPR 1 %, seen hosts removed | Recall @ FPR 1 %, seen hosts kept "
        "| Inflation (pts) | ROC-AUC inflation |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in _models(report):
        per_seed = ablation[model]["per_seed"]
        filtered = sum(entry["filtered"]["recall_at_fpr_1pct"] for entry in per_seed) / len(per_seed)
        unfiltered = sum(entry["unfiltered"]["recall_at_fpr_1pct"] for entry in per_seed) / len(per_seed)
        inflation = ablation[model]["mean_inflation"]
        lines.append(
            f"| {MODEL_LABELS[model]} | {pct(filtered)} | {pct(unfiltered)} "
            f"| {signed_points(inflation['recall_at_fpr_1pct'])} | {inflation['roc_auc']:+.4f} |"
        )
    return "\n".join(lines)


def openphish_table(report: dict) -> str:
    sets = report["meta"]["sets"]
    lines = [
        f"| Model | Detected, per host (n = {sets['openphish_hosts']['n']}) "
        f"| Detected, per URL row (n = {sets['openphish_rows']['n']}) |",
        "|---|---:|---:|",
    ]
    for model in _models(report):
        entry = report["openphish"][model]
        lines.append(
            f"| {MODEL_LABELS[model]} | {pct(entry['mean_rate_hosts'])} | {pct(entry['mean_rate_rows'])} |"
        )
    return "\n".join(lines)


def false_positive_table(report: dict) -> str:
    profile = report["false_positive_profile"]
    lines = [
        "| Benign hosts | Count | Contain a digit | Punycode (`xn--`) | Contain a hyphen | Mean length |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = [("All benign OOD hosts", profile["all_benign"])]
    rows += [(f"Flagged by {MODEL_LABELS[model]}", profile[model]) for model in _models(report)]
    for label, traits in rows:
        lines.append(
            f"| {label} | {traits['n']:,.0f} | {pct(traits['share_with_digit'])} "
            f"| {pct(traits['share_punycode'])} | {pct(traits['share_with_hyphen'])} "
            f"| {traits['mean_length']:.1f} |"
        )
    return "\n".join(lines)


def cost_table(report: dict) -> str:
    lines = [
        "| Model | Trainable parameters | Training rows | Device | Best epoch per seed "
        "| Validation ROC-AUC per seed | Training time per seed |",
        "|---|---:|---:|---|---|---|---:|",
    ]
    for model in _models(report):
        info = report["models"][model]
        if model.startswith("lightgbm"):
            parameters = f"{info['n_parameters']} trees × 31 leaves"
            epochs = "—"
        else:
            parameters = f"{info['n_parameters']:,}"
            epochs = ", ".join(str(epoch) for epoch in info["best_epochs"])
        scored = report["results"][model].get("validation")
        validation = (", ".join(f"{value:.4f}" for value in scored["roc_auc_per_seed"])
                      if scored else "in training data")
        minutes = info["train_minutes_mean"]
        duration = f"{minutes * 60:.0f} s" if minutes < 2 else f"{minutes:.1f} min"
        lines.append(
            f"| {MODEL_LABELS[model]} | {parameters} | {info['train_rows']:,} | {info['device']} "
            f"| {epochs} | {validation} | {duration} |"
        )
    return "\n".join(lines)


def reproduction_table(report: dict) -> str:
    reproduction = report["meta"]["reproduction"]
    lines = [
        "| Test set | Metric | Published (phishing-url-detector) | Reproduced here | Identical |",
        "|---|---|---:|---:|:---:|",
    ]
    for test_set in ("internal", "ood"):
        for metric, label in (("roc_auc", "ROC-AUC"), ("recall_at_fpr_1pct", "Recall @ FPR 1 %")):
            entry = reproduction[test_set][metric]
            lines.append(
                f"| {TEST_SET_LABELS[test_set]} | {label} | {entry['published']:.16g} "
                f"| {entry['reproduced']:.16g} | {'yes' if entry['match'] else '**no**'} |"
            )
    return "\n".join(lines)


def sets_table(report: dict) -> str:
    sets = report["meta"]["sets"]
    rows = (
        ("train", "Training (neural models, LightGBM control)"),
        ("val", "Validation (early stopping, threshold ablation)"),
        ("internal", "Internal hold-out (test)"),
        ("ood", "OOD + temporal (test)"),
    )
    lines = [
        "| Set | Hosts | Phishing | Registrable domains | Phishing in its 2 largest domains |",
        "|---|---:|---:|---:|---|",
    ]
    for key, label in rows:
        entry = sets[key]
        concentration = entry["phishing_concentration"]
        named = ", ".join(
            f"`{defang(domain)}` {count:,}"
            for domain, count in zip(
                concentration["top_domains"], concentration["top_domain_hosts"], strict=True
            )
        )
        lines.append(
            f"| {label} | {entry['n']:,} | {entry['n_phishing']:,} | {entry['registrable_domains']:,} "
            f"| {pct(concentration['share_of_phishing'])} ({named}) |"
        )
    return "\n".join(lines)


def headline(report: dict) -> str:
    """The one-paragraph answer, built from the numbers rather than written around them."""
    published = report["results"]["lightgbm_published"]["ood"]
    neural = [model for model in _models(report) if not model.startswith("lightgbm")]
    if not neural:
        return "_No neural model has been trained yet._"
    best = max(neural, key=lambda model: report["results"][model]["ood"]["mean"]["recall_at_fpr_1pct"])
    entry = report["results"][best]["ood"]
    delta = entry["delta_vs"]["lightgbm_published"]["recall_at_fpr_1pct"]
    control = entry["delta_vs"].get("lightgbm", {}).get("recall_at_fpr_1pct")
    lines = [
        f"On the out-of-distribution + temporal test ({report['meta']['sets']['ood']['n']:,} hosts), "
        f"the best small neural model, **{MODEL_LABELS[best]}**, reaches "
        f"**{pct(entry['mean']['recall_at_fpr_1pct'])} recall at a 1 % false-positive rate** "
        f"(95 % CI {interval_pct(entry['ci95']['recall_at_fpr_1pct'])}, mean of "
        f"{len(entry['per_seed'])} seeds), against **{pct(published['mean']['recall_at_fpr_1pct'])}** "
        f"for the published LightGBM: a paired difference of {signed_points(delta['point'])} points "
        f"[{signed_points(delta['ci95'][0])}, {signed_points(delta['ci95'][1])}]."
    ]
    if control is not None:
        lines.append(
            f" Against LightGBM trained on the same rows and averaged over the same seeds, the "
            f"difference is {signed_points(control['point'])} points "
            f"[{signed_points(control['ci95'][0])}, {signed_points(control['ci95'][1])}]."
        )
    return "".join(lines)


BLOCKS: dict[str, Callable[[dict], str]] = {
    "headline": headline,
    "results": results_table,
    "deltas": delta_table,
    "threshold": threshold_table,
    "unfiltered": unfiltered_table,
    "openphish": openphish_table,
    "false_positives": false_positive_table,
    "cost": cost_table,
    "reproduction": reproduction_table,
    "sets": sets_table,
}


def render(report: dict) -> dict[str, str]:
    return {name: build(report) for name, build in BLOCKS.items()}


def fill(document: str, blocks: dict[str, str]) -> str:
    """Replace the body of every ``BEGIN:name``/``END:name`` pair; fail on any missing pair."""
    for name, body in blocks.items():
        pattern = re.compile(
            rf"(<!-- BEGIN:{re.escape(name)} -->)\n.*?(<!-- END:{re.escape(name)} -->)", re.DOTALL
        )
        document, count = pattern.subn(
            lambda match, body=body: f"{match.group(1)}\n{body}\n{match.group(2)}", document
        )
        if count != 1:
            raise MissingMarkerError(f"expected exactly one '{name}' block in the document, found {count}")
    return document


def main(results_path: Path = RESULTS_PATH, readme: Path = README,
         model_card: Path | None = MODEL_CARD) -> None:
    blocks = render(json.loads(results_path.read_text()))
    readme.write_text(fill(readme.read_text(), blocks))
    if model_card is not None:
        card_blocks = {name: blocks[name] for name in MODEL_CARD_BLOCKS}
        model_card.write_text(fill(model_card.read_text(), card_blocks))
    print(f"Regenerated the README and model card blocks from {results_path.name}")


if __name__ == "__main__":
    main()
