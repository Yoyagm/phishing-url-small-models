"""Figures for the README, drawn from the same scores the results table is computed from.

Styling is the original project's (:mod:`phishurl.viz`); model colours follow a
fixed categorical order validated for colour-vision deficiency, and every line
also differs in dash pattern and carries a legend entry, so identity never rests
on hue alone. The table in the README is the accessible view of the same data.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from phishsmall.config import FIGURES, MODEL_LABELS, TARGET_FPR
from phishurl import viz

# Slots of the validated categorical palette, fixed per model (never re-cycled).
STYLE = {
    "lightgbm_published": ("#2a78d6", "-"),
    "lightgbm_reseeded": ("#008300", (0, (1, 1))),
    "lightgbm": ("#6250d6", (0, (1, 1))),
    "char_cnn": ("#eb6834", (0, (6, 2))),
    "bigru": ("#1baf7a", (0, (6, 2, 1.5, 2))),
    "distilbert": ("#eda100", (0, (2, 1.5))),
}
CURVE_MODELS = ("lightgbm_published", "char_cnn", "bigru", "distilbert")
SET_TITLES = {"internal": "Internal hold-out (grouped)", "ood": "OOD + temporal"}


def representative_seed(report: dict, model: str) -> int:
    """The seed whose OOD recall@1 % FPR is the median of the model's seeds."""
    per_seed = report["results"][model]["ood"]["per_seed"]
    ranked = sorted(per_seed, key=lambda entry: entry["recall_at_fpr_1pct"])
    return ranked[len(ranked) // 2]["seed"]


def _save(fig, name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    return path


def _shared_legend(fig, axis) -> None:
    """One legend under both panels: the same models appear in each, and none of the plot is covered.

    Per-model AUC/AP values live in the results table, which is the accessible view.
    """
    handles, labels = axis.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8,
               bbox_to_anchor=(0.5, -0.08), handlelength=3)


def roc_figure(report: dict, curves) -> Path:
    """ROC with a logarithmic FPR axis: the 1 % operating point sits mid-plot, not at the edge."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, test_set in zip(axes, ("internal", "ood"), strict=True):
        frame = curves.frames[test_set]
        y = frame["y"].to_numpy()
        for model in CURVE_MODELS:
            if model not in curves.models:
                continue
            seed = representative_seed(report, model)
            scores = curves.lookup(model, seed, frame["host"])
            fpr, tpr, _ = roc_curve(y, scores)
            colour, dash = STYLE[model]
            axis.plot(np.clip(fpr, 1e-5, 1), tpr, color=colour, linestyle=dash, label=MODEL_LABELS[model])
        axis.axvline(TARGET_FPR, color=viz.MUTED, linewidth=1, linestyle=(0, (2, 2)))
        axis.text(TARGET_FPR * 1.12, 0.03, "FPR 1 %", color=viz.MUTED, fontsize=7.5)
        axis.set_xscale("log")
        axis.set_xlim(1e-4, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("False positive rate (log scale)")
        axis.set_title(SET_TITLES[test_set])
        viz.strip_chrome(axis)
    axes[0].set_ylabel("True positive rate (recall)")
    _shared_legend(fig, axes[0])
    return _save(fig, "roc_curves")


def pr_figure(report: dict, curves) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), sharey=True)
    for axis, test_set in zip(axes, ("internal", "ood"), strict=True):
        frame = curves.frames[test_set]
        y = frame["y"].to_numpy()
        for model in CURVE_MODELS:
            if model not in curves.models:
                continue
            seed = representative_seed(report, model)
            scores = curves.lookup(model, seed, frame["host"])
            precision, recall, _ = precision_recall_curve(y, scores)
            colour, dash = STYLE[model]
            axis.plot(recall, precision, color=colour, linestyle=dash, label=MODEL_LABELS[model])
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel("Recall")
        axis.set_title(SET_TITLES[test_set])
        viz.strip_chrome(axis)
    axes[0].set_ylabel("Precision")
    _shared_legend(fig, axes[0])
    return _save(fig, "pr_curves")


def paired_delta_figure(report: dict) -> Path:
    """Each model's recall at 1 % FPR minus the published LightGBM's, with a paired 95 % CI.

    Paired, not marginal: both models are scored on the same domain-level resample
    each time, so the interval reflects the difference itself. Two marginal
    intervals that overlap can still hide a clear, consistent gap.
    """
    models = [model for model in STYLE if model in report["results"] and model != "lightgbm_published"]
    # One shared x-axis: the two test sets are compared on the same scale, on purpose.
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 0.5 * len(models) + 1.6), sharey=True, sharex=True)
    positions = np.arange(len(models))[::-1]
    for axis, test_set in zip(axes, ("internal", "ood"), strict=True):
        axis.axvline(0, color=viz.INK_SECONDARY, linewidth=1, zorder=0)
        for position, model in zip(positions, models, strict=True):
            delta = report["results"][model][test_set]["delta_vs"]["lightgbm_published"]["recall_at_fpr_1pct"]
            low, high = (100 * bound for bound in delta["ci95"])
            point = 100 * delta["point"]
            colour, _ = STYLE[model]
            axis.plot([low, high], [position, position], color=colour, linewidth=2, solid_capstyle="round")
            axis.plot(point, position, "o", color=colour, markersize=8,
                      markeredgecolor=viz.SURFACE, markeredgewidth=2)
            label = "0.0" if round(point, 1) == 0 else f"{point:+.1f}"
            axis.text(high + 0.6, position, label, va="center", fontsize=8, color=viz.INK_SECONDARY)
        axis.set_title(SET_TITLES[test_set])
        viz.strip_chrome(axis)
        axis.yaxis.grid(False)
        axis.xaxis.grid(True, color=viz.GRID, linewidth=0.7)
    axes[0].set_yticks(positions, [MODEL_LABELS[model] for model in models])
    axes[0].tick_params(axis="y", length=0)
    fig.supxlabel("Δ recall at FPR 1 % vs the published LightGBM (points; paired 95 % CI; right is better)",
                  fontsize=9, color=viz.INK_SECONDARY)
    return _save(fig, "paired_delta_recall")


def render(report: dict, curves) -> list[Path]:
    viz.apply_style()
    return [roc_figure(report, curves), pr_figure(report, curves), paired_delta_figure(report)]
