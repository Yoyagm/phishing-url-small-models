"""Paths, seeds and the published numbers this study is anchored to.

Everything that decides *what is being compared* lives here, so the comparison
cannot drift between the training scripts and the evaluation.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PROCESSED = DATA / "processed"
ARTIFACTS = ROOT / "artifacts"
SCORES = ARTIFACTS / "scores"
CHECKPOINTS = ARTIFACTS / "checkpoints"
LOGS = ARTIFACTS / "logs"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures"
RESULTS_PATH = REPORTS / "results.json"

# The original repository splits with GroupShuffleSplit(test_size=0.2, random_state=42).
# Reusing both values is what makes the hold-out the *same* hold-out.
SPLIT_SEED = 42
TEST_SIZE = 0.2

# Early stopping needs data the test sets never see. It is carved out of the
# original training portion, grouped by registrable domain like everything else.
VAL_SIZE = 0.1

TRAIN_SEEDS = (0, 1, 2)
PUBLISHED_BASELINE_SEED = 42

LIGHTGBM_MODELS = ("lightgbm_published", "lightgbm_reseeded", "lightgbm")
# Validation hosts are in these models' training data, so nothing may be
# calibrated or early-stopped on validation for them.
TRAINED_ON_VALIDATION = ("lightgbm_published", "lightgbm_reseeded")
NEURAL_MODELS = ("char_cnn", "bigru", "distilbert")
MODEL_NAMES = LIGHTGBM_MODELS + NEURAL_MODELS
MODEL_LABELS = {
    "lightgbm_published": "LightGBM, published (reproduced)",
    "lightgbm_reseeded": "LightGBM, published rows, seeds 0-2",
    "lightgbm": "LightGBM, neural models' rows",
    "char_cnn": "char-CNN (from scratch)",
    "bigru": "BiGRU (from scratch)",
    "distilbert": "DistilBERT (fine-tuned)",
}

TARGET_FPR = 0.01
STRICT_FPR = 0.001
BOOTSTRAP_REPS = 1000
BOOTSTRAP_SEED = 20260814

# reports/metrics.json of phishing-url-detector at commit 4283da1. The reproduced
# baseline must match these, or the comparison is not against the published model.
PUBLISHED = {
    "internal": {"roc_auc": 0.9186758425833806, "recall_at_fpr_1pct": 0.725922131147541},
    "ood": {"roc_auc": 0.9372024471270224, "recall_at_fpr_1pct": 0.705472034654096},
}
# The original computed this over 299 OpenPhish URL rows (260 distinct hosts).
PUBLISHED_OPENPHISH_RATE_ROWS = 0.5852842809364549
REPRODUCTION_TOLERANCE = 1e-9
