"""The LightGBM baseline is deterministic on a fixed mini-fixture.

Runs in a subprocess on purpose. PyTorch (imported by other test modules) and
LightGBM ship different OpenMP runtimes; on macOS one process holding both
segfaults or deadlocks. The subprocess keeps the suite honest about that rule
instead of passing only when the test order happens to be lucky.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from phishsmall.config import ROOT

# Recorded from a reference run with LightGBM 4.7.0 (pinned in pyproject.toml).
PINNED_AUC = 0.99995
PINNED_SCORE_SUM = 99.93919272177467

# Built from fixed rules, so the fixture is the same on every machine: benign hosts
# are short brand-like names, phishing hosts stack keywords and a cheap suffix, and
# a fifth of each class is deliberately drawn from the other class's pattern.
_SCRIPT = r"""
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from phishsmall import baseline

words = ["shop", "news", "mail", "cloud", "photo", "music", "travel", "food", "bank", "school"]
suffixes = ["xyz", "top", "online", "site", "info"]

def host(i, phishing):
    word = words[i % len(words)]
    if phishing == (i % 5 != 0):
        return f"secure-{word}{i}.login-verify.{word}.com.acct{i % 7}.{suffixes[i % 5]}"
    return f"{word}{i % 13}.com"

def frame(start, stop):
    hosts, labels = [], []
    for i in range(start, stop):
        for phishing in (0, 1):
            hosts.append(host(i, phishing))
            labels.append(phishing)
    return hosts, np.array(labels)

train_hosts, train_y = frame(0, 300)
probe_hosts, probe_y = frame(300, 400)
first = baseline.score(baseline.fit(train_hosts, train_y, seed=0), probe_hosts)
second = baseline.score(baseline.fit(train_hosts, train_y, seed=0), probe_hosts)
print(json.dumps({
    "identical": bool(np.array_equal(first, second)),
    "auc": float(roc_auc_score(probe_y, first)),
    "score_sum": float(first.sum()),
}))
"""


@pytest.fixture(scope="module")
def outcome() -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT], capture_output=True, text=True, env=env, cwd=ROOT, timeout=120
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_two_fits_with_the_same_seed_give_bit_identical_scores(outcome) -> None:
    assert outcome["identical"]


def test_the_fixture_baseline_gives_the_pinned_values(outcome) -> None:
    assert outcome["auc"] == pytest.approx(PINNED_AUC, abs=1e-12)
    assert outcome["score_sum"] == pytest.approx(PINNED_SCORE_SUM, abs=1e-9)

