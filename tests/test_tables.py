"""The README's numbers are generated from ``reports/results.json`` and nowhere else."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from phishsmall import tables
from phishsmall.config import RESULTS_PATH, ROOT

FIXTURE = Path(__file__).parent / "fixtures" / "results_minimal.json"


@pytest.fixture
def report() -> dict:
    return json.loads(FIXTURE.read_text())


def _template() -> str:
    return "\n\n".join(
        f"## {name}\n<!-- BEGIN:{name} -->\nstale text 99.9 %\n<!-- END:{name} -->" for name in tables.BLOCKS
    )


def test_results_table_reads_every_number_from_the_json(report) -> None:
    table = tables.results_table(report)
    assert "| LightGBM, published (reproduced) | 1 (42) | OOD + temporal | 0.9372 [0.927, 0.947] " in table
    assert "**70.5 %** [65.5, 74.5]" in table
    assert "| char-CNN (from scratch) | 3 (0, 1, 2) |" in table
    assert "| 66.0 / 63.0 / 66.0 |" in table
    assert "| 705 | 99 |" in table


def test_delta_table_reports_paired_intervals(report) -> None:
    table = tables.delta_table(report)
    assert "| char-CNN (from scratch) | OOD + temporal | vs published LightGBM | -5.5 [-8.0, -3.0] |" in table


def test_headline_names_the_best_neural_model_and_its_gap(report) -> None:
    text = tables.headline(report)
    assert "**char-CNN (from scratch)**" in text
    assert "**65.0 % recall at a 1 % false-positive rate**" in text
    assert "-5.5 points [-8.0, -3.0]" in text


def test_main_rewrites_only_the_marked_blocks(tmp_path, report) -> None:
    results = tmp_path / "results.json"
    shutil.copy(FIXTURE, results)
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nHand-written prose stays.\n\n" + _template() + "\n")

    tables.main(results_path=results, readme=readme, model_card=None)

    text = readme.read_text()
    assert "Hand-written prose stays." in text
    assert "stale text" not in text
    assert "70.5 %" in text


def test_filling_is_idempotent(report) -> None:
    once = tables.fill(_template(), tables.render(report))
    assert tables.fill(once, tables.render(report)) == once


def test_a_missing_marker_is_an_error(report) -> None:
    with pytest.raises(tables.MissingMarkerError, match="'cost'"):
        tables.fill(_template().replace("BEGIN:cost", "BEGIN:costs"), tables.render(report))


def test_the_committed_readme_and_model_card_have_every_block() -> None:
    tables.fill((ROOT / "README.md").read_text(), {name: "x" for name in tables.BLOCKS})
    tables.fill(tables.MODEL_CARD.read_text(), {name: "x" for name in tables.MODEL_CARD_BLOCKS})


@pytest.mark.skipif(not RESULTS_PATH.exists(), reason="no results.json yet")
def test_the_committed_readme_is_in_sync_with_the_committed_results() -> None:
    """If this fails, someone edited a number by hand or forgot `make tables`."""
    blocks = tables.render(json.loads(RESULTS_PATH.read_text()))
    readme = (ROOT / "README.md").read_text()
    assert tables.fill(readme, blocks) == readme
    card = tables.MODEL_CARD.read_text()
    assert tables.fill(card, {name: blocks[name] for name in tables.MODEL_CARD_BLOCKS}) == card
