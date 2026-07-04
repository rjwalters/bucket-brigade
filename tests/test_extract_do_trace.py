"""Tests for ``experiments/nash/scripts/extract_do_trace.py`` (issue #445).

Covers:

1. Parsing a synthetic verbose ``compute_nash.py`` log block (field
   extraction, support composition string, incomplete-trailing-block drop).
2. Parsing the committed rest_trap seeded-DO run log end-to-end: 50
   iterations, known min/final improvement values (the non-convergence
   evidence committed for #445).
3. CSV round-trip and markdown rendering.

The script is imported by path because ``experiments/nash/scripts`` is not a
Python package (same pattern as ``tests/test_compute_nash_seed_profiles.py``).
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "experiments" / "nash" / "scripts" / "extract_do_trace.py"
REST_TRAP_LOG = (
    REPO_ROOT
    / "experiments"
    / "nash"
    / "rest_trap_seeded_do"
    / "nash-resttrap-seeded.log"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("extract_do_trace", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def mod():
    return _load_module()


SYNTHETIC_LOG = """\
Starting Double Oracle with 9 strategies

--- Iteration 1 ---
Computing payoff matrix (9x9)...
Solving restricted game...
Equilibrium support size: 3
  Strategy 3: 0.290
  Strategy 5: 0.582
  Strategy 6: 0.128
Equilibrium payoff: 1754.88
Computing best response...
Best response payoff: 1881.94
Improvement: 127.0690
Added best response to pool (new size: 10)

--- Iteration 2 ---
Computing payoff matrix (10x10)...
Solving restricted game...
Equilibrium support size: 1
  Strategy 0: 1.000
Equilibrium payoff: -12.50
Computing best response...
Best response payoff: -12.49
Improvement: 0.0050
Added best response to pool (new size: 11)

--- Iteration 3 ---
Computing payoff matrix (11x11)...
Solving restricted game...
Equilibrium support size: 2
  Strategy 1: 0.500
  Strategy 2: 0.500
Equilibrium payoff: 100.00
Computing best response...
"""


class TestParseSyntheticLog:
    def test_two_complete_iterations_parsed(self, mod):
        rows = mod.parse_do_log(SYNTHETIC_LOG)
        assert [r["iteration"] for r in rows] == [1, 2]

    def test_field_extraction(self, mod):
        row = mod.parse_do_log(SYNTHETIC_LOG)[0]
        assert row["pool_size"] == 9
        assert row["support_size"] == 3
        assert row["eq_payoff"] == pytest.approx(1754.88)
        assert row["br_payoff"] == pytest.approx(1881.94)
        assert row["improvement"] == pytest.approx(127.0690)
        assert row["support"] == "3:0.290|5:0.582|6:0.128"

    def test_negative_payoffs_parsed(self, mod):
        row = mod.parse_do_log(SYNTHETIC_LOG)[1]
        assert row["eq_payoff"] == pytest.approx(-12.50)
        assert row["br_payoff"] == pytest.approx(-12.49)

    def test_incomplete_trailing_block_dropped(self, mod):
        rows = mod.parse_do_log(SYNTHETIC_LOG)
        assert all(r["iteration"] != 3 for r in rows)

    def test_empty_log_gives_no_rows(self, mod):
        assert mod.parse_do_log("no iterations here\n") == []


class TestRestTrapSeededLog:
    """End-to-end parse of the committed #445 run log (non-convergence evidence)."""

    @pytest.fixture(scope="class")
    def rows(self, mod):
        return mod.parse_do_log(REST_TRAP_LOG.read_text())

    def test_fifty_iterations(self, rows):
        assert [r["iteration"] for r in rows] == list(range(1, 51))

    def test_pool_grows_from_9_to_58(self, rows):
        assert rows[0]["pool_size"] == 9
        assert rows[-1]["pool_size"] == 58

    def test_min_improvement_is_iter_44(self, rows):
        best = min(rows, key=lambda r: r["improvement"])
        assert best["iteration"] == 44
        assert best["improvement"] == pytest.approx(11.7885)

    def test_improvement_never_meets_epsilon(self, rows):
        # The run's convergence threshold was epsilon=0.01 (absolute).
        assert all(r["improvement"] > 0.01 for r in rows)

    def test_final_iteration_values(self, rows):
        last = rows[-1]
        assert last["eq_payoff"] == pytest.approx(2303.53)
        assert last["improvement"] == pytest.approx(113.4554)


class TestOutputs:
    def test_csv_round_trip(self, mod, tmp_path):
        rows = mod.parse_do_log(SYNTHETIC_LOG)
        out = tmp_path / "trace.csv"
        mod.write_csv(rows, out)
        with open(out) as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == 2
        assert read_back[0]["iteration"] == "1"
        assert read_back[0]["support"] == "3:0.290|5:0.582|6:0.128"
        assert float(read_back[1]["improvement"]) == pytest.approx(0.0050)

    def test_markdown_table(self, mod):
        rows = mod.parse_do_log(SYNTHETIC_LOG)
        table = mod.markdown_table(rows)
        lines = table.splitlines()
        assert lines[0].startswith("| Iter |")
        assert len(lines) == 2 + len(rows)
        assert "| 1 | 9 | 3 | 1754.88 | 1881.94 | 127.07 |" in table
