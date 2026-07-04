"""Tests for the entropy-vs-trainability join/aggregate helpers (issue #430).

The analysis script is a pure join/stats pipeline over two committed JSON
artifacts. These tests cover the load-bearing helpers plus the end-to-end
run against the committed inputs (fast — no training, no Nash solves).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "entropy_vs_trainability.py"
)

spec = importlib.util.spec_from_file_location("entropy_vs_trainability", SCRIPT_PATH)
evt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evt)


def _entropy_cell(tag: str, beta: float, kappa: float, c: float, h_conds: list[float]):
    return {
        "cell_tag": tag,
        "beta": beta,
        "kappa": kappa,
        "c": c,
        "verdict": "asymmetric_only",
        "positions": [{"position": i, "h_cond": h} for i, h in enumerate(h_conds)],
    }


def _verdict_cell(tag: str, ne_mean, ne_std, homog_mean=0.1, homog_std=0.05):
    return {
        "cell_tag": tag,
        "ne_verdict": "asymmetric_only" if ne_mean is not None else "no_convergence",
        "n_seeds": 4,
        "gap_closed_ne_mean": ne_mean,
        "gap_closed_ne_std": ne_std,
        "gap_closed_homogeneous_mean": homog_mean,
        "gap_closed_homogeneous_std": homog_std,
    }


# ---------------------------------------------------------------------------
# entropy_aggregates
# ---------------------------------------------------------------------------


def test_entropy_aggregates_values():
    agg = evt.entropy_aggregates([1.0, 2.0, 3.0, 4.0])
    assert agg == {"mean": 2.5, "max": 4.0, "min": 1.0, "spread": 3.0}


def test_entropy_aggregates_empty_raises():
    with pytest.raises(ValueError):
        evt.entropy_aggregates([])


# ---------------------------------------------------------------------------
# join_cells
# ---------------------------------------------------------------------------


def test_join_cells_reports_verdict_only_cells():
    entropy_data = {"cells": [_entropy_cell("a", 0.1, 0.1, 1.0, [1.0, 2.0])]}
    verdict_data = {
        "cells": [
            _verdict_cell("a", 0.5, 0.1),
            _verdict_cell("b", None, None),  # no_convergence, no entropy
        ]
    }
    joined, report = evt.join_cells(entropy_data, verdict_data)
    assert len(joined) == 1
    assert report["n_joined"] == 1
    assert report["verdict_cells_without_entropy"] == ["b"]
    assert joined[0]["gap_closed_ne_mean"] == 0.5


def test_join_cells_fails_loudly_on_missing_verdict():
    entropy_data = {"cells": [_entropy_cell("orphan", 0.1, 0.1, 1.0, [1.0])]}
    verdict_data = {"cells": [_verdict_cell("other", 0.5, 0.1)]}
    with pytest.raises(ValueError, match="orphan"):
        evt.join_cells(entropy_data, verdict_data)


# ---------------------------------------------------------------------------
# spearman_table — null handling
# ---------------------------------------------------------------------------


def test_spearman_table_skips_null_targets():
    entropy_data = {
        "cells": [
            _entropy_cell("a", 0.1, 0.1, 1.0, [1.0, 2.0]),
            _entropy_cell("b", 0.5, 0.1, 1.0, [2.0, 3.0]),
            _entropy_cell("c", 0.9, 0.1, 1.0, [3.0, 4.0]),
        ]
    }
    verdict_data = {
        "cells": [
            _verdict_cell("a", 0.1, 0.1, homog_mean=0.1),
            _verdict_cell("b", None, None, homog_mean=0.2),
            _verdict_cell("c", 0.3, 0.1, homog_mean=0.3),
        ]
    }
    joined, _ = evt.join_cells(entropy_data, verdict_data)
    table = evt.spearman_table(joined, "gap_closed_ne_mean")
    assert table["n_cells"] == 2  # null cell excluded
    homog = evt.spearman_table(joined, "gap_closed_homogeneous_mean")
    assert homog["n_cells"] == 3  # all joined cells included


# ---------------------------------------------------------------------------
# beta_invariance_table
# ---------------------------------------------------------------------------


def test_beta_invariance_flags_identical_entropy():
    entropy_data = {
        "cells": [
            _entropy_cell("b0.10", 0.1, 0.1, 1.0, [1.0, 2.0]),
            _entropy_cell("b0.90", 0.9, 0.1, 1.0, [1.0, 2.0]),  # same profile
        ]
    }
    verdict_data = {
        "cells": [
            _verdict_cell("b0.10", -0.3, 0.1),
            _verdict_cell("b0.90", 0.0, 0.1),
        ]
    }
    joined, _ = evt.join_cells(entropy_data, verdict_data)
    table = evt.beta_invariance_table(joined)
    assert len(table) == 1
    row = table[0]
    assert row["entropy_identical_across_beta"] is True
    assert row["betas"] == [0.1, 0.9]
    assert row["gap_closed_ne_range"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# End-to-end on the committed artifacts
# ---------------------------------------------------------------------------


def test_end_to_end_reproduces_headline(tmp_path):
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = evt.main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0

    result = json.loads(out_json.read_text())
    ne = result["correlations"]["gap_closed_ne_mean"]
    assert ne["n_cells"] == 31
    # Dead predictor — re-verified after the #443 20-seed noise buy-down
    # updated 8 cells in recalibrated_verdict.json (rho moved 0.007 -> 0.109,
    # still nowhere near significance).
    assert abs(ne["aggregates"]["mean"]["spearman_rho"]) < 0.20
    assert ne["aggregates"]["mean"]["p_value"] > 0.05
    for agg in ("mean", "max", "min", "spread"):
        assert abs(ne["aggregates"][agg]["spearman_rho"]) <= 0.40
        assert ne["aggregates"][agg]["p_value"] > 0.05  # all insignificant

    # Every (kappa, c) group has byte-identical entropy across beta.
    assert all(r["entropy_identical_across_beta"] for r in result["beta_invariance"])

    # The 6 no_convergence cells are reported, not silently dropped.
    assert len(result["join_report"]["verdict_cells_without_entropy"]) == 6
