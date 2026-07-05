"""Tests for the 20-seed noise buy-down analysis helpers (issue #443).

The analysis script is a pure join/stats pipeline over committed per-seed
``summary.json`` artifacts. These tests cover the load-bearing helpers plus
the end-to-end run against the committed inputs (fast — no training, no
Nash solves).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "noise_buydown_precision.py"
)

spec = importlib.util.spec_from_file_location("noise_buydown_precision", SCRIPT_PATH)
nbp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nbp)


def _write_seed_summary(
    cell_dir: Path,
    seed: int,
    gap_ne: float | None,
    gap_homog: float,
) -> None:
    sdir = cell_dir / f"seed_{seed}"
    sdir.mkdir(parents=True)
    (sdir / "summary.json").write_text(
        json.dumps(
            {
                "seed": seed,
                "trailing5_team_mean": -50.0,
                "gap_closed_ne": gap_ne,
                "gap_closed_homogeneous": gap_homog,
            }
        )
    )


def _kstar_row(tag: str, ne_verdict: str, homog_mean: float) -> dict:
    return {
        "cell_tag": tag,
        "ne_verdict": ne_verdict,
        "gap_closed_homogeneous": {"n20": {"mean": homog_mean}},
    }


# ---------------------------------------------------------------------------
# verdict_for — the #360 ladder
# ---------------------------------------------------------------------------


def test_verdict_ladder_thresholds():
    assert nbp.verdict_for(-0.5) == "insufficient"
    assert nbp.verdict_for(0.19) == "insufficient"
    assert nbp.verdict_for(0.20) == "partial_lower"
    assert nbp.verdict_for(0.49) == "partial_upper"
    assert nbp.verdict_for(0.88) == "closed"


# ---------------------------------------------------------------------------
# bootstrap_ci / sample_stats
# ---------------------------------------------------------------------------


def test_bootstrap_ci_deterministic_and_contains_mean():
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    lo1, hi1 = nbp.bootstrap_ci(values, np.random.default_rng(443))
    lo2, hi2 = nbp.bootstrap_ci(values, np.random.default_rng(443))
    assert (lo1, hi1) == (lo2, hi2)  # seeded — reproducible
    mean = sum(values) / len(values)
    assert lo1 < mean < hi1
    assert lo1 >= min(values) and hi1 <= max(values)


def test_more_seeds_shrink_the_ci():
    # Same empirical distribution (values tiled 5x, identical pstdev) so the
    # only difference is n: the n=20 CI must be ~sqrt(5) narrower.
    small = [-1.5, -0.5, 0.5, 1.5]
    big = small * 5
    s4 = nbp.sample_stats(small, np.random.default_rng(443))
    s20 = nbp.sample_stats(big, np.random.default_rng(443))
    assert s4["std"] == pytest.approx(s20["std"])
    assert s20["ci95_half_width"] < s4["ci95_half_width"]
    ratio = s4["ci95_half_width"] / s20["ci95_half_width"]
    assert ratio == pytest.approx(5**0.5, rel=0.25)


# ---------------------------------------------------------------------------
# precision_rows — baseline split and null-NE handling
# ---------------------------------------------------------------------------


def test_precision_rows_splits_baseline_and_handles_null_ne(tmp_path):
    converged = tmp_path / "cell_conv"
    no_conv = tmp_path / "cell_noconv"
    for seed in range(42, 62):
        _write_seed_summary(converged, seed, gap_ne=0.01 * seed, gap_homog=0.5)
        _write_seed_summary(no_conv, seed, gap_ne=None, gap_homog=0.1)

    cells = [
        {
            "cell_tag": "conv",
            "beta": 0.5,
            "kappa": 0.1,
            "c": 1.0,
            "ne_verdict": "mixed",
        },
        {
            "cell_tag": "noconv",
            "beta": 0.5,
            "kappa": 0.1,
            "c": 0.5,
            "ne_verdict": "no_convergence",
        },
    ]
    rows = nbp.precision_rows(cells, tmp_path)

    conv = next(r for r in rows if r["cell_tag"] == "conv")
    assert conv["n_seeds"] == 20
    ne = conv["gap_closed_ne"]
    assert ne["n4"]["n"] == 4 and ne["n20"]["n"] == 20
    # n=4 baseline is exactly seeds 42-45.
    assert ne["n4"]["mean"] == pytest.approx(0.01 * (42 + 43 + 44 + 45) / 4)

    noconv = next(r for r in rows if r["cell_tag"] == "noconv")
    assert noconv["gap_closed_ne"] is None  # null NE baseline propagates
    homog = noconv["gap_closed_homogeneous"]
    assert homog["n20"]["mean"] == pytest.approx(0.1)
    assert homog["n20"]["ci95_half_width"] == pytest.approx(0.0)


def test_precision_rows_rejects_mixed_null_ne(tmp_path):
    cell = tmp_path / "cell_bad"
    _write_seed_summary(cell, 42, gap_ne=None, gap_homog=0.1)
    _write_seed_summary(cell, 43, gap_ne=0.2, gap_homog=0.1)
    cells = [
        {"cell_tag": "bad", "beta": 0.5, "kappa": 0.1, "c": 1.0, "ne_verdict": "mixed"}
    ]
    with pytest.raises(AssertionError, match="mixed null"):
        nbp.precision_rows(cells, tmp_path)


# ---------------------------------------------------------------------------
# kstar_class_test
# ---------------------------------------------------------------------------


def test_kstar_class_test_detects_clear_separation():
    rows = [
        _kstar_row("a", "no_convergence", -0.9),
        _kstar_row("b", "no_convergence", -0.8),
        _kstar_row("c", "mixed", 0.5),
        _kstar_row("d", "mixed", 0.6),
        _kstar_row("e", "symmetric_only", 0.7),
        _kstar_row("f", "symmetric_only", 0.8),
        _kstar_row("g", "asymmetric_only", 0.9),
        _kstar_row("h", "asymmetric_only", 1.0),
    ]
    result = nbp.kstar_class_test(rows)
    # Exact one-sided MW with groups of 2 and 6: minimum p = 1/28.
    assert result["mannwhitney_p_one_sided"] == pytest.approx(1 / 28)
    assert result["permutation_p_one_sided"] == pytest.approx(1 / 28)
    assert result["n_permutations"] == 28
    assert result["significant"] is True
    assert "significantly less" in result["verdict"]


def test_kstar_class_test_null_case_not_significant():
    rows = [
        _kstar_row("a", "no_convergence", 0.5),
        _kstar_row("b", "no_convergence", 0.1),
        _kstar_row("c", "mixed", 0.2),
        _kstar_row("d", "mixed", 0.6),
        _kstar_row("e", "symmetric_only", 0.05),
        _kstar_row("f", "symmetric_only", 0.4),
        _kstar_row("g", "asymmetric_only", 0.3),
        _kstar_row("h", "asymmetric_only", 0.55),
    ]
    result = nbp.kstar_class_test(rows)
    assert result["significant"] is False
    assert "no significant separation" in result["verdict"]


# ---------------------------------------------------------------------------
# load_buydown_cells — committed cells-source
# ---------------------------------------------------------------------------


def test_load_buydown_cells_committed_subset():
    cells = nbp.load_buydown_cells(nbp.DEFAULT_CELLS_SOURCE)
    assert len(cells) == 8
    tags = [c["cell_tag"] for c in cells]
    assert tags == sorted(tags)
    assert all(c["beta"] == 0.5 for c in cells)  # canonical beta (see #442)
    verdicts = [c["ne_verdict"] for c in cells]
    assert verdicts.count("no_convergence") == 2


# ---------------------------------------------------------------------------
# End-to-end on the committed artifacts
# ---------------------------------------------------------------------------


def test_end_to_end_reproduces_headline(tmp_path):
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = nbp.main(["--out-json", str(out_json), "--out-md", str(out_md)])
    assert rc == 0

    result = json.loads(out_json.read_text())
    assert result["cross_check"]["n_cells_checked"] == 8
    assert all(r["n_seeds"] == 20 for r in result["cells"])

    # The acceptance test: n=20 bought precision on every cell/column.
    for row in result["cells"]:
        for col in ("gap_closed_ne", "gap_closed_homogeneous"):
            if row[col] is None:
                assert row["ne_verdict"] == "no_convergence"
                continue
            assert row[col]["half_width_ratio_n4_over_n20"] > 1.2

    # No trainability-ladder verdict flips at n=20 (all 8 committed cells).
    assert all(not f["flipped"] for f in result["verdict_flips"])

    kstar = result["kstar_class_test"]
    assert kstar["column"] == "gap_closed_homogeneous"
    assert 0.0 < kstar["mannwhitney_p_one_sided"] <= 1.0
    assert kstar["n_permutations"] == 28

    md = out_md.read_text()
    assert "CI half-width comparison" in md
    assert "k* class-comparison test" in md
