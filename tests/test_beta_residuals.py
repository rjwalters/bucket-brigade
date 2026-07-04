"""Tests for the cross-β residual analysis helpers (issue #442).

The analysis script is a join/stats pipeline over ``results.json`` plus a
cheap CRN re-evaluation of committed NE profiles. These tests cover the
pure helpers (no Rust) and an end-to-end run against the committed
artifacts with a reduced simulation budget (fast).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "beta_residuals.py"

spec = importlib.util.spec_from_file_location("beta_residuals", SCRIPT_PATH)
br = importlib.util.module_from_spec(spec)
spec.loader.exec_module(br)


def _cell(beta, kappa, c, payoff, converged=3, verdict="symmetric_only"):
    return {
        "beta": beta,
        "kappa": kappa,
        "c": c,
        "tag": br.cell_tag(beta, kappa, c),
        "best_team_payoff": payoff,
        "converged": converged,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# cell_tag
# ---------------------------------------------------------------------------


def test_cell_tag_matches_driver_convention():
    assert br.cell_tag(0.1, 0.9, 0.5) == "b0.10_k0.90_c0.50"


# ---------------------------------------------------------------------------
# group_columns
# ---------------------------------------------------------------------------


def test_group_columns_bit_identical_column():
    cells = [
        _cell(0.1, 0.5, 1.0, -700.0),
        _cell(0.5, 0.5, 1.0, -700.0),
        _cell(0.9, 0.5, 1.0, -700.0),
    ]
    cols = br.group_columns(cells)
    assert len(cols) == 1
    col = cols[0]
    assert col["bit_identical"] is True
    assert col["payoff_residual"] == 0.0
    assert col["verdict_consistent"] is True
    assert col["convergence_consistent"] is True
    assert col["betas"] == [0.1, 0.5, 0.9]  # sorted by beta


def test_group_columns_residual_and_verdict_flip():
    cells = [
        _cell(0.9, 0.9, 0.5, 72.0095, converged=14, verdict="asymmetric_only"),
        _cell(0.1, 0.9, 0.5, 80.915, converged=17, verdict="mixed"),
        _cell(0.5, 0.9, 0.5, 72.0095, converged=14, verdict="asymmetric_only"),
    ]
    cols = br.group_columns(cells)
    col = cols[0]
    assert col["bit_identical"] is False
    assert col["payoff_residual"] == pytest.approx(80.915 - 72.0095)
    assert col["verdict_consistent"] is False
    assert col["convergence_consistent"] is False


def test_group_columns_sorts_by_kappa_then_c():
    cells = [
        _cell(0.1, 0.9, 0.5, 1.0),
        _cell(0.5, 0.9, 0.5, 1.0),
        _cell(0.1, 0.1, 2.0, 2.0),
        _cell(0.5, 0.1, 2.0, 2.0),
    ]
    cols = br.group_columns(cells)
    assert [(c["kappa"], c["c"]) for c in cols] == [(0.1, 2.0), (0.9, 0.5)]


def test_group_columns_single_beta_raises():
    with pytest.raises(ValueError, match="at least 2"):
        br.group_columns([_cell(0.5, 0.5, 0.5, -1.0)])


def test_group_columns_null_payoff_raises():
    cells = [_cell(0.1, 0.5, 0.5, None), _cell(0.5, 0.5, 0.5, -1.0)]
    with pytest.raises(ValueError, match="null"):
        br.group_columns(cells)


# ---------------------------------------------------------------------------
# genome_max_delta / dedupe_profiles
# ---------------------------------------------------------------------------


def test_genome_max_delta_values():
    a = [[0.0, 1.0], [0.5, 0.5]]
    b = [[0.0, 1.0], [0.5, 0.75]]
    assert br.genome_max_delta(a, b) == pytest.approx(0.25)
    assert br.genome_max_delta(a, a) == 0.0


def test_genome_max_delta_shape_mismatch_raises():
    with pytest.raises(ValueError):
        br.genome_max_delta([[0.0]], [[0.0], [1.0]])
    with pytest.raises(ValueError):
        br.genome_max_delta([[0.0]], [[0.0, 1.0]])


def _payload(label, payoff, genomes, symmetric=False):
    return {
        "profile_label": label,
        "team_payoff": payoff,
        "symmetric_profile": symmetric,
        "positions": [{"position": i, "genome": g} for i, g in enumerate(genomes)],
    }


def test_dedupe_profiles_collapses_identical_genomes():
    hero = [[1.0, 0.9]] * 2
    profiles = {
        "0.50": _payload("hero | hero", 72.0, hero, symmetric=True),
        "0.90": _payload("hero | hero", 72.0, hero, symmetric=True),
    }
    unique = br.dedupe_profiles(profiles)
    assert len(unique) == 1
    assert unique[0]["from_betas"] == ["0.50", "0.90"]


def test_dedupe_profiles_keeps_distinct_genomes():
    profiles = {
        "0.10": _payload("ff | hero", 80.9, [[0.5, 0.5], [1.0, 1.0]]),
        "0.50": _payload("hero | ff", 72.0, [[1.0, 1.0], [0.5, 0.5]]),
    }
    unique = br.dedupe_profiles(profiles)
    assert len(unique) == 2
    assert unique[0]["from_betas"] == ["0.10"]
    assert unique[1]["from_betas"] == ["0.50"]


# ---------------------------------------------------------------------------
# paired_stats
# ---------------------------------------------------------------------------


def test_paired_stats_known_values():
    stats = br.paired_stats([1.0, 2.0, 3.0, 4.0])
    assert stats["n"] == 4
    assert stats["mean"] == pytest.approx(2.5)
    # sample std of [1..4] is ~1.2910; se = std / 2
    assert stats["se"] == pytest.approx(0.6455, rel=1e-3)
    assert stats["t"] == pytest.approx(2.5 / 0.6455, rel=1e-3)


def test_paired_stats_too_few_samples_raises():
    with pytest.raises(ValueError):
        br.paired_stats([1.0])


# ---------------------------------------------------------------------------
# load_column_profiles — missing files reported, not fatal
# ---------------------------------------------------------------------------


def test_load_column_profiles_reports_missing(tmp_path):
    column = {
        "betas": [0.1, 0.5],
        "tags": ["b0.10_k0.10_c0.50", "b0.50_k0.10_c0.50"],
    }
    payload = _payload("hero | hero", -1.0, [[1.0], [1.0]])
    (tmp_path / "b0.50_k0.10_c0.50.json").write_text(json.dumps(payload))
    profiles, missing = br.load_column_profiles(column, tmp_path)
    assert list(profiles) == ["0.50"]
    assert missing == ["b0.10_k0.10_c0.50"]


# ---------------------------------------------------------------------------
# End-to-end on the committed artifacts (reduced sim budget, Rust engine)
# ---------------------------------------------------------------------------


def test_end_to_end_reproduces_headline(tmp_path):
    pytest.importorskip("bucket_brigade_core")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = br.main(
        [
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--n-sims",
            "400",
            "--inertness-sims",
            "100",
        ]
    )
    assert rc == 0

    result = json.loads(out_json.read_text())
    assert result["n_columns"] == 13
    assert result["n_bit_identical"] == 11
    residuals = {
        (r["kappa"], r["c"]): r["payoff_residual"] for r in result["residual_columns"]
    }
    assert set(residuals) == {(0.5, 0.5), (0.9, 0.5)}
    assert residuals[(0.9, 0.5)] == pytest.approx(80.915 - 72.0095)

    # Both residual columns have committed profiles → both CRN-evaluated.
    assert len(result["crn_evaluations"]) == 2
    k09 = next(e for e in result["crn_evaluations"] if e["kappa"] == 0.9)
    assert len(k09["profiles"]) == 2  # structurally different equilibria
    k05 = next(e for e in result["crn_evaluations"] if e["kappa"] == 0.5)
    # κ=0.5 committed profiles differ only by optimizer endpoint jitter.
    assert k05["pairwise"][0]["genome_max_delta"] < 1e-5

    # β-inertness must hold bit-exactly for every committed profile.
    assert result["inertness"]["all_identical"] is True

    md = out_md.read_text()
    assert "β-inertness" in md
    assert "#445" in md and "#429" in md
    # Profile labels contain '|' and must be escaped inside markdown tables.
    assert "\\|" in md
