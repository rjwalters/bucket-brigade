"""Tests for the k*-vs-trainability join/stats helpers (issue #430 Task 2).

The analysis script is a pure join/stats pipeline over two committed
artifacts (the thrust#269 k* phase diagram and the recalibrated PPO
verdicts). These tests cover the load-bearing helpers plus the end-to-end
run against the committed inputs (fast — no training, no Nash solves).
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "kstar_vs_trainability.py"
)

spec = importlib.util.spec_from_file_location("kstar_vs_trainability", SCRIPT_PATH)
kvt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kvt)


def _kstar_cell(beta: float, kappa: float, c: float, k_star: int) -> dict:
    return {"beta": beta, "kappa": kappa, "c": c, "k_star": k_star}


def _verdict_cell(
    tag: str,
    ne_verdict: str,
    n_seeds: int,
    gap_ne: float | None,
    gap_hom: float,
) -> dict:
    return {
        "cell_tag": tag,
        "ne_verdict": ne_verdict,
        "n_seeds": n_seeds,
        "gap_closed_ne_mean": gap_ne,
        "gap_closed_homogeneous_mean": gap_hom,
    }


# ---------------------------------------------------------------------------
# parse_cell_tag
# ---------------------------------------------------------------------------


def test_parse_cell_tag_roundtrip():
    assert kvt.parse_cell_tag("b0.10_k0.90_c2.00") == (0.1, 0.9, 2.0)


def test_parse_cell_tag_rejects_garbage():
    with pytest.raises(ValueError, match="Unparseable"):
        kvt.parse_cell_tag("x0.10_k0.90_c2.00")
    with pytest.raises(ValueError, match="Unparseable"):
        kvt.parse_cell_tag("not-a-tag")


# ---------------------------------------------------------------------------
# kstar_columns — β-dedup and the invariance guard
# ---------------------------------------------------------------------------


def test_kstar_columns_dedupes_beta_replicas():
    data = {
        "cells": [_kstar_cell(b, 0.1, 0.5, 4) for b in (0.1, 0.3, 0.5, 0.7, 0.9)]
        + [_kstar_cell(b, 0.9, 0.5, 1) for b in (0.1, 0.5)]
    }
    columns, report = kvt.kstar_columns(data)
    assert columns == {(0.1, 0.5): 4, (0.9, 0.5): 1}
    assert report["n_artifact_cells"] == 7
    assert report["n_effective_columns"] == 2
    assert report["kstar_beta_invariant"] is True
    assert report["kstar_is_function_of_kappa_only"] is True


def test_kstar_columns_fails_loudly_on_beta_variation():
    data = {
        "cells": [
            _kstar_cell(0.1, 0.5, 1.0, 2),
            _kstar_cell(0.5, 0.5, 1.0, 3),  # k* differs across β — invalid
        ]
    }
    with pytest.raises(ValueError, match="NOT β-invariant"):
        kvt.kstar_columns(data)


def test_kstar_columns_detects_kappa_dependence_on_c():
    data = {
        "cells": [
            _kstar_cell(0.1, 0.5, 0.5, 1),
            _kstar_cell(0.1, 0.5, 2.0, 2),  # same κ, different k* by c
        ]
    }
    _, report = kvt.kstar_columns(data)
    assert report["kstar_is_function_of_kappa_only"] is False


# ---------------------------------------------------------------------------
# join_columns — pooling and coverage guards
# ---------------------------------------------------------------------------


def _kstar_map():
    return {(0.1, 0.5): 4, (0.9, 0.5): 1}


def test_join_columns_pools_with_n_seeds_weights():
    verdict = {
        "cells": [
            _verdict_cell("b0.10_k0.90_c0.50", "mixed", 4, 0.10, 0.05),
            _verdict_cell("b0.50_k0.90_c0.50", "mixed", 20, 0.40, 0.10),
        ]
    }
    columns, report = kvt.join_columns(_kstar_map(), verdict)
    assert len(columns) == 1
    col = columns[0]
    assert col["k_star"] == 1
    # pooled = (0.10*4 + 0.40*20) / 24 = 0.35
    assert col["gap_closed_ne_pooled"] == pytest.approx(0.35)
    assert col["n_seeds_ne"] == 24
    assert col["n_beta_replicas"] == 2
    # unswept k* column is reported, not an error
    assert report["kstar_columns_without_ppo_sweep"] == [
        {"kappa": 0.1, "c": 0.5, "k_star": 4}
    ]


def test_join_columns_all_null_ne_gap_pools_to_none():
    verdict = {
        "cells": [
            _verdict_cell("b0.10_k0.10_c0.50", "no_convergence", 4, None, -0.01),
            _verdict_cell("b0.50_k0.10_c0.50", "no_convergence", 20, None, -0.02),
        ]
    }
    columns, _ = kvt.join_columns(_kstar_map(), verdict)
    assert columns[0]["gap_closed_ne_pooled"] is None
    assert columns[0]["gap_closed_homogeneous_pooled"] == pytest.approx(
        (-0.01 * 4 - 0.02 * 20) / 24
    )


def test_join_columns_fails_loudly_on_unknown_column():
    verdict = {"cells": [_verdict_cell("b0.10_k0.30_c1.00", "mixed", 4, 0.1, 0.05)]}
    with pytest.raises(ValueError, match="out of sync"):
        kvt.join_columns(_kstar_map(), verdict)


# ---------------------------------------------------------------------------
# exact_permutation_test
# ---------------------------------------------------------------------------


def test_exact_permutation_perfect_separation_hits_floor():
    values = [1.0, 2.0, 3.0, 10.0, 11.0]
    group1 = [False, False, False, True, True]
    out = kvt.exact_permutation_test(values, group1, alternative="greater")
    assert out["n_permutations"] == math.comb(5, 2)
    assert out["p_one_sided"] == pytest.approx(1.0 / 10.0)
    assert out["observed_mean_diff"] == pytest.approx(10.5 - 2.0)


def test_exact_permutation_less_alternative_is_mirror():
    values = [1.0, 2.0, 3.0, 10.0, 11.0]
    group1 = [True, True, False, False, False]
    out = kvt.exact_permutation_test(values, group1, alternative="less")
    assert out["p_one_sided"] == pytest.approx(1.0 / 10.0)
    assert out["observed_mean_diff"] < 0


def test_exact_permutation_null_effect_has_large_p():
    values = [1.0, 2.0, 3.0, 4.0]
    group1 = [True, False, True, False]  # mean 2.0 vs 3.0 — middling
    out = kvt.exact_permutation_test(values, group1, alternative="greater")
    assert out["p_one_sided"] > 0.5


def test_exact_permutation_rejects_empty_group():
    with pytest.raises(ValueError, match="non-empty"):
        kvt.exact_permutation_test([1.0, 2.0], [True, True], alternative="greater")


def test_rank_biserial_bounds():
    # U=0 → perfect separation in favour of group 2; U=n1*n2 → group 1.
    assert kvt.rank_biserial(0.0, 3, 10) == pytest.approx(-1.0)
    assert kvt.rank_biserial(30.0, 3, 10) == pytest.approx(1.0)
    assert kvt.rank_biserial(15.0, 3, 10) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# End-to-end against the committed artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed_result():
    import json

    with open(kvt.DEFAULT_KSTAR_JSON) as f:
        kstar_data = json.load(f)
    with open(kvt.DEFAULT_PROVENANCE_JSON) as f:
        provenance = json.load(f)
    with open(kvt.DEFAULT_VERDICT_JSON) as f:
        verdict_data = json.load(f)
    return kvt.build_result(kstar_data, provenance, verdict_data)


def test_committed_join_shape(committed_result):
    r = committed_result
    assert r["beta_dedup"]["n_artifact_cells"] == 75
    assert r["beta_dedup"]["n_effective_columns"] == 15
    assert r["beta_dedup"]["kstar_beta_invariant"] is True
    assert r["join_report"]["n_verdict_cells"] == 37
    assert r["join_report"]["n_columns_joined"] == 13


def test_committed_primary_threshold_test_is_null(committed_result):
    t = committed_result["threshold_test_primary"]
    assert (t["n1"], t["n2"]) == (3, 8)
    assert t["rank_biserial"] == pytest.approx(0.0)
    assert t["mwu_p_one_sided"] > 0.5
    assert t["permutation"]["p_two_sided"] > 0.9


def test_committed_failure_zone_is_perfect_separation(committed_result):
    t = committed_result["failure_zone_test_posthoc"]
    assert (t["n1"], t["n2"]) == (3, 10)
    assert t["rank_biserial"] == pytest.approx(-1.0)
    assert t["mwu_p_one_sided"] == pytest.approx(1.0 / math.comb(13, 3))
    assert "POST-HOC" in t["registration"]


def test_committed_no_convergence_cells_all_kstar4(committed_result):
    ct = committed_result["cross_tab"]
    assert ct["no_convergence_only_in_kstar"] == [4]
    assert ct["cells_by_kstar"]["4"]["no_convergence"] == 6


def test_committed_output_files_match_regeneration(committed_result, tmp_path):
    """The committed .json output must be regenerable byte-identically."""
    import json

    committed = kvt.DEFAULT_OUT_JSON
    assert committed.exists(), "run the script to commit its outputs"
    regenerated = json.dumps(committed_result, indent=2, sort_keys=False) + "\n"
    assert committed.read_text() == regenerated

    committed_md = kvt.DEFAULT_OUT_MD
    assert committed_md.exists()
    assert committed_md.read_text() == kvt.render_markdown(committed_result)
