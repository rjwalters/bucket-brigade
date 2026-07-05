#!/usr/bin/env python3
"""Test whether the coordination threshold k* predicts PPO trainability
(issue #430, Task 2).

The retired conditional-entropy predictor (#368, falsified by
``entropy_vs_trainability.py``: Spearman ρ = 0.109 vs ``gap_closed_ne``,
ρ = 0.342 vs ``gap_closed_homogeneous``, both n.s.) was to be replaced by
the thrust improvability-oracle **coordination threshold k\\*** — the
smallest coalition size whose simultaneous deviation from uniform yields a
significantly positive team-return gap (thrust#259 → thrust#268 →
thrust#269). The threshold prediction: k\\* = 1 cells are trainable by
unilateral-BR methods (PPO), k\\* ≥ 2 cells are not.

thrust PR #290 committed the full 75-cell k\\* artifact
(``docs/research/data/2026-07-kstar-phase-diagram.json``); a verbatim copy
lives next to this script as ``kstar_phase_diagram.json`` with provenance in
``kstar_phase_diagram.provenance.json``. This script:

1. **Dedupes β-replicas.** Per #442 / thrust#458, β is inert: the 75
   (β, κ, c) cells collapse to 15 effective (κ, c) columns and the artifact's
   k\\* is verified to be identical across β within each column (it is in
   fact a pure function of κ: κ=0.1 → k\\*=4, κ=0.3–0.7 → k\\*=2,
   κ=0.9 → k\\*=1). β-replicas are NOT independent observations of k\\* and
   are never counted as such.
2. **Joins** column-level k\\* against
   ``experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json``
   (37 cells → 13 effective columns; ``gap_closed_ne`` pooled across
   β-replicas with n_seeds weights — 20-seed #443 buy-down cells dominate
   their columns).
3. **Threshold prediction test** (primary, pre-registered direction):
   column-level ``gap_closed_ne`` for k\\* = 1 vs k\\* ≥ 2 — exact
   Mann-Whitney U + exhaustive label-permutation test, rank-biserial effect
   size, and the minimum achievable p at these class sizes (power honesty).
4. **Post-hoc failure-zone split** (exploratory, formed after seeing the
   primary null): k\\* = k_max = 4 vs k\\* ≤ 2 on ``gap_closed_homogeneous``
   (the only gap defined on ``no_convergence`` cells).
5. **Cross-tabulates** k\\* against the NE-verdict classes
   (symmetric/mixed/asymmetric/no_convergence).
6. **Benchmarks** k\\* against the retired entropy predictor (Spearman on
   the same joins).

This supersedes the partial-stratification class test in
``noise_buydown_precision.py`` §4 (thrust#259/#268 era: 2 vs 6 cells,
p = 0.43, underpowered) — the full artifact stratifies all 13 columns.

Pure join/stats over committed artifacts — no training, no Nash solves —
safe to run locally:

    uv run python experiments/nash/phase_diagram/kstar_vs_trainability.py

Outputs (written next to this script, deterministic — the permutation tests
are exhaustive, no RNG):
    - ``kstar_vs_trainability.json`` — machine-readable stats
    - ``kstar_vs_trainability.md``   — write-up
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

from scipy.stats import mannwhitneyu, spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"

DEFAULT_KSTAR_JSON = PHASE_DIAGRAM_ROOT / "kstar_phase_diagram.json"
DEFAULT_PROVENANCE_JSON = PHASE_DIAGRAM_ROOT / "kstar_phase_diagram.provenance.json"
DEFAULT_VERDICT_JSON = (
    REPO_ROOT
    / "experiments"
    / "p3_specialization"
    / "phase_diagram_ppo_v2"
    / "recalibrated_verdict.json"
)
DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "kstar_vs_trainability.json"
DEFAULT_OUT_MD = PHASE_DIAGRAM_ROOT / "kstar_vs_trainability.md"

#: Retired predictor headline (committed entropy_vs_trainability.json) for
#: the k*-vs-entropy benchmark in §6.
ENTROPY_BASELINE = {
    "gap_closed_ne_mean": {"spearman_rho": 0.109, "p_value": 0.560, "n_cells": 31},
    "gap_closed_homogeneous_mean": {
        "spearman_rho": 0.342,
        "p_value": 0.060,
        "n_cells": 31,
    },
}


def parse_cell_tag(tag: str) -> tuple[float, float, float]:
    """``b0.10_k0.10_c0.50`` → (beta, kappa, c)."""
    try:
        b_part, k_part, c_part = tag.split("_")
        if not (
            b_part.startswith("b") and k_part.startswith("k") and c_part.startswith("c")
        ):
            raise ValueError
        return float(b_part[1:]), float(k_part[1:]), float(c_part[1:])
    except ValueError as err:
        raise ValueError(f"Unparseable cell_tag: {tag!r}") from err


def kstar_columns(kstar_data: dict) -> tuple[dict[tuple[float, float], int], dict]:
    """Dedupe β-replicas: verify k* is identical across β within each (κ, c).

    β-replicas share the environment (β is inert, #442/thrust#458) so they
    are one observation of k*, not five. Fails loudly if the artifact ever
    disagrees with itself across β.
    """
    by_col: dict[tuple[float, float], dict[float, int]] = {}
    for cell in kstar_data["cells"]:
        key = (cell["kappa"], cell["c"])
        by_col.setdefault(key, {})[cell["beta"]] = cell["k_star"]

    violations = [
        {"kappa": k, "c": c, "k_star_by_beta": betas}
        for (k, c), betas in sorted(by_col.items())
        if len(set(betas.values())) != 1
    ]
    if violations:
        raise ValueError(
            "k* is NOT β-invariant in the artifact — the β-dedup assumption "
            f"(#442/thrust#458) is broken: {violations}"
        )

    columns = {key: next(iter(betas.values())) for key, betas in by_col.items()}
    kappa_map: dict[float, set[int]] = {}
    for (kappa, _), ks in columns.items():
        kappa_map.setdefault(kappa, set()).add(ks)
    report = {
        "n_artifact_cells": len(kstar_data["cells"]),
        "n_effective_columns": len(columns),
        "kstar_beta_invariant": True,
        "kstar_is_function_of_kappa_only": all(len(v) == 1 for v in kappa_map.values()),
        "kstar_by_kappa": {
            f"{kappa:.2f}": sorted(v) for kappa, v in sorted(kappa_map.items())
        },
    }
    return columns, report


def _pooled_mean(pairs: list[tuple[float, int]]) -> float | None:
    """n_seeds-weighted mean = pooled per-seed mean across β-replicas."""
    if not pairs:
        return None
    total = sum(n for _, n in pairs)
    return sum(g * n for g, n in pairs) / total


def join_columns(
    kstar_by_col: dict[tuple[float, float], int], verdict_data: dict
) -> tuple[list[dict], dict]:
    """Join verdict cells (grouped into (κ, c) columns) against column k*.

    Fails loudly if a verdict cell has no k* column — every PPO-swept cell
    must be covered by the full-grid artifact. The reverse (k* columns with
    no PPO sweep) is expected and reported.
    """
    grouped: dict[tuple[float, float], list[dict]] = {}
    for cell in verdict_data["cells"]:
        _, kappa, c = parse_cell_tag(cell["cell_tag"])
        key = (kappa, c)
        if key not in kstar_by_col:
            raise ValueError(
                f"Verdict cell {cell['cell_tag']!r} has no (κ={kappa}, c={c}) "
                "column in the k* artifact — the two artifacts are out of sync."
            )
        grouped.setdefault(key, []).append(cell)

    columns: list[dict] = []
    for (kappa, c), cells in sorted(grouped.items()):
        cells = sorted(cells, key=lambda r: r["cell_tag"])
        ne_pairs = [
            (r["gap_closed_ne_mean"], r["n_seeds"])
            for r in cells
            if r["gap_closed_ne_mean"] is not None
        ]
        hom_pairs = [(r["gap_closed_homogeneous_mean"], r["n_seeds"]) for r in cells]
        columns.append(
            {
                "kappa": kappa,
                "c": c,
                "k_star": kstar_by_col[(kappa, c)],
                "cell_tags": [r["cell_tag"] for r in cells],
                "n_beta_replicas": len(cells),
                "ne_verdicts": sorted({r["ne_verdict"] for r in cells}),
                "gap_closed_ne_pooled": _pooled_mean(ne_pairs),
                "n_seeds_ne": sum(n for _, n in ne_pairs),
                "gap_closed_ne_by_cell": [
                    {
                        "cell_tag": r["cell_tag"],
                        "n_seeds": r["n_seeds"],
                        "gap_closed_ne_mean": r["gap_closed_ne_mean"],
                    }
                    for r in cells
                ],
                "gap_closed_homogeneous_pooled": _pooled_mean(hom_pairs),
                "n_seeds_homogeneous": sum(n for _, n in hom_pairs),
            }
        )

    unswept = sorted(set(kstar_by_col) - set(grouped))
    report = {
        "n_verdict_cells": len(verdict_data["cells"]),
        "n_columns_joined": len(columns),
        "kstar_columns_without_ppo_sweep": [
            {"kappa": k, "c": c, "k_star": kstar_by_col[(k, c)]} for k, c in unswept
        ],
    }
    return columns, report


def rank_biserial(u_stat: float, n1: int, n2: int) -> float:
    """Rank-biserial correlation (= Cliff's delta) from the MWU statistic."""
    return 2.0 * u_stat / (n1 * n2) - 1.0


def exact_permutation_test(
    values: list[float], group1: list[bool], alternative: str
) -> dict:
    """Exhaustive label-permutation test on the difference of group means.

    ``alternative`` is the predicted direction for group 1: ``"greater"``
    (group-1 mean higher) or ``"less"``. Deterministic — enumerates all
    C(n, n1) assignments, no RNG.
    """
    if alternative not in ("greater", "less"):
        raise ValueError(f"Unknown alternative: {alternative!r}")
    n = len(values)
    n1 = sum(group1)
    if not 0 < n1 < n:
        raise ValueError("Both groups must be non-empty.")

    def diff(indices1: frozenset[int]) -> float:
        m1 = sum(values[i] for i in indices1) / n1
        m2 = sum(values[i] for i in range(n) if i not in indices1) / (n - n1)
        return m1 - m2

    observed = diff(frozenset(i for i, g in enumerate(group1) if g))
    signed_obs = observed if alternative == "greater" else -observed
    eps = 1e-12
    n_perm = math.comb(n, n1)
    count_one = 0
    count_two = 0
    for combo in itertools.combinations(range(n), n1):
        d = diff(frozenset(combo))
        signed = d if alternative == "greater" else -d
        if signed >= signed_obs - eps:
            count_one += 1
        if abs(d) >= abs(observed) - eps:
            count_two += 1
    return {
        "observed_mean_diff": observed,
        "alternative": alternative,
        "p_one_sided": count_one / n_perm,
        "p_two_sided": count_two / n_perm,
        "n_permutations": n_perm,
        "min_achievable_p_one_sided": 1.0 / n_perm,
    }


def class_comparison(
    rows: list[dict],
    value_key: str,
    in_group1,
    group1_label: str,
    group2_label: str,
    alternative: str,
    registration: str,
) -> dict:
    """Mann-Whitney + exact permutation comparison of two k* classes."""
    usable = [r for r in rows if r[value_key] is not None]
    values = [r[value_key] for r in usable]
    group1 = [in_group1(r) for r in usable]
    g1 = [v for v, g in zip(values, group1) if g]
    g2 = [v for v, g in zip(values, group1) if not g]
    if not g1 or not g2:
        return {
            "value_key": value_key,
            "group1": group1_label,
            "group2": group2_label,
            "skipped": f"empty class (n1={len(g1)}, n2={len(g2)})",
        }
    mwu_one = mannwhitneyu(g1, g2, alternative=alternative, method="exact")
    mwu_two = mannwhitneyu(g1, g2, alternative="two-sided", method="exact")
    perm = exact_permutation_test(values, group1, alternative)
    return {
        "value_key": value_key,
        "registration": registration,
        "group1": group1_label,
        "group2": group2_label,
        "n1": len(g1),
        "n2": len(g2),
        "group1_values": sorted(g1),
        "group2_values": sorted(g2),
        "group1_median": sorted(g1)[len(g1) // 2],
        "group2_median": sorted(g2)[len(g2) // 2],
        "mann_whitney_u": float(mwu_one.statistic),
        "mwu_p_one_sided": float(mwu_one.pvalue),
        "mwu_p_two_sided": float(mwu_two.pvalue),
        "mwu_min_achievable_p_one_sided": 1.0 / math.comb(len(values), len(g1)),
        "rank_biserial": rank_biserial(float(mwu_one.statistic), len(g1), len(g2)),
        "permutation": perm,
    }


def spearman_block(pairs: list[tuple[float, float]], label: str) -> dict:
    xs = [a for a, _ in pairs]
    ys = [b for _, b in pairs]
    rho, p = spearmanr(xs, ys)
    return {
        "label": label,
        "n": len(pairs),
        "spearman_rho": float(rho),
        "p_value": float(p),
    }


def cross_tabulate(columns: list[dict], verdict_data: dict) -> dict:
    """k* vs NE-verdict class, at cell and column granularity."""
    kstar_by_col = {(r["kappa"], r["c"]): r["k_star"] for r in columns}
    cell_counts: dict[int, dict[str, int]] = {}
    for cell in verdict_data["cells"]:
        _, kappa, c = parse_cell_tag(cell["cell_tag"])
        ks = kstar_by_col[(kappa, c)]
        cell_counts.setdefault(ks, {}).setdefault(cell["ne_verdict"], 0)
        cell_counts[ks][cell["ne_verdict"]] += 1
    col_counts: dict[int, dict[str, int]] = {}
    for r in columns:
        for v in r["ne_verdicts"]:
            col_counts.setdefault(r["k_star"], {}).setdefault(v, 0)
            col_counts[r["k_star"]][v] += 1
    no_conv_kstars = sorted(
        {ks for ks, verdicts in cell_counts.items() if "no_convergence" in verdicts}
    )
    return {
        "cells_by_kstar": {
            str(ks): dict(sorted(v.items())) for ks, v in sorted(cell_counts.items())
        },
        "columns_by_kstar": {
            str(ks): dict(sorted(v.items())) for ks, v in sorted(col_counts.items())
        },
        "no_convergence_only_in_kstar": no_conv_kstars,
    }


def n20_subset(columns: list[dict]) -> dict:
    """The #443 20-seed buy-down cells — highest-precision per-cell means."""
    rows = []
    for col in columns:
        for cell in col["gap_closed_ne_by_cell"]:
            if cell["n_seeds"] == 20:
                rows.append(
                    {
                        "cell_tag": cell["cell_tag"],
                        "k_star": col["k_star"],
                        "gap_closed_ne_mean": cell["gap_closed_ne_mean"],
                    }
                )
    rows.sort(key=lambda r: (r["k_star"], r["cell_tag"]))
    non_null = [r for r in rows if r["gap_closed_ne_mean"] is not None]
    n1 = sum(1 for r in non_null if r["k_star"] == 1)
    n2 = len(non_null) - n1
    return {
        "cells": rows,
        "n_non_null": len(non_null),
        "class_sizes_kstar1_vs_ge2": [n1, n2],
        "min_achievable_p_one_sided": (
            1.0 / math.comb(len(non_null), n1) if 0 < n1 < len(non_null) else None
        ),
        "note": (
            "Preferred for precision (n=20 seeds per cell) but only "
            f"{len(non_null)} non-null cells split {n1} vs {n2} — the minimum "
            "achievable one-sided p exceeds 0.05, so no confirmatory test is "
            "possible on this subset alone; reported descriptively."
        ),
    }


def build_result(kstar_data: dict, provenance: dict, verdict_data: dict) -> dict:
    kstar_by_col, dedup_report = kstar_columns(kstar_data)
    columns, join_report = join_columns(kstar_by_col, verdict_data)

    # Primary, pre-registered: k*=1 trainable vs k*>=2 not, on the NE gap.
    primary = class_comparison(
        columns,
        "gap_closed_ne_pooled",
        lambda r: r["k_star"] == 1,
        "k*=1",
        "k*>=2",
        alternative="greater",
        registration="pre-registered (issue #430 Task 2 / thrust#259 threshold prediction)",
    )
    # Same split on the homogeneous gap (defined on all 13 columns).
    primary_hom = class_comparison(
        columns,
        "gap_closed_homogeneous_pooled",
        lambda r: r["k_star"] == 1,
        "k*=1",
        "k*>=2",
        alternative="greater",
        registration="pre-registered split, secondary outcome (homogeneous gap)",
    )
    # Post-hoc: the failure zone is k*=k_max, not k*>1.
    failure_zone = class_comparison(
        columns,
        "gap_closed_homogeneous_pooled",
        lambda r: r["k_star"] == 4,
        "k*=4 (=k_max)",
        "k*<=2",
        alternative="less",
        registration=(
            "POST-HOC (exploratory; split chosen after observing the primary "
            "null — treat the p-value as descriptive, not confirmatory)"
        ),
    )

    ne_cols = [r for r in columns if r["gap_closed_ne_pooled"] is not None]
    cell_pairs_ne = []
    cell_pairs_hom = []
    kstar_lookup = {(r["kappa"], r["c"]): r["k_star"] for r in columns}
    for cell in verdict_data["cells"]:
        _, kappa, c = parse_cell_tag(cell["cell_tag"])
        ks = kstar_lookup[(kappa, c)]
        if cell["gap_closed_ne_mean"] is not None:
            cell_pairs_ne.append((ks, cell["gap_closed_ne_mean"]))
        cell_pairs_hom.append((ks, cell["gap_closed_homogeneous_mean"]))

    correlations = {
        "column_level": [
            spearman_block(
                [(r["k_star"], r["gap_closed_ne_pooled"]) for r in ne_cols],
                "k* vs gap_closed_ne (pooled, columns)",
            ),
            spearman_block(
                [(r["k_star"], r["gap_closed_homogeneous_pooled"]) for r in columns],
                "k* vs gap_closed_homogeneous (pooled, columns)",
            ),
        ],
        "cell_level_pseudo_replicated": [
            spearman_block(cell_pairs_ne, "k* vs gap_closed_ne (cells)"),
            spearman_block(cell_pairs_hom, "k* vs gap_closed_homogeneous (cells)"),
        ],
        "cell_level_caveat": (
            "Cell-level rows repeat each column's k* across its β-replicas "
            "(β is inert, #442) — pseudo-replication inflates n; shown only "
            "for comparability with the retired entropy predictor's join."
        ),
        "entropy_baseline_retired": ENTROPY_BASELINE,
    }

    return {
        "inputs": {
            "kstar_json": "experiments/nash/phase_diagram/kstar_phase_diagram.json",
            "verdict_json": (
                "experiments/p3_specialization/phase_diagram_ppo_v2/"
                "recalibrated_verdict.json"
            ),
        },
        "kstar_provenance": provenance,
        "kstar_protocol": kstar_data.get("protocol", {}),
        "beta_dedup": dedup_report,
        "join_report": join_report,
        "columns": columns,
        "threshold_test_primary": primary,
        "threshold_test_homogeneous": primary_hom,
        "failure_zone_test_posthoc": failure_zone,
        "correlations": correlations,
        "cross_tab": cross_tabulate(columns, verdict_data),
        "n20_subset": n20_subset(columns),
        "supersedes": (
            "noise_buydown_precision.py §4 partial-stratification class test "
            "(thrust#259/#268 era, 2 vs 6 cells, p=0.43)"
        ),
    }


def _fmt(x: float | None, digits: int = 3) -> str:
    return "null" if x is None else f"{x:.{digits}f}"


def render_markdown(result: dict) -> str:
    primary = result["threshold_test_primary"]
    primary_hom = result["threshold_test_homogeneous"]
    fz = result["failure_zone_test_posthoc"]
    corr = result["correlations"]
    ct = result["cross_tab"]
    prov = result["kstar_provenance"]
    dedup = result["beta_dedup"]

    lines = [
        "# k* vs PPO trainability: the literal threshold prediction fails;"
        " k* = k_max marks the failure zone",
        "",
        "Generated by `kstar_vs_trainability.py` (issue #430, Task 2). Inputs:",
        f"`kstar_phase_diagram.json` (verbatim copy of {prov['source_repo']}",
        f"`{prov['source_path']}` @ `{prov['source_commit'][:12]}`, PR"
        f" {prov['source_pr']}, issue {prov['source_issue']}) joined against",
        "`../../p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`.",
        "",
        "## Verdict (stated either way, per the acceptance criterion)",
        "",
        "1. **The pre-registered threshold prediction — k\\* = 1 cells trainable,",
        "   k\\* ≥ 2 cells not — is NOT supported.** On the pooled NE gap the two",
        f"   classes are indistinguishable: rank-biserial ="
        f" {primary['rank_biserial']:.2f},",
        f"   exact one-sided Mann-Whitney p = {primary['mwu_p_one_sided']:.3f},",
        f"   permutation Δmean = {primary['permutation']['observed_mean_diff']:.3f}"
        f" (two-sided p = {primary['permutation']['p_two_sided']:.2f}).",
        "   PPO closes the NE gap at k\\* = 2 just as well as at k\\* = 1 — the",
        '   literal "unilateral-BR methods fail whenever k\\* > 1" account is',
        "   falsified for PPO on this grid.",
        "2. **Post hoc, the failure zone is k\\* = k_max = 4.** On the homogeneous",
        "   gap (the only gap defined on `no_convergence` cells) the three",
        "   k\\* = 4 columns sit below all ten k\\* ≤ 2 columns — perfect",
        f"   separation (rank-biserial = {fz['rank_biserial']:.2f}, exact"
        f" one-sided p = {fz['mwu_p_one_sided']:.4f},",
        f"   the minimum achievable at 3 vs 10; permutation two-sided"
        f" p = {fz['permutation']['p_two_sided']:.3f}).",
        "   This split was chosen after seeing the primary null — treat it as",
        "   exploratory until an out-of-grid cell tests it.",
        "3. **k\\* beats the retired entropy predictor, with a caveat.** Spearman",
        "   k\\* vs homogeneous gap: ρ ="
        f" {corr['cell_level_pseudo_replicated'][1]['spearman_rho']:.3f}"
        f" (p = {corr['cell_level_pseudo_replicated'][1]['p_value']:.4f},"
        f" n = {corr['cell_level_pseudo_replicated'][1]['n']} cells) /"
        f" ρ = {corr['column_level'][1]['spearman_rho']:.3f}",
        f"   (p = {corr['column_level'][1]['p_value']:.3f},"
        f" n = {corr['column_level'][1]['n']} columns), vs the retired entropy",
        "   predictor's ρ = 0.109 / 0.342 (both n.s.). The caveat: in this",
        "   artifact k\\* is a pure function of κ (κ=0.1 → 4, κ=0.3–0.7 → 2,",
        "   κ=0.9 → 1), so its predictive content is observationally equivalent",
        '   to "κ = 0.1 is untrainable" — the coalition mechanism is a candidate',
        "   *explanation* of the κ effect, not independently identified here.",
        "",
        "## β-dedup (why n = 13 columns, not 37 cells)",
        "",
        f"The k\\* artifact has {dedup['n_artifact_cells']} (β, κ, c) cells, but β",
        "is inert (#442 / thrust#458): k\\* is byte-identical across β within",
        f"every (κ, c) column (verified — {dedup['n_effective_columns']} effective"
        " columns; k\\* is a pure function",
        "of κ). β-replicas are therefore NOT independent observations; all",
        "class tests run at column level, pooling `gap_closed_ne` across",
        "β-replicas with n_seeds weights (the #443 20-seed buy-down cells",
        "dominate their columns).",
        "",
        "| κ | k\\* |",
        "|---|---|",
    ]
    for kappa, ks in dedup["kstar_by_kappa"].items():
        lines.append(f"| {kappa} | {ks[0]} |")

    lines += [
        "",
        "## Joined columns",
        "",
        "| κ | c | k\\* | NE verdict(s) | pooled gap_closed_ne (n seeds) |"
        " pooled gap_closed_homogeneous (n seeds) |",
        "|---|---|---|---|---|---|",
    ]
    for r in result["columns"]:
        ne = (
            "null"
            if r["gap_closed_ne_pooled"] is None
            else f"{r['gap_closed_ne_pooled']:.3f} (n={r['n_seeds_ne']})"
        )
        hom = f"{r['gap_closed_homogeneous_pooled']:.3f} (n={r['n_seeds_homogeneous']})"
        lines.append(
            f"| {r['kappa']:.1f} | {r['c']:.1f} | {r['k_star']} |"
            f" {', '.join(r['ne_verdicts'])} | {ne} | {hom} |"
        )
    missing = result["join_report"]["kstar_columns_without_ppo_sweep"]
    if missing:
        miss = ", ".join(
            f"(κ={m['kappa']}, c={m['c']}, k*={m['k_star']})" for m in missing
        )
        lines.append("")
        lines.append(f"k\\* columns without a PPO sweep (not joinable): {miss}.")

    def test_block(title: str, t: dict) -> list[str]:
        return [
            "",
            f"## {title}",
            "",
            f"- Registration: {t['registration']}",
            f"- Outcome: `{t['value_key']}`; classes {t['group1']}"
            f" (n={t['n1']}) vs {t['group2']} (n={t['n2']})",
            f"- {t['group1']} values: "
            + ", ".join(f"{v:.3f}" for v in t["group1_values"])
            + f" (median {t['group1_median']:.3f})",
            f"- {t['group2']} values: "
            + ", ".join(f"{v:.3f}" for v in t["group2_values"])
            + f" (median {t['group2_median']:.3f})",
            f"- Exact Mann-Whitney U = {t['mann_whitney_u']:.1f}:"
            f" one-sided p = {t['mwu_p_one_sided']:.4f},"
            f" two-sided p = {t['mwu_p_two_sided']:.4f}",
            f"- Rank-biserial (Cliff's δ) = {t['rank_biserial']:.2f}",
            f"- Exhaustive permutation ({t['permutation']['n_permutations']}"
            f" assignments): Δmean = {t['permutation']['observed_mean_diff']:.4f},"
            f" one-sided p = {t['permutation']['p_one_sided']:.4f},"
            f" two-sided p = {t['permutation']['p_two_sided']:.4f}",
            f"- Power floor: minimum achievable one-sided p at these class sizes"
            f" = {t['mwu_min_achievable_p_one_sided']:.4f} — only near-perfect"
            " separation is detectable; a real but modest threshold effect"
            " would be missed.",
        ]

    lines += test_block(
        "Primary test: k\\* = 1 vs k\\* ≥ 2 on the NE gap (pre-registered)", primary
    )
    lines += test_block(
        "Same split on the homogeneous gap (all 13 columns)", primary_hom
    )
    lines += test_block(
        "Post-hoc failure zone: k\\* = 4 vs k\\* ≤ 2 on the homogeneous gap", fz
    )

    lines += [
        "",
        "## k\\* vs the NE-verdict classes (cross-tabulation)",
        "",
        "Cell counts (37 verdict cells; β-replicas counted — descriptive only):",
        "",
        "| k\\* | asymmetric_only | mixed | symmetric_only | no_convergence |",
        "|---|---|---|---|---|",
    ]
    for ks, verdicts in result["cross_tab"]["cells_by_kstar"].items():
        row = [
            str(verdicts.get(v, 0))
            for v in ("asymmetric_only", "mixed", "symmetric_only", "no_convergence")
        ]
        lines.append(f"| {ks} | " + " | ".join(row) + " |")
    lines += [
        "",
        "k\\* **neither reproduces nor refines** the four NE-verdict classes: it",
        "cross-cuts symmetric/mixed/asymmetric (every k\\* level contains at",
        "least two of them, and every converged class spans at least two k\\*",
        "levels). The single clean alignment is that all `no_convergence` cells",
        f"fall in k\\* = {ct['no_convergence_only_in_kstar'][0]} columns"
        " (thrust#269's headline), consistent with the",
        "post-hoc failure-zone reading. k\\* is a coarser, orthogonal axis —",
        "a κ-driven trainability marker, not a sharper NE taxonomy.",
        "",
        "## Precision subset (#443, n = 20 seeds)",
        "",
        "| cell_tag | k\\* | gap_closed_ne (n=20) |",
        "|---|---|---|",
    ]
    for r in result["n20_subset"]["cells"]:
        lines.append(
            f"| {r['cell_tag']} | {r['k_star']} | {_fmt(r['gap_closed_ne_mean'])} |"
        )
    n20_k1 = [
        r["gap_closed_ne_mean"]
        for r in result["n20_subset"]["cells"]
        if r["k_star"] == 1 and r["gap_closed_ne_mean"] is not None
    ]
    n20_ge2 = [
        r["gap_closed_ne_mean"]
        for r in result["n20_subset"]["cells"]
        if r["k_star"] >= 2 and r["gap_closed_ne_mean"] is not None
    ]
    lines += [
        "",
        result["n20_subset"]["note"],
        f"Direction check: the k\\* = 1 cells"
        f" ({', '.join(f'{v:.3f}' for v in sorted(n20_k1))}) do not exceed the"
        f" best k\\* ≥ 2 cell ({max(n20_ge2):.3f}) by any margin the noise"
        " supports — consistent with the primary null.",
        "",
        "## k\\* vs the retired entropy predictor",
        "",
        "| predictor | join | ρ | p | n |",
        "|---|---|---|---|---|",
    ]
    for blk in corr["column_level"] + corr["cell_level_pseudo_replicated"]:
        lines.append(
            f"| k\\* | {blk['label']} | {blk['spearman_rho']:.3f} |"
            f" {blk['p_value']:.4f} | {blk['n']} |"
        )
    eb = corr["entropy_baseline_retired"]
    lines += [
        f"| h_cond mean (retired) | entropy vs gap_closed_ne (cells) |"
        f" {eb['gap_closed_ne_mean']['spearman_rho']:.3f} |"
        f" {eb['gap_closed_ne_mean']['p_value']:.3f} |"
        f" {eb['gap_closed_ne_mean']['n_cells']} |",
        f"| h_cond mean (retired) | entropy vs gap_closed_homogeneous (cells) |"
        f" {eb['gap_closed_homogeneous_mean']['spearman_rho']:.3f} |"
        f" {eb['gap_closed_homogeneous_mean']['p_value']:.3f} |"
        f" {eb['gap_closed_homogeneous_mean']['n_cells']} |",
        "",
        "k\\* vs the homogeneous gap is the only (predictor, outcome) pair in",
        "either family that is significant — k\\* is the better predictor. Note",
        "the NE-gap join is selection-biased *against* k\\*: the `no_convergence`",
        "cells where k\\* makes its strongest prediction have no NE baseline and",
        "drop out of that correlation by construction.",
        "",
        "## Relationship to prior tests",
        "",
        f"Supersedes: {result['supersedes']}. That test could only stratify",
        "2 vs 6 cells from the partial thrust#259/#268 gate results; this run",
        "stratifies all 13 swept columns with the full 75-cell artifact.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "uv run python experiments/nash/phase_diagram/kstar_vs_trainability.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kstar-json", type=Path, default=DEFAULT_KSTAR_JSON)
    parser.add_argument("--provenance-json", type=Path, default=DEFAULT_PROVENANCE_JSON)
    parser.add_argument("--verdict-json", type=Path, default=DEFAULT_VERDICT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    for path in (args.kstar_json, args.provenance_json, args.verdict_json):
        if not path.exists():
            print(f"ERROR: input artifact not found: {path}", file=sys.stderr)
            return 1

    with open(args.kstar_json) as f:
        kstar_data = json.load(f)
    with open(args.provenance_json) as f:
        provenance = json.load(f)
    with open(args.verdict_json) as f:
        verdict_data = json.load(f)

    result = build_result(kstar_data, provenance, verdict_data)

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"Wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {args.out_md}")

    primary = result["threshold_test_primary"]
    fz = result["failure_zone_test_posthoc"]
    print(
        "\nHeadline: k*=1 vs k*>=2 on pooled gap_closed_ne "
        f"(n={primary['n1']} vs {primary['n2']} columns): "
        f"rank-biserial={primary['rank_biserial']:.2f}, "
        f"one-sided p={primary['mwu_p_one_sided']:.3f} — threshold prediction"
        " NOT supported."
    )
    print(
        "Post hoc: k*=4 vs k*<=2 on pooled gap_closed_homogeneous "
        f"(n={fz['n1']} vs {fz['n2']}): rank-biserial={fz['rank_biserial']:.2f}, "
        f"one-sided p={fz['mwu_p_one_sided']:.4f} (exploratory)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
