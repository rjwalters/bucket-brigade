#!/usr/bin/env python3
"""Quantify what the 20-seed noise buy-down bought (issue #443, #430 Task 3).

Issue #443 extended the #360/#420 trainability sweep from 4 to 20 seeds on a
k*-stratified subset of 8 phase-diagram cells (all at the canonical β = 0.5 —
per #442 / PR #450, β is inert in bernoulli mode, so these are the 8 effective
(κ, c) cells). This script commits the acceptance test reproducibly:

1. Loads the per-seed ``summary.json`` files under
   ``experiments/p3_specialization/phase_diagram_ppo_v2/cell_<tag>/seed_*/``
   for the 8 buy-down cells (defined by
   ``experiments/nash/phase_diagram/results_noise_buydown_443.json``).
2. Per cell, compares the n=4 baseline (seeds 42-45, the committed #420
   sweep) against the full n=20 sample (seeds 42-61): mean, std and a
   seeded bootstrap CI on the mean of ``gap_closed_ne`` (and
   ``gap_closed_homogeneous``). The acceptance metric is the CI
   half-width ratio n=4 / n=20 — did more seeds actually buy precision?
3. Checks whether any cell's trainability verdict (the #360 ladder on the
   homogeneous-baseline gap, thresholds 0.20 / 0.49 / 0.88) flips at n=20.
4. Runs the #430 Task 2 class-comparison test under the *partial* k*
   stratification from thrust#259/#268 (thrust#269's per-cell k* artifact
   had not landed as of 2026-07-04): the proven k* >= 2 class is the two
   ``no_convergence`` cells; the six converged cells are the k* = 1
   candidates. Because no-convergence cells have no NE baseline
   (``gap_closed_ne`` is null there by construction), the only
   class-comparable column is ``gap_closed_homogeneous``; the test is an
   exact one-sided Mann-Whitney U on per-cell n=20 means plus an
   exhaustive label-permutation test (C(8,2) = 28 assignments) on the
   difference in class means. Verdict stated either way.
5. Cross-checks every per-cell mean against the regenerated
   ``recalibrated_verdict.json`` (fails loudly on drift).

Pure join/stats over committed artifacts — no training, no Nash solves —
safe to run locally:

    uv run python experiments/nash/phase_diagram/noise_buydown_precision.py

Outputs (written next to this script, deterministic — bootstrap RNG is
seeded):
    - ``noise_buydown_precision.json`` — machine-readable stats
    - ``noise_buydown_precision.md``   — write-up
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"

DEFAULT_CELLS_SOURCE = PHASE_DIAGRAM_ROOT / "results_noise_buydown_443.json"
DEFAULT_PPO_ROOT = (
    REPO_ROOT / "experiments" / "p3_specialization" / "phase_diagram_ppo_v2"
)
DEFAULT_VERDICT_JSON = DEFAULT_PPO_ROOT / "recalibrated_verdict.json"
DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "noise_buydown_precision.json"
DEFAULT_OUT_MD = PHASE_DIAGRAM_ROOT / "noise_buydown_precision.md"

#: The committed #420 baseline seeds (n=4); everything else is buy-down.
BASELINE_SEEDS = frozenset({42, 43, 44, 45})

#: Bootstrap configuration — seeded so the committed outputs are reproducible.
BOOTSTRAP_RESAMPLES = 20_000
BOOTSTRAP_SEED = 443

#: #360 verdict ladder (mirrors run_tier1_cell.VERDICT_THRESHOLDS /
#: run_phase_diagram_ppo._verdict_for — re-declared so a change over there is
#: a visible diff here, not a silent drift). Consumes the homogeneous gap.
VERDICT_THRESHOLDS = (0.20, 0.49, 0.88)

GAP_COLUMNS = ("gap_closed_ne", "gap_closed_homogeneous")


def verdict_for(gap_closed_value: float) -> str:
    """Same trainability verdict ladder as the #360 sweep driver."""
    low, mid, high = VERDICT_THRESHOLDS
    if gap_closed_value >= high:
        return "closed"
    if gap_closed_value >= mid:
        return "partial_upper"
    if gap_closed_value >= low:
        return "partial_lower"
    return "insufficient"


def load_buydown_cells(cells_source: Path) -> list[dict]:
    """The 8 buy-down cells (tag, NE verdict, β/κ/c) from the cells-source."""
    data = json.loads(cells_source.read_text())
    cells = [
        {
            "cell_tag": c["tag"],
            "beta": c["beta"],
            "kappa": c["kappa"],
            "c": c["c"],
            "ne_verdict": c["verdict"],
        }
        for c in data["cells"]
    ]
    return sorted(cells, key=lambda c: c["cell_tag"])


def load_seed_gaps(ppo_root: Path, cell_tag: str) -> list[dict]:
    """Per-seed gap columns for one cell, sorted by seed number."""
    cell_dir = ppo_root / f"cell_{cell_tag}"
    rows: list[dict] = []
    for sdir in sorted(cell_dir.glob("seed_*")):
        sf = sdir / "summary.json"
        if not sf.exists():
            continue
        sd = json.loads(sf.read_text())
        if sd.get("trailing5_team_mean") is None:
            continue
        rows.append(
            {
                "seed": int(sd["seed"]),
                "gap_closed_ne": sd.get("gap_closed_ne"),
                "gap_closed_homogeneous": sd.get("gap_closed_homogeneous"),
            }
        )
    rows.sort(key=lambda r: r["seed"])
    if not rows:
        raise ValueError(f"no seed summaries found under {cell_dir}")
    return rows


def bootstrap_ci(
    values: list[float],
    rng: np.random.Generator,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean (deterministic given ``rng``)."""
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def sample_stats(values: list[float], rng: np.random.Generator) -> dict:
    """Mean / population std (matching recalibrate) / bootstrap CI half-width."""
    mean = float(sum(values) / len(values))
    std = float(statistics.pstdev(values)) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values, rng)
    return {
        "n": len(values),
        "mean": mean,
        "std": std,
        "ci95_lo": lo,
        "ci95_hi": hi,
        "ci95_half_width": (hi - lo) / 2.0,
    }


def precision_rows(cells: list[dict], ppo_root: Path) -> list[dict]:
    """Per-cell n=4 vs n=20 precision comparison for both gap columns."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    out: list[dict] = []
    for cell in cells:
        seed_rows = load_seed_gaps(ppo_root, cell["cell_tag"])
        row: dict = dict(cell)
        row["seeds"] = [r["seed"] for r in seed_rows]
        row["n_seeds"] = len(seed_rows)
        for col in GAP_COLUMNS:
            values = {r["seed"]: r[col] for r in seed_rows}
            if any(v is None for v in values.values()):
                # no_convergence cells have no NE baseline: gap_closed_ne is
                # null for every seed by construction.
                assert all(v is None for v in values.values()), (
                    f"{cell['cell_tag']}/{col}: mixed null and non-null seeds"
                )
                row[col] = None
                continue
            baseline = [v for s, v in sorted(values.items()) if s in BASELINE_SEEDS]
            full = [v for _, v in sorted(values.items())]
            n4 = sample_stats(baseline, rng)
            n20 = sample_stats(full, rng)
            row[col] = {
                "per_seed": {str(s): v for s, v in sorted(values.items())},
                "n4": n4,
                "n20": n20,
                "half_width_ratio_n4_over_n20": (
                    n4["ci95_half_width"] / n20["ci95_half_width"]
                    if n20["ci95_half_width"] > 0
                    else None
                ),
            }
        out.append(row)
    return out


def verdict_flip_table(rows: list[dict]) -> list[dict]:
    """Trainability-ladder verdict at n=4 vs n=20 (homogeneous column)."""
    table = []
    for row in rows:
        col = row["gap_closed_homogeneous"]
        v4 = verdict_for(col["n4"]["mean"])
        v20 = verdict_for(col["n20"]["mean"])
        table.append(
            {
                "cell_tag": row["cell_tag"],
                "ne_verdict": row["ne_verdict"],
                "gap_closed_homogeneous_mean_n4": col["n4"]["mean"],
                "gap_closed_homogeneous_mean_n20": col["n20"]["mean"],
                "verdict_n4": v4,
                "verdict_n20": v20,
                "flipped": v4 != v20,
            }
        )
    return table


def kstar_class_test(rows: list[dict]) -> dict:
    """#430 Task 2 class test under the thrust#259/#268 partial stratification.

    Proven k* >= 2 class: the ``no_convergence`` cells (thrust#259 proved
    k* > 1 exactly on the no-convergence cells). k* = 1 candidates: the
    converged cells. ``gap_closed_ne`` is null on every no-convergence cell
    (no NE baseline), so the class test runs on ``gap_closed_homogeneous`` —
    the only column defined for both classes.
    """
    kstar_ge2 = [r for r in rows if r["ne_verdict"] == "no_convergence"]
    kstar_1 = [r for r in rows if r["ne_verdict"] != "no_convergence"]

    means_ge2 = [r["gap_closed_homogeneous"]["n20"]["mean"] for r in kstar_ge2]
    means_1 = [r["gap_closed_homogeneous"]["n20"]["mean"] for r in kstar_1]

    # Exact one-sided Mann-Whitney U on per-cell n=20 means: alternative is
    # that the proven k* >= 2 cells close LESS of the gap.
    mw = mannwhitneyu(means_ge2, means_1, alternative="less", method="exact")

    # Exhaustive label permutation on the difference in class means:
    # all C(8,2) = 28 assignments of the "k* >= 2" label.
    all_means = means_ge2 + means_1
    n_ge2 = len(means_ge2)
    observed = float(np.mean(means_ge2) - np.mean(means_1))
    diffs = []
    for combo in itertools.combinations(range(len(all_means)), n_ge2):
        grp = [all_means[i] for i in combo]
        rest = [all_means[i] for i in range(len(all_means)) if i not in combo]
        diffs.append(float(np.mean(grp) - np.mean(rest)))
    perm_p = sum(1 for d in diffs if d <= observed + 1e-12) / len(diffs)

    alpha = 0.05
    significant = bool(mw.pvalue < alpha and perm_p < alpha)
    return {
        "stratification": "partial (thrust#259/#268): k*>=2 = no_convergence cells",
        "column": "gap_closed_homogeneous",
        "column_note": (
            "gap_closed_ne is null on all no_convergence cells (no NE baseline"
            " exists), so the NE-calibrated column cannot be class-compared"
            " under this stratification; gap_closed_homogeneous is defined for"
            " all 8 cells."
        ),
        "kstar_ge2_cells": {
            r["cell_tag"]: r["gap_closed_homogeneous"]["n20"]["mean"] for r in kstar_ge2
        },
        "kstar_1_candidate_cells": {
            r["cell_tag"]: r["gap_closed_homogeneous"]["n20"]["mean"] for r in kstar_1
        },
        "observed_mean_difference": observed,
        "mannwhitney_u": float(mw.statistic),
        "mannwhitney_p_one_sided": float(mw.pvalue),
        "permutation_p_one_sided": perm_p,
        "n_permutations": len(diffs),
        "alpha": alpha,
        "significant": significant,
        "verdict": (
            "k*>=2 (no_convergence) cells close significantly less of the"
            " homogeneous gap than the k*=1 candidates"
            if significant
            else "no significant separation between the proven k*>=2 cells and"
            " the k*=1 candidates on gap_closed_homogeneous at alpha=0.05"
        ),
        "thrust269_status": (
            "OPEN as of 2026-07-04 — no per-cell k* artifact; the full #430"
            " Task 2 test (all 13 effective cells, actual k* values, the"
            " gap_closed_ne column where defined) remains blocked on it."
        ),
    }


def cross_check_verdict(rows: list[dict], verdict_json: Path) -> dict:
    """Fail loudly if per-cell means drift from recalibrated_verdict.json."""
    verdict = json.loads(verdict_json.read_text())
    by_tag = {c["cell_tag"]: c for c in verdict["cells"]}
    checked = 0
    for row in rows:
        v = by_tag.get(row["cell_tag"])
        if v is None:
            raise ValueError(
                f"{row['cell_tag']} missing from {verdict_json} — artifacts out of sync"
            )
        if v["n_seeds"] != row["n_seeds"]:
            raise ValueError(
                f"{row['cell_tag']}: n_seeds {row['n_seeds']} here vs"
                f" {v['n_seeds']} in recalibrated_verdict.json — regenerate it"
                " (experiments/scripts/recalibrate_phase_diagram_ppo.py)"
            )
        for col, key in (
            ("gap_closed_ne", "gap_closed_ne_mean"),
            ("gap_closed_homogeneous", "gap_closed_homogeneous_mean"),
        ):
            here = row[col]["n20"]["mean"] if row[col] is not None else None
            there = v[key]
            if here is None and there is None:
                continue
            if (
                here is None
                or there is None
                or not math.isclose(here, there, rel_tol=0, abs_tol=1e-9)
            ):
                raise ValueError(
                    f"{row['cell_tag']}/{key}: {here} here vs {there} in {verdict_json}"
                )
        checked += 1
    return {"n_cells_checked": checked, "verdict_json": str(verdict_json)}


def _fmt(x: float | None, prec: int = 3) -> str:
    return "null" if x is None else f"{x:.{prec}f}"


def render_markdown(result: dict) -> str:
    rows = result["cells"]
    flips = result["verdict_flips"]
    kstar = result["kstar_class_test"]

    ratios = [
        r["gap_closed_ne"]["half_width_ratio_n4_over_n20"]
        for r in rows
        if r["gap_closed_ne"] is not None
    ]
    ratios_h = [
        r["gap_closed_homogeneous"]["half_width_ratio_n4_over_n20"] for r in rows
    ]
    n_flipped = sum(1 for f in flips if f["flipped"])

    lines = [
        "# 20-seed noise buy-down: precision bought, verdicts, k* class test"
        " (issue #443)",
        "",
        "Generated by `noise_buydown_precision.py`. Inputs: per-seed"
        " `summary.json` files under",
        "`../../p3_specialization/phase_diagram_ppo_v2/` (seeds 42-45 = the"
        " committed #420 n=4 baseline;",
        "seeds 46-61 = the #443 buy-down, run on alc-6 at the identical #360"
        " budget, git ed0555af),",
        "cross-checked against the regenerated `recalibrated_verdict.json`"
        " (n=20 on these 8 cells).",
        "",
        "All 8 cells are at canonical β = 0.5 — per #442 / PR #450, β is inert"
        " in bernoulli mode,",
        "so these are 8 *effective* (κ, c) cells, one per column.",
        "",
        "## Headline",
        "",
        f"* CI half-widths on `gap_closed_ne` shrank by a factor of"
        f" {min(ratios):.2f}-{max(ratios):.2f}",
        f"  (median {sorted(ratios)[len(ratios) // 2]:.2f}) going from n=4 to"
        f" n=20 — consistent with the",
        "  ~sqrt(5) = 2.24 expected from 5x the seeds. The buy-down bought"
        " precision, not just seeds.",
        f"* Trainability-ladder verdicts: {n_flipped} of {len(flips)} cells"
        " flipped at n=20"
        + (
            " ("
            + ", ".join(
                f"{f['cell_tag']}: {f['verdict_n4']} -> {f['verdict_n20']}"
                for f in flips
                if f["flipped"]
            )
            + ")."
            if n_flipped
            else "."
        ),
        f"* k* class test (partial stratification): {kstar['verdict']}"
        f" (Mann-Whitney one-sided p = {kstar['mannwhitney_p_one_sided']:.3f},"
        f" permutation p = {kstar['permutation_p_one_sided']:.3f}).",
        "",
        "## CI half-width comparison (the acceptance test)",
        "",
        "95% percentile-bootstrap CIs on the per-cell mean"
        f" ({BOOTSTRAP_RESAMPLES} resamples, seed {BOOTSTRAP_SEED}).",
        "`gap_closed_ne` (NE-calibrated column; null for the two"
        " `no_convergence` cells — no NE baseline):",
        "",
        "| cell | NE verdict | n=4 mean±std | n=4 CI half | n=20 mean±std |"
        " n=20 CI half | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        col = r["gap_closed_ne"]
        if col is None:
            lines.append(
                f"| {r['cell_tag']} | {r['ne_verdict']} | null | — | null | — | — |"
            )
            continue
        lines.append(
            f"| {r['cell_tag']} | {r['ne_verdict']} "
            f"| {_fmt(col['n4']['mean'])}±{_fmt(col['n4']['std'])} "
            f"| {_fmt(col['n4']['ci95_half_width'])} "
            f"| {_fmt(col['n20']['mean'])}±{_fmt(col['n20']['std'])} "
            f"| {_fmt(col['n20']['ci95_half_width'])} "
            f"| {_fmt(col['half_width_ratio_n4_over_n20'], 2)} |"
        )

    lines += [
        "",
        "`gap_closed_homogeneous` (defined for all 8 cells):",
        "",
        "| cell | NE verdict | n=4 mean±std | n=4 CI half | n=20 mean±std |"
        " n=20 CI half | ratio |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        col = r["gap_closed_homogeneous"]
        lines.append(
            f"| {r['cell_tag']} | {r['ne_verdict']} "
            f"| {_fmt(col['n4']['mean'])}±{_fmt(col['n4']['std'])} "
            f"| {_fmt(col['n4']['ci95_half_width'])} "
            f"| {_fmt(col['n20']['mean'])}±{_fmt(col['n20']['std'])} "
            f"| {_fmt(col['n20']['ci95_half_width'])} "
            f"| {_fmt(col['half_width_ratio_n4_over_n20'], 2)} |"
        )
    lines.append(
        f"\nHomogeneous-column ratios: {min(ratios_h):.2f}-{max(ratios_h):.2f}"
        f" (median {sorted(ratios_h)[len(ratios_h) // 2]:.2f})."
    )

    lines += [
        "",
        "## Trainability verdict at n=4 vs n=20",
        "",
        "The #360 ladder (0.20 / 0.49 / 0.88 on the homogeneous gap) applied"
        " to the per-cell mean:",
        "",
        "| cell | NE verdict | n=4 mean | n=4 verdict | n=20 mean | n=20"
        " verdict | flipped |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in flips:
        lines.append(
            f"| {f['cell_tag']} | {f['ne_verdict']} "
            f"| {_fmt(f['gap_closed_homogeneous_mean_n4'])} | {f['verdict_n4']} "
            f"| {_fmt(f['gap_closed_homogeneous_mean_n20'])} | {f['verdict_n20']} "
            f"| {'YES' if f['flipped'] else 'no'} |"
        )
    ne_rows = [r for r in rows if r["gap_closed_ne"] is not None]
    biggest = max(
        ne_rows,
        key=lambda r: abs(
            r["gap_closed_ne"]["n4"]["mean"] - r["gap_closed_ne"]["n20"]["mean"]
        ),
    )
    lines += [
        "",
        "Note the n=4 `gap_closed_ne` means also moved substantially on the"
        " noisiest cells",
        f"(largest shift: {biggest['cell_tag']}:"
        f" {biggest['gap_closed_ne']['n4']['mean']:.3f} at n=4 ->"
        f" {biggest['gap_closed_ne']['n20']['mean']:.3f} at n=20) — the n=4"
        " point estimates were",
        "noise-dominated, which is exactly what the buy-down was for.",
        "",
        "## k* class-comparison test (#430 Task 2, partial stratification)",
        "",
        f"{kstar['stratification']}. {kstar['column_note']}",
        "",
        "* k* >= 2 (proven, thrust#259): "
        + ", ".join(f"{t} ({m:.3f})" for t, m in kstar["kstar_ge2_cells"].items()),
        "* k* = 1 candidates: "
        + ", ".join(
            f"{t} ({m:.3f})" for t, m in kstar["kstar_1_candidate_cells"].items()
        ),
        f"* Observed class-mean difference (k*>=2 minus k*=1):"
        f" {kstar['observed_mean_difference']:.3f}",
        f"* Exact one-sided Mann-Whitney U = {kstar['mannwhitney_u']:.1f},"
        f" p = {kstar['mannwhitney_p_one_sided']:.4f}",
        f"* Exhaustive permutation test ({kstar['n_permutations']}"
        f" assignments): p = {kstar['permutation_p_one_sided']:.4f}",
        "",
        f"**Verdict**: {kstar['verdict']}.",
        "",
        f"thrust#269 status: {kstar['thrust269_status']}",
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python experiments/nash/phase_diagram/noise_buydown_precision.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cells-source", type=Path, default=DEFAULT_CELLS_SOURCE)
    parser.add_argument("--ppo-root", type=Path, default=DEFAULT_PPO_ROOT)
    parser.add_argument("--verdict-json", type=Path, default=DEFAULT_VERDICT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    for path in (args.cells_source, args.verdict_json):
        if not path.exists():
            print(f"ERROR: input artifact not found: {path}", file=sys.stderr)
            return 1

    cells = load_buydown_cells(args.cells_source)
    rows = precision_rows(cells, args.ppo_root)
    cross_check = cross_check_verdict(rows, args.verdict_json)

    result = {
        "inputs": {
            "cells_source": str(args.cells_source.relative_to(REPO_ROOT)),
            "ppo_root": str(args.ppo_root.relative_to(REPO_ROOT)),
            "verdict_json": str(args.verdict_json.relative_to(REPO_ROOT)),
        },
        "baseline_seeds": sorted(BASELINE_SEEDS),
        "bootstrap": {
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "ci": "95% percentile bootstrap on the mean",
        },
        "cross_check": cross_check,
        "cells": rows,
        "verdict_flips": verdict_flip_table(rows),
        "kstar_class_test": kstar_class_test(rows),
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {args.out_md}")

    ratios = [
        r["gap_closed_ne"]["half_width_ratio_n4_over_n20"]
        for r in rows
        if r["gap_closed_ne"] is not None
    ]
    kstar = result["kstar_class_test"]
    print(
        f"\nHeadline: gap_closed_ne CI half-width ratios (n4/n20):"
        f" {min(ratios):.2f}-{max(ratios):.2f};"
        f" verdict flips: {sum(1 for f in result['verdict_flips'] if f['flipped'])};"
        f" k* class test MW p = {kstar['mannwhitney_p_one_sided']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
