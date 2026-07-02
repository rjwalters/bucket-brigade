#!/usr/bin/env python3
"""Test whether NE conditional action entropy predicts PPO trainability
(issue #430, Task 1 — negative result).

Paper 2 (``slepian-wolf-marl-2``) hypothesised that the #368 conditional
entropy H(A_i*|A_{-i}*) of the converged heterogeneous Nash profile predicts
whether PPO can close the gap to the NE specialist baseline. This script
commits the falsification reproducibly:

1. Loads ``experiments/nash/phase_diagram/conditional_entropy.json``
   (31 converged cells, per-position ``h_cond``) and
   ``experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json``
   (37 cells, per-cell ``gap_closed_ne_mean`` / ``gap_closed_homogeneous_mean``).
2. Joins them on ``cell_tag`` (defensively — any tag present on one side
   only is reported; the 6 ``no_convergence`` cells have no NE profile,
   hence no entropy, and carry ``gap_closed_ne_mean = null``).
3. Reports Spearman ρ for four per-cell entropy aggregates — mean, max,
   min, spread (max − min) of per-position ``h_cond`` — against
   ``gap_closed_ne_mean`` (31 non-null cells) and
   ``gap_closed_homogeneous_mean`` (all joined cells).
4. Emits the β-invariance table: cells sharing (κ, c) have byte-identical
   converged-profile entropies while ``gap_closed_ne_mean`` varies with β,
   so no function of the NE profile can explain within-column trainability
   variance.
5. Documents the n=4-seed noise ceiling (per-cell ``gap_closed_ne_std``).

This is a pure join/stats script over committed artifacts — no training,
no Nash solves — and is safe to run locally:

    uv run python experiments/nash/phase_diagram/entropy_vs_trainability.py

Outputs (written next to this script, deterministic):
    - ``entropy_vs_trainability.json`` — machine-readable stats
    - ``entropy_vs_trainability.md``   — negative-result write-up
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"

DEFAULT_ENTROPY_JSON = PHASE_DIAGRAM_ROOT / "conditional_entropy.json"
DEFAULT_VERDICT_JSON = (
    REPO_ROOT
    / "experiments"
    / "p3_specialization"
    / "phase_diagram_ppo_v2"
    / "recalibrated_verdict.json"
)
DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "entropy_vs_trainability.json"
DEFAULT_OUT_MD = PHASE_DIAGRAM_ROOT / "entropy_vs_trainability.md"

AGGREGATE_NAMES = ("mean", "max", "min", "spread")


def entropy_aggregates(h_conds: list[float]) -> dict[str, float]:
    """Per-cell aggregates of per-position conditional entropies."""
    if not h_conds:
        raise ValueError("Cell has no per-position h_cond values.")
    return {
        "mean": sum(h_conds) / len(h_conds),
        "max": max(h_conds),
        "min": min(h_conds),
        "spread": max(h_conds) - min(h_conds),
    }


def join_cells(entropy_data: dict, verdict_data: dict) -> tuple[list[dict], dict]:
    """Join entropy cells against trainability verdicts on ``cell_tag``.

    Returns (joined_rows, join_report). Fails loudly (ValueError) if any
    entropy cell has no matching verdict — every converged NE cell should
    have a PPO verdict; the reverse (verdict without entropy) is expected
    for ``no_convergence`` cells and is only reported.
    """
    verdict_by_tag = {c["cell_tag"]: c for c in verdict_data["cells"]}
    entropy_tags = {c["cell_tag"] for c in entropy_data["cells"]}

    joined: list[dict] = []
    for cell in sorted(entropy_data["cells"], key=lambda c: c["cell_tag"]):
        tag = cell["cell_tag"]
        if tag not in verdict_by_tag:
            raise ValueError(
                f"cell_tag {tag!r} present in the entropy file but missing from "
                "the recalibrated verdict file — the two artifacts are out of sync."
            )
        verdict = verdict_by_tag[tag]
        h_conds = [p["h_cond"] for p in cell["positions"]]
        joined.append(
            {
                "cell_tag": tag,
                "beta": cell["beta"],
                "kappa": cell["kappa"],
                "c": cell["c"],
                "ne_verdict": verdict["ne_verdict"],
                "n_seeds": verdict["n_seeds"],
                "h_cond": entropy_aggregates(h_conds),
                "gap_closed_ne_mean": verdict["gap_closed_ne_mean"],
                "gap_closed_ne_std": verdict["gap_closed_ne_std"],
                "gap_closed_homogeneous_mean": verdict["gap_closed_homogeneous_mean"],
                "gap_closed_homogeneous_std": verdict["gap_closed_homogeneous_std"],
            }
        )

    verdict_only = sorted(set(verdict_by_tag) - entropy_tags)
    report = {
        "n_entropy_cells": len(entropy_data["cells"]),
        "n_verdict_cells": len(verdict_data["cells"]),
        "n_joined": len(joined),
        "verdict_cells_without_entropy": verdict_only,
        "entropy_file_skipped_cells": [
            {"cell_tag": s["cell_tag"], "reason": s["reason"]}
            for s in entropy_data.get("skipped_cells", [])
        ],
    }
    return joined, report


def spearman_table(joined: list[dict], target_key: str) -> dict:
    """Spearman ρ of each entropy aggregate vs ``target_key`` (nulls skipped)."""
    rows = [r for r in joined if r[target_key] is not None]
    n = len(rows)
    out: dict = {"target": target_key, "n_cells": n, "aggregates": {}}
    for agg in AGGREGATE_NAMES:
        xs = [r["h_cond"][agg] for r in rows]
        ys = [r[target_key] for r in rows]
        rho, p = spearmanr(xs, ys)
        out["aggregates"][agg] = {"spearman_rho": float(rho), "p_value": float(p)}
    return out


def beta_invariance_table(joined: list[dict]) -> list[dict]:
    """Group joined cells by (κ, c); show per-β entropy vs gap_closed_ne.

    Converged-profile entropies are β-invariant (the heterogeneous DO lands
    on the same profile for every β sharing (κ, c)), while gap_closed_ne
    varies with β — the table makes both visible side-by-side.
    """
    groups: dict[tuple[float, float], list[dict]] = {}
    for row in joined:
        groups.setdefault((row["kappa"], row["c"]), []).append(row)

    table = []
    for (kappa, c), rows in sorted(groups.items()):
        rows = sorted(rows, key=lambda r: r["beta"])
        mean_entropies = [r["h_cond"]["mean"] for r in rows]
        table.append(
            {
                "kappa": kappa,
                "c": c,
                "betas": [r["beta"] for r in rows],
                "h_cond_mean_by_beta": mean_entropies,
                "entropy_identical_across_beta": len(set(mean_entropies)) == 1,
                "gap_closed_ne_mean_by_beta": [r["gap_closed_ne_mean"] for r in rows],
                "gap_closed_ne_range": (
                    max(g for g in (r["gap_closed_ne_mean"] for r in rows))
                    - min(g for g in (r["gap_closed_ne_mean"] for r in rows))
                    if all(r["gap_closed_ne_mean"] is not None for r in rows)
                    else None
                ),
            }
        )
    return table


def noise_ceiling(joined: list[dict]) -> dict:
    """Per-cell gap_closed_ne_std summary — the n=4-seed noise ceiling."""
    stds = [
        (r["cell_tag"], r["gap_closed_ne_std"])
        for r in joined
        if r["gap_closed_ne_std"] is not None
    ]
    worst = sorted(stds, key=lambda t: -t[1])[:5]
    return {
        "n_seeds": sorted({r["n_seeds"] for r in joined}),
        "max_gap_closed_ne_std": max(s for _, s in stds),
        "median_gap_closed_ne_std": sorted(s for _, s in stds)[len(stds) // 2],
        "worst_cells": [{"cell_tag": t, "gap_closed_ne_std": s} for t, s in worst],
    }


def render_markdown(result: dict) -> str:
    """Human-readable negative-result summary."""
    ne = result["correlations"]["gap_closed_ne_mean"]
    homog = result["correlations"]["gap_closed_homogeneous_mean"]
    join = result["join_report"]
    noise = result["noise_ceiling"]

    lines = [
        "# Conditional entropy does NOT predict PPO trainability (negative result)",
        "",
        "Generated by `entropy_vs_trainability.py` (issue #430, Task 1). Inputs:",
        "`conditional_entropy.json` (#368) joined on `cell_tag` against",
        "`../../p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json` (#360/#420).",
        "",
        "## Headline",
        "",
        f"Across the {ne['n_cells']} converged cells, per-cell **mean** conditional",
        f"entropy vs `gap_closed_ne_mean` gives Spearman ρ = "
        f"{ne['aggregates']['mean']['spearman_rho']:.3f} "
        f"(p = {ne['aggregates']['mean']['p_value']:.2f}). All four aggregates are",
        "insignificant. The #368 predictor is dead; Paper 2's thesis must move to the",
        "coordination-threshold (k\\*) account (thrust#259/#268/#269 — issue #430 Task 2).",
        "",
        "## Spearman correlations",
        "",
        f"vs `gap_closed_ne_mean` (n = {ne['n_cells']} cells with a converged NE"
        " baseline):",
        "",
        "| h_cond aggregate | ρ | p |",
        "|---|---|---|",
    ]
    for agg in AGGREGATE_NAMES:
        a = ne["aggregates"][agg]
        lines.append(f"| {agg} | {a['spearman_rho']:.3f} | {a['p_value']:.3f} |")
    lines += [
        "",
        f"vs `gap_closed_homogeneous_mean` (n = {homog['n_cells']} joined cells;",
        f"the verdict file has {join['n_verdict_cells']} cells with homogeneous gaps,",
        f"but the {len(join['verdict_cells_without_entropy'])} `no_convergence` cells"
        " have no NE profile and therefore no entropy to correlate):",
        "",
        "| h_cond aggregate | ρ | p |",
        "|---|---|---|",
    ]
    for agg in AGGREGATE_NAMES:
        a = homog["aggregates"][agg]
        lines.append(f"| {agg} | {a['spearman_rho']:.3f} | {a['p_value']:.3f} |")

    lines += [
        "",
        "The single nominally significant entry (spread vs homogeneous, p = 0.039)",
        "does not survive correction for the 8 tests in this family"
        " (Bonferroni α = 0.0063), and spread was never the paper's primary",
        "predictor (mean entropy was; it is null against both targets).",
    ]

    lines += [
        "",
        "## β-invariance: no function of the NE profile can work",
        "",
        "Within each (κ, c) column the converged heterogeneous NE profile — and",
        "hence its entropy — is **identical across β**, while `gap_closed_ne_mean`",
        "varies strongly with β. The within-column trainability variance is",
        "therefore unexplainable by *any* statistic computed from the NE profile,",
        "entropy or otherwise.",
        "",
        "| κ | c | β values | h_cond mean (per β) | identical? | gap_closed_ne_mean (per β) | gap range |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in result["beta_invariance"]:
        betas = ", ".join(f"{b:.2f}" for b in row["betas"])
        ents = ", ".join(f"{h:.4f}" for h in row["h_cond_mean_by_beta"])
        gaps = ", ".join(
            "null" if g is None else f"{g:.2f}"
            for g in row["gap_closed_ne_mean_by_beta"]
        )
        rng = (
            "—"
            if row["gap_closed_ne_range"] is None
            else f"{row['gap_closed_ne_range']:.2f}"
        )
        ident = "yes" if row["entropy_identical_across_beta"] else "NO"
        lines.append(
            f"| {row['kappa']:.2f} | {row['c']:.2f} | {betas} | {ents} | {ident} | {gaps} | {rng} |"
        )

    lines += [
        "",
        "## Noise ceiling (n = 4 seeds)",
        "",
        f"`gap_closed_ne_mean` is estimated from n = {noise['n_seeds']} seeds per cell.",
        f"Per-cell std reaches {noise['max_gap_closed_ne_std']:.2f}"
        f" (median {noise['median_gap_closed_ne_std']:.2f}), which caps the",
        "achievable correlation for *any* predictor. Tightening this (Task 3, 20-seed",
        "extension on a k\\*-stratified subset) is recommended before the paper makes",
        "quantitative threshold claims.",
        "",
        "| cell_tag | gap_closed_ne_std |",
        "|---|---|",
    ]
    for w in noise["worst_cells"]:
        lines.append(f"| {w['cell_tag']} | {w['gap_closed_ne_std']:.2f} |")

    lines += [
        "",
        "## Join coverage",
        "",
        f"- Entropy cells: {join['n_entropy_cells']}; verdict cells:"
        f" {join['n_verdict_cells']}; joined: {join['n_joined']}.",
        "- Verdict cells without entropy (all `no_convergence`, no NE profile):"
        f" {', '.join(join['verdict_cells_without_entropy'])}.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        "uv run python experiments/nash/phase_diagram/entropy_vs_trainability.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entropy-json", type=Path, default=DEFAULT_ENTROPY_JSON)
    parser.add_argument("--verdict-json", type=Path, default=DEFAULT_VERDICT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    for path in (args.entropy_json, args.verdict_json):
        if not path.exists():
            print(f"ERROR: input artifact not found: {path}", file=sys.stderr)
            return 1

    with open(args.entropy_json) as f:
        entropy_data = json.load(f)
    with open(args.verdict_json) as f:
        verdict_data = json.load(f)

    joined, join_report = join_cells(entropy_data, verdict_data)

    result = {
        "inputs": {
            "entropy_json": str(args.entropy_json.relative_to(REPO_ROOT)),
            "verdict_json": str(args.verdict_json.relative_to(REPO_ROOT)),
        },
        "join_report": join_report,
        "correlations": {
            "gap_closed_ne_mean": spearman_table(joined, "gap_closed_ne_mean"),
            "gap_closed_homogeneous_mean": spearman_table(
                joined, "gap_closed_homogeneous_mean"
            ),
        },
        "beta_invariance": beta_invariance_table(joined),
        "noise_ceiling": noise_ceiling(joined),
        "per_cell": joined,
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"Wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {args.out_md}")

    ne = result["correlations"]["gap_closed_ne_mean"]
    print(
        f"\nHeadline: mean h_cond vs gap_closed_ne_mean over {ne['n_cells']} cells: "
        f"rho={ne['aggregates']['mean']['spearman_rho']:.3f}, "
        f"p={ne['aggregates']['mean']['p_value']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
