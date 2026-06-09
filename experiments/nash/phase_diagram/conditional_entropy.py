#!/usr/bin/env python3
"""Estimate per-cell, per-position conditional action entropy from
heterogeneous Nash solutions (issue #368).

For each cell in the phase diagram preview that has a converged
heterogeneous Nash profile, this script:

1. Loads the converged NE genomes from
   ``experiments/nash/phase_diagram/preview/<host>/cells/<tag>/results.json``.
2. Drives ``BucketBrigadeEnv`` step-by-step with 4 ``HeuristicAgent``
   instances seeded with those genomes for ``--n-episodes`` rollouts.
3. Tabulates per-position ``H(A_i^* | A_{-i}^*)`` (plug-in + Miller-Madow,
   in bits) with a bootstrap 95% CI that resamples *episodes*.
4. Writes:
   - ``experiments/nash/phase_diagram/conditional_entropy.json`` — full
     per-cell, per-position summary with metadata.
   - ``experiments/nash/phase_diagram/conditional_entropy.csv`` — tidy CSV
     with one row per ``(cell_tag, position)``.

Compute placement
-----------------
This rollout is CPU-bound and parallelises trivially across episodes via
``multiprocessing.Pool``. 10k episodes × ~50 steps × 4 agents × 7 cells is
bounded by Rust env throughput, well under one hour on a 32-thread alc-*
host. Do NOT run the full sweep on a laptop — see ``CLAUDE.md``.

For unit-test and smoke-test purposes the script accepts ``--n-episodes 50``
which completes in a few seconds.

Usage
-----
    # Smoke (fast, validates wiring)
    uv run python experiments/nash/phase_diagram/conditional_entropy.py \\
        --n-episodes 50 --n-boot 100 --num-workers 4

    # Production (full 10k episodes, 1k bootstrap)
    uv run python experiments/nash/phase_diagram/conditional_entropy.py \\
        --n-episodes 10000 --n-boot 1000
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Allow running from repo root or this directory.
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from bucket_brigade.analysis.conditional_entropy import (  # noqa: E402
    CellEntropyResult,
    estimate_cell_entropy,
)
from bucket_brigade.envs import get_scenario_by_name  # noqa: E402

PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"
PREVIEW_ROOT = PHASE_DIAGRAM_ROOT / "preview"
DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "conditional_entropy.json"
DEFAULT_OUT_CSV = PHASE_DIAGRAM_ROOT / "conditional_entropy.csv"

BASE_SCENARIO_NAME = "minimal_specialization"


def _discover_cells() -> list[dict]:
    """Find every per-cell results.json under preview/<host>/cells/.

    Dedup by cell tag — the same (β, κ, c) cell may exist under multiple
    host directories (alc-2 and alc-5-freqtest both contain b0.50_*).
    Prefer the host whose `results.json` has the converged equilibrium
    with the largest team payoff; tie-break by host name.
    """
    by_tag: dict[str, dict] = {}
    if not PREVIEW_ROOT.exists():
        return []
    for cell_path in PREVIEW_ROOT.glob("*/cells/*/results.json"):
        host = cell_path.parents[2].name
        tag = cell_path.parent.name
        try:
            with open(cell_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"[discover] WARN: failed to parse {cell_path}: {exc}",
                file=sys.stderr,
            )
            continue

        # Best converged team payoff in this file (for dedup tie-breaking).
        equilibria = data.get("equilibria", [])
        converged = [eq for eq in equilibria if eq.get("converged")]
        if converged:
            best_payoff = max(eq.get("team_payoff", float("-inf")) for eq in converged)
        else:
            best_payoff = float("-inf")

        entry = {
            "host": host,
            "tag": tag,
            "path": cell_path,
            "data": data,
            "best_converged_payoff": best_payoff,
        }
        if tag not in by_tag:
            by_tag[tag] = entry
        else:
            existing = by_tag[tag]
            # Prefer the file with a converged NE; if both have one, prefer
            # the higher team payoff; else lexicographic host name for
            # deterministic dedup.
            if best_payoff > existing["best_converged_payoff"]:
                by_tag[tag] = entry
            elif (
                best_payoff == existing["best_converged_payoff"]
                and host < existing["host"]
            ):
                by_tag[tag] = entry
    return sorted(by_tag.values(), key=lambda e: e["tag"])


def _select_best_equilibrium(cell_data: dict) -> Optional[dict]:
    """Pick the best converged equilibrium from a per-cell results.json.

    Mirrors the verdict logic in ``compute_nash_phase_diagram._classify_cell``:
    we want the NE that defines the cell's verdict. If a symmetric_only cell
    has both symmetric and asymmetric converged NE, we pick the one with the
    higher team payoff (which is the one classified as the cell's
    equilibrium).
    """
    equilibria = cell_data.get("equilibria", [])
    converged = [eq for eq in equilibria if eq.get("converged")]
    if not converged:
        return None
    return max(converged, key=lambda eq: eq.get("team_payoff", float("-inf")))


def _genomes_from_equilibrium(eq: dict) -> list[list[float]]:
    """Extract length-10 float genomes from an equilibrium dict."""
    profile = eq.get("strategy_profile", [])
    # Sort by position to be safe (the JSON usually has them in order).
    profile_sorted = sorted(profile, key=lambda p: p.get("position", 0))
    return [list(p["genome"]) for p in profile_sorted]


def _build_scenario(beta: float, kappa: float, c: float):
    base = get_scenario_by_name(BASE_SCENARIO_NAME, num_agents=4)
    return dataclasses.replace(
        base,
        prob_fire_spreads_to_neighbor=float(beta),
        prob_solo_agent_extinguishes_fire=float(kappa),
        cost_to_work_one_night=float(c),
    )


def _result_to_dict(res: CellEntropyResult) -> dict:
    return {
        "cell_tag": res.cell_tag,
        "beta": res.beta,
        "kappa": res.kappa,
        "c": res.c,
        "verdict": res.verdict,
        "n_episodes": res.n_episodes,
        "n_steps_total": res.n_steps_total,
        "positions": [
            {
                "position": p.position,
                "h_joint": p.h_joint,
                "h_minus_i": p.h_minus_i,
                "h_cond": p.h_cond,
                "h_cond_ci_lo": p.h_cond_ci_lo,
                "h_cond_ci_hi": p.h_cond_ci_hi,
            }
            for p in res.positions
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=10_000,
        help="Number of rollouts per cell. Default: 10000.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Bootstrap resamples for the 95%% CI. Default: 1000.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base RNG seed.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Pool size for parallel rollouts. 0 (default) -> cpu_count(); "
        "set to 1 to disable multiprocessing.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_OUT_JSON,
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=DEFAULT_OUT_CSV,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--cell-filter",
        type=str,
        default=None,
        help="Optional substring to filter cell tags (smoke testing).",
    )
    args = parser.parse_args()

    num_workers = args.num_workers if args.num_workers > 0 else None

    discovered = _discover_cells()
    if args.cell_filter:
        discovered = [c for c in discovered if args.cell_filter in c["tag"]]

    print(f"Discovered {len(discovered)} unique cells under {PREVIEW_ROOT}")

    cell_results: list[CellEntropyResult] = []
    skipped_cells: list[dict] = []

    t_total = time.time()
    for cell in discovered:
        tag = cell["tag"]
        host = cell["host"]
        data = cell["data"]

        swept = data.get("swept_parameters", {})
        beta = float(swept.get("beta", float("nan")))
        kappa = float(swept.get("kappa", float("nan")))
        c = float(swept.get("c", float("nan")))
        verdict = data.get("summary", {}).get("verdict", "unknown")

        if verdict == "no_convergence":
            skipped_cells.append(
                {
                    "cell_tag": tag,
                    "host": host,
                    "beta": beta,
                    "kappa": kappa,
                    "c": c,
                    "verdict": verdict,
                    "reason": "No converged NE in this cell's results.json.",
                }
            )
            print(f"[{tag}] SKIP: verdict={verdict}")
            continue

        eq = _select_best_equilibrium(data)
        if eq is None:
            skipped_cells.append(
                {
                    "cell_tag": tag,
                    "host": host,
                    "beta": beta,
                    "kappa": kappa,
                    "c": c,
                    "verdict": verdict,
                    "reason": "Verdict not 'no_convergence' but no converged equilibrium present.",
                }
            )
            print(f"[{tag}] SKIP: no converged equilibrium found")
            continue

        genomes = _genomes_from_equilibrium(eq)
        if len(genomes) != 4:
            skipped_cells.append(
                {
                    "cell_tag": tag,
                    "host": host,
                    "beta": beta,
                    "kappa": kappa,
                    "c": c,
                    "verdict": verdict,
                    "reason": f"Expected 4 positions, got {len(genomes)}.",
                }
            )
            print(f"[{tag}] SKIP: only {len(genomes)} positions in strategy_profile")
            continue

        scenario = _build_scenario(beta, kappa, c)

        t_cell = time.time()
        print(
            f"[{tag}] verdict={verdict} host={host} "
            f"rollout n_episodes={args.n_episodes} ..."
        )
        result = estimate_cell_entropy(
            genomes=genomes,
            scenario=scenario,
            n_episodes=args.n_episodes,
            n_boot=args.n_boot,
            seed=args.seed,
            num_workers=num_workers,
            cell_tag=tag,
            beta=beta,
            kappa=kappa,
            c=c,
            verdict=verdict,
        )
        elapsed = time.time() - t_cell
        h_cond_values = [p.h_cond for p in result.positions]
        print(
            f"[{tag}] done in {elapsed:.1f}s; n_steps_total={result.n_steps_total} "
            f"h_cond per position (bits): "
            f"{['{:.3f}'.format(v) for v in h_cond_values]}"
        )
        cell_results.append(result)

    total_elapsed = time.time() - t_total
    print(f"All cells done in {total_elapsed:.1f}s")

    # Write JSON.
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_episodes": args.n_episodes,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_cells_processed": len(cell_results),
        "n_cells_skipped": len(skipped_cells),
        "skipped_cells": skipped_cells,
        "units": "bits (Miller-Madow corrected plug-in)",
        "ci_method": "episode-level bootstrap, 95%",
        "cells": [_result_to_dict(r) for r in cell_results],
        "elapsed_seconds": total_elapsed,
    }
    with open(args.out_json, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {args.out_json}")

    # Write CSV (one row per cell × position).
    csv_columns = [
        "cell_tag",
        "beta",
        "kappa",
        "c",
        "position",
        "h_joint",
        "h_minus_i",
        "h_cond",
        "h_cond_ci_lo",
        "h_cond_ci_hi",
        "n_episodes",
        "verdict",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        for r in cell_results:
            for p in r.positions:
                writer.writerow(
                    {
                        "cell_tag": r.cell_tag,
                        "beta": r.beta,
                        "kappa": r.kappa,
                        "c": r.c,
                        "position": p.position,
                        "h_joint": p.h_joint,
                        "h_minus_i": p.h_minus_i,
                        "h_cond": p.h_cond,
                        "h_cond_ci_lo": p.h_cond_ci_lo,
                        "h_cond_ci_hi": p.h_cond_ci_hi,
                        "n_episodes": r.n_episodes,
                        "verdict": r.verdict,
                    }
                )
    print(f"Wrote {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
