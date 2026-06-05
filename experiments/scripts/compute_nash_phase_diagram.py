#!/usr/bin/env python3
"""
Compute a heterogeneous Nash phase diagram across the (β, κ, c) parameter grid.

Extends ``compute_nash_heterogeneous.py`` from a single scenario to a
parameter sweep over:

  - β = ``prob_fire_spreads_to_neighbor`` (spread rate)
  - κ = ``prob_solo_agent_extinguishes_fire`` (single-agent extinguish prob)
  - c = ``cost_to_work_one_night`` (work cost)

Per cell we run the same Heterogeneous Double Oracle search as
``compute_nash_heterogeneous.py`` and classify the converged equilibrium
profile into one of four verdicts:

  - ``symmetric_only`` — all restarts converge to symmetric profiles
  - ``asymmetric_only`` — converged asymmetric NE dominates any symmetric NE
                          by more than ``epsilon`` team payoff
  - ``mixed`` — both symmetric and asymmetric equilibria exist with comparable
                team payoff (within ``epsilon``)
  - ``no_convergence`` — no restart converged within ``max_iterations``

The grid base scenario uses the ``minimal_specialization`` reward structure
(per-agent ownership signal dominant, team signal reduced) — this is the
scenario family the asymmetric-NE search was calibrated on (#353, #355).
Only the three swept fields are varied per cell.

Output
------
    experiments/nash/phase_diagram/<tag>/
        results.json          — aggregated per-cell summaries + grid axes
        cells/<beta>_<kappa>_<c>/  — full per-cell heterogeneous-DO artifacts
            results.json
            summary.md
            summary.json
        summary.md            — top-level human-readable verdict table

Compute (CRITICAL — do NOT run locally)
----------------------------------------
Per-cell cost at the full 20 restarts × 25 iter × 1000 sims × ε=50
calibration is ~5–7 h on alc-9 (see ``memory/nash_heterogeneous_calibration.md``).
The full 5×5×3 = 75-cell grid is therefore ~450 h wall-clock on a single
host. The issue recommends a 3×3×2 = 18-cell preview first.

This script ships with a ``--smoke`` mode that runs a 1×1×1 = 1-cell grid
at trivially small budget (restarts=1, max_iter=1, simulations=20) so the
driver wiring can be smoke-tested locally in a few minutes. Real runs MUST
be launched on a remote host (alc-* / Mac Studio); see CLAUDE.md.

Usage
-----
    # Local smoke (a few minutes, validates wiring only — verdicts meaningless)
    uv run python experiments/scripts/compute_nash_phase_diagram.py --smoke

    # Preview grid on remote (3×3×2 = 18 cells, ~108h on alc-9)
    uv run python experiments/scripts/compute_nash_phase_diagram.py --preview

    # Full grid on remote (5×5×3 = 75 cells, ~450h on alc-9)
    uv run python experiments/scripts/compute_nash_phase_diagram.py

    # Custom grid + budget
    uv run python experiments/scripts/compute_nash_phase_diagram.py \\
        --beta-values 0.1,0.5,0.9 --kappa-values 0.1,0.5,0.9 \\
        --c-values 0.5,2.0 --restarts 10 --simulations 500

Resumability
------------
Per-cell artifacts are written to ``cells/<beta>_<kappa>_<c>/`` and the
driver skips cells whose ``summary.json`` already exists unless ``--force``
is passed. This lets long sweeps survive interruption and lets you fan-out
across hosts by running the same command with overlapping but disjoint
``--beta-values`` slices.
"""

from __future__ import annotations

import sys
import json
import time
import argparse
import dataclasses
from pathlib import Path

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _disk_precheck import DEFAULT_MIN_FREE_MIB, check_free_space
from bucket_brigade.envs import get_scenario_by_name
from bucket_brigade.equilibrium.double_oracle_heterogeneous import (
    HeterogeneousDoubleOracle,
    HeterogeneousNashEquilibrium,
)

# Re-use the labelling, classification, and serialisation helpers from the
# single-scenario driver so per-cell artifacts have identical schema.
from compute_nash_heterogeneous import (  # type: ignore[import-not-found]
    _eq_to_dict,
    _profile_is_symmetric,
    _profile_label,
)


# ---------------------------------------------------------------------------
# Grid defaults (per issue #358 "Concrete scope" section)
# ---------------------------------------------------------------------------

FULL_BETA_VALUES: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
FULL_KAPPA_VALUES: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 0.9)
FULL_C_VALUES: tuple[float, ...] = (0.5, 1.0, 2.0)

PREVIEW_BETA_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)
PREVIEW_KAPPA_VALUES: tuple[float, ...] = (0.1, 0.5, 0.9)
PREVIEW_C_VALUES: tuple[float, ...] = (0.5, 2.0)

# Per-cell calibrated defaults (#353 / nash_heterogeneous_calibration.md)
DEFAULT_RESTARTS = 20
DEFAULT_SIMULATIONS = 1000
DEFAULT_OPT_SIMULATIONS = 300
DEFAULT_MAX_ITER = 25
DEFAULT_EPSILON = 50.0

# Base scenario family — minimal_specialization reward structure, just
# vary β/κ/c. This is the asymmetric-NE-relevant family from #353/#355.
BASE_SCENARIO_NAME = "minimal_specialization"


# ---------------------------------------------------------------------------
# Verdict classification
# ---------------------------------------------------------------------------


def _classify_cell(
    equilibria: list[HeterogeneousNashEquilibrium],
    epsilon: float,
) -> tuple[str, str]:
    """Map a list of restart equilibria to one of four phase-diagram verdicts.

    Returns ``(verdict, detail)``.
    """
    converged = [e for e in equilibria if e.converged]
    if not converged:
        return (
            "no_convergence",
            (
                f"No restart converged within max_iterations "
                f"({len(equilibria)} attempted). Likely needs more iterations "
                "or a tighter epsilon. Compute budget?"
            ),
        )

    symmetric = [e for e in converged if _profile_is_symmetric(e.strategy_profile)]
    asymmetric = [e for e in converged if not _profile_is_symmetric(e.strategy_profile)]

    sym_best = max((e.team_payoff for e in symmetric), default=None)
    asym_best = max((e.team_payoff for e in asymmetric), default=None)

    if asym_best is None:
        return (
            "symmetric_only",
            (
                f"All {len(converged)} converged restarts produced symmetric "
                f"profiles; best symmetric team payoff = {sym_best:.2f}."
            ),
        )
    if sym_best is None:
        return (
            "asymmetric_only",
            (
                f"All {len(converged)} converged restarts produced asymmetric "
                f"profiles; best asymmetric team payoff = {asym_best:.2f}."
            ),
        )

    # Both present: compare by team payoff, epsilon-tolerance.
    gap = asym_best - sym_best
    if gap > epsilon:
        return (
            "asymmetric_only",
            (
                f"Asymmetric NE ({asym_best:.2f}) dominates best symmetric NE "
                f"({sym_best:.2f}) by gap={gap:.2f} > epsilon={epsilon}."
            ),
        )
    if gap < -epsilon:
        return (
            "symmetric_only",
            (
                f"Symmetric NE ({sym_best:.2f}) dominates best asymmetric NE "
                f"({asym_best:.2f}) by gap={-gap:.2f} > epsilon={epsilon}."
            ),
        )
    return (
        "mixed",
        (
            f"Symmetric NE ({sym_best:.2f}) and asymmetric NE ({asym_best:.2f}) "
            f"both converge within epsilon={epsilon} (|gap|={abs(gap):.2f})."
        ),
    )


# ---------------------------------------------------------------------------
# Per-cell driver
# ---------------------------------------------------------------------------


def _cell_tag(beta: float, kappa: float, c: float) -> str:
    """Filesystem-safe directory tag for one (β,κ,c) cell."""
    return f"b{beta:.2f}_k{kappa:.2f}_c{c:.2f}"


def _make_scenario(beta: float, kappa: float, c: float, num_agents: int):
    """Return a Scenario from the base family with (β, κ, c) overridden."""
    base = get_scenario_by_name(BASE_SCENARIO_NAME, num_agents=num_agents)
    # dataclasses.replace requires the swept fields to remain valid types.
    return dataclasses.replace(
        base,
        prob_fire_spreads_to_neighbor=float(beta),
        prob_solo_agent_extinguishes_fire=float(kappa),
        cost_to_work_one_night=float(c),
    )


def _run_one_cell(
    beta: float,
    kappa: float,
    c: float,
    cell_dir: Path,
    num_simulations: int,
    opt_simulations: int,
    max_iterations: int,
    epsilon: float,
    num_restarts: int,
    seed: int,
    verbose: bool,
) -> dict:
    """Run heterogeneous DO on one (β, κ, c) cell. Returns the cell summary dict."""
    scenario = _make_scenario(beta, kappa, c, num_agents=4)
    cell_dir.mkdir(parents=True, exist_ok=True)

    solver = HeterogeneousDoubleOracle(
        scenario=scenario,
        num_simulations=num_simulations,
        opt_simulations=opt_simulations,
        max_iterations=max_iterations,
        epsilon=epsilon,
        seed=seed,
        num_restarts=num_restarts,
        verbose=verbose,
    )

    t0 = time.time()
    equilibria = solver.solve()
    elapsed = time.time() - t0

    converged = [e for e in equilibria if e.converged]
    symmetric = [e for e in equilibria if _profile_is_symmetric(e.strategy_profile)]
    asymmetric = [
        e for e in equilibria if not _profile_is_symmetric(e.strategy_profile)
    ]
    best = max(equilibria, key=lambda e: e.team_payoff) if equilibria else None
    best_asym = max(asymmetric, key=lambda e: e.team_payoff) if asymmetric else None
    best_sym = max(symmetric, key=lambda e: e.team_payoff) if symmetric else None

    verdict, verdict_detail = _classify_cell(equilibria, epsilon=epsilon)

    # Per-cell full artifact (same schema as compute_nash_heterogeneous.py
    # for downstream re-use).
    cell_payload = {
        "scenario": f"{BASE_SCENARIO_NAME}@beta={beta},kappa={kappa},c={c}",
        "algorithm": "heterogeneous_double_oracle",
        "swept_parameters": {"beta": beta, "kappa": kappa, "c": c},
        "parameters": {
            "num_simulations": num_simulations,
            "opt_simulations": opt_simulations,
            "max_iterations": max_iterations,
            "epsilon": epsilon,
            "num_restarts": num_restarts,
            "seed": seed,
        },
        "timing": {"elapsed_seconds": elapsed},
        "summary": {
            "total_restarts": len(equilibria),
            "converged": len(converged),
            "symmetric_profiles": len(symmetric),
            "asymmetric_profiles": len(asymmetric),
            "best_team_payoff": float(best.team_payoff) if best else None,
            "best_team_payoff_is_symmetric": (
                _profile_is_symmetric(best.strategy_profile) if best else None
            ),
            "verdict": verdict,
        },
        "equilibria": [_eq_to_dict(e) for e in equilibria],
    }

    with open(cell_dir / "results.json", "w") as f:
        json.dump(cell_payload, f, indent=2)

    cell_summary = {
        "beta": beta,
        "kappa": kappa,
        "c": c,
        "tag": _cell_tag(beta, kappa, c),
        "elapsed_seconds": elapsed,
        "total_restarts": len(equilibria),
        "converged": len(converged),
        "symmetric_profiles": len(symmetric),
        "asymmetric_profiles": len(asymmetric),
        "best_team_payoff": float(best.team_payoff) if best else None,
        "best_symmetric_team_payoff": (
            float(best_sym.team_payoff) if best_sym else None
        ),
        "best_asymmetric_team_payoff": (
            float(best_asym.team_payoff) if best_asym else None
        ),
        "best_asymmetric_profile_label": (
            _profile_label(best_asym.strategy_profile) if best_asym else None
        ),
        "verdict": verdict,
        "verdict_detail": verdict_detail,
    }

    with open(cell_dir / "summary.json", "w") as f:
        json.dump(cell_summary, f, indent=2)

    return cell_summary


# ---------------------------------------------------------------------------
# Top-level grid driver
# ---------------------------------------------------------------------------


def _parse_value_list(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def compute_phase_diagram(
    output_dir: Path,
    beta_values: list[float],
    kappa_values: list[float],
    c_values: list[float],
    num_simulations: int,
    opt_simulations: int,
    max_iterations: int,
    epsilon: float,
    num_restarts: int,
    seed: int,
    verbose: bool,
    force: bool,
) -> None:
    cells_dir = output_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)

    total = len(beta_values) * len(kappa_values) * len(c_values)
    print("=" * 70)
    print("Heterogeneous Nash Phase Diagram")
    print("=" * 70)
    print(f"Output dir:      {output_dir}")
    print(f"Base scenario:   {BASE_SCENARIO_NAME}")
    print(
        f"Grid:            {len(beta_values)}×{len(kappa_values)}×{len(c_values)} = {total} cells"
    )
    print(f"  β values:      {beta_values}")
    print(f"  κ values:      {kappa_values}")
    print(f"  c values:      {c_values}")
    print("Per-cell DO budget:")
    print(
        f"  restarts={num_restarts}, max_iter={max_iterations}, "
        f"simulations={num_simulations} (opt={opt_simulations}), ε={epsilon}"
    )
    print()

    grid_t0 = time.time()
    cell_summaries: list[dict] = []
    skipped: list[str] = []

    for ci, c in enumerate(c_values):
        for ki, kappa in enumerate(kappa_values):
            for bi, beta in enumerate(beta_values):
                tag = _cell_tag(beta, kappa, c)
                cell_dir = cells_dir / tag
                idx = (
                    ci * len(beta_values) * len(kappa_values)
                    + ki * len(beta_values)
                    + bi
                    + 1
                )
                cell_summary_path = cell_dir / "summary.json"

                if cell_summary_path.exists() and not force:
                    print(f"[{idx}/{total}] {tag}: cached (use --force to recompute)")
                    with open(cell_summary_path) as f:
                        cell_summaries.append(json.load(f))
                    skipped.append(tag)
                    continue

                print(
                    f"[{idx}/{total}] {tag}: β={beta:.2f} κ={kappa:.2f} c={c:.2f} ..."
                )
                # Use a deterministic per-cell seed derived from the master
                # seed and the cell coordinates so resuming or re-running a
                # subset is reproducible.
                cell_seed = seed + bi * 10007 + ki * 1009 + ci * 101
                cell_summary = _run_one_cell(
                    beta=beta,
                    kappa=kappa,
                    c=c,
                    cell_dir=cell_dir,
                    num_simulations=num_simulations,
                    opt_simulations=opt_simulations,
                    max_iterations=max_iterations,
                    epsilon=epsilon,
                    num_restarts=num_restarts,
                    seed=cell_seed,
                    verbose=verbose,
                )
                cell_summaries.append(cell_summary)
                print(
                    f"    -> verdict={cell_summary['verdict']} "
                    f"best_payoff={cell_summary['best_team_payoff']} "
                    f"elapsed={cell_summary['elapsed_seconds']:.1f}s"
                )

    grid_elapsed = time.time() - grid_t0

    # --- Aggregate ---
    verdict_counts: dict[str, int] = {}
    for cs in cell_summaries:
        verdict_counts[cs["verdict"]] = verdict_counts.get(cs["verdict"], 0) + 1

    aggregate = {
        "base_scenario": BASE_SCENARIO_NAME,
        "algorithm": "heterogeneous_double_oracle",
        "grid": {
            "beta_values": beta_values,
            "kappa_values": kappa_values,
            "c_values": c_values,
            "total_cells": total,
        },
        "parameters": {
            "num_simulations": num_simulations,
            "opt_simulations": opt_simulations,
            "max_iterations": max_iterations,
            "epsilon": epsilon,
            "num_restarts": num_restarts,
            "seed": seed,
        },
        "timing": {
            "elapsed_seconds": grid_elapsed,
            "skipped_cached": skipped,
        },
        "verdict_counts": verdict_counts,
        "cells": cell_summaries,
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(aggregate, f, indent=2)

    # --- Markdown summary (compact verdict table per c-slice) ---
    lines = [
        "# Heterogeneous Nash Phase Diagram\n",
        f"Base scenario: `{BASE_SCENARIO_NAME}`\n",
        f"Grid: {len(beta_values)}×{len(kappa_values)}×{len(c_values)} = {total} cells\n",
        f"Per-cell budget: restarts={num_restarts}, max_iter={max_iterations}, "
        f"simulations={num_simulations}, ε={epsilon}\n",
        f"Grid elapsed: {grid_elapsed:.0f}s ({grid_elapsed / 3600:.2f}h)\n",
        "## Verdict counts\n",
    ]
    for v, n in sorted(verdict_counts.items()):
        lines.append(f"- `{v}`: {n}\n")

    # Index cell summaries by tag for quick lookup.
    by_tag = {cs["tag"]: cs for cs in cell_summaries}

    for c in c_values:
        lines.append(f"\n## c = {c}\n")
        # Header row: kappa values
        header = (
            "| β \\\\ κ | "
            + " | ".join(f"{kappa:.2f}" for kappa in kappa_values)
            + " |\n"
        )
        sep = "|---" * (len(kappa_values) + 1) + "|\n"
        lines.append(header)
        lines.append(sep)
        for beta in beta_values:
            row = [f"{beta:.2f}"]
            for kappa in kappa_values:
                tag = _cell_tag(beta, kappa, c)
                cs = by_tag.get(tag)
                row.append(cs["verdict"] if cs else "—")
            lines.append("| " + " | ".join(row) + " |\n")

    with open(output_dir / "summary.md", "w") as f:
        f.writelines(lines)

    print()
    print("=" * 70)
    print(f"Grid complete in {grid_elapsed / 3600:.2f}h ({grid_elapsed:.0f}s)")
    print(f"Cells:          {total} ({len(skipped)} cached)")
    print(f"Verdicts:       {verdict_counts}")
    print(f"Aggregate JSON → {results_path}")
    print(f"Summary MD     → {output_dir / 'summary.md'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute heterogeneous Nash phase diagram across (β, κ, c) grid"
    )
    grid_group = parser.add_mutually_exclusive_group()
    grid_group.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Local smoke test: 1×1×1 grid (single mid-range cell), trivially small "
            "compute budget. Validates driver wiring only — verdicts are meaningless."
        ),
    )
    grid_group.add_argument(
        "--preview",
        action="store_true",
        help=("Preview grid: 3×3×2 = 18 cells (~108h on alc-9). REMOTE ONLY."),
    )
    parser.add_argument(
        "--beta-values",
        type=str,
        default=None,
        help="Comma-separated β values (overrides --smoke/--preview default)",
    )
    parser.add_argument(
        "--kappa-values",
        type=str,
        default=None,
        help="Comma-separated κ values (overrides --smoke/--preview default)",
    )
    parser.add_argument(
        "--c-values",
        type=str,
        default=None,
        help="Comma-separated c values (overrides --smoke/--preview default)",
    )
    parser.add_argument(
        "--restarts",
        type=int,
        default=None,
        help=f"Per-cell restarts (default: {DEFAULT_RESTARTS}, smoke: 1)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=None,
        help=f"MC simulations per payoff estimate (default: {DEFAULT_SIMULATIONS}, smoke: 20)",
    )
    parser.add_argument(
        "--opt-simulations",
        type=int,
        default=None,
        help=f"Cheaper simulations during gradient search (default: {DEFAULT_OPT_SIMULATIONS}, smoke: 10)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help=f"BR rounds per restart (default: {DEFAULT_MAX_ITER}, smoke: 1)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=DEFAULT_EPSILON,
        help=f"Payoff comparison epsilon (default: {DEFAULT_EPSILON})",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/nash/phase_diagram[/smoke])",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="Minimum free disk space in MiB before aborting",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute cells even if cached summary.json exists",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-position DO logs"
    )

    args = parser.parse_args()

    # --- Resolve grid axes ---
    if args.smoke:
        default_betas = [0.5]
        default_kappas = [0.5]
        default_cs = [0.5]
    elif args.preview:
        default_betas = list(PREVIEW_BETA_VALUES)
        default_kappas = list(PREVIEW_KAPPA_VALUES)
        default_cs = list(PREVIEW_C_VALUES)
    else:
        default_betas = list(FULL_BETA_VALUES)
        default_kappas = list(FULL_KAPPA_VALUES)
        default_cs = list(FULL_C_VALUES)

    beta_values = (
        _parse_value_list(args.beta_values) if args.beta_values else default_betas
    )
    kappa_values = (
        _parse_value_list(args.kappa_values) if args.kappa_values else default_kappas
    )
    c_values = _parse_value_list(args.c_values) if args.c_values else default_cs

    # --- Resolve compute budget ---
    if args.smoke:
        num_restarts = args.restarts if args.restarts is not None else 1
        num_simulations = args.simulations if args.simulations is not None else 20
        opt_simulations = (
            args.opt_simulations if args.opt_simulations is not None else 10
        )
        max_iterations = args.max_iterations if args.max_iterations is not None else 1
    else:
        num_restarts = args.restarts if args.restarts is not None else DEFAULT_RESTARTS
        num_simulations = (
            args.simulations if args.simulations is not None else DEFAULT_SIMULATIONS
        )
        opt_simulations = (
            args.opt_simulations
            if args.opt_simulations is not None
            else DEFAULT_OPT_SIMULATIONS
        )
        max_iterations = (
            args.max_iterations if args.max_iterations is not None else DEFAULT_MAX_ITER
        )

    if args.output_dir is None:
        if args.smoke:
            args.output_dir = Path("experiments/nash/phase_diagram/smoke")
        elif args.preview:
            args.output_dir = Path("experiments/nash/phase_diagram/preview")
        else:
            args.output_dir = Path("experiments/nash/phase_diagram")

    check_free_space(args.output_dir, min_free_mib=args.min_free_mib)

    compute_phase_diagram(
        output_dir=args.output_dir,
        beta_values=beta_values,
        kappa_values=kappa_values,
        c_values=c_values,
        num_simulations=num_simulations,
        opt_simulations=opt_simulations,
        max_iterations=max_iterations,
        epsilon=args.epsilon,
        num_restarts=num_restarts,
        seed=args.seed,
        verbose=not args.quiet,
        force=args.force,
    )


if __name__ == "__main__":
    main()
