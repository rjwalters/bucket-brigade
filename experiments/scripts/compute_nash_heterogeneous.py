#!/usr/bin/env python3
"""
Compute asymmetric Nash equilibria via Heterogeneous Double Oracle.

Unlike compute_nash.py / compute_nash_v2.py (which find *symmetric* equilibria
where all agents play the same strategy), this script searches for equilibria
where different agent positions can play different strategies — the type needed
to confirm or rule out role-differentiated specialisation equilibria.

Primary use-case: minimal_specialization (issue #353).
Secondary use-case: rest_trap cycling diagnosis (issue #352 / #353).

Usage
-----
    # Minimal specialisation — default settings, run remotely
    uv run python experiments/scripts/compute_nash_heterogeneous.py minimal_specialization

    # More restarts, more simulations, quiet output (for overnight runs)
    uv run python experiments/scripts/compute_nash_heterogeneous.py minimal_specialization \\
        --restarts 30 --simulations 1500 --quiet

    # rest_trap (the cycling scenario)
    uv run python experiments/scripts/compute_nash_heterogeneous.py rest_trap \\
        --restarts 20 --simulations 1000

Output
------
    experiments/nash/heterogeneous/<scenario>/
        results.json          — full equilibrium list (all restarts)
        summary.md            — human-readable verdict
        summary.json          — machine-readable summary
"""

from __future__ import annotations

import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np

# Allow running from repo root or scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _disk_precheck import DEFAULT_MIN_FREE_MIB, check_free_space
from bucket_brigade.envs import get_scenario_by_name
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)
from bucket_brigade.equilibrium.double_oracle_heterogeneous import (
    HeterogeneousDoubleOracle,
    HeterogeneousNashEquilibrium,
)


# ---------------------------------------------------------------------------
# Strategy labelling helpers
# ---------------------------------------------------------------------------

ARCHETYPE_POOL = {
    "firefighter": FIREFIGHTER_PARAMS,
    "free_rider": FREE_RIDER_PARAMS,
    "hero": HERO_PARAMS,
    "coordinator": COORDINATOR_PARAMS,
    "liar": LIAR_PARAMS,
}

PARAM_NAMES = [
    "honesty",
    "work_tendency",
    "neighbor_help",
    "own_priority",
    "risk_aversion",
    "coordination",
    "exploration",
    "fatigue_memory",
    "rest_bias",
    "altruism",
]


def _classify(theta: np.ndarray) -> str:
    """Return the name of the closest archetype by L2 distance."""
    best_name, best_dist = "unknown", np.inf
    for name, params in ARCHETYPE_POOL.items():
        d = float(np.linalg.norm(theta - params))
        if d < best_dist:
            best_dist = d
            best_name = name
    return f"{best_name}(d={best_dist:.2f})"


def _profile_label(profile: list[np.ndarray]) -> str:
    return " | ".join(_classify(t) for t in profile)


def _profile_is_symmetric(profile: list[np.ndarray], atol: float = 0.05) -> bool:
    """True if all 4 positions play effectively the same strategy."""
    return all(np.allclose(profile[0], t, atol=atol) for t in profile[1:])


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _eq_to_dict(eq: HeterogeneousNashEquilibrium) -> dict:
    return {
        "converged": eq.converged,
        "iterations": eq.iterations,
        "team_payoff": eq.team_payoff,
        "per_position_payoffs": [float(p) for p in eq.payoffs],
        "symmetric_profile": _profile_is_symmetric(eq.strategy_profile),
        "profile_label": _profile_label(eq.strategy_profile),
        "strategy_profile": [
            {
                "position": i,
                "closest_archetype": _classify(t),
                "parameters": {n: float(v) for n, v in zip(PARAM_NAMES, t)},
                "genome": [float(v) for v in t],
            }
            for i, t in enumerate(eq.strategy_profile)
        ],
        "start_profile_label": _profile_label(eq.start_profile),
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------


def compute_heterogeneous_nash(
    scenario_name: str,
    output_dir: Path,
    num_simulations: int,
    opt_simulations: int,
    max_iterations: int,
    epsilon: float,
    num_restarts: int,
    seed: int,
    verbose: bool,
) -> None:
    print("=" * 70)
    print("Heterogeneous Nash Equilibrium — Double Oracle (asymmetric)")
    print("=" * 70)
    print(f"Scenario:      {scenario_name}")
    print(f"Output dir:    {output_dir}")
    print(f"Restarts:      {num_restarts}")
    print(f"Simulations:   {num_simulations} (opt: {opt_simulations})")
    print(f"Max iter:      {max_iterations}")
    print(f"Epsilon:       {epsilon}")
    print(f"Seed:          {seed}")
    print()

    scenario = get_scenario_by_name(scenario_name, num_agents=4)
    print("Scenario parameters:")
    print(f"  β (spread):      {scenario.prob_fire_spreads_to_neighbor:.2f}")
    print(f"  κ (extinguish):  {scenario.prob_solo_agent_extinguishes_fire:.2f}")
    print(f"  c (work cost):   {scenario.cost_to_work_one_night:.2f}")
    print(f"  num_agents:      {scenario.num_agents}")
    print()

    output_dir.mkdir(parents=True, exist_ok=True)

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

    print("\n" + "=" * 70)
    print(f"Run complete in {elapsed / 3600:.2f}h ({elapsed:.0f}s)")
    print(f"Restarts completed: {len(equilibria)}")
    converged = [e for e in equilibria if e.converged]
    print(f"Converged:          {len(converged)}/{len(equilibria)}")
    print()

    # --- Analyse results ---
    symmetric = [e for e in equilibria if _profile_is_symmetric(e.strategy_profile)]
    asymmetric = [
        e for e in equilibria if not _profile_is_symmetric(e.strategy_profile)
    ]

    # Partition converged equilibria by symmetry up front — both the reporting
    # block below and the verdict block further down reference these.
    converged_asymmetric = [e for e in asymmetric if e.converged]
    converged_symmetric = [e for e in symmetric if e.converged]

    print(f"Symmetric profiles:   {len(symmetric)}")
    print(f"Asymmetric profiles:  {len(asymmetric)}")
    print()

    if equilibria:
        best = max(equilibria, key=lambda e: e.team_payoff)
        print(f"Best team payoff:  {best.team_payoff:.2f}")
        print(f"Profile:           {_profile_label(best.strategy_profile)}")
        print(f"Symmetric:         {_profile_is_symmetric(best.strategy_profile)}")
        print()

    if converged_asymmetric:
        _best_print = max(converged_asymmetric, key=lambda e: e.team_payoff)
        print(
            f"Best converged ASYMMETRIC equilibrium ({len(converged_asymmetric)} total converged):"
        )
        print(f"  Team payoff:  {_best_print.team_payoff:.2f}")
        print(f"  Profile:      {_profile_label(_best_print.strategy_profile)}")
        print(f"  Iterations:   {_best_print.iterations}")
        for i, (t, p) in enumerate(
            zip(_best_print.strategy_profile, _best_print.payoffs)
        ):
            print(f"  Pos {i}: payoff={p:.1f}  {_classify(t)}")
            print(f"         work_tendency={t[1]:.3f}  honesty={t[0]:.3f}")
        print()
    elif asymmetric:
        best_asym = max(asymmetric, key=lambda e: e.team_payoff)
        print("Best ASYMMETRIC profile (not converged):")
        print(f"  Team payoff:  {best_asym.team_payoff:.2f}")
        print(f"  Profile:      {_profile_label(best_asym.strategy_profile)}")
        print(f"  Converged:    {best_asym.converged}")
        for i, (t, p) in enumerate(zip(best_asym.strategy_profile, best_asym.payoffs)):
            print(f"  Pos {i}: payoff={p:.1f}  {_classify(t)}")
            print(f"         work_tendency={t[1]:.3f}  honesty={t[0]:.3f}")
        print()
    else:
        print(
            "No asymmetric equilibria found — all restarts converged to symmetric profiles."
        )
        print()

    # --- Determine verdict ---
    # Use best *converged* equilibria for the verdict. A non-converged profile
    # having a higher payoff doesn't mean converged equilibria don't exist.
    # (converged_asymmetric / converged_symmetric were partitioned above so the
    # reporting block could reference them.)
    best_conv_asym = (
        max(converged_asymmetric, key=lambda e: e.team_payoff)
        if converged_asymmetric
        else None
    )
    best_conv_sym = (
        max(converged_symmetric, key=lambda e: e.team_payoff)
        if converged_symmetric
        else None
    )
    sym_best = best_conv_sym.team_payoff if best_conv_sym else None

    if best_conv_asym is not None and sym_best is not None:
        if best_conv_asym.team_payoff > sym_best + abs(epsilon):
            verdict = "asymmetric_ne_superior"
            verdict_detail = (
                f"Asymmetric NE (payoff={best_conv_asym.team_payoff:.2f}) outperforms "
                f"best symmetric NE (payoff={sym_best:.2f}). "
                "Role differentiation is a genuine equilibrium advantage."
            )
        elif sym_best > best_conv_asym.team_payoff + abs(epsilon):
            verdict = "symmetric_ne_superior"
            verdict_detail = (
                f"Symmetric NE (payoff={sym_best:.2f}) outperforms best converged "
                f"asymmetric NE (payoff={best_conv_asym.team_payoff:.2f}). "
                "Hero-like symmetric play is the dominant equilibrium."
            )
        else:
            verdict = "asymmetric_ne_exists_not_superior"
            verdict_detail = (
                f"Asymmetric NE (payoff={best_conv_asym.team_payoff:.2f}) and "
                f"symmetric NE (payoff={sym_best:.2f}) are within ε={epsilon} of each other."
            )
    elif best_conv_asym is not None:
        verdict = "asymmetric_only"
        verdict_detail = (
            f"All {len(converged)} converged equilibria are asymmetric "
            f"(best payoff={best_conv_asym.team_payoff:.2f}). "
            "No symmetric NE exists — role differentiation is required."
        )
    elif best_conv_sym is not None:
        verdict = "symmetric_only"
        verdict_detail = (
            f"All {len(converged)} converged equilibria are symmetric "
            f"(best payoff={sym_best:.2f}). "
            "Role-differentiated specialization is not a Nash equilibrium."
        )
    elif asymmetric:
        verdict = "asymmetric_profile_found_not_converged"
        verdict_detail = (
            "Asymmetric profiles were found during search but none converged. "
            "Increase restarts/iterations for a definitive answer."
        )
    else:
        verdict = "no_convergence"
        verdict_detail = "No restarts converged within max_iterations."

    print(f"VERDICT: {verdict}")
    print(f"  {verdict_detail}")

    # --- Serialise ---
    results = {
        "scenario": scenario_name,
        "algorithm": "heterogeneous_double_oracle",
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
            "best_team_payoff": float(best.team_payoff) if equilibria else None,
            "best_team_payoff_is_symmetric": _profile_is_symmetric(
                best.strategy_profile
            )
            if equilibria
            else None,
            "verdict": verdict,
        },
        "equilibria": [_eq_to_dict(e) for e in equilibria],
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results → {results_path}")

    # --- Markdown summary ---
    lines = [
        f"# Heterogeneous Nash — {scenario_name}\n",
        f"**Verdict**: `{verdict}`\n",
        f"> {verdict_detail}\n",
        "## Run configuration\n",
        f"- restarts={num_restarts}, simulations={num_simulations}, "
        f"max_iter={max_iterations}, ε={epsilon}, seed={seed}\n",
        f"- Elapsed: {elapsed:.0f}s\n",
        "## Results overview\n",
        f"| | Count |\n|---|---|\n"
        f"| Total restarts | {len(equilibria)} |\n"
        f"| Converged | {len(converged)} |\n"
        f"| Symmetric profiles | {len(symmetric)} |\n"
        f"| Asymmetric profiles | {len(asymmetric)} |\n",
    ]

    if equilibria:
        best = max(equilibria, key=lambda e: e.team_payoff)
        lines += [
            "## Best equilibrium\n",
            f"- Team payoff: **{best.team_payoff:.2f}**\n",
            f"- Profile: `{_profile_label(best.strategy_profile)}`\n",
            f"- Symmetric: {_profile_is_symmetric(best.strategy_profile)}\n",
            f"- Converged: {best.converged}\n",
        ]

    if converged_asymmetric:
        lines += [
            "## Best converged asymmetric equilibrium\n",
            f"- Team payoff: **{best_conv_asym.team_payoff:.2f}**\n",
            f"- Converged: {best_conv_asym.converged}\n",
            f"- Iterations: {best_conv_asym.iterations}\n",
            "\n| Position | Payoff | Closest archetype | work_tendency |\n"
            "|---|---|---|---|\n",
        ] + [
            f"| {i} | {p:.1f} | {_classify(t)} | {t[1]:.3f} |\n"
            for i, (t, p) in enumerate(
                zip(best_conv_asym.strategy_profile, best_conv_asym.payoffs)
            )
        ]
    elif asymmetric:
        best_asym = max(asymmetric, key=lambda e: e.team_payoff)
        lines += [
            "## Best asymmetric profile (not converged)\n",
            f"- Team payoff: **{best_asym.team_payoff:.2f}**\n",
            f"- Converged: {best_asym.converged}\n",
            f"- Iterations: {best_asym.iterations}\n",
            "\n| Position | Payoff | Closest archetype | work_tendency |\n"
            "|---|---|---|---|\n",
        ] + [
            f"| {i} | {p:.1f} | {_classify(t)} | {t[1]:.3f} |\n"
            for i, (t, p) in enumerate(
                zip(best_asym.strategy_profile, best_asym.payoffs)
            )
        ]

    summary_md = output_dir / "summary.md"
    summary_json = output_dir / "summary.json"

    with open(summary_md, "w") as f:
        f.writelines(lines)

    with open(summary_json, "w") as f:
        json.dump(
            {
                "scenario": scenario_name,
                "verdict": verdict,
                "verdict_detail": verdict_detail,
                "elapsed_seconds": elapsed,
                "converged": len(converged),
                "total_restarts": len(equilibria),
                "symmetric_profiles": len(symmetric),
                "asymmetric_profiles": len(asymmetric),
                "converged_asymmetric_profiles": len(converged_asymmetric),
                "converged_symmetric_profiles": len(converged_symmetric),
                "best_team_payoff": float(best.team_payoff) if equilibria else None,
                "best_converged_symmetric_payoff": float(best_conv_sym.team_payoff)
                if best_conv_sym
                else None,
                "best_converged_asymmetric_payoff": float(best_conv_asym.team_payoff)
                if best_conv_asym
                else None,
            },
            f,
            indent=2,
        )

    print(f"Summary     → {summary_md}")
    print(f"Summary JSON→ {summary_json}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Heterogeneous Nash equilibrium via asymmetric Double Oracle"
    )
    parser.add_argument("scenario", help="Scenario name (e.g. minimal_specialization)")
    parser.add_argument(
        "--restarts",
        type=int,
        default=20,
        help="Random starting profiles to try (default: 20)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="MC episodes per payoff estimate (default: 1000)",
    )
    parser.add_argument(
        "--opt-simulations",
        type=int,
        default=300,
        help="Cheaper simulations during gradient search (default: 300)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=25,
        help="BR rounds per restart (default: 25)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
        help="Minimum payoff improvement to update a position (default: 2.0)",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/nash/heterogeneous/<scenario>)",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="Minimum free disk space in MiB before aborting",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-position logs"
    )

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = Path("experiments/nash/heterogeneous") / args.scenario

    check_free_space(args.output_dir, min_free_mib=args.min_free_mib)

    compute_heterogeneous_nash(
        scenario_name=args.scenario,
        output_dir=args.output_dir,
        num_simulations=args.simulations,
        opt_simulations=args.opt_simulations,
        max_iterations=args.max_iterations,
        epsilon=args.epsilon,
        num_restarts=args.restarts,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
