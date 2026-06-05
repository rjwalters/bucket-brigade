#!/usr/bin/env python3
"""
Symmetric specialist Nash verification for ``minimal_specialization`` (issue #354).

Given the converged specialist baseline policy on ``minimal_specialization``,
verify whether the *symmetric* profile (all 4 agents play specialist) is a
Nash equilibrium by computing the best response of a 4th deviating agent
when the other 3 play specialist.

Algorithm
---------
1. Load the specialist policy parameters. Per ``experiments/p3_specialization/
   diagnostics/issue199_baselines.py``, the specialist on
   ``minimal_specialization`` is the *hand-coded* ``specialist_action_joint``
   (`bucket_brigade.baselines.specialist`), NOT an MLP.  Since the
   best-response infrastructure (`compute_best_response`) operates on
   10-dimensional heuristic-agent parameter vectors, we encode the specialist
   as the **closest heuristic approximation** ``SPECIALIST_HEURISTIC_PARAMS``
   (documented inline below).  This is faithful to the issue's "convert to
   heuristic params" branch and matches the path taken by sibling PRs #355 /
   #376 (heterogeneous Nash + per-position exploitability).

2. Compute the baseline payoff of (specialist, specialist, specialist,
   specialist) using the shared Rust evaluator.

3. Multi-round best-response sweep:
     - Round 1: BR vs (specialist, specialist, specialist)
     - Round 2: BR vs (BR_1, specialist, specialist) using best of round 1
     - Round 3: BR vs (BR_2, BR_1, specialist) ...
   Each round uses ``compute_best_response_global`` (differential evolution,
   robust to local optima) followed by full-MC re-evaluation.

4. Compute exploitability gap = ``BR_payoff - specialist_payoff`` (per-step
   normalisation reported alongside the cumulative per-episode value).

5. Classify per the issue's pre-registered verdict criteria:
     - improvement < 1.0 per step  → ε-Nash equilibrium
     - 1.0 < improvement < 10.0   → weakly exploitable
     - improvement > 10.0          → strongly exploitable

Compute
-------
CPU-bound; the full sweep at default settings (1000 sims, 3 rounds, global
optimisation via differential evolution at ~20 maxiter × 4-position payoff
calls) is on the order of **1–2 hours on $COMPUTE_HOST_PRIMARY** per the
issue's own estimate.  **DO NOT RUN LOCALLY** — push to
$COMPUTE_HOST_PRIMARY or $COMPUTE_HOST_CLUSTER from .env.

Usage
-----
    # Full run on remote host
    uv run python experiments/scripts/test_specialist_nash.py

    # Smoke test (1 round, tiny budget) — safe to run locally in a few seconds
    uv run python experiments/scripts/test_specialist_nash.py --smoke

    # Custom rounds / budget
    uv run python experiments/scripts/test_specialist_nash.py \\
        --rounds 3 --simulations 1000 --opt-method global

Output
------
``experiments/nash/specialist_nash_test/minimal_specialization/``:
  - ``br_sweep.json``    per-round BR strategy + payoff
  - ``exploitability.json``    final gap + pre-registered verdict
  - ``summary.md``    human-readable Markdown summary
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

from bucket_brigade.envs import get_scenario_by_name
from bucket_brigade.equilibrium.best_response import (
    compute_best_response,
    compute_best_response_global,
)
from bucket_brigade.equilibrium.payoff_evaluator_rust import RustPayoffEvaluator
from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    FREE_RIDER_PARAMS,
    HERO_PARAMS,
    COORDINATOR_PARAMS,
    LIAR_PARAMS,
)


# ---------------------------------------------------------------------------
# Specialist heuristic-space encoding
# ---------------------------------------------------------------------------

# Heuristic-space approximation of the hand-coded specialist policy from
# `bucket_brigade.baselines.specialist.specialist_action_joint`.
#
# The hand-coded specialist (per `specialist.py`, lines 87-93):
#     1. If any owned house is BURNING, WORK the lowest-index owned burning house.
#     2. Otherwise REST.
#     3. Signal honestly (signal == mode).
#
# The heuristic agent's decision rule (per `bucket-brigade-core/src/agents/
# heuristic.rs` lines 62-93, mirrored in `payoff_evaluator_rust._heuristic_action`):
#     work_p = work_tendency * (1 - rest_reward_bias)
#     if rng < work_p:
#         if owned_house is burning AND rng < own_house_priority: pick owned
#         else: pick a random burning house (or fall back to owned)
#         mode = WORK
#     else:
#         pick owned_house, mode = REST
#     if rng < honesty_bias: signal = mode else signal = 1 - mode
#
# Closest match for "always work owned burning house, else rest":
#     work_tendency = 1.0,  rest_reward_bias = 0.0     → always work
#     own_house_priority = 1.0                          → always pick owned if burning
#     honesty_bias = 1.0                                → honest signals
#
# **Known approximation limit**: the heuristic's "owned house" is `agent_id % 10`
# (single house), while the specialist's ownership is round-robin
# `np.arange(num_houses) % num_agents == agent_id` (multiple houses per agent for
# num_agents < num_houses).  For num_agents=4, num_houses=10:
#     specialist: agent 0 owns {0,4,8}, agent 1 owns {1,5,9}, etc.
#     heuristic:  agent i "owns" house i (only).
# This means the heuristic approximation will work-on-the-wrong-house when the
# agent's true owned-set has multiple fires (specialist would pick the lowest;
# heuristic only knows about agent_id % 10).  Other 10 params are zeroed
# (irrelevant to the decision rule under work_tendency=1, own_house_priority=1).
#
# Documenting this approximation explicitly so downstream readers understand
# what is being tested: this is the "heuristic-space proxy" Nash check, not
# a policy-space check against the literal hand-coded specialist.
SPECIALIST_HEURISTIC_PARAMS = np.array(
    [
        1.0,  # honesty_bias       - honest signals (matches specialist)
        1.0,  # work_tendency      - always work (gated by own_house priority below)
        0.0,  # neighbor_help_bias - irrelevant under own_house_priority=1
        1.0,  # own_house_priority - always pick owned house if burning
        0.0,  # risk_aversion      - irrelevant in this rule
        0.0,  # coordination_weight- irrelevant in this rule
        0.0,  # exploration_rate   - irrelevant in this rule
        0.0,  # fatigue_memory     - irrelevant in this rule
        0.0,  # rest_reward_bias   - don't bias toward rest (specialist works when fires)
        0.0,  # altruism_factor    - irrelevant in this rule
    ]
)


# ---------------------------------------------------------------------------
# Classification helper (for reporting BR strategies)
# ---------------------------------------------------------------------------

ARCHETYPE_POOL: dict[str, np.ndarray] = {
    "firefighter": FIREFIGHTER_PARAMS,
    "free_rider": FREE_RIDER_PARAMS,
    "hero": HERO_PARAMS,
    "coordinator": COORDINATOR_PARAMS,
    "liar": LIAR_PARAMS,
    "specialist": SPECIALIST_HEURISTIC_PARAMS,
}


def _classify(theta: np.ndarray) -> str:
    """Return name of closest archetype by L2 distance."""
    best_name, best_dist = "unknown", np.inf
    for name, params in ARCHETYPE_POOL.items():
        d = float(np.linalg.norm(theta - params))
        if d < best_dist:
            best_dist = d
            best_name = name
    return f"{best_name}(d={best_dist:.3f})"


# ---------------------------------------------------------------------------
# Pre-registered verdict
# ---------------------------------------------------------------------------


def _verdict(improvement_per_step: float) -> str:
    """Apply the issue's pre-registered verdict criteria.

    Translates raw BR-improvement gaps into one of three verdict labels
    matching the table in issue #354's design section.
    """
    if improvement_per_step < 1.0:
        return "epsilon_nash"
    if improvement_per_step < 10.0:
        return "weakly_exploitable"
    return "strongly_exploitable"


def _verdict_text(label: str) -> str:
    return {
        "epsilon_nash": (
            "Specialist is approximately a Nash equilibrium "
            "(improvement < 1.0 per step). P3 failure is a gradient/basin problem."
        ),
        "weakly_exploitable": (
            "Specialist is weakly exploitable (1.0 ≤ improvement < 10.0 per step). "
            "Specialization is nearly stable but not exactly; run heterogeneous "
            "Nash (#353) to find the true NE."
        ),
        "strongly_exploitable": (
            "Specialist is strongly exploitable (improvement ≥ 10.0 per step). "
            "Specialization is NOT a Nash equilibrium. Free-riding dominates; "
            "update P3 research framing."
        ),
    }[label]


# ---------------------------------------------------------------------------
# Best-response sweep
# ---------------------------------------------------------------------------


def _best_response_round(
    opponents: np.ndarray,
    scenario,
    num_simulations: int,
    opt_method: str,
    seed: int,
) -> tuple[np.ndarray, float]:
    """One round of best response: find BR vs ``opponents``.

    Returns the BR strategy and its full-MC payoff.
    """
    if opt_method == "global":
        br, br_payoff = compute_best_response_global(
            theta_opponents=opponents,
            scenario=scenario,
            num_simulations=num_simulations,
            seed=seed,
        )
    elif opt_method == "local":
        br, br_payoff = compute_best_response(
            theta_opponents=opponents,
            scenario=scenario,
            num_simulations=num_simulations,
            method="L-BFGS-B",
            x0=opponents,
            seed=seed,
        )
    else:
        raise ValueError(
            f"Unknown opt_method={opt_method!r}; expected 'global'|'local'"
        )
    return br, float(br_payoff)


def run_specialist_nash_test(
    scenario_name: str,
    rounds: int,
    num_simulations: int,
    opt_method: str,
    seed: int,
    num_workers: int | None,
    episode_length_for_per_step: int,
    skip_optimization: bool = False,
) -> dict:
    """Run the full BR-sweep Nash verification.

    If ``skip_optimization`` is True (smoke harness mode), each round skips
    the BR optimizer and reports a no-op BR equal to ``SPECIALIST_HEURISTIC_PARAMS``.
    Used to verify that the script wires correctly + writes artifacts without
    incurring the multi-minute Pool-per-call cost of the real optimizer.
    """

    print("=" * 70)
    print("Symmetric specialist Nash verification — issue #354")
    print("=" * 70)
    print(f"Scenario:        {scenario_name}")
    print("Specialist proxy: SPECIALIST_HEURISTIC_PARAMS (heuristic-space encoding)")
    print(f"Rounds:          {rounds}")
    print(f"Simulations:     {num_simulations}")
    print(f"Optimisation:    {opt_method}")
    print(f"Seed:            {seed}")
    print()

    scenario = get_scenario_by_name(scenario_name, num_agents=4)
    print(
        f"Scenario params: β={scenario.prob_fire_spreads_to_neighbor:.2f} "
        f"κ={scenario.prob_solo_agent_extinguishes_fire:.2f} "
        f"c={scenario.cost_to_work_one_night:.2f} "
        f"team_reward={scenario.team_reward_house_survives:.1f} "
        f"team_penalty={scenario.team_penalty_house_burns:.1f}"
    )
    print()

    # --- Baseline: symmetric specialist payoff ---
    print("Evaluating symmetric specialist baseline payoff...", flush=True)
    t0 = time.time()
    evaluator = RustPayoffEvaluator(
        scenario=scenario,
        num_simulations=num_simulations,
        seed=seed,
        parallel=num_workers is None or num_workers > 1,
        num_workers=num_workers,
    )
    specialist_payoff = float(
        evaluator.evaluate_symmetric_payoff(
            theta_focal=SPECIALIST_HEURISTIC_PARAMS,
            theta_opponents=SPECIALIST_HEURISTIC_PARAMS,
        )
    )
    spec_elapsed = time.time() - t0
    print(f"  specialist_payoff = {specialist_payoff:.2f}  ({spec_elapsed:.1f}s)")
    print()

    # --- Multi-round best-response sweep ---
    # The "opponents" vector represents what 3 opponents play (assumed symmetric
    # in `evaluate_symmetric_payoff`).  In each round we set opponents to the
    # PREVIOUS round's BR (so by round k we are asking: what's the best response
    # if all 3 opponents have switched to the BR-against-specialist?).  This
    # is the classic BR-chain convergence test.
    sweep = []
    current_opponents = SPECIALIST_HEURISTIC_PARAMS.copy()
    for round_idx in range(1, rounds + 1):
        print(
            f"--- Round {round_idx}: BR vs opponents = {_classify(current_opponents)} ---",
            flush=True,
        )
        t_round = time.time()
        round_seed = seed + 1000 * round_idx
        if skip_optimization:
            # Harness-only path: no BR optimization, report identity BR for
            # artifact-shape verification. Real verdict requires the real
            # optimizer (full remote run).
            br_strategy = SPECIALIST_HEURISTIC_PARAMS.copy()
            br_payoff = float(
                evaluator.evaluate_symmetric_payoff(
                    theta_focal=br_strategy,
                    theta_opponents=current_opponents,
                )
            )
        else:
            br_strategy, br_payoff = _best_response_round(
                opponents=current_opponents,
                scenario=scenario,
                num_simulations=num_simulations,
                opt_method=opt_method,
                seed=round_seed,
            )
        round_elapsed = time.time() - t_round

        # Re-evaluate specialist-payoff under the *same* opponents profile so
        # the gap is apples-to-apples (round 1 reference is the symmetric
        # specialist baseline; later rounds compare BR vs specialist-playing-
        # against-the-updated-opponents).
        if round_idx == 1:
            reference_payoff = specialist_payoff
        else:
            reference_payoff = float(
                evaluator.evaluate_symmetric_payoff(
                    theta_focal=SPECIALIST_HEURISTIC_PARAMS,
                    theta_opponents=current_opponents,
                )
            )

        gap = br_payoff - reference_payoff
        gap_per_step = gap / max(episode_length_for_per_step, 1)
        print(
            f"  BR = {_classify(br_strategy)}  payoff={br_payoff:.2f}  "
            f"specialist_payoff_vs_same_opponents={reference_payoff:.2f}  "
            f"gap={gap:+.2f}  per-step={gap_per_step:+.3f}  ({round_elapsed:.1f}s)",
            flush=True,
        )

        sweep.append(
            {
                "round": round_idx,
                "opponents_classification": _classify(current_opponents),
                "opponents_genome": [float(v) for v in current_opponents],
                "br_strategy_classification": _classify(br_strategy),
                "br_strategy_genome": [float(v) for v in br_strategy],
                "br_payoff": br_payoff,
                "reference_specialist_payoff_vs_same_opponents": reference_payoff,
                "gap_total": gap,
                "gap_per_step_approx": gap_per_step,
                "elapsed_seconds": round_elapsed,
                "seed": round_seed,
            }
        )

        # Update opponents for next round
        current_opponents = br_strategy.copy()

    # --- Final exploitability verdict (use round 1 gap, as it directly tests
    # "is symmetric specialist a NE?") ---
    final_gap = sweep[0]["gap_total"]
    final_gap_per_step = sweep[0]["gap_per_step_approx"]
    verdict_label = _verdict(final_gap_per_step)
    verdict_text = _verdict_text(verdict_label)

    return {
        "scenario": scenario_name,
        "skip_optimization": bool(skip_optimization),
        "specialist_params": [float(v) for v in SPECIALIST_HEURISTIC_PARAMS],
        "specialist_params_note": (
            "Heuristic-space approximation of the hand-coded "
            "`specialist_action_joint` baseline. See module docstring for the "
            "approximation derivation and known fidelity limits."
        ),
        "num_simulations": num_simulations,
        "opt_method": opt_method,
        "seed": seed,
        "episode_length_for_per_step": episode_length_for_per_step,
        "symmetric_specialist_payoff": specialist_payoff,
        "rounds": sweep,
        "exploitability_gap_total": final_gap,
        "exploitability_gap_per_step": final_gap_per_step,
        "verdict_label": verdict_label,
        "verdict_text": verdict_text,
    }


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_br_sweep(results: dict, path: Path) -> None:
    payload = {
        "scenario": results["scenario"],
        "specialist_params": results["specialist_params"],
        "num_simulations": results["num_simulations"],
        "opt_method": results["opt_method"],
        "seed": results["seed"],
        "rounds": results["rounds"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_exploitability(results: dict, path: Path) -> None:
    payload = {
        "scenario": results["scenario"],
        "symmetric_specialist_payoff": results["symmetric_specialist_payoff"],
        "round1_br_payoff": results["rounds"][0]["br_payoff"],
        "round1_br_classification": results["rounds"][0]["br_strategy_classification"],
        "exploitability_gap_total": results["exploitability_gap_total"],
        "exploitability_gap_per_step": results["exploitability_gap_per_step"],
        "episode_length_for_per_step": results["episode_length_for_per_step"],
        "verdict_label": results["verdict_label"],
        "verdict_text": results["verdict_text"],
        "num_simulations": results["num_simulations"],
        "opt_method": results["opt_method"],
        "seed": results["seed"],
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _write_summary(results: dict, path: Path) -> None:
    lines = [
        f"# Symmetric Specialist Nash Test — `{results['scenario']}`",
        "",
        "Issue #354.  Per-position best-response check: does any single deviating",
        "agent gain by switching strategy when the other 3 play specialist?",
        "",
        "## Setup",
        "",
        "- Specialist proxy: heuristic-space encoding "
        "(`SPECIALIST_HEURISTIC_PARAMS`, see script docstring for derivation).",
        f"- Simulations per payoff estimate: **{results['num_simulations']}**",
        f"- BR optimisation method: **{results['opt_method']}**",
        f"- Episode length used for per-step normalisation: **{results['episode_length_for_per_step']}**",
        f"- Seed: {results['seed']}",
        "",
        "## Symmetric specialist baseline",
        "",
        f"- `specialist_payoff` (all 4 play specialist): **{results['symmetric_specialist_payoff']:.2f}** "
        f"per episode "
        f"(≈ **{results['symmetric_specialist_payoff'] / max(results['episode_length_for_per_step'], 1):.2f}** per step)",
        "",
        "## Best-response sweep",
        "",
        "| Round | Opponents | BR closest archetype | BR payoff | Specialist payoff (vs same opp) | Gap (total) | Gap (per step) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results["rounds"]:
        lines.append(
            f"| {r['round']} | {r['opponents_classification']} | "
            f"{r['br_strategy_classification']} | {r['br_payoff']:.2f} | "
            f"{r['reference_specialist_payoff_vs_same_opponents']:.2f} | "
            f"{r['gap_total']:+.2f} | {r['gap_per_step_approx']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Pre-registered verdict",
            "",
            f"- Exploitability gap (per step, round 1): **{results['exploitability_gap_per_step']:+.3f}**",
            f"- Verdict label: **`{results['verdict_label']}`**",
            f"- Interpretation: {results['verdict_text']}",
            "",
            "## Caveats",
            "",
            "- The specialist policy is encoded as a heuristic-space approximation",
            "  (see script docstring).  The hand-coded `specialist_action_joint`",
            "  rule and the heuristic-agent decision rule differ in how house",
            "  ownership is computed (round-robin set vs. `agent_id % 10` single",
            "  house).  Treat the verdict as 'heuristic-space proxy NE' rather",
            "  than a literal policy-space claim.",
            "- Best-response optimisation searches the 10-dim heuristic parameter",
            "  cube `[0,1]^10`; deviations outside the heuristic family (e.g., a",
            "  trained MLP) are not tested.",
            "- Episode-length per-step normalisation uses a fixed scalar; if the",
            "  scenario's actual mean episode length differs, the verdict",
            "  threshold may shift by a small factor.",
        ]
    )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Symmetric specialist Nash verification for minimal_specialization (#354)."
    )
    parser.add_argument(
        "--scenario",
        default="minimal_specialization",
        help="Scenario name (default: minimal_specialization, the issue #354 target).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of BR-chain rounds to run (default: 3, per issue spec).",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=1000,
        help="MC simulations per payoff estimate (default: 1000).",
    )
    parser.add_argument(
        "--opt-method",
        choices=["global", "local"],
        default="global",
        help="BR optimisation: 'global' (differential evolution, robust) "
        "or 'local' (L-BFGS-B, faster).  Default: global.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel workers for MC evaluation (default: cpu_count()).",
    )
    parser.add_argument(
        "--episode-length",
        type=int,
        default=13,
        help="Episode length used to normalise gap to per-step units. "
        "Default: 13, matching the median episode length observed in "
        "issue #199 baselines (`minimal_specialization` scenarios).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke-test mode: --rounds 1 --simulations 30 and SKIP the BR "
        "optimizer (uses identity BR = specialist). Verifies the script wires "
        "correctly and writes artifacts without spending hours of compute. The "
        "real verdict requires the full remote run (no --smoke).",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip the BR optimizer (debug). Implied by --smoke.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory (default: "
            "experiments/nash/specialist_nash_test/<scenario>/)."
        ),
    )
    args = parser.parse_args()

    if args.smoke:
        args.rounds = min(args.rounds, 1)
        args.simulations = min(args.simulations, 30)
        args.opt_method = "local"
        args.skip_optimization = True

    if args.output_dir is None:
        repo_root = Path(__file__).resolve().parents[2]
        args.output_dir = (
            repo_root / "experiments" / "nash" / "specialist_nash_test" / args.scenario
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = run_specialist_nash_test(
        scenario_name=args.scenario,
        rounds=args.rounds,
        num_simulations=args.simulations,
        opt_method=args.opt_method,
        seed=args.seed,
        num_workers=args.num_workers,
        episode_length_for_per_step=args.episode_length,
        skip_optimization=args.skip_optimization,
    )
    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed
    results["smoke_test"] = bool(args.smoke)

    suffix = "_smoke" if args.smoke else ""
    br_sweep_path = args.output_dir / f"br_sweep{suffix}.json"
    exploitability_path = args.output_dir / f"exploitability{suffix}.json"
    summary_path = args.output_dir / f"summary{suffix}.md"

    _write_br_sweep(results, br_sweep_path)
    _write_exploitability(results, exploitability_path)
    _write_summary(results, summary_path)

    print()
    print("=" * 70)
    print(f"Wrote {br_sweep_path}")
    print(f"Wrote {exploitability_path}")
    print(f"Wrote {summary_path}")
    print(f"Total elapsed: {elapsed:.1f}s")
    print(
        f"Exploitability gap (per step): {results['exploitability_gap_per_step']:+.3f}"
    )
    print(f"Verdict: {results['verdict_label']}  — {results['verdict_text']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
