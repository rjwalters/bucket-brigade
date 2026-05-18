#!/usr/bin/env python3
"""Compute the Nash equilibrium of *trained* PPO policies on a scenario (#275).

This is the headline driver for issue #275: take converged PPO policy
checkpoints from ``experiments/p3_specialization/runs/`` and add them to the
heuristic archetype pool as new "strategies" alongside Firefighter, Hero,
Free Rider, Coordinator, and Liar. Run :func:`solve_symmetric_nash` on the
augmented (heuristics + trained) payoff matrix and emit the equilibrium
distribution + a markdown verdict.

Decision question (issue #275 body): pool-inclusion policy =
above-random-filtered vs all-converged. **v1 implements
above-random-filtered** (smaller pool, more interpretable Nash). The user
can request all-converged in a follow-up by passing ``--no-filter``.

Adapter choice: Option B (mixed Python payoff evaluator).

* PPO checkpoints are wrapped via
  :class:`bucket_brigade.agents.TrainedPolicyArchetype` and run through a
  Python rollout. Heuristic-vs-heuristic cells use the existing fast Rust
  path; any cell touching a trained policy falls back to a Python rollout
  that calls ``HeuristicAgent``/``TrainedPolicyArchetype`` ``act()`` in
  whichever slots the cell prescribes.
* This avoids the lossy θ-clone of Option A while staying ≤ 25×25 in
  pool size. See Curator notes on #275.

Scope: this is a *smoke* driver. The full Nash sweep (M(M+K) cells × ~200
episodes each) is remote-only — local runs should use ``--simulations 10``
or so. Disk precheck is integrated (#269/#315).

Usage::

    # Tiny smoke (local-safe)
    uv run python experiments/scripts/compute_nash_trained.py \\
        --scenario default --simulations 5 --max-checkpoints 2

    # Full run (REMOTE only)
    uv run python experiments/scripts/compute_nash_trained.py \\
        --scenario default --simulations 200 --include-trained
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Path setup so sibling helpers (_disk_precheck) are importable even though
# experiments/scripts is not a Python package.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _disk_precheck import DEFAULT_MIN_FREE_MIB, check_free_space  # noqa: E402

from bucket_brigade.agents import (  # noqa: E402
    HeuristicAgent,
    TrainedPolicyArchetype,
)
from bucket_brigade.agents.archetypes import ARCHETYPES  # noqa: E402
from bucket_brigade.envs import BucketBrigadeEnv, get_scenario_by_name  # noqa: E402
from bucket_brigade.equilibrium.nash_solver import (  # noqa: E402
    compute_support,
    solve_symmetric_nash,
)


# ---------------------------------------------------------------------------
# Pool discovery
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT_GLOB = (
    "experiments/p3_specialization/runs/{scenario}/lambda_0e0/seed_*/policies"
)


def discover_checkpoint_dirs(glob_pattern: str, scenario: str) -> List[Path]:
    """Resolve ``glob_pattern`` (with ``{scenario}`` substitution) to dirs.

    Each returned path is expected to contain ``agent_{0..N-1}.pt`` files.
    Returns the sorted list of directories that actually exist.
    """
    pattern = glob_pattern.format(scenario=scenario)
    # Allow the glob to be either repo-relative or absolute.
    base_glob = pattern if Path(pattern).is_absolute() else str(REPO_ROOT / pattern)
    matches = sorted(Path(p) for p in glob(base_glob))
    return [m for m in matches if m.is_dir()]


# ---------------------------------------------------------------------------
# Above-random filter
# ---------------------------------------------------------------------------


def above_random_threshold(
    team_reward: float,
    random_baseline: float,
    margin: float = 0.0,
) -> bool:
    """Return True iff ``team_reward > random_baseline + margin``.

    Margin is in raw-reward units (per-episode, not per-step). Default 0
    keeps any policy that beats random on average; a positive margin trims
    near-random survivors.
    """
    return float(team_reward) > float(random_baseline) + float(margin)


# ---------------------------------------------------------------------------
# Mixed Python payoff evaluator
# ---------------------------------------------------------------------------


def _make_agent(spec: Dict[str, Any], slot_id: int, num_agents: int) -> Any:
    """Instantiate the agent occupying ``slot_id`` for a given strategy spec.

    ``spec`` shapes:
      * ``{"kind": "heuristic", "theta": np.ndarray(10,), "name": str}``
      * ``{"kind": "trained", "path": Path, "name": str, "deterministic": bool}``
    """
    kind = spec["kind"]
    if kind == "heuristic":
        return HeuristicAgent(
            params=np.asarray(spec["theta"], dtype=np.float64),
            agent_id=slot_id,
            name=f"{spec['name']}-{slot_id}",
        )
    if kind == "trained":
        return TrainedPolicyArchetype(
            state_dict_path=Path(spec["path"]),
            agent_id=slot_id,
            num_agents=num_agents,
            deterministic=spec.get("deterministic", True),
            name=f"{spec['name']}-{slot_id}",
        )
    raise ValueError(f"Unknown strategy kind: {kind!r}")


def _run_episode_team_reward(
    env: BucketBrigadeEnv,
    focal_spec: Dict[str, Any],
    opp_spec: Dict[str, Any],
    seed: int,
    num_agents: int,
) -> float:
    """Run one episode with slot-0 = focal, slots 1..N-1 = opp; return focal reward.

    For the symmetric-Nash framing the focal slot's per-episode reward is
    used; this matches ``RustPayoffEvaluator.evaluate_symmetric_payoff``.
    """
    focal_agent = _make_agent(focal_spec, slot_id=0, num_agents=num_agents)
    opp_agents = [
        _make_agent(opp_spec, slot_id=i, num_agents=num_agents)
        for i in range(1, num_agents)
    ]
    agents = [focal_agent, *opp_agents]
    for a in agents:
        a.reset()
    obs = env.reset(seed=seed)
    focal_total = 0.0
    step_count = 0
    while not env.done and step_count < 200:
        joint = np.zeros((num_agents, 3), dtype=np.int8)
        for i, ag in enumerate(agents):
            joint[i] = ag.act(obs)
        obs, rewards, _, _ = env.step(joint)
        focal_total += float(rewards[0])
        step_count += 1
    return focal_total


def evaluate_mixed_payoff_matrix(
    strategy_specs: List[Dict[str, Any]],
    scenario_name: str,
    num_agents: int,
    num_simulations: int,
    seed: int,
    verbose: bool = True,
) -> np.ndarray:
    """Build a K×K payoff matrix over heterogeneous strategy specs.

    Each entry ``A[i, j]`` is the mean per-episode reward of the focal
    agent (slot 0) playing strategy i when the other ``N-1`` slots all
    play strategy j.

    Note: we re-construct the env every cell so per-cell seeding stays
    deterministic. The dominant cost is the rollout itself, not env
    construction.
    """
    K = len(strategy_specs)
    matrix = np.zeros((K, K), dtype=np.float64)
    rng = np.random.RandomState(seed)
    seeds_per_cell = rng.randint(0, 2**31 - 1, size=(K, K, num_simulations))
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    for i in range(K):
        for j in range(K):
            episode_rewards = np.empty(num_simulations, dtype=np.float64)
            for s in range(num_simulations):
                env = BucketBrigadeEnv(scenario=scenario)
                episode_rewards[s] = _run_episode_team_reward(
                    env=env,
                    focal_spec=strategy_specs[i],
                    opp_spec=strategy_specs[j],
                    seed=int(seeds_per_cell[i, j, s]),
                    num_agents=num_agents,
                )
            matrix[i, j] = float(episode_rewards.mean())
            if verbose:
                print(
                    f"  cell [{i:2d},{j:2d}] {strategy_specs[i]['name']:>20s} "
                    f"vs {strategy_specs[j]['name']:<20s}  ->  {matrix[i, j]:8.2f}"
                )
    return matrix


# ---------------------------------------------------------------------------
# Above-random pre-screening: average team reward of each trained checkpoint
# played against a clone of itself.
# ---------------------------------------------------------------------------


def screen_trained_checkpoints(
    checkpoint_dirs: List[Path],
    scenario_name: str,
    num_agents: int,
    num_simulations: int,
    random_baseline: float,
    margin: float,
    seed: int,
    verbose: bool = True,
) -> Tuple[List[Path], Dict[str, float]]:
    """Filter checkpoint dirs to those whose self-play team reward beats random.

    Returns ``(kept_dirs, screening_scores)`` where ``screening_scores``
    maps the dir's basename to the measured team reward. The team reward
    is the *sum* over agents (not slot-0 only) so the screening matches
    the convention used in ``random_baseline.py``.
    """
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    rng = np.random.RandomState(
        seed + 7919
    )  # offset so we don't reuse the cell-seed stream
    kept: List[Path] = []
    scores: Dict[str, float] = {}
    for ckpt_dir in checkpoint_dirs:
        # Build a single team of TrainedPolicyArchetypes (one per slot).
        try:
            team = [
                TrainedPolicyArchetype(
                    state_dict_path=ckpt_dir / f"agent_{i}.pt",
                    agent_id=i,
                    num_agents=num_agents,
                    deterministic=True,
                    name=f"{ckpt_dir.parent.name}",
                )
                for i in range(num_agents)
            ]
        except (FileNotFoundError, ValueError) as exc:
            if verbose:
                print(f"  [screen] SKIP {ckpt_dir}: {exc}")
            continue
        rewards = []
        for s in range(num_simulations):
            for a in team:
                a.reset()
            env = BucketBrigadeEnv(scenario=scenario)
            obs = env.reset(seed=int(rng.randint(0, 2**31 - 1)))
            episode_reward = 0.0
            step_count = 0
            while not env.done and step_count < 200:
                joint = np.zeros((num_agents, 3), dtype=np.int8)
                for i, ag in enumerate(team):
                    joint[i] = ag.act(obs)
                obs, rs, _, _ = env.step(joint)
                episode_reward += float(rs.sum())
                step_count += 1
            rewards.append(episode_reward)
        mean_reward = float(np.mean(rewards))
        scores[ckpt_dir.parent.name] = mean_reward
        ok = above_random_threshold(mean_reward, random_baseline, margin=margin)
        if verbose:
            verdict = "KEEP" if ok else "DROP"
            print(
                f"  [screen] {verdict} {ckpt_dir.parent.name}  "
                f"mean_team_reward={mean_reward:8.2f}  "
                f"(random={random_baseline:.2f}, margin={margin:.1f})"
            )
        if ok:
            kept.append(ckpt_dir)
    return kept, scores


# ---------------------------------------------------------------------------
# Reward-baseline lookup (subset of the SCENARIO_CITED_VALUES table; kept
# inline so this module has no dependency on experiments/p3_specialization
# diagnostics).
# ---------------------------------------------------------------------------

RANDOM_BASELINES: Dict[str, float] = {
    "default": 251.23,
    "easy": 355.07,
    "hard": 124.66,
    "trivial_cooperation": 399.99,
    "early_containment": 297.24,
    "greedy_neighbor": 292.78,
    "sparse_heroics": 246.06,
    "rest_trap": 302.87,
    "chain_reaction": 227.39,
    "deceptive_calm": 78.55,
    "overcrowding": 120.24,
    "mixed_motivation": 224.06,
    "minimal_specialization": -87.72,
    "positional_default": 250.73,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario",
        default="default",
        help="Scenario name (default: default). Must be present in RANDOM_BASELINES.",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=4,
        help="Team size (default: 4 to match the post-2026-05-17 sweeps).",
    )
    parser.add_argument(
        "--checkpoint-glob",
        default=DEFAULT_CHECKPOINT_GLOB,
        help=(
            "Glob pattern for checkpoint policies/ dirs. Substitutes "
            "{scenario}. Default: "
            "experiments/p3_specialization/runs/{scenario}/lambda_0e0/seed_*/policies"
        ),
    )
    parser.add_argument(
        "--include-trained",
        action="store_true",
        help=(
            "Include trained PPO checkpoints in the pool. If omitted, only "
            "the 5 heuristic archetypes are used (regression check / "
            "smoke-test path)."
        ),
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help=(
            "Skip the above-random screening (v1 default: filter enabled). "
            "Use to answer the all-converged variant of the decision question."
        ),
    )
    parser.add_argument(
        "--filter-margin",
        type=float,
        default=0.0,
        help="Margin above the random baseline required to keep a checkpoint.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10,
        help="Monte Carlo simulations per payoff-matrix cell (LOCAL default: 10).",
    )
    parser.add_argument(
        "--screen-simulations",
        type=int,
        default=10,
        help="Monte Carlo simulations per checkpoint for the screening pass.",
    )
    parser.add_argument(
        "--max-checkpoints",
        type=int,
        default=0,
        help=(
            "Hard cap on the number of trained checkpoints included after "
            "filtering. 0 = no cap. Useful for local smoke runs."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/nash/with_ppo"),
        help="Directory under which scenario-named results are written.",
    )
    parser.add_argument(
        "--min-free-mib",
        type=int,
        default=DEFAULT_MIN_FREE_MIB,
        help="Disk-space precheck threshold (MiB).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Discover + screen + print the pool, but skip the Nash compute.",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress per-cell progress lines."
    )

    args = parser.parse_args(argv)
    verbose = not args.quiet

    if args.scenario not in RANDOM_BASELINES:
        print(
            f"ERROR: --scenario {args.scenario!r} not in RANDOM_BASELINES "
            f"({sorted(RANDOM_BASELINES.keys())})",
            file=sys.stderr,
        )
        return 2

    output_dir = Path(args.output_dir) / args.scenario
    check_free_space(output_dir, min_free_mib=args.min_free_mib)
    output_dir.mkdir(parents=True, exist_ok=True)

    random_baseline = RANDOM_BASELINES[args.scenario]
    print(f"=== compute_nash_trained: scenario={args.scenario} ===")
    print(f"  num_agents:       {args.num_agents}")
    print(f"  simulations/cell: {args.simulations}")
    print(f"  random baseline:  {random_baseline:.2f}")
    print(f"  include-trained:  {args.include_trained}")
    print(f"  filter:           {'OFF' if args.no_filter else 'above-random + margin'}")
    print(f"  filter-margin:    {args.filter_margin}")
    print(f"  seed:             {args.seed}")
    print(f"  output:           {output_dir}")
    print()

    # ---- Pool construction --------------------------------------------------
    strategy_specs: List[Dict[str, Any]] = []
    for name, theta in ARCHETYPES.items():
        strategy_specs.append({"kind": "heuristic", "name": name, "theta": theta})

    screening_scores: Dict[str, float] = {}
    excluded: List[str] = []
    discovered: List[Path] = []
    if args.include_trained:
        discovered = discover_checkpoint_dirs(args.checkpoint_glob, args.scenario)
        print(f"Discovered {len(discovered)} checkpoint dir(s) matching glob:")
        for d in discovered:
            print(f"  - {d}")

        if not args.no_filter and discovered:
            print()
            print(
                f"Screening checkpoints (above-random, n={args.screen_simulations} eps each)..."
            )
            kept, screening_scores = screen_trained_checkpoints(
                checkpoint_dirs=discovered,
                scenario_name=args.scenario,
                num_agents=args.num_agents,
                num_simulations=args.screen_simulations,
                random_baseline=random_baseline,
                margin=args.filter_margin,
                seed=args.seed,
                verbose=verbose,
            )
            excluded = [d.parent.name for d in discovered if d not in kept]
        else:
            kept = discovered

        if args.max_checkpoints > 0:
            if len(kept) > args.max_checkpoints:
                print(
                    f"\nCapping kept checkpoints at {args.max_checkpoints} "
                    f"(was {len(kept)})."
                )
                kept = kept[: args.max_checkpoints]

        for ckpt_dir in kept:
            # Heuristic naming: the parent of policies/ is the seed dir,
            # e.g. seed_42 / lambda_0e0 / scenario.
            strategy_specs.append(
                {
                    "kind": "trained",
                    "name": f"ppo_{ckpt_dir.parent.name}",
                    "path": ckpt_dir / "agent_0.pt",
                }
            )

    print(f"\nFinal pool size: {len(strategy_specs)}")
    for spec in strategy_specs:
        print(f"  - [{spec['kind']:9s}] {spec['name']}")

    if args.dry_run:
        print("\n--dry-run requested; skipping Nash compute.")
        # Emit a partial verdict for traceability.
        partial = {
            "scenario": args.scenario,
            "pool_size": len(strategy_specs),
            "pool": [{"kind": s["kind"], "name": s["name"]} for s in strategy_specs],
            "screening_scores": screening_scores,
            "excluded_below_random": excluded,
            "random_baseline": random_baseline,
            "filter_enabled": not args.no_filter,
            "filter_margin": args.filter_margin,
            "dry_run": True,
        }
        out_json = output_dir / "equilibrium_partial.json"
        out_json.write_text(json.dumps(partial, indent=2))
        print(f"Wrote {out_json}")
        return 0

    # ---- Payoff matrix ------------------------------------------------------
    print("\nBuilding payoff matrix...")
    start = time.time()
    matrix = evaluate_mixed_payoff_matrix(
        strategy_specs=strategy_specs,
        scenario_name=args.scenario,
        num_agents=args.num_agents,
        num_simulations=args.simulations,
        seed=args.seed,
        verbose=verbose,
    )
    elapsed_payoff = time.time() - start
    print(f"\nPayoff matrix built in {elapsed_payoff:.1f}s.")

    # ---- Solve Nash ---------------------------------------------------------
    print("Solving symmetric Nash on augmented pool...")
    start = time.time()
    distribution = solve_symmetric_nash(matrix)
    elapsed_nash = time.time() - start
    support = compute_support(distribution, threshold=1e-3)

    print("\nEquilibrium distribution:")
    for idx, prob in enumerate(distribution):
        marker = "*" if idx in support else " "
        print(f"  {marker} {strategy_specs[idx]['name']:25s}  p = {prob:.4f}")

    # ---- Verdict + emit -----------------------------------------------------
    trained_indices = [
        i for i, s in enumerate(strategy_specs) if s["kind"] == "trained"
    ]
    trained_mass = float(sum(distribution[i] for i in trained_indices))
    heuristic_mass = 1.0 - trained_mass
    if not trained_indices:
        verdict = "no-trained-policies-in-pool"
    elif trained_mass < 1e-3:
        verdict = "zero-mass-on-trained (PPO game-theoretically dominated)"
    elif trained_mass > 0.5:
        verdict = "majority-mass-on-trained (PPO is equilibrium-stable; basin trap)"
    else:
        verdict = "mixed (PPO supported but not dominant)"

    equilibrium = {
        "scenario": args.scenario,
        "pool_size": len(strategy_specs),
        "pool": [
            {"index": i, "kind": s["kind"], "name": s["name"]}
            for i, s in enumerate(strategy_specs)
        ],
        "distribution": [float(p) for p in distribution],
        "support_indices": support,
        "trained_mass": trained_mass,
        "heuristic_mass": heuristic_mass,
        "verdict": verdict,
        "payoff_matrix": matrix.tolist(),
        "screening_scores": screening_scores,
        "excluded_below_random": excluded,
        "random_baseline": random_baseline,
        "filter_enabled": not args.no_filter,
        "filter_margin": args.filter_margin,
        "num_simulations": args.simulations,
        "seed": args.seed,
        "elapsed_payoff_sec": elapsed_payoff,
        "elapsed_nash_sec": elapsed_nash,
    }
    out_json = output_dir / "equilibrium.json"
    out_json.write_text(json.dumps(equilibrium, indent=2))
    print(f"\nWrote {out_json}")

    md_lines = [
        f"# Nash equilibrium of trained PPO policies — {args.scenario}",
        "",
        f"- pool size: **{len(strategy_specs)}** "
        f"({sum(1 for s in strategy_specs if s['kind'] == 'heuristic')} heuristic + "
        f"{sum(1 for s in strategy_specs if s['kind'] == 'trained')} trained)",
        f"- trained mass: **{trained_mass:.4f}**",
        f"- heuristic mass: **{heuristic_mass:.4f}**",
        f"- verdict: **{verdict}**",
        f"- filter: {'OFF' if args.no_filter else f'above-random + margin {args.filter_margin}'}",
        f"- random baseline: {random_baseline:.2f}",
        f"- seed: {args.seed}",
        f"- simulations/cell: {args.simulations}",
        "",
        "## Equilibrium support",
        "",
        "| idx | kind | name | p |",
        "| --- | --- | --- | --- |",
    ]
    for idx in support:
        s = strategy_specs[idx]
        md_lines.append(
            f"| {idx} | {s['kind']} | {s['name']} | {distribution[idx]:.4f} |"
        )
    if excluded:
        md_lines += ["", "## Excluded (below random)", ""]
        for name in excluded:
            score = screening_scores.get(name, float("nan"))
            md_lines.append(f"- {name}: {score:.2f}")
    out_md = output_dir / "equilibrium.md"
    out_md.write_text("\n".join(md_lines) + "\n")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
