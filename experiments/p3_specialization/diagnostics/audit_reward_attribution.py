"""
H2 diagnostic: per-step reward attribution audit.

Decomposes each agent's per-step reward into:
  - team_component[t]: scalar, identical across all 4 agents
  - ownership_component[i][t]: per-agent (save bonus + ruined penalty for owned houses)
  - work_cost[i][t]: per-agent (-cost_to_work_one_night if WORK, +0.5 if REST)

Reports magnitude ratios (median / p75 / p95), the 4x4 pairwise reward
correlation matrix, and the team-variance lower bound on cross-agent
correlation. Diagnostic-only — does not modify env or training code.

Run from repo root::

    # default scenario (legacy default)
    uv run python experiments/p3_specialization/diagnostics/audit_reward_attribution.py

    # any other registered scenario (e.g. minimal_specialization, issue #199)
    uv run python experiments/p3_specialization/diagnostics/audit_reward_attribution.py \\
        --scenario minimal_specialization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name


def run_rollouts(scenario_name="default", seeds=tuple(range(20)), max_nights_per_ep=200):
    """Roll out a few episodes with uniform-random actions and decompose
    each per-step reward into (team, ownership, work_cost).

    The decomposition mirrors `_compute_rewards` exactly so the three
    components sum to the env-returned reward (verified at runtime).
    """
    scenario = get_scenario_by_name(scenario_name, num_agents=4)
    env = BucketBrigadeEnv(scenario=scenario, num_agents=4)

    team_t: list[float] = []
    own_ti: list[np.ndarray] = []
    work_ti: list[np.ndarray] = []
    reward_ti: list[np.ndarray] = []

    for seed in seeds:
        env.reset(seed=seed)
        # Mirror env's internal `_prev_houses_state` semantics: it is reset to
        # all-zeros (SAFE) in `reset()` and only overwritten with `self.houses`
        # at the *start* of each `_compute_rewards` call. So on the first step,
        # prev_houses is all-SAFE even if `_initialize_houses` lit some fires.
        prev_houses = np.zeros(10, dtype=np.int8)
        rng = np.random.RandomState(seed + 10_000)
        steps = 0
        while not env.done and steps < max_nights_per_ep:
            # Uniform-random actions: (house_idx in [0,10), mode in {REST, WORK})
            houses = rng.randint(0, 10, size=4).astype(np.int8)
            modes = rng.randint(0, 2, size=4).astype(np.int8)
            actions = np.stack([houses, modes], axis=1)

            _, rewards, _, _ = env.step(actions)

            # Decompose using the state observed *after* step (env.houses)
            # and the prev_houses captured at the start of the step.
            saved = int(np.sum(env.houses == env.SAFE))
            ruined = int(np.sum(env.houses == env.RUINED))
            team = (
                scenario.team_reward_house_survives * (saved / 10.0)
                - scenario.team_penalty_house_burns * (ruined / 10.0)
            )

            own = np.zeros(4, dtype=np.float64)
            work = np.zeros(4, dtype=np.float64)
            for a in range(4):
                work[a] = (
                    -scenario.cost_to_work_one_night
                    if actions[a, 1] == env.WORK
                    else 0.5
                )
                for h in range(10):
                    is_own = env.house_owners[h] == a
                    # Save event (any non-SAFE -> SAFE transition this step).
                    # Post-#198, the four ownership reward fields are per-agent
                    # vectors of length num_agents; index by agent.
                    if prev_houses[h] != env.SAFE and env.houses[h] == env.SAFE:
                        own[a] += (
                            scenario.reward_own_house_survives[a]
                            if is_own
                            else scenario.reward_other_house_survives[a]
                        )
                    # Currently-ruined penalty (applied every step)
                    if env.houses[h] == env.RUINED:
                        own[a] -= (
                            scenario.penalty_own_house_burns[a]
                            if is_own
                            else scenario.penalty_other_house_burns[a]
                        )

            # Sanity: components must sum to env.rewards
            recon = team + own + work
            assert np.allclose(recon, rewards, atol=1e-5), (
                f"Decomposition mismatch step={steps} seed={seed}: "
                f"recon={recon} env={rewards}"
            )

            team_t.append(team)
            own_ti.append(own)
            work_ti.append(work)
            reward_ti.append(rewards.astype(np.float64).copy())

            prev_houses = env.houses.copy()
            steps += 1

    return (
        np.asarray(team_t, dtype=np.float64),
        np.asarray(own_ti, dtype=np.float64),
        np.asarray(work_ti, dtype=np.float64),
        np.asarray(reward_ti, dtype=np.float64),
    )


def summarize(team, own, work, reward):
    T, A = own.shape

    # Magnitude ratios per timestep, averaged across agents for ownership/work.
    # We compute |team|/mean(|own_a|) and |team|/mean(|work_a|); when an
    # ownership component is exactly zero we skip that step (ratio undefined).
    abs_team = np.abs(team)
    mean_abs_own = np.mean(np.abs(own), axis=1)
    mean_abs_work = np.mean(np.abs(work), axis=1)

    own_nonzero = mean_abs_own > 1e-9
    work_nonzero = mean_abs_work > 1e-9

    ratio_to_own = abs_team[own_nonzero] / mean_abs_own[own_nonzero]
    ratio_to_work = abs_team[work_nonzero] / mean_abs_work[work_nonzero]

    def pcts(x):
        if x.size == 0:
            return {"median": None, "p75": None, "p95": None, "n": 0}
        return {
            "median": float(np.median(x)),
            "p75": float(np.percentile(x, 75)),
            "p95": float(np.percentile(x, 95)),
            "n": int(x.size),
        }

    ratios = {
        "team_over_ownership": pcts(ratio_to_own),
        "team_over_work_cost": pcts(ratio_to_work),
        "frac_steps_ownership_zero": float(np.mean(~own_nonzero)),
    }

    # Pairwise agent reward correlation (4x4)
    corr = np.corrcoef(reward.T)  # reward is (T, A); .T -> (A, T)

    # Lower bound on cross-agent correlation from shared team component.
    # If r_i = team + own_i + work_i and team is identical across agents,
    # while (own + work) is independent across agents in the limit, then
    # corr(r_i, r_j) >= var(team) / var(r_i) for any j != i (when var(r_i)
    # is similar across agents). We report this scalar bound using the
    # mean per-agent variance.
    team_var = float(np.var(team))
    per_agent_var = np.var(reward, axis=0)
    mean_agent_var = float(np.mean(per_agent_var))
    team_share_lower_bound = team_var / mean_agent_var if mean_agent_var > 0 else None

    # Mean off-diagonal correlation
    mask = ~np.eye(A, dtype=bool)
    mean_off_diag = float(corr[mask].mean())
    min_off_diag = float(corr[mask].min())

    return {
        "n_steps": int(T),
        "n_agents": int(A),
        "magnitudes": {
            "mean_abs_team": float(np.mean(abs_team)),
            "mean_abs_ownership_per_agent": float(np.mean(np.abs(own))),
            "mean_abs_work_per_agent": float(np.mean(np.abs(work))),
        },
        "ratios": ratios,
        "pairwise_corr": corr.tolist(),
        "mean_off_diag_corr": mean_off_diag,
        "min_off_diag_corr": min_off_diag,
        "team_var": team_var,
        "mean_per_agent_var": mean_agent_var,
        "team_share_lower_bound_on_corr": team_share_lower_bound,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scenario",
        default="default",
        help="Scenario name from SCENARIO_REGISTRY (e.g. default, minimal_specialization).",
    )
    ap.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix appended to results filename (default: derived from scenario "
        "name, except 'default' which writes to the canonical h2_reward_attribution.json).",
    )
    args = ap.parse_args()

    team, own, work, reward = run_rollouts(scenario_name=args.scenario)
    summary = summarize(team, own, work, reward)
    summary["scenario"] = args.scenario

    print("=" * 70)
    print(f"H2 reward-attribution audit — {args.scenario}_scenario(num_agents=4)")
    print("=" * 70)
    print(f"Steps analyzed: {summary['n_steps']}  (uniform-random rollouts)")
    print()
    print("Magnitudes (mean of absolute value):")
    m = summary["magnitudes"]
    print(f"  team component         {m['mean_abs_team']:8.3f}")
    print(f"  ownership component    {m['mean_abs_ownership_per_agent']:8.3f}  (per agent)")
    print(f"  work cost              {m['mean_abs_work_per_agent']:8.3f}  (per agent)")
    print()
    print("Per-step magnitude ratios:")
    r = summary["ratios"]
    to = r["team_over_ownership"]
    tw = r["team_over_work_cost"]
    print(f"  |team| / |ownership|  median={to['median']:.2f}  "
          f"p75={to['p75']:.2f}  p95={to['p95']:.2f}  (n={to['n']})")
    print(f"  |team| / |work_cost|  median={tw['median']:.2f}  "
          f"p75={tw['p75']:.2f}  p95={tw['p95']:.2f}  (n={tw['n']})")
    print(f"  fraction of steps with zero ownership: {r['frac_steps_ownership_zero']:.3f}")
    print()
    print("Pairwise agent reward correlation matrix (4x4):")
    for row in summary["pairwise_corr"]:
        print("  " + "  ".join(f"{v:+.4f}" for v in row))
    print()
    print(f"Mean off-diagonal corr: {summary['mean_off_diag_corr']:.4f}")
    print(f"Min off-diagonal corr:  {summary['min_off_diag_corr']:.4f}")
    print()
    print(f"Team variance:                  {summary['team_var']:.4f}")
    print(f"Mean per-agent reward variance: {summary['mean_per_agent_var']:.4f}")
    print(f"Team-share lower bound on corr: "
          f"{summary['team_share_lower_bound_on_corr']:.4f}")
    print()
    verdict_h2 = (
        (to["median"] is not None and to["median"] >= 10.0)
        or summary["min_off_diag_corr"] > 0.95
    )
    print(f"H2 VERDICT: {'CONFIRMED' if verdict_h2 else 'NOT CONFIRMED'}")

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.output_suffix is not None:
        suffix = args.output_suffix
    elif args.scenario == "default":
        suffix = ""
    else:
        suffix = f"_{args.scenario}"
    out_path = out_dir / f"h2_reward_attribution{suffix}.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
