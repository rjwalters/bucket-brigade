"""H3 diagnostic: re-derive the "308" random baseline used in P3 specialization.

The acceptance bar in ``analyze_plateau.py`` (``BASELINES["default"]["random"] =
308.0``) and ``analyze_174.py`` (``RAND_BASELINE = 308.0``) traces back to
issue #145's body, which reports::

    Random actions across 50 episodes on ``default``: 4012.34 per episode
    (~+308/step).

That measurement has no committed script, so this diagnostic is the durable
artifact: re-derive the number from scratch under conditions matching the #183
phase-3 training cells (``default`` scenario, ``num_agents=4``), and put a
random-init MLP iter-0 baseline next to it for comparison.

Latest re-derivation (issue #237, 2026-05-16, commit ``dffe1060``): post-#236
sweep across all 14 named scenarios with the 3-dim action space
``MultiDiscrete([10, 2, 2])`` (signal channel sampled independently). See
``SCENARIO_CITED_VALUES`` for the per-scenario table.

Three numbers are reported, each as ``mean ± 95% bootstrap CI`` over the per-
episode samples:

1. **Uniform-random per-episode team reward** (matches #145's
   "4012.34 per episode" framing).
2. **Uniform-random per-step team reward**, ``ep_reward / nights_played``
   where ``nights_played`` is the actual ``env.night`` counter at done — *not*
   a fixed 13. ``default_scenario`` has ``min_nights=12``; episodes can run
   slightly longer if fires are still active when night 12 ends.
3. **Random-init MLP iter-0 per-step team reward** (``JointPPOTrainer`` with
   the #183 phase-3 ``CellConfig`` defaults — ``hidden_size=64``,
   ``num_agents=4`` — seeded but never trained).

Per-step normalization uses each episode's own ``nights_played``, which is
``≥ min_nights=12`` (see ``scenarios_generated.default_scenario:83`` and the
termination check at ``bucket_brigade_env.py:303-314``).

Run locally — this is pure env stepping plus a few un-trained forward passes,
no PPO updates, ~1-2 minutes total. Safe per CLAUDE.md compute guidelines.

Usage::

    uv run python experiments/p3_specialization/diagnostics/random_baseline.py
    uv run python experiments/p3_specialization/diagnostics/random_baseline.py \\
        --episodes-per-seed 50 --seeds 1 --no-mlp   # reproduce #145 protocol
    uv run python experiments/p3_specialization/diagnostics/random_baseline.py \\
        --scenario positional_default --no-mlp      # issue #221
    uv run python experiments/p3_specialization/diagnostics/random_baseline.py \\
        --scenario chain_reaction                   # issue #219

Issue #219 added the ``SCENARIO_CITED_VALUES`` table below. For each named
scenario in the table, the verdict block compares the re-derived value against
the cited number from #145 (and, for ``default``, also against the iter-0 MLP
number from #183). For scenarios outside the table, raw measurements still
print but the verdict comparison is suppressed.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import get_scenario_by_name
from bucket_brigade.training.joint_trainer import JointPPOTrainer, flatten_dict_obs

# CellConfig defaults from experiments/p3_specialization/train.py.
# Kept in sync by hand because importing CellConfig would drag in the whole
# training stack (info_theory, torch optim, etc.) just to read three numbers.
HIDDEN_SIZE = 64
NUM_AGENTS = 4
ACTION_DIMS = [10, 2, 2]  # [house, mode, signal] (issue #235)
# Default scenario name; override via ``--scenario`` (issue #221).
DEFAULT_SCENARIO_NAME = "default"

# Per-scenario cited-value table (issue #219; extended to all 14 scenarios by #237).
#
# ``random`` is the per-step team-reward number cited in the literature. Values
# here are the **post-#236 (signal-as-first-class-action) re-derivation** from
# issue #237 run on ``COMPUTE_HOST_PRIMARY`` at commit ``dffe1060``: n=1000
# episodes per scenario (200 episodes × 5 seeds 42..46), MultiDiscrete([10, 2, 2])
# uniform sampling. Logs committed under
# ``experiments/p3_specialization/diagnostics/results/issue237_postmerge/``.
#
# ``mlp_iter0`` is the iter-0 per-step MLP team reward cited alongside it
# (e.g. the L1_norm phase-3 cell from #183) — currently only defined for
# ``default``. ``note`` records the provenance so re-derivations stay auditable.
#
# When ``--scenario`` matches an entry here, the verdict block compares the
# re-derived value against ``random`` (and, if non-None, against ``mlp_iter0``).
# All 14 named scenarios from ``bucket-brigade-core/src/scenarios.rs`` are
# present post-#237; the verdict comparison should never be suppressed for a
# named scenario.
#
# Cross-reference (issue #323): a value-only mirror of the ``random`` column
# below lives in :data:`bucket_brigade.baselines.SCENARIO_RANDOM_BASELINES`
# for non-torch consumers (this module transitively imports torch via the
# training package). ``tests/test_baselines_constants.py`` asserts the two
# stay in sync.
SCENARIO_CITED_VALUES: dict[str, dict[str, float | str | None]] = {
    "default": {
        "random": 251.23,
        "mlp_iter0": 290.52,
        "note": "#237 post-#236 (n=1000, dffe1060); MLP iter-0 from #183 (untrained path unchanged by #236)",
    },
    "easy": {
        "random": 355.07,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "hard": {
        "random": 124.66,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "trivial_cooperation": {
        "random": 399.99,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060); fixed-reward scenario, CI ≈ [399.98, 400.01]",
    },
    "early_containment": {
        "random": 297.24,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "greedy_neighbor": {
        "random": 292.78,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "sparse_heroics": {
        "random": 246.06,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060); long episodes (median 21 nights)",
    },
    "rest_trap": {
        "random": 302.87,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "chain_reaction": {
        "random": 227.39,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060); previously 220.75 pre-#236 (PR #229)",
    },
    "deceptive_calm": {
        "random": 78.55,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "overcrowding": {
        "random": 120.24,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "mixed_motivation": {
        "random": 224.06,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060)",
    },
    "minimal_specialization": {
        "random": -87.72,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060); per-agent ownership dominates (#199), sign preserved",
    },
    "positional_default": {
        "random": 250.73,
        "mlp_iter0": None,
        "note": "#237 post-#236 (n=1000, dffe1060); tracks default closely (alpha=0.1 spatial cost)",
    },
}

# Backwards-compat aliases for callers that imported these constants directly.
# Kept for one release cycle; new code should consult ``SCENARIO_CITED_VALUES``.
# Names retained for historical continuity even though the values now reflect
# the post-#237 re-derivation rather than the original #145 "308" / #183 "290".
CITED_308 = SCENARIO_CITED_VALUES["default"]["random"]
CITED_290 = SCENARIO_CITED_VALUES["default"]["mlp_iter0"]


def run_random_episode(
    env: BucketBrigadeEnv, rng: np.random.Generator
) -> Tuple[float, int]:
    """One episode of uniform-random actions. Returns (team_reward, nights_played)."""
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_reward = 0.0
    while not env.done:
        # MultiDiscrete([10, 2, 2]) per agent (issue #235).
        # Shape (N, 3) = [house_index, mode_flag, signal]. Pre-#235 was
        # (N, 2); the third column is the broadcast signal, sampled
        # independently here to fully exercise the action space.
        actions = np.stack(
            [
                rng.integers(0, 10, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
            ],
            axis=-1,
        ).astype(np.int64)
        _, rewards, _, _ = env.step(actions)
        total_reward += float(rewards.sum())
    return total_reward, int(env.night)


def run_mlp_episode(
    trainer: JointPPOTrainer, env: BucketBrigadeEnv, seed: int
) -> Tuple[float, int]:
    """One episode under a fixed (untrained) random-init MLP policy.

    Mirrors ``JointPPOTrainer._act_all`` but for one episode at a time so we
    can record the actual ``env.night`` at done for proper normalization.

    Issue #204: each agent now sees its own row with a distinct identity
    one-hot tail, matching ``JointPPOTrainer.collect_rollout``'s per-agent
    obs layout. Previously the same flat obs was fed to every policy.
    """
    import torch  # local import to keep `--no-mlp` light

    obs_dict = env.reset(seed=seed)
    n = trainer.num_agents
    obs_rows = np.stack(
        [flatten_dict_obs(obs_dict, agent_id=i, num_agents=n) for i in range(n)],
        axis=0,
    )
    total_reward = 0.0
    while not env.done:
        obs_t = torch.from_numpy(obs_rows)  # [N, obs_dim]
        joint_action = np.zeros(
            (trainer.num_agents, len(trainer.action_dims)), dtype=np.int64
        )
        with torch.no_grad():
            for i, policy in enumerate(trainer.policies):
                a, _, _, _ = policy.get_action_and_value(obs_t[i : i + 1])
                joint_action[i] = a[0].cpu().numpy()
        next_obs_dict, rewards, _, _ = env.step(joint_action)
        total_reward += float(rewards.sum())
        if not env.done:
            obs_rows = np.stack(
                [
                    flatten_dict_obs(next_obs_dict, agent_id=i, num_agents=n)
                    for i in range(n)
                ],
                axis=0,
            )
    return total_reward, int(env.night)


def bootstrap_ci(
    arr: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05
) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(np.percentile(boots, 100 * alpha / 2)), float(
        np.percentile(boots, 100 * (1 - alpha / 2))
    )


def summarize(label: str, arr: np.ndarray) -> str:
    lo, hi = bootstrap_ci(arr)
    return f"{label}: mean={arr.mean():.2f}, 95% CI=[{lo:.2f}, {hi:.2f}], n={len(arr)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--episodes-per-seed",
        type=int,
        default=200,
        help="Episodes per seed for uniform-random. #145 used 50 (single seed); "
        "default 200 across 5 seeds tightens the CI.",
    )
    ap.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds (42, 43, ...)."
    )
    ap.add_argument(
        "--mlp-episodes-per-seed",
        type=int,
        default=50,
        help="Episodes per seed for random-init MLP (slower).",
    )
    ap.add_argument(
        "--no-mlp", action="store_true", help="Skip the random-init MLP pass."
    )
    ap.add_argument(
        "--scenario",
        type=str,
        default=DEFAULT_SCENARIO_NAME,
        help=(
            "Scenario name to evaluate (e.g. 'default', 'chain_reaction', "
            "'positional_default', 'minimal_specialization'). Scenarios listed "
            "in SCENARIO_CITED_VALUES get a verdict comparison against the cited "
            "number; others print raw measurements only. See issue #219."
        ),
    )
    args = ap.parse_args()

    scenario_name = args.scenario
    scenario = get_scenario_by_name(scenario_name, num_agents=NUM_AGENTS)
    cited = SCENARIO_CITED_VALUES.get(scenario_name)
    print(
        f"Scenario: {scenario_name}, num_agents={NUM_AGENTS}, min_nights={scenario.min_nights}"
    )
    if cited is not None:
        rand_cite = cited["random"]
        mlp_cite = cited["mlp_iter0"]
        provenance = cited["note"]
        mlp_str = f", iter-0 MLP={mlp_cite}" if mlp_cite is not None else ""
        print(f"Cited values: random={rand_cite}{mlp_str} ({provenance})")
    else:
        print(
            f"Cited values: n/a for scenario '{scenario_name}' "
            "(not in SCENARIO_CITED_VALUES; raw measurements only)."
        )
    print()

    seeds = list(range(42, 42 + args.seeds))

    # ----- Uniform random -----
    rand_per_episode: list[float] = []
    rand_per_step: list[float] = []
    rand_lengths: list[int] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = BucketBrigadeEnv(scenario=scenario)
        for _ in range(args.episodes_per_seed):
            ep_reward, nights = run_random_episode(env, rng)
            rand_per_episode.append(ep_reward)
            rand_per_step.append(ep_reward / nights)
            rand_lengths.append(nights)

    rand_per_episode_arr = np.array(rand_per_episode)
    rand_per_step_arr = np.array(rand_per_step)
    rand_lengths_arr = np.array(rand_lengths)
    print("=== Uniform-random ===")
    print(summarize("  per-episode  ", rand_per_episode_arr))
    print(summarize("  per-step     ", rand_per_step_arr))
    print(
        f"  episode length: median={int(np.median(rand_lengths_arr))}, "
        f"mean={rand_lengths_arr.mean():.2f}, "
        f"min={rand_lengths_arr.min()}, max={rand_lengths_arr.max()}"
    )
    # Sanity: reproduce the 4012.34 / 13 ≈ 308.6 framing.
    print(
        f"  per-episode / median-length = {rand_per_episode_arr.mean() / np.median(rand_lengths_arr):.2f}"
    )
    print()

    # ----- Random-init MLP iter-0 -----
    mlp_per_step_arr = None
    if not args.no_mlp:
        # obs_dim from one reset. Issue #204: include the per-agent identity
        # one-hot tail (length N), giving 10 + 3N + 10 + N = 22 + 4N.
        env = BucketBrigadeEnv(scenario=scenario)
        obs_dim = flatten_dict_obs(
            env.reset(seed=0), agent_id=0, num_agents=NUM_AGENTS
        ).shape[0]
        mlp_per_step: list[float] = []
        mlp_lengths: list[int] = []
        for seed in seeds:
            # Construct an untrained trainer; we only use ``trainer.policies``.
            trainer = JointPPOTrainer(
                env_fn=lambda s=scenario: BucketBrigadeEnv(scenario=s),
                num_agents=NUM_AGENTS,
                obs_dim=obs_dim,
                action_dims=ACTION_DIMS,
                hidden_size=HIDDEN_SIZE,
                seed=seed,
            )
            mlp_env = BucketBrigadeEnv(scenario=scenario)
            for ep in range(args.mlp_episodes_per_seed):
                ep_reward, nights = run_mlp_episode(
                    trainer, mlp_env, seed=seed * 1000 + ep
                )
                mlp_per_step.append(ep_reward / nights)
                mlp_lengths.append(nights)
        mlp_per_step_arr = np.array(mlp_per_step)
        print("=== Random-init MLP (iter-0, untrained PolicyNetwork) ===")
        print(summarize("  per-step     ", mlp_per_step_arr))
        print(
            f"  episode length: median={int(np.median(mlp_lengths))}, "
            f"mean={np.mean(mlp_lengths):.2f}"
        )
        print()

    # ----- Verdict -----
    print("=== Verdict ===")
    if cited is not None:
        rand_cite = cited["random"]
        mlp_cite = cited["mlp_iter0"]
        lo, hi = bootstrap_ci(rand_per_step_arr)
        rand_agrees = lo <= rand_cite <= hi
        print(f"Uniform-random per-step CI contains cited {rand_cite}: {rand_agrees}")
        if mlp_per_step_arr is not None:
            if mlp_cite is not None:
                lo_m, hi_m = bootstrap_ci(mlp_per_step_arr)
                mlp_agrees_cite = lo_m <= mlp_cite <= hi_m
                print(
                    f"Random-init MLP per-step CI contains cited {mlp_cite}: {mlp_agrees_cite}"
                )
            mlp_agrees_rand = (
                abs(mlp_per_step_arr.mean() - rand_per_step_arr.mean()) < 5.0
            )
            print(f"Random-init MLP within ±5 of uniform-random: {mlp_agrees_rand}")
    else:
        print(
            f"(No cited verdict for scenario '{scenario_name}'; "
            "add an entry to SCENARIO_CITED_VALUES to enable.)"
        )


if __name__ == "__main__":
    main()
