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
ACTION_DIMS = [10, 2]
SCENARIO_NAME = "default"

# The widely cited number we are re-deriving.
CITED_308 = 308.0
# Iter-0 per-step team reward from issue #183's phase-3 L1_norm cell.
CITED_290 = 290.52


def run_random_episode(
    env: BucketBrigadeEnv, rng: np.random.Generator
) -> Tuple[float, int]:
    """One episode of uniform-random actions. Returns (team_reward, nights_played)."""
    env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    total_reward = 0.0
    while not env.done:
        # MultiDiscrete([10, 2]) per agent. See bucket_brigade_env.py:117 and
        # puffer_env.py:77 for the verified shape (N, 2) = [house_index, mode_flag].
        actions = np.stack(
            [
                rng.integers(0, 10, size=env.num_agents),
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
    """
    import torch  # local import to keep `--no-mlp` light

    obs_dict = env.reset(seed=seed)
    obs = flatten_dict_obs(obs_dict)
    total_reward = 0.0
    while not env.done:
        obs_t = torch.from_numpy(obs).unsqueeze(0)
        joint_action = np.zeros((trainer.num_agents, len(trainer.action_dims)), dtype=np.int64)
        with torch.no_grad():
            for i, policy in enumerate(trainer.policies):
                a, _, _, _ = policy.get_action_and_value(obs_t)
                joint_action[i] = a[0].cpu().numpy()
        next_obs_dict, rewards, _, _ = env.step(joint_action)
        total_reward += float(rewards.sum())
        if not env.done:
            obs = flatten_dict_obs(next_obs_dict)
    return total_reward, int(env.night)


def bootstrap_ci(arr: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05) -> Tuple[float, float]:
    rng = np.random.default_rng(0)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def summarize(label: str, arr: np.ndarray) -> str:
    lo, hi = bootstrap_ci(arr)
    return f"{label}: mean={arr.mean():.2f}, 95% CI=[{lo:.2f}, {hi:.2f}], n={len(arr)}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes-per-seed", type=int, default=200,
                    help="Episodes per seed for uniform-random. #145 used 50 (single seed); "
                         "default 200 across 5 seeds tightens the CI.")
    ap.add_argument("--seeds", type=int, default=5, help="Number of seeds (42, 43, ...).")
    ap.add_argument("--mlp-episodes-per-seed", type=int, default=50,
                    help="Episodes per seed for random-init MLP (slower).")
    ap.add_argument("--no-mlp", action="store_true", help="Skip the random-init MLP pass.")
    args = ap.parse_args()

    scenario = get_scenario_by_name(SCENARIO_NAME, num_agents=NUM_AGENTS)
    print(f"Scenario: {SCENARIO_NAME}, num_agents={NUM_AGENTS}, min_nights={scenario.min_nights}")
    print(f"Cited values: random={CITED_308} (issue #145), iter-0 MLP={CITED_290} (issue #183)")
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
    print(f"  episode length: median={int(np.median(rand_lengths_arr))}, "
          f"mean={rand_lengths_arr.mean():.2f}, "
          f"min={rand_lengths_arr.min()}, max={rand_lengths_arr.max()}")
    # Sanity: reproduce the 4012.34 / 13 ≈ 308.6 framing.
    print(f"  per-episode / median-length = {rand_per_episode_arr.mean() / np.median(rand_lengths_arr):.2f}")
    print()

    # ----- Random-init MLP iter-0 -----
    mlp_per_step_arr = None
    if not args.no_mlp:
        # obs_dim from one reset: flatten_dict_obs layout = 10 + 3N + 10 = 22 + 3N
        env = BucketBrigadeEnv(scenario=scenario)
        obs_dim = flatten_dict_obs(env.reset(seed=0)).shape[0]
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
                ep_reward, nights = run_mlp_episode(trainer, mlp_env, seed=seed * 1000 + ep)
                mlp_per_step.append(ep_reward / nights)
                mlp_lengths.append(nights)
        mlp_per_step_arr = np.array(mlp_per_step)
        print("=== Random-init MLP (iter-0, untrained PolicyNetwork) ===")
        print(summarize("  per-step     ", mlp_per_step_arr))
        print(f"  episode length: median={int(np.median(mlp_lengths))}, "
              f"mean={np.mean(mlp_lengths):.2f}")
        print()

    # ----- Verdict -----
    print("=== Verdict ===")
    lo, hi = bootstrap_ci(rand_per_step_arr)
    rand_agrees = lo <= CITED_308 <= hi
    print(f"Uniform-random per-step CI contains cited 308: {rand_agrees}")
    if mlp_per_step_arr is not None:
        lo_m, hi_m = bootstrap_ci(mlp_per_step_arr)
        mlp_agrees_290 = lo_m <= CITED_290 <= hi_m
        mlp_agrees_rand = abs(mlp_per_step_arr.mean() - rand_per_step_arr.mean()) < 5.0
        print(f"Random-init MLP per-step CI contains cited 290.52: {mlp_agrees_290}")
        print(f"Random-init MLP within ±5 of uniform-random: {mlp_agrees_rand}")


if __name__ == "__main__":
    main()
