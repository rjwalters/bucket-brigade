# Bucket Brigade — Environment API

Reference for the public `bucket_brigade` Python entry points: `make()`,
`make_vec()`, `list_envs()`, and the versioned scenario registry. This is
the *programming surface*. The *mathematical contract* lives in
[`paper/anvil_memo.env_spec.1/env_spec.md`](../paper/anvil_memo.env_spec.1/env_spec.md)
(paper §2) — read that for the formal definition of states, dynamics,
and rewards. This document covers only what you need to wire the env
into a training pipeline or evaluation harness.

## Quick reference

```python
import bucket_brigade

# 1. Pick a scenario.
print(bucket_brigade.list_envs())
# ['chain_reaction-v1', 'deceptive_calm-v1', 'default-v1', ...]

# 2. Make a single env.
env = bucket_brigade.make("minimal_specialization-v1")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

# 3. Or a synchronous batched vector env (issue #370).
vec = bucket_brigade.make_vec("minimal_specialization-v1", num_envs=8)
obs, info = vec.reset(seed=0)  # obs.shape == (8, obs_dim)
```

Both `make()` and `make_vec()` accept an optional `num_agents` override.
Overriding produces an env that is **not covered by the frozen scenario
ID's reproducibility guarantee**; record the override in your experiment
metadata if you use it.

## Versioned scenario registry

Frozen scenario IDs are the **reproducibility surface**. Every paper
result is keyed on an ID like `minimal_specialization-v1`, and the
registry is **append-only in the version direction**: once shipped, a
`-vN` ID never changes its meaning. If a scenario's parameters, obs/action
shape, or RNG sequence change, the new behavior gets a new `-vN+1` ID
and the old one stays frozen.

See [`bucket_brigade/envs/registry.py`](../bucket_brigade/envs/registry.py)
for the full version-bump policy and the live registry table.

| Scenario ID | Brief |
|---|---|
| `default-v1` | 10 houses, 4 agents, mid-difficulty baseline |
| `hard-v1` | High-pressure variant of `default` |
| `trivial_cooperation-v1` | Sanity-check scenario — cooperation is trivially optimal |
| `early_containment-v1` | Tests early fire suppression |
| `greedy_neighbor-v1` | Owner-only reward — encourages neighbor neglect |
| `sparse_heroics-v1` | Rare high-payoff cooperative moments |
| `rest_trap-v1` | Asymmetric Nash — three free-riders + one firefighter ([see PAPER_RESULTS.md](PAPER_RESULTS.md#3-heterogeneous-nash-equilibria)) |
| `chain_reaction-v1` | Tests fire-spread containment |
| `deceptive_calm-v1` | Tests handling of misleading low-fire periods |
| `overcrowding-v1` | Many agents competing for few targets |
| `mixed_motivation-v1` | Mixed individual + team incentives |
| `minimal_specialization-v1` | P3 specialization diagnostic — symmetric all-Hero NE ([see PAPER_RESULTS.md](PAPER_RESULTS.md#3-heterogeneous-nash-equilibria)) |
| `v2_minimal-v1` | 2-house topology, PPO learnability diagnostic (#254) |
| `positional_default-v1` | Positional-reward variant of `default` — PPO baseline (#384) + frozen-baseline release (#371) |

`bucket_brigade.list_envs()` returns the live sorted list. The default
number of agents for every frozen ID is **4**
(`bucket_brigade.DEFAULT_NUM_AGENTS`).

## Observation space

`env.observation_space` is a `gymnasium.spaces.Box` of dtype `float32`.
At the default parameterization (`num_agents=4`, ring size depends on
scenario), the flat observation vector — per agent, concatenated along
the joint-controller view — packs the following channels in order:

| Channel | Shape | Source |
|---|---|---|
| `houses` | `(H,)` | House status one-hot/encoding per house |
| `signals` | `(N,)` | Previous-night signals broadcast by each agent |
| `locations` | `(N,)` | Previous-night agent locations |
| `last_actions` | `(2N,)` | Previous-night `(house, mode)` per agent |
| `scenario_info` | varies | Static scenario parameters (work cost, etc.) |
| `identity` | `(N,)` | One-hot identity tail (per-agent in the multi-agent flattener) |

The joint-controller wrapper (used by `make()` / `make_vec()`)
concatenates `num_agents` of these per-agent vectors into a single flat
observation. The shape is published on `env.observation_space.shape`
— do not hard-code it; the layout is stable per frozen ID but the exact
dimension depends on `(H, N)`.

For the formal observation tuple $o_t = (h_t, \ell_{t-1}, \zeta_{t-1},
\alpha_{t-1}, t)$ and the underlying state, see
[paper/anvil_memo.env_spec.1/env_spec.md §2.3](../paper/anvil_memo.env_spec.1/env_spec.md).

## Action space

`env.action_space` is a `gymnasium.spaces.MultiDiscrete`. Per-agent
action layout is `[house, mode, signal]` (length 3):

| Component | Range | Meaning |
|---|---|---|
| `house` | `[0, H)` | Target house index |
| `mode` | `{0, 1}` | `0` = REST, `1` = WORK |
| `signal` | `{0, 1}` | Broadcast bit (observable next night) |

The joint action layout for `N` agents is
`[house_0, mode_0, signal_0, house_1, mode_1, signal_1, ...]`
(length `3N`). At `num_agents=4`, `num_houses=10` (the default), this is
`MultiDiscrete([10, 2, 2, 10, 2, 2, 10, 2, 2, 10, 2, 2])`.

The signal bit is **strictly informational** — it does not constrain the
agent's `mode` or `house` choice, and other agents cannot reject it.
Signals support cheap-talk equilibria; their interpretation is not part
of the game spec
([env_spec §2.2](../paper/anvil_memo.env_spec.1/env_spec.md)).

## Reward

`env.step()` returns a scalar `reward` — the **team-summed reward**
across all agents for the night. Per-agent rewards are surfaced
losslessly via `info["per_agent_rewards"]`, a length-`N` `numpy.ndarray`,
so multi-agent consumers retain the full decomposition.

The per-agent reward decomposes into a private work/rest cost, a
per-house ownership term, and a team-welfare term; the formal
specification is in
[env_spec §4](../paper/anvil_memo.env_spec.1/env_spec.md).

## Episode lifecycle

- `reset(seed=int) -> (obs, info)` — standard Gymnasium contract; `info`
  on reset contains at minimum `scenario_id`.
- `step(action) -> (obs, reward, terminated, truncated, info)` —
  `truncated` is always `False` (the env signals end-of-episode via
  `terminated` only; episodes are bounded by `min_nights` + natural
  termination, not a fixed time cap).
- `info["scenario_id"]` is set on every `reset` and `step` for
  traceability of downstream artifacts.

Episode-termination conditions and the `T_min` clamp are documented in
[env_spec §5.1](../paper/anvil_memo.env_spec.1/env_spec.md).

## Vectorized env (`make_vec`)

`bucket_brigade.make_vec(id, num_envs)` returns a
`bucket_brigade.envs.vector.SyncVectorEnv` — a synchronous batched
wrapper around `num_envs` independent `BucketBrigadeGymEnv` instances
built from the same versioned scenario ID. The vectorized API follows
Gymnasium's batched conventions with auto-reset semantics: terminal
observations surface via `info["final_observation"]` and
`info["final_info"]`. See
[`bucket_brigade/envs/vector.py`](../bucket_brigade/envs/vector.py) for
implementation details and the issue #370 motivation.

```python
vec = bucket_brigade.make_vec("default-v1", num_envs=8)
obs, info = vec.reset(seed=42)
# obs.shape == (8, obs_dim); per-sub-env actions are stacked the same way
actions = np.stack([vec.action_space.sample() for _ in range(8)])
obs, rewards, terminated, truncated, info = vec.step(actions)
```

## Compatibility notes

- The adapter is a `gymnasium.Env`. Stable-Baselines3, CleanRL, Tianshou,
  and similar **single-agent** pipelines consume it directly.
- The adapter is a **joint-controller** single-agent view. External
  multi-agent researchers who want per-agent reward streams should read
  `info["per_agent_rewards"]` (preserved verbatim).
- A PettingZoo Parallel surface is **not** shipped in this slice (see
  issue #369 scope notes). The native multi-agent surface remains
  available via `bucket_brigade.envs.bucket_brigade_env.BucketBrigadeEnv`
  for in-repo research; the Gym adapter is intentionally minimal so
  external pipelines work without Rust extension builds.
- The Gym adapter does **not** require the Rust `bucket_brigade_core`
  extension. Pure-Python users get a fully functional env from
  `pip install bucket-brigade` alone. The Rust extension stays optional
  and gives ~100× throughput for evolution/Nash workloads (see
  [README.md Quickstart](../README.md#quickstart)).

## See also

- **Formal spec (paper §2)**: [`paper/anvil_memo.env_spec.1/env_spec.md`](../paper/anvil_memo.env_spec.1/env_spec.md)
- **Game-mechanics walkthrough**: [`docs/game_mechanics.md`](game_mechanics.md)
- **Design philosophy**: [`docs/game_description.md`](game_description.md)
- **Paper results**: [`docs/PAPER_RESULTS.md`](PAPER_RESULTS.md)
- **Registry implementation**: [`bucket_brigade/envs/registry.py`](../bucket_brigade/envs/registry.py)
- **Gym adapter**: [`bucket_brigade/envs/gym_adapter.py`](../bucket_brigade/envs/gym_adapter.py)
- **Vector env**: [`bucket_brigade/envs/vector.py`](../bucket_brigade/envs/vector.py)
