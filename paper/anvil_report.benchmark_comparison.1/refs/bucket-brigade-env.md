# In-repo reference: Bucket Brigade environment specification

This file collects load-bearing source citations from the Bucket Brigade repository for the Findings in `../report.md`.

## Action space (Finding 1)

Per-agent action: `MultiDiscrete([num_houses, 2, 2])`, where the three sub-actions are `(house, mode, signal)`.

- Source: `bucket_brigade/envs/scenarios_generated.py` line 85 (comment):
  > wrappers read ``scenario.num_houses`` to size their action space
  > (``MultiDiscrete([num_houses, 2, 2])``) and observation arrays.
- Source: `bucket_brigade/baselines/specialist.py` line 81 (docstring):
  > ``MultiDiscrete([num_houses, 2, 2])`` action space (issue #235).
- Source: `bucket_brigade/baselines/__init__.py` line 41:
  > measured with the post-#236 3-dim sampler ``MultiDiscrete([10, 2, 2])``

Concrete per-agent action cardinalities:
- `v2_minimal` (num_houses=2): 2 · 2 · 2 = **8**
- default (num_houses=10): 10 · 2 · 2 = **40**

## State space (Finding 2)

Per-timestep state at the env level:

- `houses: np.int8[num_houses]` in `{SAFE=0, BURNING=1, RUINED=2}` — source: `bucket_brigade/envs/bucket_brigade_env.py` lines 23–26, 53–55.
- `locations: np.int8[num_agents]` ranging over house indices — source: same file, line 56.
- `signals: np.int8[num_agents]` in `{REST=0, WORK=1}` — source: same file, lines 28–29, 57–59.

State cardinality (ignoring `last_actions`, which is a redundant cache of the previous action):
`|S| = 3^H · H^A · 2^A`

- `v2_minimal` (H=2, A=4): `3^2 · 2^4 · 2^4 = 9 · 16 · 16 = 2304`
- default (H=10, A=4): `3^10 · 10^4 · 2^4 = 59049 · 10000 · 16 ≈ 9.45 × 10^9`

This is the *enumerable* state cardinality. The dynamics over this state space are stochastic (fire spread / burn-out are Bernoulli per `bucket_brigade/envs/bucket_brigade_env.py` later sections), so the *trajectory* space is larger, but the per-step state space remains polynomial in (H, A).

## Scenario parameterization (Finding 6)

The per-scenario data class `Scenario` (defined in `bucket_brigade/envs/scenarios_generated.py`) carries the load-bearing scalars used by the phase-diagram sweep in issue #358. The parameter triple `(β, κ, c)` referenced in Finding 6 corresponds to:

- β = fire-spread probability per timestep per neighbouring burning house
- κ = burn-out probability per timestep for a burning house transitioning to RUINED
- c = per-step work cost incurred when an agent's signal is WORK

See `bucket_brigade/envs/scenarios_generated.py` (auto-generated from `definitions/scenarios.json`) for the precise field names. The NE structure observed in the hetero-DO sweep (#355) splits `v2_minimal` (Hero NE dominates) from rest-trap (only NE is 3 FreeRider + 1 FullForce) as a function of these scalars.

## Number-of-agents constraints (table row, Finding 1)

- Default: 4. Source: `bucket_brigade/envs/bucket_brigade_env.py` line 32 (`def __init__(self, scenario: Optional[Scenario] = None, num_agents: int = 4)`).
- Allowed range: 4–10 per the docstring comment on the same line: "Number of agents (4-10)".

## NE-characterization tracker references

- Issue #358: NE phase diagram across (β, κ, c) grid (compute-bound, in progress).
- Issue #359: Analytical NE characterization for 4-agent Bucket Brigade (theoretical, in progress).
- Issue #355: Hetero-DO sweep on 2 cells; output cited in Findings 3, 4.
- Issue #356: P3 specialization research wall; output cited in Finding 7.
- Issue #357: Workshop-paper roadmap (tracker).
