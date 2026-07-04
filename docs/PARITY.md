# Verifying Reward-Scale Parity

**If you consume Bucket Brigade outside this repo — a fork, a re-binding, a
re-implementation, or an external evaluation harness — run the parity check
before scoring anything.**

## Why

PR #432 documented the motivating incident: a downstream harness ran a full
study of this benchmark and drew conclusions from numbers that were **~7x**
larger than repo-native values — a differently-weighted reward
configuration. Nothing failed loudly; the mismatch was only exposed later by
a dedicated repo-side oracle. Results about Bucket Brigade produced
off-scale are worse than no results.

The parity check converts that class of silent config mismatch into an
immediate, loud, seconds-cheap failure.

## What it checks

1. **Reward scale (statistical)** — rolls out a few hundred uniform-random
   episodes against your installed env and compares the measured per-step
   team reward to the canonical reference
   (`bucket_brigade.baselines.SCENARIO_RANDOM_BASELINES`, derived at n=1000
   episodes per scenario under issue #237). Tolerance is derived from the
   committed measurement CI, not a made-up epsilon.
2. **Scenario parameters (fingerprint)** — hashes the resolved `Scenario`
   dataclass and compares against the pinned manifest fingerprint. This
   catches parameter drift even when the reward scale happens to coincide.

## How to run

Seconds-cheap (a few hundred pure env episodes, no training, no Nash
computation) — safe to run locally per the CLAUDE.md compute guidelines.

```bash
# One scenario:
python -m bucket_brigade.baselines.parity --scenario-id rest_trap-v1

# Every manifest scenario (~20 seconds):
python -m bucket_brigade.baselines.parity --all

# Tighter check / machine-readable output:
python -m bucket_brigade.baselines.parity --scenario-id default-v1 \
    --episodes 2000 --json
```

Pass looks like:

```
PARITY OK   rest_trap-v1: observed per-step random team reward 301.07 vs
expected 302.87 — observed/expected ratio = 0.994; |diff| 1.80 <= tolerance
11.91 (z=3, n=500)
```

A #432-class mismatch looks like (non-zero exit code):

```
PARITY FAIL default-v1: observed per-step random team reward 1725.71 vs
expected 251.23 — observed/expected ratio = 6.869; |diff| 1474.48 > tolerance
17.93 (z=3, n=500). Your build/binding is likely on a different reward scale
than the canonical scenario (see docs/PARITY.md and PR #432 for the
motivating 7x incident).
```

Exit codes: `0` all checks passed, `1` parity or fingerprint mismatch,
`2` usage error (e.g. unknown scenario ID).

## The reference manifest

```bash
python -m bucket_brigade.baselines.parity --manifest > parity_manifest.json
```

The manifest is keyed by **frozen scenario ID**
(`bucket_brigade/envs/registry.py`; e.g. `rest_trap-v1`) and carries, per
scenario: the canonical uniform-random per-step team reward, its n=1000 95%
bootstrap CI, and the scenario fingerprint. A top-level
`measurement_convention` block records exactly how the reference numbers
were measured. Drift guards in `tests/test_parity.py` keep the manifest
aligned with `SCENARIO_RANDOM_BASELINES` and `SCENARIO_VERSIONS`.

### Manifest version history

- **v1** (issue #437): initial manifest.
- **v2** (issue #447): the `Scenario` dataclass gained the `reward_rest`
  field (default `0.5`), promoting the per-step rest reward — historically
  a hardcoded `+0.5` in the reward implementations — to a scenario weight.
  Every reward term is now a scenario parameter, so
  `definitions/scenarios.json` fully determines the reward surface and
  reward-weight scaling is exact. **All pinned scenario fingerprints
  changed** (they hash the resolved dataclass); reward behavior and every
  reference baseline are bit-identical, and the frozen `-v1` scenario IDs
  are unchanged (bit-exact refactors do not require a new `-vN` ID per the
  registry version-bump policy). Consumers pinned to v1 fingerprints
  should re-pin against v2.

## For re-implementations (no Python `Scenario` available)

Reproduce the measurement convention from the manifest and compare your own
measurement against `random_per_step_team`:

- Policy: uniform random over `MultiDiscrete([num_houses, 2, 2])`
  (`[house, mode, signal]`), sampled independently per agent per step,
  `num_agents = 4`.
- Per-step team reward: total episode team reward (summed over all agents
  and all steps) divided by the number of nights actually played
  (`env.night` at termination — episodes end naturally via the
  `min_nights` + no-active-fire rule, **not** at a fixed night count).
- A few hundred episodes puts your standard error well inside the
  manifest CI; anything like a 2x (let alone 7x) ratio is unmissable.

The fingerprint check additionally requires constructing the Python
`Scenario` (`bucket_brigade.baselines.parity.scenario_fingerprint`); pure
re-implementations should instead verify scenario parameters field-by-field
against `definitions/scenarios.json` / `bucket-brigade-core/src/scenarios.rs`.

## Reporting results

When publishing numbers about this benchmark, cite the **frozen scenario ID**
(e.g. `rest_trap-v1`, never just "rest_trap") and the **manifest version**
(`manifest_version` in the manifest JSON, currently 2), and state that the
parity check passed on the build that produced your results.
