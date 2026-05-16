# Env-health diagnostics (issue #201)

Three standalone scripts plus an aggregator + pytest regression suite that
together answer the question:

> Is the env in a state where independent PPO *could plausibly* learn?

If any of the H-checks below regress, gradient signal has been silently
broken — usually by an env-side change (reward rebalance, new scenario,
ownership-vector refactor, etc.).

---

## Scripts

| File | Hypothesis | What it measures |
|------|------------|------------------|
| `inspect_rollout_rewards.py` | **H1** (#190) | Per-agent reward CV + action-reward R² from one rollout. |
| `audit_reward_attribution.py` | **H2** (#191) | Per-step decomposition into (team, ownership, work_cost) + pairwise reward correlation. |
| `random_baseline.py` | **H3** (#192) | Uniform-random per-step team reward; re-derives `default` (cited 308 → post-#197/#198 247.58) and `chain_reaction` (cited 233 → re-derived; issue #219). Other scenarios via `--scenario`. |
| `check_env_health.sh` | aggregator | Runs H1+H2+H3 sequentially, tees logs, prints a summary table. |
| `issue199_baselines.py` | (separate) | Specialist baseline harness for issue #199. |

### H1 (hermetic mode)

`inspect_rollout_rewards.py` originally rsynced a trained cell into
`/tmp/h1_cell`. The `--hermetic` flag added in #201 skips that path and
runs only a random-init rollout against a freshly-constructed
`JointPPOTrainer`. This is what the aggregator and the pytest regression
test use; the trained-cell path is still available for ad-hoc
investigation.

```bash
# Aggregator — what you want 90% of the time.
bash experiments/p3_specialization/diagnostics/check_env_health.sh           # default scenario
bash experiments/p3_specialization/diagnostics/check_env_health.sh minimal_specialization

# Just the regression bar (asserts thresholds, fails CI-style):
uv run pytest tests/test_env_health_diagnostics.py --run-slow -v
```

---

## Thresholds (the "plausibly trainable" rubric)

These match `tests/test_env_health_diagnostics.py` exactly — single source
of truth lives in that file's module-level constants.

| Check | Threshold | Meaning of a regression |
|-------|-----------|-------------------------|
| H1: per-agent CV | `> 0.05` for **any** agent | Reward variance has collapsed; PPO sees a flat signal. |
| H1: action-reward R² | `> 0.01` for **any** agent | Actions don't move reward at all; gradient is noise. |
| H2: median \|team\|/\|ownership\| ratio | `< 5×` | Team term dominates per-agent reward; nothing left to specialize on. |
| H2: min pairwise reward correlation | `< 0.95` | All four agent reward streams are basically the same scalar. |
| H3: uniform-random per-step reward (default) | `∈ [220, 290]` | Env reward magnitudes have shifted from the post-#197/#198 baseline (~250). |

H1 is an "OR" gate per agent (CV or R²) because either side is enough to
prove the signal isn't degenerate. H2 is two independent "AND" gates
(ratio AND correlation) because both pathologies can independently kill
PPO. H3 is an absolute scale check on `default` only — `random_baseline.py`
hard-codes that scenario because it re-derives a specific cited number.

### Why these thresholds and not stricter ones

We want this suite to catch *order-of-magnitude* regressions, not
statistical jitter. The committed result JSONs under `results/` show
typical values well inside the bounds:

- `h2_reward_attribution.json` (default): median ratio = 3.0, min corr = 0.71
- `h2_reward_attribution_minimal_specialization.json`: median ratio = 0.12,
  min corr = 0.04

Tightening past those values would couple the test to seed-level noise.

---

## Runtime budget

- `check_env_health.sh default` — under 2 min wall time on a Mac Studio (CPU).
- `pytest tests/test_env_health_diagnostics.py --run-slow` — under 1 min;
  H1 fixtures use 1024 rollout steps (vs the script's 2048) to keep it fast
  while preserving statistical signal.

Both are gated behind `slow` because even sub-second-per-call workloads
add up across the matrix and the suite is run on-demand, not per-commit.

---

## Non-goals

- **CI fast-lane integration** — see issue #201 Layer 3. The default CI
  lane uses `-m "not slow and not integration"`; this suite is correctly
  excluded from that lane.
- **Auto-running on every env change** — these are diagnostic tools, not
  acceptance gates. Run them when you change reward shaping or add a
  scenario.
- **Negative-case automated test** — verifying the suite fails on a
  deliberately broken scenario is a one-shot human-driven check, not
  worth the maintenance cost of fixturing.

---

## Adding new scenarios to the regression suite

1. Add the scenario name to the two parametrize lists in
   `test_h1_reward_signal_not_degenerate` and the H2 tests.
2. Add entries to the `H1_MIN_CV_OR_R2`, `H2_MAX_TEAM_TO_OWN_RATIO`, and
   `H2_MAX_MIN_PAIRWISE_CORR` dicts at the top of the test file.
3. Run `bash check_env_health.sh <new_scenario>` once to confirm the
   numbers are in the same regime, then commit.

H3's range is `default`-specific by design and does not generalize across
scenarios — extending it would require re-running the protocol from
issue #145 on each new scenario.

---

## Related issues / PRs

- #190, #191, #192 — the three diagnostic hypotheses (H1/H2/H3).
- PR #194 (H2), PR #195 / commit `2130586d` (H1), PR #196 / commit `ec6e521c` (H3) — original commits.
- #197 (team-vs-ownership rebalance), #198 (per-agent ownership vectors),
  #199 (minimal_specialization scenario) — the env changes this suite is
  meant to catch silently regressing.
