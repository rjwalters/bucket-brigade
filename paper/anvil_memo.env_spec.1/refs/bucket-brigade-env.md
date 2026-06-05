# Source: in-repo Bucket Brigade environment

In-repo references for the formal env specification:

- `bucket_brigade/envs/bucket_brigade_env.py` — the canonical Python reference implementation. Class `BucketBrigadeEnv`. The `step()` method (lines ~177–277) encodes the seven-phase night ordering: signal write, location/mode write, extinguish, burn-out, spread, spontaneous ignition, reward.
- `bucket_brigade/envs/scenarios_generated.py` — `Scenario` dataclass (lines ~17–202) defining the parameter family $(\beta, \kappa, \rho, c, H, N, T_{\min}, W)$ plus the optional variant parameters (action-validity mode, extinguish mode, commitment mode, distance-cost asymmetry).
- `bucket-brigade-core/src/engine/phases.rs` — Rust kernel mirroring the Python phase dispatch in bit-exact parity under seeded RNG.
- `bucket-brigade-core/src/engine/core.rs` — top-level `step` and `step_two_phase` dispatchers in Rust.
- `bucket-brigade-core/src/scenarios.rs` — Rust validator for scenario parameter allowlists (`ALLOWED_DISTANCE_METRICS`, `ALLOWED_TEAM_WELFARE_KINDS`, `ALLOWED_ACTION_VALIDITY_MODES`, `ALLOWED_EXTINGUISH_MODES`, `ALLOWED_COMMITMENT_MODES`).

Implementation invariants relied on by the specification:

- The seven-phase order is load-bearing. Reordering changes the transition kernel observably (the spread phase in particular composes nontrivially with burn-out — see §3.4 of the spec).
- The extinguish formula uses the independent-workers model: $P(\text{extinguish} \mid w \text{ workers}) = 1 - (1-\kappa)^w$. The implementation realizes this as a single Bernoulli draw at each BURNING house with probability $1 - (1-\kappa)^w$.
- Default rest payoff is $c_{\text{rest}} = 0.5$ (hardcoded at line 669 of `bucket_brigade_env.py`); the work cost $c$ is the scenario parameter `cost_to_work_one_night`.
- House ownership is round-robin by default: $\text{owner}(h) = h \bmod N$ (line 103 of `bucket_brigade_env.py`).
- Ruin penalty is applied **every step** the house is RUINED, not just on the ruin transition (lines 700–705 of `bucket_brigade_env.py`).

Issue tracker:

- #357 — workshop paper tracker (M3.1).
- #358 — NE phase diagram across $(\beta, \kappa, c)$.
- #359 — analytical NE characterization for 4-agent Bucket Brigade.
- #362 — this issue (formal environment specification).
- #363 — comparative benchmark report (merged in PR #374).
