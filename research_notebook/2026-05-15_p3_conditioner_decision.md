---
title: P3 CMI Conditioner — Architect Decision
date: 2026-05-15
status: decided
relates_to: ["issue #172", "issue #154", "PR #168"]
companion: 2026-05-14_p3_specialization_results.md
---

# P3 CMI Conditioner — Architect Decision

## TL;DR

**Decision:** Keep Option 1 (state-summary conditioner: `num_burning + 11 * day_bin`) as the **primary** conditioner. Add Option 3 (other-agent lagged action) as a **secondary sensitivity check**, reported alongside in the same measurement pass. Drop Option 2 (trajectory bucket) — its near-deterministic-reward failure mode is the same pathology that motivated #154 in the first place.

This satisfies Acceptance Criterion #1 of #172 ("Architect documents rationale"). Implementation is a small Builder follow-up — see "Implementation scope" below.

## The scientific claim, stated precisely

P3 measures whether agents under a redundancy penalty develop **specialized encoder outputs** — i.e., whether `Ẑ_i` and `Ẑ_j` (the two agents' projected encoder activations) carry **different** information about whatever they're tracking. The pre-registered falsifier F1 reads CMI(`Ẑ_i`; `Ẑ_j` | Z) as a function of `λ_red`: if the redundancy term is doing work, CMI should monotonically decrease with `λ_red`.

The choice of conditioning variable Z **defines what "different" means**. Each candidate answers a different scientific question:

- **Option 1** (state summary): "specialized beyond what the *shared environment state* forces both encoders to track"
- **Option 2** (trajectory bucket): "specialized beyond what the *shared reward trajectory* forces both encoders to track"
- **Option 3** (lagged other-agent action): "specialized beyond what their *observed coordination* forces both encoders to track"

These are not interchangeable. They are three different empirical claims, and which one we make is a research call — not a code-level pick.

## Why Option 1 as primary

Option 1 conditions on **exogenous** environment state. `(num_burning, day_bin)` is upstream of both encoders: it is a property of the world both agents observe, not a function of either encoder's output. Conditioning on an exogenous variable cleanly removes "redundancy that exists because both agents are looking at the same world" without removing redundancy that exists for other reasons (e.g., both encoders converging on the same compression of the same observations, or both tracking a feature that isn't in the conditioner).

Option 1's degenerate cases are well-bounded and already guarded:
- `is_degenerate_conditioner` (`train.py:189-207`, landed in PR #149) fires if `(num_burning, day_bin)` collapses to near-constant on a future scenario. The May 14 results (`2026-05-14_p3_specialization_results.md`) confirm the guard does **not** fire on any of `trivial_cooperation`, `default`, `chain_reaction` post-#168.
- The "every episode ends at step 0" edge case noted in `train.py:187` would collapse both components, but no current scenario exhibits this.

## Why Option 3 as sensitivity check, not primary

Option 3 conditions on the **other agent's lagged action**. This is **downstream** of the other agent's encoder output: action `A_j[t-1]` is a function (via the policy head) of `Ẑ_j[t-1]`. Conditioning on a downstream signal partially conditions on the very variable we are trying to detect — it can suppress CMI(`Ẑ_i`; `Ẑ_j` | Z) for the wrong reason and produce artificially low values that look like a successful specialization signal.

That circularity makes Option 3 unsuitable as the **primary** measurement. But it is informative as a **secondary**: if Option 1 and Option 3 disagree, the disagreement *itself* is diagnostic. Specifically:

- **Both low** (Option 1 CMI low, Option 3 CMI low): strongest reading — encoders are specialized in a way that survives conditioning on both shared environment *and* observed coordination. Robust positive.
- **Option 1 low, Option 3 high**: encoders are specialized w.r.t. environment state, but their joint information is mediated through coordination (action observations). Plausible specialization story, weaker.
- **Option 1 high, Option 3 low**: suspicious — Option 3's circularity is artificially suppressing CMI. Treat the Option 3 number as a measurement artefact, not a result.
- **Both high**: no specialization signal under any conditioner. Falsified.

This four-way disambiguation is genuinely worth the cost-1 second measurement.

## Why Option 2 is dropped

Option 2 (sliding-window reward mean, quantized) inherits the **near-deterministic-reward failure mode** that motivated #154. On `trivial_cooperation`, rewards are functionally constant in `λ_red`; the trajectory bucket collapses to near-constant; `I(Ẑ_i; Ẑ_j | Z)` numerically equals `I(Ẑ_i; Ẑ_j)`. This is exactly the pathology #154 was filed to fix. Option 2 has no advantage over Option 1 on the other scenarios and a known failure mode on the easy scenario.

## Caveat to record alongside results

Option 1 has a subtle **under-reporting** failure mode that is worth stating explicitly in any results writeup: if both encoders happen to compress *exactly* `(num_burning, day_bin)` and nothing else, conditioning on `(num_burning, day_bin)` removes everything they share. CMI drops to zero. This reads as "specialization" but really means "both encoders converged on a low-dimensional summary that the conditioner happens to fully capture."

This is not a bug in the conditioner — it is an intrinsic limitation of any conditional measurement with a coarse conditioner. The honest interpretation of "Option 1 CMI low" is therefore: *"specialization beyond what `(num_burning, day_bin)` explains."* Not: *"specialization, full stop."*

The Option 3 sensitivity check partially mitigates this — if both encoders truly compress to `(num_burning, day_bin)`, Option 3's CMI will *also* drop (because both agents' actions are then state-determined), and the diagnostic table above flags the "both low" outcome as the strongest specialization reading. But fundamentally this is a measurement-floor problem inherent to the question; the right response is documentation, not a different conditioner.

## Implementation scope (Builder follow-up)

Small. ~30-45 minutes of work:

1. In `experiments/p3_specialization/train.py`:
   - Add `_other_agent_action_codes(rollout, lag=1)` — for each pair `(i, j)`, the conditioner for agent `i`'s measurement against agent `j` is `j`'s packed action at `t-1` (using the existing pack: `a[:, 0] * 2 + a[:, 1]`, range 0..19). Handle `t=0` defensively (no prior action — either drop the first step or assign a sentinel code).
   - In `_measure_information`, after the existing Option 1 block, compute a second CMI per pair: `cmi_action_cond = conditional_mutual_information(codes[i], codes[j], action_codes_for_pair_ij)`. Emit as `cmi_action/agent_{i}_{j}` and `cmi_action/mean_pair`.
   - Apply `is_degenerate_conditioner` to the action-codes conditioner as well and emit `cmi_action/conditioner_*` diagnostics. The action-codes conditioner has a degenerate mode of its own (e.g., one agent always taking the same action), which would silently collapse Option 3 the same way the team-reward conditioner used to collapse Option 1.
   - Remove the "Builder-pick" caveat comments at `train.py:114-118` and `:162-165`. Replace with a one-line pointer: *"Architect-validated; see `research_notebook/2026-05-15_p3_conditioner_decision.md`."*

2. No changes needed to `joint_trainer.py` — the training-side penalty is a separate methodological gap (per #146 curator note) and out of scope for #172.

3. Acceptance check before merge:
   - `is_degenerate_conditioner` does not fire on either conditioner on `default`, `chain_reaction`, `trivial_cooperation` at λ=0 on a short sandbox run (10 iters × 512 steps is enough — we are checking the conditioner's marginal distribution, not the CMI trend).
   - Both `cmi/mean_pair` and `cmi_action/mean_pair` are emitted in the per-iteration JSON log and consumed by `analyze.py` aggregation. (If `analyze.py` hardcodes the metric key, that aggregation update is part of this PR's scope.)

4. **Not** in scope for this PR:
   - Re-running the full sweep with the new conditioner (that's a sandbox-1 job, separate from this code change).
   - Updating the writeup in `2026-05-14_p3_specialization_results.md` with the new numbers (post-sweep, separate notebook).
   - The training-side penalty methodology gap from #146.

## What this does *not* settle

This decision is about the **measurement** conditioner. The training-side redundancy penalty in `bucket_brigade/training/joint_trainer.py:414-420` is a linear cross-correlation surrogate (Frobenius), not the CMI being measured. That mismatch is real and is worth its own architect pass eventually, but it is a separate question from #172. Tracked in #146's curator note; not addressed here.

The May 14 sweep falsified F1 under the Option 1 conditioner alone. Adding Option 3 will give us a second view but is not expected to overturn the falsification on these three scenarios. If the goal is to *un-falsify* F1, the lever to pull is the training-side penalty design (#146), not the conditioner.
