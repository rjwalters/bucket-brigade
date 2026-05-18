"""Minimal 2-player iterated public-goods dilemma — toy basin-trap demo (issue #292).

This sub-package implements a referee-friendly Python-only reproduction of the
basin-trap signature observed in the bucket-brigade env (issues #270, #271).
The demo target (post-#271 reframing): PPO from random init converges to
mutual-defect on a 2-player iterated public-goods game where the cooperative
equilibrium yields strictly higher reward; BC-init from `always_cooperate`
holds the cooperative basin under PPO continuation; best-of-N random nets do
NOT bridge to the cooperative basin under the K=20 → K=200 stability protocol
validated in #271.

Modules:
    env: 2-player iterated public-goods env (Option A; m=1.6, 50-step episodes).
    specialists: hand-coded `always_cooperate` and `tit_for_tat` callables.
    train_ippo: thin wrapper invoking `JointPPOTrainer` on the dilemma env.
    bc_init: BC pipeline adapted from `bc_init.py` for the dilemma env.
    best_of_n: random-init best-of-N with K=20 → K=200 stability re-eval.
    verdict: 4-gate verdict classifier (basin_trap vs anti_attractor vs null).

See issue #292 (`gh issue view 292 --comments`) for full spec and `run_all.sh`
for the orchestration entry point. Heavy runs go on `COMPUTE_HOST_PRIMARY`.
"""
