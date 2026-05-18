"""4-gate verdict classifier for the minimal dilemma (issue #292).

Consumes the JSON outputs of the IPPO baseline, best-of-N, and BC-init
arms and emits one of three verdicts:

- ``basin_trap_replicated`` — all 4 gates pass; the minimal env reproduces
  the bucket-brigade basin-trap signature.
- ``anti_attractor_in_toy`` — gate 3 fails (BC-init decays under PPO
  continuation); the toy env exhibits an actively misleading gradient.
  Publishable as a falsification (re-opens the anti-attractor thesis).
- ``null_result`` — at least one of the supporting gates (1, 2, or 4)
  fails, leaving the basin-trap reading unresolved.

The 4 gates (from the curator's success criterion):

1. **IPPO converges to defection** — mean per-step per-agent reward at
   final iter within ``[-0.05, 0.10]`` (≈ mutual-defect = 0.0) AND
   cooperation fraction < 10%. Averaged over ``>=`` 5 seeds.
2. **Specialist dominates** — ``always_cooperate`` self-play yields per-
   step per-agent reward >= 0.5 (mutual-C = 0.6). This is a pure
   analytic check — read from the env spec, not from a trained model.
3. **BC-init holds the basin** — BC clones ``always_cooperate`` to
   >= 95% action accuracy AND PPO continuation for 100 iters keeps
   per-step per-agent reward >= 0.4 (i.e. does NOT decay to defect).
4. **Best-of-N does not bridge** — random-init top-1% K=200 phase-2
   mean stays at or below IPPO converged value + 0.10 slack. The
   stability gate in ``best_of_n.py`` already encodes this; we
   simply read its verdict.

Failure of gate 3 specifically flips the verdict to
``anti_attractor_in_toy``; failure of any other gate downgrades to
``null_result``. This matches the issue's explicit falsification path:
basin-trap and anti-attractor are mutually exclusive readings of gate 3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


# Gate thresholds. Pinned constants so the classifier is reproducible across
# downstream analyses and tests can exercise the boundary cases.
GATE1_REWARD_MAX: float = 0.10  # IPPO must not exceed this per-step per-agent reward
GATE1_REWARD_MIN: float = -0.05  # nor fall below this (sanity floor)
GATE1_COOP_FRAC_MAX: float = 0.10  # IPPO cooperation fraction ceiling
GATE2_SPECIALIST_REWARD_MIN: float = 0.50  # always_cooperate self-play floor
GATE3_BC_ACTION_ACC_MIN: float = 0.95  # BC action accuracy gate
GATE3_BC_PPO_REWARD_MIN: float = 0.40  # post-PPO reward floor for "basin holds"
GATE4_BRIDGE_SLACK: float = 0.10  # random-best-of-N must stay within slack of IPPO


def gate1_ippo_defects(ippo_seed_summaries: List[Dict]) -> Dict[str, object]:
    """Gate 1: IPPO converges to defection across all seeds."""
    if not ippo_seed_summaries:
        return {"passed": False, "reason": "no IPPO seeds provided"}
    rewards = [s["final_per_agent_reward"] for s in ippo_seed_summaries]
    coops = [s["final_coop_fraction"] for s in ippo_seed_summaries]
    mean_reward = sum(rewards) / len(rewards)
    mean_coop = sum(coops) / len(coops)
    reward_ok = GATE1_REWARD_MIN <= mean_reward <= GATE1_REWARD_MAX
    coop_ok = mean_coop < GATE1_COOP_FRAC_MAX
    return {
        "passed": bool(reward_ok and coop_ok),
        "mean_final_per_agent_reward": mean_reward,
        "mean_coop_fraction": mean_coop,
        "n_seeds": len(rewards),
        "reward_in_range": reward_ok,
        "coop_below_threshold": coop_ok,
        "threshold_reward_max": GATE1_REWARD_MAX,
        "threshold_reward_min": GATE1_REWARD_MIN,
        "threshold_coop_max": GATE1_COOP_FRAC_MAX,
    }


def gate2_specialist_dominates(specialist_per_agent_reward: float) -> Dict[str, object]:
    """Gate 2: ``always_cooperate`` self-play exceeds the cooperative threshold."""
    passed = specialist_per_agent_reward >= GATE2_SPECIALIST_REWARD_MIN
    return {
        "passed": bool(passed),
        "specialist_per_agent_reward": float(specialist_per_agent_reward),
        "threshold": GATE2_SPECIALIST_REWARD_MIN,
    }


def gate3_bc_holds_basin(
    bc_summary: Dict, ppo_continuation_seed_summaries: List[Dict]
) -> Dict[str, object]:
    """Gate 3: BC clones the specialist AND PPO continuation keeps cooperative reward."""
    bc_accuracy = float(bc_summary.get("min_test_accuracy", 0.0))
    accuracy_ok = bc_accuracy >= GATE3_BC_ACTION_ACC_MIN
    if not ppo_continuation_seed_summaries:
        return {
            "passed": False,
            "reason": "no BC-init PPO continuation seeds provided",
            "bc_action_accuracy": bc_accuracy,
            "accuracy_ok": accuracy_ok,
        }
    cont_rewards = [
        s["final_per_agent_reward"] for s in ppo_continuation_seed_summaries
    ]
    mean_cont_reward = sum(cont_rewards) / len(cont_rewards)
    reward_ok = mean_cont_reward >= GATE3_BC_PPO_REWARD_MIN
    return {
        "passed": bool(accuracy_ok and reward_ok),
        "bc_action_accuracy": bc_accuracy,
        "accuracy_ok": accuracy_ok,
        "mean_post_ppo_per_agent_reward": mean_cont_reward,
        "reward_ok": reward_ok,
        "n_seeds": len(cont_rewards),
        "threshold_accuracy": GATE3_BC_ACTION_ACC_MIN,
        "threshold_reward": GATE3_BC_PPO_REWARD_MIN,
    }


def gate4_bridge_fails(
    bestofn_stability_gate: Dict, ippo_mean_reward: float
) -> Dict[str, object]:
    """Gate 4: random-init best-of-N (K=200 confirmed) does NOT bridge to cooperation."""
    phase2_mean = float(bestofn_stability_gate.get("phase2_mean", float("nan")))
    # The bridge "fails" (= basin-trap consistent) when the phase-2 confirmed
    # random-net top-1% reward stays within ``GATE4_BRIDGE_SLACK`` of IPPO.
    bridge_failed = phase2_mean <= ippo_mean_reward + GATE4_BRIDGE_SLACK
    # Also require the stability gate's own check (drift <= slack) — the #271
    # safeguard. If phase-2 reward is much higher than phase-1, even a low
    # absolute value could indicate the K=20 ranking missed something.
    stability_ok = bool(bestofn_stability_gate.get("passed", False))
    return {
        "passed": bool(bridge_failed and stability_ok),
        "phase2_mean": phase2_mean,
        "ippo_reference": float(ippo_mean_reward),
        "slack": GATE4_BRIDGE_SLACK,
        "bridge_below_ippo_plus_slack": bridge_failed,
        "stability_gate_passed": stability_ok,
    }


def classify_verdict(gate1: Dict, gate2: Dict, gate3: Dict, gate4: Dict) -> str:
    """Map the four gate verdicts to one of three named outcomes.

    - All four pass → ``basin_trap_replicated`` (textbook target).
    - Gate 3 fails specifically → ``anti_attractor_in_toy`` (BC decays;
      this falsifies basin-trap and reopens the anti-attractor reading).
    - Any other configuration → ``null_result``.
    """
    g1 = bool(gate1["passed"])
    g2 = bool(gate2["passed"])
    g3 = bool(gate3["passed"])
    g4 = bool(gate4["passed"])
    if g1 and g2 and g3 and g4:
        return "basin_trap_replicated"
    if g1 and g2 and (not g3) and g4:
        return "anti_attractor_in_toy"
    return "null_result"


def compute_verdict(
    ippo_seed_summaries: List[Dict],
    specialist_per_agent_reward: float,
    bc_summary: Dict,
    ppo_continuation_seed_summaries: List[Dict],
    bestofn_stability_gate: Dict,
) -> Dict[str, object]:
    """Top-level entry: assemble all four gates and the named verdict."""
    g1 = gate1_ippo_defects(ippo_seed_summaries)
    g2 = gate2_specialist_dominates(specialist_per_agent_reward)
    g3 = gate3_bc_holds_basin(bc_summary, ppo_continuation_seed_summaries)
    ippo_reference = g1.get("mean_final_per_agent_reward", 0.0)
    g4 = gate4_bridge_fails(bestofn_stability_gate, float(ippo_reference))
    verdict = classify_verdict(g1, g2, g3, g4)
    return {
        "issue": 292,
        "verdict": verdict,
        "gate1_ippo_defects": g1,
        "gate2_specialist_dominates": g2,
        "gate3_bc_holds_basin": g3,
        "gate4_bridge_fails": g4,
    }


def _load_metrics(path: Path) -> List[Dict]:
    with path.open() as f:
        return json.load(f)


def _seed_summary_from_metrics(metrics_path: Path) -> Dict:
    """Extract gate-relevant scalars from a single training cell's metrics.json."""
    metrics = _load_metrics(metrics_path)
    final = metrics[-1]
    return {
        "metrics_path": str(metrics_path),
        "final_per_agent_reward": float(final["mean_step_reward_per_agent"]),
        "final_coop_fraction": float(final["cooperation_fraction"]),
        "iteration": int(final["iteration"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ippo-runs",
        nargs="+",
        required=True,
        help="Paths to IPPO seed cell metrics.json files.",
    )
    p.add_argument(
        "--bc-continuation-runs",
        nargs="+",
        required=True,
        help="Paths to BC-init PPO continuation seed metrics.json files.",
    )
    p.add_argument(
        "--bc-summary",
        type=Path,
        required=True,
        help="Path to bc_init.py's bc_summary.json output.",
    )
    p.add_argument(
        "--bestofn-summary",
        type=Path,
        required=True,
        help="Path to best_of_n.py's summary.json output.",
    )
    p.add_argument(
        "--bestofn-protocol",
        default="independent",
        help="Which init protocol's stability gate to consume (default: independent).",
    )
    p.add_argument(
        "--specialist-per-agent-reward",
        type=float,
        default=0.6,
        help="Analytic always_cooperate self-play per-step per-agent reward.",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    ippo_summaries = [_seed_summary_from_metrics(Path(x)) for x in args.ippo_runs]
    cont_summaries = [
        _seed_summary_from_metrics(Path(x)) for x in args.bc_continuation_runs
    ]
    with args.bc_summary.open() as f:
        bc_summary = json.load(f)
    with args.bestofn_summary.open() as f:
        bestofn = json.load(f)
    stab = bestofn[args.bestofn_protocol]["stability_gate"]

    verdict = compute_verdict(
        ippo_seed_summaries=ippo_summaries,
        specialist_per_agent_reward=args.specialist_per_agent_reward,
        bc_summary=bc_summary,
        ppo_continuation_seed_summaries=cont_summaries,
        bestofn_stability_gate=stab,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(verdict, f, indent=2)

    print(f"\n== Verdict: {verdict['verdict']} ==")
    for k in (
        "gate1_ippo_defects",
        "gate2_specialist_dominates",
        "gate3_bc_holds_basin",
        "gate4_bridge_fails",
    ):
        print(f"  {k}: passed={verdict[k]['passed']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
