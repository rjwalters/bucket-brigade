"""Reward-scale parity check for downstream consumers (issue #437).

Motivation: PR #432 found that a downstream harness's headline claim about
this benchmark did not reproduce repo-natively — the quoted magnitudes were
~7x larger than repo-native numbers, "consistent with a differently-weighted
reward configuration". A full study ran to completion on the wrong reward
scale before a repo-side oracle exposed it. This module converts that class
of silent, study-invalidating config mismatch into an immediate loud failure
that costs seconds to run.

It packages the *existing* canonical numbers — no new measurement — as an
executable check:

1. **Reference manifest** (:func:`build_manifest`): machine-readable, keyed
   by frozen scenario ID (``bucket_brigade.envs.registry.SCENARIO_VERSIONS``).
   Each entry carries the canonical uniform-random per-step team reward
   (mirroring :data:`bucket_brigade.baselines.SCENARIO_RANDOM_BASELINES`),
   the measurement convention it was derived under, the committed n=1000
   95% bootstrap CI from the issue #237 derivation logs, and a stable
   **scenario fingerprint** (sha256 over the resolved
   :class:`~bucket_brigade.envs.scenarios_generated.Scenario` fields).

2. **Statistical parity check** (:func:`check_scenario` / the CLI): runs a
   few hundred cheap uniform-random episodes with the documented convention
   against the caller's installed env and compares the measured per-step
   team reward to the manifest within a tolerance derived from the committed
   measurement CI. On mismatch the CLI exits non-zero and prints the
   observed/expected **ratio** — a #432-style 7x scale error is unmissable.

3. **Fingerprint check**: for consumers that can construct the Python
   ``Scenario``, the pinned :data:`SCENARIO_FINGERPRINTS` hashes catch
   scenario *parameter* drift even when the reward scale happens to
   coincide. In-repo, ``tests/test_parity.py`` recomputes these hashes so a
   parameter change forces a conscious version-bump decision (see the
   version-bump policy in :mod:`bucket_brigade.envs.registry`).

Measurement convention (must be reproduced exactly by re-implementations)
--------------------------------------------------------------------------

* Policy: uniform random over ``MultiDiscrete([num_houses, 2, 2])``
  (``[house, mode, signal]``; post-#236 signal-as-first-class-action),
  sampled independently per agent per step.
* Per-step team reward: total episode team reward (``rewards.sum()``
  accumulated over all steps and agents) divided by ``env.night`` at done.
  Episodes run to natural termination (``min_nights`` + no-active-fire
  rule), NOT a fixed night count.
* ``num_agents = 4`` (the registry default for frozen IDs).
* The rollout loop is shared with the calibration path
  (:func:`bucket_brigade.baselines.per_cell._run_random_episode`) so the
  check and the canonical derivation cannot drift apart.

Reference derivation: issue #237 post-#236 re-derivation at commit
``dffe1060``, n=1000 episodes per scenario (200 episodes x 5 seeds 42..46).
Committed logs: ``experiments/p3_specialization/diagnostics/results/
issue237_postmerge/``.

Usage
-----

Run locally — this is a few hundred cheap env episodes (seconds), not
training. Safe per the CLAUDE.md compute guidelines. ::

    # Check one scenario (exits non-zero on mismatch, with the ratio):
    python -m bucket_brigade.baselines.parity --scenario-id rest_trap-v1

    # Check every manifest scenario:
    python -m bucket_brigade.baselines.parity --all

    # Dump the machine-readable reference manifest:
    python -m bucket_brigade.baselines.parity --manifest

    # More episodes -> tighter check:
    python -m bucket_brigade.baselines.parity --scenario-id default-v1 \\
        --episodes 2000

See ``docs/PARITY.md`` for the downstream-consumer workflow.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from bucket_brigade.baselines import SCENARIO_RANDOM_BASELINES
from bucket_brigade.baselines.per_cell import _run_random_episode, _seeds_for
from bucket_brigade.envs.registry import (
    DEFAULT_NUM_AGENTS,
    get_scenario_by_id,
    parse_scenario_id,
)
from bucket_brigade.envs.scenarios_generated import Scenario

__all__ = [
    "MANIFEST_VERSION",
    "DEFAULT_EPISODES",
    "DEFAULT_SEED",
    "DEFAULT_Z",
    "REFERENCE_CI95",
    "SCENARIO_FINGERPRINTS",
    "ParityResult",
    "scenario_fingerprint",
    "build_manifest",
    "measure_random_per_step",
    "check_scenario",
    "main",
]


# Bump when the manifest schema or any reference value changes, so
# downstream results can cite "scenario ID + manifest version" (issue #437
# acceptance criteria / docs/PARITY.md).
MANIFEST_VERSION: int = 1

# CLI defaults. 500 episodes is a few seconds of pure env stepping and
# shrinks the observed-side standard error well below the reference CI for
# every manifest scenario; ``--episodes`` can raise it for a tighter check.
DEFAULT_EPISODES: int = 500
DEFAULT_SEED: int = 42
# Tolerance multiplier: |observed - expected| must be <= z * combined SE
# (see :func:`check_scenario`). z=3 keeps false alarms rare across the whole
# 16-scenario manifest while a 7x scale error sits hundreds of SEs out.
DEFAULT_Z: float = 3.0

# Absolute tolerance floor (reward units per step). Covers the 2-decimal
# rounding of the published reference means (max 0.005) with margin, and
# keeps near-deterministic scenarios (trivial_cooperation's reference CI is
# [399.98, 400.01]) from producing a vacuously tight tolerance.
_TOLERANCE_FLOOR: float = 0.05


# ---------------------------------------------------------------------------
# Committed reference measurement CIs (issue #237 derivation, n=1000)
# ---------------------------------------------------------------------------
#
# Per-step team-reward 95% bootstrap CIs lifted from the committed issue #237
# derivation logs (``experiments/p3_specialization/diagnostics/results/
# issue237_postmerge/<base_name>.log``, "per-step" line, n=1000 episodes,
# commit ``dffe1060``). These are the same runs that produced
# ``SCENARIO_RANDOM_BASELINES``; lifting the CIs into code (issue #437) lets
# the parity tolerance derive from the committed measurement uncertainty
# instead of a made-up epsilon. Keyed by frozen scenario ID.
#
# The two ``asym_*`` entries come from the issue #435 measurement (same
# n=1000 protocol: 200 episodes x 5 seeds 42..46 via
# ``experiments/p3_specialization/diagnostics/random_baseline.py``, host
# studio, commit ``866f43dd``). They are identical by construction: beta
# is inert in bernoulli extinguish mode, so the two phase-diagram cells
# are the same effective environment (see the provenance comments in
# ``bucket_brigade/baselines/__init__.py``).
#
# Drift guard: ``tests/test_parity.py`` asserts each interval contains the
# corresponding ``SCENARIO_RANDOM_BASELINES`` mean and that the key set
# aligns with both ``SCENARIO_RANDOM_BASELINES`` and ``SCENARIO_VERSIONS``.
REFERENCE_CI95: Dict[str, Tuple[float, float]] = {
    "default-v1": (244.86, 257.51),
    "easy-v1": (352.07, 358.06),
    "hard-v1": (118.62, 130.63),
    "trivial_cooperation-v1": (399.98, 400.01),
    "early_containment-v1": (292.88, 301.55),
    "greedy_neighbor-v1": (288.46, 297.13),
    "sparse_heroics-v1": (240.99, 251.08),
    "rest_trap-v1": (298.63, 307.07),
    "chain_reaction-v1": (221.96, 232.70),
    "deceptive_calm-v1": (72.68, 84.49),
    "overcrowding-v1": (116.93, 123.42),
    "mixed_motivation-v1": (218.83, 229.26),
    "minimal_specialization-v1": (-93.31, -82.16),
    "positional_default-v1": (244.36, 257.01),
    "asym_b05_k09_c05-v1": (-83.88, -72.81),
    "asym_b09_k09_c05-v1": (-83.88, -72.81),
}

# Reference sample size behind every entry above (episodes per scenario in
# the issue #237 derivation).
REFERENCE_N_EPISODES: int = 1000


# ---------------------------------------------------------------------------
# Scenario fingerprints
# ---------------------------------------------------------------------------


def scenario_fingerprint(scenario: Scenario) -> str:
    """Stable hash of a resolved :class:`Scenario`'s parameters.

    Canonicalizes ``dataclasses.asdict(scenario)`` as compact JSON with
    sorted keys and returns ``"sha256:<hexdigest>"``. Any single-field
    change to the resolved scenario (reward weights, fire dynamics, costs,
    topology, num_agents, shaping knobs, ...) changes the fingerprint.

    Note: this fingerprints the *constructed Python dataclass*, so it is
    only directly usable by consumers that can build the Python
    ``Scenario``. Pure re-implementations should verify parameter-by-
    parameter against their scenario source and rely on the statistical
    check for end-to-end reward-scale parity.
    """
    payload = dataclasses.asdict(scenario)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Pinned fingerprints of every manifest scenario, computed from
# ``get_scenario_by_id(<id>, num_agents=4)`` at the commit that landed
# issue #437. ``tests/test_parity.py`` recomputes these from the live
# registry: if a scenario's parameters drift, the test fails and forces the
# registry version-bump decision (frozen IDs must not mutate — see
# ``bucket_brigade/envs/registry.py``). Downstream consumers compare their
# constructed scenario's fingerprint against these published values.
SCENARIO_FINGERPRINTS: Dict[str, str] = {
    "asym_b05_k09_c05-v1": (
        "sha256:ce3aa75d21c70ce88f2041d6be0f52dfde76f006ae7f616cd5d1633d37376f89"
    ),
    "asym_b09_k09_c05-v1": (
        "sha256:c7e6d12f80befccf4c77b321d4c63bf36f0dcb8d23e37f926d520084ebc5f5fb"
    ),
    "chain_reaction-v1": (
        "sha256:f1386042d1618ab5b5429e4290cb9bc435e0643f34d7bc2e279d04e3c5443bf9"
    ),
    "deceptive_calm-v1": (
        "sha256:867f9bd71bcaa68cc4a94f4e5a2c6e4f4e9584d019ac2ead42c9047eb54d4c20"
    ),
    "default-v1": (
        "sha256:d5171b48b046c330fa860d6cb87032d6970f92879375f8623901312af5920fbd"
    ),
    "early_containment-v1": (
        "sha256:8c78037a7730ef8d8f0032465f1cd7d00f9283a1b3dd82c6ea7f5bf2fc5275ae"
    ),
    "easy-v1": (
        "sha256:5d63a61a5cc7b0f3b3dbda0bd040f88985d780674c1f74d32a3eb46eb83216a2"
    ),
    "greedy_neighbor-v1": (
        "sha256:ecf39439a702ac1c70f8d0c5fa42e28912290c9fee93227880cb8a342501711f"
    ),
    "hard-v1": (
        "sha256:48b414edefb5697b806b2a92eeb45fc61e62168736443d6be069ccbde685cce0"
    ),
    "minimal_specialization-v1": (
        "sha256:eb0c93b8d45550d8adea36d499ef2a5994e7d2198de14fc908991b99e61cfc4f"
    ),
    "mixed_motivation-v1": (
        "sha256:c191757e40a194339f20ade84e946c2457f7d3be81f6373470c09cc238f1fb56"
    ),
    "overcrowding-v1": (
        "sha256:70c2da7347a9f41b17dd8f99615020e95dcb409290bea3c6c776adc5a3cd9999"
    ),
    "positional_default-v1": (
        "sha256:9786f49b2480031b59c88fb61bc3c81595aea2e43c4b7424d37dd8651deb75e6"
    ),
    "rest_trap-v1": (
        "sha256:098ba0ed1cb67779bfb3abd4aea769b6de49d827e3c693554284f00feeabcf5f"
    ),
    "sparse_heroics-v1": (
        "sha256:e6f4304bdba5dd444f09ee5a2279e56c1d9353f579d2ebe127dc4783ea84fd21"
    ),
    "trivial_cooperation-v1": (
        "sha256:15a5878e07a1d8d6f5ea75518f39818fd66504783405ecf2b8914f67e6f9e2a4"
    ),
}


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest() -> Dict[str, object]:
    """Build the machine-readable parity reference manifest.

    Generated from the existing canonical sources — the baseline means come
    straight from :data:`SCENARIO_RANDOM_BASELINES`, so the manifest cannot
    diverge from the single source of truth by construction. Keyed by
    frozen scenario ID from :data:`SCENARIO_VERSIONS`.
    """
    scenarios: Dict[str, object] = {}
    for scenario_id in sorted(REFERENCE_CI95):
        base_name, _version = parse_scenario_id(scenario_id)
        lo, hi = REFERENCE_CI95[scenario_id]
        scenarios[scenario_id] = {
            "base_name": base_name,
            "random_per_step_team": SCENARIO_RANDOM_BASELINES[base_name],
            "ci95": [lo, hi],
            "n_episodes": REFERENCE_N_EPISODES,
            "scenario_fingerprint": SCENARIO_FINGERPRINTS[scenario_id],
        }
    return {
        "manifest_version": MANIFEST_VERSION,
        "measurement_convention": {
            "policy": "uniform_random",
            "action_layout": "[house, mode, signal]",
            "sampler": (
                "MultiDiscrete([num_houses, 2, 2]) uniform, sampled "
                "independently per agent per step (post-#236 signal-as-"
                "first-class-action)"
            ),
            "per_step_normalization": (
                "total episode team reward (rewards.sum() over all agents "
                "and steps) / env.night at natural episode termination"
            ),
            "num_agents": DEFAULT_NUM_AGENTS,
            "n_episodes": REFERENCE_N_EPISODES,
            "seeds": "42..46 (200 episodes per seed)",
            "source_commit": "dffe1060",
            "provenance": (
                "issue #237 post-#236 re-derivation; committed logs under "
                "experiments/p3_specialization/diagnostics/results/"
                "issue237_postmerge/"
            ),
        },
        "fingerprint_convention": {
            "algorithm": (
                "sha256 over json.dumps(dataclasses.asdict(scenario), "
                "sort_keys=True, separators=(',', ':')) of the Scenario "
                "resolved via get_scenario_by_id(scenario_id, num_agents=4)"
            ),
            "num_agents": DEFAULT_NUM_AGENTS,
        },
        "scenarios": scenarios,
    }


# ---------------------------------------------------------------------------
# Statistical parity check
# ---------------------------------------------------------------------------


def measure_random_per_step(
    scenario: Scenario, n_episodes: int, seed: int
) -> np.ndarray:
    """Per-episode per-step team rewards for uniform-random play.

    Uses the exact rollout loop shared with the calibration path
    (:func:`bucket_brigade.baselines.per_cell._run_random_episode`) and the
    same deterministic per-episode seed derivation, so results are
    reproducible for a given ``(scenario, n_episodes, seed)``.
    """
    if n_episodes < 2:
        raise ValueError("n_episodes must be >= 2 to estimate a standard error")
    seeds = _seeds_for(seed, n_episodes)
    return np.asarray([_run_random_episode((scenario, s)) for s in seeds])


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a single-scenario parity check."""

    scenario_id: str
    expected: float
    observed: float
    ratio: float
    diff: float
    tolerance: float
    z: float
    n_episodes: int
    seed: int
    reward_ok: bool
    fingerprint_expected: str
    fingerprint_observed: str
    fingerprint_ok: bool

    @property
    def passed(self) -> bool:
        return self.reward_ok and self.fingerprint_ok

    def to_dict(self) -> Dict[str, object]:
        d = dataclasses.asdict(self)
        d["passed"] = self.passed
        return d

    def summary(self) -> str:
        """One-line human-readable verdict (the ratio is load-bearing)."""
        if self.reward_ok:
            head = f"PARITY OK   {self.scenario_id}:"
            tail = (
                f"|diff| {abs(self.diff):.2f} <= tolerance {self.tolerance:.2f} "
                f"(z={self.z:g}, n={self.n_episodes})"
            )
        else:
            head = f"PARITY FAIL {self.scenario_id}:"
            tail = (
                f"|diff| {abs(self.diff):.2f} > tolerance {self.tolerance:.2f} "
                f"(z={self.z:g}, n={self.n_episodes}). Your build/binding is "
                "likely on a different reward scale than the canonical "
                "scenario (see docs/PARITY.md and PR #432 for the motivating "
                "7x incident)."
            )
        line = (
            f"{head} observed per-step random team reward {self.observed:.2f} "
            f"vs expected {self.expected:.2f} — observed/expected ratio = "
            f"{self.ratio:.3f}; {tail}"
        )
        if not self.fingerprint_ok:
            line += (
                f"\nFINGERPRINT MISMATCH {self.scenario_id}: constructed "
                f"scenario hash {self.fingerprint_observed} != manifest "
                f"{self.fingerprint_expected} — scenario parameters differ "
                "from the frozen definition even if the reward scale "
                "coincides."
            )
        return line


def check_scenario(
    scenario_id: str,
    n_episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    z: float = DEFAULT_Z,
    scenario: Optional[Scenario] = None,
) -> ParityResult:
    """Run the reward-scale parity check for one frozen scenario ID.

    Tolerance policy: the reference side contributes a standard error
    recovered from the committed n=1000 95% bootstrap CI
    (``SE_ref = half_width / 1.96``); the observed side contributes its own
    sample standard error (``sample std / sqrt(n)``). The check passes when
    ``|observed - expected| <= max(z * sqrt(SE_ref^2 + SE_obs^2), 0.05)``.
    Raising ``n_episodes`` shrinks the observed-side term (issue #437 risk
    mitigation for false alarms on legitimate RNG variation).

    Args:
        scenario_id: Frozen ID present in the manifest, e.g. ``rest_trap-v1``.
        n_episodes: Uniform-random episodes to roll out (seconds-cheap).
        seed: Base seed for the deterministic per-episode seed derivation.
        z: Tolerance multiplier on the combined standard error.
        scenario: Optional pre-constructed scenario override. Used by tests
            to inject a perturbed scenario; normal callers omit it so the
            check exercises the registry construction path end-to-end.

    Returns:
        :class:`ParityResult` with both the statistical verdict and the
        fingerprint verdict.

    Raises:
        KeyError: If ``scenario_id`` is not in the parity manifest.
    """
    if scenario_id not in REFERENCE_CI95:
        available = ", ".join(sorted(REFERENCE_CI95))
        raise KeyError(
            f"Scenario ID {scenario_id!r} is not in the parity manifest. "
            f"Available IDs: {available}"
        )
    base_name, _version = parse_scenario_id(scenario_id)
    expected = float(SCENARIO_RANDOM_BASELINES[base_name])
    if scenario is None:
        scenario = get_scenario_by_id(scenario_id, num_agents=DEFAULT_NUM_AGENTS)

    fingerprint_observed = scenario_fingerprint(scenario)
    fingerprint_expected = SCENARIO_FINGERPRINTS[scenario_id]
    fingerprint_ok = fingerprint_observed == fingerprint_expected

    values = measure_random_per_step(scenario, n_episodes=n_episodes, seed=seed)
    observed = float(values.mean())

    lo, hi = REFERENCE_CI95[scenario_id]
    se_ref = (hi - lo) / 2.0 / 1.96
    se_obs = float(values.std(ddof=1)) / math.sqrt(n_episodes)
    tolerance = max(z * math.sqrt(se_ref**2 + se_obs**2), _TOLERANCE_FLOOR)

    diff = observed - expected
    # All manifest baselines are far from zero (|expected| >= 78), so the
    # ratio is always well-defined; guard anyway for future entries.
    ratio = observed / expected if expected != 0.0 else float("inf")

    return ParityResult(
        scenario_id=scenario_id,
        expected=expected,
        observed=observed,
        ratio=ratio,
        diff=diff,
        tolerance=tolerance,
        z=z,
        n_episodes=n_episodes,
        seed=seed,
        reward_ok=abs(diff) <= tolerance,
        fingerprint_expected=fingerprint_expected,
        fingerprint_observed=fingerprint_observed,
        fingerprint_ok=fingerprint_ok,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bucket_brigade.baselines.parity",
        description=(
            "Reward-scale parity check: verify that this build/binding of "
            "Bucket Brigade is on the canonical reward scale by comparing "
            "a cheap uniform-random rollout against the committed reference "
            "manifest. Exits non-zero (with the observed/expected ratio) on "
            "mismatch. Seconds-cheap; safe to run locally."
        ),
    )
    parser.add_argument(
        "--scenario-id",
        action="append",
        default=None,
        metavar="ID",
        help=(
            "Frozen scenario ID to check (repeatable), e.g. rest_trap-v1. "
            "See --manifest for the full list."
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Check every scenario in the manifest.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"Uniform-random episodes per scenario (default {DEFAULT_EPISODES}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed for episode seed derivation (default {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=DEFAULT_Z,
        help=f"Tolerance multiplier on the combined SE (default {DEFAULT_Z}).",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help="Print the machine-readable reference manifest as JSON and exit.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit per-scenario results as JSON instead of text lines.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point. Returns 0 iff every requested check passed."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.manifest:
        print(json.dumps(build_manifest(), indent=2, sort_keys=True))
        return 0

    if args.all:
        scenario_ids = sorted(REFERENCE_CI95)
    elif args.scenario_id:
        scenario_ids = list(args.scenario_id)
    else:
        parser.error("provide --scenario-id ID (repeatable), --all, or --manifest")

    results: List[ParityResult] = []
    for scenario_id in scenario_ids:
        try:
            result = check_scenario(
                scenario_id,
                n_episodes=args.episodes,
                seed=args.seed,
                z=args.z,
            )
        except KeyError as exc:
            print(f"ERROR: {exc.args[0]}", file=sys.stderr)
            return 2
        results.append(result)
        if not args.json:
            print(result.summary())

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))

    failed = [r for r in results if not r.passed]
    if failed:
        ratios = ", ".join(
            f"{r.scenario_id}: observed/expected = {r.ratio:.3f}" for r in failed
        )
        print(
            f"PARITY CHECK FAILED for {len(failed)}/{len(results)} "
            f"scenario(s) — {ratios}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
