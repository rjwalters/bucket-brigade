"""Tests for the reward-scale parity check (issue #437).

Three families of guarantees:

1. **Drift guards**: the parity manifest is keyed by frozen scenario IDs
   and generated from the canonical sources, so its keys/values must stay
   aligned with ``SCENARIO_RANDOM_BASELINES`` and ``SCENARIO_VERSIONS``
   (same pattern as ``tests/test_baselines_constants.py``), and the pinned
   ``SCENARIO_FINGERPRINTS`` must match hashes recomputed from the live
   registry — a scenario-parameter change fails here and forces the
   registry version-bump decision.

2. **Falsification** (the #432-class scenario): a 7x scaling of every
   reward/cost weight must make the CLI exit non-zero with the
   observed/expected ratio in the failure output.

3. **Fingerprint sensitivity**: a single-field scenario parameter change
   must flip the fingerprint verdict even when run against an otherwise
   healthy env.

Rollouts here are deliberately tiny (tens to low hundreds of pure-Python
env episodes, fixed seeds — deterministic and seconds-cheap; safe per the
CLAUDE.md compute guidelines).
"""

from __future__ import annotations

import dataclasses
import json
import re

import pytest

from bucket_brigade.baselines import SCENARIO_RANDOM_BASELINES
from bucket_brigade.baselines import parity
from bucket_brigade.baselines.parity import (
    REFERENCE_CI95,
    SCENARIO_FINGERPRINTS,
    build_manifest,
    check_scenario,
    main,
    scenario_fingerprint,
)
from bucket_brigade.envs.registry import (
    SCENARIO_VERSIONS,
    get_scenario_by_id,
    parse_scenario_id,
)
from bucket_brigade.envs.scenarios_generated import Scenario


def _scale_reward_weights(scenario: Scenario, factor: float) -> Scenario:
    """Scale every reward/penalty/cost weight by ``factor``.

    Fire dynamics and topology are untouched, so trajectories are
    identical and the per-step team reward scales by exactly ``factor`` —
    the cleanest model of the PR #432 "differently-weighted reward
    configuration" incident.
    """
    return dataclasses.replace(
        scenario,
        team_reward_house_survives=scenario.team_reward_house_survives * factor,
        team_penalty_house_burns=scenario.team_penalty_house_burns * factor,
        reward_own_house_survives=[
            v * factor for v in scenario.reward_own_house_survives
        ],
        reward_other_house_survives=[
            v * factor for v in scenario.reward_other_house_survives
        ],
        penalty_own_house_burns=[v * factor for v in scenario.penalty_own_house_burns],
        penalty_other_house_burns=[
            v * factor for v in scenario.penalty_other_house_burns
        ],
        cost_to_work_one_night=scenario.cost_to_work_one_night * factor,
    )


# ---------------------------------------------------------------------------
# Manifest drift guards
# ---------------------------------------------------------------------------


class TestManifestDriftGuards:
    def test_manifest_ids_are_frozen_registry_ids(self) -> None:
        """Every parity entry must be keyed by an ID in SCENARIO_VERSIONS."""
        unknown = sorted(set(REFERENCE_CI95) - set(SCENARIO_VERSIONS))
        assert not unknown, (
            f"Parity manifest references scenario IDs absent from the frozen "
            f"registry: {unknown}. The manifest must be keyed by "
            "SCENARIO_VERSIONS IDs."
        )

    def test_manifest_covers_every_canonical_random_baseline(self) -> None:
        """Base names must be exactly the SCENARIO_RANDOM_BASELINES keys.

        One parity entry per canonical random baseline — no extras, no
        gaps. (``easy-v1`` was added to the registry by #437 to close the
        one gap; ``v2_minimal-v1`` has no canonical random baseline and is
        correctly absent.)
        """
        manifest_bases = {parse_scenario_id(sid)[0] for sid in REFERENCE_CI95}
        assert manifest_bases == set(SCENARIO_RANDOM_BASELINES), (
            "Parity manifest base names diverge from SCENARIO_RANDOM_BASELINES: "
            f"missing={sorted(set(SCENARIO_RANDOM_BASELINES) - manifest_bases)}, "
            f"extra={sorted(manifest_bases - set(SCENARIO_RANDOM_BASELINES))}."
        )

    def test_manifest_values_mirror_scenario_random_baselines(self) -> None:
        """Manifest baselines must equal the single-source-of-truth table."""
        manifest = build_manifest()
        scenarios = manifest["scenarios"]
        assert isinstance(scenarios, dict)
        for scenario_id, entry in scenarios.items():
            base = entry["base_name"]
            assert entry["random_per_step_team"] == SCENARIO_RANDOM_BASELINES[base], (
                f"{scenario_id}: manifest baseline "
                f"{entry['random_per_step_team']!r} disagrees with "
                f"SCENARIO_RANDOM_BASELINES[{base!r}] = "
                f"{SCENARIO_RANDOM_BASELINES[base]!r}."
            )

    def test_reference_ci_contains_canonical_mean(self) -> None:
        """Each committed n=1000 CI must bracket its canonical mean —
        catches transcription errors in the lifted CI endpoints."""
        for scenario_id, (lo, hi) in REFERENCE_CI95.items():
            base, _ = parse_scenario_id(scenario_id)
            mean = SCENARIO_RANDOM_BASELINES[base]
            assert lo < hi, f"{scenario_id}: degenerate CI [{lo}, {hi}]"
            assert lo <= mean <= hi, (
                f"{scenario_id}: canonical mean {mean} outside the committed "
                f"reference CI [{lo}, {hi}] — transcription error in "
                "REFERENCE_CI95?"
            )

    def test_pinned_fingerprints_match_live_registry(self) -> None:
        """Recompute every pinned fingerprint from the live registry.

        A mismatch means a frozen scenario's parameters drifted: per the
        registry version-bump policy that requires a NEW ``-vN`` ID, not a
        silent mutation of the old one.
        """
        assert set(SCENARIO_FINGERPRINTS) == set(REFERENCE_CI95)
        for scenario_id, pinned in SCENARIO_FINGERPRINTS.items():
            live = scenario_fingerprint(get_scenario_by_id(scenario_id))
            assert live == pinned, (
                f"{scenario_id}: live scenario fingerprint {live} != pinned "
                f"{pinned}. Frozen scenario parameters changed — this "
                "requires a new -vN registry entry (see "
                "bucket_brigade/envs/registry.py version-bump policy), not "
                "an in-place edit."
            )

    def test_manifest_is_json_serializable_with_convention(self) -> None:
        manifest = build_manifest()
        payload = json.loads(json.dumps(manifest))
        assert payload["manifest_version"] == parity.MANIFEST_VERSION
        convention = payload["measurement_convention"]
        assert convention["policy"] == "uniform_random"
        assert convention["num_agents"] == 4
        assert convention["n_episodes"] == 1000
        assert "MultiDiscrete" in convention["sampler"]
        assert "env.night" in convention["per_step_normalization"]
        assert len(payload["scenarios"]) == len(SCENARIO_RANDOM_BASELINES)


# ---------------------------------------------------------------------------
# Fingerprint sensitivity
# ---------------------------------------------------------------------------


class TestFingerprint:
    def test_fingerprint_is_deterministic(self) -> None:
        a = scenario_fingerprint(get_scenario_by_id("default-v1"))
        b = scenario_fingerprint(get_scenario_by_id("default-v1"))
        assert a == b
        assert a.startswith("sha256:")

    def test_fingerprint_catches_single_field_change(self) -> None:
        """Acceptance criterion: a single-field parameter change flips the
        fingerprint (here the #432-adjacent knob, a team-reward weight)."""
        base = get_scenario_by_id("default-v1")
        perturbed = dataclasses.replace(
            base,
            team_reward_house_survives=base.team_reward_house_survives + 1.0,
        )
        assert scenario_fingerprint(perturbed) != scenario_fingerprint(base)

    def test_check_scenario_reports_fingerprint_mismatch(self) -> None:
        """A parameter change that barely moves the reward scale must still
        fail the check via the fingerprint verdict."""
        base = get_scenario_by_id("default-v1")
        perturbed = dataclasses.replace(base, min_nights=base.min_nights)
        # Sanity: identical scenario passes the fingerprint side.
        result = check_scenario("default-v1", n_episodes=10, seed=0, scenario=perturbed)
        assert result.fingerprint_ok

        drifted = dataclasses.replace(
            base, cost_to_work_one_night=base.cost_to_work_one_night + 0.01
        )
        result = check_scenario("default-v1", n_episodes=10, seed=0, scenario=drifted)
        assert not result.fingerprint_ok
        assert not result.passed
        assert "FINGERPRINT MISMATCH" in result.summary()


# ---------------------------------------------------------------------------
# Statistical parity check
# ---------------------------------------------------------------------------


class TestParityCheck:
    def test_passes_on_repo_native_env_all_manifest_scenarios(self) -> None:
        """Acceptance criterion: the check passes on the repo-native env for
        every manifest scenario, at the CLI defaults (n=500, seed=42) —
        deterministic, ~20s of pure env stepping for the full sweep.
        (Smaller n is NOT safe here: heavy-tailed scenarios like ``hard``
        can legitimately sit >3 combined SEs out at n=150.)"""
        failures = []
        for scenario_id in sorted(REFERENCE_CI95):
            result = check_scenario(scenario_id, n_episodes=500, seed=42)
            if not result.passed:
                failures.append(result.summary())
        assert not failures, "Repo-native parity check failed:\n" + "\n".join(failures)

    def test_fails_on_7x_reward_scale_with_ratio_in_error(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Acceptance criterion (falsification): inject the PR #432-style 7x
        reward scaling and assert the CLI exits non-zero with the
        observed/expected ratio in the error output."""
        scaled = _scale_reward_weights(get_scenario_by_id("default-v1"), 7.0)
        monkeypatch.setattr(parity, "get_scenario_by_id", lambda sid, **kw: scaled)

        exit_code = main(["--scenario-id", "default-v1", "--episodes", "60"])
        captured = capsys.readouterr()

        assert exit_code != 0
        assert "PARITY FAIL" in captured.out
        assert "observed/expected ratio" in captured.out
        # stderr carries the machine-grep-able failure roll-up with the
        # ratio; a ~7x scale error must be unmissable in it. (The ratio is
        # observed/canonical-expected, so at small n it is 7x the run's own
        # sampling wobble around 1.0 — bound it rather than pin it.)
        match = re.search(r"observed/expected = (-?\d+\.\d+)", captured.err)
        assert match is not None, f"no ratio in stderr: {captured.err!r}"
        assert 6.0 < float(match.group(1)) < 8.0

    def test_check_scenario_scales_with_injected_factor(self) -> None:
        """Reward-weight scaling leaves trajectories untouched (dynamics
        depend only on probabilities/actions), so the observed value scales
        by ~7x against the same-seed unscaled run. Not *exactly* 7x: the
        env's flat ``+0.5`` rest reward is hardcoded in
        ``BucketBrigadeEnv._compute_rewards``, not a scenario weight, so it
        stays at 1x (a sub-1% effect on ``default``)."""
        scaled = _scale_reward_weights(get_scenario_by_id("default-v1"), 7.0)
        result = check_scenario("default-v1", n_episodes=60, seed=42, scenario=scaled)
        unscaled = check_scenario(
            "default-v1",
            n_episodes=60,
            seed=42,
            scenario=get_scenario_by_id("default-v1"),
        )
        assert not result.reward_ok
        assert not result.passed
        assert result.observed == pytest.approx(7.0 * unscaled.observed, rel=0.02)
        # The scaled scenario also trips the fingerprint check.
        assert not result.fingerprint_ok

    def test_unknown_scenario_id_raises_keyerror(self) -> None:
        with pytest.raises(KeyError, match="not in the parity manifest"):
            check_scenario("no_such_scenario-v1", n_episodes=2)

    def test_v2_minimal_is_not_in_manifest(self) -> None:
        """v2_minimal-v1 is a frozen ID without a canonical random baseline;
        it must be rejected rather than checked against a wrong yardstick."""
        with pytest.raises(KeyError):
            check_scenario("v2_minimal-v1", n_episodes=2)


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


class TestCli:
    def test_manifest_flag_emits_valid_json(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert main(["--manifest"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["manifest_version"] == parity.MANIFEST_VERSION
        assert "rest_trap-v1" in payload["scenarios"]
        entry = payload["scenarios"]["rest_trap-v1"]
        assert entry["random_per_step_team"] == SCENARIO_RANDOM_BASELINES["rest_trap"]
        assert entry["scenario_fingerprint"] == SCENARIO_FINGERPRINTS["rest_trap-v1"]

    def test_single_scenario_pass_exit_zero(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        assert (
            main(["--scenario-id", "trivial_cooperation-v1", "--episodes", "30"]) == 0
        )
        assert "PARITY OK" in capsys.readouterr().out

    def test_json_output_shape(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert (
            main(
                [
                    "--scenario-id",
                    "trivial_cooperation-v1",
                    "--episodes",
                    "30",
                    "--json",
                ]
            )
            == 0
        )
        results = json.loads(capsys.readouterr().out)
        assert isinstance(results, list) and len(results) == 1
        record = results[0]
        assert record["scenario_id"] == "trivial_cooperation-v1"
        assert record["passed"] is True
        assert {"expected", "observed", "ratio", "tolerance", "n_episodes"} <= set(
            record
        )

    def test_unknown_id_exits_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert main(["--scenario-id", "bogus-v1", "--episodes", "5"]) == 2
        assert "not in the parity manifest" in capsys.readouterr().err

    def test_requires_target_selection(self) -> None:
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 2
