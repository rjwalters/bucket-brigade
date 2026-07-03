"""Anti-regression guard for the canonical minimal_specialization baselines
(issue #293) and the scenario-wide random-baselines mirror (issue #323).

This test pins the per-step team-reward references published by
:mod:`bucket_brigade.baselines` to the canonical post-#246 derivation
documented in that module's docstring:

* ``MINSPEC_RANDOM`` — uniform-random per-step team reward on
  ``minimal_specialization`` with the post-#236 3-dim sampler
  ``MultiDiscrete([10, 2, 2])``, n=1000 episodes (200 episodes × 5 seeds
  42..46), commit ``dffe1060``. Issue #237 / PR #244.

* ``MINSPEC_SPECIALIST`` — hand-coded ``SpecialistPolicy`` per-step team
  reward on the same scenario. Re-derived under issue #416 from the
  n=10k per-cell calibration landed in issue #413 / PR #415. The
  canonical cell (β=0.25, κ=0.50, c=0.50) yields mean ``-28.376``,
  CI95 [-29.748, -26.987] at n=10000 episodes (seed=0). Historical
  n=50 value was ``-22.07`` (PR #243); the shift is a precision
  correction, not a sampler-bug correction.

* ``SCENARIO_RANDOM_BASELINES`` — value-only mirror of the ``random``
  column of ``random_baseline.SCENARIO_CITED_VALUES`` (issue #323). The
  drift guard asserts every key/value pair matches the diagnostics table
  so non-torch consumers (e.g. ``compute_nash_trained.py``) get the same
  numbers as the authoritative measurement record.

Before issue #293 the codebase had three coexisting MINSPEC_RANDOM values
spread across analyzers: ``-96.07`` (pre-#246 2-dim sampler bug),
``-92.92`` (intermediate n=50), and the canonical ``-87.72``. This test
locks in the canonical value so the unification cannot silently regress.

We deliberately compare against the table in
``random_baseline.SCENARIO_CITED_VALUES`` only via plain grep, not import,
because ``random_baseline.py`` transitively pulls in torch via the
training package and would break the no-RL CI install — matching the
pattern in :mod:`tests.test_issue199_baselines_sampler`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from bucket_brigade.baselines import (
    MINSPEC_RANDOM,
    MINSPEC_SPECIALIST,
    SCENARIO_GAP_REFERENCES,
    SCENARIO_RANDOM_BASELINES,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RANDOM_BASELINE = (
    _REPO_ROOT
    / "experiments"
    / "p3_specialization"
    / "diagnostics"
    / "random_baseline.py"
)


def test_minspec_random_is_canonical() -> None:
    """Canonical uniform-random per-step team reward (n=1000 × 5 seeds)."""
    assert MINSPEC_RANDOM == -87.72, (
        f"MINSPEC_RANDOM = {MINSPEC_RANDOM!r}; expected -87.72 (issue #237 / "
        "PR #244 post-#246 3-dim sampler, n=1000 × 5 seeds). See "
        "``bucket_brigade.baselines`` module docstring for derivation."
    )


def test_minspec_specialist_is_canonical() -> None:
    """Canonical specialist per-step team reward (n=10000, seed 0)."""
    assert MINSPEC_SPECIALIST == -28.38, (
        f"MINSPEC_SPECIALIST = {MINSPEC_SPECIALIST!r}; expected -28.38 "
        "(issue #416 re-derivation from the issue #413 / PR #415 per-cell "
        "calibration, n=10000 at the canonical cell β=0.25 κ=0.50 c=0.50)."
    )


def test_minspec_random_matches_measurement_table() -> None:
    """The constant must agree with the entry in ``random_baseline.py``.

    ``SCENARIO_CITED_VALUES['minimal_specialization']['random']`` is the
    authoritative per-scenario measurement table — it tracks all 14
    scenarios with provenance notes. Issue #293 promoted just the
    minimal_specialization value to ``bucket_brigade.baselines`` for easy
    import, so the two must stay in sync.
    """
    src = _RANDOM_BASELINE.read_text()
    # Match the entry pattern:
    #     "minimal_specialization": {
    #         "random": -87.72,
    pattern = re.compile(
        r'"minimal_specialization"\s*:\s*\{[^}]*?"random"\s*:\s*(-?\d+\.\d+)',
        re.DOTALL,
    )
    match = pattern.search(src)
    assert match is not None, (
        "Could not locate ``SCENARIO_CITED_VALUES['minimal_specialization']"
        "['random']`` in random_baseline.py — has the table layout changed?"
    )
    table_value = float(match.group(1))
    assert table_value == MINSPEC_RANDOM, (
        f"random_baseline.SCENARIO_CITED_VALUES['minimal_specialization']"
        f"['random'] = {table_value!r} disagrees with "
        f"bucket_brigade.baselines.MINSPEC_RANDOM = {MINSPEC_RANDOM!r}. "
        "Issue #293 requires these to stay aligned."
    )


def _parse_scenario_cited_values_random_column() -> dict[str, float]:
    """Extract ``scenario -> random`` pairs from ``random_baseline.py`` via grep.

    We avoid importing ``random_baseline`` because it transitively pulls in
    torch (see module docstring). This mirrors the approach in
    ``test_minspec_random_matches_measurement_table``.
    """
    src = _RANDOM_BASELINE.read_text()
    # Match each `"scenario": { ... "random": <float>, ...` entry.
    pattern = re.compile(
        r'"(?P<name>[a-z_][a-z0-9_]*)"\s*:\s*\{[^}]*?"random"\s*:\s*(?P<value>-?\d+\.\d+)',
        re.DOTALL,
    )
    return {m.group("name"): float(m.group("value")) for m in pattern.finditer(src)}


def test_scenario_random_baselines_matches_measurement_table() -> None:
    """``SCENARIO_RANDOM_BASELINES`` must equal the diagnostics ``random`` column.

    Issue #323: ``bucket_brigade.baselines.SCENARIO_RANDOM_BASELINES`` is a
    value-only mirror of
    ``random_baseline.SCENARIO_CITED_VALUES[<scenario>]["random"]``. The two
    must agree exactly on every scenario the mirror exposes.
    """
    table = _parse_scenario_cited_values_random_column()
    assert table, (
        "Could not parse any scenario entries from random_baseline.py — "
        "has the SCENARIO_CITED_VALUES layout changed?"
    )
    missing = sorted(set(SCENARIO_RANDOM_BASELINES) - set(table))
    assert not missing, (
        f"SCENARIO_RANDOM_BASELINES references scenarios absent from "
        f"random_baseline.SCENARIO_CITED_VALUES: {missing}"
    )
    mismatches = {
        name: (SCENARIO_RANDOM_BASELINES[name], table[name])
        for name in SCENARIO_RANDOM_BASELINES
        if SCENARIO_RANDOM_BASELINES[name] != table[name]
    }
    assert not mismatches, (
        "SCENARIO_RANDOM_BASELINES disagrees with "
        "random_baseline.SCENARIO_CITED_VALUES on these scenarios "
        f"(mirror, source): {mismatches}. Issue #323 requires these to stay "
        "aligned — update both tables together when re-deriving baselines."
    )


# ---------------------------------------------------------------------------
# SCENARIO_GAP_REFERENCES drift guards (issue #434)
# ---------------------------------------------------------------------------


def test_gap_references_random_side_matches_scenario_random_baselines() -> None:
    """The ``random`` side of every gap-reference pair must match the
    canonical ``SCENARIO_RANDOM_BASELINES`` entry — the two tables must not
    silently diverge (same yardstick, two consumers)."""
    for scenario, entry in SCENARIO_GAP_REFERENCES.items():
        assert scenario in SCENARIO_RANDOM_BASELINES, (
            f"SCENARIO_GAP_REFERENCES[{scenario!r}] has no matching "
            "SCENARIO_RANDOM_BASELINES entry — gap references must be built "
            "on the canonical random baselines."
        )
        assert entry["random"] == SCENARIO_RANDOM_BASELINES[scenario], (
            f"SCENARIO_GAP_REFERENCES[{scenario!r}]['random'] = "
            f"{entry['random']!r} disagrees with "
            f"SCENARIO_RANDOM_BASELINES[{scenario!r}] = "
            f"{SCENARIO_RANDOM_BASELINES[scenario]!r}."
        )


def test_gap_references_minspec_entry_matches_minspec_constants() -> None:
    """The minimal_specialization pair must be exactly (MINSPEC_RANDOM,
    MINSPEC_SPECIALIST) so recalibrated Tier-1 summaries stay bit-for-bit
    compatible with the historical MINSPEC formula (#434)."""
    entry = SCENARIO_GAP_REFERENCES["minimal_specialization"]
    assert entry["random"] == MINSPEC_RANDOM
    assert entry["reference"] == MINSPEC_SPECIALIST
    assert entry["reference_kind"] == "specialist_homogeneous"


def test_gap_references_rest_trap_is_degenerate() -> None:
    """rest_trap must have NO upper reference (NE-below-random social trap):
    the fraction ladder is not applicable there until a measured reference
    that beats random exists (#434)."""
    entry = SCENARIO_GAP_REFERENCES["rest_trap"]
    assert entry["reference"] is None
    assert entry["degenerate_reason"] == "social_trap_ne_below_random"
    # The provenance must record the per-episode vs per-step unit caveat.
    assert "PER EPISODE" in str(entry["provenance"])


def test_gap_references_rest_trap_trap_anchors_match_committed_artifacts() -> None:
    """The #436 trap-verdict anchors must stay in sync with their committed
    measurement artifacts:

    * ``ne_per_step_bound`` == frozen NE per-episode payoff / min_nights
      (rest_trap-v1.json, min_nights = 12).
    * ``scripted_best`` == the final-stage winner recorded by the scripted
      battery run (scripted_battery/rest_trap.json), and it must actually
      beat random (otherwise it must not be recorded as an anchor).
    """
    entry = SCENARIO_GAP_REFERENCES["rest_trap"]

    ne_path = (
        _REPO_ROOT
        / "bucket_brigade"
        / "baselines"
        / "release"
        / "local"
        / "nash"
        / "rest_trap-v1.json"
    )
    ne = json.loads(ne_path.read_text())
    from bucket_brigade.envs.scenarios_generated import get_scenario_by_name

    min_nights = get_scenario_by_name("rest_trap", num_agents=4).min_nights
    assert min_nights == 12
    assert entry["ne_per_step_bound"] == ne["team_payoff"] / min_nights

    battery_path = (
        _REPO_ROOT
        / "experiments"
        / "p3_specialization"
        / "scripted_battery"
        / "rest_trap.json"
    )
    measured = json.loads(battery_path.read_text())["scripted_best"]
    sb = entry["scripted_best"]
    assert sb["value"] == measured["value"]
    assert sb["ci95_lo"] == measured["ci95_lo"]
    assert sb["ci95_hi"] == measured["ci95_hi"]
    assert sb["n_episodes"] == measured["n_episodes"]
    assert sb["kind"] == f"scripted_battery:{measured['name']}"
    assert measured["beats_random"] is True
    assert sb["value"] > entry["random"]
    # Anchor ordering sanity: NE bound < random < scripted_best.
    assert entry["ne_per_step_bound"] < entry["random"] < sb["value"]


def test_gap_references_valid_pairs_have_positive_denominator() -> None:
    """Any entry with a non-null reference must satisfy reference > random —
    otherwise it belongs in the degenerate branch, not the ladder."""
    for scenario, entry in SCENARIO_GAP_REFERENCES.items():
        reference = entry.get("reference")
        if reference is None:
            assert "degenerate_reason" in entry, (
                f"{scenario!r}: degenerate entries must document why "
                "(degenerate_reason)."
            )
            continue
        assert float(reference) > float(entry["random"]), (  # type: ignore[arg-type]
            f"{scenario!r}: reference {reference!r} must exceed random "
            f"{entry['random']!r} for the gap fraction to be meaningful."
        )
