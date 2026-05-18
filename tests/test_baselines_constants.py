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
  reward on the same scenario. Re-derived post-#236 under issue #238 /
  PR #243; specialist policies signal honestly so the value is unchanged
  from pre-#236 at the precision of the n=50 sampler.

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

import re
from pathlib import Path

from bucket_brigade.baselines import (
    MINSPEC_RANDOM,
    MINSPEC_SPECIALIST,
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
    """Canonical specialist per-step team reward (n=50, seed 42)."""
    assert MINSPEC_SPECIALIST == -22.07, (
        f"MINSPEC_SPECIALIST = {MINSPEC_SPECIALIST!r}; expected -22.07 "
        "(issue #238 / PR #243 post-#236 re-derivation, n=50)."
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
