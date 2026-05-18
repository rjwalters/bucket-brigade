"""Anti-regression guard for the canonical minimal_specialization baselines
(issue #293).

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

from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

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
