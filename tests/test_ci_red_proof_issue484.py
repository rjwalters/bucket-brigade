"""Deliberately-failing test proving the test-python job can go RED.

Acceptance criterion for issue #484: a failing test must fail the CI job.
This commit is reverted immediately after the red run is captured; both
commits stay in history as the audit trail.
"""


def test_ci_goes_red_on_failure_issue484():
    assert False, "deliberate failure: issue #484 red-run proof (reverted next commit)"
