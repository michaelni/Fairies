"""Tests for handling of CANCELLED commit-status state.

A cancelled CI job is usually an infra hiccup or a superseded run that
just needs a click on the UI's "Rerun" button. Forgejo's HTTP API does
NOT expose a rerun endpoint (verified against the swagger spec; first
observed on Forgejo 15.0.0+gitea-1.22.0): the only ``POST`` routes
under ``/repos/{owner}/{repo}/actions/`` are ``runners`` registration,
``workflows/{file}/dispatches`` (manual ``workflow_dispatch`` on a
branch -- not a "rerun this run") and secrets/variables management.
If a future Forgejo release adds a rerun endpoint, the bot can be
taught to drive the retry directly; until then:

1.  preserve ``CANCELLED`` as a distinct logical state so it is NOT
    folded into ``FAILURE`` (which would have included it in the LLM
    CI-triage nag payload), and
2.  collect the affected ``(PR, context)`` pairs so the end-of-run
    summary can list them for an admin operator to walk through and
    click the Rerun button in the UI.

Detection has two paths because forges disagree on how cancellations
surface in the legacy commit-status API:

- **State-based (GitLab and any future Forgejo version):** the row's
  ``state`` field is literally ``cancelled`` (or ``canceled``,
  GitLab/GitHub spelling). ``normalize_status_state`` collapses both
  spellings to the canonical ``CANCELLED``.

- **Description-based (Forgejo Actions):** the row's ``state`` field
  is ``failure`` and the cancellation hint lives in the description,
  e.g. ``"Has been cancelled"``. ``row_effective_state`` recovers
  this signal by reclassifying ``FAILURE``/``ERROR`` rows whose
  description matches a cancellation marker.

These tests pin the behavior at the smallest unit possible.
``row_effective_state`` is the single funnel; downstream helpers
(``effective_commit_statuses``, ``build_ci_failure_details``,
``extract_contexts_with_state``) all consult it.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402
import forge_gcli  # noqa: E402


def _row(
    *,
    context: str,
    state: str,
    description: str = "",
    when: str = "2026-04-26T20:00:00Z",
) -> dict:
    return {
        "context": context,
        "state": state,
        "description": description,
        "target_url": f"/path/{context}",
        "created_at": when,
        "updated_at": when,
    }


class NormalizeStatusStateTests(unittest.TestCase):
    def test_cancelled_double_l_is_preserved(self) -> None:
        self.assertEqual(
            fairy.normalize_status_state("cancelled"), "CANCELLED"
        )

    def test_canceled_single_l_is_normalized_to_double_l(self) -> None:
        # GitHub-style and GitLab-style spelling. We collapse to the
        # same logical state so downstream code only has to test for
        # one value.
        self.assertEqual(
            fairy.normalize_status_state("canceled"), "CANCELLED"
        )

    def test_failure_is_unchanged(self) -> None:
        self.assertEqual(fairy.normalize_status_state("failure"), "FAILURE")

    def test_timed_out_still_collapses_to_failure(self) -> None:
        # We deliberately keep TIMED_OUT in FAILURE: a timeout is far more
        # likely to be a real test problem than a transient infra issue,
        # so we still want the LLM to nag about it.
        self.assertEqual(
            fairy.normalize_status_state("timed_out"), "FAILURE"
        )


class RowEffectiveStateTests(unittest.TestCase):
    def test_state_field_pure_passthrough_for_success(self) -> None:
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="ok", state="success", description="anything goes",
            )),
            "SUCCESS",
        )

    def test_gitlab_style_canceled_state_is_cancelled(self) -> None:
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="ok", state="canceled", description="",
            )),
            "CANCELLED",
        )

    def test_forgejo_actions_failure_with_cancelled_description(self) -> None:
        # Real-world Forgejo Actions shape captured from
        # ``GET /repos/.../commits/<sha>/statuses``: cancellations
        # surface as ``status="failure"`` with this exact
        # description. The new code must reclassify them.
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="/ lint (pull_request)",
                state="failure",
                description="Has been cancelled",
            )),
            "CANCELLED",
        )

    def test_real_failure_with_unrelated_description_stays_failure(self) -> None:
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="/ unit_tests",
                state="failure",
                description="Tests failed in 2m13s",
            )),
            "FAILURE",
        )

    def test_cancellation_word_does_not_match_word_with_extra_letters(self) -> None:
        # Word-boundary anchor: a description such as "Cancellation
        # tests passing" must not be reclassified as cancelled.
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="/ unit_tests",
                state="failure",
                description="Cancellation tests passing",
            )),
            "FAILURE",
        )

    def test_status_field_alias_works_for_forgejo_shape(self) -> None:
        # Forgejo's actual API uses the field name ``status`` rather
        # than ``state``. The projection settles that spelling, so a
        # row that arrives either way reaches the same verdict.
        row = forge_gcli._project_status_row({
            "context": "/ run_fate",
            "status": "failure",
            "description": "Has been cancelled",
        })
        self.assertEqual(
            fairy.row_effective_state(row), "CANCELLED",
        )

    def test_pending_with_blocked_by_required_conditions_is_blocked(self) -> None:
        # Real-world Forgejo Actions shape captured from
        # ``GET /repos/.../commits/<sha>/statuses``: jobs gated on a
        # required-environment review or a manual approval surface
        # as ``status="pending"`` with this exact description and
        # never run on their own.
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="/ lint (pull_request)",
                state="pending",
                description="Blocked by required conditions",
            )),
            "BLOCKED",
        )

    def test_pending_with_unrelated_description_stays_pending(self) -> None:
        for description in ("Has started running", "Waiting to run", ""):
            with self.subTest(description=description):
                self.assertEqual(
                    fairy.row_effective_state(_row(
                        context="/ lint",
                        state="pending",
                        description=description,
                    )),
                    "PENDING",
                )

    def test_failure_with_blocked_description_stays_failure(self) -> None:
        # The BLOCKED reclassification only fires from PENDING; a
        # failure row whose description happens to mention "blocked"
        # is still a real failure.
        self.assertEqual(
            fairy.row_effective_state(_row(
                context="/ unit_tests",
                state="failure",
                description="Blocked by required conditions",
            )),
            "FAILURE",
        )


class BuildCiFailureDetailsExcludesCancelledTests(unittest.TestCase):
    def test_state_based_cancelled_is_excluded(self) -> None:
        # GitLab-style state="cancelled".
        statuses = [
            _row(context="cancelled-job", state="cancelled"),
            _row(context="real-failure", state="failure"),
        ]
        out = fairy.build_ci_failure_details(statuses)
        self.assertEqual([d["context"] for d in out], ["real-failure"])

    def test_description_based_cancelled_is_excluded(self) -> None:
        # Forgejo-Actions-style state="failure" + description match.
        statuses = [
            _row(
                context="cancelled-job",
                state="failure",
                description="Has been cancelled",
            ),
            _row(context="real-failure", state="failure",
                 description="Tests failed"),
        ]
        out = fairy.build_ci_failure_details(statuses)
        self.assertEqual([d["context"] for d in out], ["real-failure"])

    def test_only_cancelled_yields_empty_failure_details_state_path(self) -> None:
        statuses = [_row(context="cancelled-job", state="cancelled")]
        self.assertEqual(fairy.build_ci_failure_details(statuses), [])

    def test_only_cancelled_yields_empty_failure_details_description_path(self) -> None:
        statuses = [
            _row(
                context="cancelled-job",
                state="failure",
                description="Has been cancelled",
            )
        ]
        self.assertEqual(fairy.build_ci_failure_details(statuses), [])


class ExtractCancelledContextsTests(unittest.TestCase):
    def test_state_based_only_cancelled_returned(self) -> None:
        statuses = [
            _row(context="real-failure", state="failure"),
            _row(context="cancelled-job", state="cancelled"),
            _row(context="green-job", state="success"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "CANCELLED"),
            ("cancelled-job",),
        )

    def test_description_based_cancelled_is_picked_up(self) -> None:
        statuses = [
            _row(context="real-failure", state="failure",
                 description="Tests failed"),
            _row(
                context="cancelled-job",
                state="failure",
                description="Has been cancelled",
            ),
            _row(context="green-job", state="success"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "CANCELLED"),
            ("cancelled-job",),
        )

    def test_returns_tuple_sorted_by_context_name(self) -> None:
        statuses = [
            _row(context="zeta", state="cancelled"),
            _row(context="alpha", state="cancelled"),
            _row(context="middle", state="cancelled"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "CANCELLED"),
            ("alpha", "middle", "zeta"),
        )

    def test_only_latest_state_per_context_counts(self) -> None:
        # Older cancelled run, then a fresh success: should NOT appear
        # in the cancelled list.
        statuses = [
            _row(context="ctx", state="cancelled", when="2026-04-26T10:00:00Z"),
            _row(context="ctx", state="success", when="2026-04-26T20:00:00Z"),
        ]
        self.assertEqual(fairy.extract_contexts_with_state(statuses, "CANCELLED"), ())

    def test_only_latest_state_per_context_counts_other_direction(self) -> None:
        # Older success, then a fresh cancellation: SHOULD appear.
        statuses = [
            _row(context="ctx", state="success", when="2026-04-26T10:00:00Z"),
            _row(context="ctx", state="cancelled", when="2026-04-26T20:00:00Z"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "CANCELLED"), ("ctx",)
        )

    def test_empty_status_list_returns_empty_tuple(self) -> None:
        self.assertEqual(fairy.extract_contexts_with_state([], "CANCELLED"), ())

    def test_returns_a_real_tuple_not_a_list(self) -> None:
        # ``Decision`` is a frozen dataclass, so the field's default has
        # to be an immutable type. Pin that the helper actually returns
        # a tuple -- if it ever drifts to a list, attaching the result
        # to a frozen Decision would still work today (the dataclass
        # type hint is informational) but a future ``frozen=True``
        # equality check could surprise us.
        self.assertIsInstance(
            fairy.extract_contexts_with_state(
                [_row(context="ctx", state="cancelled")], "CANCELLED",
            ),
            tuple,
        )


class ExtractBlockedContextsTests(unittest.TestCase):
    """Mirror of cancelled-extraction tests for BLOCKED state."""

    def test_pending_with_blocked_description_is_picked_up(self) -> None:
        statuses = [
            _row(context="real-failure", state="failure",
                 description="Tests failed"),
            _row(
                context="blocked-job",
                state="pending",
                description="Blocked by required conditions",
            ),
            _row(context="green-job", state="success"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "BLOCKED"),
            ("blocked-job",),
        )

    def test_returns_tuple_sorted_by_context_name(self) -> None:
        statuses = [
            _row(context="zeta", state="pending",
                 description="Blocked by required conditions"),
            _row(context="alpha", state="pending",
                 description="Blocked by required conditions"),
            _row(context="middle", state="pending",
                 description="Blocked by required conditions"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "BLOCKED"),
            ("alpha", "middle", "zeta"),
        )

    def test_only_latest_state_per_context_counts(self) -> None:
        # Older blocked row, then a fresh success: should NOT appear.
        statuses = [
            _row(context="ctx", state="pending",
                 description="Blocked by required conditions",
                 when="2026-04-26T10:00:00Z"),
            _row(context="ctx", state="success",
                 when="2026-04-26T20:00:00Z"),
        ]
        self.assertEqual(fairy.extract_contexts_with_state(statuses, "BLOCKED"), ())

    def test_only_latest_state_per_context_counts_other_direction(self) -> None:
        # Older success, then a fresh blocked-by-conditions row: SHOULD appear.
        statuses = [
            _row(context="ctx", state="success", when="2026-04-26T10:00:00Z"),
            _row(context="ctx", state="pending",
                 description="Blocked by required conditions",
                 when="2026-04-26T20:00:00Z"),
        ]
        self.assertEqual(
            fairy.extract_contexts_with_state(statuses, "BLOCKED"), ("ctx",)
        )

    def test_empty_status_list_returns_empty_tuple(self) -> None:
        self.assertEqual(fairy.extract_contexts_with_state([], "BLOCKED"), ())

    def test_returns_a_real_tuple_not_a_list(self) -> None:
        self.assertIsInstance(
            fairy.extract_contexts_with_state(
                [_row(context="ctx", state="pending",
                      description="Blocked by required conditions")],
                "BLOCKED",
            ),
            tuple,
        )

    def test_pending_without_blocked_description_is_excluded(self) -> None:
        # PENDING without the specific description must NOT count as
        # blocked; that would mis-flag every CI run that's just
        # waiting for a runner.
        statuses = [
            _row(context="just-pending", state="pending",
                 description="Has started running"),
            _row(context="waiting", state="pending",
                 description="Waiting to run"),
        ]
        self.assertEqual(fairy.extract_contexts_with_state(statuses, "BLOCKED"), ())


class CancelledStillBlocksAutoApproveTests(unittest.TestCase):
    def test_cancelled_state_is_not_success(self) -> None:
        # Sanity: ``failing_contexts`` in ``prepare_pr`` is built as
        # ``state != "SUCCESS"``, so as long as ``CANCELLED`` is not
        # ``SUCCESS`` the PR cannot be auto-approved while a cancelled
        # job is the latest reported state. This pins that property
        # explicitly so a future refactor cannot quietly let a PR with
        # only-cancelled CI fall through to auto-approve.
        self.assertNotEqual(
            fairy.normalize_status_state("cancelled"), "SUCCESS"
        )


class ForgejoActionsRealStatusPayloadTests(unittest.TestCase):
    """End-to-end regression test against a real Forgejo Actions payload.

    The status rows below are an anonymized but otherwise verbatim
    copy of the response from
    ``GET /repos/<owner>/<repo>/commits/<head>/statuses`` for a PR
    whose CI was cancelled. Before this fix the bot saw 4 ``failure``
    rows and fed them to the LLM as failures (which then produced a
    "heads-up: cancelled CI jobs" comment). The end-of-run summary
    listed nothing because no row was ``state=cancelled``.

    After the fix:
    - ``effective_commit_statuses`` reports 4 contexts as
      ``CANCELLED`` and 1 (``/ pr_labeler``) as ``SUCCESS``.
    - ``build_ci_failure_details`` returns ``[]`` (cancelled rows
      are not nag-worthy and the success row is fine).
    - ``extract_contexts_with_state(.., "CANCELLED")`` returns the
      4 cancelled context names.
    """

    WIRE = [
        # Latest per context: 4 failures, all "Has been cancelled"
        {"id": 18, "status": "failure", "description": "Has been cancelled",
         "target_url": "/o/r/actions/runs/2705/jobs/0",
         "context": "/ lint (pull_request)",
         "created_at": "2025-08-18T09:21:17Z",
         "updated_at": "2025-08-18T09:21:17Z"},
        {"id": 19, "status": "failure", "description": "Has been cancelled",
         "target_url": "/o/r/actions/runs/2706/jobs/0",
         "context": "/ run_fate (linux-aarch64) (pull_request)",
         "created_at": "2025-08-18T09:21:17Z",
         "updated_at": "2025-08-18T09:21:17Z"},
        {"id": 20, "status": "failure", "description": "Has been cancelled",
         "target_url": "/o/r/actions/runs/2706/jobs/1",
         "context": "/ run_fate (linux-amd64) (pull_request)",
         "created_at": "2025-08-18T09:21:17Z",
         "updated_at": "2025-08-18T09:21:17Z"},
        {"id": 21, "status": "failure", "description": "Has been cancelled",
         "target_url": "/o/r/actions/runs/2706/jobs/2",
         "context": "/ compile_only (img:latest) (pull_request)",
         "created_at": "2025-08-18T09:21:17Z",
         "updated_at": "2025-08-18T09:21:17Z"},
        # One real success.
        {"id": 15, "status": "success", "description": "Successful in 8s",
         "target_url": "/o/r/actions/runs/2707/jobs/0",
         "context": "/ pr_labeler (pull_request_target)",
         "created_at": "2025-08-17T05:10:09Z",
         "updated_at": "2025-08-17T05:10:09Z"},
        # Older pending rows that were superseded by the failures /
        # success above. effective_commit_statuses must keep only the
        # latest per context.
        {"id": 6, "status": "pending", "description": "Has started running",
         "target_url": "/o/r/actions/runs/2707/jobs/0",
         "context": "/ pr_labeler (pull_request_target)",
         "created_at": "2025-08-17T05:10:01Z",
         "updated_at": "2025-08-17T05:10:01Z"},
        {"id": 1, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/o/r/actions/runs/2705/jobs/0",
         "context": "/ lint (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 2, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/o/r/actions/runs/2706/jobs/0",
         "context": "/ run_fate (linux-aarch64) (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 3, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/o/r/actions/runs/2706/jobs/1",
         "context": "/ run_fate (linux-amd64) (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 4, "status": "pending", "description": "Blocked by required conditions",
         "target_url": "/o/r/actions/runs/2706/jobs/2",
         "context": "/ compile_only (img:latest) (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 5, "status": "pending", "description": "Waiting to run",
         "target_url": "/o/r/actions/runs/2707/jobs/0",
         "context": "/ pr_labeler (pull_request_target)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
    ]
    SAMPLE = [forge_gcli._project_status_row(r) for r in WIRE]

    def test_effective_commit_statuses_classifies_cancellations_correctly(self) -> None:
        eff = fairy.effective_commit_statuses(self.SAMPLE)
        # Five contexts in total (one per unique context string).
        self.assertEqual(len(eff), 5)
        cancelled = sorted(c for c, (s, _) in eff.items() if s == "CANCELLED")
        successes = sorted(c for c, (s, _) in eff.items() if s == "SUCCESS")
        self.assertEqual(
            cancelled,
            [
                "/ compile_only (img:latest) (pull_request)",
                "/ lint (pull_request)",
                "/ run_fate (linux-aarch64) (pull_request)",
                "/ run_fate (linux-amd64) (pull_request)",
            ],
        )
        self.assertEqual(successes, ["/ pr_labeler (pull_request_target)"])

    def test_build_ci_failure_details_returns_empty(self) -> None:
        # All 4 "failure" rows are cancellations and the only real
        # success row is success: nothing nag-worthy remains.
        self.assertEqual(
            fairy.build_ci_failure_details(self.SAMPLE), []
        )

    def test_cancelled_extraction_returns_all_four(self) -> None:
        self.assertEqual(
            fairy.extract_contexts_with_state(self.SAMPLE, "CANCELLED"),
            (
                "/ compile_only (img:latest) (pull_request)",
                "/ lint (pull_request)",
                "/ run_fate (linux-aarch64) (pull_request)",
                "/ run_fate (linux-amd64) (pull_request)",
            ),
        )

    def test_blocked_extraction_returns_empty(self) -> None:
        # In this fixture the older "Blocked by required conditions"
        # rows are superseded by the newer "Has been cancelled"
        # failure rows, so the LATEST per context is CANCELLED, not
        # BLOCKED. The blocked-state filter must therefore return
        # nothing -- otherwise a context would be double-counted in
        # both the cancelled and blocked end-of-run summary lists.
        self.assertEqual(
            fairy.extract_contexts_with_state(self.SAMPLE, "BLOCKED"), ()
        )


class ForgejoActionsBlockedJobsPayloadTests(unittest.TestCase):
    """End-to-end regression test for a PR whose CI is genuinely blocked.

    Sample shape: required-environment review never granted, so the
    legacy commit-status API only ever shows ``status="pending"``
    rows with ``description="Blocked by required conditions"``. No
    failure rows ever appear because the jobs never start.

    Expected behavior:
    - ``effective_commit_statuses`` reports each context as ``BLOCKED``.
    - ``extract_contexts_with_state(.., "BLOCKED")`` returns the
      affected names.
    - ``extract_contexts_with_state(.., "CANCELLED")`` returns
      nothing -- a genuinely blocked job is not the same as a
      cancelled one.
    - ``build_ci_failure_details`` returns ``[]`` -- the LLM is never
      asked to nag about a job that is gated on an admin action.
    """

    WIRE = [
        {"id": 1, "status": "pending",
         "description": "Blocked by required conditions",
         "context": "/ lint (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 2, "status": "pending",
         "description": "Blocked by required conditions",
         "context": "/ build (pull_request)",
         "created_at": "2025-08-17T05:10:00Z",
         "updated_at": "2025-08-17T05:10:00Z"},
        {"id": 3, "status": "success", "description": "Successful in 8s",
         "context": "/ pr_labeler (pull_request_target)",
         "created_at": "2025-08-17T05:10:09Z",
         "updated_at": "2025-08-17T05:10:09Z"},
    ]
    SAMPLE = [forge_gcli._project_status_row(r) for r in WIRE]

    def test_effective_commit_statuses_classifies_blocked(self) -> None:
        eff = fairy.effective_commit_statuses(self.SAMPLE)
        blocked = sorted(c for c, (s, _) in eff.items() if s == "BLOCKED")
        self.assertEqual(
            blocked,
            ["/ build (pull_request)", "/ lint (pull_request)"],
        )

    def test_build_ci_failure_details_returns_empty(self) -> None:
        self.assertEqual(
            fairy.build_ci_failure_details(self.SAMPLE), []
        )

    def test_blocked_extraction_returns_both(self) -> None:
        self.assertEqual(
            fairy.extract_contexts_with_state(self.SAMPLE, "BLOCKED"),
            ("/ build (pull_request)", "/ lint (pull_request)"),
        )

    def test_cancelled_extraction_returns_empty(self) -> None:
        self.assertEqual(
            fairy.extract_contexts_with_state(self.SAMPLE, "CANCELLED"), ()
        )


if __name__ == "__main__":
    unittest.main()
