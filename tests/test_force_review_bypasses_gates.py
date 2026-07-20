"""``--force-review-pr`` bypasses the state/WIP/mergeable skip gates.

Without the bypass these gates fire before ``forced_review_reason``
ever sees the override, so a forced run on a closed/merged, draft,
or conflicting PR returns a Decision skip and the LLM never looks
at the PR. Bit the simulate-past harness on its first end-to-end
run (3 of 4 candidate PRs got 0 LLM responses).

We patch ``get_pr_discussion`` to a sentinel exception: a forced
run reaches the patched call and raises; an unforced run returns a
Decision before ever touching it.
"""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gcli_cache  # noqa: E402
import fairy  # noqa: E402
import workset  # noqa: E402

WIP_RE = re.compile(r"\b(WIP|DRAFT)\b", re.IGNORECASE)


class _PastGates(Exception):
    """Sentinel: ``prepare_pr`` reached past the early-skip gates."""


def _call(pr: dict, *, force_review: set[int] = frozenset(),
          force_skip: set[int] = frozenset(),
          force_review_non_open: bool = False,
          force_review_skip: bool = False,
          workset_dir: Path | None = None,
          now: datetime | None = None) -> object:
    args = SimpleNamespace(
        force_skip_prs=force_skip,
        force_review_prs=force_review,
        force_review_non_open=force_review_non_open,
        force_review_skip=force_review_skip,
        owner="o",
        repo="r",
        forge_type="gitea",
        gcli_account=None,
        workset_dir=workset_dir,
        simulate_past=None,
        verbose=False,
        llm_review_cmd=None,
    )
    return fairy.prepare_pr(
        args, pr, now=now, self_login=None, wip_re=WIP_RE,
        cache=gcli_cache.Cache(),
        discussion_cache_max_age=timedelta(hours=1),
    )


class ForceReviewBypassesGatesTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.object(
            fairy, "get_pr_discussion", side_effect=_PastGates(),
        )
        self.mock_disc = patcher.start()
        self.addCleanup(patcher.stop)

    def test_wip_and_mergeable_gates_always_bypassed_by_force(self) -> None:
        # WIP/draft and conflicting PRs are reviewed whenever forced,
        # independent of --force-review-non-open.
        cases = [
            ("mergeable", {"state": "open", "mergeable": False, "title": "x"},
             "has conflicts with the target branch"),
            ("wip-title", {"state": "open", "mergeable": True,
                           "title": "WIP: refactor"},
             "marked WIP/draft"),
            ("draft-flag", {"state": "open", "mergeable": True, "title": "ok",
                            "draft": True},
             "marked WIP/draft"),
        ]
        for label, fields, expected_reason in cases:
            with self.subTest(label):
                pr = {"number": 1, **fields}
                self.mock_disc.reset_mock()
                # Unforced: returns Decision skip with the gate's reason.
                decision = _call(pr)
                self.assertEqual(decision.action, "skip")
                self.assertEqual(decision.reason, expected_reason)
                self.mock_disc.assert_not_called()
                # Forced: proceeds past the gate (sentinel fires).
                self.mock_disc.reset_mock()
                with self.assertRaises(_PastGates):
                    _call(pr, force_review={1})
                self.mock_disc.assert_called_once()

    def test_non_open_gate_is_opt_in_via_force_review_non_open(self) -> None:
        # A non-open PR is skipped on a bare forced run; only
        # --force-review-non-open lets the forced review reach the PR.
        pr = {"number": 1, "state": "closed", "mergeable": True, "title": "x"}

        decision = _call(pr)
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "not open")
        self.mock_disc.assert_not_called()

        decision = _call(pr, force_review={1})
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "not open")
        self.mock_disc.assert_not_called()

        with self.assertRaises(_PastGates):
            _call(pr, force_review={1}, force_review_non_open=True)
        self.mock_disc.assert_called_once()

    def test_force_skip_takes_precedence_over_force_review(self) -> None:
        # Pinned so a refactor cannot quietly invert the documented
        # precedence of --force-skip-pr over --force-review-pr.
        pr = {"number": 5, "state": "open", "mergeable": True, "title": "ok"}
        decision = _call(pr, force_review={5}, force_skip={5})
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "forced skip by --force-skip-pr")
        self.mock_disc.assert_not_called()


class ForceReviewBypassesBackoffGateTests(unittest.TestCase):
    """``--force-review-pr`` must also bypass the LLM skip-backoff gate.

    Regression: the backoff gate sat after the discussion fetch and
    returned a free skip even for a PR named on --force-review-pr, so
    an operator could not force a fresh review of a PR parked in its
    doubling window (observed on #22624: "in LLM skip-backoff window
    after 1 consecutive skip(s)"). The state/WIP/mergeable gates above
    it already honored the force; this gate did not.

    Unlike the gates above, this one lives past ``get_pr_discussion``,
    so here that call returns an empty discussion (rather than the
    sentinel) and a triggering backoff entry is seeded. The forced run
    is expected to reach ``get_auto_merge_info`` on the forced-review
    path -- patched to the sentinel -- proving the gate was skipped.
    """

    HEAD = "abc123"
    NUMBER = 7
    NOW = datetime(2026, 5, 29, tzinfo=timezone.utc)

    def setUp(self) -> None:
        patcher = patch.object(
            fairy, "get_pr_discussion", return_value=([], [], []),
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _pr(self) -> dict:
        return {
            "number": self.NUMBER,
            "state": "open",
            "mergeable": True,
            "title": "ok",
            "head": {"sha": self.HEAD},
        }

    def _workset_in_window(self) -> Path:
        # one prior skip on this exact head; the 24h window is still open at NOW
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        now = self.NOW.isoformat()
        workset.save_item(
            root / "gitea~default~o~r" / f"pr-{self.NUMBER}.json",
            workset.WorkItem(
                kind="pr", forge_type="gitea", account="", owner="o", repo="r",
                number=self.NUMBER, state=workset.WorkState.REVIEWED,
                created_at=now, state_changed_at=now,
                expected_head_ref=self.HEAD, last_activity_iso=None,
                llm_at=(self.NOW - timedelta(hours=1)).isoformat(),
                consecutive_skip_count=1,
                review=workset.ReviewResult(classification="skip", message=""),
            ))
        return root

    def test_unforced_skips_inside_window(self) -> None:
        decision = _call(self._pr(), workset_dir=self._workset_in_window(),
                         now=self.NOW)
        self.assertEqual(decision.action, "skip")
        self.assertIn("skip-backoff window", decision.reason)

    def test_forced_bypasses_window(self) -> None:
        with patch.object(
            fairy, "get_auto_merge_info", side_effect=_PastGates(),
        ):
            with self.assertRaises(_PastGates):
                _call(
                    self._pr(),
                    force_review={self.NUMBER},
                    workset_dir=self._workset_in_window(),
                    now=self.NOW,
                )


class ForceReviewSkipPayloadTests(unittest.TestCase):
    """``--force-review-skip`` / ``--force-engage`` ride to the wrapper as
    the ``ignore_triage_skip`` / ``force_engage`` request fields.

    The override is enforced inside the --llm-review-cmd subprocess, so
    the only thing fairy owns is putting the flag into the
    request payload it writes to that process's stdin. Pin that each
    field appears only when asked, so the cross-process contract with
    pr_review_wrapper does not silently drift.
    """

    def _payload_for(self, **flags: bool) -> dict:
        args = SimpleNamespace(
            llm_review_cmd="./wrapper",
            triage_labels=[],
            llm_max_patch_bytes=10_000,
            verbose=0,
            llm_timeout=None,
        )
        captured: dict[str, str] = {}

        def fake_run_cmd(_cmd, *, input_text: str, **_kw):
            captured["input_text"] = input_text
            return SimpleNamespace(
                returncode=0,
                stdout='{"classification": "approve", "message": ""}',
            )

        with (
            patch.object(fairy, "patch_shas_for_run",
                         return_value=("base", "head")),
            patch.object(fairy, "fetch_patch_for_llm",
                         return_value=("patch", False)),
            patch.object(fairy, "run_cmd", side_effect=fake_run_cmd),
        ):
            fairy.run_llm_review(
                args,
                {"number": 1, "title": "t"},
                "no",
                [],
                None,
                **flags,
            )
        import json
        return json.loads(captured["input_text"])

    def test_flag_set_adds_field(self) -> None:
        self.assertTrue(self._payload_for(ignore_triage_skip=True)["ignore_triage_skip"])

    def test_flag_unset_omits_field(self) -> None:
        self.assertNotIn("ignore_triage_skip", self._payload_for(ignore_triage_skip=False))

    def test_force_engage_set_adds_field(self) -> None:
        self.assertTrue(self._payload_for(force_engage=True)["force_engage"])

    def test_force_engage_unset_omits_field(self) -> None:
        self.assertNotIn("force_engage", self._payload_for())


if __name__ == "__main__":
    unittest.main()
