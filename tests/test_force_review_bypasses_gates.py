"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License version 2 for more details.
 *
 * Additional permission:
 *
 * Michael Niedermayer is permitted to relicense this file, in whole or
 * in part, under any version of the GNU General Public License, the GNU
 * Affero General Public License, or the GNU Lesser General Public License
 * published by the Free Software Foundation.
 *
 * This additional permission is personal to Michael Niedermayer.  It is
 * not transferable and does not grant any other person permission to
 * relicense this file under a different license.
 *
 * This additional permission may be removed from modified copies of this
 * file.  Removal of this additional permission does not affect the
 * licensing of the file under the GNU General Public License version 2.
 */

``--force-review`` bypasses the state/WIP/mergeable skip gates.

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
        # precedence of --force-skip over --force-review.
        pr = {"number": 5, "state": "open", "mergeable": True, "title": "ok"}
        decision = _call(pr, force_review={5}, force_skip={5})
        self.assertEqual(decision.action, "skip")
        self.assertEqual(decision.reason, "forced skip by --force-skip")
        self.mock_disc.assert_not_called()
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
