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

The CI-failure heads-up must not re-announce jobs fairy already named.

Regression: FFmpeg #21079. Fairy posted "CI is still red … `Test / Fate
(Full, wine)` …" on 2026-05-24 and again on 2026-06-08; after each push the
bot re-announced the identical four FATE jobs. The dedup compared the raw
status context ``Test / Fate (Full, wine) (pull_request)`` verbatim against
her comment, which quotes the bare job name without the ``(pull_request)``
trigger-event suffix, so it never matched -- and target_url is per-run so it
cannot dedup across CI re-runs.

The contexts below are the real failing statuses on head
``79f38e97`` (event suffix appended by Forgejo Actions); the body is her
actual 2026-06-08 comment text (trimmed).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy as paa  # noqa: E402

PR_21079_FAILING_CONTEXTS = [
    "Test / Fate (Full, wine) (pull_request)",
    "Test / Fate (linux-amd64, static, 32 bit) (pull_request)",
    "Test / Fate (linux-amd64, shared, 64 bit) (pull_request)",
    "Test / Fate (linux-aarch64, static, 64 bit) (pull_request)",
]

# Her 2026-06-08 heads-up (the run URL is an older run than the current head).
PR_21079_PRIOR_COMMENT = (
    "LLM: heads-up, CI is still red on the current head. The failing FATE "
    "jobs are `Test / Fate (linux-amd64, static, 32 bit)`, `Test / Fate "
    "(linux-amd64, shared, 64 bit)`, `Test / Fate (linux-aarch64, static, 64 "
    "bit)`, and `Test / Fate (Full, wine)`; for example, see "
    "https://code.ffmpeg.org/FFmpeg/FFmpeg/actions/runs/56427/jobs/0 and the "
    "other job links in the PR status."
)


def _details(contexts: list[str]) -> list[dict[str, object]]:
    # Current-head target_urls differ from the run cited in her comment.
    return [
        {"context": c, "target_url": f"/FFmpeg/FFmpeg/actions/runs/59701/jobs/{i}"}
        for i, c in enumerate(contexts)
    ]


class StripCiEventSuffixTests(unittest.TestCase):
    def test_strips_known_event_suffixes(self) -> None:
        self.assertEqual(
            paa.strip_ci_event_suffix("Test / Fate (Full, wine) (pull_request)"),
            "Test / Fate (Full, wine)",
        )
        self.assertEqual(
            paa.strip_ci_event_suffix("/ pr_labeler (pull_request_target)"),
            "/ pr_labeler",
        )

    def test_leaves_matrix_parenthetical_intact(self) -> None:
        # No event suffix, and a real matrix value carries spaces/commas/
        # digits so it must never be stripped.
        self.assertEqual(
            paa.strip_ci_event_suffix("Test / Fate (linux-amd64, static, 32 bit)"),
            "Test / Fate (linux-amd64, static, 32 bit)",
        )


class CiMentionDedupTests(unittest.TestCase):
    def test_already_announced_jobs_not_reannounced(self) -> None:
        mentioned, need = paa.partition_ci_announcement(
            _details(PR_21079_FAILING_CONTEXTS),
            [PR_21079_PRIOR_COMMENT],
        )
        self.assertEqual(mentioned, PR_21079_FAILING_CONTEXTS)
        self.assertEqual(need, [])

    def test_a_newly_failing_context_still_needs_announcement(self) -> None:
        contexts = [*PR_21079_FAILING_CONTEXTS, "Test / Fate (macos-arm64) (pull_request)"]
        mentioned, need = paa.partition_ci_announcement(
            _details(contexts), [PR_21079_PRIOR_COMMENT],
        )
        self.assertEqual(need, ["Test / Fate (macos-arm64) (pull_request)"])

    def test_no_prior_comment_means_all_need_announcement(self) -> None:
        mentioned, need = paa.partition_ci_announcement(
            _details(PR_21079_FAILING_CONTEXTS), [],
        )
        self.assertEqual(mentioned, [])
        self.assertEqual(need, PR_21079_FAILING_CONTEXTS)


if __name__ == "__main__":
    unittest.main()
