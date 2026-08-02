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

Tests for CI failure-detail URL absolutization.

Forgejo's ``GET /commits/<sha>/statuses`` endpoint returns ``target_url``
as a path-only string for some integrations (notably Forgejo Actions,
which emits e.g. ``/<owner>/<repo>/actions/runs/<id>/jobs/<id>``). The
LLM was forwarding that path verbatim into the CI heads-up reply, so
the bot was posting comments containing literal paths that were not
clickable links anywhere they were rendered (PR thread, email digests,
etc.).

``build_ci_failure_details`` now takes a ``base_url`` keyword and
resolves every ``target_url`` against it via ``urllib.parse.urljoin``.
That makes the LLM see absolute URLs only, regardless of which shape
the upstream API returned.

These tests pin the three input shapes ``urljoin`` has to handle:
already-absolute URL (left alone), absolute path (host substitution),
and relative path (resolved against the base path). They also cover
the empty/missing edge cases so a malformed status row never crashes
the wrapper or strips the existing data.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


PR_HTML_URL = "https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/20493"


def _row(
    *,
    context: str,
    state: str,
    target_url: str,
    description: str = "",
) -> dict:
    return {
        "context": context,
        "state": state,
        "description": description,
        "target_url": target_url,
        "created_at": "2026-04-26T20:00:00Z",
        "updated_at": "2026-04-26T20:00:00Z",
    }


class AbsolutizeTargetUrlUnitTests(unittest.TestCase):
    def test_path_only_url_gets_scheme_and_host_from_base(self) -> None:
        self.assertEqual(
            fairy.absolutize_target_url(
                "/FFmpeg/FFmpeg/actions/runs/6735/jobs/0", PR_HTML_URL
            ),
            "https://code.ffmpeg.org/FFmpeg/FFmpeg/actions/runs/6735/jobs/0",
        )

    def test_already_absolute_url_is_passed_through(self) -> None:
        absolute = "https://other.example.com/some/run/42"
        self.assertEqual(
            fairy.absolutize_target_url(absolute, PR_HTML_URL),
            absolute,
        )

    def test_relative_path_is_resolved_against_base_path(self) -> None:
        # urljoin replaces the last path segment of the base.
        self.assertEqual(
            fairy.absolutize_target_url("foo", PR_HTML_URL),
            "https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/foo",
        )

    def test_empty_url_stays_empty(self) -> None:
        self.assertEqual(fairy.absolutize_target_url("", PR_HTML_URL), "")

    def test_empty_base_url_passes_url_through_verbatim(self) -> None:
        self.assertEqual(
            fairy.absolutize_target_url("/x/y", ""),
            "/x/y",
        )


class BuildCiFailureDetailsAbsolutizationTests(unittest.TestCase):
    def test_path_only_target_url_is_absolutized(self) -> None:
        statuses = [
            _row(
                context="/ run_fate (32, linux-amd64) (pull_request)",
                state="failure",
                target_url="/FFmpeg/FFmpeg/actions/runs/6735/jobs/0",
                description="exited with code 1",
            )
        ]
        out = fairy.build_ci_failure_details(statuses, base_url=PR_HTML_URL)
        self.assertEqual(len(out), 1)
        self.assertEqual(
            out[0]["target_url"],
            "https://code.ffmpeg.org/FFmpeg/FFmpeg/actions/runs/6735/jobs/0",
        )

    def test_already_absolute_target_url_is_passed_through(self) -> None:
        statuses = [
            _row(
                context="ci/jenkins",
                state="error",
                target_url="https://ci.example.org/build/123",
            )
        ]
        out = fairy.build_ci_failure_details(statuses, base_url=PR_HTML_URL)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["target_url"], "https://ci.example.org/build/123")

    def test_no_base_url_keeps_target_url_verbatim(self) -> None:
        statuses = [
            _row(
                context="ci/jenkins",
                state="failure",
                target_url="/path/only",
            )
        ]
        out = fairy.build_ci_failure_details(statuses)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["target_url"], "/path/only")

    def test_missing_target_url_field_yields_empty_string(self) -> None:
        statuses = [
            {
                "context": "ci/jenkins",
                "state": "failure",
                "description": "boom",
                "created_at": "2026-04-26T20:00:00Z",
                "updated_at": "2026-04-26T20:00:00Z",
            }
        ]
        out = fairy.build_ci_failure_details(statuses, base_url=PR_HTML_URL)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["target_url"], "")

    def test_succeeded_context_is_filtered_out(self) -> None:
        """Sanity check: absolutization must not change which contexts
        get reported -- only failing contexts surface, regardless of
        what shape their target_url has."""
        statuses = [
            _row(
                context="green",
                state="success",
                target_url="/ok",
            ),
            _row(
                context="red",
                state="failure",
                target_url="/bad",
            ),
        ]
        out = fairy.build_ci_failure_details(statuses, base_url=PR_HTML_URL)
        self.assertEqual([d["context"] for d in out], ["red"])
        self.assertEqual(out[0]["target_url"], "https://code.ffmpeg.org/bad")


if __name__ == "__main__":
    unittest.main()
