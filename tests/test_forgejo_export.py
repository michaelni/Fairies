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

Regression tests for ``forgejo_export``: attachment export and
``get_item_fields``.

The exporter is a thin adapter on top of ``gcli_cache.get`` -- it
just collapses gcli_cache's atomic-or-raise contract into a
best-effort ``(fields_dict, warnings_list)`` pair so a single failing
fetch does not abort the whole export run.
"""

from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forgejo_export  # noqa: E402
import gcli_cache  # noqa: E402

NOW = datetime(2026, 5, 26, tzinfo=timezone.utc)
LIVE_ISO = "2026-05-25T12:00:00Z"
LIVE = datetime(2026, 5, 25, 12, 0, tzinfo=timezone.utc)
MAX_AGE = timedelta(hours=24)


def _args() -> SimpleNamespace:
    return SimpleNamespace(owner="o", repo="r")


def _item(number: int = 42, updated_at: str | None = LIVE_ISO) -> dict:
    return {"number": number, "updated_at": updated_at}


class AttachmentExportTests(unittest.TestCase):
    """Exports must list forge attachments: issue 20572's ZIP was attached
    but not linked from the body, so the exported 020572.md had no trace
    of it. Reuses the real capture from fixtures/issue_fairy/."""

    def setUp(self) -> None:
        self.issue = json.loads(
            (Path(__file__).parent / "fixtures" / "issue_fairy"
             / "ffmpeg_issue_20572.json").read_text()
        )

    def _render(self, comments: list[dict]) -> str:
        return forgejo_export.render_issue_md({
            "number": self.issue["number"], "title": self.issue["title"],
            "issue": forgejo_export.norm_issue(self.issue),
            "comments": comments, "timeline": [], "fetch_warnings": [],
        })

    def test_issue_md_lists_unlinked_attachment(self) -> None:
        zip_asset = self.issue["assets"][3]
        self.assertIn(
            f"- Attachment: {zip_asset['name']} ({zip_asset['size']} bytes) "
            f"{zip_asset['browser_download_url']}",
            self._render([]),
        )

    def test_no_assets_omits_attachment_urls(self) -> None:
        # An always-present empty list rewrote every attachment-free
        # export file, forcing a full vector-store resync.
        self.assertNotIn("attachment_urls",
                         forgejo_export.norm_issue({**self.issue, "assets": []}))
        self.assertNotIn("attachment_urls",
                         forgejo_export.norm_comment({"id": 1, "body": "hi"}))

    def test_comment_attachment_renders(self) -> None:
        zip_asset = self.issue["assets"][3]
        comment = forgejo_export.norm_comment(
            {"id": 1, "user": self.issue["user"], "body": "sample attached",
             "assets": [zip_asset]}
        )
        self.assertIn(
            f"Attachment: {zip_asset['name']} ({zip_asset['size']} bytes) "
            f"{zip_asset['browser_download_url']}\n",
            self._render([comment]),
        )


class GetItemFieldsTests(unittest.TestCase):

    def test_success_returns_fields_and_no_warnings(self) -> None:
        with patch.object(gcli_cache, "get") as gc:
            gc.return_value = {
                "timeline": ({"type": "label", "id": 1},),
                "issue_comments": ({"id": 1, "body": "hi"},),
                "reviews": (),
                "review_comments": (),
                "commits": ({"sha": "abc"},),
                "files": (),
            }
            fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(), "pulls",
                forgejo_export.PR_FIELDS,
                now=NOW, cache_max_age=MAX_AGE,
            )
        self.assertEqual(warnings, [])
        self.assertEqual(fields["issue_comments"], [{"id": 1, "body": "hi"}])
        self.assertEqual(fields["commits"], [{"sha": "abc"}])
        self.assertEqual(fields["files"], [])
        # Sanity: gcli_cache.get was invoked with the parsed datetime,
        # not the ISO string, so the staleness check works.
        self.assertEqual(gc.call_args.args[6], LIVE)
        self.assertEqual(gc.call_args.kwargs["max_age"], MAX_AGE)
        self.assertEqual(gc.call_args.kwargs["now"], NOW)

    def test_missing_updated_at_returns_empty_fields_and_warning(self) -> None:
        # No updated_at -> cannot gate the cache -> we'd refetch every
        # run forever. Better to skip with a warning than fake freshness.
        with patch.object(gcli_cache, "get") as gc:
            fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(updated_at=None), "pulls",
                forgejo_export.PR_FIELDS,
                now=NOW, cache_max_age=MAX_AGE,
            )
        self.assertEqual(gc.call_count, 0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing updated_at", warnings[0])
        self.assertEqual(set(fields), set(forgejo_export.PR_FIELDS))
        self.assertTrue(all(v == [] for v in fields.values()))

    def test_fetch_failure_returns_empty_fields_and_warning(self) -> None:
        # A failed sub-fetch must not abort the whole export run.
        with patch.object(gcli_cache, "get", side_effect=RuntimeError("boom")):
            fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(number=7), "issues",
                forgejo_export.ISSUE_FIELDS,
                now=NOW, cache_max_age=MAX_AGE,
            )
        self.assertEqual(len(warnings), 1)
        self.assertIn("issues #7", warnings[0])
        self.assertIn("boom", warnings[0])
        self.assertEqual(fields, {"timeline": [], "issue_comments": []})


class TimelineRenderTests(unittest.TestCase):
    """The timeline section is the only new render path; pin its shape."""

    def test_empty_timeline_renders_placeholder(self) -> None:
        lines: list[str] = []
        forgejo_export.add_timeline_section(lines, [])
        self.assertEqual(lines, ["## Timeline", "", "_No timeline events._", ""])

    def test_event_renders_type_author_body(self) -> None:
        event = forgejo_export.norm_timeline_event({
            "type": "pull_push",
            "id": 7,
            "user": {"login": "alice", "id": 1},
            "created_at": "2026-05-25T10:00:00Z",
            "body": '{"is_force_push": false, "commit_ids": ["abc123"]}',
        })
        lines: list[str] = []
        forgejo_export.add_timeline_section(lines, [event])
        text = "\n".join(lines)
        self.assertIn("### Event 1: pull_push", text)
        self.assertIn("From: alice (id=1)", text)
        self.assertIn("When: 2026-05-25 10:00:00 UTC", text)
        self.assertIn('"commit_ids": ["abc123"]', text)

    def test_bodyless_event_still_renders_header(self) -> None:
        # Most typed events (label, assignee, milestone) carry no
        # ``body`` -- the type+when alone is enough chronology.
        event = forgejo_export.norm_timeline_event({
            "type": "label",
            "user": {"login": "bob"},
            "created_at": "2026-05-25T10:00:00Z",
        })
        lines: list[str] = []
        forgejo_export.add_timeline_section(lines, [event])
        text = "\n".join(lines)
        self.assertIn("### Event 1: label", text)
        self.assertIn("From: bob", text)


if __name__ == "__main__":
    unittest.main()
