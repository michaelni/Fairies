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

import forge_gcli  # noqa: E402
import forgejo_export  # noqa: E402
import gcli_cache  # noqa: E402

NOW = datetime(2026, 5, 26, tzinfo=timezone.utc)
LIVE_ISO = "2026-05-25T12:00:00Z"
MAX_AGE = timedelta(hours=24)


def _args() -> SimpleNamespace:
    return SimpleNamespace(owner="o", repo="r")


def _item(number: int = 42, updated_at: str | None = LIVE_ISO) -> dict:
    return {"number": number, "updated_at": updated_at}


class OpenCachesTests(unittest.TestCase):
    """open_caches loads one Cache per distinct path: derived per-kind
    files by default, a single shared object under an explicit --cache
    so the whole-file saves cannot discard each other's entries."""

    def _open(self, cache: Path | None):
        args = SimpleNamespace(cache=cache, forge_type="gitea",
                               gcli_account="acct", owner="o", repo="r")
        with patch.object(gcli_cache, "load_cache",
                          side_effect=lambda path: gcli_cache.Cache()):
            return forgejo_export.open_caches(args)

    def test_default_is_one_derived_file_per_kind(self) -> None:
        caches, by_path = self._open(None)
        self.assertEqual({path.name for path in by_path},
                         {"gitea_acct_o_r_issues.pkl",
                          "gitea_acct_o_r_pulls.pkl"})
        self.assertIsNot(caches["issues"], caches["pulls"])

    def test_explicit_cache_is_one_shared_object(self) -> None:
        caches, by_path = self._open(Path("x.pkl"))
        self.assertEqual(set(by_path), {Path("x.pkl")})
        self.assertIs(caches["issues"], caches["pulls"])


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
            gc.return_value = _item(), {
                "timeline": ({"type": "label", "id": 1},),
                "issue_comments": ({"id": 1, "body": "hi"},),
                "reviews": (),
                "review_comments": (),
                "commits": ({"sha": "abc"},),
                "files": (),
            }
            _, fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(), "pulls",
                forgejo_export.PR_FIELDS,
                now=NOW, cache_max_age=MAX_AGE, reactions_max_age=MAX_AGE,
            )
        self.assertEqual(warnings, [])
        self.assertEqual(fields["issue_comments"], [{"id": 1, "body": "hi"}])
        self.assertEqual(fields["commits"], [{"sha": "abc"}])
        self.assertEqual(fields["files"], [])
        self.assertEqual(gc.call_args.args[5], _item())
        self.assertEqual(gc.call_args.kwargs["max_age"], MAX_AGE)
        self.assertEqual(gc.call_args.kwargs["now"], NOW)

    def test_missing_updated_at_returns_empty_fields_and_warning(self) -> None:
        # No updated_at -> cannot gate the cache -> we'd refetch every
        # run forever. Better to skip with a warning than fake freshness.
        _, fields, warnings = forgejo_export.get_item_fields(
            _args(), gcli_cache.Cache(), _item(updated_at=None), "pulls",
            forgejo_export.PR_FIELDS,
            now=NOW, cache_max_age=MAX_AGE, reactions_max_age=MAX_AGE,
        )
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing or has unparseable updated_at", warnings[0])
        self.assertEqual(set(fields), set(forgejo_export.PR_FIELDS))
        self.assertTrue(all(v == [] for v in fields.values()))

    def test_fetch_failure_returns_empty_fields_and_warning(self) -> None:
        # A failed sub-fetch must not abort the whole export run.
        with patch.object(gcli_cache, "get", side_effect=RuntimeError("boom")):
            _, fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(number=7), "issues",
                forgejo_export.ISSUE_FIELDS,
                now=NOW, cache_max_age=MAX_AGE, reactions_max_age=MAX_AGE,
            )
        self.assertEqual(len(warnings), 1)
        self.assertIn("issues #7", warnings[0])
        self.assertIn("boom", warnings[0])
        self.assertEqual(fields, {"timeline": [], "issue_comments": []})


    def test_the_item_comes_back_as_of_its_fields(self) -> None:
        moved = _item(updated_at="2026-05-21T10:05:00Z")
        with patch.object(gcli_cache, "get", return_value=(
                moved, {"timeline": (), "issue_comments": ({"id": 1},)})):
            item, fields, warnings = forgejo_export.get_item_fields(
                _args(), gcli_cache.Cache(), _item(), "issues",
                forgejo_export.ISSUE_FIELDS,
                now=NOW, cache_max_age=MAX_AGE, reactions_max_age=MAX_AGE,
            )
        self.assertIs(item, moved)
        self.assertEqual(fields["issue_comments"], [{"id": 1}])
        self.assertEqual(warnings, [])


class TimelineRenderTests(unittest.TestCase):
    def test_empty_timeline_renders_placeholder(self) -> None:
        lines: list[str] = []
        forgejo_export.add_timeline_section(lines, [])
        self.assertEqual(lines, ["## Timeline", "", "_No timeline events._", ""])

    def test_typed_events_render_without_the_comments(self) -> None:
        raw = json.loads((Path(__file__).parent / "fixtures" / "issue_fairy"
                          / "ffmpeg_issue_23738_timeline.json").read_text())
        lines: list[str] = []
        forgejo_export.add_timeline_section(lines, forgejo_export.norm_timeline(
            [forge_gcli.project_timeline_event(event) for event in raw]))
        text = "\n".join(lines)
        self.assertIn("### Event 1: label_added\n\nFrom: ffmpeg-devel\n"
                      "When: 2026-07-08 10:42:44 UTC\nlabel: new\n", text)
        self.assertIn("### Event 2: label_removed", text)
        self.assertNotIn("### Event 3", text)


if __name__ == "__main__":
    unittest.main()
