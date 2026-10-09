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

What fairy gives the LLM and what forgejo_export writes about an item:
one timeline holding each comment once."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402
import forge_item  # noqa: E402
import forgejo_export  # noqa: E402
import issue_fairy  # noqa: E402
from fairy import LLMReview  # noqa: E402

import test_issue_fairy  # noqa: E402
from test_issue_fairy import load_fixture, make_args, prepare, real_issue  # noqa: E402
from test_own_pull_request import PR, _args, llm_payload  # noqa: E402

COMMENTS = load_fixture("ffmpeg_issue_23738_comments.json")
TIMELINE = [forge_gcli.project_timeline_event(e)
            for e in load_fixture("ffmpeg_issue_23738_timeline.json")]


def issue_llm_discussion() -> list[dict]:
    prepared = prepare(make_args(), real_issue(),
                       now=test_issue_fairy.PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
    with mock.patch.object(issue_fairy, "invoke_llm_wrapper",
                           return_value=LLMReview("skip", "")) as invoke:
        issue_fairy.run_llm_issue(make_args(), prepared, None)
    return invoke.call_args.args[1]["discussion"]


def pr_llm_discussion(reviews: list[dict], comments: list[dict], timeline: list[dict]) -> list[dict]:
    return llm_payload(_args(), discussion=forge_item.build_llm_discussion(
        reviews, comments, [], timeline))["discussion"]


def exported_issue() -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as target:
        forgejo_export.export_issue(Path(target), real_issue(), COMMENTS, TIMELINE, [], [])
        base = Path(target) / "issues" / "023738"
        return json.loads(base.with_suffix(".json").read_text()), base.with_suffix(".md").read_text()


def exported_pr(reviews: list[dict], comments: list[dict], timeline: list[dict]) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as target:
        forgejo_export.export_pr(Path(target), PR, comments, reviews, [], [], [], timeline, [], "no", [])
        base = Path(target) / "pulls" / f"{PR['number']:06d}"
        return json.loads(base.with_suffix(".json").read_text()), base.with_suffix(".md").read_text()


class DuplicateCommentTests(unittest.TestCase):
    """Forgejo's timeline repeats every comment of the comment listing;
    issue 23738's timeline carries all 8 of its comments."""

    def assert_each_comment_once(self, discussion: list[dict]) -> None:
        self.assertEqual(sorted(item["id"] for item in discussion if item["kind"] == "comment"),
                         sorted(comment["id"] for comment in COMMENTS))

    def test_the_llm_gets_each_comment_once(self) -> None:
        self.assert_each_comment_once(issue_llm_discussion())
        self.assert_each_comment_once(pr_llm_discussion([], COMMENTS, TIMELINE))

    def test_the_export_writes_each_comment_once(self) -> None:
        for data, text in (exported_issue(), exported_pr([], COMMENTS, TIMELINE)):
            self.assert_each_comment_once(data["timeline"])
            for comment in COMMENTS:
                self.assertEqual(text.count(comment["body"].rstrip()), 1)


if __name__ == "__main__":
    unittest.main()
