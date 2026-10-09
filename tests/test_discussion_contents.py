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
one timeline holding each comment once, the full name of each author
named and every commit of each push."""

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
import llm_prompt  # noqa: E402
from fairy import LLMReview  # noqa: E402

import test_issue_fairy  # noqa: E402
from test_issue_fairy import load_fixture, make_args, prepare, real_issue  # noqa: E402
from test_own_pull_request import PR, _args, llm_payload  # noqa: E402

COMMENTS = load_fixture("ffmpeg_issue_23738_comments.json")
TIMELINE = [forge_gcli.project_timeline_event(e)
            for e in load_fixture("ffmpeg_issue_23738_timeline.json")]
NAMED_PR = next(pr for pr in load_fixture("ffmpeg_open_pulls.json") if pr["user"]["full_name"])
RAW_PR_TIMELINE = json.loads((REPO_ROOT / "tests" / "fixtures" / "forgejo_pr_timeline"
                              / "ffmpeg_pr_22729_timeline.json").read_text())
PR_TIMELINE = [forge_gcli.project_timeline_event(e) for e in RAW_PR_TIMELINE]
PUSHED = [json.loads(e["body"])["commit_ids"] for e in RAW_PR_TIMELINE if e["type"] == "pull_push"]


def issue_llm_input() -> tuple[dict, str]:
    """The payload fairy hands the LLM wrapper for issue 23738, and the
    prompt text the wrapper makes of it."""
    prepared = prepare(make_args(), real_issue(),
                       now=test_issue_fairy.PrepareIssueGateTests.LAST_COMMENT + timedelta(days=2))
    with mock.patch.object(issue_fairy, "invoke_llm_wrapper",
                           return_value=LLMReview("skip", "")) as invoke:
        issue_fairy.run_llm_issue(make_args(), prepared, None)
    payload = invoke.call_args.args[1]
    return payload, llm_prompt.make_issue_user_text(payload)


def pr_llm_input(pr: dict, reviews: list[dict], comments: list[dict],
                 timeline: list[dict]) -> tuple[dict, str]:
    payload = llm_payload(_args(), pr, discussion=forge_item.build_llm_discussion(
        reviews, comments, [], timeline))
    return payload, llm_prompt.make_triage_user_text(payload, False)


def exported_issue() -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as target:
        forgejo_export.export_issue(Path(target), real_issue(), COMMENTS, TIMELINE, [], [])
        base = Path(target) / "issues" / "023738"
        return json.loads(base.with_suffix(".json").read_text()), base.with_suffix(".md").read_text()


def exported_pr(pr: dict, reviews: list[dict], comments: list[dict],
                timeline: list[dict]) -> tuple[dict, str]:
    with tempfile.TemporaryDirectory() as target:
        forgejo_export.export_pr(Path(target), pr, comments, reviews, [], [], [], timeline, [], "no", [])
        base = Path(target) / "pulls" / f"{pr['number']:06d}"
        return json.loads(base.with_suffix(".json").read_text()), base.with_suffix(".md").read_text()


class DuplicateCommentTests(unittest.TestCase):
    """Forgejo's timeline repeats every comment of the comment listing;
    issue 23738's timeline carries all 8 of its comments."""

    def assert_each_comment_once(self, discussion: list[dict]) -> None:
        self.assertEqual(sorted(item["id"] for item in discussion if item["kind"] == "comment"),
                         sorted(comment["id"] for comment in COMMENTS))

    def test_the_llm_gets_each_comment_once(self) -> None:
        for payload, text in (issue_llm_input(), pr_llm_input(PR, [], COMMENTS, TIMELINE)):
            self.assert_each_comment_once(payload["discussion"])
            for comment in COMMENTS:
                self.assertEqual(text.count(json.dumps(comment["body"], ensure_ascii=False)), 1)

    def test_the_export_writes_each_comment_once(self) -> None:
        for data, text in (exported_issue(), exported_pr(PR, [], COMMENTS, TIMELINE)):
            self.assert_each_comment_once(data["timeline"])
            for comment in COMMENTS:
                self.assertEqual(text.count(comment["body"].rstrip()), 1)


class FullNameTests(unittest.TestCase):
    """Whatever names an author carries the author's full name, so no
    reader has to look the login up elsewhere."""

    NAMED = {comment["id"]: comment["user"]["full_name"]
             for comment in COMMENTS if comment["user"]["full_name"]}

    def assert_comment_authors_named(self, discussion: list[dict]) -> None:
        self.assertTrue(self.NAMED)
        self.assertEqual({item["id"]: item.get("author_full_name") for item in discussion
                          if item.get("id") in self.NAMED}, self.NAMED)

    def test_the_llm_gets_the_full_names(self) -> None:
        for (payload, text), names in (
                (issue_llm_input(), self.NAMED.values()),
                (pr_llm_input(NAMED_PR, [], COMMENTS, TIMELINE),
                 (NAMED_PR["user"]["full_name"], *self.NAMED.values()))):
            self.assert_comment_authors_named(payload["discussion"])
            for name in names:
                self.assertIn(json.dumps(name, ensure_ascii=False), text)

    def test_the_export_writes_the_full_names(self) -> None:
        data, text = exported_issue()
        self.assert_comment_authors_named(data["timeline"])
        for comment in COMMENTS:
            if comment["user"]["full_name"]:
                self.assertIn(f"From: {comment['user']['login']} ({comment['user']['full_name']})", text)
        data, text = exported_pr(NAMED_PR, [], COMMENTS, TIMELINE)
        self.assertIn(f"- Author: {NAMED_PR['user']['login']} ({NAMED_PR['user']['full_name']}, ", text)



class PushTests(unittest.TestCase):
    """PR 22729 got a push of one commit, then a force-push of two."""

    def assert_every_commit(self, discussion: list[dict], text: str) -> None:
        self.assertEqual([len(commit_ids) for commit_ids in PUSHED], [1, 2])
        self.assertEqual([item["commit_ids"] for item in discussion if item["kind"] == "push"], PUSHED)
        for sha in sum(PUSHED, []):
            self.assertIn(sha, text)

    def test_the_llm_gets_every_pushed_commit(self) -> None:
        payload, text = pr_llm_input(PR, [], [], PR_TIMELINE)
        self.assert_every_commit(payload["discussion"], text)

    def test_the_export_writes_every_pushed_commit(self) -> None:
        data, text = exported_pr(PR, [], [], PR_TIMELINE)
        self.assert_every_commit(data["timeline"], text)


if __name__ == "__main__":
    unittest.main()
