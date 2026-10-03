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

Inline review comments cost one request on GitHub, not one per review.

The per-review walk skips a review whose ``comments_count`` is 0. GitHub's
review objects have no such field -- ``testrepo_pr4_reviews.json``, an
unmodified ``gcli -t github api`` capture from the project's own scratch
repository, carries
``id``/``state``/``submitted_at``/``user`` and no count -- so nothing can
be skipped and the walk costs one request per review against a 5000/hour
budget. GitHub serves every inline comment on the PR from a single
endpoint instead, captured here as
``testrepo_pr4_review_comments.json``.

Forgejo keeps the per-review walk and its skip; both are asserted, since
a silent extra request per review is exactly the kind of regression that
only shows up as a rate-limit failure in production.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402

FIXTURES = REPO_ROOT / "tests" / "fixtures" / "github"


def _load(name: str):
    with (FIXTURES / name).open() as f:
        return json.load(f)


def _fetch(forge_type: str, reviews: list[dict], payload: list[dict]):
    """Returns ``(comments, requested_paths)``."""
    paths: list[str] = []

    def fake_api(args, path, **kwargs):
        paths.append(path)
        return payload

    args = SimpleNamespace(forge_type=forge_type, gcli_account="", verbose=0)
    with mock.patch.object(forge_gcli, "gcli_api", fake_api):
        got = forge_gcli.list_pr_review_comments(args, "o", "r", 4, reviews)
    return got, paths


class GitHubTests(unittest.TestCase):

    def test_capture_has_no_comments_count_to_skip_on(self) -> None:
        review = _load("testrepo_pr4_reviews.json")[0]
        self.assertNotIn("comments_count", review)

    def test_one_request_regardless_of_review_count(self) -> None:
        reviews = [{"id": i} for i in range(1, 8)]
        comments, paths = _fetch(
            "github", reviews, _load("testrepo_pr4_review_comments.json"))
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("/pulls/4/comments"))
        self.assertEqual(len(comments), 1)

    def test_comments_carry_the_keys_the_contract_promises(self) -> None:
        comments, _ = _fetch(
            "github", [], _load("testrepo_pr4_review_comments.json"))
        self.assertLessEqual(
            {"id", "body", "user", "created_at", "updated_at", "path"},
            set(comments[0]),
        )


class ReactionTests(unittest.TestCase):
    """Real capture: FFmpeg issue #22546, the fairy's comment and its
    one reaction. Forgejo answers null where there are none."""

    def _attach(self, forge_type: str, comments: list[dict], reactions,
                scope: str = "issues", known: dict | None = None):
        paths: list[str] = []

        def fake_api(args, path, **kwargs):
            paths.append(path)
            return reactions

        args = SimpleNamespace(forge_type=forge_type, gcli_account="", verbose=0)
        with mock.patch.object(forge_gcli, "gcli_api", fake_api):
            forge_gcli.attach_reactions(args, "o", "r", comments, scope, known or {})
        return paths

    def test_forgejo_asks_per_comment_and_keeps_who_reacted_how(self) -> None:
        reactions_dir = REPO_ROOT / "tests" / "fixtures" / "reactions"
        with (reactions_dir / "ffmpeg_issue_22546_comments.json").open() as f:
            comments = json.load(f)
        with (reactions_dir / "ffmpeg_issue_22546_comment_37290_reactions.json").open() as f:
            reactions = json.load(f)
        paths = self._attach("forgejo", comments, reactions)
        self.assertEqual(paths, ["/repos/o/r/issues/comments/37290/reactions"])
        self.assertEqual(comments[0]["reactions"], [{"user": "michaelni", "content": "hooray"}])

    def test_forgejo_null_means_none(self) -> None:
        comments = [{"id": 5}]
        self._attach("forgejo", comments, None)
        self.assertEqual(comments[0]["reactions"], [])

    def test_known_reactions_spare_the_request(self) -> None:
        comments = [{"id": 5}, {"id": 6}]
        paths = self._attach("forgejo", comments, [], known={5: [{"user": "a", "content": "+1"}]})
        self.assertEqual(paths, ["/repos/o/r/issues/comments/6/reactions"])
        self.assertEqual([c["reactions"] for c in comments], [[{"user": "a", "content": "+1"}], []])

    def test_github_rollup_spares_the_request(self) -> None:
        comments = [{"id": 5, "reactions": {"total_count": 0}}]
        paths = self._attach("github", comments, [{"content": "+1"}])
        self.assertEqual(paths, [])
        self.assertEqual(comments[0]["reactions"], [])

    def test_github_review_comments_live_under_pulls(self) -> None:
        comments = [{"id": 7, "reactions": {"total_count": 1}}]
        paths = self._attach("github", comments,
                             [{"user": {"login": "michaelni"}, "content": "-1"}], scope="pulls")
        self.assertEqual(paths, ["/repos/o/r/pulls/comments/7/reactions"])
        self.assertEqual(comments[0]["reactions"], [{"user": "michaelni", "content": "-1"}])


class ForgejoTests(unittest.TestCase):

    def test_one_request_per_review_that_may_have_comments(self) -> None:
        reviews = [{"id": 1}, {"id": 2}]
        _, paths = _fetch("gitea", reviews, [])
        self.assertEqual(len(paths), 2)
        self.assertTrue(paths[0].endswith("/reviews/1/comments"))

    def test_an_empty_review_is_still_skipped(self) -> None:
        reviews = [{"id": 1, "comments_count": 0}, {"id": 2, "comments_count": 3}]
        _, paths = _fetch("gitea", reviews, [])
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("/reviews/2/comments"))


if __name__ == "__main__":
    unittest.main()
