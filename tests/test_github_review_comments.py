"""Inline review comments cost one request on GitHub, not one per review.

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
