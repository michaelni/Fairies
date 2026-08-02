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

Opt-in check that fairy's write paths reach GitHub.

Every write fairy performs -- a comment, a label change, an approving
review -- goes through a gcli subcommand that gcli implements per
backend. That is exactly where the Gitea label bug lived: ``labels add``
resolved a name to an id and sent it as a JSON string, so the forge
returned 200 and attached nothing. A fixture cannot catch that class of
bug, because the request is well-formed and the failure is on the far
side. Only a real round-trip can.

Writes to whatever repo it is pointed at, so it skips unless told where:

    FAIRY_GITHUB_WRITE_REPO=michaelni/testrepo \\
    FAIRY_GITHUB_WRITE_ISSUE=1 \\        # or FAIRY_GITHUB_WRITE_PR=N
    FAIRY_GITHUB_ACCOUNT=<gcli account name> \\
    python3 -m unittest tests.test_github_write_live -v

Set FAIRY_GITHUB_APP_ID and FAIRY_GITHUB_APP_KEY instead of configuring a
gcli account to run the same checks as a GitHub App.

Point it only at a scratch repo. Each test writes a marker carrying the
PID so concurrent runs cannot read each other's, and removes what it
added where GitHub allows it (comments are left in place; a scratch PR
is expected to accumulate them).

Approving is covered separately, because GitHub refuses a review on your
own pull request: FAIRY_GITHUB_APPROVE_PR must name a PR opened by
somebody other than the token's owner, or the approval test stays
skipped.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402

REPO = os.environ.get("FAIRY_GITHUB_WRITE_REPO")
PR = os.environ.get("FAIRY_GITHUB_WRITE_PR")
ISSUE = os.environ.get("FAIRY_GITHUB_WRITE_ISSUE")
ACCOUNT = os.environ.get("FAIRY_GITHUB_ACCOUNT", "")
APPROVE_PR = os.environ.get("FAIRY_GITHUB_APPROVE_PR")

MARKER = f"fairy github write-path selftest pid={os.getpid()}"
LABEL = os.environ.get("FAIRY_GITHUB_LABEL", "bug")

APP_ID = os.environ.get("FAIRY_GITHUB_APP_ID")
APP_KEY = os.environ.get("FAIRY_GITHUB_APP_KEY")

ARGS = SimpleNamespace(forge_type="github", gcli_account=ACCOUNT, verbose=1,
                       approve_message="", self_login=None,
                       github_app_id=APP_ID,
                       github_app_key=Path(APP_KEY) if APP_KEY else None,
                       github_app_installation=os.environ.get(
                           "FAIRY_GITHUB_APP_INSTALLATION"))


def _owner_repo() -> tuple[str, str]:
    owner, repo = REPO.split("/", 1)
    return owner, repo


@unittest.skipUnless(REPO and (PR or ISSUE),
                     "set FAIRY_GITHUB_WRITE_REPO and _PR or _ISSUE to run")
class GitHubWritePathTests(unittest.TestCase):
    """An issue and a PR take the same comment path and differ only in
    the gcli subcommand labels go through, so either exercises both."""

    KIND = forge_gcli.KIND_PR if PR else forge_gcli.KIND_ISSUE

    def test_a_comment_round_trips(self) -> None:
        owner, repo = _owner_repo()
        number = int(PR or ISSUE)
        forge_gcli.post_issue_comment(ARGS, owner, repo, number, MARKER,
                                      kind=self.KIND)
        bodies = [c.get("body") for c in
                  forge_gcli.list_issue_comments(ARGS, owner, repo, number,
                                                 kind=self.KIND)]
        self.assertIn(MARKER, bodies)

    def test_a_label_attaches_and_detaches(self) -> None:
        owner, repo = _owner_repo()
        number = int(PR or ISSUE)

        def names() -> set[str]:
            pr = forge_gcli.gcli_api(ARGS, forge_gcli.build_repo_path(
                owner, repo, f"/issues/{number}"))
            return {lbl["name"] for lbl in pr.get("labels") or []}

        forge_gcli.apply_issue_label_changes(
            ARGS, owner, repo, number, [LABEL], [], names(), kind=self.KIND)
        attached = names()
        # The Gitea bug returned 200 and attached nothing, so assert on
        # the forge's own view of the PR rather than on the exit code.
        self.assertIn(LABEL, attached)

        forge_gcli.apply_issue_label_changes(
            ARGS, owner, repo, number, [], [LABEL], attached, kind=self.KIND)
        self.assertNotIn(LABEL, names())


@unittest.skipUnless(REPO and APPROVE_PR,
                     "set FAIRY_GITHUB_APPROVE_PR to a PR the token's owner "
                     "did not open; GitHub refuses self-approval")
class GitHubApproveTests(unittest.TestCase):

    def test_an_approval_is_recorded(self) -> None:
        import fairy
        owner, repo = _owner_repo()
        number = int(APPROVE_PR)
        args = SimpleNamespace(**vars(ARGS), owner=owner, repo=repo)
        fairy.gcli_approve(args, number, MARKER)
        states = [r.get("state") for r in
                  forge_gcli.list_pr_reviews(args, owner, repo, number)]
        self.assertIn("APPROVED", states)


if __name__ == "__main__":
    unittest.main()
