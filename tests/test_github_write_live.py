"""Opt-in check that fairy's write paths reach GitHub.

Every write fairy performs -- a comment, a label change, an approving
review -- goes through a gcli subcommand that gcli implements per
backend. That is exactly where the Gitea label bug lived: ``labels add``
resolved a name to an id and sent it as a JSON string, so the forge
returned 200 and attached nothing. A fixture cannot catch that class of
bug, because the request is well-formed and the failure is on the far
side. Only a real round-trip can.

Writes to whatever repo it is pointed at, so it skips unless told where:

    FAIRY_GITHUB_WRITE_REPO=michaelni/testrepo \\
    FAIRY_GITHUB_WRITE_PR=1 \\
    FAIRY_GITHUB_ACCOUNT=<gcli account name> \\
    python3 -m unittest tests.test_github_write_live -v

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
ACCOUNT = os.environ.get("FAIRY_GITHUB_ACCOUNT", "")
APPROVE_PR = os.environ.get("FAIRY_GITHUB_APPROVE_PR")

MARKER = f"fairy github write-path selftest pid={os.getpid()}"
LABEL = "fairy-selftest"

ARGS = SimpleNamespace(forge_type="github", gcli_account=ACCOUNT, verbose=1,
                       approve_message="")


def _owner_repo() -> tuple[str, str]:
    owner, repo = REPO.split("/", 1)
    return owner, repo


@unittest.skipUnless(REPO and PR,
                     "set FAIRY_GITHUB_WRITE_REPO and _PR to run")
class GitHubWritePathTests(unittest.TestCase):

    def test_a_comment_round_trips(self) -> None:
        owner, repo = _owner_repo()
        number = int(PR)
        forge_gcli.post_issue_comment(ARGS, owner, repo, number, MARKER)
        bodies = [c.get("body") for c in
                  forge_gcli.list_issue_comments(ARGS, owner, repo, number)]
        self.assertIn(MARKER, bodies)

    def test_a_label_attaches_and_detaches(self) -> None:
        owner, repo = _owner_repo()
        number = int(PR)

        def names() -> set[str]:
            pr = forge_gcli.gcli_api(ARGS, forge_gcli.build_repo_path(
                owner, repo, f"/issues/{number}"))
            return {lbl["name"] for lbl in pr.get("labels") or []}

        forge_gcli.apply_issue_label_changes(
            ARGS, owner, repo, number, [LABEL], [], names())
        attached = names()
        # The Gitea bug returned 200 and attached nothing, so assert on
        # the forge's own view of the PR rather than on the exit code.
        self.assertIn(LABEL, attached)

        forge_gcli.apply_issue_label_changes(
            ARGS, owner, repo, number, [], [LABEL], attached)
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
