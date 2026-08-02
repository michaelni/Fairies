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

Opt-in read-only check that the GitHub adapters still fit live GitHub.

The recorded fixtures pin what GitHub returned on the day they were
captured. By construction they cannot notice GitHub changing its API,
which is the failure this test exists for. It performs GETs against a
public repository -- no token, no writes -- and skips unless pointed at
one:

    FAIRY_GITHUB_LIVE_REPO=michaelni/testrepo FAIRY_GITHUB_LIVE_PR=4 \\
    python3 -m unittest tests.test_github_live -v

Unauthenticated GitHub allows 60 requests an hour; this test uses four.
``tools/capture_github_fixtures.py --check`` is the companion that
compares the stored fixtures against live shapes.
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

REPO = os.environ.get("FAIRY_GITHUB_LIVE_REPO")
PR = os.environ.get("FAIRY_GITHUB_LIVE_PR")

ARGS = SimpleNamespace(forge_type="github", gcli_account="", verbose=0)


@unittest.skipUnless(REPO and PR, "set FAIRY_GITHUB_LIVE_REPO and _PR to run")
class GitHubLiveReadTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.owner, cls.repo = REPO.split("/", 1)
        cls.number = int(PR)
        cls.pr = forge_gcli.gcli_api(
            ARGS, forge_gcli.build_repo_path(
                cls.owner, cls.repo, f"/pulls/{cls.number}"))

    def test_commit_statuses_carry_the_contract_keys(self) -> None:
        rows = forge_gcli.list_commit_statuses(
            ARGS, self.owner, self.repo, self.pr["head"]["sha"])
        self.assertTrue(rows, "no CI rows; pick a PR whose checks have run")
        for row in rows:
            self.assertEqual(
                sorted(row),
                ["context", "created_at", "description", "state",
                 "target_url", "updated_at"],
            )

    def test_timeline_events_are_dated_and_typed(self) -> None:
        events = forge_gcli.list_issue_timeline(
            ARGS, self.owner, self.repo, self.number)
        self.assertTrue(events)
        for event in events:
            self.assertLessEqual(
                {"type", "id", "user", "created_at", "body"}, set(event))
            self.assertIsNotNone(event["created_at"], event["type"])

    def test_a_push_is_recovered_from_the_commit_entries(self) -> None:
        events = forge_gcli.list_issue_timeline(
            ARGS, self.owner, self.repo, self.number)
        pushes = [e for e in events if e["type"] == forge_gcli.PUSH_EVENT]
        self.assertTrue(pushes, "PR has no commits in its timeline")
        self.assertTrue(all(p["commit_ids"] for p in pushes))


if __name__ == "__main__":
    unittest.main()
