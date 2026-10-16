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
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import author_settle  # noqa: E402
import filedb  # noqa: E402

FAIRY = "Forgejo_Fairy"


def pr(author: str, state: str, reviews: int, patch_ids: list[str]) -> dict:
    return {"author": author, "state": state, "patch_ids": patch_ids,
            "discussion": [{"kind": "review", "author": FAIRY}] * reviews
            + [{"kind": "push", "author": author},
               {"kind": "comment", "author": "dave"}]}


class SettleHoursTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name))

    def test_ratio_steps_and_hand_applied_landing(self) -> None:
        # alice: 24 reviews over 3 PRs; one merged on the forge, one
        # closed but every patch on master, one open: 24/(2+1) = 8.
        self.db.push("items", "pr", "1", pr("alice", "merged", 10, ["p1"]))
        self.db.push("items", "pr", "2", pr("alice", "closed", 8, ["p2", "p3"]))
        self.db.push("items", "pr", "3", pr("alice", "open", 6, ["p4"]))
        # bob: 12 reviews, one PR closed with only part of it on master.
        self.db.push("items", "pr", "4", pr("bob", "closed", 12, ["p5", "p6"]))
        self.db.push("items", "pr", "5", pr("carol", "open", 7, ["p7"]))
        self.db.push("items", "issue", "6", {"author": "bob", "discussion": []})
        hours = author_settle.settle_hours(self.db, FAIRY, {"p2", "p3", "p5"})
        self.assertEqual(hours, {"alice": 12.0, "bob": 48.0})
        # Without p2/p3 on master alice has one landed PR: 24/2 = 12;
        # bob's PR landed whole: 12/2 = 6, below every step.
        self.assertEqual(author_settle.settle_hours(self.db, FAIRY, {"p5", "p6"}),
                         {"alice": 48.0})


if __name__ == "__main__":
    unittest.main()
