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

workset: path layout and the sidecar-locked dict read-modify-write."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import workset  # noqa: E402


class PathTests(unittest.TestCase):
    def test_layout_and_sanitization(self) -> None:
        d = workset.repo_dir(Path("/root"), forge_type="gitea", account="",
                             owner="own/er", repo="re~po")
        self.assertEqual(d, Path("/root/gitea~default~own_er~re_po"))


class UpdateJsonTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "pr-5.json"

    def test_mutation_lands_atomically_with_a_sidecar_lock(self) -> None:
        self.path.write_text('{"stage": "triage"}')
        saved = workset.update_json(self.path,
                                    lambda d: d.__setitem__("cancel", True))
        self.assertEqual(saved, {"stage": "triage", "cancel": True})
        self.assertEqual(json.loads(self.path.read_text()), saved)
        self.assertTrue(self.path.with_suffix(".lock").exists())

    def test_missing_file_is_none_and_mutation_not_applied(self) -> None:
        called = []
        self.assertIsNone(workset.update_json(self.path, called.append))
        self.assertEqual(called, [])
        self.assertFalse(self.path.exists())

    def test_unparseable_file_is_none_and_left_alone(self) -> None:
        self.path.write_text("{ torn")
        self.assertIsNone(workset.update_json(self.path, lambda d: d.clear()))
        self.assertEqual(self.path.read_text(), "{ torn")


if __name__ == "__main__":
    unittest.main()
