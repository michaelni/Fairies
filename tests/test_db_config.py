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

db_config: the configurator-written config.toml round-trips."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import db_config  # noqa: E402


class DbConfigTests(unittest.TestCase):
    def test_round_trip_one_key_per_option(self) -> None:
        pr = {"owner": "o", "repo": "r", "auto-mode": True,
              "triage-label": ["important", "fix/bug"],
              "llm-review-cmd": './w.py "quoted" \\ x\n    --model m\n'}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_config.write_config(root, "o/r", {Path("logs/x.log")}, pr, None)
            text = (root / "config.toml").read_text(encoding="utf-8")
            cfg = db_config.read_config(root)
            pr_argv, issue_argv = db_config.read_side_argv(root)
        self.assertIn("[pr]", text)
        self.assertIn("llm-review-cmd = '''\n"
                      './w.py "quoted" \\ x\n    --model m\n'
                      "'''", text)
        self.assertEqual(cfg["label"], "o/r")
        self.assertEqual(cfg["log_files"], [str(Path("logs/x.log").resolve())])
        self.assertIsNone(issue_argv)
        self.assertEqual(pr_argv, [
            "--owner=o", "--repo=r", "--auto-mode",
            "--triage-label=important", "--triage-label=fix/bug",
            '--llm-review-cmd=./w.py "quoted" \\ x\n    --model m\n'])

    def test_non_bmp_and_leading_dash_values_round_trip(self) -> None:
        """ASCII-escaping JSON would spell the emoji as a surrogate
        pair, which tomllib rejects; argparse takes a leading-dash
        value only as one --key=value token."""
        pr = {"owner": "o", "triage-label": ["bug \U0001f41b"],
              "approve-message": "-x"}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_config.write_config(root, "o/r \U0001f9da", set(), pr, None)
            cfg = db_config.read_config(root)
            pr_argv, _ = db_config.read_side_argv(root)
        self.assertEqual(cfg["label"], "o/r \U0001f9da")
        self.assertEqual(pr_argv, ["--owner=o",
                                   "--triage-label=bug \U0001f41b",
                                   "--approve-message=-x"])

    def test_a_false_flag_reads_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db_config.write_config(root, "o/r", set(),
                                   {"owner": "o", "auto-mode": True}, None)
            path = root / "config.toml"
            path.write_text(path.read_text(encoding="utf-8")
                            .replace("auto-mode = true", "auto-mode = false"),
                            encoding="utf-8")
            pr_argv, _ = db_config.read_side_argv(root)
        self.assertEqual(pr_argv, ["--owner=o"])


if __name__ == "__main__":
    unittest.main()
