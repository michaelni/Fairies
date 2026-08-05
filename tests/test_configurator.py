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

configurator: side-string validation and the config.toml it writes."""

from __future__ import annotations

import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import configurator  # noqa: E402
import db_config  # noqa: E402
import fairy  # noqa: E402


class ValidationTests(unittest.TestCase):
    """Broken side configs exit rc=2."""

    def pr(self, extra: str) -> object:
        return fairy.parse_args(shlex.split("--owner o --repo r " + extra))

    def test_llm_review_cmd_requires_patch_repo(self) -> None:
        ns = self.pr("--llm-review-cmd wrapper")
        with self.assertRaises(SystemExit) as ctx:
            configurator.validate_sides(ns, None)
        self.assertEqual(ctx.exception.code, 2)  # rc 2, not 1: cron distinguishes config from crash

    def test_patch_repo_satisfies_the_check(self) -> None:
        configurator.validate_sides(
            self.pr("--llm-review-cmd wrapper --patch-repo p"), None)

    def test_simulate_past_template_must_contain_number(self) -> None:
        base = "--simulate-past 2026-07-01T00:00:00+00:00 "
        with self.assertRaises(SystemExit):
            configurator.validate_sides(self.pr(base), None)
        with self.assertRaises(SystemExit):
            configurator.validate_sides(
                self.pr(base + "--patch-pr-ref-template fforge/pr/"), None)
        configurator.validate_sides(
            self.pr(base + "--patch-pr-ref-template fforge/pr/{number}"), None)

    def test_forced_only_requires_a_force_review(self) -> None:
        with self.assertRaises(SystemExit):
            configurator.validate_sides(self.pr("--forced-only"), None)
        configurator.validate_sides(
            self.pr("--forced-only --force-review-pr 5"), None)


class SideOptionsTests(unittest.TestCase):
    def test_flags_values_and_repeats(self) -> None:
        opts = configurator.side_options(
            fairy.make_parser(),
            "--owner o --repo r --approve --triage-label a --triage-label b "
            "--min-age-days=3")
        self.assertEqual(opts, {"owner": "o", "repo": "r", "approve": True,
                                "triage-label": ["a", "b"],
                                "min-age-days": "3"})

    def test_an_abbreviated_option_is_rejected(self) -> None:
        with self.assertRaisesRegex(SystemExit, "abbreviated"):
            configurator.side_options(fairy.make_parser(), "--owne o --repo r")


class MainTests(unittest.TestCase):
    def test_pr_side_config_file_option_is_stored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            side = Path(tmp) / "side.toml"
            side.write_text('min-age-days = 3\n', encoding="utf-8")
            argv = ["configurator.py", "--db-root", tmp, "--pr-args",
                    f"--owner o --repo r --config {side}"]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(configurator, "setup_logging"):
                configurator.main()
            cfg = db_config.read_config(Path(tmp))
            pr_argv, _ = db_config.read_side_argv(Path(tmp))
            self.assertEqual(cfg["pr"]["config"], str(side))
            self.assertEqual(fairy.parse_args(pr_argv).min_age_days, 3)

    def test_main_round_trips_the_side_namespace(self) -> None:
        pr = ("--owner o --repo r --log-file logs/x.log --approve "
              "--patch-repo p --triage-label a --triage-label b "
              "--llm-review-cmd './w.py\n    --model m'")
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["configurator.py", "--db-root", tmp, "--pr-args", pr]
            with mock.patch.object(sys, "argv", argv), \
                    mock.patch.object(configurator, "setup_logging"):
                rc = configurator.main()
            cfg = db_config.read_config(Path(tmp))
            pr_argv, issue_argv = db_config.read_side_argv(Path(tmp))
        self.assertEqual(rc, 0)
        self.assertEqual(cfg["label"], "o/r")
        self.assertEqual(cfg["log_files"], [str(Path("logs/x.log").resolve())])
        self.assertIsNone(issue_argv)
        self.assertEqual(fairy.parse_args(pr_argv),
                         fairy.parse_args(shlex.split(pr)))

    def test_db_root_defaults_to_the_side_identity(self) -> None:
        ns = fairy.parse_args(["--owner", "O", "--repo", "R",
                               "--gcli-account", "a"])
        self.assertEqual(configurator.db_root_for(ns),
                         Path.home() / ".fairy" / "db" / "gitea~a~O~R")


if __name__ == "__main__":
    unittest.main()
