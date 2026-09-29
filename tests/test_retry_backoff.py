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

retry_backoff: the wait doubles per failure, is capped at a week and a
success forgets the failures."""

from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import retry_backoff  # noqa: E402

_ERROR = "podman exec: Error: container fairy-pr-23750 is not running"


class RetryBackoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.enterContext(mock.patch.object(
            retry_backoff, "default_cache_path",
            lambda name: Path(self.tmp.name) / name))

    def fail_times(self, key: str, times: int, seconds_ago: float) -> None:
        for _ in range(times):
            retry_backoff.record(key, _ERROR)
        failures_file, = (Path(self.tmp.name) / "retry_backoff").iterdir()
        os.utime(failures_file, times=(time.time() - seconds_ago,) * 2)

    def wait_for(self, key: str, base: float) -> float:
        with self.assertRaises(retry_backoff.RetryTooSoon) as caught:
            retry_backoff.ensure_due(key, base)
        return caught.exception.wait_seconds

    def test_unknown_key_and_truncated_record_are_due(self) -> None:
        retry_backoff.ensure_due("pr/1", 60)
        retry_backoff.record("pr/1", _ERROR)
        failures_file, = (Path(self.tmp.name) / "retry_backoff").iterdir()
        failures_file.write_text("")
        retry_backoff.ensure_due("pr/1", 60)

    def test_wait_doubles_per_failure(self) -> None:
        self.fail_times("pr/1", 1, seconds_ago=10)
        self.assertAlmostEqual(self.wait_for("pr/1", 60), 50, delta=2)
        self.fail_times("pr/1", 2, seconds_ago=10)
        self.assertAlmostEqual(self.wait_for("pr/1", 60), 230, delta=2)
        self.fail_times("pr/1", 0, seconds_ago=500)
        retry_backoff.ensure_due("pr/1", 60)

    def test_wait_is_capped_at_a_week(self) -> None:
        self.fail_times("pr/1", 30, seconds_ago=7 * 24 * 3600 - 100)
        self.assertAlmostEqual(self.wait_for("pr/1", 60), 100, delta=2)
        self.fail_times("pr/1", 0, seconds_ago=7 * 24 * 3600 + 1)
        retry_backoff.ensure_due("pr/1", 60)

    def test_the_servers_retry_after_outlasts_a_short_backoff(self) -> None:
        retry_backoff.record("pr/1", _ERROR, retry_after=time.time() + 3600)
        self.assertAlmostEqual(self.wait_for("pr/1", 60), 3600, delta=2)
        retry_backoff.record("pr/1", _ERROR)
        self.assertAlmostEqual(self.wait_for("pr/1", 60), 120, delta=2)

    def test_a_success_forgets_the_failures(self) -> None:
        self.fail_times("pr/1", 3, seconds_ago=0)
        retry_backoff.record("pr/1", "")
        retry_backoff.ensure_due("pr/1", 60)
        retry_backoff.record("pr/1", "")

    def test_keys_are_independent_even_when_they_sanitize_alike(self) -> None:
        retry_backoff.record("pr/1", _ERROR)
        retry_backoff.record("pr-1", _ERROR)
        retry_backoff.record("pr/2", "")
        self.assertEqual(
            sorted(p.name.split(".")[0]
                   for p in (Path(self.tmp.name) / "retry_backoff").iterdir()),
            ["pr-1", "pr-1"])
        self.wait_for("pr/1", 60)
        retry_backoff.ensure_due("pr/2", 60)

    def test_each_failure_is_one_line_of_the_record(self) -> None:
        retry_backoff.record("pr/1", "first\nline two")
        retry_backoff.record("pr/1", "second")
        failures_file, = (Path(self.tmp.name) / "retry_backoff").iterdir()
        lines = failures_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual([l.split(" ", 2)[::2] for l in lines],
                         [["0", "first line two"], ["0", "second"]])


if __name__ == "__main__":
    unittest.main()
