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

concurrency: slot limits hold across real processes."""

import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import concurrency

_CHILD = """
import json, sys, time
sys.path.insert(0, {repo!r})
from pathlib import Path
import common
common.default_cache_path = lambda name: Path({lock_root!r}) / name
import concurrency
concurrency.configure([("prov", {limit})])
with concurrency.slot("prov"):
    start = time.monotonic()
    time.sleep(0.3)
    print(json.dumps([start, time.monotonic()]))
"""


def _max_overlap(intervals: list[tuple[float, float]]) -> int:
    events = [(s, 1) for s, _ in intervals] + [(e, -1) for _, e in intervals]
    events.sort()
    held = peak = 0
    for _, delta in events:
        held += delta
        peak = max(peak, held)
    return peak


class ParseLimitTest(unittest.TestCase):
    def test_valid(self):
        self.assertEqual(concurrency.parse_limit("codex:2"), ("codex", 2))

    def test_rejects_malformed(self):
        for bad in ("codex", "codex:", ":2", "codex:0", "codex:x"):
            with self.assertRaises(argparse.ArgumentTypeError):
                concurrency.parse_limit(bad)

    def test_rejects_provider_escaping_the_lock_dir(self):
        for bad in ("../../etc/x:1", "a/b:1", ".:1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                concurrency.parse_limit(bad)


class SlotTest(unittest.TestCase):
    def test_uncapped_provider_passes_through(self):
        concurrency.configure([])
        with concurrency.slot("codex"):
            pass

    def test_cap_holds_across_processes(self):
        repo = str(Path(__file__).resolve().parent.parent)
        limit = 2
        with tempfile.TemporaryDirectory() as lock_root:
            script = _CHILD.format(repo=repo, lock_root=lock_root, limit=limit)
            children = [
                subprocess.Popen(
                    [sys.executable, "-c", script],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                )
                for _ in range(6)
            ]
            intervals = []
            for child in children:
                out, err = child.communicate(timeout=60)
                self.assertEqual(child.returncode, 0, err)
                intervals.append(tuple(json.loads(out.strip().splitlines()[-1])))

        self.assertLessEqual(_max_overlap(intervals), limit)
        # must not pass on mere serialization
        self.assertGreater(_max_overlap(intervals), 1)


if __name__ == "__main__":
    unittest.main()
