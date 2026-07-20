"""concurrency: slot limits hold across real processes."""

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
