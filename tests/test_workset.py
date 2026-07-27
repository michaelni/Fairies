"""workset: path layout and the sidecar-locked dict read-modify-write."""

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
