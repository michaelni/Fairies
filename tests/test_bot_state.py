"""Tests for ``bot_state``.

Coverage: schema constant pinned; load handles missing / corrupt /
wrong-shape / wrong-version inputs by returning an empty State; save
+ load round-trips entry contents and keys.
"""
from __future__ import annotations

import pickle
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import bot_state  # noqa: E402
from bot_state import Entry, Key, State, load, save  # noqa: E402


class SchemaTests(unittest.TestCase):

    def test_schema_version_pinned(self) -> None:
        # Bumping is deliberate; the bookkeeping is informational so we
        # cold-warm rather than migrate, but the constant still pins the
        # on-disk shape that ``load`` will accept.
        self.assertEqual(bot_state.SCHEMA_VERSION, 1)


class PersistenceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = REPO_ROOT / "tests" / "_tmp_bot_state.pkl"
        if self.tmp.exists():
            self.tmp.unlink()

    def tearDown(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()

    def test_missing_file_yields_empty_state(self) -> None:
        self.assertEqual(load(self.tmp), State())

    def test_corrupt_pickle_yields_empty_state(self) -> None:
        self.tmp.write_bytes(b"\xff\xff not a pickle \xff\xff")
        self.assertEqual(load(self.tmp), State())

    def test_wrong_shape_yields_empty_state(self) -> None:
        with self.tmp.open("wb") as f:
            pickle.dump({"some": "dict"}, f)
        self.assertEqual(load(self.tmp), State())

    def test_wrong_version_yields_empty_state(self) -> None:
        save(self.tmp, State(version=bot_state.SCHEMA_VERSION + 1))
        self.assertEqual(load(self.tmp), State())

    def test_roundtrip_preserves_entry_and_key(self) -> None:
        state = State()
        entry: Entry = {
            "last_llm_decision": "skip",
            "last_llm_at": "2026-05-20T00:00:00+00:00",
            "last_llm_head_sha": "abc123",
            "last_llm_last_activity_iso": "2026-05-19T22:00:00+00:00",
            "consecutive_skip_count": 3,
        }
        state.entries[Key("ffmpeg", "FFmpeg", 42)] = entry

        save(self.tmp, state)
        loaded = load(self.tmp)

        self.assertEqual(loaded.entries[Key("ffmpeg", "FFmpeg", 42)], entry)


if __name__ == "__main__":
    unittest.main()
