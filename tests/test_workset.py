"""Tests for ``workset``.

Coverage: item file roundtrip; loud rejection of invalid / hand-edit-broken
/ wrong-version files vs silent None for missing ones; path layout and
segment sanitization; flock'd ``update_item`` applying a state transition
without losing an operator's concurrent edit.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import workset  # noqa: E402
from workset import (  # noqa: E402
    ReviewResult,
    WorkItem,
    WorkState,
    item_path,
    load_item,
    save_item,
    update_item,
)

NOW = datetime(2026, 7, 20, 12, 0, tzinfo=timezone.utc)


def make_item(**overrides: object) -> WorkItem:
    fields: dict = dict(
        kind="pr",
        forge_type="gitea",
        account="",
        owner="ffmpeg",
        repo="FFmpeg",
        number=42,
        state=WorkState.REVIEWED,
        created_at=NOW.isoformat(),
        state_changed_at=NOW.isoformat(),
        title="lavc: fix things",
        html_url="https://example.org/ffmpeg/FFmpeg/pulls/42",
        expected_updated_at="2026-07-19T10:00:00Z",
        expected_head_ref="abc123",
        review=ReviewResult(classification="needs-changes", message="body"),
    )
    fields.update(overrides)
    return WorkItem(**fields)


class PathTests(unittest.TestCase):

    def test_layout_and_sanitization(self) -> None:
        p = item_path(
            Path("/root"), forge_type="gitea", account="", owner="own/er",
            repo="re~po", kind="issue", number=7,
        )
        self.assertEqual(p, Path("/root/gitea~default~own_er~re_po/issue-7.json"))


class PersistenceTests(unittest.TestCase):

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "pr-42.json"

    def test_roundtrip(self) -> None:
        item = make_item()
        save_item(self.path, item)
        self.assertEqual(load_item(self.path), item)

    def test_missing_file_is_silent_none(self) -> None:
        with self.assertNoLogs(workset.logger, level="ERROR"):
            self.assertIsNone(load_item(self.path))

    def test_invalid_json_is_loud_none(self) -> None:
        self.path.write_text("{ not json", encoding="utf-8")
        with self.assertLogs(workset.logger, level="ERROR"):
            self.assertIsNone(load_item(self.path))

    def test_unknown_field_is_loud_none(self) -> None:
        save_item(self.path, make_item())
        text = self.path.read_text(encoding="utf-8")
        self.path.write_text(text.replace('"title"', '"titel"'), encoding="utf-8")
        with self.assertLogs(workset.logger, level="ERROR"):
            self.assertIsNone(load_item(self.path))

    def test_wrong_schema_version_is_loud_none(self) -> None:
        save_item(self.path, make_item(schema_version=workset.SCHEMA_VERSION + 1))
        with self.assertLogs(workset.logger, level="ERROR"):
            self.assertIsNone(load_item(self.path))

    def test_update_item_missing_file_is_none(self) -> None:
        called = []
        self.assertIsNone(update_item(self.path, called.append))
        self.assertEqual(called, [])

    def test_update_item_keeps_operator_edit(self) -> None:
        save_item(self.path, make_item())
        edited = load_item(self.path)
        assert edited is not None and edited.review is not None
        edited.review.message = "operator-corrected body"
        save_item(self.path, edited)

        updated = update_item(
            self.path, lambda it: it.set_state(WorkState.POSTED, NOW)
        )
        assert updated is not None and updated.review is not None
        self.assertEqual(updated.state, WorkState.POSTED)
        self.assertEqual(updated.review.message, "operator-corrected body")
        self.assertEqual(load_item(self.path), updated)


class PruneTests(unittest.TestCase):

    OLD = "2026-06-01T00:00:00+00:00"
    CUTOFF = datetime(2026, 7, 1, tzinfo=timezone.utc)

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)

    def _write(self, number: int, state: WorkState, changed: str) -> Path:
        path = self.dir / f"pr-{number}.json"
        save_item(path, make_item(number=number, state=state,
                                  state_changed_at=changed))
        return path

    def test_old_finished_closed_item_is_deleted(self) -> None:
        path = self._write(1, WorkState.POSTED, self.OLD)
        path.with_suffix(".lock").touch()
        workset.prune(self.dir, "pr", set(), self.CUTOFF)
        self.assertFalse(path.exists())
        self.assertFalse(path.with_suffix(".lock").exists())

    def test_open_item_is_kept(self) -> None:
        path = self._write(1, WorkState.POSTED, self.OLD)
        workset.prune(self.dir, "pr", {1}, self.CUTOFF)
        self.assertTrue(path.exists())

    def test_reviewed_item_is_never_pruned(self) -> None:
        path = self._write(1, WorkState.REVIEWED, self.OLD)
        workset.prune(self.dir, "pr", set(), self.CUTOFF)
        self.assertTrue(path.exists())

    def test_recent_finished_item_is_kept(self) -> None:
        path = self._write(1, WorkState.SKIPPED, self.CUTOFF.isoformat())
        workset.prune(self.dir, "pr", set(), self.CUTOFF)
        self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
