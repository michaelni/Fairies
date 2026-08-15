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

fairy_tui: headless model round-trips over a filedb and a paint smoke."""

from __future__ import annotations

import io
import logging
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blessed  # noqa: E402
import fairy_tui  # noqa: E402
import filedb  # noqa: E402
import tui_core  # noqa: E402

R1, R2 = "o/r", "o/r2"


class Key(str):
    """Keystroke stand-in: dispatch() reads the str value and .name."""
    name = None


def NamedKey(name: str) -> Key:
    k = Key("")
    k.name = name
    return k


def make_term(stream: io.StringIO | None = None, cols: int = 100) -> blessed.Terminal:
    """Headless terminal fixture. blessed re-reads COLUMNS/LINES on
    every width query, so they are set for good rather than patched."""
    os.environ.update({"COLUMNS": str(cols), "LINES": "40"})
    return blessed.Terminal(kind="xterm-256color", stream=stream or io.StringIO(),
                            force_styling=True)


def verdict(n: int, classification: str = "moderate_issues", msg: str = "m",
            labels: list | None = None, **fields) -> dict:
    t = {"title": "from disk", "author": "a", "html_url": "u",
         "review": {"classification": classification, "message": msg,
                    "label_changes": labels or []},
         "expected_updated_at": "2026-07-19T10:00:00Z",
         "expected_head_ref": f"h{n}", "llm_at": "2026-07-20T00:00:00+00:00"}
    t.update(fields)
    return t


def make_ui(model: fairy_tui.Model, stream: io.StringIO | None = None,
            cols: int = 100, save_dir: Path = Path(".")) -> fairy_tui.UILoop:
    ring = tui_core.RingBuffer()
    sink = fairy_tui.OutputSink(ring, model.dirty, None)
    return fairy_tui.UILoop(make_term(stream, cols), model, ring, save_dir,
                            fairy_tui.LogTail([], sink))


class DbCase(unittest.TestCase):
    """Base: a Model over two temp filedbs, plus ticket writers."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db = filedb.Db(Path(tmp.name) / "r")
        self.db2 = filedb.Db(Path(tmp.name) / "r2")
        self.model = fairy_tui.Model([(R1, self.db), (R2, self.db2)])

    def keys(self) -> list:
        with self.model.lock:
            return [(it.repo, it.number) for it in self.model.visible()]


class PollTests(DbCase):
    """poll() drives rows purely from the state directories."""

    def test_rows_are_files_and_state_is_the_directory(self) -> None:
        self.db.push("queued", "pr", "5", dict(verdict(5), prepared={"pr": {}}))
        self.model.poll()
        item = self.model.items[(R1, "pr", "5")]
        self.assertEqual(item.state, "queued")
        self.assertEqual(item.data["title"], "from disk")
        self.assertNotIn("prepared", item.data)  # multi-MB, never shown
        self.db.try_move("queued", "llm", "pr", "5",
                     mutate=lambda d: d.update(stage="triage"))
        self.model.poll()
        self.assertEqual(item.state, "llm")
        self.assertEqual(item.data["stage"], "triage")

    def test_invalid_file_flags_the_row_and_recovers(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        path = self.db.path("reviewed", "pr", "5")
        path.write_text("{ broken", encoding="utf-8")
        os.utime(path, (200.0, 200.0))
        self.model.poll()
        item = self.model.items[(R1, "pr", "5")]
        self.assertEqual(item.state, fairy_tui.INVALID)
        self.assertTrue(item.error)
        self.assertEqual(item.data["review"]["message"], "m")  # last good parse
        self.assertIn((R1, "5"), self.keys())
        self.db.push("reviewed", "pr", "5", verdict(5))  # operator fixed it
        self.model.poll()
        self.assertEqual(item.state, "reviewed")
        self.assertEqual(item.error, "")

    def test_removed_file_drops_the_row_after_the_grace_polls(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.db.try_pop("reviewed", "pr", "5")
        for _ in range(fairy_tui.GONE_POLLS - 1):
            self.model.poll()
        self.assertIn((R1, "pr", "5"), self.model.items)
        with self.assertLogs("fairy_tui", level="INFO"):
            self.model.poll()
        self.assertNotIn((R1, "pr", "5"), self.model.items)

    def test_a_reappearing_ticket_resets_the_miss_counter(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.db.try_pop("reviewed", "pr", "5")
        for _ in range(fairy_tui.GONE_POLLS - 1):
            self.model.poll()
        self.db.push("llm", "pr", "5", verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "llm")
        self.assertNotIn((R1, "pr", "5"), self.model.missing)

    def test_a_merged_cancellation_shows_merged_not_cancelled(self) -> None:
        self.db.push("cancelled", "pr", "5",
                     dict(verdict(5), reason="merged"))
        self.model.poll()
        with self.model.lock:
            self.model.filter_mode = "all"
        ui = make_ui(self.model)
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        self.assertTrue(any("merged" in t and "cancelled" not in t
                            for t in rows), rows)

    def test_a_missing_ticket_is_marked_in_the_list(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.db.try_pop("reviewed", "pr", "5")
        self.model.poll()
        ui = make_ui(self.model)
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        self.assertTrue(any("reviewed?" in t for t in rows), rows)

    def test_crash_remnant_shows_the_later_state(self) -> None:
        self.db.push("queued", "pr", "5", verdict(5))
        self.db.push("posted", "pr", "5", verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "posted")

    def test_relevant_filter_hides_only_unseen_settled_rows(self) -> None:
        self.db.push("skipped", "pr", "1", verdict(1, "skip"))   # old backlog
        self.db.push("posted", "pr", "2", verdict(2))            # old backlog
        self.db.push("ci-blocked", "pr", "3", verdict(3))        # attention
        self.db.push("reviewed", "pr", "4", verdict(4))
        self.db.push("error", "pr", "6", {"error": "boom"})
        self.model.poll()
        self.assertEqual(self.keys(), [(R1, "3"), (R1, "4"), (R1, "6")])
        with self.model.lock:
            self.model.filter_mode = "all"
        self.assertEqual(self.keys(), [(R1, "1"), (R1, "2"), (R1, "3"), (R1, "4"), (R1, "6")])

    def test_a_row_seen_live_stays_listed_after_it_settles(self) -> None:
        # The operator watched this item head into the LLM and wants to
        # inspect why it skipped; without the session memory the row
        # would vanish the moment the verdict lands in skipped/.
        self.db.push("llm", "pr", "5", verdict(5))
        self.model.poll()
        self.db.try_move("llm", "skipped", "pr", "5")
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "skipped")
        self.assertIn((R1, "5"), self.keys())


class DirMtimeGateTests(DbCase):
    """Unchanged state dirs are not re-listed: every filedb write lands
    by rename into its dir, so the dir mtime is the change signal."""

    def age_dirs(self, seconds: float = 10) -> None:
        for db in (self.db, self.db2):
            for state in (*filedb.STATES, filedb.ITEM_STATE):
                t = time.time() - seconds
                os.utime(db.root / state, (t, t))

    def listing_calls(self) -> list[str]:
        calls: list[str] = []
        orig = filedb.Db.list_state_stat
        with mock.patch.object(
                filedb.Db, "list_state_stat", autospec=True,
                side_effect=lambda db, s: (calls.append(s), orig(db, s))[1]):
            self.model.poll()
        return calls

    def test_unchanged_aged_dirs_skip_the_rescan(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.age_dirs()
        self.model.poll()  # caches every (old enough) dir listing
        self.assertEqual(self.listing_calls(), [])
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "reviewed")

    def test_renames_are_seen_through_the_gate(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.age_dirs()
        self.model.poll()
        self.db.try_move("reviewed", "outgoing", "pr", "5")  # bumps both dirs
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "outgoing")

    def test_a_recently_modified_dir_is_never_trusted(self) -> None:
        # Coarse file timestamps: a rename in the same clock tick as the
        # scan can leave the dir mtime unchanged, so fresh mtimes must
        # not enter the cache.
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()  # dir mtimes are "now": nothing may be cached
        self.assertIn("reviewed", self.listing_calls())

    def test_a_vanished_root_still_counts_misses(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.age_dirs()
        self.model.poll()
        shutil.rmtree(self.db.root)
        for _ in range(fairy_tui.GONE_POLLS):
            self.model.poll()
        self.assertNotIn((R1, "pr", "5"), self.model.items)

    def test_a_crash_remnant_recovers_when_the_shadow_dies(self) -> None:
        self.db.push("queued", "pr", "5", verdict(5))
        self.db.push("skipped", "pr", "5", verdict(5))
        self.age_dirs()
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "skipped")
        # hand cleanup deletes the shadow; queued/ itself stays quiet
        self.db.path("skipped", "pr", "5").unlink()
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "queued")


class ActTests(DbCase):
    """y/s/x/r/f are file operations on the cursor row's db."""

    def test_apply_moves_reviewed_to_outgoing(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("apply")
        self.assertEqual(self.db.find("pr", "5"), "outgoing")
        self.assertIn((R1, "pr", "5"), self.model.acted)

    def test_force_apply_flags_the_ticket_for_a_guardless_send(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("apply-force")
        self.assertEqual(self.db.find("pr", "5"), "outgoing")
        self.assertTrue(self.db.get("outgoing", "pr", "5")["force_post"])

    def test_apply_stages_a_sample_verdict_like_the_base(self) -> None:
        """y works on a sample row: the send pass posts under the forge
        number either way, and the sample may hold the fresher verdict
        (the base reviewed before the latest discussion)."""
        self.db.push("reviewed", "pr", "5s1", verdict(5))
        self.model.poll()
        self.model.act("apply")
        self.assertEqual(self.db.find("pr", "5s1"), "outgoing")

    def test_R_adds_evaluations_without_clobbering_earlier_ones(self) -> None:
        """Each R takes the next free sample slot: a verdict for 5s1 on
        disk plus a pending 5s2 request mean R asks for 5s3."""
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("skipped", "pr", "5s1", verdict(5, "skip"))
        self.model.poll()
        self.model.act("sample")
        self.assertIsNotNone(self.db.get("requests", "pr", "5s2"))
        self.model.act("sample")
        self.assertIsNotNone(self.db.get("requests", "pr", "5s3"))
        self.assertIsNone(self.db.get("requests", "pr", "5"))

    def test_apply_on_a_skip_verdict_says_nothing_to_post(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, "skip"))
        self.model.poll()
        with self.assertLogs(fairy_tui.logger, level="INFO") as logs:
            self.model.act("apply")
        self.assertEqual(self.db.find("pr", "5"), "reviewed")
        self.assertTrue(any("nothing to post" in ln for ln in logs.output))

    def test_apply_refused_while_a_worker_holds_the_item(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        claim = self.db.claim("reviewed", "reviewed", "pr", "5")
        try:
            self.model.act("apply")
        finally:
            claim.abort()
        self.assertEqual(self.db.find("pr", "5"), "reviewed")

    def test_rerun_writes_a_request_and_respects_in_flight(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("rerun")
        self.assertEqual(self.db.get("requests", "pr", "5"), mock.ANY)
        self.assertEqual(self.db.get("requests", "pr", "5")["action"], "rerun")
        self.db.try_pop("requests", "pr", "5")
        self.db.try_move("reviewed", "queued", "pr", "5")
        self.model.poll()
        self.model.act("rerun")
        self.assertIsNone(self.db.get("requests", "pr", "5"))

    def test_skip_and_cancel_move_with_a_reason(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("merge-ready", "pr", "6", {"title": "t"})
        self.model.poll()
        self.model.act("skip")
        skipped = self.db.get("skipped", "pr", "5")
        self.assertEqual(skipped["reason"], "operator skip")
        self.assertNotIn("snoozed_at", skipped)  # one-shot: no snooze
        self.assertNotIn("llm_at", skipped)  # next scan reconsiders it
        self.model.poll()
        with self.model.lock:
            self.model._move_cursor_to((R1, "pr", "6"))
        self.model.act("cancel")
        self.assertEqual(self.db.get("cancelled", "pr", "6")["reason"],
                         "operator cancel")

    def test_snooze_stamps_the_press_time(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("snooze")
        skipped = self.db.get("skipped", "pr", "5")
        self.assertEqual(skipped["reason"], "operator snooze")
        self.assertTrue(skipped["snoozed_at"])  # the snooze starts at the press

    def test_x_on_a_running_review_requests_the_cancel(self) -> None:
        self.db.push("llm", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("cancel")
        t = self.db.get("llm", "pr", "5")
        self.assertTrue(t["cancel"])
        self.assertEqual(t["reason"], "operator cancel")
        self.assertEqual(self.db.find("pr", "5"), "llm")

    def test_act_advances_to_the_next_reviewed_row(self) -> None:
        self.db.push("reviewed", "pr", "1", verdict(1))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "1"))
        self.model.act("apply")
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))

    def test_applied_row_stays_listed_through_posted(self) -> None:
        # After y the agent moves the file to posted/; the row must not
        # vanish, or the operator cannot verify the post landed.
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.model.act("apply")
        self.db.try_move("outgoing", "posted", "pr", "5")  # the agent's send pass
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "posted")
        self.assertIn((R1, "5"), self.keys())


class MultiSideTests(DbCase):
    """N repos: items are keyed per repo and actions stay side-local."""

    def test_same_number_in_two_repos_is_two_rows(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db2.push("queued", "pr", "5", verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "reviewed")
        self.assertEqual(self.model.items[(R2, "pr", "5")].state, "queued")

    def test_act_works_on_the_cursor_rows_db(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db2.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        with self.model.lock:
            self.model._move_cursor_to((R2, "pr", "5"))
        self.model.act("apply")
        self.assertEqual(self.db2.find("pr", "5"), "outgoing")
        self.assertEqual(self.db.find("pr", "5"), "reviewed")

    def test_stats_show_what_could_be_applied(self) -> None:
        self.db.push("reviewed", "pr", "5", dict(verdict(5), action="comment"))
        self.db.push("reviewed", "pr", "6", dict(verdict(6), action="approve"))
        self.db.push("reviewed", "pr", "7", verdict(7, "skip"))  # not appliable
        self.model.poll()
        ui = make_ui(self.model)
        with self.model.lock:
            text = fairy_tui._plain(ui.stats_lines(120))
        self.assertIn("awaiting you: approve=1, comment=1", text)

    def test_stats_tile_one_block_per_repo(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db2.push("queued", "issue", "7", verdict(7))
        self.model.poll()
        ui = make_ui(self.model)
        with self.model.lock:
            wide = fairy_tui._plain(ui.stats_lines(120)).split("\n")
            narrow = fairy_tui._plain(ui.stats_lines(24)).split("\n")
        # Wide: both repo headings land on the same tiled line (short
        # repo display names, as in the list column).
        self.assertTrue(any("r " in ln and "r2 " in ln for ln in wide), wide)
        # Narrow: the blocks stack, one heading per line.
        self.assertTrue(any("r2 " in ln and " r " not in ln for ln in narrow),
                        narrow)

    def _rows_text(self, model: fairy_tui.Model) -> list[str]:
        ui = make_ui(model)
        with model.lock:
            rows = ui.list_rows()
        return fairy_tui._plain(rows).split("\n")

    def test_a_dash_classification_stays_out_of_the_stats(self) -> None:
        self.db.push("skipped", "pr", "5", verdict(5, classification="-"))
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model.side_stats()[R1]["cls"], {})

    def test_repo_column_only_when_sides_span_repos(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db2.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        rows = self._rows_text(self.model)
        self.assertIn("PR    r  #5", rows[0])   # short name, padded to "r2"
        self.assertIn("PR    r2 #5", rows[1])
        single = fairy_tui.Model([(R1, self.db)])
        single.poll()
        self.assertIn("PR    #5", self._rows_text(single)[0])

    def test_colliding_short_names_fall_back_to_owner_repo(self) -> None:
        model = fairy_tui.Model([("a/x", self.db), ("b/x", self.db2),
                                 ("c/y", self.db)])
        ui = make_ui(model)
        self.assertEqual(ui._repo_disp,
                         {"a/x": "a/x", "b/x": "b/x", "c/y": "y"})

    def test_visible_stats_export_matches_the_painted_pane(self) -> None:
        # e must export what is on screen: the tiled layout depends on
        # the pane width, which the draggable divider controls.
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db2.push("queued", "pr", "7", verdict(7))
        self.model.poll()
        save = tempfile.TemporaryDirectory()
        self.addCleanup(save.cleanup)
        ui = make_ui(self.model, cols=160, save_dir=Path(save.name))
        ui.focus = "tl"
        ui.layout.fx_top = 0.15  # narrow stats pane: blocks stack on screen
        rects = ui.layout.rects(ui.term.width, max(3, ui.term.height - 1))
        with self.model.lock:
            painted = fairy_tui._plain(
                ui.stats_lines(ui._text_width(rects["tl"])))
        ui.export(full=False)
        out = next(Path(save.name).glob("fairy_tui-stats-*.txt"))
        expected = painted.split("\n")[:ui._page() + 1]
        self.assertEqual(out.read_text().rstrip("\n").split("\n"), expected)

    def test_pr_and_issue_kinds_share_one_db(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("queued", "issue", "5", verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", "5")].state, "reviewed")
        self.assertEqual(self.model.items[(R1, "issue", "5")].state, "queued")


class SideBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()

    def root(self, name: str, label: str, log_files: set[Path]) -> Path:
        root = self.base / name
        root.mkdir()
        fairy_tui.db_config.write_config(root, label, log_files, None, None)
        return root

    def test_sides_come_from_each_roots_config(self) -> None:
        x = self.root("a~x", "a/x", {self.base / "x.log"})
        y = self.root("a~y", "a/y", {self.base / "y.log"})
        args = fairy_tui.parse_args(
            ["--db-root", str(x), "--db-root", str(y),
             "--db-root", str(x),
             "--tail", str(self.base / "nosuch" / ".." / "x.log"),
             "--tail", "extra.log"])
        sides, tails = fairy_tui.build_sides(args)
        self.assertEqual([label for label, _ in sides], ["a/x", "a/y"])
        self.assertEqual([db.root for _, db in sides], [x, y])
        self.assertEqual(tails, [self.base / "x.log", self.base / "y.log",
                                 Path("extra.log")])

    def test_missing_config_names_a_case_sibling(self) -> None:
        self.root("FFmpeg~web", "FFmpeg/web", set())
        args = fairy_tui.parse_args(["--db-root", str(self.base / "ffmpeg~Web")])
        with self.assertRaisesRegex(SystemExit, "FFmpeg~web exists"):
            fairy_tui.build_sides(args)

    def test_existing_root_without_config_does_not_blame_its_own_case(self) -> None:
        (self.base / "a~x").mkdir()
        args = fairy_tui.parse_args(["--db-root", str(self.base / "a~x")])
        with self.assertRaises(SystemExit) as ctx:
            fairy_tui.build_sides(args)
        self.assertNotIn("check the case", str(ctx.exception))


class SortTests(DbCase):
    """t cycles the visible-list sort; sorts are stable over arrival."""

    def _numbers(self) -> list[filedb.TicketId]:
        return [n for _, n in self.keys()]

    def test_status_sort_bubbles_actionable_rows_stably(self) -> None:
        for n, state in ((1, "queued"), (2, "reviewed"),
                         (3, "queued"), (4, "reviewed")):
            self.db.push(state, "pr", str(n), verdict(n))
        self.model.poll()
        self.assertEqual(self._numbers(), ["1", "2", "3", "4"])
        self.assertEqual(self.model.cycle_sort(), "status")
        # reviewed first; arrival order preserved within equal status.
        self.assertEqual(self._numbers(), ["2", "4", "1", "3"])

    def test_repo_and_number_modes(self) -> None:
        # "z/AA" sorts last by raw owner/repo but its displayed short
        # name "AA" sorts first: repo mode must follow the column.
        tmp3 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp3.cleanup)
        db3 = filedb.Db(Path(tmp3.name))
        self.model.sides.append(("z/AA", db3))
        self.db.push("queued", "pr", "5", verdict(5))
        self.db.push("queued", "pr", "9", verdict(9))
        self.db2.push("queued", "pr", "7", verdict(7))
        db3.push("queued", "pr", "3", verdict(3))
        self.model.poll()
        self.model.sort_mode = "repo"
        self.assertEqual(self.keys(),
                         [("z/AA", "3"), (R1, "5"), (R1, "9"), (R2, "7")])
        self.model.sort_mode = "number"
        self.assertEqual(self._numbers(), ["3", "5", "7", "9"])

    def test_cursor_follows_its_item_when_a_poll_reorders(self) -> None:
        """The cursor IS a key, not an index: a status flip elsewhere
        must never move the selection off its PR."""
        self.db.push("queued", "pr", "1", verdict(1))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.model.poll()
        self.model.sort_mode = "status"
        with self.model.lock:
            self.model._move_cursor_to((R1, "pr", "2"))
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
        self.db.try_move("queued", "reviewed", "pr", "1")  # bubbles above #2
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
            self.assertEqual(self.model.cursor, 1)

    def test_cursor_stays_through_a_transient_rename_blip(self) -> None:
        """A poll racing a rename can see the item in no state dir for
        a tick; the grace polls keep the row alive and the sticky key
        keeps the cursor on it -- it never visits a neighbour."""
        self.db.push("reviewed", "pr", "1", verdict(1))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.model.poll()
        with self.model.lock:
            self.model._move_cursor_to((R1, "pr", "2"))
        self.db.try_pop("reviewed", "pr", "2")
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
        self.db.push("llm", "pr", "2", verdict(2))
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
            self.assertEqual(self.model.items[(R1, "pr", "2")].state, "llm")

    def test_cycle_wraps_back_to_arrival(self) -> None:
        for expected in ("status", "repo", "number", "arrival"):
            self.assertEqual(self.model.cycle_sort(), expected)

    def test_every_state_has_a_sort_priority(self) -> None:
        # A filedb state missing from _SORT_STATES would KeyError on
        # the UI thread the first time t reaches status mode.
        self.assertEqual(set(fairy_tui._SORT_STATES),
                         set(filedb.STATES) | {fairy_tui.INVALID})

    def test_t_key_keeps_the_cursor_on_its_row_and_labels_the_bar(self) -> None:
        self.db.push("queued", "pr", "1", verdict(1))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.model.poll()
        stream = io.StringIO()
        ui = make_ui(self.model, stream, cols=160)
        self.model.cursor = 1            # on #2 in arrival order
        ui.dispatch(Key("t"))            # -> status sort: #2 is first
        self.assertEqual(self.model.sort_mode, "status")
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
        self.assertEqual(self.model.cursor, 0)
        ui.paint()
        self.assertIn("sort:status", stream.getvalue())


class WheelTests(DbCase):
    """The wheel moves the view; only cursor keys and clicks move the
    cursor, and the window stops chasing it until the next cursor move.

    On a real tty blessed reads the ioctl window size, not the COLUMNS/
    LINES the fixture sets, so coordinates and row counts are derived
    from the actual layout instead of hardcoded."""

    def _mouse(self, ui, name: str):
        rects = ui.layout.rects(ui.term.width, max(3, ui.term.height - 1))
        k = NamedKey(name)
        k.mouse_yx = (rects["tr"].y + 1, rects["tr"].x + 2)
        return k

    def _ui_with_rows(self):
        ui = make_ui(self.model)
        for n in range(1, ui._page() + 13):
            self.db.push("reviewed", "pr", str(n), verdict(n))
        self.model.poll()
        return ui

    def test_wheel_scrolls_the_view_and_leaves_the_cursor(self) -> None:
        ui = self._ui_with_rows()
        with self.model.lock:
            self.model.select_index(0)
        ui.paint()
        key = self.model.cursor_key
        ui.dispatch(self._mouse(ui, "MOUSE_SCROLL_DOWN"))
        self.assertEqual(ui.list_top, 3)
        self.assertEqual(self.model.cursor_key, key)
        ui.paint()
        self.assertEqual(ui.list_top, 3)  # timer repaints do not snap back

    def test_message_pane_arrows_step_the_list_cursor(self) -> None:
        ui = self._ui_with_rows()
        ui.focus = "br"
        ui.scroll["br"] = 4
        with self.model.lock:
            self.model.select_index(1)
        ui.dispatch(NamedKey("KEY_RIGHT"))
        with self.model.lock:
            self.model._sync_cursor()
            self.assertEqual(self.model.cursor, 2)
        self.assertEqual(ui.scroll["br"], 0)  # the next ticket reads from its top
        ui.dispatch(NamedKey("KEY_LEFT"))
        ui.dispatch(NamedKey("KEY_LEFT"))
        with self.model.lock:
            self.model._sync_cursor()
            self.assertEqual(self.model.cursor, 0)

    def test_home_and_end_jump_the_cursor(self) -> None:
        ui = self._ui_with_rows()
        with self.model.lock:
            self.model.select_index(3)
        ui.dispatch(NamedKey("KEY_END"))
        with self.model.lock:
            vis = self.model._sync_cursor()
            self.assertEqual(self.model.cursor, len(vis) - 1)
        ui.dispatch(NamedKey("KEY_HOME"))
        with self.model.lock:
            self.model._sync_cursor()
            self.assertEqual(self.model.cursor, 0)

    def test_an_arrow_after_wheeling_returns_the_view_to_the_cursor(self) -> None:
        ui = self._ui_with_rows()
        with self.model.lock:
            self.model.select_index(0)
        ui.paint()
        ui.dispatch(self._mouse(ui, "MOUSE_SCROLL_DOWN"))
        ui.dispatch(self._mouse(ui, "MOUSE_SCROLL_DOWN"))
        ui.dispatch(NamedKey("KEY_DOWN"))
        ui.paint()
        self.assertLessEqual(ui.list_top, 1)
        with self.model.lock:
            self.assertIsNotNone(self.model._cursor_key())
            self.assertEqual(self.model.cursor, 1)


class CopyMessageTests(DbCase):
    def test_clicking_the_message_title_copies_the_raw_message(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, msg="full **body**"))
        self.model.poll()
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui.paint()
        self.assertIn("⧉", stream.getvalue())
        rect = ui._shown["br"][0]
        sent: list[str] = []
        with mock.patch.object(ui, "_to_clipboard",
                               side_effect=lambda t, d: sent.append(t)):
            ui._copy_click("br", rect.x + 2, rect.y)
        self.assertEqual(sent, ["full **body**"])

    def test_an_error_ticket_copies_the_plain_pane_text(self) -> None:
        self.db.push("error", "pr", "5", {"title": "t", "error": "boom"})
        self.model.poll()
        ui = make_ui(self.model)
        ui.paint()
        rect = ui._shown["br"][0]
        sent: list[str] = []
        with mock.patch.object(ui, "_to_clipboard",
                               side_effect=lambda t, d: sent.append(t)):
            ui._copy_click("br", rect.x + 2, rect.y)
        self.assertEqual(len(sent), 1)
        self.assertIn("boom", sent[0])


class ErrorAgeColumnTests(DbCase):
    def test_error_rows_show_the_errors_age_in_the_llm_column(self) -> None:
        from datetime import datetime, timedelta, timezone
        stamp = (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
        self.db.push("error", "pr", "5", {"title": "t", "error": "boom",
                                        "llm_at": stamp})
        self.model.poll()
        ui = make_ui(self.model)
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        self.assertTrue(any("err=2d" in t for t in rows), rows)


class StatusColumnTests(DbCase):
    """The letter cluster between #number and state is drawn from the
    items/ snapshot's status fields, with the cancelled ticket's reason
    taking over once the item is closed and no longer snapshotted."""

    def rows(self) -> list[str]:
        ui = make_ui(self.model)
        return ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]

    def test_open_pr_shows_automerge_and_review_counts(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("items", "pr", "5", {
            "title": "t", "state": "open", "auto_merge": "merge",
            "approvals": 2, "change_requests": 1, "discussion": []})
        self.model.poll()
        self.assertIn("#5       a21", self.rows()[0])

    def test_sample_tickets_share_the_base_items_status(self) -> None:
        self.db.push("reviewed", "pr", "5s2", verdict(5))
        self.db.push("items", "pr", "5", {
            "state": "open", "approvals": 1, "change_requests": 0})
        self.model.poll()
        self.assertIn("#5s2     " + " 10", self.rows()[0])

    def test_a_long_sample_token_keeps_the_columns_aligned(self) -> None:
        self.db.push("reviewed", "pr", "21684s1", verdict(21684))
        self.db.push("reviewed", "pr", "9", verdict(9))
        self.model.poll()
        self.assertEqual(len({r.index("reviewed") for r in self.rows()}), 1)

    def test_merged_pr_distinguishes_auto_from_manual(self) -> None:
        self.db.push("cancelled", "pr", "5",
                     dict(verdict(5), reason="merged", auto_merge="merge"))
        self.db.push("cancelled", "pr", "6", dict(verdict(6), reason="merged"))
        self.db.push("cancelled", "pr", "7",
                     dict(verdict(7), reason="closed without merge"))
        with self.model.lock:
            self.model.filter_mode = "all"
        self.model.poll()
        clusters = [r[19:23] for r in self.rows()]
        self.assertEqual(clusters, ["M   ", "m   ", "R   "])

    def test_issue_letters_come_from_labels_and_state(self) -> None:
        self.db.push("reviewed", "issue", "3", verdict(3))
        self.db.push("items", "issue", "3", {
            "state": "open",
            "labels": ["bug", "repro/yes", "resolution/fixed"]})
        self.db.push("reviewed", "issue", "4", verdict(4))
        self.db.push("items", "issue", "4", {
            "state": "closed", "labels": ["enhancement", "repro/no(env)",
                                          "resolution/wontfix"]})
        self.model.poll()
        clusters = [r[19:23] for r in self.rows()]
        self.assertEqual(clusters, ["BYfO", "EnwC"])

    def test_forced_reviews_carry_a_plus_through_queued_and_llm(self) -> None:
        self.db.push("queued", "pr", "5", dict(verdict(5), forced=True))
        self.db.push("queued", "pr", "6", dict(verdict(6), forced=False))
        self.model.poll()
        self.assertIn("queued+", self.rows()[0])
        self.assertIn("queued ", self.rows()[1])
        self.db.try_move("queued", "llm", "pr", "5",
                         mutate=lambda d: d.update(stage="triage"))
        self.model.poll()
        self.assertIn("llm+", self.rows()[0])

    def test_without_a_snapshot_the_cluster_is_blank(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        self.assertIn("#5            reviewed", self.rows()[0])

    def test_a_snapshot_rewrite_updates_the_status(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("items", "pr", "5", {"state": "open", "approvals": 0,
                                          "change_requests": 0})
        self.model.poll()
        self.assertIn("#5        00", self.rows()[0])
        self.db.push("items", "pr", "5", {"state": "open", "approvals": 3,
                                          "change_requests": 0})
        self.model.poll()
        self.assertIn("#5        30", self.rows()[0])


class HelpTests(DbCase):
    """? renders README-TUI.md full-screen until any non-scroll key."""

    def test_help_opens_scrolls_and_any_key_returns(self) -> None:
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui.dispatch(Key("?"))
        self.assertIn("fairy_tui.py", ui.help_text)
        ui.paint()
        self.assertIn("README-TUI.md", stream.getvalue())
        ui.dispatch(NamedKey("KEY_UP"))
        self.assertEqual(ui.help_scroll, 0)  # never above the top
        ui.dispatch(NamedKey("KEY_DOWN"))
        self.assertEqual(ui.help_scroll, 1)
        ui.dispatch(Key("q"))
        self.assertIsNone(ui.help_text)
        self.assertFalse(ui.model.quit_flag)  # q closed help, not the app


class PauseTests(DbCase):
    """p freezes the session's agents/workers; a second p thaws them."""

    def test_session_pids_finds_marked_siblings_with_children(self) -> None:
        child = subprocess.Popen(
            [sys.executable, "-c",
             "import subprocess, sys, time\n"
             "p = subprocess.Popen([sys.executable, '-c',"
             " 'import time; time.sleep(30)'])\n"
             "print(p.pid, flush=True)\n"
             "time.sleep(30)",
             "agent.py"],
            stdout=subprocess.PIPE, text=True)
        self.addCleanup(child.kill)
        self.addCleanup(child.stdout.close)
        grandchild = int(child.stdout.readline())
        self.addCleanup(lambda: os.kill(grandchild, signal.SIGKILL))
        other = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"])
        self.addCleanup(other.kill)
        pids = fairy_tui.session_pids(parent=os.getpid())
        self.assertIn(child.pid, pids)
        self.assertIn(grandchild, pids)
        self.assertNotIn(other.pid, pids)
        self.assertNotIn(os.getpid(), pids)

    def test_p_stops_the_tree_and_a_second_p_continues_it(self) -> None:
        ui = make_ui(self.model)
        sent: list[tuple[int, int]] = []
        with mock.patch.object(fairy_tui, "session_pids",
                               return_value=[111, 222]), \
                mock.patch.object(fairy_tui.os, "kill",
                                  side_effect=lambda p, s: sent.append((p, s))):
            ui.dispatch(Key("p"))
            self.assertEqual(sent, [(111, signal.SIGSTOP),
                                    (222, signal.SIGSTOP)])
            self.assertEqual(ui.paused, [111, 222])
            self.assertIn("PAUSED",
                          fairy_tui._plain(ui.stats_lines(80)))
            sent.clear()
            ui.dispatch(Key("p"))
            self.assertEqual(sent, [(111, signal.SIGCONT),
                                    (222, signal.SIGCONT)])
            self.assertEqual(ui.paused, [])

    def test_quit_thaws_paused_processes(self) -> None:
        """A frozen daemon never sees the launcher's exit-trap SIGTERM,
        so quitting the TUI while paused must thaw first."""
        ui = make_ui(self.model)
        ui.paused = [111]
        self.model.quit_all()
        sent: list[tuple[int, int]] = []
        with mock.patch.object(fairy_tui, "Thread"), \
                mock.patch.object(fairy_tui.os, "kill",
                                  side_effect=lambda p, s: sent.append((p, s))):
            ui.run()
        self.assertEqual(sent, [(111, signal.SIGCONT)])
        self.assertEqual(ui.paused, [])


class DetailTests(DbCase):
    """The detail pane renders everything from the ticket dict."""

    def _detail_text(self) -> str:
        ui = make_ui(self.model)
        with self.model.lock:
            return fairy_tui._plain(ui.detail_lines(100))

    def test_reviewed_ticket_renders_message_and_action(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, msg="persisted body"))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("persisted body", text)
        self.assertIn("state reviewed", text)
        self.assertIn("comment", text)  # rebuilt decision's action

    def test_the_opening_description_leads_the_thread(self) -> None:
        """An item whose only text is its description (no comments yet)
        showed an empty thread while the forge web UI showed the post."""
        self.db.push("reviewed", "pr", "5", verdict(
            5, body="the initial message", discussion=[
                {"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T09:00:00Z", "body": "ack"}]))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("discussion (2)", text)
        self.assertIn("a  description", text)
        self.assertIn("the initial message", text)
        self.assertLess(text.index("the initial message"), text.index("ack"))

    def test_the_review_renders_at_the_threads_end_marked_unposted(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(
            5, msg="persisted body", discussion=[
                {"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T09:00:00Z", "body": "please rebase"}]))
        self.model.poll()
        text = self._detail_text()
        self.assertLess(text.index("please rebase"),
                        text.index("persisted body"))
        self.assertLess(text.index("NOT POSTED"), text.index("persisted body"))

    def test_a_posted_review_loses_the_unposted_mark(self) -> None:
        self.model.filter_mode = "all"  # posted rows hide from "relevant"
        self.db.push("posted", "pr", "5", verdict(
            5, msg="persisted body",
            posted_at="2026-07-21T00:00:00+00:00"))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("persisted body", text)
        self.assertIn("review  2026-07-21", text)
        self.assertNotIn("NOT POSTED", text)

    def test_the_unposted_review_renders_below_the_thread(self) -> None:
        disc = [
            {"kind": "comment", "author": "carol",
             "created_at": "2026-07-18T09:00:00Z", "body": "please **rebase**",
             "attachment_urls": ["https://forge/attachments/1"]},
            {"kind": "review", "author": "dave", "state": "APPROVED",
             "submitted_at": "2026-07-18T10:00:00Z", "body": "LGTM"},
            {"kind": "review_comment", "author": "erin",
             "created_at": "2026-07-18T11:00:00Z", "path": "src/x.c",
             "line": 42, "body": "off by one"},
            {"kind": "push", "author": "a",
             "created_at": "2026-07-18T12:00:00Z",
             "head_sha": "abcdef1234567890", "is_force_push": True,
             "commit_count": 2},
        ]
        self.db.push("reviewed", "pr", "5",
                     verdict(5, msg="persisted body", discussion=disc))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("discussion (4)", text)
        self.assertIn("carol  comment  2026-07-18", text)
        self.assertIn("rebase", text)
        self.assertIn("https://forge/attachments/1", text)
        self.assertIn("dave  review APPROVED", text)
        self.assertIn("erin  review_comment  src/x.c:42", text)
        self.assertIn("a  force-pushed 2 commit(s) abcdef1234", text)
        self.assertLess(text.index("carol"), text.index("persisted body"))

    def test_queued_ticket_shows_the_thread_from_its_prepared_payload(self) -> None:
        self.db.push("queued", "pr", "5", {"title": "t", "prepared": {
            "discussion": [{"kind": "comment", "author": "carol",
                            "created_at": "2026-07-18T09:00:00Z",
                            "body": "still applies?"}]}})
        self.model.poll()
        text = self._detail_text()
        self.assertIn("discussion (1)", text)
        self.assertIn("still applies?", text)

    def test_author_and_branch_are_shown(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, head_branch="fix-lavc"))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("author a", text)
        self.assertIn("branch fix-lavc", text)

    def test_error_ticket_shows_reason_and_its_age(self) -> None:
        """An error without a date reads as current; the operator must
        see whether it predates their retry (production: #23903)."""
        self.db.push("error", "pr", "5", {
            "title": "t", "error": "LLM exploded",
            "llm_at": "2026-07-27T04:15:00+00:00"})
        self.model.poll()
        text = self._detail_text()
        self.assertIn("ago): LLM exploded", text)
        self.assertIn("2026-07-27", text)

    def test_gate_ticket_shows_attention_context(self) -> None:
        self.db.push("ci-blocked", "pr", "5", {
            "title": "t", "reason": "ci red",
            "cancelled_ci_contexts": ["job1"], "blocked_ci_contexts": []})
        self.model.poll()
        text = self._detail_text()
        self.assertIn("reason: ci red", text)
        self.assertIn("cancelled ci contexts: job1", text)

    def test_review_age_is_shown(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("reviewed 2026-07-20", text)
        self.assertIn("ago)", text)

    def test_failed_reviewers_are_shown(self) -> None:
        """Production #23901: the GPT reviewer died (provider content
        flag), the verdict silently came from GLM alone."""
        self.db.push("reviewed", "pr", "5", verdict(
            5, failed_reviewers=["codex:gpt-5.6-sol: content flagged"]))
        self.model.poll()
        self.assertIn("reviewer failed: codex:gpt-5.6-sol: content flagged",
                      self._detail_text())

    def test_long_error_wraps_instead_of_clipping(self) -> None:
        """Regression (review finding): the error line was clipped at
        terminal width, hiding the provider's actual failure text
        behind ~190 chars of prefixes (observed with the cybersecurity
        content flag)."""
        error = ("LLM analysis failed: LLM review gave up: provider-ended "
                 "turns exhausted the wrapper's in-run retry budget: "
                 "codex:gpt-5.6-sol: codex exec produced no final message "
                 '(rc=1); errors: {"type": "error", "message": "This '
                 "content was flagged for possible cybersecurity risk. If "
                 "this seems wrong, try rephrasing your request. To get "
                 "authorized for security work, join the Trusted Access "
                 'for Cyber program: https://chatgpt.com/cyber"}')
        self.db.push("reviewed", "pr", "5", verdict(5, error=error))
        self.model.poll()
        text = " ".join(self._detail_text().split())
        self.assertIn("flagged for possible cybersecurity risk", text)
        self.assertIn("https://chatgpt.com/cyber", text)

    def test_send_blocked_note_is_shown(self) -> None:
        self.db.push("reviewed", "pr", "5",
                     verdict(5, send_blocked="PR updated_at changed"))
        self.model.poll()
        self.assertIn("send blocked: PR updated_at changed",
                      self._detail_text())

    def test_snapshot_replaces_the_thread_with_a_separator(self) -> None:
        old = {"kind": "comment", "author": "carol",
               "created_at": "2026-07-18T09:00:00Z", "body": "please rebase"}
        new = {"kind": "comment", "author": "dave",
               "created_at": "2026-07-20T09:00:00Z", "body": "rebased now"}
        self.db.push("reviewed", "pr", "5", verdict(5, discussion=[old]))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "the initial message",
            "updated_at": "2026-07-20T09:00:00Z", "discussion": [old, new]})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertIn("discussion (3)", text)  # description + both comments
        self.assertIn("rebased now", text)
        self.assertLess(text.index("please rebase"),
                        text.index("sampled for the review"))
        self.assertLess(text.index("sampled for the review"),
                        text.index("rebased now"))

    def test_separator_trails_a_thread_without_new_activity(self) -> None:
        old = [{"kind": "comment", "author": "carol",
                "created_at": "2026-07-18T09:00:00Z", "body": "please rebase"},
               {"kind": "comment", "author": "carol",
                "created_at": "2026-07-18T10:00:00Z", "body": "last word"}]
        self.db.push("reviewed", "pr", "5", verdict(5, discussion=old))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": old})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertLess(text.index("last word"),
                        text.index("sampled for the review"))

    def test_edited_comment_stays_above_the_separator(self) -> None:
        """A comment edit bumps the comment's updated_at but not the
        item's (see gcli_cache), so the sampling watermark can predate
        the edit; placement goes by arrival, keeping the comment above
        the separator. Times from FFmpeg issue #22240."""
        edited = {"kind": "comment", "author": "carol",
                  "created_at": "2026-02-22T09:29:02Z",
                  "updated_at": "2026-02-22T09:30:32Z", "body": "edited later"}
        self.db.push("reviewed", "pr", "5", verdict(
            5, expected_updated_at="2026-02-22T09:29:02Z",
            discussion=[edited]))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": [edited]})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertLess(text.index("edited later"),
                        text.index("sampled for the review"))

    def test_no_review_row_gets_the_plain_sampled_wording(self) -> None:
        self.db.push("merge-ready", "pr", "5", {
            "title": "t", "reason": "approved",
            "expected_updated_at": "2026-07-19T10:00:00Z"})
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": [
                {"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T09:00:00Z", "body": "hi"}]})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertIn("── sampled 2026-07-19", text)
        self.assertNotIn("sampled for the review", text)

    def test_no_separator_without_a_sampling_stamp(self) -> None:
        self.db.push("queued", "pr", "5", {"title": "t"})
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": [
                {"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T09:00:00Z", "body": "hi"}]})
        self.model.poll()
        self.model.poll_snapshot()
        self.assertNotIn("sampled", self._detail_text())

    def test_empty_thread_renders_no_lone_separator(self) -> None:
        self.db.push("reviewed", "pr", "5",
                     verdict(5, body="", discussion=[]))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": []})
        self.model.poll()
        self.model.poll_snapshot()
        self.assertNotIn("sampled", self._detail_text())

    def test_junk_entries_do_not_shift_the_separator(self) -> None:
        disc = [{"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T09:00:00Z", "body": "old one"},
                "junk",
                {"kind": "comment", "author": "carol",
                 "created_at": "2026-07-18T10:00:00Z", "body": "old two"},
                {"kind": "comment", "author": "dave",
                 "created_at": "2026-07-20T09:00:00Z", "body": "new one"}]
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": disc})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertLess(text.index("old two"), text.index("sampled"))
        self.assertLess(text.index("sampled"), text.index("new one"))

    def test_description_survives_an_emptied_snapshot_body(self) -> None:
        self.db.push("reviewed", "pr", "5",
                     verdict(5, body="the initial message"))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "", "discussion": []})
        self.model.poll()
        self.model.poll_snapshot()
        self.assertIn("the initial message", self._detail_text())

    def test_header_prefers_the_snapshot_title(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))  # title "from disk"
        self.db.push("items", "pr", "5", {
            "title": "renamed since", "author": "a", "body": "",
            "discussion": []})
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertIn("renamed since", text)
        self.assertNotIn("from disk", text.split("\n")[0])

    def test_without_a_snapshot_the_ticket_thread_renders(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, discussion=[
            {"kind": "comment", "author": "carol",
             "created_at": "2026-07-18T09:00:00Z", "body": "please rebase"}]))
        self.model.poll()
        self.model.poll_snapshot()
        text = self._detail_text()
        self.assertIn("please rebase", text)
        self.assertNotIn("sampled for the review", text)

    def test_sample_ticket_reads_the_base_items_snapshot(self) -> None:
        self.db.push("reviewed", "pr", "5s1", verdict(5))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "",
            "updated_at": "2026-07-19T10:00:00Z", "discussion": [
                {"kind": "comment", "author": "dave",
                 "created_at": "2026-07-20T09:00:00Z", "body": "fresh word"}]})
        self.model.poll()
        self.model.poll_snapshot()
        self.assertIn("fresh word", self._detail_text())

    def test_closed_snapshots_keep_serving_the_pane(self) -> None:
        self.model.filter_mode = "all"  # cancelled rows hide from "relevant"
        self.db.push("cancelled", "pr", "5", verdict(5, reason="merged"))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "",
            "updated_at": "2026-07-19T10:00:00Z", "discussion": [
                {"kind": "comment", "author": "dave",
                 "created_at": "2026-07-18T09:00:00Z", "body": "final word"}]})
        self.model.poll()
        self.model.poll_snapshot()
        self.assertIn("final word", self._detail_text())

    def test_poll_snapshot_is_quiet_while_nothing_changes(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5))
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "",
            "updated_at": "2026-07-19T10:00:00Z", "discussion": []})
        self.model.poll()
        self.model.poll_snapshot()
        self.model.dirty.clear()
        self.model.poll_snapshot()
        self.assertFalse(self.model.dirty.is_set())
        self.db.push("items", "pr", "5", {
            "title": "t5", "author": "a", "body": "",
            "updated_at": "2026-07-21T00:00:00Z", "discussion": []})
        self.model.poll_snapshot()
        self.assertTrue(self.model.dirty.is_set())

    def test_invalid_file_shows_reason(self) -> None:
        self.db.path("reviewed", "pr", "5").write_text("{ broken",
                                                     encoding="utf-8")
        self.model.poll()
        self.assertIn("file invalid:", self._detail_text())


class EditReviewTests(DbCase):
    def test_o_key_round_trips_the_message_through_the_editor(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, msg="original"))
        self.model.poll()

        def fake_call(cmd, **kw):
            Path(cmd[-1]).write_text("edited body", encoding="utf-8")
            return 0

        ui = make_ui(self.model)
        with mock.patch.dict(os.environ, {"EDITOR": "myeditor"}), \
                mock.patch.object(fairy_tui.subprocess, "call",
                                  side_effect=fake_call) as call:
            ui.edit_review()
        self.assertEqual(call.call_args.args[0][0], "myeditor")
        self.assertEqual(self.db.get("reviewed", "pr", "5")["review"]["message"],
                         "edited body")

    def test_edit_refused_while_a_worker_holds_the_item(self) -> None:
        self.db.push("reviewed", "pr", "5", verdict(5, msg="original"))
        self.model.poll()

        def fake_call(cmd, **kw):
            Path(cmd[-1]).write_text("edited body", encoding="utf-8")
            return 0

        ui = make_ui(self.model)
        claim = self.db.claim("reviewed", "reviewed", "pr", "5")
        try:
            with mock.patch.dict(os.environ, {"EDITOR": "e"}), \
                    mock.patch.object(fairy_tui.subprocess, "call",
                                      side_effect=fake_call):
                ui.edit_review()
        finally:
            claim.abort()
        self.assertEqual(self.db.get("reviewed", "pr", "5")["review"]["message"],
                         "original")


class FilterToggleTests(DbCase):
    def test_a_cycles_the_lenses_and_each_shows_its_states(self) -> None:
        fixtures = (("reviewed", "1"), ("merge-ready", "2"), ("ci-blocked", "3"),
                    ("awaiting-approver", "4"), ("posted", "5"), ("queued", "6"),
                    ("llm", "7"), ("outgoing", "8"))
        for state, n in fixtures:
            self.db.push(state, "pr", str(n), verdict(n))
        self.model.poll()
        expect = {
            "relevant": ["1", "2", "3", "4", "6", "7", "8"],  # settled posted/ hidden
            "review": ["1", "6", "7", "8"],           # pipeline around reviewed/
            "merge": ["2", "4"],                  # merge-ready and merge-ready*
            "ci": ["3"],
            "actionable": ["1", "2", "3", "4"],
            "all": ["1", "2", "3", "4", "5", "6", "7", "8"],
        }
        for mode in fairy_tui.FILTER_MODES[1:] + ("relevant",):
            self.assertEqual(self.model.cycle_filter(), mode)
            self.assertEqual([n for _, n in self.keys()], expect[mode], mode)

    def test_cursor_follows_selection_across_the_a_lens_cycle(self) -> None:
        self.db.push("skipped", "pr", "1", verdict(1, "skip"))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.db.push("skipped", "pr", "3", verdict(3, "skip"))
        self.model.poll()
        self.model.filter_mode = "all"
        ui = make_ui(self.model)

        with self.model.lock:
            self.model.select_index(1)   # on #2 in the "all" view
        ui.dispatch(Key("a"))            # all -> relevant: only #2 visible
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))
        self.assertEqual(self.model.cursor, 0)
        ui.dispatch(Key("a"))            # relevant -> review lens: #2 remains
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))

    def test_a_lens_hiding_the_key_hides_the_cursor_until_it_returns(self) -> None:
        """The invariant: the highlight only ever sits on the key's own
        row. A lens that hides the key shows NO cursor, actions refuse,
        and the key survives to be highlighted again -- never a
        neighbour."""
        self.db.push("skipped", "pr", "1", verdict(1, "skip"))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.db.push("skipped", "pr", "3", verdict(3, "skip"))
        self.model.poll()
        self.model.filter_mode = "all"
        ui = make_ui(self.model)
        with self.model.lock:
            self.model.select_index(2)
        ui.dispatch(Key("a"))
        with self.model.lock:
            self.assertIsNone(self.model._cursor_key())
            self.assertFalse(self.model.cursor_shown)
        self.model.act("apply")
        self.assertEqual(self.db.find("pr", "2"), "reviewed")
        for _ in range(len(fairy_tui.FILTER_MODES) - 1):
            ui.dispatch(Key("a"))
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "3"))

    def test_arrow_summons_a_hidden_cursor_at_its_old_spot(self) -> None:
        self.db.push("skipped", "pr", "1", verdict(1, "skip"))
        self.db.push("reviewed", "pr", "2", verdict(2))
        self.model.poll()
        self.model.filter_mode = "all"
        ui = make_ui(self.model)
        with self.model.lock:
            self.model.select_index(0)
        ui.dispatch(Key("a"))
        ui.dispatch(NamedKey("KEY_DOWN"))
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "2"))


class FsWatchTests(DbCase):
    def test_watch_paths_fires_on_new_files(self) -> None:
        import common
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        fired = Event()
        observer = common.watch_paths([Path(tmp.name)], fired.set)
        if observer is None:
            self.skipTest("watchdog not installed")
        self.addCleanup(observer.stop)
        (Path(tmp.name) / "pr-1.json").write_text("{}", encoding="utf-8")
        self.assertTrue(fired.wait(2.0), "no event within 2s")

    def test_a_state_directory_is_seen_through_its_root(self) -> None:
        """Watching each state directory cost one inotify instance per
        state per side -- 85 for seven sides, against the 128 a user
        gets for every program they run, which is what
        ``fs.inotify.max_user_instances`` refused. One recursive watch
        on the db root has to see the same change for that to hold."""
        import common
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        for state in filedb.STATES:
            (root / state).mkdir()
        fired = Event()
        observer = common.watch_paths([root], fired.set, recursive=True)
        if observer is None:
            self.skipTest("watchdog not installed")
        self.addCleanup(observer.stop)
        (root / "queued" / "pr-1.json").write_text("{}", encoding="utf-8")
        self.assertTrue(fired.wait(2.0), "no event within 2s")

    def test_without_recursion_the_root_sees_nothing_below_it(self) -> None:
        import common
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / "queued").mkdir()
        fired = Event()
        observer = common.watch_paths([root], fired.set)
        if observer is None:
            self.skipTest("watchdog not installed")
        self.addCleanup(observer.stop)
        (root / "queued" / "pr-1.json").write_text("{}", encoding="utf-8")
        self.assertFalse(fired.wait(1.0))

    def test_needs_poll_wakes_the_poll_thread(self) -> None:
        from threading import Thread
        ui = make_ui(self.model)
        polled = Event()
        with mock.patch.object(self.model, "poll", side_effect=polled.set):
            thread = Thread(target=ui._poll_loop, daemon=True)
            thread.start()
            ui.needs_poll.set()
            self.assertTrue(polled.wait(2.0), "poll thread did not wake")
            self.model.quit_flag = True
            ui.needs_poll.set()
            thread.join(2.0)
        self.assertFalse(thread.is_alive())

    def test_the_poll_thread_survives_a_poll_exception(self) -> None:
        from threading import Thread
        ui = make_ui(self.model)
        calls: list[int] = []

        def poll() -> None:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("bad ticket json")

        with mock.patch.object(self.model, "poll", side_effect=poll):
            thread = Thread(target=ui._poll_loop, daemon=True)
            thread.start()
            for _ in range(200):
                if len(calls) >= 2:
                    break
                ui.needs_poll.set()
                time.sleep(0.01)
            self.model.quit_flag = True
            ui.needs_poll.set()
            thread.join(2.0)
        self.assertGreaterEqual(len(calls), 2)


class CountAndSearchTests(DbCase):
    def setUp(self) -> None:
        super().setUp()
        for n, title in ((1, "hevc sao fix"), (2, "rtp muxer"), (3, "lut parse")):
            self.db.push("reviewed", "pr", str(n), verdict(n, msg=title, title=title))
        self.model.poll()
        self.ui = make_ui(self.model)

    def test_count_prefix_spawns_n_sample_requests(self) -> None:
        self.ui.dispatch(Key("3"))
        self.ui.dispatch(Key("r"))
        self.assertEqual([n for k, n in self.db.list_state("requests")],
                         ["1s1", "1s2", "1s3"])
        self.assertEqual(self.ui.count_buf, "")

    def test_count_ten_or_more_is_refused(self) -> None:
        self.ui.dispatch(Key("1"))
        self.ui.dispatch(Key("0"))
        with self.assertLogs(fairy_tui.logger, level="ERROR"):
            self.ui.dispatch(Key("r"))
        self.assertEqual(self.db.list_state("requests"), [])

    def test_backspace_edits_the_count(self) -> None:
        self.ui.dispatch(Key("1"))
        self.ui.dispatch(Key("0"))
        self.ui.dispatch(NamedKey("KEY_BACKSPACE"))
        self.ui.dispatch(Key("r"))  # count 1: plain rerun of the row
        self.assertEqual([n for k, n in self.db.list_state("requests")], ["1"])

    def test_count_scrolls_by_n_lines(self) -> None:
        self.ui.focus = "tr"
        self.model.cursor = 0
        self.ui.dispatch(Key("2"))
        self.ui.dispatch(NamedKey("KEY_DOWN"))
        self.assertEqual(self.model.cursor, 2)

    def test_search_over_an_empty_list_reports_no_match(self) -> None:
        for n in (1, 2, 3):
            self.db.try_pop("reviewed", "pr", str(n))
        self.model.poll()
        for ch in "/11":
            self.ui.dispatch(Key(ch))
        with self.assertLogs(fairy_tui.logger, level="INFO"):
            self.ui.dispatch(NamedKey("KEY_ENTER"))

    def test_search_with_a_stale_cursor_still_wraps(self) -> None:
        self.model.cursor = 99
        for ch in "/lut":
            self.ui.dispatch(Key(ch))
        self.ui.dispatch(NamedKey("KEY_ENTER"))
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "3"))

    def test_search_jumps_and_n_repeats(self) -> None:
        self.model.cursor = 0
        for ch in "/lut":
            self.ui.dispatch(Key(ch))
        self.ui.dispatch(NamedKey("KEY_ENTER"))
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "3"))
        self.ui.dispatch(Key("n"))  # wraps: only one match, stays
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", "3"))


class LogTailTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "agent.log"
        self.ring = tui_core.RingBuffer()
        self.sink = fairy_tui.OutputSink(self.ring, Event(), None)
        self.tail = fairy_tui.LogTail([self.path], self.sink)

    def lines(self) -> list:
        return list(self.ring.view(0, 100))

    def test_appended_lines_arrive_with_parsed_levels(self) -> None:
        self.tail.poll()  # file does not exist yet: retried, not fatal
        self.path.write_text("2026-07-24T10:00:00 I hello\n")
        self.tail.poll()
        with open(self.path, "a") as fh:
            fh.write("2026-07-24T10:00:01 W look out\n")
        self.tail.poll()
        tags = {text: tag for tag, text in self.lines()}
        self.assertEqual(tags["agent: 2026-07-24T10:00:00 I hello"], logging.INFO)
        self.assertEqual(tags["agent: 2026-07-24T10:00:01 W look out"],
                         logging.WARNING)

    def test_partial_line_is_withheld_until_complete(self) -> None:
        self.path.write_text("2026-07-24T10:00:00 I hel")
        self.tail.poll()
        self.assertEqual(self.lines(), [])
        with open(self.path, "a") as fh:
            fh.write("lo\n")
        self.tail.poll()
        self.assertEqual([t for _, t in self.lines()],
                         ["agent: 2026-07-24T10:00:00 I hello"])

    def test_truncation_restarts_from_the_top(self) -> None:
        self.path.write_text("2026-07-24T10:00:00 I one\n")
        self.tail.poll()
        self.path.write_text("2026-07-24T11:00:00 I 2\n")  # rotated, shorter
        self.tail.poll()
        self.assertIn("agent: 2026-07-24T11:00:00 I 2",
                      [t for _, t in self.lines()])

    def test_source_tags_align_in_one_column(self) -> None:
        base = self.path.parent
        paths = [base / f"{n}.log" for n in ("web", "fateserver", "fairies")]
        for p in paths:
            p.write_text("2026-07-24T10:00:00 I hi\n")
        tail = fairy_tui.LogTail(paths, self.sink)
        tail.poll()
        prefixes = {t.split(": ", 1)[0] for _, t in self.lines()}
        self.assertEqual(prefixes, {"web    ", "fateser", "fairies"})

    def test_tag_width_is_unique_prefix_plus_four_capped(self) -> None:
        self.assertEqual(
            fairy_tui._tag_width(["web", "fateserver", "fairies"]), 7)
        self.assertEqual(
            fairy_tui._tag_width(["ffmpeg", "fairies", "fairy_tui"]), 9)
        self.assertEqual(fairy_tui._tag_width(["agent"]), 5)
        self.assertEqual(fairy_tui._tag_width([]), 0)


class AgeColumnTests(DbCase):
    def test_age_formats_days_and_hours(self) -> None:
        from datetime import datetime, timezone
        now = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)
        self.assertEqual(fairy_tui._age("2026-07-15T12:00:00+00:00", now), "12d")
        self.assertEqual(fairy_tui._age("2026-07-27T07:00:00+00:00", now), "5h")
        self.assertEqual(fairy_tui._age("2026-07-14T19:57:30Z", now), "12d")
        self.assertEqual(fairy_tui._age(None, now), "")
        self.assertEqual(fairy_tui._age("garbage", now), "")

    def test_rows_show_activity_age_and_approval_age(self) -> None:
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        self.db.push("reviewed", "pr", "1", verdict(
            1, last_activity_iso=(now - timedelta(hours=5)).isoformat()))
        self.db.push("merge-ready", "pr", "2", {
            "title": "t", "approved_at": (now - timedelta(days=12)).isoformat(),
            "expected_updated_at": (now - timedelta(days=3)).isoformat()})
        self.model.poll()
        ui = make_ui(self.model)
        texts = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                 for r in ui.list_rows()]
        self.assertTrue(any(" 5h  " in t for t in texts), texts)
        merge_row = next(t for t in texts if "merge-ready" in t)
        self.assertIn("appr=12d", merge_row)
        self.assertIn(" 3d  ", merge_row)


class StatsWrapTests(DbCase):
    def test_state_counts_wrap_to_the_pane_width(self) -> None:
        states = ("ci-blocked", "merge-ready", "queued",
                  "posted", "skipped")
        for n, state in enumerate(states, 1):
            self.db.push(state, "pr", str(n), verdict(n))
        self.model.poll()
        with self.model.lock:
            self.model.filter_mode = "all"
        ui = make_ui(self.model)
        texts = ["".join(t for _, t in ln) for ln in ui.stats_lines(24)]
        for state in states:
            self.assertTrue(any(f"{state}=1" in t for t in texts), state)
        count_lines = [t for t in texts if "=1" in t]
        self.assertGreater(len(count_lines), 1)
        for t in count_lines:
            self.assertLessEqual(len(t.rstrip()), 24, t)

    def test_awaiting_approver_folds_into_merge_ready(self) -> None:
        self.db.push("merge-ready", "pr", "1", verdict(1))
        self.db.push("merge-ready", "pr", "2", verdict(2))
        self.db.push("awaiting-approver", "pr", "3", verdict(3))
        self.model.poll()
        ui = make_ui(self.model)
        joined = "\n".join("".join(t for _, t in ln)
                           for ln in ui.stats_lines(80))
        self.assertIn("merge-ready=2/1", joined)
        self.assertNotIn("awaiting-approver", joined)
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        approver_row = next(t for t in rows if "#3" in t)
        self.assertIn("merge-ready*", approver_row)
        self.assertNotIn("awaiting-approver", approver_row)


class PaintSmokeTests(DbCase):
    def test_paint_one_frame_headless(self) -> None:
        self.db.push("reviewed", "pr", "1",
                     verdict(1, msg="# Head\n**bold** and `code`",
                             title="hello title"))
        self.model.poll()
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui.ring.append("a log line")
        ui.ring.append("a warning line", tag=logging.WARNING)
        ui.paint()
        out = stream.getvalue()
        for expected in ("stats", "logs", "message", "#1", "a log line", "Head"):
            self.assertIn(expected, out)
        self.assertIn("\x1b[33ma warning line", out)

    def test_palette_covers_every_markdown_style(self) -> None:
        term = make_term()
        # "text" deliberately has no entry: it means unstyled.
        self.assertLessEqual(tui_core.MARKDOWN_STYLES - {"text"},
                             set(fairy_tui._styles(term)))

    def test_paint_strips_hostile_escape_sequences(self) -> None:
        self.db.push("reviewed", "pr", "2",
                     verdict(2, msg="body\x1b]0;pwned\x07text",
                             title="evil\x1b]0;pwned\x07title"))
        self.model.poll()
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui.ring.append("wrapper says \x1b]0;pwned\x07hi")
        ui.paint()
        out = stream.getvalue()
        self.assertNotIn("\x1b]0;", out)
        self.assertNotIn("\x07", out)

    def test_log_scrollback_stops_at_the_oldest_line(self) -> None:
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        for i in range(5):
            ui.ring.append(f"line{i}")
        ui.scroll["bl"] = 10_000
        ui.paint()
        out = stream.getvalue()
        self.assertLessEqual(ui.scroll["bl"], 5)
        self.assertIn("line0", out)

    def test_label_explanations_wrap_instead_of_clipping(self) -> None:
        reason = ("the reproduction requires a sample clip that the reporter "
                  "has not attached and cannot be synthesized locally")
        self.db.push("reviewed", "pr", "1", verdict(
            1, msg="msg", labels=[{"label": "needs sample", "op": "add",
                                   "reason": reason, "post": True}]))
        self.model.poll()
        ui = make_ui(self.model)
        lines = ui.detail_lines(32)
        text = " ".join("".join(t for _, t in ln) for ln in lines)
        for word in ("synthesized", "locally", "[posted]"):
            self.assertIn(word, text)
        self.assertTrue(all(
            sum(len(t) for _, t in ln) <= 32 for ln in lines))

    def test_divider_gutter_keeps_selection_clean(self) -> None:
        # A URL in a right pane must not sit directly against the "│"
        # divider: the terminal's own shift/double-click selection would
        # copy the divider with it. One blank gutter column separates them.
        ui = make_ui(self.model)
        buf: list = []
        ui._blit(buf, tui_core.Rect(10, 0, 20, 3), "br", ["https://x/y"])
        self.assertTrue(buf[1].endswith(" "))          # gutter after divider
        self.assertEqual(buf[2], "https://x/y" + " " * 8)  # 18 wide + right gutter

    def test_click_copies_url_via_osc52(self) -> None:
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui._clip_cmd = None  # pin the terminal-escape fallback path
        buf: list = []
        # pane narrower than the URL: the copy must still be whole
        ui._blit(buf, tui_core.Rect(10, 0, 16, 4), "br",
                 ["see https://ffmpeg.org/very/long now"])
        ui._copy_click("br", 10 + 1 + 6, 1)   # inside the URL, row 0
        ui._copy_click("br", 10 + 1 + 1, 1)   # on plain text: no-op
        payload = fairy_tui.base64.b64encode(
            b"https://ffmpeg.org/very/long").decode()
        self.assertEqual(stream.getvalue().count("\x1b]52;c;"), 1)
        self.assertIn(f"\x1b]52;c;{payload}\x07", stream.getvalue())

    def test_click_prefers_the_external_clipboard_helper(self) -> None:
        # xclip/wl-copy work in terminals without OSC 52 support (rxvt).
        stream = io.StringIO()
        ui = make_ui(self.model, stream)
        ui._clip_cmd = ["xclip", "-selection", "primary"]
        buf: list = []
        ui._blit(buf, tui_core.Rect(10, 0, 20, 4), "br", ["see 5144acb now"])
        with mock.patch.object(fairy_tui.subprocess, "run") as run:
            ui._copy_click("br", 10 + 1 + 5, 1)
        self.assertEqual(run.call_args.args[0], ["xclip", "-selection", "primary"])
        self.assertEqual(run.call_args.kwargs["input"], b"5144acb")
        self.assertNotIn("\x1b]52;", stream.getvalue())  # no fallback needed

    def test_clipboard_helper_targets_the_primary_selection(self) -> None:
        with mock.patch.dict(os.environ,
                             {"DISPLAY": ":0", "WAYLAND_DISPLAY": ""}), \
             mock.patch.object(fairy_tui.shutil, "which",
                               lambda n: "/usr/bin/xclip" if n == "xclip" else None):
            self.assertEqual(fairy_tui._clipboard_cmd(),
                             ["xclip", "-selection", "primary"])

    def test_export_failure_is_logged_not_fatal(self) -> None:
        ui = make_ui(self.model, save_dir=Path("/proc/no-such-dir"))
        with self.assertLogs(fairy_tui.logger, level="ERROR") as logs:
            ui.export(full=True)  # must not raise
        self.assertIn("export", logs.output[0])


if __name__ == "__main__":
    unittest.main()


class RequestedBadgeTests(DbCase):
    def test_a_pending_rerun_request_shows_on_the_row(self) -> None:
        """Production #20893: r created the request but the row was
        pixel-identical until the agent's next full scan."""
        self.db.push("skipped", "pr", "5", verdict(5, "skip"))
        self.model.poll()
        with self.model.lock:
            self.model.filter_mode = "all"
        ui = make_ui(self.model)
        self.db.request("pr", "5", {"action": "rerun"})
        self.model.poll()
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        self.assertTrue(any("requested" in t for t in rows), rows)
        self.db.try_pop("requests", "pr", "5")
        self.model.poll()
        rows = ["".join(t for _, t in r) if isinstance(r, list) else r[1]
                for r in ui.list_rows()]
        self.assertFalse(any("requested" in t for t in rows), rows)
