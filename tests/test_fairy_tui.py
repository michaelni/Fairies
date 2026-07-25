"""fairy_tui: headless model round-trips over a filedb and a paint smoke."""

from __future__ import annotations

import io
import logging
import os
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
        self.db.push("queued", "pr", 5, verdict(5))
        self.model.poll()
        item = self.model.items[(R1, "pr", 5)]
        self.assertEqual(item.state, "queued")
        self.assertEqual(item.data["title"], "from disk")
        self.db.move("queued", "llm", "pr", 5,
                     mutate=lambda d: d.update(stage="triage"))
        self.model.poll()
        self.assertEqual(item.state, "llm")
        self.assertEqual(item.data["stage"], "triage")

    def test_invalid_file_flags_the_row_and_recovers(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        path = self.db.path("reviewed", "pr", 5)
        path.write_text("{ broken", encoding="utf-8")
        os.utime(path, (200.0, 200.0))
        self.model.poll()
        item = self.model.items[(R1, "pr", 5)]
        self.assertEqual(item.state, fairy_tui.INVALID)
        self.assertTrue(item.error)
        self.assertEqual(item.data["review"]["message"], "m")  # last good parse
        self.assertIn((R1, 5), self.keys())
        self.db.push("reviewed", "pr", 5, verdict(5))  # operator fixed it
        self.model.poll()
        self.assertEqual(item.state, "reviewed")
        self.assertEqual(item.error, "")

    def test_removed_file_drops_the_row(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        self.db.pop("reviewed", "pr", 5)
        self.model.poll()
        self.assertNotIn((R1, "pr", 5), self.model.items)

    def test_crash_remnant_shows_the_later_state(self) -> None:
        self.db.push("queued", "pr", 5, verdict(5))
        self.db.push("posted", "pr", 5, verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "posted")

    def test_relevant_filter_hides_only_unseen_settled_rows(self) -> None:
        self.db.push("skipped", "pr", 1, verdict(1, "skip"))   # old backlog
        self.db.push("posted", "pr", 2, verdict(2))            # old backlog
        self.db.push("ci-blocked", "pr", 3, verdict(3))        # attention
        self.db.push("reviewed", "pr", 4, verdict(4))
        self.db.push("error", "pr", 6, {"error": "boom"})
        self.model.poll()
        self.assertEqual(self.keys(), [(R1, 3), (R1, 4), (R1, 6)])
        with self.model.lock:
            self.model.show_all = True
        self.assertEqual(self.keys(), [(R1, 1), (R1, 2), (R1, 3), (R1, 4), (R1, 6)])

    def test_a_row_seen_live_stays_listed_after_it_settles(self) -> None:
        # The operator watched this item head into the LLM and wants to
        # inspect why it skipped; without the session memory the row
        # would vanish the moment the verdict lands in skipped/.
        self.db.push("llm", "pr", 5, verdict(5))
        self.model.poll()
        self.db.move("llm", "skipped", "pr", 5)
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "skipped")
        self.assertIn((R1, 5), self.keys())


class DirMtimeGateTests(DbCase):
    """Unchanged state dirs are not re-listed: every filedb write lands
    by rename into its dir, so the dir mtime is the change signal."""

    def age_dirs(self, seconds: float = 10) -> None:
        for db in (self.db, self.db2):
            for state in filedb.STATES:
                t = time.time() - seconds
                os.utime(db.root / state, (t, t))

    def listing_calls(self) -> list[str]:
        calls: list[str] = []
        orig = filedb.Db.list_state
        with mock.patch.object(
                filedb.Db, "list_state", autospec=True,
                side_effect=lambda db, s: (calls.append(s), orig(db, s))[1]):
            self.model.poll()
        return calls

    def test_unchanged_aged_dirs_skip_the_rescan(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.age_dirs()
        self.model.poll()  # caches every (old enough) dir listing
        self.assertEqual(self.listing_calls(), [])
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "reviewed")

    def test_renames_are_seen_through_the_gate(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.age_dirs()
        self.model.poll()
        self.db.move("reviewed", "outgoing", "pr", 5)  # bumps both dirs
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "outgoing")

    def test_a_recently_modified_dir_is_never_trusted(self) -> None:
        # Coarse file timestamps: a rename in the same clock tick as the
        # scan can leave the dir mtime unchanged, so fresh mtimes must
        # not enter the cache.
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()  # dir mtimes are "now": nothing may be cached
        self.assertIn("reviewed", self.listing_calls())


class ActTests(DbCase):
    """y/s/x/r/f are file operations on the cursor row's db."""

    def test_apply_moves_reviewed_to_outgoing(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        self.model.act("apply")
        self.assertEqual(self.db.find("pr", 5), "outgoing")
        self.assertIn((R1, "pr", 5), self.model.acted)

    def test_apply_on_a_skip_verdict_says_nothing_to_post(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5, "skip"))
        self.model.poll()
        with self.assertLogs(fairy_tui.logger, level="INFO") as logs:
            self.model.act("apply")
        self.assertEqual(self.db.find("pr", 5), "reviewed")
        self.assertTrue(any("nothing to post" in ln for ln in logs.output))

    def test_apply_refused_while_a_worker_holds_the_item(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        claim = self.db.claim("reviewed", "reviewed", "pr", 5)
        try:
            self.model.act("apply")
        finally:
            claim.abort()
        self.assertEqual(self.db.find("pr", 5), "reviewed")

    def test_rerun_writes_a_request_and_respects_in_flight(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        self.model.act("rerun")
        self.assertEqual(self.db.get("requests", "pr", 5), mock.ANY)
        self.assertEqual(self.db.get("requests", "pr", 5)["action"], "rerun")
        self.db.pop("requests", "pr", 5)
        self.db.move("reviewed", "queued", "pr", 5)
        self.model.poll()
        self.model.act("rerun")
        self.assertIsNone(self.db.get("requests", "pr", 5))

    def test_skip_and_cancel_move_with_a_reason(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db.push("merge-ready", "pr", 6, {"title": "t"})
        self.model.poll()
        self.model.act("skip")
        self.assertEqual(self.db.get("skipped", "pr", 5)["reason"],
                         "operator skip")
        self.model.poll()
        with self.model.lock:
            keys = [(it.repo, it.kind, it.number) for it in self.model.visible()]
            self.model.cursor = keys.index((R1, "pr", 6))
        self.model.act("cancel")
        self.assertEqual(self.db.get("cancelled", "pr", 6)["reason"],
                         "operator cancel")

    def test_act_advances_to_the_next_reviewed_row(self) -> None:
        self.db.push("reviewed", "pr", 1, verdict(1))
        self.db.push("reviewed", "pr", 2, verdict(2))
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 1))
        self.model.act("apply")
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))

    def test_applied_row_stays_listed_through_posted(self) -> None:
        # After y the agent moves the file to posted/; the row must not
        # vanish, or the operator cannot verify the post landed.
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        self.model.act("apply")
        self.db.move("outgoing", "posted", "pr", 5)  # the agent's send pass
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "posted")
        self.assertIn((R1, 5), self.keys())


class MultiSideTests(DbCase):
    """N repos: items are keyed per repo and actions stay side-local."""

    def test_same_number_in_two_repos_is_two_rows(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db2.push("queued", "pr", 5, verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "reviewed")
        self.assertEqual(self.model.items[(R2, "pr", 5)].state, "queued")

    def test_act_works_on_the_cursor_rows_db(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db2.push("reviewed", "pr", 5, verdict(5))
        self.model.poll()
        with self.model.lock:
            keys = [(it.repo, it.kind, it.number) for it in self.model.visible()]
            self.model.cursor = keys.index((R2, "pr", 5))
        self.model.act("apply")
        self.assertEqual(self.db2.find("pr", 5), "outgoing")
        self.assertEqual(self.db.find("pr", 5), "reviewed")

    def test_stats_tile_one_block_per_repo(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db2.push("queued", "issue", 7, verdict(7))
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

    def test_repo_column_only_when_sides_span_repos(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db2.push("reviewed", "pr", 5, verdict(5))
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
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db2.push("queued", "pr", 7, verdict(7))
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
        self.db.push("reviewed", "pr", 5, verdict(5))
        self.db.push("queued", "issue", 5, verdict(5))
        self.model.poll()
        self.assertEqual(self.model.items[(R1, "pr", 5)].state, "reviewed")
        self.assertEqual(self.model.items[(R1, "issue", 5)].state, "queued")


class SideBuildTests(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        patcher = mock.patch.object(
            fairy_tui.agent, "db_root_for",
            side_effect=lambda ns: Path(tmp.name) / f"{ns.owner}~{ns.repo}")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_repeated_side_args_build_one_side_per_repo(self) -> None:
        args = fairy_tui.parse_args(
            ["--pr-args", "--owner a --repo x",
             "--pr-args", "--owner a --repo y",
             "--issue-args", "--owner a --repo x"])  # shares a/x's db
        sides, _ = fairy_tui.build_sides(args)
        self.assertEqual([label for label, _ in sides], ["a/x", "a/y"])

    def test_case_variant_repos_merge_into_one_side(self) -> None:
        # The forge routes owner/repo case-insensitively, so a case slip
        # must not become a second side over the same repo.
        args = fairy_tui.parse_args(["--pr-args", "--owner FFmpeg --repo web",
                                     "--pr-args", "--owner ffmpeg --repo Web"])
        sides, _ = fairy_tui.build_sides(args)
        self.assertEqual([label for label, _ in sides], ["FFmpeg/web"])

    def test_side_log_files_become_tails(self) -> None:
        # A side's --log-file is where its agent and worker write; the
        # UI tails it without a separate --tail flag.
        args = fairy_tui.parse_args(
            ["--pr-args", "--owner a --repo x --log-file logs/x.log",
             "--issue-args", "--owner a --repo x --log-file logs/x.log",
             "--pr-args", "--owner a --repo y --log-file logs/y.log",
             "--tail", "extra.log"])
        _, tails = fairy_tui.build_sides(args)
        self.assertEqual(tails, [Path("logs/x.log"), Path("logs/y.log"),
                                 Path("extra.log")])


class SortTests(DbCase):
    """t cycles the visible-list sort; sorts are stable over arrival."""

    def _numbers(self) -> list[int]:
        return [n for _, n in self.keys()]

    def test_status_sort_bubbles_actionable_rows_stably(self) -> None:
        for n, state in ((1, "queued"), (2, "reviewed"),
                         (3, "queued"), (4, "reviewed")):
            self.db.push(state, "pr", n, verdict(n))
        self.model.poll()
        self.assertEqual(self._numbers(), [1, 2, 3, 4])
        self.assertEqual(self.model.cycle_sort(), "status")
        # reviewed first; arrival order preserved within equal status.
        self.assertEqual(self._numbers(), [2, 4, 1, 3])

    def test_repo_and_number_modes(self) -> None:
        # "z/AA" sorts last by raw owner/repo but its displayed short
        # name "AA" sorts first: repo mode must follow the column.
        tmp3 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp3.cleanup)
        db3 = filedb.Db(Path(tmp3.name))
        self.model.sides.append(("z/AA", db3))
        self.db.push("queued", "pr", 5, verdict(5))
        self.db.push("queued", "pr", 9, verdict(9))
        self.db2.push("queued", "pr", 7, verdict(7))
        db3.push("queued", "pr", 3, verdict(3))
        self.model.poll()
        self.model.sort_mode = "repo"
        self.assertEqual(self.keys(),
                         [("z/AA", 3), (R1, 5), (R1, 9), (R2, 7)])
        self.model.sort_mode = "number"
        self.assertEqual(self._numbers(), [3, 5, 7, 9])

    def test_cursor_follows_its_item_when_a_poll_reorders(self) -> None:
        # A status flip elsewhere must not move the operator's selection:
        # y/s/x/r act on whatever the cursor points at, so a reorder
        # under a stationary index would hit a different row.
        self.db.push("queued", "pr", 1, verdict(1))
        self.db.push("reviewed", "pr", 2, verdict(2))
        self.model.poll()
        self.model.sort_mode = "status"
        with self.model.lock:
            self.model._move_cursor_to((R1, "pr", 2))
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))
        self.db.move("queued", "reviewed", "pr", 1)  # bubbles above #2
        self.model.poll()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))
            self.assertEqual(self.model.cursor, 1)

    def test_cycle_wraps_back_to_arrival(self) -> None:
        for expected in ("status", "repo", "number", "arrival"):
            self.assertEqual(self.model.cycle_sort(), expected)

    def test_every_state_has_a_sort_priority(self) -> None:
        # A filedb state missing from _SORT_STATES would KeyError on
        # the UI thread the first time t reaches status mode.
        self.assertEqual(set(fairy_tui._SORT_STATES),
                         set(filedb.STATES) | {fairy_tui.INVALID})

    def test_t_key_keeps_the_cursor_on_its_row_and_labels_the_bar(self) -> None:
        self.db.push("queued", "pr", 1, verdict(1))
        self.db.push("reviewed", "pr", 2, verdict(2))
        self.model.poll()
        stream = io.StringIO()
        ui = make_ui(self.model, stream, cols=160)
        self.model.cursor = 1            # on #2 in arrival order
        ui.dispatch(Key("t"))            # -> status sort: #2 is first
        self.assertEqual(self.model.sort_mode, "status")
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))
        self.assertEqual(self.model.cursor, 0)
        ui.paint()
        self.assertIn("sort:status", stream.getvalue())


class DetailTests(DbCase):
    """The detail pane renders everything from the ticket dict."""

    def _detail_text(self) -> str:
        ui = make_ui(self.model)
        with self.model.lock:
            return fairy_tui._plain(ui.detail_lines(100))

    def test_reviewed_ticket_renders_message_and_action(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5, msg="persisted body"))
        self.model.poll()
        text = self._detail_text()
        self.assertIn("persisted body", text)
        self.assertIn("state reviewed", text)
        self.assertIn("comment", text)  # rebuilt decision's action

    def test_error_ticket_shows_reason(self) -> None:
        self.db.push("error", "pr", 5, {"title": "t", "error": "LLM exploded"})
        self.model.poll()
        self.assertIn("error: LLM exploded", self._detail_text())

    def test_gate_ticket_shows_attention_context(self) -> None:
        self.db.push("ci-blocked", "pr", 5, {
            "title": "t", "reason": "ci red",
            "cancelled_ci_contexts": ["job1"], "blocked_ci_contexts": []})
        self.model.poll()
        text = self._detail_text()
        self.assertIn("reason ci red", text)
        self.assertIn("cancelled ci contexts: job1", text)

    def test_send_blocked_note_is_shown(self) -> None:
        self.db.push("reviewed", "pr", 5,
                     verdict(5, send_blocked="PR updated_at changed"))
        self.model.poll()
        self.assertIn("send blocked: PR updated_at changed",
                      self._detail_text())

    def test_invalid_file_shows_reason(self) -> None:
        self.db.path("reviewed", "pr", 5).write_text("{ broken",
                                                     encoding="utf-8")
        self.model.poll()
        self.assertIn("file invalid:", self._detail_text())


class EditReviewTests(DbCase):
    def test_o_key_round_trips_the_message_through_the_editor(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5, msg="original"))
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
        self.assertEqual(self.db.get("reviewed", "pr", 5)["review"]["message"],
                         "edited body")

    def test_edit_refused_while_a_worker_holds_the_item(self) -> None:
        self.db.push("reviewed", "pr", 5, verdict(5, msg="original"))
        self.model.poll()

        def fake_call(cmd, **kw):
            Path(cmd[-1]).write_text("edited body", encoding="utf-8")
            return 0

        ui = make_ui(self.model)
        claim = self.db.claim("reviewed", "reviewed", "pr", 5)
        try:
            with mock.patch.dict(os.environ, {"EDITOR": "e"}), \
                    mock.patch.object(fairy_tui.subprocess, "call",
                                      side_effect=fake_call):
                ui.edit_review()
        finally:
            claim.abort()
        self.assertEqual(self.db.get("reviewed", "pr", 5)["review"]["message"],
                         "original")


class FilterToggleTests(DbCase):
    def test_cursor_follows_selection_across_the_a_toggle(self) -> None:
        self.db.push("skipped", "pr", 1, verdict(1, "skip"))
        self.db.push("reviewed", "pr", 2, verdict(2))
        self.db.push("skipped", "pr", 3, verdict(3, "skip"))
        self.model.poll()
        self.model.show_all = True
        ui = make_ui(self.model)

        self.model.cursor = 1            # on #2 in the "all" view
        ui.dispatch(Key("a"))            # -> relevant view: only #2
        self.assertEqual(self.model.cursor, 0)
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))
        ui.dispatch(Key("a"))            # back to "all": still on #2
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))

        self.model.cursor = 2            # on filtered-out #3
        ui.dispatch(Key("a"))            # nearest preceding visible: #2
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (R1, "pr", 2))


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

    def test_needs_poll_triggers_an_immediate_refresh(self) -> None:
        import time
        ui = make_ui(self.model)
        ui._last_poll = time.monotonic()  # the 1s fallback is not due
        with mock.patch.object(self.model, "poll") as poll:
            ui._maybe_poll()
            poll.assert_not_called()
            ui.needs_poll.set()
            ui._maybe_poll()
            poll.assert_called_once()
        self.assertFalse(ui.needs_poll.is_set())


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
        self.assertEqual(tags["2026-07-24T10:00:00 I hello"], logging.INFO)
        self.assertEqual(tags["2026-07-24T10:00:01 W look out"],
                         logging.WARNING)

    def test_partial_line_is_withheld_until_complete(self) -> None:
        self.path.write_text("2026-07-24T10:00:00 I hel")
        self.tail.poll()
        self.assertEqual(self.lines(), [])
        with open(self.path, "a") as fh:
            fh.write("lo\n")
        self.tail.poll()
        self.assertEqual([t for _, t in self.lines()],
                         ["2026-07-24T10:00:00 I hello"])

    def test_truncation_restarts_from_the_top(self) -> None:
        self.path.write_text("2026-07-24T10:00:00 I one\n")
        self.tail.poll()
        self.path.write_text("2026-07-24T11:00:00 I 2\n")  # rotated, shorter
        self.tail.poll()
        self.assertIn("2026-07-24T11:00:00 I 2",
                      [t for _, t in self.lines()])


class StatsWrapTests(DbCase):
    def test_state_counts_wrap_to_the_pane_width(self) -> None:
        states = ("ci-blocked", "merge-ready", "awaiting-approver",
                  "posted", "skipped")
        for n, state in enumerate(states, 1):
            self.db.push(state, "pr", n, verdict(n))
        self.model.poll()
        with self.model.lock:
            self.model.filter_mode = "all"
        ui = make_ui(self.model)
        texts = ["".join(t for _, t in ln) for ln in ui.stats_lines(30)]
        for state in states:
            self.assertTrue(any(f"{state}=1" in t for t in texts), state)
        count_lines = [t for t in texts if "=1" in t]
        self.assertGreater(len(count_lines), 1)
        for t in count_lines:
            self.assertLessEqual(len(t.rstrip()), 30, t)


class PaintSmokeTests(DbCase):
    def test_paint_one_frame_headless(self) -> None:
        self.db.push("reviewed", "pr", 1,
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
        self.db.push("reviewed", "pr", 2,
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
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            model = fairy_tui.Model()
            model.add_candidates("PR", [{"number": 1, "title": "t"}])
            d = fairy.Decision(
                1, "t", "a", "-", "comment", "llm", None, "reply", "msg",
                label_changes=(fairy.LabelChange(
                    "needs sample", "add", reason, post=True),),
            )
            model.finish("PR", d)
            ui = fairy_tui.UILoop(term, model, tui_core.RingBuffer(),
                                  Path("."), ["PR"])
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
