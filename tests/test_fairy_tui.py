"""fairy_tui: headless model round-trips and a one-frame paint smoke."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from queue import SimpleQueue
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blessed  # noqa: E402
import fairy  # noqa: E402
import fairy_tui  # noqa: E402
import tui_core  # noqa: E402
import workset  # noqa: E402


PR = ("PR", "o/r")
PR2 = ("PR", "o/r2")


def decision(n: int, action: str = "comment", msg: str = "msg") -> fairy.Decision:
    return fairy.Decision(n, "t", "a", "-", action, "llm", None, "reply", msg)


def make_pipe() -> fairy_tui.Pipeline:
    return fairy_tui.Pipeline(SimpleQueue(), fairy.PendingCount(0), set(),
                              SimpleQueue())


class ModelTests(unittest.TestCase):
    def test_decide_holds_and_records(self) -> None:
        # The controller never blocks: decide records the decision and
        # returns "hold"; acting is the operator's move on the table.
        model = fairy_tui.Model()
        model.add_candidates(PR, [{"number": 7, "title": "t", "html_url": "u"}])
        ui = fairy_tui.SideUI(PR, model, set())
        self.assertEqual(ui.decide(None, decision(7), "u"), "hold")
        item = model.items[(*PR, 7)]
        self.assertIsNotNone(item.decision)
        self.assertIs(item.status, fairy_tui.Status.PENDING)  # file drives status

    def test_relevant_filter_hides_gate_skips(self) -> None:
        model = fairy_tui.Model()
        model.add_candidates(PR, [{"number": 1, "title": "a"},
                                    {"number": 2, "title": "b"}])
        model.finish(PR, decision(1))                       # actionable
        model.finish(PR, decision(2, action="skip", msg=""))  # gate skip
        with model.lock:
            self.assertEqual([it.number for it in model.visible()], [1])
            model.show_all = True
            self.assertEqual([it.number for it in model.visible()], [1, 2])

    def test_finish_marks_only_non_actionable_done(self) -> None:
        model = fairy_tui.Model()
        model.add_candidates(PR, [{"number": 1, "title": "a"},
                                    {"number": 2, "title": "b"}])
        model.finish(PR, decision(1, action="skip", msg=""))  # gate skip
        model.finish(PR, decision(2))                         # actionable
        self.assertIs(model.items[(*PR, 1)].status, fairy_tui.Status.DONE)
        # Actionable items keep their file-driven status so the operator
        # can still act on the row.
        self.assertIs(model.items[(*PR, 2)].status, fairy_tui.Status.PENDING)


class WorksetDirCase(unittest.TestCase):
    """Base: a Model wired to a temp workset dir, plus a file writer."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = fairy_tui.Model()
        self.model.workset_dirs[PR] = self.dir

    def _write(self, number: int, state: workset.WorkState,
               message: str = "m", classification: str = "reply",
               labels: list[workset.LabelChange] | None = None,
               mtime: float | None = None, d: Path | None = None,
               kind: str = "pr") -> Path:
        path = (d or self.dir) / f"{kind}-{number}.json"
        now = "2026-07-20T00:00:00+00:00"
        workset.save_item(path, workset.WorkItem(
            kind=kind, forge_type="gitea", account="", owner="o", repo="r",
            number=number, state=state, created_at=now, state_changed_at=now,
            title="from disk", html_url="u",
            review=workset.ReviewResult(
                classification=classification, message=message,
                label_changes=labels or [])
            if state >= workset.WorkState.REVIEWED else None,
        ))
        if mtime is not None:
            os.utime(path, (mtime, mtime))
        return path


class WorksetPollTests(WorksetDirCase):
    """poll_workset drives item state from the on-disk files."""

    def test_invalid_file_flags_the_row_and_recovers(self) -> None:
        path = self._write(5, workset.WorkState.REVIEWED, mtime=100.0)
        self.model.poll_workset()
        item = self.model.items[(*PR, 5)]
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        path.write_text("{ broken", encoding="utf-8")
        os.utime(path, (200.0, 200.0))
        self.model.poll_workset()
        self.assertIs(item.status, fairy_tui.Status.INVALID)
        self.assertTrue(item.ws_error)
        self.assertEqual(item.ws.review.message, "m")
        with self.model.lock:
            self.assertIn(5, [it.number for it in self.model.visible()])
        self._write(5, workset.WorkState.REVIEWED)  # operator fixed it
        self.model.poll_workset()
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        self.assertEqual(item.ws_error, "")

    def test_states_map_to_status_and_stage(self) -> None:
        self.model.add_candidates(PR, [{"number": 5, "title": "t"}])
        self._write(5, workset.WorkState.REVIEW, mtime=100.0)
        self.model.poll_workset()
        item = self.model.items[(*PR, 5)]
        self.assertIs(item.status, fairy_tui.Status.IN_LLM)
        self.assertEqual(item.stage, "review")
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        self.assertEqual(item.stage, "")

    def test_unknown_disk_item_is_added_and_relevant(self) -> None:
        self._write(9, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        item = self.model.items[(*PR, 9)]
        self.assertEqual(item.title, "from disk")
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        with self.model.lock:
            self.assertEqual([it.number for it in self.model.visible()], [9])

    def test_persisted_llm_skip_is_done_not_awaiting(self) -> None:
        # Regression: at startup every REVIEWED file counted as "awaiting
        # you", including LLM-skip bookkeeping, and the number shrank as
        # the gates re-finished them. Skips with nothing to post are done;
        # a skip carrying label changes still awaits the operator.
        self._write(5, workset.WorkState.REVIEWED, classification="skip",
                    message="only rewraps a comment")
        self._write(6, workset.WorkState.REVIEWED, classification="skip",
                    labels=[workset.LabelChange(label="needs docs", op="add")])
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status, fairy_tui.Status.DONE)
        self.assertIs(self.model.items[(*PR, 6)].status,
                      fairy_tui.Status.REVIEWED)

    def test_deleted_file_cancels_in_pipeline_item(self) -> None:
        path = self._write(5, workset.WorkState.QUEUED)
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status, fairy_tui.Status.QUEUED)
        path.unlink()
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status,
                      fairy_tui.Status.CANCELLED)

    def test_operator_states_are_not_clobbered(self) -> None:
        # APPLIED (just posted this run) must not be downgraded by a poll
        # that still sees the file in REVIEWED for a moment.
        self.model.add_candidates(PR, [{"number": 5, "title": "t"}])
        self.model.items[(*PR, 5)].status = fairy_tui.Status.APPLIED
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status,
                      fairy_tui.Status.APPLIED)


class ActTests(WorksetDirCase):
    """y/s/x/r route table actions for the cursor row to the controller."""

    def setUp(self) -> None:
        super().setUp()
        self.pipe = make_pipe()
        with self.model.lock:
            self.model.pipelines[PR] = self.pipe
            self.model.forced[PR] = set()

    def _actions(self) -> list:
        out = []
        while True:
            try:
                out.append(self.pipe.actions.get_nowait())
            except Exception:
                return out

    def test_apply_on_reviewed_row_is_routed(self) -> None:
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.model.act("apply")
        self.assertEqual(self._actions(), [(5, "apply")])

    def test_apply_without_reviewed_file_is_refused(self) -> None:
        self.model.add_candidates(PR, [{"number": 5, "title": "t"}])
        self.model.act("apply")
        self.assertEqual(self._actions(), [])

    def test_apply_on_llm_skip_says_nothing_to_post(self) -> None:
        self._write(5, workset.WorkState.REVIEWED, classification="skip")
        self.model.poll_workset()
        self.model.show_all = True
        with self.assertLogs(fairy_tui.logger, level="INFO") as logs:
            self.model.act("apply")
        self.assertEqual(self._actions(), [])
        self.assertTrue(any("nothing to post" in ln for ln in logs.output))

    def test_rerun_refused_while_evaluating(self) -> None:
        self._write(5, workset.WorkState.REVIEW)  # wrapper running
        self.model.poll_workset()
        self.model.act("rerun")
        self.assertEqual(self._actions(), [])

    def test_rerun_on_reviewed_row_is_routed(self) -> None:
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.model.act("rerun")
        self.assertEqual(self._actions(), [(5, "rerun")])

    def test_act_advances_to_the_next_reviewed_row(self) -> None:
        self._write(1, workset.WorkState.REVIEWED)
        self._write(2, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (*PR, 1))
        self.model.act("apply")
        with self.model.lock:
            self.assertEqual(self.model._cursor_key(), (*PR, 2))

    def test_x_cancels_pending_via_set_and_reviewed_via_action(self) -> None:
        self.model.add_candidates(PR, [{"number": 1, "title": "t"}])
        self.model.show_all = True  # pending rows live in the "all" view
        self.model.cancel()
        self.assertIn(1, self.pipe.cancelled)
        self.assertIs(self.model.items[(*PR, 1)].status,
                      fairy_tui.Status.CANCELLED)
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        with self.model.lock:
            self.model.cursor = [it.number for it in self.model.visible()].index(5)
        self.model.cancel()
        self.assertEqual(self._actions(), [(5, "cancel")])


class MultiSideTests(WorksetDirCase):
    """N sides: items are keyed per repo and actions stay side-local."""

    def setUp(self) -> None:
        super().setUp()
        tmp2 = tempfile.TemporaryDirectory()
        self.addCleanup(tmp2.cleanup)
        self.dir2 = Path(tmp2.name)
        self.model.workset_dirs[PR2] = self.dir2

    def test_same_number_in_two_repos_is_two_rows(self) -> None:
        self._write(5, workset.WorkState.REVIEWED)
        self._write(5, workset.WorkState.QUEUED, d=self.dir2)
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status,
                      fairy_tui.Status.REVIEWED)
        self.assertIs(self.model.items[(*PR2, 5)].status,
                      fairy_tui.Status.QUEUED)

    def test_act_routes_to_the_cursor_rows_side(self) -> None:
        pipe1, pipe2 = make_pipe(), make_pipe()
        with self.model.lock:
            self.model.pipelines[PR] = pipe1
            self.model.pipelines[PR2] = pipe2
            self.model.forced[PR] = set()
            self.model.forced[PR2] = set()
        self._write(5, workset.WorkState.REVIEWED)
        self._write(5, workset.WorkState.REVIEWED, d=self.dir2)
        self.model.poll_workset()
        with self.model.lock:
            keys = [(it.kind, it.repo, it.number) for it in self.model.visible()]
            self.model.cursor = keys.index((*PR2, 5))
        self.model.act("apply")
        self.assertEqual(pipe2.actions.get_nowait(), (5, "apply"))
        self.assertTrue(pipe1.actions.empty())

    def test_pr_and_issue_sides_share_one_dir(self) -> None:
        self.model.workset_dirs[("issue", "o/r")] = self.dir
        self._write(5, workset.WorkState.REVIEWED)
        self._write(5, workset.WorkState.QUEUED, kind="issue")
        self.model.poll_workset()
        self.assertIs(self.model.items[(*PR, 5)].status,
                      fairy_tui.Status.REVIEWED)
        self.assertIs(self.model.items[("issue", "o/r", 5)].status,
                      fairy_tui.Status.QUEUED)


class SideBuildTests(unittest.TestCase):
    def test_repeated_side_args_build_one_side_each(self) -> None:
        args = fairy_tui.parse_args(
            ["--pr-args", "--owner a --repo x",
             "--pr-args", "--owner a --repo y",
             "--issue-args", "--owner a --repo x"])
        self.assertEqual([s.key for s in fairy_tui.build_sides(args)],
                         [("PR", "a/x"), ("PR", "a/y"), ("issue", "a/x")])

    def test_duplicate_side_is_rejected(self) -> None:
        args = fairy_tui.parse_args(["--pr-args", "--owner a --repo x",
                                     "--pr-args", "--owner a --repo x"])
        with self.assertRaises(SystemExit):
            fairy_tui.build_sides(args)

    def test_case_variant_duplicate_side_is_rejected(self) -> None:
        # The forge routes owner/repo case-insensitively, so a case slip
        # must not slip past the duplicate check as a "different" repo.
        args = fairy_tui.parse_args(["--pr-args", "--owner FFmpeg --repo web",
                                     "--pr-args", "--owner ffmpeg --repo Web"])
        with self.assertRaises(SystemExit):
            fairy_tui.build_sides(args)


class DetailFromFileTests(WorksetDirCase):
    """The detail pane renders review content and error reasons from the
    workset file, even when no in-memory decision exists."""

    def _detail_text(self) -> str:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
        ui = fairy_tui.UILoop(term, self.model, tui_core.RingBuffer(), Path("."), [PR])
        with self.model.lock:
            return "\n".join(
                "".join(seg[1] for seg in line) for line in ui.detail_lines(100))

    def test_orphan_review_renders_message_from_file(self) -> None:
        self._write(5, workset.WorkState.REVIEWED, message="persisted body")
        self.model.poll_workset()
        text = self._detail_text()
        self.assertIn("persisted body", text)
        self.assertIn("status reviewed", text)

    def test_error_file_shows_reason(self) -> None:
        now = "2026-07-20T00:00:00+00:00"
        workset.save_item(self.dir / "pr-5.json", workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=5, state=workset.WorkState.ERROR,
            created_at=now, state_changed_at=now, title="t",
            error="LLM exploded",
        ))
        self.model.poll_workset()
        self.model.show_all = True
        self.assertIn("error: LLM exploded", self._detail_text())

    def test_invalid_file_shows_reason(self) -> None:
        (self.dir / "pr-5.json").write_text("{ broken", encoding="utf-8")
        self.model.poll_workset()
        self.assertIn("file invalid:", self._detail_text())


class EditReviewTests(unittest.TestCase):
    def test_o_key_round_trips_the_message_through_the_editor(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = Path(tmp.name)
        now = "2026-07-20T00:00:00+00:00"
        workset.save_item(d / "pr-5.json", workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=5, state=workset.WorkState.REVIEWED,
            created_at=now, state_changed_at=now, title="t",
            review=workset.ReviewResult(classification="reply", message="original"),
        ))
        model = fairy_tui.Model()
        model.workset_dirs[PR] = d
        model.poll_workset()

        def fake_call(cmd, **kw):
            Path(cmd[-1]).write_text("edited body", encoding="utf-8")
            return 0

        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40",
                                          "EDITOR": "myeditor"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            ui = fairy_tui.UILoop(term, model, tui_core.RingBuffer(), Path("."), [PR])
            with mock.patch.object(fairy_tui.subprocess, "call",
                                   side_effect=fake_call) as call:
                ui.edit_review()
        self.assertEqual(call.call_args.args[0][0], "myeditor")
        item = workset.load_item(d / "pr-5.json")
        self.assertEqual(item.review.message, "edited body")


class FilterToggleTests(unittest.TestCase):
    def test_cursor_follows_selection_across_the_a_toggle(self) -> None:
        class Key(str):
            name = None

        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            model = fairy_tui.Model()
            model.add_candidates(
                PR, [{"number": n, "title": "t"} for n in (1, 2, 3)])
            model.finish(PR, decision(1, action="skip", msg=""))
            model.finish(PR, decision(2))                       # actionable
            model.finish(PR, decision(3, action="skip", msg=""))
            model.show_all = True
            ui = fairy_tui.UILoop(
                term, model, tui_core.RingBuffer(), Path("."), [PR])

            model.cursor = 1                 # on #2 in the "all" view
            ui.dispatch(Key("a"))            # -> relevant view: only #2
            self.assertEqual(model.cursor, 0)
            with model.lock:
                self.assertEqual(model._cursor_key(), (*PR, 2))
            ui.dispatch(Key("a"))            # back to "all": still on #2
            with model.lock:
                self.assertEqual(model._cursor_key(), (*PR, 2))

            model.cursor = 2                 # on filtered-out #3
            ui.dispatch(Key("a"))            # nearest preceding visible: #2
            with model.lock:
                self.assertEqual(model._cursor_key(), (*PR, 2))


class PaintSmokeTests(unittest.TestCase):
    def test_paint_one_frame_headless(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            model = fairy_tui.Model()
            model.add_candidates(PR, [{"number": 1, "title": "hello title"}])
            model.finish(PR, decision(1, msg="# Head\n**bold** and `code`"))
            ring = tui_core.RingBuffer()
            ring.append("a debug line")
            ring.append("a warning line", tag=fairy_tui.logging.WARNING)
            ui = fairy_tui.UILoop(term, model, ring, Path("."), [PR])
            ui.paint()
            out = stream.getvalue()
        for expected in ("stats", "debug", "message", "#1", "a debug line", "Head"):
            self.assertIn(expected, out)
        self.assertIn("\x1b[33ma warning line", out)

    def test_palette_covers_every_markdown_style(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
        # "text" deliberately has no entry: it means unstyled.
        self.assertLessEqual(tui_core.MARKDOWN_STYLES - {"text"},
                             set(fairy_tui._styles(term)))

    def test_paint_strips_hostile_escape_sequences(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            model = fairy_tui.Model()
            model.add_candidates(
                PR, [{"number": 2, "title": "evil\x1b]0;pwned\x07title"}])
            model.finish(PR, decision(2, msg="body\x1b]0;pwned\x07text"))
            ring = tui_core.RingBuffer()
            ring.append("wrapper says \x1b]0;pwned\x07hi")
            ui = fairy_tui.UILoop(term, model, ring, Path("."), [PR])
            ui.paint()
            out = stream.getvalue()
        self.assertNotIn("\x1b]0;", out)
        self.assertNotIn("\x07", out)

    def test_debug_scrollback_stops_at_the_oldest_line(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            ring = tui_core.RingBuffer()
            for i in range(5):
                ring.append(f"line{i}")
            ui = fairy_tui.UILoop(
                term, fairy_tui.Model(), ring, Path("."), [PR])
            ui.scroll["bl"] = 10_000
            ui.paint()
            out = stream.getvalue()
        self.assertLessEqual(ui.scroll["bl"], 5)
        self.assertIn("line0", out)

    def test_divider_gutter_keeps_selection_clean(self) -> None:
        # A URL in a right pane must not sit directly against the "│"
        # divider: the terminal's own shift/double-click selection would
        # copy the divider with it. One blank gutter column separates them.
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            ui = fairy_tui.UILoop(term, fairy_tui.Model(),
                                  tui_core.RingBuffer(), Path("."), [PR])
            buf: list = []
            ui._blit(buf, tui_core.Rect(10, 0, 20, 3), "br", ["https://x/y"])
        self.assertTrue(buf[1].endswith(" "))          # gutter after divider
        self.assertEqual(buf[2], "https://x/y" + " " * 8)  # 18 wide + right gutter

    def test_click_copies_url_via_osc52(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            ui = fairy_tui.UILoop(term, fairy_tui.Model(),
                                  tui_core.RingBuffer(), Path("."), [PR])
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
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            ui = fairy_tui.UILoop(term, fairy_tui.Model(),
                                  tui_core.RingBuffer(), Path("."), [PR])
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
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            model = fairy_tui.Model()
            ui = fairy_tui.UILoop(
                term, model, tui_core.RingBuffer(),
                Path("/proc/no-such-dir"), [PR])
            with self.assertLogs(fairy_tui.logger, level="ERROR") as logs:
                ui.export(full=True)  # must not raise
        self.assertIn("export", logs.output[0])


if __name__ == "__main__":
    unittest.main()
