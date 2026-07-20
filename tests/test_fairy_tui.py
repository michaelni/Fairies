"""fairy_tui: headless model round-trips and a one-frame paint smoke."""

from __future__ import annotations

import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from threading import Thread
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import blessed  # noqa: E402
import fairy  # noqa: E402
import fairy_tui  # noqa: E402
import tui_core  # noqa: E402
import workset  # noqa: E402


def decision(n: int, action: str = "comment", msg: str = "msg") -> fairy.Decision:
    return fairy.Decision(n, "t", "a", "-", action, "llm", None, "reply", msg)


class ModelTests(unittest.TestCase):
    def test_ask_answer_roundtrip(self) -> None:
        model = fairy_tui.Model()
        model.add_candidates("PR", [{"number": 7, "title": "t", "html_url": "u"}])
        got: dict[str, str] = {}
        th = Thread(
            target=lambda: got.setdefault("choice", model.ask("PR", decision(7), "u")))
        th.start()
        for _ in range(500):
            with model.lock:
                if model.prompts:
                    break
            time.sleep(0.01)
        self.assertTrue(model.answer("apply"))
        th.join(timeout=2)
        self.assertEqual(got.get("choice"), "apply")
        self.assertIs(model.items[("PR", 7)].status, fairy_tui.Status.APPLIED)

    def test_answer_without_prompt_is_refused(self) -> None:
        model = fairy_tui.Model()
        model.add_candidates("PR", [{"number": 1, "title": "t"}])
        self.assertFalse(model.answer("apply"))

    def test_relevant_filter_hides_gate_skips(self) -> None:
        model = fairy_tui.Model()
        model.add_candidates("PR", [{"number": 1, "title": "a"},
                                    {"number": 2, "title": "b"}])
        model.finish("PR", decision(1))                       # actionable
        model.finish("PR", decision(2, action="skip", msg=""))  # gate skip
        with model.lock:
            self.assertEqual([it.number for it in model.visible()], [1])
            model.show_all = True
            self.assertEqual([it.number for it in model.visible()], [1, 2])

    def test_quit_answers_pending_prompts(self) -> None:
        model = fairy_tui.Model()
        got: dict[str, str] = {}
        th = Thread(
            target=lambda: got.setdefault("choice", model.ask("PR", decision(3), "")))
        th.start()
        for _ in range(500):
            with model.lock:
                if model.prompts:
                    break
            time.sleep(0.01)
        model.quit_all()
        th.join(timeout=2)
        self.assertEqual(got.get("choice"), "quit")


class WorksetDirCase(unittest.TestCase):
    """Base: a Model wired to a temp workset dir, plus a file writer."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.model = fairy_tui.Model()
        self.model.workset_dirs["PR"] = self.dir

    def _write(self, number: int, state: workset.WorkState,
               message: str = "m", mtime: float | None = None) -> Path:
        path = self.dir / f"pr-{number}.json"
        now = "2026-07-20T00:00:00+00:00"
        workset.save_item(path, workset.WorkItem(
            kind="pr", forge_type="gitea", account="", owner="o", repo="r",
            number=number, state=state, created_at=now, state_changed_at=now,
            title="from disk", html_url="u",
            review=workset.ReviewResult(classification="reply", message=message)
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
        item = self.model.items[("PR", 5)]
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
        self.model.add_candidates("PR", [{"number": 5, "title": "t"}])
        self._write(5, workset.WorkState.REVIEW, mtime=100.0)
        self.model.poll_workset()
        item = self.model.items[("PR", 5)]
        self.assertIs(item.status, fairy_tui.Status.IN_LLM)
        self.assertEqual(item.stage, "review")
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        self.assertEqual(item.stage, "")

    def test_unknown_disk_item_is_added_and_relevant(self) -> None:
        self._write(9, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        item = self.model.items[("PR", 9)]
        self.assertEqual(item.title, "from disk")
        self.assertIs(item.status, fairy_tui.Status.REVIEWED)
        with self.model.lock:
            self.assertEqual([it.number for it in self.model.visible()], [9])

    def test_deleted_file_cancels_in_pipeline_item(self) -> None:
        path = self._write(5, workset.WorkState.QUEUED)
        self.model.poll_workset()
        self.assertIs(self.model.items[("PR", 5)].status, fairy_tui.Status.QUEUED)
        path.unlink()
        self.model.poll_workset()
        self.assertIs(self.model.items[("PR", 5)].status,
                      fairy_tui.Status.CANCELLED)

    def test_operator_states_are_not_clobbered(self) -> None:
        self.model.add_candidates("PR", [{"number": 5, "title": "t"}])
        self.model.items[("PR", 5)].status = fairy_tui.Status.AWAITING
        self._write(5, workset.WorkState.REVIEWED)
        self.model.poll_workset()
        self.assertIs(self.model.items[("PR", 5)].status,
                      fairy_tui.Status.AWAITING)


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
        model.workset_dirs["PR"] = d
        model.poll_workset()

        def fake_call(cmd, **kw):
            Path(cmd[-1]).write_text("edited body", encoding="utf-8")
            return 0

        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40",
                                          "EDITOR": "myeditor"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            ui = fairy_tui.UILoop(term, model, tui_core.RingBuffer(), Path("."), ["PR"])
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
                "PR", [{"number": n, "title": "t"} for n in (1, 2, 3)])
            model.finish("PR", decision(1, action="skip", msg=""))
            model.finish("PR", decision(2))                       # actionable
            model.finish("PR", decision(3, action="skip", msg=""))
            model.show_all = True
            ui = fairy_tui.UILoop(
                term, model, tui_core.RingBuffer(), Path("."), ["PR"])

            model.cursor = 1                 # on #2 in the "all" view
            ui.dispatch(Key("a"))            # -> relevant view: only #2
            self.assertEqual(model.cursor, 0)
            with model.lock:
                self.assertEqual(model._cursor_key(), ("PR", 2))
            ui.dispatch(Key("a"))            # back to "all": still on #2
            with model.lock:
                self.assertEqual(model._cursor_key(), ("PR", 2))

            model.cursor = 2                 # on filtered-out #3
            ui.dispatch(Key("a"))            # nearest preceding visible: #2
            with model.lock:
                self.assertEqual(model._cursor_key(), ("PR", 2))


class PaintSmokeTests(unittest.TestCase):
    def test_paint_one_frame_headless(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            stream = io.StringIO()
            term = blessed.Terminal(
                kind="xterm-256color", stream=stream, force_styling=True)
            model = fairy_tui.Model()
            model.add_candidates("PR", [{"number": 1, "title": "hello title"}])
            model.finish("PR", decision(1, msg="# Head\n**bold** and `code`"))
            ring = tui_core.RingBuffer()
            ring.append("a debug line")
            ring.append("a warning line", tag=fairy_tui.logging.WARNING)
            ui = fairy_tui.UILoop(term, model, ring, Path("."), ["PR"])
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
                "PR", [{"number": 2, "title": "evil\x1b]0;pwned\x07title"}])
            model.finish("PR", decision(2, msg="body\x1b]0;pwned\x07text"))
            ring = tui_core.RingBuffer()
            ring.append("wrapper says \x1b]0;pwned\x07hi")
            ui = fairy_tui.UILoop(term, model, ring, Path("."), ["PR"])
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
                term, fairy_tui.Model(), ring, Path("."), ["PR"])
            ui.scroll["bl"] = 10_000
            ui.paint()
            out = stream.getvalue()
        self.assertLessEqual(ui.scroll["bl"], 5)
        self.assertIn("line0", out)

    def test_export_failure_is_logged_not_fatal(self) -> None:
        with mock.patch.dict(os.environ, {"COLUMNS": "100", "LINES": "40"}):
            term = blessed.Terminal(
                kind="xterm-256color", stream=io.StringIO(), force_styling=True)
            model = fairy_tui.Model()
            ui = fairy_tui.UILoop(
                term, model, tui_core.RingBuffer(),
                Path("/proc/no-such-dir"), ["PR"])
            with self.assertLogs(fairy_tui.logger, level="ERROR") as logs:
                ui.export(full=True)  # must not raise
        self.assertIn("export", logs.output[0])


if __name__ == "__main__":
    unittest.main()
