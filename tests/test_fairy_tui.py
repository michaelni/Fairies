"""fairy_tui: headless model round-trips and a one-frame paint smoke."""

from __future__ import annotations

import io
import os
import sys
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
            ui = fairy_tui.UILoop(term, model, ring, Path("."), ["PR"])
            ui.paint()
            out = stream.getvalue()
        for expected in ("stats", "debug", "message", "#1", "a debug line", "Head"):
            self.assertIn(expected, out)

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
