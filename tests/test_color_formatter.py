"""Tests for ``common._ColorFormatter`` and the ``--color`` CLI plumbing.

We pin the per-level color choice and the wrap-and-reset shape so a
future tweak to the palette is intentional rather than accidental, and
so the ``--color={auto,always,never}`` selection wires through
``setup_logging`` correctly without spawning a real terminal.
"""

from __future__ import annotations

import argparse
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import common  # noqa: E402


def _record(level: int, msg: str = "hello") -> logging.LogRecord:
    record = logging.LogRecord(
        name="t", level=level, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    record.thread_prefix = "T "
    return record


class ColorFormatterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fmt = common._ColorFormatter(
            fmt='%(thread_prefix)s%(message)s',
        )

    def test_info_is_uncolored(self) -> None:
        # INFO is the visual baseline -- no escape sequences at all.
        out = self.fmt.format(_record(logging.INFO))
        self.assertEqual(out, "T hello")

    def test_debug_is_dim_gray(self) -> None:
        # ``2;90`` combines faint (``2``) with bright-black foreground
        # (``90``) so the debug shade is visibly darker than default
        # on every terminal -- bare faint is widely ignored, see the
        # comment on ``_LEVEL_COLOR`` for why both are required.
        out = self.fmt.format(_record(logging.DEBUG))
        self.assertTrue(out.startswith("\x1b[2;90m"))
        self.assertTrue(out.endswith("\x1b[0m"))

    def test_warning_is_bold_yellow(self) -> None:
        out = self.fmt.format(_record(logging.WARNING))
        self.assertTrue(out.startswith("\x1b[1;33m"))
        self.assertTrue(out.endswith("\x1b[0m"))

    def test_error_is_bold_red(self) -> None:
        out = self.fmt.format(_record(logging.ERROR))
        self.assertTrue(out.startswith("\x1b[1;31m"))
        self.assertTrue(out.endswith("\x1b[0m"))

    def test_critical_is_bold_red(self) -> None:
        # CRITICAL shares ERROR's red rather than introducing a third
        # color; the bot doesn't currently emit CRITICAL but we still
        # want a sensible color if some future code does.
        out = self.fmt.format(_record(logging.CRITICAL))
        self.assertTrue(out.startswith("\x1b[1;31m"))
        self.assertTrue(out.endswith("\x1b[0m"))

    def test_message_text_is_preserved_inside_wrapping(self) -> None:
        out = self.fmt.format(_record(logging.WARNING, "watch out"))
        # The formatted body must appear verbatim between the two
        # escape sequences -- no truncation, no extra wrapping.
        self.assertIn("T watch out", out)


class ColorModeWiringTests(unittest.TestCase):
    """``setup_logging(color=...)`` must pick the right formatter.

    We can't poke at the formatter directly (it's wrapped inside
    a private handler list) so we exercise the public API: install
    a fresh logger, log at WARNING, capture the rendered line, and
    check whether ANSI escapes are present.
    """

    def _captured_warning(self, *, color: str, isatty: bool) -> str:
        # Build a throwaway logger so we don't touch global state.
        target = logging.getLogger(f"_color_mode_test_{color}_{isatty}")
        target.handlers.clear()
        # ``setup_logging`` writes to ``sys.stderr`` directly via
        # ``StreamHandler(sys.stderr)``; we patch the module-level
        # ``sys.stderr`` lookup AND its ``isatty`` so the auto path
        # observes a deterministic value.
        import io
        fake_stderr = io.StringIO()
        fake_stderr.isatty = lambda: isatty  # type: ignore[method-assign]
        with patch.object(common.sys, "stderr", fake_stderr):
            common.setup_logging(target, False, color=color)
            target.warning("warned")
        return fake_stderr.getvalue()

    def test_always_forces_color_even_without_tty(self) -> None:
        out = self._captured_warning(color="always", isatty=False)
        self.assertIn("\x1b[1;33m", out)
        self.assertIn("\x1b[0m", out)

    def test_never_disables_color_even_on_tty(self) -> None:
        out = self._captured_warning(color="never", isatty=True)
        self.assertNotIn("\x1b[", out)

    def test_auto_off_for_non_tty(self) -> None:
        out = self._captured_warning(color="auto", isatty=False)
        self.assertNotIn("\x1b[", out)

    def test_auto_on_for_tty(self) -> None:
        # Make sure NO_COLOR / CLICOLOR_FORCE are unset while we test
        # the TTY-on path so the assertion can attribute the color to
        # the TTY check rather than to a stray env var.
        import os
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NO_COLOR", None)
            os.environ.pop("CLICOLOR_FORCE", None)
            out = self._captured_warning(color="auto", isatty=True)
        self.assertIn("\x1b[1;33m", out)

    def test_auto_with_clicolor_force_forces_color_without_tty(self) -> None:
        # CLICOLOR_FORCE set in the environment must enable color on
        # the ``auto`` path even when stderr is not a TTY -- this is
        # the env-var counterpart to ``--color always`` and is what
        # makes a whole subprocess pipeline pick up color from a
        # single shell-level setting.
        import os
        with patch.dict(os.environ, {"CLICOLOR_FORCE": "1"}, clear=False):
            out = self._captured_warning(color="auto", isatty=False)
        self.assertIn("\x1b[1;33m", out)

    def test_never_overrides_clicolor_force(self) -> None:
        # Explicit opt-out wins over the env-var force-on so users
        # can locally disable color without unsetting CLICOLOR_FORCE
        # in their whole shell.
        import os
        with patch.dict(os.environ, {"CLICOLOR_FORCE": "1"}, clear=False):
            out = self._captured_warning(color="never", isatty=True)
        self.assertNotIn("\x1b[", out)


class AddColorArgTests(unittest.TestCase):
    def test_default_is_auto(self) -> None:
        p = argparse.ArgumentParser()
        common.add_color_arg(p)
        self.assertEqual(p.parse_args([]).color, "auto")

    def test_choices_are_locked(self) -> None:
        p = argparse.ArgumentParser()
        common.add_color_arg(p)
        for value in ("auto", "always", "never"):
            self.assertEqual(p.parse_args(["--color", value]).color, value)
        # An unknown value must be rejected so a typo can't silently
        # disable color.
        with self.assertRaises(SystemExit):
            p.parse_args(["--color", "rainbow"])


class CustomHandlersTests(unittest.TestCase):
    def test_custom_handlers_replace_stream_handlers(self) -> None:
        records: list[logging.LogRecord] = []

        class Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        lg = logging.getLogger("test_setup_logging_custom_handlers")
        handler = Capture()
        common.setup_logging(lg, True, handlers=[handler])
        self.assertEqual(lg.handlers, [handler])
        self.assertFalse(lg.propagate)
        self.assertEqual(lg.level, logging.DEBUG)
        lg.info("hello")
        self.assertEqual(records[0].thread_prefix, "M ")


if __name__ == "__main__":
    unittest.main()
