#!/usr/bin/env python3
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

Blessed 4-pane operator UI over the filedb ticket directories.

A pure view: the rows ARE the ticket files of every side's filedb (a
row's state is the directory its file sits in) and every operator
action is a file operation on the same db -- y moves a reviewed
verdict to outgoing/ for the agent's send pass, s/x move it to
skipped/cancelled/, r and f write a request the agent answers with a
fresh gate-bypassing ticket, o edits the persisted review message.
The agent and LLM workers are separate processes; the UI composes
with them but none of them needs it running. Panes: stats (top left),
ticket list (top right), merged tail of the processes' log files
(bottom left), rendered review message + label changes (bottom
right). Dividers move with the mouse.

What belongs here: everything terminal-facing -- blessed painting,
key/mouse dispatch, the directory poll and the log tail.

What does NOT belong: layout/scrollback/markdown logic (tui_core),
file atomicity (filedb), gates/posting policy (agent), review logic
(fairy, issue_fairy).
"""

from __future__ import annotations

import argparse
import base64
import faulthandler
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock

import blessed

import agent
import fairy
import filedb
import issue_fairy
import tui_core
from common import setup_logging, watch_paths

__all__ = ["main"]

logger = logging.getLogger(__name__)

# Row states are exactly the filedb directory names, plus "invalid"
# for a ticket file that no longer parses (broken hand-edit).
INVALID = "invalid"
# Being seen in one of these means work is happening on the row; such
# rows stay listed for the whole session so their outcome (posted,
# skipped, errored) remains inspectable after they settle.
LIVE_STATES = ("requests", "queued", "llm", "outgoing")
# What the relevant filter hides -- unless the row was acted on or
# seen live this session.
HIDDEN_SETTLED = ("posted", "skipped", "cancelled")
_KIND_DISP = {"pr": "PR", "issue": "issue"}
# status sort: operator-actionable rows first, then the live pipeline
# states, then attention, then the settled ones
_SORT_STATES = {s: i for i, s in enumerate((
    "reviewed", INVALID, "llm", "queued", "outgoing", "requests",
    "merge-ready", "ci-blocked", "awaiting-approver",
    "error", "posted", "skipped", "cancelled"))}
assert set(_SORT_STATES) == set(filedb.STATES) | {INVALID}, \
    "every filedb state needs a sort priority"
STATE_W = max(len(s) for s in _SORT_STATES)


@dataclass
class Item:
    repo: str            # side repo label, "owner/repo"
    kind: str            # filedb kind: "pr" | "issue"
    number: int | str  # ticket token; see filedb
    state: str           # filedb directory name, or "invalid"
    data: dict = field(default_factory=dict)  # last good ticket content
    error: str = ""      # why the current file fails to parse


def _repo_short(repo: str) -> str:
    """Display name for an "owner/repo" side label."""
    return repo.rsplit("/", 1)[-1]


SORT_MODES = ("arrival", "status", "repo", "number")
# the a key cycles these lenses: the default working view, then one
# per attention surface (review/merge/CI are different jobs), any of
# them, and everything
FILTER_MODES = ("relevant", "review", "merge", "ci", "actionable", "all")
_FILTER_STATES = {
    "review": ("reviewed", INVALID),
    "merge": ("merge-ready",),
    "ci": ("ci-blocked",),
    "actionable": ("reviewed", INVALID, "merge-ready", "ci-blocked",
                   "awaiting-approver", "error"),
}
_SORT_KEYS = {
    "status": lambda it: _SORT_STATES[it.state],
    # repo mode orders by the short name the list column displays, or
    # rows would look unsorted whenever owners differ.
    "repo": lambda it: (_repo_short(it.repo).casefold(), it.repo, it.kind),
    # tokens sort with their base item, samples/reviews after it
    "number": lambda it: (filedb.forge_number(it.number), str(it.number)),
}


class Model:
    """Shared state between the 1 Hz directory poll and the painter.
    Every mutation happens under ``lock``; ``dirty`` wakes the painter.

    A side is a repo label plus its filedb; items are keyed by
    ``(repo, kind, number)`` -- two repos can share a PR number."""

    def __init__(self, sides: list[tuple[str, filedb.Db]]) -> None:
        self.lock = Lock()
        self.dirty = Event()
        self.sides = sides
        self.items: dict[tuple[str, str, int], Item] = {}
        self.order: list[tuple[str, str, int]] = []
        self._read: dict[tuple[str, str, int], tuple[Path, float]] = {}
        # (repo, state) -> (dir mtime, listing): an unchanged state dir
        # is not re-listed and its files are not re-stat'ed.
        self._dirs: dict[tuple[str, str], tuple[float, list[tuple[str, int]]]] = {}
        self.filter_mode = FILTER_MODES[0]
        self.sort_mode = SORT_MODES[0]
        # Rows the operator acted on (y/s/x) and rows seen in a live
        # state this session: they stay listed after settling so the
        # outcome is verifiable and "why did this one skip" has an
        # answer on screen.
        self.acted: set[tuple[str, str, int]] = set()
        self.seen_live: set[tuple[str, str, int]] = set()
        self.cursor = 0
        self.quit_flag = False
        self.started = time.monotonic()

    def db(self, item: Item) -> filedb.Db:
        return dict(self.sides)[item.repo]

    def poll(self) -> None:
        """Rescan every side's state directories: the files are the whole
        truth -- agent, workers and operator hand-edits all land here.
        File IO happens outside ``lock``.

        Every filedb write lands by rename INTO its state directory
        (content rewrites included), so a directory whose mtime has not
        moved needs no re-listing and none of its files re-stat'ed:
        steady state costs a dozen directory stats per side, not one
        stat per ticket."""
        now = time.time()
        found: dict[tuple[str, str, int], tuple[str, filedb.Db, bool]] = {}
        for repo, db in self.sides:
            for state in filedb.STATES:
                try:
                    dir_mtime = (db.root / state).stat().st_mtime
                except OSError:
                    self._dirs.pop((repo, state), None)
                    continue
                cached = self._dirs.get((repo, state))
                rescanned = cached is None or cached[0] != dir_mtime
                if rescanned:
                    listing = db.list_state(state)
                    # Linux file timestamps come from the coarse clock: a
                    # rename in the same tick as this scan could leave the
                    # mtime unchanged, so a just-modified directory is
                    # never trusted as clean.
                    if dir_mtime < now - 2.0:
                        self._dirs[(repo, state)] = (dir_mtime, listing)
                    else:
                        self._dirs.pop((repo, state), None)
                else:
                    listing = cached[1]
                for kind, number in listing:
                    # later directory wins: crash-remnant precedence
                    found[(repo, kind, number)] = (state, db, rescanned)
        updates: list[tuple[tuple[str, str, int], str, dict | None, str]] = []
        for key in sorted(found):
            state, db, rescanned = found[key]
            item = self.items.get(key)
            if not rescanned and item is not None and item.state == state:
                continue  # unchanged dir => unchanged files inside it
            path = db.path(state, key[1], key[2])
            try:
                tag = (path, path.stat().st_mtime)
            except OSError:
                continue  # racing a rename; the next poll sees the new dir
            if self._read.get(key) == tag:
                continue
            try:
                updates.append((key, state, json.loads(
                    path.read_text(encoding="utf-8")), ""))
            except (FileNotFoundError, IsADirectoryError):
                continue
            except (OSError, ValueError) as exc:
                updates.append((key, state, None, str(exc)))
            self._read[key] = tag
        removed = [k for k in self.items if k not in found]
        if not updates and not removed:
            return
        with self.lock, self._cursor_anchored():
            for key, state, data, error in updates:
                item = self.items.get(key)
                if item is None:
                    item = Item(*key, state)
                    self.items[key] = item
                    self.order.append(key)
                if error:
                    item.state, item.error = INVALID, error
                else:
                    item.state, item.error, item.data = state, "", data
                if state in LIVE_STATES:
                    self.seen_live.add(key)
            for key in removed:  # pruned, or a consumed request
                del self.items[key]
                self.order.remove(key)
                self._read.pop(key, None)
        self.dirty.set()

    # ---- UI-thread side ----

    def visible(self) -> list[Item]:
        """Items for the list pane, honoring the all/relevant filter and
        the sort mode (stable, so arrival order breaks ties). Caller
        holds ``lock``."""
        items = [self.items[k] for k in self.order]
        if self.filter_mode == "relevant":
            items = [it for it in items if self._relevant(it)]
        elif self.filter_mode != "all":
            allowed = _FILTER_STATES[self.filter_mode]
            items = [it for it in items if it.state in allowed]
        if self.sort_mode != "arrival":
            items.sort(key=_SORT_KEYS[self.sort_mode])
        return items

    def cycle_sort(self) -> str:
        self.sort_mode = SORT_MODES[
            (SORT_MODES.index(self.sort_mode) + 1) % len(SORT_MODES)]
        return self.sort_mode

    def cycle_filter(self) -> str:
        self.filter_mode = FILTER_MODES[
            (FILTER_MODES.index(self.filter_mode) + 1) % len(FILTER_MODES)]
        return self.filter_mode

    def act(self, action: str) -> None:
        """Execute a table action on the cursor row as a file operation:
        apply = reviewed -> outgoing (the agent's send pass posts it),
        skip/cancel = -> skipped/cancelled, rerun/force = a request the
        agent answers with a fresh gate-bypassing ticket. try_move never
        blocks: a row a worker holds, or one that changed under the
        cursor, refuses with a log line instead. No cursor anchor here:
        after y the cursor jumps to the next reviewed row instead of
        following the acted one."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key) if key else None
            if item is None:
                return
            db = self.db(item)
            label = f"{_KIND_DISP[item.kind]} {item.repo}#{item.number}"
            if action == "apply" and not filedb.is_base(item.number):
                logger.info("%s is a sample/review evaluation: post the "
                            "base ticket, or mv it to outgoing/ to force",
                            label)
                return
            if action in ("rerun", "force"):
                if item.state in ("queued", "llm", "outgoing"):
                    logger.info("%s is already in flight", label)
                    return
                db.request(item.kind, item.number, {"action": "rerun"})
                self.seen_live.add(key)
                logger.info("requested a fresh review of %s", label)
            elif action == "apply":
                if item.state != "reviewed" or not agent.postable(
                        agent.ticket_decision(item.kind, item.number, item.data)):
                    logger.info("%s has nothing to post (state %s, llm %s)",
                                label, item.state,
                                (item.data.get("review") or {}).get(
                                    "classification", "-"))
                    return
                if not db.try_move("reviewed", "outgoing", item.kind, item.number):
                    logger.info("%s changed under the cursor; not applied", label)
                    return
                self.acted.add(key)
                logger.info("%s -> outgoing/ (the agent's send pass posts it)",
                            label)
                self._advance_to_reviewed(key)
            elif action in ("skip", "cancel"):
                dst = "skipped" if action == "skip" else "cancelled"
                if item.state in ("llm", INVALID) or item.state in HIDDEN_SETTLED:
                    logger.info("cannot %s %s in state %s%s", action, label,
                                item.state,
                                " (fix or delete the file by hand)"
                                if item.state == INVALID else "")
                    return
                note = {"reason": "operator " + action}
                if action == "skip":  # a real snooze even on old verdicts
                    note["snoozed_at"] = datetime.now(timezone.utc).isoformat()
                if not db.try_move(item.state, dst, item.kind, item.number,
                                   mutate=lambda d: d.update(note)):
                    logger.info("%s is busy or changed; not %sed", label, action)
                    return
                self.acted.add(key)
                logger.info("%s -> %s/", label, dst)
        self.dirty.set()

    def quit_all(self) -> None:
        with self.lock:
            self.quit_flag = True
        self.dirty.set()

    # ---- internals (caller holds ``lock``) ----

    def _advance_to_reviewed(self, key: tuple[str, str, int]) -> None:
        """Jump to the next reviewed row waiting for the operator: the
        first at/after the cursor, wrapping to the first overall."""
        remaining = {(it.repo, it.kind, it.number) for it in self.visible()
                     if it.state == "reviewed"} - {key}
        if not remaining:
            return
        keys = [(it.repo, it.kind, it.number) for it in self.visible()]
        nxt = next((k for k in keys[self.cursor:] if k in remaining),
                   next((k for k in keys if k in remaining), None))
        if nxt is not None:
            self._move_cursor_to(nxt)

    def _relevant(self, item: Item) -> bool:
        key = (item.repo, item.kind, item.number)
        return (item.state not in HIDDEN_SETTLED
                or key in self.acted or key in self.seen_live)

    def _cursor_key(self) -> tuple[str, str, int] | None:
        vis = self.visible()
        if not vis:
            return None
        self.cursor = max(0, min(self.cursor, len(vis) - 1))
        it = vis[self.cursor]
        return (it.repo, it.kind, it.number)

    def _move_cursor_to(self, key: tuple[str, str, int]) -> None:
        """Cursor onto ``key``; if the filter hides it, onto the visible
        item nearest before it in arrival order."""
        keys = [(it.repo, it.kind, it.number) for it in self.visible()]
        order_pos = {k: i for i, k in enumerate(self.order)}
        pos = order_pos.get(key)
        if pos is None:
            return
        self.cursor = 0
        best = -1
        for i, k in enumerate(keys):
            if best < order_pos[k] <= pos:
                best = order_pos[k]
                self.cursor = i

    @contextmanager
    def _cursor_anchored(self):
        """Keep the cursor on its item across a mutation: any status or
        membership change can reorder the sorted visible list, and a
        bare index would silently land on a different row. Caller holds
        ``lock``."""
        key = self._cursor_key()
        try:
            yield
        finally:
            if key is not None:
                self._move_cursor_to(key)


class OutputSink:
    """Fan-in for every captured line: the debug-pane ring buffer, an
    optional tee file, and the painter's dirty event."""

    def __init__(self, ring: tui_core.RingBuffer, dirty: Event, path: Path | None) -> None:
        self.ring = ring
        self.dirty = dirty
        self._lock = Lock()
        self._fh = open(path, "a", encoding="utf-8") if path else None

    def line(self, text: str, level: int | None = None) -> None:
        for ln in text.splitlines() or [""]:
            self.ring.append(ln, level)
            if self._fh is not None:
                with self._lock:
                    self._fh.write(ln + "\n")
                    self._fh.flush()
        self.dirty.set()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()


class RingLogHandler(logging.Handler):
    def __init__(self, sink: OutputSink) -> None:
        super().__init__()
        self.sink = sink
        self.setFormatter(logging.Formatter(
            '%(asctime)s %(message)s', '%Y-%m-%dT%H:%M:%S',
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.line(self.format(record), record.levelno)
        except Exception:
            pass  # never let UI logging kill the loop


_TAIL_LEVELS = {"D": logging.DEBUG, "I": logging.INFO, "W": logging.WARNING,
                "E": logging.ERROR, "C": logging.CRITICAL}


def _line_level(line: str) -> int | None:
    """Level of an ``ISO8601 L message`` log line (add_file_log's shape),
    None for anything else."""
    parts = line.split(" ", 2)
    if len(parts) > 2 and len(parts[1]) == 1:
        return _TAIL_LEVELS.get(parts[1])
    return None


class LogTail:
    """Merge appended lines of the agent/worker log files into the
    debug ring; the processes run and log independently, the UI only
    watches. A file may not exist yet (process not started): retried
    every poll. Truncation/rotation restarts from the top."""

    TAIL_BYTES = 64 * 1024  # first sight: recent end only, not months of log

    def __init__(self, paths: list[Path], sink: OutputSink) -> None:
        self.paths = paths
        self.sink = sink
        self._pos: dict[Path, int] = {}

    def poll(self) -> None:
        for path in self.paths:
            try:
                size = path.stat().st_size
                pos = self._pos.get(path)
                if pos is None:
                    pos = max(0, size - self.TAIL_BYTES)
                if size < pos:
                    pos = 0
                if size == pos:
                    continue
                with open(path, "rb") as fh:
                    fh.seek(pos)
                    chunk = fh.read()
            except OSError:
                continue
            cut = chunk.rfind(b"\n")
            if cut < 0:
                self._pos[path] = pos  # no complete line yet
                continue
            self._pos[path] = pos + cut + 1
            for ln in chunk[:cut].decode("utf-8", "replace").splitlines():
                self.sink.line(ln, _line_level(ln))


class StreamToRing:
    """File-like stand-in for sys.stdout/sys.stderr while the UI owns the
    screen; complete lines (stray prints) go to the sink."""

    def __init__(self, sink: OutputSink) -> None:
        self.sink = sink
        self._buf = ""
        self._lock = Lock()

    def write(self, s: str) -> int:
        with self._lock:
            self._buf += s
            *lines, self._buf = self._buf.split("\n")
        for ln in lines:
            self.sink.line(ln)
        return len(s)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


@contextmanager
def captured_output(sink: OutputSink):
    saved = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = StreamToRing(sink)
    try:
        yield
    finally:
        sys.stdout, sys.stderr = saved


MIN_TEXT_W = 8       # narrowest width a pane renders text at
EXPORT_FULL_W = 200  # E: full exports reflow at this fixed width
PANES = {"tl": "stats", "tr": "list", "bl": "logs", "br": "message"}
PANE_GLYPHS = {"tl": "Σ", "tr": "☰", "bl": "≣", "br": "¶"}
FOCUS_ORDER = ("tl", "tr", "bl", "br")
ACTION_KEYS = {"y": "apply", "s": "skip", "r": "rerun", "f": "force",
               "x": "cancel"}
KEYMAP = (("q", "quit"), ("y", "apply"), ("s", "skip"), ("r", "rerun"),
          ("f", "force"), ("x", "drop"), ("o", "edit msg"),
          ("a", "filter"), ("t", "sort"), ("e/E", "export"),
          ("Tab/click", "focus"), ("↑↓ PgUp/PgDn", "scroll"))


def _styles(t: blessed.Terminal) -> dict:
    """Style-name -> callable(text). 256-color palette, degrading to the
    16-color attribute names on lesser terminals; empty (= everything
    plain) when the terminal does no styling at all."""
    if not t.does_styling:
        return {}

    def strike(s: str) -> str:
        return f"\x1b[9m{s}\x1b[29m"

    def mix(*fns):
        def go(s: str) -> str:
            for fn in reversed(fns):
                s = fn(s)
            return s
        return go

    if t.number_of_colors >= 256:
        c, on = t.color, t.on_color
        return {
            "h1": mix(t.bold, c(212)),        "h2": mix(t.bold, c(141)),
            "h3": mix(t.bold, c(75)),         "h4": mix(t.bold, c(73)),
            "bold": t.bold,                   "italic": t.italic,
            "bold_italic": mix(t.bold, t.italic), "strike": strike,
            "code": mix(c(203), on(236)),     "codeblock": mix(c(252), on(235)),
            "codeblock_lang": c(244),
            "quote": mix(t.italic, c(246)),   "quote_bar": c(141),
            "bullet": c(212),
            "checkbox_on": c(78),             "checkbox_off": c(244),
            "link": mix(t.underline, c(75)),  "url": c(244),
            "hr": c(240),
            "table_border": c(240),           "th": mix(t.bold, c(223)),
            "divider": c(240),                "divider_drag": mix(t.bold, c(81)),
            "bar_focus": mix(t.bold, c(231), on(25)), "bar_blur": mix(c(245), on(236)),
            "key": mix(t.bold, c(81)),        "label": c(245),
            "num": t.bold,                    "mark": mix(t.bold, c(203)),
            "kind_pr": c(75),                 "kind_issue": c(176),
            "llm": c(141),                    "title": c(252),
            "st_requests": c(244),
            "st_queued": c(117),              "st_llm": mix(t.bold, c(45)),
            "st_reviewed": mix(t.bold, c(214)),
            "st_outgoing": mix(t.bold, c(81)),
            "st_posted": c(78),               "st_skipped": c(244),
            "st_cancelled": c(167),           "st_error": mix(t.bold, c(203)),
            "st_invalid": mix(t.bold, c(196)),
            "st_ci-blocked": c(209),          "st_merge-ready": mix(t.bold, c(78)),
            "st_awaiting-approver": c(179),
            "cursor": t.reverse,
            # log-pane levels; palette mirrors common._ColorFormatter
            "log_debug": t.dim_bright_black,  "log_warn": t.bold_yellow,
            "log_err": t.bold_red,
        }
    return {
        "h1": t.bold_magenta,   "h2": t.bold_blue,  "h3": t.bold_cyan,
        "h4": t.cyan,           "bold": t.bold,     "italic": t.italic,
        "bold_italic": t.bold,  "strike": strike,
        "code": t.reverse,      "codeblock": t.reverse,
        "codeblock_lang": t.bright_black,
        "quote": t.bright_black, "quote_bar": t.magenta,
        "bullet": t.bold,
        "checkbox_on": t.green, "checkbox_off": t.bright_black,
        "link": t.underline_blue, "url": t.bright_black,
        "hr": t.bright_black,
        "table_border": t.bright_black, "th": t.bold,
        "divider": t.bright_black, "divider_drag": t.bold_cyan,
        "bar_focus": t.reverse, "bar_blur": t.underline,
        "key": t.bold_cyan,     "label": t.bright_black,
        "num": t.bold,          "mark": t.bold_red,
        "kind_pr": t.cyan,      "kind_issue": t.magenta,
        "llm": t.magenta,       "title": t.white,
        "st_requests": t.bright_black,
        "st_queued": t.cyan,    "st_llm": t.bold_cyan,
        "st_reviewed": t.bold_yellow,
        "st_outgoing": t.bold_cyan,
        "st_posted": t.green,   "st_skipped": t.bright_black,
        "st_cancelled": t.red,  "st_error": t.bold_red,
        "st_invalid": t.bold_red,
        "st_ci-blocked": t.red, "st_merge-ready": t.bold_green,
        "st_awaiting-approver": t.yellow,
        "cursor": t.reverse,
        "log_debug": t.dim_bright_black, "log_warn": t.bold_yellow,
        "log_err": t.bold_red,
    }


def _clipboard_cmd() -> list[str] | None:
    """External clipboard helper when a display is reachable. Unlike
    OSC 52 this works with every terminal (rxvt has no OSC 52), and over
    ssh -X/-Y the forwarded connection carries the clipboard home."""
    if os.environ.get("WAYLAND_DISPLAY") and shutil.which("wl-copy"):
        return ["wl-copy", "--primary"]
    if os.environ.get("DISPLAY"):
        # PRIMARY, not CLIPBOARD: a mouse click is a selection in Unix
        # terms -- it must middle-click-paste, and must never clobber an
        # explicitly Ctrl-C'd clipboard.
        if shutil.which("xclip"):
            return ["xclip", "-selection", "primary"]
        if shutil.which("xsel"):
            return ["xsel", "-ip"]
    return None


def _plain(lines: list) -> str:
    """Pane content back to text for exports. Lines are the same three
    shapes _blit paints: plain strings, one (style, text) row, or a
    StyledLine segment list."""
    def one(line) -> str:
        if isinstance(line, str):
            return line
        if line and isinstance(line[0], str):
            return line[1]
        return "".join(t for _, t in line)
    return "\n".join(one(x) for x in lines)


class UILoop:
    def __init__(self, term: blessed.Terminal, model: Model, ring: tui_core.RingBuffer,
                 save_dir: Path, tail: LogTail) -> None:
        self.term = term
        self.model = model
        self.ring = ring
        self.save_dir = save_dir
        self.tail = tail
        repos = [repo for repo, _ in model.sides]
        short = [_repo_short(r) for r in repos]
        self._repo_disp = {r: (s if short.count(s) == 1 else r)
                           for r, s in zip(repos, short)}
        # 0 hides every repo label: labels only disambiguate when the
        # sides span more than one repo.
        self._repo_w = (max(len(d) for d in self._repo_disp.values())
                        if len(repos) > 1 else 0)
        self.layout = tui_core.GridLayout()
        self.focus = "tr"
        self.scroll = {"tl": 0, "bl": 0, "br": 0}
        self.list_top = 0
        self.drag: str | None = None
        self._last_size: tuple[int, int] | None = None
        self._last_paint = 0.0
        # pane -> (rect, gutter, unclipped plain rows) as last painted;
        # click-to-copy resolves the token under the mouse from this.
        self._shown: dict[str, tuple[tui_core.Rect, int, list[str]]] = {}
        self._clip_cmd = _clipboard_cmd()
        self.styles = _styles(term)
        self.needs_poll = Event()  # set by the filesystem watcher
        self._last_poll = 0.0
        self._build_status()

    def _build_status(self) -> None:
        """(Re)build the bottom-bar legend; the ``t`` entry names the
        current sort mode."""
        key = self.styles.get("key") or (lambda s: s)
        label = self.styles.get("label") or (lambda s: s)
        self._status_mode = (self.model.sort_mode, self.model.filter_mode)
        pairs = [(k, f"sort:{self.model.sort_mode}" if k == "t"
                  else f"filter:{self.model.filter_mode}" if k == "a" else d)
                 for k, d in KEYMAP]
        self._status_plain = "  ".join(f"{k} {d}" for k, d in pairs)
        self._status_styled = "  ".join(f"{key(k)} {label(d)}" for k, d in pairs)

    # ---- pane content (caller holds model.lock) ----

    def stats_lines(self, width: int) -> list[tui_core.StyledLine]:
        """Full-width header, then one block per repo tiled into as many
        columns as ``width`` fits. The counts are simply the number of
        files per state directory."""
        m = self.model
        items = [m.items[k] for k in m.order]
        actionable = sum(1 for it in items if it.state == "reviewed")
        header: tui_core.StyledLine = [
            ("label", "elapsed "),
            ("num", f"{int(time.monotonic() - m.started)}s"),
            ("label", "   reviewed, awaiting you "),
            ("st_reviewed" if actionable else "num", str(actionable)),
        ]
        blocks: list[list[tui_core.StyledLine]] = []
        for repo, _db in m.sides:
            group = [it for it in items if it.repo == repo]
            kinds = Counter(it.kind for it in group)
            block: list[tui_core.StyledLine] = [[
                ("title", f"{self._repo_disp[repo]}  "),
                ("kind_pr", f"{kinds['pr']} PRs"),
                ("text", "  "),
                ("kind_issue", f"{kinds['issue']} issues"),
            ]]
            by = Counter(it.state for it in group)
            if by:
                row: tui_core.StyledLine = [("text", "  ")]
                used = 2
                for s in (*filedb.STATES, INVALID):
                    if not by[s]:
                        continue
                    part_len = len(s) + 1 + len(str(by[s])) + 2
                    if used > 2 and used + part_len > width:
                        block.append(row)
                        row = [("text", "  ")]
                        used = 2
                    row += [(f"st_{s}", f"{s}="),
                            ("num", str(by[s])), ("text", "  ")]
                    used += part_len
                block.append(row)
            # what could be applied right now, by action -- distinct
            # from the review/investigate rows
            ready = Counter(it.data.get("action") for it in group
                            if it.state == "reviewed"
                            and it.data.get("action") in fairy.ACTIONABLE_DECISIONS)
            if ready:
                block.append([("text", "  "), ("label", "awaiting you: "),
                              ("st_reviewed", ", ".join(
                                  f"{k}={v}" for k, v in sorted(ready.items())))])
            stages = Counter((it.data.get("stage") or "starting")
                             for it in group if it.state == "llm")
            if stages:
                block.append([("text", "  "), ("label", "llm stage: "), ("st_llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))])
            cls = Counter(
                fairy.format_llm_classification(
                    (it.data.get("review") or {}).get("classification") or "-")
                for it in group if it.data.get("review"))
            cls.pop("-", None)
            if cls:
                block.append([("text", "  "), ("label", "llm: "), ("llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(cls.items())))])
            acts = Counter(it.data.get("action") for it in group
                           if it.state == "posted" and it.data.get("action"))
            if acts:
                block.append([("text", "  "), ("label", "posted: "), ("st_posted",
                    ", ".join(f"{k}={v}" for k, v in sorted(acts.items())))])
            blocks.append(block)
        return [header, []] + tui_core.tile_blocks(blocks, width)

    def _llm_col(self, it: Item) -> str:
        if it.state == "llm":
            return it.data.get("stage") or "llm"
        review = it.data.get("review") or {}
        if review.get("classification"):
            return fairy.format_llm_classification(review["classification"])
        return ""

    def list_rows(self) -> list:
        m = self.model
        vis = m.visible()
        m.cursor = max(0, min(m.cursor, len(vis) - 1)) if vis else 0
        rows: list = []
        for i, it in enumerate(vis):
            llm = self._llm_col(it)
            mark = "▶" if it.state == "reviewed" else " "
            repo_col = (f"{self._repo_disp[it.repo]:<{self._repo_w}} "
                        if self._repo_w else "")
            title = it.data.get("title") or ""
            if i == m.cursor:
                rows.append(("cursor",
                             f"{mark}{_KIND_DISP[it.kind]:<5} {repo_col}#{it.number:<6} "
                             f"{it.state:<{STATE_W}} {llm:<9}  {title}"))
                continue
            rows.append([
                ("mark", mark),
                ("kind_pr" if it.kind == "pr" else "kind_issue",
                 f"{_KIND_DISP[it.kind]:<5} "),
                ("label", repo_col),
                ("num", f"#{it.number:<6} "),
                (f"st_{it.state}", f"{it.state:<{STATE_W}} "),
                ("llm", f"{llm:<9}  "),
                ("title", title),
            ])
        return rows

    def detail_lines(self, width: int) -> list[tui_core.StyledLine]:
        m = self.model
        key = m._cursor_key()
        item = m.items.get(key) if key else None
        if item is None:
            return [[("text", "(no item selected)")]]
        data = item.data
        where = f"{item.repo}#{item.number}" if self._repo_w else f"#{item.number}"
        head: list[tui_core.StyledLine] = [
            [("h2", f"{_KIND_DISP[item.kind]} {where}  {data.get('title') or ''}"[:width])],
        ]
        if data.get("html_url"):
            head.append([("link", str(data["html_url"])[:width])])
        if item.error:
            head.append([("log_err", f"file invalid: {item.error}"[:width])])
        if data.get("error"):
            head.append([("log_err", f"error: {data['error']}"[:width])])
        if data.get("send_blocked"):
            head.append([("log_warn", f"send blocked: {data['send_blocked']}"[:width])])
        review = data.get("review") or {}
        decision = agent.ticket_decision(item.kind, item.number, data)
        if decision is not None:
            head.append([("bold", fairy.manual_action_description(decision)[:width])])
        status_line = f"state {item.state}"
        if review.get("classification"):
            status_line += ("   llm "
                            + fairy.format_llm_classification(review["classification"]))
        if data.get("reason"):
            status_line += f"   reason {data['reason']}"
        head += [[("text", status_line[:width])]]
        for fieldname in ("cancelled_ci_contexts", "blocked_ci_contexts",
                          "external_approvers"):
            vals = data.get(fieldname) or []
            if vals:
                head.append([("label", fieldname.replace("_", " ") + ": "),
                             ("text", ", ".join(map(str, vals))[:width])])
        head.append([])
        if not review:
            return head + [[("text", f"({item.state}: no review)")]]
        labels = [
            [("bullet", f"label {c.get('op')} {c.get('label')}"),
             ("text", (f" ({c['reason']})" if c.get("reason") else "")
                      + (" [posted]" if c.get("post") else ""))]
            for c in review.get("label_changes") or []
        ]
        if labels:
            labels.append([])
        return head + labels + tui_core.render_markdown(
            review.get("message") or "", width)

    def paint(self) -> None:
        self._last_paint = time.monotonic()
        t = self.term
        w, h = t.width, t.height
        body_h = max(3, h - 1)
        rects = self.layout.rects(w, body_h)
        col_t, col_b, row = self.layout.splits(w, body_h)
        self.scroll["bl"] = min(self.scroll["bl"],
                                max(0, len(self.ring) - (rects["bl"].h - 1)))
        with self.model.lock:
            content: dict[str, list] = {
                "tl": self._scrolled("tl", self.stats_lines(self._text_width(rects["tl"])),
                                     rects["tl"].h - 1),
                "tr": self._list_window(self.list_rows(), rects["tr"].h - 1),
                "bl": [(self._log_style(tag), text) for tag, text in
                       self.ring.view(self.scroll["bl"], rects["bl"].h - 1)],
                "br": self._scrolled("br", self.detail_lines(self._text_width(rects["br"])),
                                     rects["br"].h - 1),
            }
            nact = sum(1 for it in self.model.items.values()
                       if it.state == "reviewed")
        buf = []
        for pane, rect in rects.items():
            self._blit(buf, rect, pane, content[pane])
        divider = self.styles.get("divider_drag" if self.drag else "divider") \
            or (lambda s: s)
        for y in range(row):
            buf.append(t.move_xy(col_t, y) + divider("│"))
        for y in range(row + 1, body_h):
            buf.append(t.move_xy(col_b, y) + divider("│"))
        buf.append(t.move_xy(0, row) + divider("─" * w))
        for col in {col_t, col_b}:
            buf.append(t.move_xy(col, row) + divider("┼"))
        if self._status_mode != (self.model.sort_mode, self.model.filter_mode):
            self._build_status()
        note = f" ▶ {nact} reviewed  " if nact else " "
        if len(note) + len(self._status_plain) > w:
            status = (note + self._status_plain)[:w].ljust(w)
        else:
            mark = self.styles.get("st_reviewed") or (lambda s: s)
            status = ((mark(note) if nact else note) + self._status_styled
                      + " " * (w - len(note) - len(self._status_plain)))
        buf.append(t.move_xy(0, h - 1) + status)
        print("".join(buf), end="", flush=True, file=t.stream)

    def _log_style(self, level: int | None) -> str:
        if level is None or logging.INFO <= level < logging.WARNING:
            return "plain"
        if level >= logging.ERROR:
            return "log_err"
        return "log_warn" if level >= logging.WARNING else "log_debug"

    @staticmethod
    def _text_width(rect: tui_core.Rect) -> int:
        return max(MIN_TEXT_W, rect.w - 1)

    def _scrolled(self, pane: str, lines: list, inner_h: int) -> list:
        self.scroll[pane] = max(0, min(self.scroll[pane], len(lines) - inner_h))
        return lines[self.scroll[pane]:self.scroll[pane] + inner_h]

    def _list_window(self, rows: list, inner_h: int) -> list:
        cursor = self.model.cursor
        if cursor < self.list_top:
            self.list_top = cursor
        if inner_h > 0 and cursor >= self.list_top + inner_h:
            self.list_top = cursor - inner_h + 1
        self.list_top = max(0, min(self.list_top, max(0, len(rows) - inner_h)))
        return rows[self.list_top:self.list_top + inner_h]

    def _blit(self, buf: list, rect: tui_core.Rect, pane: str, lines: list) -> None:
        t = self.term
        if rect.w <= 0 or rect.h <= 0:
            return
        title = f" {PANE_GLYPHS[pane]} {PANES[pane]} "
        if pane == "tr":
            title += f"[{self.model.filter_mode}] "
        bar = title[:rect.w].ljust(rect.w)
        bar_fn = self.styles.get("bar_focus" if pane == self.focus else "bar_blur") \
            or (t.reverse if pane == self.focus else (lambda s: s))
        buf.append(t.move_xy(rect.x, rect.y) + bar_fn(bar))
        # One blank gutter column on any divider-adjacent edge: without it
        # the terminal's own shift-click / double-click selection glues the
        # "│" to the pane text (e.g. copying a URL picks up the divider).
        lpad = " " if rect.x > 0 else ""
        w = max(1, rect.w - len(lpad) - (1 if rect.x + rect.w < t.width else 0))
        shown: list[str] = []
        self._shown[pane] = (rect, len(lpad), shown)
        for i in range(rect.h - 1):
            buf.append(t.move_xy(rect.x, rect.y + 1 + i) + lpad)
            line = lines[i] if i < len(lines) else ""
            shown.append(tui_core.sanitize(_plain([line])))
            # sanitize(): forge/LLM text must not inject escape sequences.
            if isinstance(line, str):
                buf.append(tui_core.sanitize(line)[:w].ljust(rect.w - len(lpad)))
            elif line and isinstance(line[0], str):
                style, text = line
                text = tui_core.sanitize(text)[:w].ljust(rect.w - len(lpad))
                fn = self.styles.get(style)
                buf.append(fn(text) if fn else text)
            else:
                buf.append(self._styled_line(line, w, rect.w - len(lpad)))

    def _styled_line(self, segs: tui_core.StyledLine, width: int,
                     pad_to: int | None = None) -> str:
        out = []
        used = 0
        for style, text in segs:
            if used >= width:
                break
            text = tui_core.sanitize(text)[:width - used]
            fn = self.styles.get(style)
            out.append(fn(text) if fn else text)
            used += len(text)
        return "".join(out) + " " * ((pad_to or width) - used)

    def _maybe_poll(self) -> None:
        """Refresh from disk when the filesystem watcher fired (within
        one input tick, <=100ms) or on the 1s fallback interval."""
        if not self.needs_poll.is_set() \
                and time.monotonic() - self._last_poll < 1.0:
            return
        self.needs_poll.clear()
        self._last_poll = time.monotonic()
        self.model.poll()
        self.tail.poll()
        self.model.dirty.set()

    def run(self) -> None:
        try:
            while not self.model.quit_flag:
                ks = self.term.inkey(timeout=0.1)
                while ks:
                    self.dispatch(ks)
                    ks = self.term.inkey(timeout=0)
                size = (self.term.width, self.term.height)
                if size != self._last_size:
                    self._last_size = size
                    self.model.dirty.set()
                self._maybe_poll()
                if self.model.dirty.is_set():
                    self.model.dirty.clear()
                    self.paint()
        except KeyboardInterrupt:
            pass
        self.model.quit_all()

    def _scroll_pane(self, pane: str, delta: int) -> None:
        if pane == "tr":
            with self.model.lock:
                self.model.cursor = max(0, self.model.cursor + delta)
        elif pane == "bl":
            # offset counts back from the newest line; 0 follows the tail
            self.scroll["bl"] = max(0, self.scroll["bl"] - delta)
        elif pane in self.scroll:
            self.scroll[pane] = max(0, self.scroll[pane] + delta)

    def _page(self) -> int:
        rects = self.layout.rects(self.term.width, max(3, self.term.height - 1))
        return max(1, rects[self.focus].h - 2)

    def dispatch(self, ks) -> None:
        name = ks.name or ""
        body_h = max(3, self.term.height - 1)
        if name.startswith("MOUSE_"):
            y, x = ks.mouse_yx
            hit = (self.layout.hit(x, y, self.term.width, body_h)
                   if 0 <= y < body_h else "")
            if name in ("MOUSE_SCROLL_UP", "MOUSE_SCROLL_DOWN"):
                if hit in PANES:
                    self._scroll_pane(hit, -3 if name.endswith("UP") else 3)
            elif name == "MOUSE_LEFT":
                if hit in ("vt", "vb", "h"):
                    self.drag = hit
                elif hit in PANES:
                    self.focus = hit
                    if hit == "tr":
                        row = y - 1  # list rows start under the title bar
                        with self.model.lock:
                            self.model.cursor = max(0, self.list_top + row)
                    self._copy_click(hit, x, y)
            elif name.endswith("_MOTION") and self.drag:
                before = self.layout.splits(self.term.width, body_h)
                self.layout.drag(self.drag, x, y, self.term.width, body_h)
                if self.layout.splits(self.term.width, body_h) == before:
                    return  # divider did not actually move; skip the repaint
            elif name.endswith("_RELEASED"):
                self.drag = None
            self.model.dirty.set()
            return
        if name == "KEY_TAB":
            self.focus = FOCUS_ORDER[(FOCUS_ORDER.index(self.focus) + 1) % 4]
        elif name == "KEY_UP":
            self._scroll_pane(self.focus, -1)
        elif name == "KEY_DOWN":
            self._scroll_pane(self.focus, 1)
        elif name == "KEY_PGUP":
            self._scroll_pane(self.focus, -self._page())
        elif name == "KEY_PGDOWN":
            self._scroll_pane(self.focus, self._page())
        elif ks == "q":
            self.model.quit_all()
        elif ks == "a":
            with self.model.lock:
                key = self.model._cursor_key()
                mode = self.model.cycle_filter()
                if key is not None:
                    self.model._move_cursor_to(key)
            logger.info("list filter: %s", mode)
        elif ks == "t":
            with self.model.lock:
                key = self.model._cursor_key()
                mode = self.model.cycle_sort()
                if key is not None:
                    self.model._move_cursor_to(key)
            logger.info("list sort: %s", mode)
        elif str(ks) in ACTION_KEYS:
            self.model.act(ACTION_KEYS[str(ks)])
        elif ks == "o":
            self.edit_review()
        elif ks in ("e", "E"):
            self.export(full=(ks == "E"))
        else:
            return
        self.model.dirty.set()

    def edit_review(self) -> None:
        """o: open $EDITOR on the cursor item's persisted review message.

        The message round-trips through a temp ``.md`` file (the raw JSON
        stays editable by hand outside the TUI); the result is written
        back in place under the item's lock -- refused (with a log line)
        when a worker holds the item or it moved on meanwhile."""
        with self.model.lock:
            key = self.model._cursor_key()
            item = self.model.items.get(key) if key else None
        if item is None:
            return
        message = (item.data.get("review") or {}).get("message")
        if message is None:
            logger.info("%s %s#%s has no review message to edit",
                        _KIND_DISP[item.kind], item.repo, item.number)
            return
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
        fd, tmp_name = tempfile.mkstemp(
            suffix=".md", prefix=f"fairy-{item.kind}-{item.number}-")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(message)
            cmd = [*shlex.split(editor), str(tmp)]
            logger.info("editing %s#%s via: %s", item.repo, item.number,
                        shlex.join(cmd))
            t = self.term
            print(t.exit_fullscreen + t.normal_cursor, end="", flush=True,
                  file=t.stream)
            try:
                rc = subprocess.call(
                    cmd, stdin=sys.__stdin__, stdout=sys.__stdout__,
                    stderr=sys.__stderr__)
            finally:
                print(t.enter_fullscreen + t.hide_cursor, end="", flush=True,
                      file=t.stream)
                self.model.dirty.set()
            if rc != 0:
                logger.warning("editor exited rc=%d; review unchanged", rc)
                return
            edited = tmp.read_text(encoding="utf-8")
            if edited == message:
                logger.info("%s#%s review unchanged", item.repo, item.number)
                return

            def record(d: dict) -> None:
                d.setdefault("review", {})["message"] = edited

            db = self.model.db(item)
            state = db.find(item.kind, item.number)  # it may have moved on
            if state and db.try_move(state, state, item.kind, item.number,
                                     mutate=record):
                logger.info("%s#%s review message updated (%d -> %d chars)",
                            item.repo, item.number, len(message), len(edited))
            else:
                logger.warning("%s#%s is busy or gone; review unchanged",
                               item.repo, item.number)
        finally:
            tmp.unlink(missing_ok=True)

    def _copy_click(self, pane: str, x: int, y: int) -> None:
        """Copy the URL / git hash / #number under a left click to the
        system clipboard via OSC 52 (needs terminal support; tmux wants
        set-clipboard on). The rows are stored unclipped, so a visually
        truncated URL still copies whole."""
        rect, gutter, rows = self._shown.get(pane, (None, 0, []))
        if rect is None:
            return
        row, col = y - rect.y - 1, x - rect.x - gutter
        if not (0 <= row < len(rows)) or col < 0:
            return
        token = tui_core.token_at(rows[row], col)
        if token is None:
            return
        if self._clip_cmd is not None:
            try:
                subprocess.run(
                    self._clip_cmd, input=token.encode(), timeout=2, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.info("copied %r via %s (primary selection)",
                            token, self._clip_cmd[0])
                return
            except Exception as exc:
                logger.debug("clipboard helper %s failed (%s); trying OSC 52",
                             self._clip_cmd, exc)
        # Display-less fallback stays on the c (clipboard) target: no
        # PRIMARY reaches the local end of a plain ssh session and several
        # terminals ignore 52;p.
        b64 = base64.b64encode(token.encode()).decode()
        print(f"\x1b]52;c;{b64}\x07", end="", flush=True, file=self.term.stream)
        logger.info("copied %r to the clipboard (OSC 52)", token)

    def export(self, full: bool) -> None:
        pane = self.focus
        inner_h = self._page() + 1
        # The visible export must reproduce the pane as painted: same
        # divider-dependent width, or the tiling/wrapping (and thus the
        # scroll window) would differ from the screen.
        rects = self.layout.rects(self.term.width, max(3, self.term.height - 1))
        with self.model.lock:
            if pane == "bl":
                text = (self.ring.all_text() if full
                        else "\n".join(t for _, t in
                                       self.ring.view(self.scroll["bl"], inner_h)))
            elif pane == "tl":
                lines = self.stats_lines(
                    EXPORT_FULL_W if full else self._text_width(rects["tl"]))
                text = _plain(lines if full else self._scrolled("tl", lines, inner_h))
            elif pane == "tr":
                rows = self.list_rows()
                if not full:
                    rows = self._list_window(rows, inner_h)
                text = _plain(rows)
            else:
                lines = self.detail_lines(
                    EXPORT_FULL_W if full else self._text_width(rects["br"]))
                if not full:
                    lines = self._scrolled("br", lines, inner_h)
                text = _plain(lines)
        path = self.save_dir / (
            f"fairy_tui-{PANES[pane]}-{datetime.now():%Y%m%d-%H%M%S}.txt")
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            path.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            logger.error("export to %s failed: %s", path, exc)
            return
        logger.info("exported %s pane (%s) to %s",
                    PANES[pane], "full" if full else "visible", path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Blessed 4-pane operator UI over the filedb ticket directories.",
    )
    p.add_argument("--pr-args", metavar="ARGS", action="append",
                   help="fairy.py argument string naming a PR side's repo; "
                        "one per use")
    p.add_argument("--issue-args", metavar="ARGS", action="append",
                   help="issue_fairy.py argument string naming an issue side's "
                        "repo; one per use")
    p.add_argument("--tail", metavar="FILE", action="append", type=Path,
                   help="follow this agent/worker --log-file in the logs pane "
                        "(repeatable)")
    p.add_argument("--log-file", type=Path,
                   help="tee every line shown in the logs pane to this file")
    p.add_argument("--save-dir", type=Path, default=Path("."),
                   help="directory for e/E pane exports (default: cwd)")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def build_sides(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, filedb.Db]], list[Path]]:
    """One (repo label, filedb) side per distinct repo: the PR and issue
    argument strings of one repo share a db, and Forgejo routes
    owner/repo case-insensitively, so case variants merge too. Every
    side's --log-file is collected for the logs pane, so the agent and
    worker logs arrive without separate --tail flags."""
    sides: list[tuple[str, filedb.Db]] = []
    tails: list[Path] = []
    for parse, arg_strs in ((fairy.parse_args, args.pr_args),
                            (issue_fairy.parse_args, args.issue_args)):
        for arg_str in arg_strs or []:
            ns = parse(shlex.split(arg_str))
            if ns.log_file and ns.log_file not in tails:
                tails.append(ns.log_file)
            label = f"{ns.owner}/{ns.repo}"
            if any(known.casefold() == label.casefold() for known, _ in sides):
                continue
            sides.append((label, filedb.Db(agent.db_root_for(ns))))
    tails += [t for t in args.tail or [] if t not in tails]
    return sides, tails


def main() -> int:
    args = parse_args()
    ring = tui_core.RingBuffer()
    sides, tails = build_sides(args)
    model = Model(sides)
    sink = OutputSink(ring, model.dirty, args.log_file)
    setup_logging(logger, False, handlers=[RingLogHandler(sink)])
    for repo, db in sides:
        logger.info("side %s: db %s", repo, db.root)
    for path in tails:
        logger.info("tailing %s", path)

    term = blessed.Terminal(stream=sys.__stdout__)
    faulthandler.enable(file=sys.__stderr__)
    ui = UILoop(term, model, ring, args.save_dir, LogTail(tails, sink))
    # File changes repaint within one 100ms input tick instead of the
    # 1s fallback rescan; without watchdog only the fallback remains.
    watch_paths(
        [db.root / state for _, db in sides for state in filedb.STATES]
        + sorted({t.parent for t in tails}),
        ui.needs_poll.set)
    with term.fullscreen(), term.cbreak(), term.hidden_cursor(), \
            term.mouse_enabled(report_drag=True, timeout=0.2), \
            captured_output(sink):
        model.poll()
        ui.run()
    sink.close()
    if args.log_file:
        print(f"captured output: {args.log_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
