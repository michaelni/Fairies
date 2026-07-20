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

Blessed 4-pane operator UI over the fairy PR and issue pipelines.

One tool for both sides: ``--pr-args``/``--issue-args`` each take the
full argument string of fairy.py / issue_fairy.py; each enabled side
runs run_reviews() on its own controller thread and talks to the screen
through a fairy.ReviewUI adapter. Panes: stats (top left), PR/issue
list (top right), captured debug output (bottom left), rendered
review message + label changes (bottom right). Dividers move with the
mouse.

The list is a table over the workset files, not a queue: the operator
selects any row and acts on it whenever they choose -- y posts a
reviewed item (guard-checked), s skips it, x drops it, r reruns the
LLM, f force-queues a candidate. Actions travel over a per-side channel
and execute on that side's controller thread; nothing ever waits for a
prompt.

What belongs here: everything terminal-facing -- blessed painting,
key/mouse dispatch, output capture, the ReviewUI adapters and the
shared item model.

What does NOT belong: layout/scrollback/markdown logic (tui_core) and
any review logic (fairy, issue_fairy).
"""

from __future__ import annotations

import argparse
import base64
import faulthandler
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
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Lock, Thread

import blessed

import ci_log
import fairy
import forge_gcli
import gcli_cache
import issue_fairy
import tui_core
import workset
from common import setup_logging

__all__ = ["main"]

logger = logging.getLogger(__name__)


class Status(IntEnum):
    PENDING = 0    # candidate listed; nothing back from the pipeline yet
    QUEUED = 1     # passed the gates; waiting for an LLM worker
    IN_LLM = 2     # LLM evaluation running
    REVIEWED = 3   # verdict in the file; actionable with y/s/x/r any time
    APPLIED = 4    # posted to the forge
    SKIPPED = 5    # operator answered skip
    CANCELLED = 6  # operator threw it out (x)
    DONE = 7       # non-actionable decision arrived (gate/LLM skip)
    INVALID = 8    # workset file failed validation (broken hand-edit)


@dataclass
class Item:
    kind: str            # "PR" | "issue"
    number: int
    api: dict            # forge ApiObject from the candidate listing
    title: str
    status: Status = Status.PENDING
    decision: fairy.Decision | None = None
    url: str = ""
    stage: str = ""  # wrapper sub-stage while IN_LLM: triage/review/combine
    ws: workset.WorkItem | None = None  # last good parse of the item file
    ws_error: str = ""                  # why the file currently fails to parse


_IN_PIPELINE = (Status.PENDING, Status.QUEUED, Status.IN_LLM,
                Status.REVIEWED, Status.INVALID)
_KIND_FILE = {"PR": "pr", "issue": "issue"}  # TUI kind -> workset file kind
_WORKSET_STATUS = {
    workset.WorkState.QUEUED: Status.QUEUED,
    workset.WorkState.TRIAGE: Status.IN_LLM,
    workset.WorkState.REVIEW: Status.IN_LLM,
    workset.WorkState.COMBINE: Status.IN_LLM,
    workset.WorkState.REVIEWED: Status.REVIEWED,
    workset.WorkState.POSTED: Status.APPLIED,
    workset.WorkState.SKIPPED: Status.SKIPPED,
    workset.WorkState.CANCELLED: Status.CANCELLED,
    workset.WorkState.ERROR: Status.DONE,
}
_WORKSET_STAGE = {
    workset.WorkState.TRIAGE: "triage",
    workset.WorkState.REVIEW: "review",
    workset.WorkState.COMBINE: "combine",
}


def _ws_actionable(ws: workset.WorkItem) -> bool:
    """Something to post: a posting classification or label changes.
    LLM-skip verdicts stay REVIEWED on disk (they carry the skip-backoff
    memory) but there is nothing for the operator to send."""
    return ws.review is not None and (
        ws.review.classification not in ("skip", "error", "-", "")
        or bool(ws.review.label_changes)
    )


@dataclass
class Pipeline:
    input_queue: SimpleQueue
    pending: fairy.PendingCount
    cancelled: set[int]
    actions: SimpleQueue  # (number, action) tuples for the controller


class Model:
    """Shared state between the controller threads (via SideUI) and the
    UI loop. Every mutation happens under ``lock``; ``dirty`` wakes the
    painter."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.dirty = Event()
        self.items: dict[tuple[str, int], Item] = {}
        self.order: list[tuple[str, int]] = []
        self.pipelines: dict[str, Pipeline] = {}
        self.forced: dict[str, set[int]] = {}
        self.workset_dirs: dict[str, Path] = {}  # kind -> per-repo dir
        self._ws_mtimes: dict[Path, float] = {}  # poll_workset change detection
        self.show_all = False
        self.cursor = 0
        self.quit_flag = False
        self.started = time.monotonic()

    # ---- controller-thread side (called through SideUI) ----

    def add_candidates(self, kind: str, apis: list[dict]) -> None:
        with self.lock:
            for api in apis:
                number = api.get("number")
                if not str(number).isdigit():
                    continue
                key = (kind, int(number))
                if key in self.items:
                    continue
                url = api.get("html_url")
                self.items[key] = Item(
                    kind, int(number), api, str(api.get("title") or ""),
                    url=url if isinstance(url, str) else "",
                )
                self.order.append(key)
        self.dirty.set()

    def attach_pipeline(self, kind: str, pipe: Pipeline, forced: set[int]) -> None:
        with self.lock:
            self.pipelines[kind] = pipe
            self.forced[kind] = forced
        self.dirty.set()

    def note_reviewed(self, kind: str, decision: fairy.Decision, url: str) -> None:
        """A decision arrived; record it for the detail pane. The row's
        status comes from the workset file (poll), the operator acts on
        it whenever they choose."""
        with self.lock:
            item = self._ensure(kind, decision)
            item.decision = decision
            item.url = url or item.url
        self.dirty.set()

    def finish(self, kind: str, decision: fairy.Decision) -> None:
        with self.lock:
            item = self._ensure(kind, decision)
            item.decision = decision
            actionable = (decision.action in fairy.ACTIONABLE_DECISIONS
                          or bool(decision.label_changes))
            # Actionable items stay at their file state (reviewed) so the
            # operator can still act on them; everything else is done.
            if item.status in _IN_PIPELINE and not actionable:
                item.status = Status.DONE
        self.dirty.set()

    def workset_file(self, kind: str, number: int) -> Path | None:
        d = self.workset_dirs.get(kind)
        return workset.item_path_in(d, _KIND_FILE[kind], number) if d else None

    def poll_workset(self) -> None:
        """Refresh item state from the on-disk work files (1 Hz, UI loop).

        The files are the durable source of truth: they carry the pipeline
        and wrapper stage transitions, items left behind by a previous or
        killed run, and external edits/deletions by the operator. File IO
        happens outside ``lock``."""
        changed: list[tuple[str, int, workset.WorkItem | None, str]] = []
        removed: list[tuple[str, int]] = []
        for kind, d in self.workset_dirs.items():
            prefix = f"{_KIND_FILE[kind]}-"
            try:
                paths = set(d.glob(prefix + "*.json"))
            except OSError:
                continue
            gone = [p for p in self._ws_mtimes
                    if p.parent == d and p.name.startswith(prefix) and p not in paths]
            for path in gone:
                del self._ws_mtimes[path]
                number = path.stem.removeprefix(prefix)
                if number.isdigit():
                    removed.append((kind, int(number)))
            for path in paths:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    continue
                if self._ws_mtimes.get(path) == mtime:
                    continue
                self._ws_mtimes[path] = mtime
                ws, error = workset.load_item_result(path)
                if ws is not None:
                    changed.append((kind, ws.number, ws, ""))
                elif error is not None:
                    number = path.stem.removeprefix(prefix)
                    if number.isdigit():
                        changed.append((kind, int(number), None, error))
        if not changed and not removed:
            return
        with self.lock:
            # Deterministic row order for newly discovered items: the
            # scan iterates a set of paths, which has no stable order.
            for kind, number, ws, error in sorted(
                    changed, key=lambda c: (c[0], c[1])):
                key = (kind, number)
                item = self.items.get(key)
                if item is None:
                    item = Item(kind, number, {}, ws.title if ws else "")
                    self.items[key] = item
                    self.order.append(key)
                item.ws_error = error
                if ws is None:
                    item.stage = ""
                    if item.status in _IN_PIPELINE:
                        item.status = Status.INVALID
                    continue
                item.ws = ws
                item.title = item.title or ws.title
                item.url = item.url or ws.html_url
                item.stage = _WORKSET_STAGE.get(ws.state, "")
                if item.status in _IN_PIPELINE:
                    status = _WORKSET_STATUS[ws.state]
                    # A persisted LLM skip is bookkeeping, not work: it
                    # must not show (or count) as awaiting the operator.
                    if status is Status.REVIEWED and not _ws_actionable(ws):
                        status = Status.DONE
                    item.status = status
            for kind, number in removed:
                item = self.items.get((kind, number))
                if item is not None and item.status in _IN_PIPELINE:
                    item.status = Status.CANCELLED
                    item.stage = ""
                    item.ws_error = ""
        self.dirty.set()

    # ---- UI-thread side ----

    def visible(self) -> list[Item]:
        """Items for the list pane, honoring the all/relevant filter.
        Caller holds ``lock``."""
        items = [self.items[k] for k in self.order]
        if self.show_all:
            return items
        return [it for it in items if self._relevant(it)]

    def act(self, action: str) -> None:
        """Route a table action ("apply"/"skip"/"cancel"/"rerun") on the
        cursor row to its side's controller, which executes it with the
        run's args and caches. Any row is actionable at any time; the
        controller re-validates against the file before doing anything."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key) if key else None
            pipe = self.pipelines.get(item.kind) if item else None
            if item is None or pipe is None:
                return
            if action == "rerun" and item.status in (Status.PENDING, Status.QUEUED,
                                                     Status.IN_LLM):
                logger.info("%s #%s is already being evaluated", item.kind, item.number)
                return
            if action in ("apply", "skip", "cancel") and (
                item.ws is None
                or item.ws.state not in (workset.WorkState.REVIEWED,
                                         workset.WorkState.ERROR)
            ):
                logger.info("%s #%s has no reviewed workset file to %s",
                            item.kind, item.number, action)
                return
            if action == "apply" and not _ws_actionable(item.ws):
                logger.info(
                    "%s #%s has nothing to post (LLM verdict: %s)",
                    item.kind, item.number,
                    item.ws.review.classification if item.ws.review else "-",
                )
                return
            logger.info("requested %s for %s #%s", action, item.kind, item.number)
            pipe.actions.put((item.number, action))
            # Jump to the next reviewed row waiting for the operator: the
            # first at/after the cursor, wrapping to the first overall.
            remaining = {(it.kind, it.number) for it in self.visible()
                         if it.status is Status.REVIEWED} - {key}
            if remaining:
                keys = [(it.kind, it.number) for it in self.visible()]
                nxt = next((k for k in keys[self.cursor:] if k in remaining),
                           next((k for k in keys if k in remaining), None))
                if nxt is not None:
                    self._move_cursor_to(nxt)
        self.dirty.set()

    def force(self) -> None:
        """Force-queue the cursor item for (re-)review, bypassing gates."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key) if key else None
            pipe = self.pipelines.get(item.kind) if item else None
            if item is None or pipe is None:
                return
            if item.status in (Status.PENDING, Status.QUEUED, Status.IN_LLM):
                logger.info("%s #%s is already in flight", item.kind, item.number)
                return
            if not item.api:
                logger.info("%s #%s has no fetched data to re-queue", item.kind, item.number)
                return
            self.forced[item.kind].add(item.number)
            pipe.cancelled.discard(item.number)
            item.status = Status.PENDING
            pipe.pending.add(1)
            pipe.input_queue.put(item.api)
            logger.info("force-queued %s #%s for review", item.kind, item.number)
        self.dirty.set()

    def cancel(self) -> None:
        """x: throw the cursor item out -- stop an upcoming evaluation,
        or cancel a persisted review via the controller."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key) if key else None
            pipe = self.pipelines.get(item.kind) if item else None
            if item is None or pipe is None:
                return
            if item.status in (Status.PENDING, Status.QUEUED):
                pipe.cancelled.add(item.number)
                item.status = Status.CANCELLED
                logger.info("cancelled %s #%s", item.kind, item.number)
                self.dirty.set()
                return
        self.act("cancel")

    def quit_all(self) -> None:
        with self.lock:
            if self.quit_flag:
                return
            self.quit_flag = True
        logger.info("quit requested; waiting for the pipelines to wind down")
        self.dirty.set()

    # ---- internals (caller holds ``lock``) ----

    def _ensure(self, kind: str, decision: fairy.Decision) -> Item:
        # Forced items can have numbers absent from the candidate listing.
        key = (kind, decision.pr_number)
        item = self.items.get(key)
        if item is None:
            item = Item(kind, decision.pr_number, {}, decision.title)
            self.items[key] = item
            self.order.append(key)
        return item

    def _relevant(self, item: Item) -> bool:
        if item.number in self.forced.get(item.kind, ()):
            return True
        if item.status in (Status.QUEUED, Status.IN_LLM, Status.REVIEWED,
                           Status.INVALID):
            return True
        d = item.decision
        return d is not None and (
            d.action in fairy.ACTIONABLE_DECISIONS or bool(d.label_changes)
        )

    def _cursor_key(self) -> tuple[str, int] | None:
        vis = self.visible()
        if not vis:
            return None
        self.cursor = max(0, min(self.cursor, len(vis) - 1))
        it = vis[self.cursor]
        return (it.kind, it.number)

    def _move_cursor_to(self, key: tuple[str, int]) -> None:
        """Cursor onto ``key``; if the filter hides it, onto the nearest
        preceding visible item."""
        order_pos = {k: i for i, k in enumerate(self.order)}
        pos = order_pos.get(key)
        if pos is None:
            return
        self.cursor = 0
        for i, it in enumerate(self.visible()):
            if order_pos[(it.kind, it.number)] <= pos:
                self.cursor = i
            else:
                break


class SideUI:
    """fairy.ReviewUI adapter for one side ("PR" or "issue")."""

    def __init__(self, kind: str, model: Model, forced: set[int]) -> None:
        self.kind = kind
        self.model = model
        self.forced = forced

    def candidates(self, items: list[dict]) -> None:
        self.model.add_candidates(self.kind, items)

    def pipeline(self, input_queue, pending, cancelled, actions) -> None:
        self.model.attach_pipeline(
            self.kind, Pipeline(input_queue, pending, cancelled, actions),
            self.forced,
        )

    def decide(self, prepared, decision, url) -> str:
        # Table model: never block the controller; the operator acts on
        # the row (y/s/x/r) whenever they choose.
        self.model.note_reviewed(self.kind, decision, url)
        return "hold"

    def item_done(self, prepared, decision) -> None:
        self.model.finish(self.kind, decision)

    def keep_open(self) -> bool:
        return not self.model.quit_flag

    def stopped(self) -> bool:
        return self.model.quit_flag


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
            '%(asctime)s %(thread_prefix)s%(message)s', '%Y-%m-%d %H:%M:%S',
        ))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.line(self.format(record), record.levelno)
        except Exception:
            pass  # never let UI logging kill a worker


class StreamToRing:
    """File-like stand-in for sys.stdout/sys.stderr while the UI owns the
    screen; complete lines (wrapper stderr pump, stray prints) go to the
    sink."""

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


PANES = {"tl": "stats", "tr": "list", "bl": "debug", "br": "message"}
PANE_GLYPHS = {"tl": "Σ", "tr": "☰", "bl": "≣", "br": "¶"}
FOCUS_ORDER = ("tl", "tr", "bl", "br")
ACTION_KEYS = {"y": "apply", "s": "skip", "r": "rerun"}
KEYMAP = (("q", "quit"), ("y", "apply"), ("s", "skip"), ("r", "rerun"),
          ("f", "force"), ("x", "drop"), ("o", "edit msg"),
          ("a", "all/relevant"), ("e/E", "export"),
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
            "st_pending": c(244),
            "st_queued": c(117),              "st_in_llm": mix(t.bold, c(45)),
            "st_reviewed": mix(t.bold, c(214)),
            "st_applied": c(78),              "st_skipped": c(244),
            "st_cancelled": c(167),           "st_done": c(108),
            "st_invalid": mix(t.bold, c(196)),
            "cursor": t.reverse,
            # debug-pane log levels; palette mirrors common._ColorFormatter
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
        "st_pending": t.bright_black,
        "st_queued": t.cyan, "st_in_llm": t.bold_cyan,
        "st_reviewed": t.bold_yellow,
        "st_applied": t.green,  "st_skipped": t.bright_black,
        "st_cancelled": t.red,  "st_done": t.cyan, "st_invalid": t.bold_red,
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
                 save_dir: Path, kinds: list[str]) -> None:
        self.term = term
        self.model = model
        self.ring = ring
        self.save_dir = save_dir
        self.kinds = kinds
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
        key = self.styles.get("key") or (lambda s: s)
        label = self.styles.get("label") or (lambda s: s)
        self._status_plain = "  ".join(f"{k} {d}" for k, d in KEYMAP)
        self._status_styled = "  ".join(f"{key(k)} {label(d)}" for k, d in KEYMAP)

    # ---- pane content (caller holds model.lock) ----

    def stats_lines(self) -> list[tui_core.StyledLine]:
        m = self.model
        items = [m.items[k] for k in m.order]
        actionable = sum(1 for it in items if it.status is Status.REVIEWED)
        lines: list[tui_core.StyledLine] = [[
            ("label", "elapsed "),
            ("num", f"{int(time.monotonic() - m.started)}s"),
            ("label", "   reviewed, awaiting you "),
            ("st_reviewed" if actionable else "num", str(actionable)),
        ]]
        for kind in self.kinds:
            group = [it for it in items if it.kind == kind]
            by = Counter(it.status for it in group)
            pipe = m.pipelines.get(kind)
            lines += [[], [
                ("kind_pr" if kind == "PR" else "kind_issue", f"{kind}s "),
                ("num", str(len(group))),
                ("label", " candidates, in flight "),
                ("num", str(pipe.pending.value) if pipe else "-"),
            ]]
            if by:
                row: tui_core.StyledLine = [("text", "  ")]
                for s in Status:
                    if by[s]:
                        row += [(f"st_{s.name.lower()}", f"{s.name.lower()}="),
                                ("num", str(by[s])), ("text", "  ")]
                lines.append(row)
            stages = Counter(it.stage or "starting" for it in group
                             if it.status is Status.IN_LLM)
            if stages:
                lines.append([("text", "  "), ("label", "llm stage: "), ("st_in_llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(stages.items())))])
            cls = Counter(
                fairy.format_llm_classification(it.decision.llm_classification)
                for it in group if it.decision)
            cls.pop("-", None)
            if cls:
                lines.append([("text", "  "), ("label", "llm: "), ("llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(cls.items())))])
            acts = Counter(it.decision.action for it in group
                           if it.status is Status.APPLIED and it.decision)
            if acts:
                lines.append([("text", "  "), ("label", "applied: "), ("st_applied",
                    ", ".join(f"{k}={v}" for k, v in sorted(acts.items())))])
        return lines

    def list_rows(self) -> list:
        m = self.model
        vis = m.visible()
        m.cursor = max(0, min(m.cursor, len(vis) - 1)) if vis else 0
        rows: list = []
        for i, it in enumerate(vis):
            d = it.decision
            if d is not None:
                llm = fairy.format_llm_classification(d.llm_classification)
            elif not it.stage and it.ws is not None and it.ws.review is not None:
                llm = fairy.format_llm_classification(it.ws.review.classification)
            else:
                llm = it.stage or ("llm" if it.status is Status.IN_LLM else "")
            mark = "▶" if it.status is Status.REVIEWED else " "
            if i == m.cursor:
                rows.append(("cursor",
                             f"{mark}{it.kind:<5} #{it.number:<6} "
                             f"{it.status.name.lower():<9} {llm:<9}  {it.title}"))
                continue
            rows.append([
                ("mark", mark),
                ("kind_pr" if it.kind == "PR" else "kind_issue", f"{it.kind:<5} "),
                ("num", f"#{it.number:<6} "),
                (f"st_{it.status.name.lower()}", f"{it.status.name.lower():<9} "),
                ("llm", f"{llm:<9}  "),
                ("title", it.title),
            ])
        return rows

    def detail_lines(self, width: int) -> list[tui_core.StyledLine]:
        m = self.model
        key = m._cursor_key()
        item = m.items.get(key) if key else None
        if item is None:
            return [[("text", "(no item selected)")]]
        d = item.decision
        ws = item.ws
        review = ws.review if ws is not None else None
        head: list[tui_core.StyledLine] = [
            [("h2", f"{item.kind} #{item.number}  {item.title}"[:width])],
        ]
        if item.url:
            head.append([("link", item.url[:width])])
        if item.ws_error:
            head.append([("log_err", f"file invalid: {item.ws_error}"[:width])])
        if ws is not None and ws.error:
            head.append([("log_err", f"error: {ws.error}"[:width])])
        if d is None and review is None:
            return head + [[], [("text", f"({item.status.name.lower()}: no review yet)")]]
        classification = review.classification if review else d.llm_classification
        message = review.message if review else d.llm_message
        label_changes = review.label_changes if review else d.label_changes
        if d is not None:
            head.append([("bold", fairy.manual_action_description(d)[:width])])
        status_line = (f"status {item.status.name.lower()}   llm "
                       f"{fairy.format_llm_classification(classification)}")
        if d is not None:
            status_line += f"   reason {d.reason}"
        head += [[("text", status_line[:width])], []]
        labels = [
            [("bullet", f"label {c.op} {c.label}"),
             ("text", (f" ({c.reason})" if c.reason else "") + (" [posted]" if c.post else ""))]
            for c in label_changes
        ]
        if labels:
            labels.append([])
        return head + labels + tui_core.render_markdown(message, width)

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
                "tl": self._scrolled("tl", self.stats_lines(), rects["tl"].h - 1),
                "tr": self._list_window(self.list_rows(), rects["tr"].h - 1),
                "bl": [(self._log_style(tag), text) for tag, text in
                       self.ring.view(self.scroll["bl"], rects["bl"].h - 1)],
                "br": self._scrolled("br", self.detail_lines(max(8, rects["br"].w - 1)),
                                     rects["br"].h - 1),
            }
            nact = sum(1 for it in self.model.items.values()
                       if it.status is Status.REVIEWED)
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
            title += f"[{'all' if self.model.show_all else 'relevant'}] "
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
                if time.monotonic() - self._last_paint >= 1.0:
                    self.model.poll_workset()
                    self.model.dirty.set()
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
                self.model.show_all = not self.model.show_all
                if key is not None:
                    self.model._move_cursor_to(key)
        elif str(ks) in ACTION_KEYS:
            self.model.act(ACTION_KEYS[str(ks)])
        elif ks == "f":
            self.model.force()
        elif ks == "x":
            self.model.cancel()
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
        back through the flock'd update path."""
        with self.model.lock:
            key = self.model._cursor_key()
            item = self.model.items.get(key) if key else None
        if item is None:
            return
        path = self.model.workset_file(item.kind, item.number)
        ws = workset.load_item(path) if path is not None else None
        if ws is None or ws.review is None:
            logger.info("%s #%s has no persisted review to edit",
                        item.kind, item.number)
            return
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
        fd, tmp_name = tempfile.mkstemp(
            suffix=".md", prefix=f"fairy-{_KIND_FILE[item.kind]}-{item.number}-")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(ws.review.message)
            cmd = [*shlex.split(editor), str(tmp)]
            logger.info("editing %s via: %s", path, shlex.join(cmd))
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
            if edited == ws.review.message:
                logger.info("%s #%s review unchanged", item.kind, item.number)
                return

            def record(it: workset.WorkItem) -> None:
                if it.review is not None:
                    it.review.message = edited

            if workset.update_item(path, record) is not None:
                logger.info("%s #%s review message updated (%d -> %d chars)",
                            item.kind, item.number,
                            len(ws.review.message), len(edited))
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
        with self.model.lock:
            if pane == "bl":
                text = (self.ring.all_text() if full
                        else "\n".join(t for _, t in
                                       self.ring.view(self.scroll["bl"], inner_h)))
            elif pane == "tl":
                lines = self.stats_lines()
                text = _plain(lines if full else self._scrolled("tl", lines, inner_h))
            elif pane == "tr":
                rows = self.list_rows()
                if not full:
                    rows = self._list_window(rows, inner_h)
                text = _plain(rows)
            else:
                lines = self.detail_lines(200 if full else max(8, self.term.width // 2))
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
        description="Blessed 4-pane operator UI over the fairy PR and issue pipelines.",
    )
    p.add_argument("--pr-args", metavar="ARGS",
                   help="fairy.py argument string; enables the PR side")
    p.add_argument("--issue-args", metavar="ARGS",
                   help="issue_fairy.py argument string; enables the issue side")
    p.add_argument("--log-file", type=Path,
                   help="tee every captured log/output line to this file")
    p.add_argument("--save-dir", type=Path, default=Path("."),
                   help="directory for e/E pane exports (default: cwd)")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def main() -> int:
    args = parse_args()
    ring = tui_core.RingBuffer()
    model = Model()
    sink = OutputSink(ring, model.dirty, args.log_file)

    sides = []
    if args.pr_args:
        ns = fairy.parse_args(shlex.split(args.pr_args))
        sides.append(("PR", fairy.run_reviews, ns, ns.force_review_prs))
    if args.issue_args:
        ns = issue_fairy.parse_args(shlex.split(args.issue_args))
        sides.append(("issue", issue_fairy.run_reviews, ns, ns.force_review_issues))

    # issue_fairy.logger is fairy.logger, so listing fairy's covers both.
    setup_logging(
        fairy.logger, max(ns.verbose for _, _, ns, _ in sides),
        forge_gcli.logger, gcli_cache.logger, workset.logger, ci_log.logger,
        logger,
        handlers=[RingLogHandler(sink)],
    )
    for kind, _, ns, _forced in sides:
        if ns.approve:
            ns.approve = False
            logger.warning("--approve on the %s side is ignored: the TUI always "
                           "asks per decision", kind)
        logger.info("%s side enabled: %s/%s", kind, ns.owner, ns.repo)
        ws_dir = fairy.workset_repo_dir(ns)
        if ws_dir is not None:
            model.workset_dirs[kind] = ws_dir
            logger.info("%s workset dir: %s", kind, ws_dir)
    if args.log_file:
        logger.info("teeing captured output to %s", args.log_file)

    term = blessed.Terminal(stream=sys.__stdout__)
    faulthandler.enable(file=sys.__stderr__)
    threads = []
    for kind, run, ns, forced in sides:
        def run_side(run=run, ns=ns, kind=kind, forced=forced) -> None:
            try:
                logger.info("%s side finished rc=%s",
                            kind, run(ns, SideUI(kind, model, forced)))
            except Exception:
                logger.exception("%s side crashed", kind)
        threads.append(Thread(target=run_side, name=f"{kind}-controller", daemon=True))

    ui = UILoop(term, model, ring, args.save_dir, [k for k, *_ in sides])
    with term.fullscreen(), term.cbreak(), term.hidden_cursor(), \
            term.mouse_enabled(report_drag=True, timeout=0.2), \
            captured_output(sink):
        for th in threads:
            th.start()
        ui.run()
    if any(th.is_alive() for th in threads):
        print("waiting up to 10s for the pipelines to wind down"
              + (f" (their logs land in {args.log_file})" if args.log_file else "")
              + " ...", file=sys.stderr)
    for th in threads:
        th.join(timeout=10)
    sink.close()
    for th in threads:
        if th.is_alive():
            print(f"{th.name} still winding down (LLM call in flight?); "
                  "its cache saves may be incomplete", file=sys.stderr)
    if args.log_file:
        print(f"captured output: {args.log_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
