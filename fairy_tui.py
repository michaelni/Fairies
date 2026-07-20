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
mouse; the operator answers each actionable decision with the
prompt_manual letters (y/s/d/r, q quits).

What belongs here: everything terminal-facing -- blessed painting,
key/mouse dispatch, output capture, the ReviewUI adapters and the
shared item model.

What does NOT belong: layout/scrollback/markdown logic (tui_core) and
any review logic (fairy, issue_fairy).
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import shlex
import sys
import time
from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path
from queue import SimpleQueue
from threading import Event, Lock, Thread

import blessed

import bot_state
import ci_log
import fairy
import forge_gcli
import gcli_cache
import issue_fairy
import tui_core
from common import setup_logging

__all__ = ["main"]

logger = logging.getLogger(__name__)


class Status(IntEnum):
    PENDING = 0    # candidate listed; nothing back from the pipeline yet
    AWAITING = 1   # actionable decision waiting for the operator
    RETRYING = 2   # operator sent it back to the LLM
    DEFERRED = 3   # operator pushed it to the back of the queue
    APPLIED = 4
    SKIPPED = 5    # operator answered skip
    CANCELLED = 6  # operator threw it out (x)
    DONE = 7       # non-actionable decision arrived (gate/LLM skip)


@dataclass
class Item:
    kind: str            # "PR" | "issue"
    number: int
    api: dict            # forge ApiObject from the candidate listing
    title: str
    status: Status = Status.PENDING
    decision: fairy.Decision | None = None
    url: str = ""


@dataclass
class PromptReq:
    key: tuple[str, int]
    reply: SimpleQueue = field(default_factory=SimpleQueue)


@dataclass
class Pipeline:
    input_queue: SimpleQueue
    llm_queue: SimpleQueue
    pending: fairy.PendingCount
    cancelled: set[int]


class Model:
    """Shared state between the controller threads (via SideUI) and the
    UI loop. Every mutation happens under ``lock``; ``dirty`` wakes the
    painter."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.dirty = Event()
        self.items: dict[tuple[str, int], Item] = {}
        self.order: list[tuple[str, int]] = []
        self.prompts: deque[PromptReq] = deque()
        self.pipelines: dict[str, Pipeline] = {}
        self.forced: dict[str, set[int]] = {}
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

    def ask(self, kind: str, decision: fairy.Decision, url: str) -> str:
        req = PromptReq((kind, decision.pr_number))
        with self.lock:
            if self.quit_flag:
                return "quit"
            item = self._ensure(kind, decision)
            item.decision = decision
            item.url = url or item.url
            item.status = Status.AWAITING
            self.prompts.append(req)
            if self._prompt_for(self._cursor_key()) is None:
                self._move_cursor_to(req.key)
        self.dirty.set()
        return req.reply.get()

    def finish(self, kind: str, decision: fairy.Decision) -> None:
        with self.lock:
            item = self._ensure(kind, decision)
            item.decision = decision
            if item.status in (Status.PENDING, Status.RETRYING):
                item.status = Status.DONE
        self.dirty.set()

    # ---- UI-thread side ----

    def visible(self) -> list[Item]:
        """Items for the list pane, honoring the all/relevant filter.
        Caller holds ``lock``."""
        items = [self.items[k] for k in self.order]
        if self.show_all:
            return items
        return [it for it in items if self._relevant(it)]

    def answer(self, choice: str) -> bool:
        """Answer the cursor item's pending prompt; False if it has none."""
        with self.lock:
            key = self._cursor_key()
            req = self._prompt_for(key)
            if req is None:
                return False
            self.items[key].status = {
                "apply": Status.APPLIED, "skip": Status.SKIPPED,
                "defer": Status.DEFERRED, "retry": Status.RETRYING,
            }[choice]
            self.prompts.remove(req)
        req.reply.put(choice)
        self.dirty.set()
        return True

    def force(self) -> None:
        """Force-queue the cursor item for (re-)review, bypassing gates."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key)
            pipe = self.pipelines.get(item.kind) if item else None
            if item is None or pipe is None or self._prompt_for(key):
                return
            if item.status in (Status.PENDING, Status.RETRYING, Status.DEFERRED):
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
        """Throw the cursor item out: skip its prompt if one is pending,
        else cancel it before/instead of its LLM evaluation."""
        if self.answer("skip"):
            return
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key)
            pipe = self.pipelines.get(item.kind) if item else None
            if item is None or pipe is None:
                return
            if item.status in (Status.PENDING, Status.RETRYING, Status.DEFERRED):
                pipe.cancelled.add(item.number)
                item.status = Status.CANCELLED
                logger.info("cancelled %s #%s", item.kind, item.number)
        self.dirty.set()

    def quit_all(self) -> None:
        with self.lock:
            if self.quit_flag:
                return
            self.quit_flag = True
            reqs = list(self.prompts)
            self.prompts.clear()
        for req in reqs:
            req.reply.put("quit")
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
        # "Today's relevant set": what this run works on -- forced items,
        # items in flight toward/awaiting the operator, and finished ones
        # whose decision proposed an action or label change.
        if item.number in self.forced.get(item.kind, ()):
            return True
        if item.status in (Status.AWAITING, Status.RETRYING, Status.DEFERRED):
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

    def _prompt_for(self, key: tuple[str, int] | None) -> PromptReq | None:
        return next((r for r in self.prompts if r.key == key), None)

    def _move_cursor_to(self, key: tuple[str, int]) -> None:
        for i, it in enumerate(self.visible()):
            if (it.kind, it.number) == key:
                self.cursor = i
                return


class SideUI:
    """fairy.ReviewUI adapter for one side ("PR" or "issue")."""

    def __init__(self, kind: str, model: Model, forced: set[int]) -> None:
        self.kind = kind
        self.model = model
        self.forced = forced

    def candidates(self, items: list[dict]) -> None:
        self.model.add_candidates(self.kind, items)

    def pipeline(self, input_queue, llm_queue, pending, cancelled) -> None:
        self.model.attach_pipeline(
            self.kind, Pipeline(input_queue, llm_queue, pending, cancelled),
            self.forced,
        )

    def decide(self, prepared, decision, url) -> str:
        return self.model.ask(self.kind, decision, url)

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
FOCUS_ORDER = ("tl", "tr", "bl", "br")
CHOICE_KEYS = {"y": "apply", "s": "skip", "d": "defer", "r": "retry"}
STATUS_LINE = (
    " q quit  y/s/d/r decide  f force  x drop  a all/relevant  "
    "e/E export  Tab/click focus  arrows/pgup/pgdn scroll "
)


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
        t = term
        self.styles = {
            "h1": t.bold_underline, "h2": t.bold, "h3": t.underline,
            "bold": t.bold, "italic": t.italic, "code": t.reverse,
            "codeblock": t.on_bright_black, "quote": t.bright_black,
            "bullet": t.bold,
            "cursor": t.reverse, "awaiting": t.bold_red, "plain": None,
            "text": None,
            # debug-pane log levels; palette mirrors common._ColorFormatter
            "log_debug": t.dim_bright_black, "log_warn": t.bold_yellow,
            "log_err": t.bold_red,
        }

    # ---- pane content (caller holds model.lock) ----

    def stats_lines(self) -> list[str]:
        m = self.model
        lines = [
            f"elapsed {int(time.monotonic() - m.started)}s"
            f"   prompts waiting {len(m.prompts)}",
        ]
        items = [m.items[k] for k in m.order]
        for kind in self.kinds:
            group = [it for it in items if it.kind == kind]
            by = Counter(it.status for it in group)
            pipe = m.pipelines.get(kind)
            in_flight = pipe.pending.value if pipe else "-"
            lines += ["", f"{kind}s: {len(group)} candidates, in flight {in_flight}"]
            if by:
                lines.append("  " + "  ".join(
                    f"{s.name.lower()}={by[s]}" for s in Status if by[s]))
            cls = Counter(
                fairy.format_llm_classification(it.decision.llm_classification)
                for it in group if it.decision)
            cls.pop("-", None)
            if cls:
                lines.append("  llm: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(cls.items())))
            acts = Counter(it.decision.action for it in group
                           if it.status is Status.APPLIED and it.decision)
            if acts:
                lines.append("  applied: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(acts.items())))
        return lines

    def list_rows(self) -> list[tuple[str, str]]:
        m = self.model
        vis = m.visible()
        m.cursor = max(0, min(m.cursor, len(vis) - 1)) if vis else 0
        rows = []
        for i, it in enumerate(vis):
            d = it.decision
            llm = fairy.format_llm_classification(d.llm_classification) if d else ""
            mark = ">" if m._prompt_for((it.kind, it.number)) else " "
            text = (f"{mark}{it.kind:<5} #{it.number:<6} "
                    f"{it.status.name.lower():<9} {llm:<9} {it.title}")
            style = "cursor" if i == m.cursor else (
                "awaiting" if mark == ">" else "plain")
            rows.append((style, text))
        return rows

    def detail_lines(self, width: int) -> list[tui_core.StyledLine]:
        m = self.model
        key = m._cursor_key()
        item = m.items.get(key) if key else None
        if item is None:
            return [[("text", "(no item selected)")]]
        d = item.decision
        head: list[tui_core.StyledLine] = [
            [("h2", f"{item.kind} #{item.number}  {item.title}"[:width])],
        ]
        if item.url:
            head.append([("text", item.url[:width])])
        if d is None:
            return head + [[], [("text", f"({item.status.name.lower()}: no decision yet)")]]
        head += [
            [("bold", fairy.manual_action_description(d)[:width])],
            [("text", f"status {item.status.name.lower()}   llm "
                      f"{fairy.format_llm_classification(d.llm_classification)}   "
                      f"reason {d.reason}"[:width])],
            [],
        ]
        labels = [
            [("bullet", f"label {c.op} {c.label}"),
             ("text", (f" ({c.reason})" if c.reason else "") + (" [posted]" if c.post else ""))]
            for c in d.label_changes
        ]
        if labels:
            labels.append([])
        return head + labels + tui_core.render_markdown(d.llm_message, width)

    # ---- painting ----

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
            status = (f" {len(self.model.prompts)} pending |{STATUS_LINE}"
                      if self.model.prompts else STATUS_LINE)
        buf = []
        for pane, rect in rects.items():
            self._blit(buf, rect, pane, content[pane])
        divider = t.bold if self.drag else (lambda s: s)
        for y in range(row):
            buf.append(t.move_xy(col_t, y) + divider("|"))
        for y in range(row + 1, body_h):
            buf.append(t.move_xy(col_b, y) + divider("|"))
        buf.append(t.move_xy(0, row) + divider("-" * w))
        for col in {col_t, col_b}:
            buf.append(t.move_xy(col, row) + divider("+"))
        buf.append(t.move_xy(0, h - 1) + t.reverse(status[:w].ljust(w)))
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
        title = f" {PANES[pane]} "
        if pane == "tr":
            title += f"[{'all' if self.model.show_all else 'relevant'}] "
        bar = title[:rect.w].ljust(rect.w)
        buf.append(t.move_xy(rect.x, rect.y)
                   + (t.reverse(bar) if pane == self.focus else t.underline(bar)))
        for i in range(rect.h - 1):
            buf.append(t.move_xy(rect.x, rect.y + 1 + i))
            line = lines[i] if i < len(lines) else ""
            # sanitize(): forge/LLM text must not inject escape sequences.
            if isinstance(line, str):
                buf.append(tui_core.sanitize(line)[:rect.w].ljust(rect.w))
            elif line and isinstance(line[0], str):
                style, text = line
                text = tui_core.sanitize(text)[:rect.w].ljust(rect.w)
                fn = self.styles.get(style)
                buf.append(fn(text) if fn else text)
            else:
                buf.append(self._styled_line(line, rect.w))

    def _styled_line(self, segs: tui_core.StyledLine, width: int) -> str:
        out = []
        used = 0
        for style, text in segs:
            if used >= width:
                break
            text = tui_core.sanitize(text)[:width - used]
            fn = self.styles.get(style)
            out.append(fn(text) if fn else text)
            used += len(text)
        return "".join(out) + " " * (width - used)

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
                # 1 Hz heartbeat so the elapsed clock moves without input.
                if time.monotonic() - self._last_paint >= 1.0:
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
                self.model.show_all = not self.model.show_all
                self.model.cursor = 0
                self.list_top = 0
        elif str(ks) in CHOICE_KEYS:
            if not self.model.answer(CHOICE_KEYS[str(ks)]):
                logger.debug("key %r: no pending prompt under the cursor", str(ks))
        elif ks == "f":
            self.model.force()
        elif ks == "x":
            self.model.cancel()
        elif ks in ("e", "E"):
            self.export(full=(ks == "E"))
        else:
            return
        self.model.dirty.set()

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
                text = "\n".join(lines if full else self._scrolled("tl", lines, inner_h))
            elif pane == "tr":
                rows = self.list_rows()
                if not full:
                    rows = self._list_window(rows, inner_h)
                text = "\n".join(r for _, r in rows)
            else:
                lines = self.detail_lines(200 if full else max(8, self.term.width // 2))
                if not full:
                    lines = self._scrolled("br", lines, inner_h)
                text = "\n".join("".join(t for _, t in ln) for ln in lines)
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
        forge_gcli.logger, gcli_cache.logger, bot_state.logger, ci_log.logger,
        logger,
        handlers=[RingLogHandler(sink)],
    )
    for kind, _, ns, _forced in sides:
        if ns.approve:
            ns.approve = False
            logger.warning("--approve on the %s side is ignored: the TUI always "
                           "asks per decision", kind)
        logger.info("%s side enabled: %s/%s", kind, ns.owner, ns.repo)
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
