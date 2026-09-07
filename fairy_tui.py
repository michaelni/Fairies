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
(bottom left), rendered review message + label changes + the
discussion the review replied to (bottom right). Dividers move
with the mouse.

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
import signal
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
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread

import blessed
from blessed.dec_modes import DecPrivateMode

import agent
import branch_persist
import db_config
import diff_render
import fairy
import filedb
import git_util
import tui_core
import workset
from common import iso_to_dt, setup_logging, watch_paths

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
# A ticket found in no state dir is almost always a poll racing a
# rename (the destination dir listed before the write, the source
# after the unlink); a row only dies after GONE_POLLS consecutive
# misses, marked '?' in the list meanwhile.
GONE_POLLS = 10
_KIND_DISP = {"pr": "PR", "issue": "issue"}
_TTY_TIMEOUT = 0.2
_MOUSE_MODES = (DecPrivateMode.MOUSE_EXTENDED_SGR,
                DecPrivateMode.MOUSE_REPORT_DRAG)
# status sort: operator-actionable rows first, then the live pipeline
# states, then attention, then the settled ones
_SORT_STATES = {s: i for i, s in enumerate((
    "reviewed", INVALID, "llm", "queued", "outgoing", "requests",
    "merge-ready", "ci-blocked", "awaiting-approver",
    "error", "posted", "skipped", "cancelled"))}
assert set(_SORT_STATES) == set(filedb.STATES) | {INVALID}, \
    "every filedb state needs a sort priority"
# awaiting-approver IS merge-ready, just approved by someone other
# than fairy; the display folds them and the * marks the external case
_STATE_DISP = {"awaiting-approver": "merge-ready*"}
STATE_W = max(len(_STATE_DISP.get(s, s)) for s in _SORT_STATES)


@dataclass
class Item:
    repo: str            # side repo label, "owner/repo"
    kind: str            # filedb kind: "pr" | "issue"
    number: filedb.TicketId
    state: str           # filedb directory name, or "invalid"
    data: dict = field(default_factory=dict)  # last good ticket content
    error: str = ""      # why the current file fails to parse


def _repo_short(repo: str) -> str:
    """Display name for an "owner/repo" side label."""
    return repo.rsplit("/", 1)[-1]


def _ticket_branches(data: dict) -> list[dict]:
    """The verdict's collected branch records, [] when there are none."""
    return [b for b in (data.get("review") or {}).get("branches") or []
            if isinstance(b, dict)]


def _branch_mark(data: dict) -> tuple[str, str]:
    """The row's branch marker cell: orange when the review persists
    branches, red when one of them asks to open a PR."""
    branches = _ticket_branches(data)
    if not branches:
        return ("text", " ")
    if any(isinstance(b.get("pr"), dict) for b in branches):
        return ("br_pr_mark", "»")
    return ("br_mark", "»")


STATUS_FIELDS = ("state", "auto_merge", "approvals", "change_requests",
                 "labels")
_REPRO_CODES = {"repro/yes": ("sc_good", "Y"),
                "repro/flaky": ("sc_warn", "F"),
                "repro/no(env)": ("sc_warn", "n"),
                "repro/no": ("sc_bad", "N")}
_RESOLUTION_CODES = {"duplicate": ("sc_warn", "d"),
                     "external": ("sc_info", "e"),
                     "fixed": ("sc_good", "f"),
                     "invalid": ("sc_bad", "i"),
                     "wontfix": ("sc_dim", "w")}


def _count_col(count, style: str) -> tuple[str, str]:
    """A 0-9 review-count cell; blank when the snapshot has no count,
    dim when it is zero. ``count`` is snapshot JSON: int when present."""
    if count is None:
        return ("text", " ")
    return (style if count else "sc_dim", str(min(int(count), 9)))


def _status_cols(item: Item, snap: dict) -> tui_core.StyledLine:
    """The row's forge-status letters, 4 cells wide (README-TUI.md
    documents the codes). ``snap`` is the item's polled STATUS_FIELDS,
    {} when no snapshot exists yet -- every cell blank then. Closure
    outlives the snapshot (the scan stops refreshing a closed item),
    so the ticket's cancelled/ reason takes precedence over the
    snapshot's last-seen state."""
    reason = item.data.get("reason") if item.state == "cancelled" else None
    if item.kind == "pr":
        if snap.get("state") == "merged" or reason == agent.REASON_MERGED:
            auto = (snap.get("auto_merge")
                    or item.data.get("auto_merge")) == "merge"
            return [("sc_done", "M" if auto else "m"), ("text", "   ")]
        if snap.get("state") == "closed" \
                or reason == agent.REASON_CLOSED_UNMERGED:
            return [("sc_bad", "R"), ("text", "   ")]
        return [
            ("sc_info", "a") if snap.get("auto_merge") == "merge"
            else ("text", " "),
            _count_col(snap.get("approvals"), "sc_good"),
            _count_col(snap.get("change_requests"), "sc_bad"),
            ("text", " "),
        ]
    labels = snap.get("labels") or []
    resolution = next((l.removeprefix("resolution/") for l in labels
                       if l.startswith("resolution/")), None)
    return [
        ("sc_bad", "B") if {"bug", "regression"} & set(labels)
        else ("sc_info", "E") if "enhancement" in labels else ("text", " "),
        next((_REPRO_CODES[l] for l in _REPRO_CODES if l in labels),
             ("text", " ")),
        ("text", " ") if resolution is None
        else _RESOLUTION_CODES.get(resolution, ("sc_dim", "?")),
        ("sc_done", "C") if snap.get("state") == "closed"
        or reason == agent.REASON_CLOSED
        else ("sc_good", "O") if snap.get("state") == "open"
        else ("text", " "),
    ]


SORT_MODES = ("arrival", "status", "repo", "number")
# the a key cycles these lenses: the default working view, then one
# per attention surface (review/merge/CI are different jobs), any of
# them, and everything
FILTER_MODES = ("relevant", "review", "merge", "ci", "actionable", "all")
_FILTER_STATES = {
    # queued/llm rows are not reviewable yet, but they are on their way,
    # and a y-approved row stays as outgoing/ until the send: the review
    # lens shows the whole y-session, its near future and its aftermath
    "review": ("reviewed", INVALID, "llm", "queued", "outgoing"),
    "merge": ("merge-ready", "awaiting-approver"),
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
    "number": lambda it: (filedb.forge_number(it.number), it.number),
}


class Model:
    """Shared state between the 1 Hz directory poll and the painter.
    Every mutation happens under ``lock``; ``dirty`` wakes the painter.

    A side is a repo label plus its filedb; items are keyed by
    ``(repo, kind, number)`` -- two repos can share a PR number."""

    def __init__(self, sides: list[tuple[str, filedb.Db]]) -> None:
        self.lock = Lock()
        self.dirty = Event()
        self.requested: set[tuple[str, str, int]] = set()
        self.sides = sides
        self.items: dict[tuple[str, str, int], Item] = {}
        self.order: list[tuple[str, str, int]] = []
        self._read: dict[tuple[str, str, int], tuple[str, float]] = {}
        self.status: dict[tuple[str, str, filedb.TicketId], dict] = {}
        self._snap_read: dict[tuple[str, str, filedb.TicketId], float] = {}
        # Bumped under ``lock`` on every change that can alter list
        # membership; the O(N) visible/stats work is cached against it
        # so a repaint costs O(window) even with 100k tickets.
        self.revision = 0
        self._vis_cache: tuple[tuple, list[Item],
                               dict[tuple[str, str, int], int]] | None = None
        self._stats_agg: tuple[int, dict[str, dict[str, Counter]]] | None = None
        # (repo, state) -> (dir mtime, listing): an unchanged state dir
        # is not re-listed and its files are not re-stat'ed.
        self._dirs: dict[tuple[str, str],
                         tuple[float, list[tuple[str, filedb.TicketId]]]] = {}
        self.filter_mode = FILTER_MODES[0]
        self.sort_mode = SORT_MODES[0]
        # Rows the operator acted on (y/s/x) and rows seen in a live
        # state this session: they stay listed after settling so the
        # outcome is verifiable and "why did this one skip" has an
        # answer on screen.
        self.acted: set[tuple[str, str, int]] = set()
        self.seen_live: set[tuple[str, str, int]] = set()
        self.cursor = 0
        self.cursor_key: tuple[str, str, int] | None = None
        self.cursor_shown = False
        # items/ snapshot behind the cursor row (single slot: the
        # message pane is its only consumer); the slot names the
        # snapshot file as (repo, kind, forge number, mtime)
        self.snapshot: dict | None = None
        self._snapshot_slot: tuple[str, str, int, float] | None = None
        self.missing: dict[tuple[str, str, int], int] = {}
        self.quit_flag = False
        self.started = time.monotonic()

    def db(self, item: Item) -> filedb.Db:
        return self.db_for(item.repo)

    def db_for(self, repo: str) -> filedb.Db:
        return dict(self.sides)[repo]

    def poll(self) -> None:
        """Rescan every side's state directories: the files are the whole
        truth -- agent, workers and operator hand-edits all land here.
        File IO happens outside ``lock``.

        Every filedb write lands by rename INTO its state directory
        (content rewrites included), so a directory whose mtime has not
        moved needs no re-listing and none of its files re-stat'ed:
        steady state costs a dozen directory stats per side, not one
        stat per ticket.

        The items/ snapshots go through the same gate; only their
        STATUS_FIELDS are kept (into ``status``), feeding the list's
        status letters -- the message pane reads its one full snapshot
        through poll_snapshot instead."""
        now = time.time()
        listings: list[tuple[str, str, filedb.Db,
                             list[tuple[str, filedb.TicketId, float]] | None,
                             list[tuple[str, filedb.TicketId]]]] = []
        any_rescanned = False
        for repo, db in self.sides:
            for state in (*filedb.STATES, filedb.ITEM_STATE):
                try:
                    dir_mtime = (db.root / state).stat().st_mtime
                except OSError:
                    # a vanished dir IS a change: without the rescan flag
                    # the early return below would skip the miss counting
                    # forever and freeze the table
                    self._dirs.pop((repo, state), None)
                    any_rescanned = True
                    continue
                cached = self._dirs.get((repo, state))
                if cached is None or cached[0] != dir_mtime:
                    stat_listing = db.list_state_stat(state)
                    listing = [(k, n) for k, n, _ in stat_listing]
                    # Linux file timestamps come from the coarse clock: a
                    # rename in the same tick as this scan could leave the
                    # mtime unchanged, so a just-modified directory is
                    # never trusted as clean.
                    if dir_mtime < now - 2.0:
                        self._dirs[(repo, state)] = (dir_mtime, listing)
                    else:
                        self._dirs.pop((repo, state), None)
                    any_rescanned = True
                else:
                    stat_listing = None
                    listing = cached[1]
                listings.append((repo, state, db, stat_listing, listing))
        if not any_rescanned and not self.missing:
            # every dir unchanged: same membership, same content, same
            # request set -- the poll is over after a dozen stats, and
            # the per-ticket work below never scales with a quiet 100k
            # backlog
            return
        found: dict[tuple[str, str, int], str] = {}
        todo: list[tuple[tuple[str, str, int], str, filedb.Db, float]] = []
        snaps: list[tuple[tuple[str, str, filedb.TicketId],
                          filedb.Db, float]] = []
        requested: set[tuple[str, str, int]] = set()
        for repo, state, db, stat_listing, listing in listings:
            if state == filedb.ITEM_STATE:
                if stat_listing is not None:
                    snaps += (((repo, k, n), db, mt)
                              for k, n, mt in stat_listing)
                continue
            for kind, number in listing:
                # later directory wins: crash-remnant precedence
                found[(repo, kind, number)] = state
            if stat_listing is not None:
                todo += (((repo, k, n), state, db, mt)
                         for k, n, mt in stat_listing)
            if state == "requests":
                requested.update((repo, k, n) for k, n in listing)
        updates: list[tuple[tuple[str, str, int], str, dict | None, str]] = []

        def read_ticket(key: tuple[str, str, int], state: str,
                        db: filedb.Db, mtime: float) -> None:
            if self._read.get(key) == (state, mtime):
                return
            path = db.path(state, key[1], key[2])
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                # multi-MB payload; only its discussion is shown
                prepared = data.pop("prepared", None)
                if isinstance(prepared, dict) and prepared.get("discussion"):
                    data.setdefault("discussion", prepared["discussion"])
                for record in (data.get("review") or {}).get("branches") or []:
                    if isinstance(record, dict):
                        # up to MAX_BUNDLE_BYTES each; _branch_diff_body
                        # re-reads the ticket when it needs one
                        record.pop("bundle", None)
                updates.append((key, state, data, ""))
            except (FileNotFoundError, IsADirectoryError):
                return  # racing a rename; the next poll sees the new dir
            except (OSError, ValueError) as exc:
                updates.append((key, state, None, str(exc)))
            self._read[key] = (state, mtime)

        for key, state, db, mtime in todo:
            if found[key] != state:
                continue  # a later state dir won; its own entry decides
            read_ticket(key, state, db, mtime)
        status_updates: list[tuple[tuple[str, str, filedb.TicketId],
                                   dict]] = []
        for key, db, mtime in snaps:
            if self._snap_read.get(key) == mtime:
                continue
            try:
                snap = json.loads(db.path(filedb.ITEM_STATE, key[1], key[2])
                                  .read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(snap, dict):
                continue  # hand-edited/foreign file of unknown shape
            self._snap_read[key] = mtime
            status_updates.append(
                (key, {f: snap[f] for f in STATUS_FIELDS if f in snap}))
        dbs = dict(self.sides)
        removed = []
        for key, item in self.items.items():
            state = found.get(key)
            if state is not None:
                self.missing.pop(key, None)
                if item.state != state:
                    # the winning dir was served from cache while a
                    # remnant in a rescanned dir vanished: re-check the
                    # surviving file (tag-gated: one stat, rarely a read)
                    try:
                        mtime = dbs[key[0]].path(
                            state, key[1], key[2]).stat().st_mtime
                    except OSError:
                        continue
                    read_ticket(key, state, dbs[key[0]], mtime)
                continue
            misses = self.missing[key] = self.missing.get(key, 0) + 1
            if misses == 1:
                logger.debug("%s %s#%s is in no state dir; keeping the row "
                             "for %d grace polls", key[1], key[0], key[2],
                             GONE_POLLS)
            if misses >= GONE_POLLS:
                logger.info("%s %s#%s found nowhere for %d consecutive "
                            "polls; dropping the row", key[1], key[0],
                            key[2], misses)
                removed.append(key)
        updates.sort(key=lambda u: u[0])  # deterministic arrival order
        if not updates and not removed and not status_updates \
                and requested == self.requested:
            return
        with self.lock:
            self.revision += 1
            self.requested = requested
            self.status.update(status_updates)
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
                self.missing.pop(key, None)
                if key == self.cursor_key:
                    # a dead key must not teleport the cursor back if
                    # the item is re-ticketed hours later
                    self.cursor_key = None
            # warmed here, on the polling thread, so the repaint this
            # wakes finds them cached instead of paying the O(N) pass
            self.visible()
            self.reviewed_count()
        self.dirty.set()

    def poll_snapshot(self) -> None:
        """Refresh the items/ snapshot behind the cursor row when the
        row or its file changed; quiet otherwise, so calling this every
        UI tick costs a stat and never a repaint loop."""
        with self.lock:
            cursor = self._cursor_key() or self.cursor_key
            slot = None
            if cursor is not None:
                repo, kind, number = cursor
                base = str(filedb.forge_number(number))
                db = self.db_for(repo)
                try:
                    mtime = db.path(filedb.ITEM_STATE, kind,
                                    base).stat().st_mtime
                    slot = (repo, kind, filedb.forge_number(number), mtime)
                except OSError:
                    pass
            if slot == self._snapshot_slot:
                return
            self.snapshot = db.get(filedb.ITEM_STATE, kind, base) \
                if slot else None
            self._snapshot_slot = slot
        self.dirty.set()

    def snapshot_for(self, key: tuple[str, str, filedb.TicketId] | None
                     ) -> dict | None:
        """The polled snapshot, when it is of ``key``'s forge item
        (sample tickets share their base item's snapshot)."""
        if key is None or self._snapshot_slot is None:
            return None
        repo, kind, number = key
        if self._snapshot_slot[:3] == (repo, kind,
                                       filedb.forge_number(number)):
            return self.snapshot
        return None

    # ---- UI-thread side ----

    def visible(self) -> list[Item]:
        """Items for the list pane, honoring the all/relevant filter and
        the sort mode (stable, so arrival order breaks ties). Cached
        against (revision, filter, sort): repaints between changes cost
        nothing here. Caller holds ``lock``."""
        sig = (self.revision, self.filter_mode, self.sort_mode)
        if self._vis_cache is not None and self._vis_cache[0] == sig:
            return self._vis_cache[1]
        items = [self.items[k] for k in self.order]
        if self.filter_mode == "relevant":
            items = [it for it in items if self._relevant(it)]
        elif self.filter_mode != "all":
            allowed = _FILTER_STATES[self.filter_mode]
            items = [it for it in items if it.state in allowed]
        if self.sort_mode != "arrival":
            items.sort(key=_SORT_KEYS[self.sort_mode])
        self._vis_cache = (sig, items, {
            (it.repo, it.kind, it.number): i for i, it in enumerate(items)})
        return items

    def reviewed_count(self) -> int:
        """How many rows await the operator; cached like ``visible``.
        Caller holds ``lock``."""
        return sum(agg["by"]["reviewed"] for agg in self.side_stats().values())

    def side_stats(self) -> dict[str, dict[str, Counter]]:
        """Per-repo aggregates for the stats pane -- one pass over the
        items instead of one Counter pass per fact, cached against the
        revision and warmed on the polling thread. Keys per repo:
        ``kinds``, ``by`` (state), ``ready`` (postable action of
        reviewed rows), ``stages`` (llm), ``cls`` (verdicts), ``acts``
        (posted actions). Caller holds ``lock``."""
        if self._stats_agg is not None and self._stats_agg[0] == self.revision:
            return self._stats_agg[1]
        per = {repo: {name: Counter() for name in
                      ("kinds", "by", "ready", "stages", "cls", "acts")}
               for repo, _db in self.sides}
        for it in self.items.values():
            agg = per[it.repo]
            agg["kinds"][it.kind] += 1
            agg["by"][it.state] += 1
            data = it.data
            action = data.get("action")
            if it.state == "reviewed" and action in fairy.ACTIONABLE_DECISIONS:
                agg["ready"][action] += 1
            elif it.state == "llm":
                agg["stages"][data.get("stage") or "starting"] += 1
            elif it.state == "posted" and action:
                agg["acts"][action] += 1
            classification = (data.get("review") or {}).get("classification")
            if classification and classification != "-":
                agg["cls"][fairy.format_llm_classification(
                    classification)] += 1
        self._stats_agg = (self.revision, per)
        return per

    def cycle_sort(self) -> str:
        self.sort_mode = SORT_MODES[
            (SORT_MODES.index(self.sort_mode) + 1) % len(SORT_MODES)]
        return self.sort_mode

    def cycle_filter(self) -> str:
        self.filter_mode = FILTER_MODES[
            (FILTER_MODES.index(self.filter_mode) + 1) % len(FILTER_MODES)]
        return self.filter_mode

    def act(self, action: str, count: int = 1) -> None:
        """Execute a table action on the cursor row as a file operation:
        apply = reviewed -> outgoing (the agent's send pass posts it),
        skip/cancel = -> skipped/cancelled, rerun = a request the
        agent answers with a fresh gate-bypassing ticket. try_move never
        blocks: a row a worker holds, or one that changed under the
        cursor, refuses with a log line instead. No cursor anchor here:
        after y the cursor jumps to the next reviewed row instead of
        following the acted one."""
        with self.lock:
            key = self._cursor_key()
            item = self.items.get(key) if key else None
            if item is None:
                if self.cursor_key is not None:
                    logger.info("%s %s#%s is outside this lens; no action "
                                "taken -- move the cursor to select a row",
                                self.cursor_key[1], self.cursor_key[0],
                                self.cursor_key[2])
                return
            db = self.db(item)
            label = f"{_KIND_DISP[item.kind]} {item.repo}#{item.number}"
            if action == "sample":
                # R: one MORE evaluation next to whatever exists -- the
                # next free sample slot, never clobbering base or earlier
                # samples the way a repeated <n>r would
                base = filedb.forge_number(item.number)
                used = {0}
                for r, k, num in set(self.items) | self.requested:
                    if r == item.repo and k == item.kind \
                            and filedb.forge_number(num) == base:
                        used.add(filedb.sample_index(num))
                for k2, num in db.list_state("requests"):
                    # rapid presses: the model lags a poll behind the
                    # request files, and two R must not share a slot
                    if k2 == item.kind and filedb.forge_number(num) == base:
                        used.add(filedb.sample_index(num))
                nxt = max(used) + 1
                if nxt > 9:
                    logger.error("%s already has 9 evaluations", label)
                    return
                db.request(item.kind, f"{base}s{nxt}", {"action": "rerun"})
                self.seen_live.add(key)
                logger.info("requested one more evaluation of %s (slot s%d)",
                            label, nxt)
            elif action == "rerun":
                if item.state in ("queued", "llm", "outgoing"):
                    logger.info("%s is already in flight", label)
                    return
                if count >= 10:
                    # fat-finger guard: a stray count must not spawn a
                    # gigantic evaluation batch
                    logger.error("%s: refusing %d evaluations (max 9)",
                                 label, count)
                    return
                if count == 1:
                    db.request(item.kind, item.number, {"action": "rerun"})
                    logger.info("requested a fresh review of %s", label)
                else:
                    base = filedb.forge_number(item.number)
                    for i in range(1, count + 1):
                        db.request(item.kind, f"{base}s{i}",
                                   {"action": "rerun"})
                    logger.info("requested %d sample evaluations of %s",
                                count, label)
                self.seen_live.add(key)
            elif action in ("apply", "apply-force"):
                if item.state != "reviewed" or not agent.postable(
                        agent.ticket_decision(item.kind, item.number, item.data)):
                    logger.info("%s has nothing to post (state %s, llm %s)",
                                label, item.state,
                                (item.data.get("review") or {}).get(
                                    "classification", "-"))
                    return
                force = action == "apply-force"
                if not db.try_move("reviewed", "outgoing", item.kind,
                                   item.number,
                                   mutate=(lambda d: d.update(force_post=True))
                                   if force else None):
                    logger.info("%s changed under the cursor; not applied", label)
                    return
                self.acted.add(key)
                logger.info("%s -> outgoing/ (%s)", label,
                            "Y: posts even if the item moved since the review"
                            if force else "the agent's send pass posts it")
                self._advance_to_reviewed(key)
            elif action == "cancel" and item.state == "llm":
                workset.update_json(db.path("llm", item.kind, item.number), lambda d: d.update(cancel=True, reason="operator cancel"))
            elif action in ("skip", "snooze", "cancel"):
                dst = "cancelled" if action == "cancel" else "skipped"
                if item.state in ("llm", INVALID) or item.state in HIDDEN_SETTLED:
                    logger.info("cannot %s %s in state %s%s", action, label,
                                item.state,
                                " (fix or delete the file by hand)"
                                if item.state == INVALID else "")
                    return
                note = {"reason": "operator " + action}
                if action == "snooze":  # a real snooze even on old verdicts
                    note["snoozed_at"] = datetime.now(timezone.utc).isoformat()

                def mutate(d: dict) -> None:
                    d.update(note)
                    if action == "skip":
                        # no timestamps: the next scan reconsiders it
                        d.pop("llm_at", None)
                        d.pop("snoozed_at", None)

                if not db.try_move(item.state, dst, item.kind, item.number,
                                   mutate=mutate):
                    logger.info("%s is busy or changed; not %sed", label, action)
                    return
                self.acted.add(key)
                logger.info("%s -> %s/", label, dst)
            self.revision += 1
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
        keys = [(it.repo, it.kind, it.number) for it in self._sync_cursor()]
        nxt = next((k for k in keys[self.cursor:] if k in remaining),
                   next((k for k in keys if k in remaining), None))
        if nxt is not None:
            self._move_cursor_to(nxt)

    def _relevant(self, item: Item) -> bool:
        key = (item.repo, item.kind, item.number)
        return (item.state not in HIDDEN_SETTLED
                or key in self.acted or key in self.seen_live)

    def _sync_cursor(self) -> list[Item]:
        """The KEY is the cursor; the index is only where it paints
        this frame. Invariant: the highlight is ONLY ever drawn on the
        key's own row -- when the lens does not show the key, no row is
        highlighted and actions refuse, never landing on a ticket the
        operator did not pick. ``cursor`` keeps the last shown index
        purely as the landing spot for the next arrow press. Without
        any key (startup, or the key's item was dropped) the first
        listed ticket is adopted. Caller holds ``lock``; returns the
        visible list."""
        vis = self.visible()
        if not vis:
            self.cursor_shown = False
            return vis
        index = self._vis_cache[2]
        if self.cursor_key is None:
            it = vis[0]
            self.cursor_key = (it.repo, it.kind, it.number)
        self.cursor_shown = self.cursor_key in index
        if self.cursor_shown:
            self.cursor = index[self.cursor_key]
        return vis

    def select_index(self, i: int) -> None:
        """An explicit operator move (arrow, click, search hit): the row
        at index ``i`` of the visible list becomes the cursor key."""
        vis = self.visible()
        if not vis:
            self.cursor, self.cursor_key = 0, None
            self.cursor_shown = False
            return
        self.cursor = max(0, min(i, len(vis) - 1))
        it = vis[self.cursor]
        self.cursor_key = (it.repo, it.kind, it.number)
        self.cursor_shown = True

    def _cursor_key(self) -> tuple[str, str, int] | None:
        vis = self._sync_cursor()
        if not self.cursor_shown:
            return None
        it = vis[self.cursor]
        return (it.repo, it.kind, it.number)

    def _move_cursor_to(self, key: tuple[str, str, int]) -> None:
        self.cursor_key = key
        self._sync_cursor()

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


def _age(iso: str | None, now: datetime | None = None) -> str:
    """Compact age of an ISO timestamp: 12d for 12 days, 5h below a day."""
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = max(0.0, ((now or datetime.now(timezone.utc)) - then).total_seconds())
    return f"{int(seconds // 86400)}d" if seconds >= 86400 \
        else f"{int(seconds // 3600)}h"


def _when(iso: str | None) -> str:
    """``2026-07-29 06:15 (3h ago)`` in local time; '' without a stamp."""
    if not iso:
        return ""
    try:
        then = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return f"{then.astimezone().strftime('%Y-%m-%d %H:%M')} ({_age(iso)} ago)"


def _line_level(line: str) -> int | None:
    """Level of an ``ISO8601 L message`` log line (add_file_log's shape),
    None for anything else."""
    parts = line.split(" ", 2)
    if len(parts) > 2 and len(parts[1]) == 1:
        return _TAIL_LEVELS.get(parts[1])
    return None


def _tag_width(stems: list[str]) -> int:
    """Column width for the tail-source tags: the shortest prefix that
    keeps the stems distinct, plus 4, capped at the longest stem."""
    if not stems:
        return 0
    longest = max(len(s) for s in stems)
    for n in range(1, longest + 1):
        if len({s[:n] for s in stems}) == len(set(stems)):
            return min(n + 4, longest)
    return longest


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
        width = _tag_width([p.stem for p in paths])
        self._tag = {p: f"{p.stem[:width]:<{width}}" for p in paths}

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
                # four repos, one pane: "pr #7" alone names no repo
                self.sink.line(f"{self._tag[path]}: {ln}", _line_level(ln))


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
ACTION_KEYS = {"y": "apply", "Y": "apply-force", "s": "skip", "S": "snooze",
               "r": "rerun", "R": "sample", "x": "cancel"}
KEYMAP = (("q", "quit"), ("y", "apply"), ("Y", "post anyway"), ("s", "skip"),
          ("S", "snooze"), ("r", "rerun"), ("R", "+eval"),
          ("x", "drop"), ("o", "edit msg"), ("p", "pause"), ("a", "filter"),
          ("t", "sort"), ("m", "diff"), ("d", "logs diff"),
          ("b", "branch diff"),
          ("[/] {/}", "patch/hunk"), ("/", "search"), ("e/E", "export"),
          ("?", "help"), ("Tab/click", "focus"),
          ("↑↓ PgUp/PgDn Home/End", "scroll"))
DIFF_MODES = ("patches", "merge diff")
BRANCH_DIFF_MODES = ("commits", "range-diff")
# Rendered-diff cache entries: the two diff panes plus one review's
# persisted-branch diffs (up to MAX_PERSIST_BRANCHES) fit without
# re-running git every repaint.
_DIFF_CACHE_ENTRIES = 8
DETAIL_MODES = ("message", *DIFF_MODES)
LOGS_MODES = ("logs", *DIFF_MODES)
DIFF_MAX_LINES = 20_000


def _next_mode(modes: tuple[str, ...], current: str) -> str:
    return modes[(modes.index(current) + 1) % len(modes)]


def _proc_children(pid: int) -> list[int]:
    kids: list[int] = []
    for task in Path(f"/proc/{pid}/task").iterdir():
        kids += [int(c) for c in (task / "children").read_text().split()]
    return kids


def _proc_tree(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        out.append(p)
        try:
            todo += _proc_children(p)
        except OSError:
            pass
    return out


def session_pids(parent: int | None = None) -> list[int]:
    """This session's agent/worker processes plus their whole subprocess
    trees (wrappers, ssh, podman clients). The launcher shell is our own
    parent, so its children whose command line names agent.py or
    worker.py are exactly the daemons started next to us; anything else
    sharing the parent (an unrelated job under an interactive shell)
    never matches."""
    me = os.getpid()
    pids: list[int] = []
    try:
        siblings = _proc_children(os.getppid() if parent is None else parent)
    except OSError:
        return []
    for pid in siblings:
        if pid == me:
            continue
        try:
            argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except OSError:
            continue
        if "agent.py" in argv or "worker.py" in argv:
            pids += _proc_tree(pid)
    return pids


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
        styles = {
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
            "comment": c(244),
            "hr": c(240),
            "table_border": c(240),           "th": mix(t.bold, c(223)),
            "divider": c(240),                "divider_drag": mix(t.bold, c(81)),
            "bar_focus": mix(t.bold, c(231), on(25)), "bar_blur": mix(c(245), on(236)),
            "key": mix(t.bold, c(81)),        "label": c(245),
            "num": t.bold,                    "mark": mix(t.bold, c(203)),
            "br_mark": mix(t.bold, c(208)),   "br_pr_mark": mix(t.bold, c(196)),
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
            "sc_good": c(78),                 "sc_bad": c(203),
            "sc_warn": c(179),                "sc_info": c(75),
            "sc_done": c(135),                "sc_dim": c(244),
            "sampled": c(114),
            "cursor": t.reverse,
            # log-pane levels; palette mirrors common._ColorFormatter
            "log_debug": t.dim_bright_black,  "log_warn": t.bold_yellow,
            "log_err": t.bold_red,
            "diff_file": mix(t.bold, c(117)), "diff_hunk": c(73),
            "diff_meta": c(244),              "diff_commit": mix(t.bold, c(179)),
        }
        for fg_name, fg in zip(diff_render.DIFF_FGS, (
                c(252), c(176), c(116), c(75), c(114), c(245), c(179))):
            for bg_name, bg in {"ctx": None, "add": on(22), "del": on(52),
                                "addhl": on(28), "delhl": on(88)}.items():
                styles[f"df_{bg_name}_{fg_name}"] = mix(bg, fg) if bg else fg
        return styles
    styles = {
        "h1": t.bold_magenta,   "h2": t.bold_blue,  "h3": t.bold_cyan,
        "h4": t.cyan,           "bold": t.bold,     "italic": t.italic,
        "bold_italic": t.bold,  "strike": strike,
        "code": t.reverse,      "codeblock": t.reverse,
        "codeblock_lang": t.bright_black,
        "quote": t.bright_black, "quote_bar": t.magenta,
        "bullet": t.bold,
        "checkbox_on": t.green, "checkbox_off": t.bright_black,
        "link": t.underline_blue, "url": t.bright_black,
        "comment": t.bright_black,
        "hr": t.bright_black,
        "table_border": t.bright_black, "th": t.bold,
        "divider": t.bright_black, "divider_drag": t.bold_cyan,
        "bar_focus": t.reverse, "bar_blur": t.underline,
        "key": t.bold_cyan,     "label": t.bright_black,
        "num": t.bold,          "mark": t.bold_red,
        "br_mark": t.bold_yellow, "br_pr_mark": t.bold_red,
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
        "sc_good": t.green,     "sc_bad": t.red,
        "sc_warn": t.yellow,    "sc_info": t.cyan,
        "sc_done": t.magenta,   "sc_dim": t.bright_black,
        "sampled": t.green,
        "cursor": t.reverse,
        "log_debug": t.dim_bright_black, "log_warn": t.bold_yellow,
        "log_err": t.bold_red,
        "diff_file": t.bold,    "diff_hunk": t.cyan,
        "diff_meta": t.bright_black, "diff_commit": t.bold_yellow,
    }
    # 16 colors cannot carry syntax foregrounds over add/del backgrounds:
    # the whole line takes the diff color, word marks reverse it, and
    # context lines stay plain (None renders unstyled).
    for bg_name, fn in {"ctx": None, "add": t.green, "del": t.red,
                        "addhl": mix(t.reverse, t.green),
                        "delhl": mix(t.reverse, t.red)}.items():
        for fg_name in diff_render.DIFF_FGS:
            styles[f"df_{bg_name}_{fg_name}"] = fn
    return styles


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
                 save_dir: Path, tail: LogTail,
                 patch_repos: dict[str, Path] | None = None) -> None:
        self.term = term
        self.model = model
        self.ring = ring
        self.save_dir = save_dir
        self.tail = tail
        self.patch_repos = patch_repos or {}
        self.detail_mode = DETAIL_MODES[0]
        self.logs_mode = LOGS_MODES[0]
        self.branch_diff_mode = BRANCH_DIFF_MODES[0]
        self._diff_cache: dict[
            tuple, tuple[tui_core.Chain, float | None]] = {}
        repos = [repo for repo, _ in model.sides]
        short = [_repo_short(r) for r in repos]
        self._repo_disp = {r: (s if short.count(s) == 1 else r)
                           for r, s in zip(repos, short)}
        # 0 hides every repo label: labels only disambiguate when the
        # sides span more than one repo.
        self._repo_w = (max(len(d) for d in self._repo_disp.values())
                        if len(repos) > 1 else 0)
        self.layout = tui_core.GridLayout()
        self._keys: SimpleQueue = SimpleQueue()
        self.focus = "tr"
        self.scroll = {"tl": 0, "bl": 0, "br": 0}
        self.list_top = 0
        self.drag: str | None = None
        self.follow_cursor = True
        self._last_size: tuple[int, int] | None = None
        self._last_paint = 0.0
        # pane -> (rect, gutter, unclipped plain rows) as last painted;
        # click-to-copy resolves the token under the mouse from this.
        self._shown: dict[str, tuple[tui_core.Rect, int, list[str]]] = {}
        # pane -> the line sequence last painted into it; [] and {}
        # ask it where its patches and hunks start
        self._painted: dict[str, list] = {}
        self._clip_cmd = _clipboard_cmd()
        self.styles = _styles(term)
        self.needs_poll = Event()  # set by the filesystem watcher
        self.help_text: str | None = None
        self.help_scroll = 0
        self.count_buf = ""     # 0-9 prefix for r/f and arrow scrolling
        self.search_mode = False
        self.search_buf = ""
        self.last_search = ""
        self.paused: list[int] = []
        self._stdin_free = Event()
        self._stdin_free.set()
        self._stdin_reader = Lock()
        self._stats_cache: tuple[tuple[int, int],
                                 list[tui_core.StyledLine]] | None = None
        self._build_status()

    def _build_status(self) -> None:
        """(Re)build the bottom-bar legend; the ``t`` entry names the
        current sort mode."""
        key = self.styles.get("key") or (lambda s: s)
        label = self.styles.get("label") or (lambda s: s)
        self._status_mode = (self.model.sort_mode, self.model.filter_mode,
                             bool(self.paused))
        pairs = [(k, f"sort:{self.model.sort_mode}" if k == "t"
                  else f"filter:{self.model.filter_mode}" if k == "a"
                  else ("resume" if self.paused else "pause") if k == "p"
                  else d)
                 for k, d in KEYMAP]
        self._status_plain = "  ".join(f"{k} {d}" for k, d in pairs)
        self._status_styled = "  ".join(f"{key(k)} {label(d)}" for k, d in pairs)

    # ---- pane content (caller holds model.lock) ----

    def stats_lines(self, width: int) -> list[tui_core.StyledLine]:
        """Full-width header, then one block per repo tiled into as many
        columns as ``width`` fits. The counts are simply the number of
        files per state directory. The per-repo blocks are cached
        against (model revision, width); only the ticking header is
        rebuilt per paint."""
        m = self.model
        actionable = m.reviewed_count()
        header: tui_core.StyledLine = [
            ("label", "elapsed "),
            ("num", f"{int(time.monotonic() - m.started)}s"),
            ("label", "   reviewed, awaiting you "),
            ("st_reviewed" if actionable else "num", str(actionable)),
        ]
        if self.paused:
            header[:0] = [("log_err", "PAUSED   ")]
        sig = (m.revision, width)
        if self._stats_cache is not None and self._stats_cache[0] == sig:
            return [header, []] + self._stats_cache[1]
        blocks: list[list[tui_core.StyledLine]] = []
        for repo, agg in m.side_stats().items():
            kinds = agg["kinds"]
            block: list[tui_core.StyledLine] = [[
                ("title", f"{self._repo_disp[repo]}  "),
                ("kind_pr", f"{kinds['pr']} PRs"),
                ("text", "  "),
                ("kind_issue", f"{kinds['issue']} issues"),
            ]]
            by = agg["by"]
            if by:
                counts: list[tuple[str, str]] = []
                for s in (*filedb.STATES, INVALID):
                    if s == "awaiting-approver":
                        continue  # folded into merge-ready=N/M below
                    if s == "merge-ready" and by["awaiting-approver"]:
                        counts.append((s, f"{by[s]}/{by['awaiting-approver']}"))
                    elif by[s]:
                        counts.append((s, str(by[s])))
                row: tui_core.StyledLine = [("text", "  ")]
                used = 2
                for s, n in counts:
                    part_len = len(s) + 1 + len(n) + 1
                    if used > 2 and used + part_len > width:
                        block.append(row)
                        row = [("text", "  ")]
                        used = 2
                    row += [(f"st_{s}", f"{s}="), ("num", n), ("text", " ")]
                    used += part_len
                block.append(row)
            # what could be applied right now, by action -- distinct
            # from the review/investigate rows
            if agg["ready"]:
                block.append([("text", "  "), ("label", "awaiting you: "),
                              ("st_reviewed", ", ".join(
                                  f"{k}={v}" for k, v in
                                  sorted(agg["ready"].items())))])
            if agg["stages"]:
                block.append([("text", "  "), ("label", "llm stage: "), ("st_llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(agg["stages"].items())))])
            if agg["cls"]:
                block.append([("text", "  "), ("label", "llm: "), ("llm",
                    ", ".join(f"{k}={v}" for k, v in sorted(agg["cls"].items())))])
            if agg["acts"]:
                block.append([("text", "  "), ("label", "posted: "), ("st_posted",
                    ", ".join(f"{k}={v}" for k, v in sorted(agg["acts"].items())))])
            blocks.append(block)
        self._stats_cache = (sig, tui_core.tile_blocks(blocks, width))
        return [header, []] + self._stats_cache[1]

    def _llm_col(self, it: Item) -> str:
        if it.state == "llm":
            return it.data.get("stage") or "llm"
        if (it.repo, it.kind, it.number) in self.model.requested:
            return "requested"
        if it.state == "merge-ready" and it.data.get("approved_at"):
            return f"appr={_age(it.data['approved_at'])}"
        if it.state == "error":
            age = _age(it.data.get("llm_at") or it.data.get("state_changed_at"))
            return f"err={age}" if age else ""
        review = it.data.get("review") or {}
        if review.get("classification"):
            return fairy.format_llm_classification(review["classification"])
        return ""

    def _row(self, it: Item, is_cursor: bool) -> list | tuple:
        m = self.model
        llm = self._llm_col(it)
        mark = "▶" if it.state == "reviewed" else " "
        repo_col = (f"{self._repo_disp[it.repo]:<{self._repo_w}} "
                    if self._repo_w else "")
        act = _age(it.data.get("last_activity_iso")
                   or it.data.get("expected_updated_at"))
        title = it.data.get("title") or ""
        state_disp = _STATE_DISP.get(it.state, it.state)
        if it.state == "cancelled" \
                and it.data.get("reason") == agent.REASON_MERGED:
            state_disp = "merged"
        if it.state in ("queued", "llm") and it.data.get("forced"):
            state_disp += "+"
        if (it.repo, it.kind, it.number) in m.missing:
            state_disp += "?"
        status = _status_cols(it, m.status.get(
            (it.repo, it.kind, str(filedb.forge_number(it.number)))) or {})
        branch_mark = _branch_mark(it.data)
        if is_cursor:
            return ("cursor",
                    f"{mark}{_KIND_DISP[it.kind]:<5} {repo_col}#{it.number:<7} "
                    f"{''.join(c for _, c in status)}{branch_mark[1]} "
                    f"{state_disp:<{STATE_W}} {llm:<9} {act:>4}  {title}")
        return [
            ("mark", mark),
            ("kind_pr" if it.kind == "pr" else "kind_issue",
             f"{_KIND_DISP[it.kind]:<5} "),
            ("label", repo_col),
            ("num", f"#{it.number:<7} "),
            *status,
            branch_mark,
            ("text", " "),
            (f"st_{it.state}", f"{state_disp:<{STATE_W}} "),
            ("llm", f"{llm:<9} "),
            ("num", f"{act:>4}  "),
            ("title", title),
        ]

    def list_rows(self) -> list:
        """Every visible row, formatted; the e/E exports and tests read
        this. Painting goes through _list_window instead, which formats
        only the rows on screen."""
        m = self.model
        vis = m._sync_cursor()
        return [self._row(it, i == m.cursor and m.cursor_shown)
                for i, it in enumerate(vis)]

    def detail_lines(
            self, width: int) -> list[tui_core.StyledLine] | tui_core.Chain:
        m = self.model
        # an off-lens cursor still has a key: keep showing its ticket,
        # so the operator sees where it went (e.g. "state posted")
        key = m._cursor_key() or m.cursor_key
        item = m.items.get(key) if key else None
        if item is None:
            return [[("text", "(no item selected)")]]
        data = item.data
        # the items/ snapshot outdates the ticket wherever both carry a
        # field, so what it has wins
        snapshot = m.snapshot_for(key)
        shown = data if snapshot is None else snapshot
        title = shown.get("title") or data.get("title") or ""
        url = shown.get("html_url") or data.get("html_url")
        author = shown.get("author") or data.get("author")
        where = f"{item.repo}#{item.number}" if self._repo_w else f"#{item.number}"
        head: list[tui_core.StyledLine] = [
            [("h2", f"{_KIND_DISP[item.kind]} {where}  {title}"[:width])],
        ]
        if url:
            head.append([("link", str(url)[:width])])
        if author or data.get("head_branch"):
            byline = [("label", "author "), ("text", str(author or "?"))]
            if data.get("head_branch"):
                byline += [("label", "   branch "),
                           ("text", str(data["head_branch"])[:width])]
            head.append(byline)
        if item.error:
            head += tui_core.wrap([("log_err", str(item.error))], width,
                                  initial=("log_err", "file invalid: "))
        if data.get("error"):
            when = _when(data.get("llm_at") or data.get("state_changed_at"))
            head += tui_core.wrap(
                [("log_err", str(data["error"]))], width,
                initial=("log_err", f"error {when}: " if when else "error: "))
        if data.get("send_blocked"):
            head += tui_core.wrap(
                [("log_warn", str(data["send_blocked"]))], width,
                initial=("log_warn", "send blocked: "))
        for failed in data.get("failed_reviewers") or []:
            head += tui_core.wrap([("log_warn", str(failed))], width,
                                  initial=("log_warn", "reviewer failed: "))
        if self.detail_mode in DIFF_MODES and item.kind == "pr":
            return tui_core.Chain(
                head, self._diff_body(item, snapshot, self.detail_mode))
        review = data.get("review") or {}
        decision = agent.ticket_decision(item.kind, item.number, data)
        if decision is not None:
            # render_markdown, not a clipped line: long apply
            # descriptions and reasons must wrap, never silently vanish
            head += tui_core.render_markdown(
                fairy.manual_action_description(decision), width)
        status_line = f"state {item.state}"
        if review.get("classification"):
            status_line += ("   llm "
                            + fairy.format_llm_classification(review["classification"]))
            if data.get("llm_at"):
                status_line += "   reviewed " + _when(data["llm_at"])
        head += [[("text", status_line[:width])]]
        if data.get("reason"):
            head += tui_core.render_markdown(f"reason: {data['reason']}", width)
        for fieldname in ("cancelled_ci_contexts", "blocked_ci_contexts",
                          "external_approvers"):
            vals = data.get(fieldname) or []
            if vals:
                head.append([("label", fieldname.replace("_", " ") + ": "),
                             ("text", ", ".join(map(str, vals))[:width])])
        head.append([])
        tail: list[tui_core.StyledLine] = []
        disc = shown.get("discussion") or []
        body = shown.get("body") or data.get("body")
        if body:
            disc = [{"kind": "description", "author": author,
                     "body": body}] + disc

        def entries_before(when: datetime) -> int:
            return sum(1 for entry in disc if not isinstance(entry, dict)
                       or fairy.discussion_time(entry) <= when)

        separator_at = None
        if snapshot is not None:
            sampled = iso_to_dt(str(data.get("expected_updated_at") or ""))
            if sampled is not None:
                separator_at = entries_before(sampled)
        review_lines = self._review_lines(item, width)
        review_at = len(disc)
        if item.state == "posted" and review_lines:
            posted = iso_to_dt(str(data.get("posted_at") or ""))
            if posted is not None:
                review_at = entries_before(posted)
        if disc:
            tail += [[], [("h3", f"discussion ({len(disc)})"[:width])]]
            separator = [("sampled", (
                f"── sampled {'for the review ' if review else ''}"
                f"{_when(data.get('expected_updated_at'))} ──")[:width])]
            for i, c in enumerate(disc):
                if i == separator_at:
                    tail += [[], separator]
                if i == review_at:
                    tail += review_lines
                if not isinstance(c, dict):
                    continue
                when = _when(str(c.get("submitted_at") or c.get("updated_at")
                                 or c.get("created_at") or ""))
                if c.get("kind") == "push":
                    what = ("force-pushed" if c.get("is_force_push") else "pushed") \
                        + f" {c.get('commit_count')} commit(s) {str(c.get('head_sha') or '')[:10]}"
                elif c.get("kind") == "review_request":
                    what = (f"withdrew the review request for {c.get('reviewer')}"
                            if c.get("removed")
                            else f"requested a review from {c.get('reviewer')}")
                else:
                    what = " ".join(str(c[k]) for k in ("kind", "state") if c.get(k))
                    if c.get("path"):
                        what += f"  {c['path']}" \
                            + (f":{c['line']}" if c.get("line") is not None else "")
                tail.append([])
                tail.append([("h4", str(c.get("author") or "?")),
                             ("label", f"  {what}  {when}"[:width])])
                if c.get("body"):
                    tail += tui_core.render_markdown(str(c["body"]), width)
                for att in c.get("attachment_urls") or []:
                    tail.append([("link", str(att)[:width])])
            if separator_at == len(disc):
                tail += [[], separator]
        if review_at == len(disc):
            tail += review_lines
        if not review:
            return head + [[("text", f"({item.state}: no review)")]] + tail
        if not review_lines:
            return head + tail
        fairy_block = head + tail
        branch_parts: list = []
        for record in _ticket_branches(data):
            pr = record.get("pr")
            style = "br_pr_mark" if isinstance(pr, dict) else "br_mark"
            if record.get("mode") == "delete":
                desc = (f"» {record.get('repo')}: delete "
                        f"{branch_persist.FAIRY_BRANCH_PREFIX}{record.get('branch')}")
                branch_parts.append([
                    [],
                    [(style, desc[:width])],
                    [("label", (f"  was {str(record.get('sha'))[:12]}"
                                "  deleted when the review is sent (y)")[:width])],
                ])
                continue
            desc = (f"» {record.get('repo')}: "
                    f"{branch_persist.FAIRY_BRANCH_PREFIX}{record.get('branch')}"
                    f"  {record.get('mode')}")
            if isinstance(pr, dict):
                desc += f'  → PR "{pr.get("title")}" into {pr.get("target")}'
            branch_parts.append([
                [],
                [(style, desc[:width])],
                [("label", (f"  {str(record.get('sha'))[:12]}"
                            "  published when the review is sent (y)")[:width])],
            ])
            if isinstance(pr, dict) and pr.get("body"):
                branch_parts.append(
                    tui_core.render_markdown(str(pr["body"]), width))
            branch_parts.append(self._branch_diff_body(item, record))
        if not branch_parts:
            return fairy_block
        return tui_core.Chain(fairy_block, *branch_parts)

    def _review_lines(self, item: Item, width: int) -> list[tui_core.StyledLine]:
        """The ticket's review as a thread entry: a ``fairy`` header
        with the posted / NOT POSTED byline, the label changes and the
        message; empty when there is nothing to show."""
        data = item.data
        review = data.get("review") or {}
        if not review:
            return []
        labels = tui_core.render_markdown("\n".join(
            f"- {c.get('op')} **{c.get('label')}**"
            + (f" — {c.get('reason')}" if c.get("reason") else "")
            + (" *[posted]*" if c.get("post") else "")
            for c in review.get("label_changes") or []
            if isinstance(c, dict)), width)
        message = tui_core.render_markdown(review.get("message") or "",
                                           width)
        if not labels and not message and not _ticket_branches(data):
            return []
        w = width - len("fairy")
        if item.state == "posted":
            when = _when(data.get("posted_at") or data.get("llm_at"))
            byline = [("label", f"  review  {when}"[:w])]
        else:
            byline = [("st_reviewed", ("  review — NOT POSTED  "
                                       + _when(data.get("llm_at")))[:w])]
        if labels:
            labels.append([])
        return [[], [("h4", "fairy")] + byline] + labels + message

    def _branch_diff_body(self, item: Item, record: dict) -> tui_core.Chain:
        """The commits a persisted branch would publish -- git log with
        patches from its recorded diff base to its tip, materialized
        from the record's bundle -- or, when ``b`` selected range-diff
        mode, for a force push whose published fairy/<name> tip is
        resolvable in the patch repo, the range-diff against that tip.
        Cached like ``_diff_body``; failures retry after a few
        seconds."""
        branch = str(record.get("branch") or "")
        sha = str(record.get("sha") or "")
        base = record.get("diff_base_sha")
        want_range_diff = (self.branch_diff_mode == "range-diff"
                           and record.get("mode") == "force")
        cache_key = ("branch", item.repo, sha, base, want_range_diff)
        cached = self._diff_cache_get(cache_key)
        if cached is not None:
            return cached
        if "bundle" not in record:
            # the model strips bundles from resident tickets
            ticket = self.model.db_for(item.repo).get(
                item.state, item.kind, item.number) or {}
            record = next(
                (b for b in _ticket_branches(ticket)
                 if b.get("branch") == branch and b.get("sha") == sha),
                record)
        patch_repo = self.patch_repos.get(item.repo)
        old_tip = git_util.git_resolve_first(
            patch_repo,
            [f"{remote}/{branch_persist.FAIRY_BRANCH_PREFIX}{branch}"
             for remote in branch_persist.FAIRY_BRANCH_REMOTES]
        ) if want_range_diff and patch_repo is not None else None
        retry_at = None
        caption = body = tail = None
        try:
            with branch_persist.materialized_record(
                    record, extra_objects=patch_repo) as store:
                if old_tip:
                    try:
                        raw = git_util.git_range_diff(store, old_tip, sha)
                        caption = [("label",
                                    f"range-diff {old_tip[:12]}...{sha[:12]}")]
                        text_lines = raw.decode("utf-8", errors="replace").splitlines()
                        body = [[("text", line)]
                                for line in text_lines[:DIFF_MAX_LINES]]
                        if len(text_lines) > DIFF_MAX_LINES:
                            tail = [[("log_warn",
                                      f"… truncated at {DIFF_MAX_LINES}"
                                      f" of {len(text_lines)} lines")]]
                    except RuntimeError as exc:
                        # e.g. the published tip's objects live in neither
                        # the record's checkout nor the patch repo
                        logger.info("range-diff unavailable (%s); showing "
                                    "the commits", exc)
                if body is None and isinstance(base, str) and base:
                    caption = [("label", f"branch commits {base[:12]}..{sha[:12]}")]
                    text_lines = git_util.git_patch_stream(store, base, sha) \
                        .decode("utf-8", errors="replace").split("\n")
                    body = diff_render.DiffView(
                        "\n".join(text_lines[:DIFF_MAX_LINES])) \
                        or [[("label", "(no commits)")]]
                    if len(text_lines) > DIFF_MAX_LINES:
                        tail = [[("log_warn", f"… truncated at {DIFF_MAX_LINES}"
                                              f" of {len(text_lines)} lines")]]
                elif body is None:
                    caption = [("label", "branch commits")]
                    body = [[("log_warn", "no diff base recorded for this branch")]]
        except RuntimeError as exc:
            caption = [("label", "branch commits")]
            logger.error("%s", exc)
            retry_at = time.monotonic() + 5
            body = [[("log_err", line)] for line in str(exc).splitlines()]
        return self._diff_cache_put(
            cache_key, tui_core.Chain([caption], body, tail or []), retry_at)

    def _diff_cache_get(self, cache_key: tuple) -> tui_core.Chain | None:
        """A still-valid cached rendering; re-inserting keeps the entry
        newest, so eviction takes the least recently shown one."""
        entry = self._diff_cache.pop(cache_key, None)
        if entry is not None:
            cached, valid_until = entry
            if valid_until is None or time.monotonic() < valid_until:
                self._diff_cache[cache_key] = entry
                return cached
        return None

    def _diff_cache_put(self, cache_key: tuple, lines: tui_core.Chain,
                        retry_at: float | None) -> tui_core.Chain:
        if len(self._diff_cache) >= _DIFF_CACHE_ENTRIES:
            self._diff_cache.pop(next(iter(self._diff_cache)))
        self._diff_cache[cache_key] = (lines, retry_at)
        return lines

    def _diff_body(self, item: Item, snapshot: dict | None,
                   mode: str) -> tui_core.Chain:
        """The "patches" / "merge diff" of ``item`` at the base/head
        SHAs its items/ snapshot reports, as a lazily rendering line
        sequence, cached until the mode or those SHAs change (a cache
        miss re-runs git; hunks highlight when first shown -- the
        message and logs panes each keep an entry, so the cache holds
        a few). A git failure is cached only briefly: it usually means
        the head is not fetched yet, and the view must recover once
        the operator fetches the mirror."""
        base_sha = (snapshot or {}).get("base_sha")
        head_sha = (snapshot or {}).get("head_sha")
        cache_key = (item.repo, item.number, mode, base_sha, head_sha)
        cached = self._diff_cache_get(cache_key)
        if cached is not None:
            return cached
        repo = self.patch_repos.get(item.repo)
        retry_at = None
        tail: list[tui_core.StyledLine] = []
        caption = [("label", mode + (f" {base_sha[:12]}..{head_sha[:12]}"
                                     if base_sha and head_sha else ""))]
        if repo is None:
            body = [[("log_warn", "no patch-repo configured for this repo")]]
        elif not base_sha or not head_sha:
            body = [[("log_warn", "no item snapshot with base/head SHAs "
                                  "yet; the next agent scan writes it")]]
        else:
            try:
                raw = (git_util.git_patch_stream(repo, base_sha, head_sha)
                       if mode == "patches"
                       else git_util.git_diff(repo, base_sha, head_sha))
                patch_lines = raw.decode("utf-8", errors="replace").split("\n")
                logger.info("%s %s#%s: %d lines from %s %s..%s",
                            mode, item.repo, item.number,
                            len(patch_lines), repo, base_sha[:12],
                            head_sha[:12])
                body = diff_render.DiffView(
                    "\n".join(patch_lines[:DIFF_MAX_LINES])) \
                    or [[("label", "(empty diff)")]]
                if len(patch_lines) > DIFF_MAX_LINES:
                    tail = [[("log_warn", f"… truncated at {DIFF_MAX_LINES}"
                                          f" of {len(patch_lines)} lines")]]
            except RuntimeError as exc:
                logger.error("%s", exc)
                retry_at = time.monotonic() + 5
                body = [[("log_err", line)] for line in str(exc).splitlines()]
                body.append([("label", "if the head is not fetched yet: "
                                       f"git -C {repo} fetch --all")])
        return self._diff_cache_put(
            cache_key, tui_core.Chain([caption, []], body, tail), retry_at)

    def _logs_diff_lines(self) -> list[tui_core.StyledLine] | tui_core.Chain:
        """The logs pane's content for ``d``'s non-logs modes: the
        cursor PR's diff. Caller holds ``model.lock``."""
        m = self.model
        key = m._cursor_key() or m.cursor_key
        item = m.items.get(key) if key else None
        if item is None or item.kind != "pr":
            return [[("log_warn", "no PR selected")]]
        return self._diff_body(item, m.snapshot_for(key), self.logs_mode)

    def _warm_diff_cache(self) -> None:
        """Fetch and window-render every diff a pane is about to show,
        before paint takes ``model.lock`` for good: git and pygments
        are too slow to run under it (the poll thread would stall
        behind them), so paint's locked pass must find its slices
        already rendered."""
        panes = [(pane, mode) for pane, mode in
                 (("br", self.detail_mode), ("bl", self.logs_mode))
                 if mode in DIFF_MODES]
        warm_branches = self.detail_mode == "message"
        if not panes and not warm_branches:
            return
        with self.model.lock:
            key = self.model._cursor_key() or self.model.cursor_key
            item = self.model.items.get(key) if key else None
            snapshot = self.model.snapshot_for(key)
        if item is None:
            return
        if warm_branches:
            # issue verdicts carry branches too, so no kind guard here
            h = self.term.height
            for record in _ticket_branches(item.data):
                if record.get("mode") == "delete":
                    continue
                lines = self._branch_diff_body(item, record)
                # like the panes loop below: the discarded slice is what
                # renders the hunks before paint takes model.lock
                s = min(self.scroll["br"], len(lines))
                lines[max(0, s - h):s + 2 * h]
        if item.kind != "pr":
            return
        for pane, mode in panes:
            lines = self._diff_body(item, snapshot, mode)
            # the discarded slice is what renders: paint's own slice of
            # these lines then hits rendered hunks; one screen of margin
            # covers the detail head sitting above the diff. The scroll
            # is clamped only inside paint (an End jump parks it at
            # 10**9), so clamp here too or the slice misses the tail.
            h = self.term.height
            s = min(self.scroll[pane], len(lines))
            lines[max(0, s - h):s + 2 * h]

    def paint(self) -> None:
        self._last_paint = time.monotonic()
        if self.help_text is not None:
            self._paint_help()
            return
        t = self.term
        w, h = t.width, t.height
        body_h = max(3, h - 1)
        rects = self.layout.rects(w, body_h)
        col_t, col_b, row = self.layout.splits(w, body_h)
        if self.logs_mode not in DIFF_MODES:
            self.scroll["bl"] = min(self.scroll["bl"],
                                    max(0, len(self.ring) - (rects["bl"].h - 1)))
        self._warm_diff_cache()
        # only the _scrolled panes below refill this
        self._painted.clear()
        with self.model.lock:
            content: dict[str, list] = {
                "tl": self._scrolled("tl", self.stats_lines(self._text_width(rects["tl"])),
                                     rects["tl"].h - 1),
                "tr": self._list_window(rects["tr"].h - 1),
                "bl": [(self._log_style(tag), text) for tag, text in
                       self.ring.view(self.scroll["bl"], rects["bl"].h - 1)]
                if self.logs_mode not in DIFF_MODES
                else self._scrolled("bl", self._logs_diff_lines(),
                                    rects["bl"].h - 1),
                "br": self._scrolled("br", self.detail_lines(self._text_width(rects["br"])),
                                     rects["br"].h - 1),
            }
            nact = self.model.reviewed_count()
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
        if self._status_mode != (self.model.sort_mode, self.model.filter_mode,
                                 bool(self.paused)):
            self._build_status()
        note = f" ▶ {nact} reviewed  " if nact else " "
        if self.search_mode:
            note = f" /{self.search_buf}▏" + note
        elif self.count_buf:
            note = f" {self.count_buf}x " + note
        if len(note) + len(self._status_plain) > w:
            status = (note + self._status_plain)[:w].ljust(w)
        else:
            mark = self.styles.get("st_reviewed") or (lambda s: s)
            status = ((mark(note) if nact else note) + self._status_styled
                      + " " * (w - len(note) - len(self._status_plain)))
        buf.append(t.move_xy(0, h - 1) + status)
        print("".join(buf), end="", flush=True, file=t.stream)

    def _paint_help(self) -> None:
        """Full-screen render of README-TUI.md; any key returns."""
        t = self.term
        w, h = t.width, t.height
        lines = tui_core.render_markdown(self.help_text, max(MIN_TEXT_W, w - 2))
        inner_h = max(1, h - 2)
        self.help_scroll = max(0, min(self.help_scroll, len(lines) - inner_h))
        bar = self.styles.get("bar_focus") or t.reverse
        buf = [t.move_xy(0, 0) + bar(" ? help — README-TUI.md"[:w].ljust(w))]
        for i in range(inner_h):
            row = self.help_scroll + i
            line = lines[row] if row < len(lines) else []
            buf.append(t.move_xy(0, 1 + i) + " "
                       + self._styled_line(line, w - 1))
        label = self.styles.get("label") or (lambda s: s)
        buf.append(t.move_xy(0, h - 1)
                   + label(" ↑↓ PgUp/PgDn Home/End scroll   any other key "
                           "closes"[:w].ljust(w)))
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
        self._painted[pane] = lines
        return lines[self.scroll[pane]:self.scroll[pane] + inner_h]

    def _list_window(self, inner_h: int) -> list:
        """The on-screen slice of the list, formatted; only these rows
        pay the formatting cost. The view chases the cursor only right
        after a cursor move (the consumed ``follow_cursor``): a
        wheel-scrolled window must stay where the operator put it
        across timer repaints. Caller holds ``model.lock``."""
        m = self.model
        vis = m._sync_cursor()
        cursor = m.cursor
        if m.cursor_shown and self.follow_cursor:
            if cursor < self.list_top:
                self.list_top = cursor
            if inner_h > 0 and cursor >= self.list_top + inner_h:
                self.list_top = cursor - inner_h + 1
        self.follow_cursor = False
        self.list_top = max(0, min(self.list_top, max(0, len(vis) - inner_h)))
        return [self._row(it, m.cursor_shown and self.list_top + i == cursor)
                for i, it in enumerate(
                    vis[self.list_top:self.list_top + inner_h])]

    def _blit(self, buf: list, rect: tui_core.Rect, pane: str, lines: list) -> None:
        t = self.term
        if rect.w <= 0 or rect.h <= 0:
            return
        title = f" {PANE_GLYPHS[pane]} {PANES[pane]} "
        if pane == "tr":
            title += f"[{self.model.filter_mode}] "
        elif pane == "br":
            title += "⧉ " if self.detail_mode not in DIFF_MODES \
                else f"⧉ [{self.detail_mode}] "
        elif pane == "bl" and self.logs_mode in DIFF_MODES:
            title += f"[{self.logs_mode}] "
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
            if line and not isinstance(line, str) \
                    and not isinstance(line[0], str) \
                    and line[0][0] == "diff_commit":
                line = [("bold", "⧉ ")] + line
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

    def _poll_loop(self) -> None:
        """Polling thread: the directory scans, ticket parses and log
        tailing run off the UI thread, so a heavy poll (an agent scan
        pass churning a 100k-file dir) delays repaints by lock time
        only, never by IO time. Wakes on the filesystem watcher within
        one tick, or on the 1s fallback interval; results land in the
        model under its lock and reach the painter through ``dirty``."""
        while not self.model.quit_flag:
            self.needs_poll.wait(1.0)
            self.needs_poll.clear()
            try:
                self.model.poll()
                self.tail.poll()
            except Exception:
                logger.exception("poll failed; the next tick retries")
            self.model.dirty.set()

    def _read_key(self) -> blessed.keyboard.Keystroke | None:
        """One keystroke, or None for a control sequence blessed's
        keymap does not cover: of those it returns the bare ``ESC [`` /
        ``ESC O`` introducer and hands out the remaining bytes as
        ordinary keys, so rxvt-unicode's shift-Up -- ``ESC [ a``, which
        no terminfo capability describes -- would arrive as the lens
        key ``a``. Its tail, up to a final byte in 0x40..0x7e, is
        buffered by then: blessed waited out its escape delay."""
        ks = self.term.inkey(timeout=_TTY_TIMEOUT)
        if str(ks) not in ("\x1b[", "\x1bO"):
            return ks
        sequence = str(ks)
        while tail := self.term.inkey(timeout=0):
            if len(tail) > 1:
                # a sequence blessed did resolve: the next key, not a
                # tail byte of this one (Alt+[ ahead of an arrow)
                self.term.ungetch(str(tail))
                break
            sequence += str(tail)
            if "\x40" <= str(tail) <= "\x7e":
                break
        logger.debug("dropped %r: no keymap entry for it", sequence)
        return None

    def _read_keys(self) -> None:
        """Input thread: ``inkey()`` sits in select() and returns the
        instant bytes arrive; each key lands in the queue and wakes the
        main loop through ``dirty``. A daemon, so quit need not wait
        out its read."""
        while not self.model.quit_flag:
            self._stdin_free.wait()
            with self._stdin_reader:
                ks = self._read_key()
            if ks:
                self._keys.put(ks)
                self.model.dirty.set()

    @contextmanager
    def _tty_lent(self):
        """Hand the terminal to a child process: once the input
        thread's pending ``inkey()`` returns it is held out of stdin,
        keys read meanwhile are dropped rather than run as commands
        later, mouse reporting is off and the normal screen is shown.
        All of it is undone afterwards."""
        t = self.term
        self._stdin_free.clear()
        try:
            with self._stdin_reader, \
                    t.dec_modes_disabled(*_MOUSE_MODES, timeout=_TTY_TIMEOUT):
                while not self._keys.empty():
                    self._keys.get_nowait()
                while t.inkey(timeout=0):
                    pass
                print(t.exit_fullscreen + t.normal_cursor, end="",
                      flush=True, file=t.stream)
                logger.debug("terminal handed over")
                try:
                    yield
                finally:
                    print(t.enter_fullscreen + t.hide_cursor, end="",
                          flush=True, file=t.stream)
                    logger.debug("terminal taken back")
        finally:
            self._stdin_free.set()
            self.model.dirty.set()

    def run(self) -> None:
        Thread(target=self._read_keys, name="input", daemon=True).start()
        Thread(target=self._poll_loop, name="poll", daemon=True).start()
        try:
            while not self.model.quit_flag:
                # keys, watcher events and repaints wake this instantly;
                # the 1s tick only keeps the clock and log tail moving
                self.model.dirty.wait(1.0)
                self.model.dirty.clear()
                while True:
                    try:
                        self.dispatch(self._keys.get_nowait())
                    except Empty:
                        break
                size = (self.term.width, self.term.height)
                if size != self._last_size:
                    self._last_size = size
                self.model.poll_snapshot()
                self.paint()
        except KeyboardInterrupt:
            pass
        if self.paused:
            self.toggle_pause()
        self.model.quit_all()

    def toggle_pause(self) -> None:
        """p: freeze this session's agents and workers (SIGSTOP on their
        whole subprocess trees); the next press thaws them (SIGCONT).
        Stopping sweeps until no new pid appears, or a process forking
        between the scan and its stop would leak a running child.
        run() thaws before exiting: a stopped process never sees the
        launcher's exit-trap SIGTERM and would stay frozen forever."""
        if self.paused:
            for pid in self.paused:
                try:
                    os.kill(pid, signal.SIGCONT)
                except OSError:
                    pass
            logger.info("resumed %d paused processes", len(self.paused))
            self.paused = []
            return
        stopped: list[int] = []
        while True:
            new = [p for p in session_pids() if p not in stopped]
            if not new:
                break
            for pid in new:
                try:
                    os.kill(pid, signal.SIGSTOP)
                    stopped.append(pid)
                except OSError:
                    pass
        if stopped:
            logger.info("paused %d processes; p resumes them", len(stopped))
        else:
            logger.info("no agent/worker processes found to pause")
        self.paused = stopped

    def _scroll_pane(self, pane: str, delta: int) -> None:
        if pane == "tr":
            with self.model.lock:
                self.model._sync_cursor()
                # a hidden cursor is summoned back to its last screen
                # spot by the first press; only then do arrows move it
                self.model.select_index(
                    self.model.cursor + (delta if self.model.cursor_shown
                                         else 0))
            self.follow_cursor = True
        elif pane == "bl" and self.logs_mode not in DIFF_MODES:
            # offset counts back from the newest line; 0 follows the tail
            self.scroll["bl"] = max(0, self.scroll["bl"] - delta)
        elif pane in self.scroll:
            self.scroll[pane] = max(0, self.scroll[pane] + delta)

    def _jump_pane(self, pane: str, top: bool) -> None:
        """Home/End: the list jumps the cursor to the first/last row;
        the other panes jump their scroll to the oldest/newest end
        (the log pane's offset counts back from the tail, hence its
        inverted arithmetic; the huge value is clamped at paint)."""
        if pane == "tr":
            with self.model.lock:
                vis = self.model._sync_cursor()
                self.model.select_index(0 if top else len(vis) - 1)
            self.follow_cursor = True
        elif pane == "bl" and self.logs_mode not in DIFF_MODES:
            self.scroll["bl"] = 10 ** 9 if top else 0
        elif pane in self.scroll:
            self.scroll[pane] = 0 if top else 10 ** 9

    def _jump_section(self, levels: tuple[str, ...], forward: bool,
                      count: int) -> None:
        """[ ] and { }: park a diff pane ``count`` patches or hunks
        before / after its top line, stopping at the outermost one.
        ``levels`` lists the header styles of one step, most specific
        first -- a series steps by commit, a merge diff, which has
        none, by file. Acts on the focused pane, or on whichever pane
        shows a diff while the focus is elsewhere: reading a patch with
        the cursor still on the list is the normal case."""
        # the offsets below describe the painted frame; repaint, or a
        # mode or cursor key dispatched since the last one would have
        # this step a pane against the diff it no longer shows
        self.paint()
        for pane in (self.focus, "br", "bl"):
            lines = self._painted.get(pane)
            starts = next((s for s in map(lines.sections, levels) if s), []) \
                if hasattr(lines, "sections") else []
            if not starts:
                continue
            top = self.scroll[pane]
            reachable = [i for i in starts if i > top] if forward \
                else [i for i in reversed(starts) if i < top]
            if step := reachable[:count]:
                self.scroll[pane] = step[-1]
            return

    def _page(self) -> int:
        rects = self.layout.rects(self.term.width, max(3, self.term.height - 1))
        return max(1, rects[self.focus].h - 2)

    def _take_count(self) -> int:
        n = int(self.count_buf or "1")
        self.count_buf = ""
        return n

    def _search_jump(self) -> None:
        query = (self.search_buf or self.last_search).casefold()
        if not query:
            return
        self.last_search = self.search_buf or self.last_search
        with self.model.lock:
            vis = self.model._sync_cursor()
            start = min(self.model.cursor + 1, len(vis))
            for i in list(range(start, len(vis))) + list(range(0, start)):
                it = vis[i]
                if query in str(it.number).casefold() \
                        or query in (it.data.get("title") or "").casefold() \
                        or query in it.state:
                    self.model.select_index(i)
                    self.follow_cursor = True
                    return
        logger.info("no ticket matches %r", query)

    def dispatch(self, ks) -> None:
        name = ks.name or ""
        if self.help_text is not None:
            page = max(1, self.term.height - 3)
            delta = {"KEY_UP": -1, "KEY_DOWN": 1, "KEY_PGUP": -page,
                     "KEY_PGDOWN": page, "MOUSE_SCROLL_UP": -3,
                     "MOUSE_SCROLL_DOWN": 3}.get(name)
            if delta is not None:
                self.help_scroll = max(0, self.help_scroll + delta)
            elif name == "KEY_HOME":
                self.help_scroll = 0
            elif name == "KEY_END":
                self.help_scroll = 10 ** 9
            elif not name.startswith("MOUSE_"):
                self.help_text = None
            self.model.dirty.set()
            return
        if self.search_mode:
            if name == "KEY_ENTER" or str(ks) in ("\n", "\r"):
                self.search_mode = False
                self._search_jump()
            elif name == "KEY_ESCAPE":
                self.search_mode = False
                self.search_buf = ""
            elif name in ("KEY_BACKSPACE", "KEY_DELETE") \
                    or str(ks) in ("\x7f", "\x08"):
                self.search_buf = self.search_buf[:-1]
            elif not name and str(ks).isprintable():
                self.search_buf += str(ks)
            self.model.dirty.set()
            return
        if not name and str(ks).isdigit():
            # two digits suffice: anything >=10 is refused anyway
            self.count_buf = (self.count_buf + str(ks))[-2:]
            self.model.dirty.set()
            return
        if name in ("KEY_BACKSPACE", "KEY_DELETE") and self.count_buf:
            self.count_buf = self.count_buf[:-1]
            self.model.dirty.set()
            return
        body_h = max(3, self.term.height - 1)
        if name.startswith("MOUSE_"):
            y, x = ks.mouse_yx
            hit = (self.layout.hit(x, y, self.term.width, body_h)
                   if 0 <= y < body_h else "")
            if name in ("MOUSE_SCROLL_UP", "MOUSE_SCROLL_DOWN"):
                delta = -3 if name.endswith("UP") else 3
                if hit == "tr":
                    # the wheel moves the VIEW; only cursor keys and
                    # clicks move the cursor
                    self.list_top = max(0, self.list_top + delta)
                elif hit in PANES:
                    self._scroll_pane(hit, delta)
            elif name == "MOUSE_LEFT":
                if hit in ("vt", "vb", "h"):
                    self.drag = hit
                elif hit in PANES:
                    self.focus = hit
                    if hit == "tr":
                        row = y - 1  # list rows start under the title bar
                        with self.model.lock:
                            self.model.select_index(self.list_top + row)
                        self.follow_cursor = True
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
            self._scroll_pane(self.focus, -self._take_count())
        elif name == "KEY_DOWN":
            self._scroll_pane(self.focus, self._take_count())
        elif name == "KEY_PGUP":
            self._scroll_pane(self.focus, -self._page())
        elif name == "KEY_PGDOWN":
            self._scroll_pane(self.focus, self._page())
        elif name in ("KEY_HOME", "KEY_END"):
            self._jump_pane(self.focus, top=name == "KEY_HOME")
        elif name in ("KEY_LEFT", "KEY_RIGHT") and self.focus in ("bl", "br"):
            # flip through tickets while reading: each lands at its top
            self.scroll[self.focus] = 0
            self._scroll_pane("tr", (-1 if name == "KEY_LEFT" else 1)
                              * self._take_count())
        elif ks == "q":
            self.model.quit_all()
        elif ks == "a":
            with self.model.lock:
                mode = self.model.cycle_filter()
            self.follow_cursor = True
            logger.info("list filter: %s", mode)
        elif ks == "t":
            with self.model.lock:
                mode = self.model.cycle_sort()
            self.follow_cursor = True
            logger.info("list sort: %s", mode)
        elif str(ks) == "r":
            self.model.act("rerun", self._take_count())
        elif str(ks) in ACTION_KEYS:
            self.count_buf = ""
            self.model.act(ACTION_KEYS[str(ks)])
            self.follow_cursor = True
        elif ks == "p":
            self.toggle_pause()
        elif ks == "/":
            self.search_mode = True
            self.search_buf = ""
        elif ks == "?":
            try:
                self.help_text = Path(__file__).with_name(
                    "README-TUI.md").read_text(encoding="utf-8")
                self.help_scroll = 0
            except OSError as exc:
                logger.error("cannot read README-TUI.md: %s", exc)
        elif ks == "n":
            self._search_jump()
        elif ks == "o":
            self.edit_review()
        elif ks == "m":
            self.detail_mode = _next_mode(DETAIL_MODES, self.detail_mode)
            self.scroll["br"] = 0
            logger.info("message pane: %s", self.detail_mode)
        elif ks == "d":
            self.logs_mode = _next_mode(LOGS_MODES, self.logs_mode)
            self.scroll["bl"] = 0
            logger.info("logs pane: %s", self.logs_mode)
        elif ks == "b":
            self.branch_diff_mode = _next_mode(BRANCH_DIFF_MODES,
                                               self.branch_diff_mode)
            logger.info("branch previews: %s", self.branch_diff_mode)
        elif ks in ("[", "]", "{", "}"):
            self._jump_section(("diff_hunk",) if ks in ("{", "}")
                               else ("diff_commit", "diff_file"),
                               ks in ("]", "}"), self._take_count())
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
        edited = self._edit_text(message, f"fairy-{item.kind}-{item.number}-")
        if edited is None or edited == message:
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

    def _edit_text(self, text: str, prefix: str) -> str | None:
        """``text`` as $EDITOR left it in a temp ``.md`` file; None when
        the editor could not start or exited non-zero."""
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
        fd, tmp_name = tempfile.mkstemp(suffix=".md", prefix=prefix)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            cmd = [*shlex.split(editor), str(tmp)]
            logger.info("editing via: %s", shlex.join(cmd))
            try:
                with self._tty_lent():
                    rc = subprocess.call(
                        cmd, stdin=sys.__stdin__, stdout=sys.__stdout__,
                        stderr=sys.__stderr__)
            except OSError as e:
                logger.warning("editor failed to start: %s", e)
                return None
            if rc != 0:
                logger.warning("editor exited rc=%d; its text is dropped", rc)
                return None
            return tmp.read_text(encoding="utf-8")
        finally:
            tmp.unlink(missing_ok=True)

    def _copy_click(self, pane: str, x: int, y: int) -> None:
        """Copy the URL / git hash / #number under a left click to the
        system clipboard via OSC 52 (needs terminal support; tmux wants
        set-clipboard on). The rows are stored unclipped, so a visually
        truncated URL still copies whole. Off a token, a click on a
        commit header row (its ⧉ advertises it) copies the whole
        patch."""
        rect, gutter, rows = self._shown.get(pane, (None, 0, []))
        if rect is None:
            return
        row, col = y - rect.y - 1, x - rect.x - gutter
        if row == -1 and pane == "br":
            self._copy_message()
            return
        if not (0 <= row < len(rows)) or col < 0:
            return
        token = tui_core.token_at(rows[row], col)
        if token is not None:
            self._to_clipboard(token, repr(token))
            return
        self._copy_patch_at(pane, row)

    def _copy_patch_at(self, pane: str, row: int) -> None:
        """The patch whose commit header sits on the pane's visible
        ``row``, taken whole from the pane's unclipped content: what
        the patches views show per commit is a git-am-able format-patch
        (the git log text for a merge), so the copy is too."""
        content = self._painted.get(pane)
        if not hasattr(content, "sections"):
            return
        clicked = self.scroll[pane] + row
        starts = content.sections("diff_commit")
        if clicked not in starts:
            return
        end = next((s for s in starts if s > clicked), len(content))
        text = _plain(content[clicked:end]).rstrip("\n") + "\n"
        sha = text.split()[1]
        self._to_clipboard(text, f"patch {sha[:12]} ({len(text)} chars)")

    def _copy_message(self) -> None:
        """A click on the message pane's title bar (the ⧉ glyph
        advertises it) copies the cursor item's raw review message --
        the markdown source, not the rendered pane -- ready to paste
        into a mail or forge comment; without a review the plain pane
        text (error and reason lines) is copied instead."""
        with self.model.lock:
            key = self.model._cursor_key()
            item = self.model.items.get(key) if key else None
        if item is None:
            return
        text = ((item.data.get("review") or {}).get("message")
                or "\n".join(self._shown["br"][2]).rstrip())
        self._to_clipboard(
            text, f"{item.kind} #{item.number} message ({len(text)} chars)")

    def _to_clipboard(self, text: str, desc: str) -> None:
        if self._clip_cmd is not None:
            try:
                subprocess.run(
                    self._clip_cmd, input=text.encode(), timeout=2, check=True,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.info("copied %s via %s (primary selection)",
                            desc, self._clip_cmd[0])
                return
            except Exception as exc:
                logger.debug("clipboard helper %s failed (%s); trying OSC 52",
                             self._clip_cmd, exc)
        # Display-less fallback stays on the c (clipboard) target: no
        # PRIMARY reaches the local end of a plain ssh session and several
        # terminals ignore 52;p.
        b64 = base64.b64encode(text.encode()).decode()
        print(f"\x1b]52;c;{b64}\x07", end="", flush=True, file=self.term.stream)
        logger.info("copied %s to the clipboard (OSC 52)", desc)

    def export(self, full: bool) -> None:
        pane = self.focus
        inner_h = self._page() + 1
        # The visible export must reproduce the pane as painted: same
        # divider-dependent width, or the tiling/wrapping (and thus the
        # scroll window) would differ from the screen.
        rects = self.layout.rects(self.term.width, max(3, self.term.height - 1))
        with self.model.lock:
            if pane == "bl" and self.logs_mode in DIFF_MODES:
                lines = self._logs_diff_lines()
                text = _plain(lines if full
                              else self._scrolled("bl", lines, inner_h))
            elif pane == "bl":
                text = (self.ring.all_text() if full
                        else "\n".join(t for _, t in
                                       self.ring.view(self.scroll["bl"], inner_h)))
            elif pane == "tl":
                lines = self.stats_lines(
                    EXPORT_FULL_W if full else self._text_width(rects["tl"]))
                text = _plain(lines if full else self._scrolled("tl", lines, inner_h))
            elif pane == "tr":
                text = _plain(self.list_rows() if full
                              else self._list_window(inner_h))
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
    p.add_argument("--db-root", metavar="DIR", action="append", type=Path,
                   help="a repo's filedb root, as its agent logs at startup; "
                        "one per use")
    p.add_argument("--tail", metavar="FILE", action="append", type=Path,
                   help="follow this agent/worker --log-file in the logs pane "
                        "(repeatable)")
    p.add_argument("--log-file", type=Path,
                   help="tee every line shown in the logs pane to this file")
    p.add_argument("--save-dir", type=Path, default=Path("."),
                   help="directory for e/E pane exports (default: cwd)")
    args = p.parse_args(argv)
    if not args.db_root:
        p.error("at least one --db-root is required")
    return args


def build_sides(
    args: argparse.Namespace,
) -> tuple[list[tuple[str, filedb.Db]], list[Path], dict[str, Path]]:
    """One (repo label, filedb) side per distinct --db-root; each root's
    config.toml, written by configurator.py, names the repo, the log
    files the logs pane tails without separate --tail flags, and the
    [pr] patch-repo mirror the ``m`` diff views read (label-keyed in
    the returned dict; the path is relative to the agent's cwd, so the
    TUI must run from the same directory)."""
    sides: list[tuple[str, filedb.Db]] = []
    tails: list[Path] = []
    patch_repos: dict[str, Path] = {}
    seen: set[Path] = set()

    def add_tail(path: Path) -> None:
        if path.resolve() not in seen:
            seen.add(path.resolve())
            tails.append(path)

    for root in args.db_root:
        if any(db.root.resolve() == root.resolve() for _, db in sides):
            continue
        cfg = db_config.read_config(root)
        for f in map(Path, cfg["log_files"]):
            add_tail(f)
        sides.append((cfg["label"], filedb.Db(root)))
        if (cfg.get("pr") or {}).get("patch-repo"):
            patch_repos[cfg["label"]] = Path(cfg["pr"]["patch-repo"])
    for t in args.tail or []:
        add_tail(t)
    return sides, tails, patch_repos


def main() -> int:
    args = parse_args()
    ring = tui_core.RingBuffer()
    sides, tails, patch_repos = build_sides(args)
    model = Model(sides)
    sink = OutputSink(ring, model.dirty, args.log_file)
    setup_logging(logger, False, db_config.logger,
                  handlers=[RingLogHandler(sink)])
    for repo, db in sides:
        logger.info("side %s: db %s", repo, db.root)
    for path in tails:
        logger.info("tailing %s", path)

    term = blessed.Terminal(stream=sys.__stdout__)
    faulthandler.enable(file=sys.__stderr__)
    ui = UILoop(term, model, ring, args.save_dir, LogTail(tails, sink),
                patch_repos)
    # File changes repaint within one 100ms input tick instead of the
    # 1s fallback rescan; without watchdog only the fallback remains.
    watch_paths(
        [db.root for _, db in sides] + sorted({t.parent for t in tails}),
        ui.needs_poll.set, recursive=True)
    with term.fullscreen(), term.cbreak(), term.hidden_cursor(), \
            term.dec_modes_enabled(*_MOUSE_MODES, timeout=_TTY_TIMEOUT), \
            captured_output(sink):
        model.poll()
        ui.run()
    sink.close()
    if args.log_file:
        print(f"captured output: {args.log_file}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
