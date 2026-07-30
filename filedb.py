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

The file database: one JSON ticket per PR/issue, state = directory name.

A ticket's state is encoded ONLY by the directory it sits in (English
words, ``mv`` is a state change); the JSON content never carries the
state. Every operation is atomic on a local filesystem: writes go
through tmp-file + rename, transitions and claims serialize on a
stable per-item sidecar lock in ``locks/`` (content writes replace the
inode, so the payload file itself can never carry a lock). Readers
need no locks. A crash between the two steps of a transition leaves
the item in two directories; the later pipeline state wins
(``find()`` precedence). Remnants are deliberately NOT deleted
eagerly -- they are crash evidence: ``reap()`` recovers llm/ claims
whose worker died, the agent clears reviewed/ remnants that shadow a
later state, and everything else is resolved by precedence until
retention pruning.

What belongs here: the per-repo directory layout, atomic push/get/pop/
move, the worker claim protocol (lock -> rename -> work -> finish),
reaping dead workers' claims, and retention pruning.

What does NOT belong: ticket schemas and validation (workset), gates,
forge access, review logic, and any policy about which state an item
should be in next (fairy).
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["Db", "Claim", "STATES", "KINDS", "forge_number",
           "is_base", "logger"]

logger = logging.getLogger(__name__)

# Pipeline order doubles as crash-remnant precedence: an item present in
# two directories is a remnant in the earlier one.
STATES = ("requests", "queued", "llm", "reviewed", "outgoing",
          "ci-blocked", "merge-ready", "awaiting-approver",
          "posted", "skipped", "cancelled", "error")
KINDS = ("pr", "issue")
_TMP = "tmp"
_LOCKS = "locks"
# Ticket identity: the forge number, optionally refined by a sample
# and/or review dimension -- "12345", "12345s2", "12345s1r2". Suffixed
# tickets are operator/tooling-created evaluations of the same forge
# item; only the base ticket takes part in scanning, gating and
# posting by default.
_TOKEN_RE = re.compile(r"(\d+)(?:s\d+)?(?:r\d+)?")


def _token(number) -> str:
    token = str(number)
    if not _TOKEN_RE.fullmatch(token):
        raise ValueError(f"invalid ticket token {number!r}")
    return token


def forge_number(number) -> int:
    """The forge item number behind any ticket token."""
    return int(_TOKEN_RE.fullmatch(_token(number)).group(1))


def is_base(number) -> bool:
    """True for the plain per-item ticket (no sample/review suffix)."""
    return str(number).isdigit()


def _name(kind: str, number) -> str:
    if kind not in KINDS:
        raise ValueError(f"unknown kind {kind!r}")
    return f"{kind}-{_token(number)}.json"


class Claim:
    """A worker's exclusive hold on one ticket (see ``Db.claim``).

    The sidecar flock is held from before the claim rename until
    ``finish``/``abort``/process death; a dead worker's claim is
    recognizable by its acquirable lock and reaped back to the source
    state."""

    def __init__(self, db: "Db", fd: int, kind: str, number: int,
                 src_state: str, path: Path) -> None:
        self._db = db
        self._fd = fd
        self.kind = kind
        self.number = number
        self.src_state = src_state
        self.path = path  # the claimed file, owned by this claim holder

    def read(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def write(self, data: dict) -> None:
        """Update the claimed file in place (atomic replace, same dir)."""
        self._db._write(self.path, data)

    def finish(self, dst_state: str, data: dict) -> Path:
        """Write ``data`` into ``dst_state`` and drop the claimed file."""
        dst = self._db._write_state(dst_state, self.kind, self.number, data)
        if dst != self.path:  # a same-state finish is an in-place update
            self.path.unlink(missing_ok=True)
        self._release()
        return dst

    def abort(self) -> None:
        """Return the ticket to its source state (clean shutdown); an
        in-place claim just releases -- no rename, no watcher event."""
        target = self._db.path(self.src_state, self.kind, self.number)
        if target != self.path:
            os.replace(self.path, target)
        self._release()

    def _release(self) -> None:
        if self._fd >= 0:
            os.close(self._fd)
            self._fd = -1


class Db:
    """File database rooted at one repository directory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        for d in (*STATES, _TMP, _LOCKS):
            (self.root / d).mkdir(parents=True, exist_ok=True)

    def path(self, state: str, kind: str, number: int) -> Path:
        if state not in STATES:
            raise ValueError(f"unknown state {state!r}")
        return self.root / state / _name(kind, number)

    def _load(self, path: Path) -> dict | None:
        """The decoded ticket at ``path``; ``path`` need not exist.

        None when the file is absent, unreadable or not valid JSON:
        none of those raise, and everything but absence is logged.
        Callers take None as no-prior-data."""
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.error("unreadable %s: %s", path, exc)
            return None

    def _write(self, dst: Path, data: dict) -> Path:
        fd, tmp = tempfile.mkstemp(dir=self.root / _TMP, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=1)
                f.write("\n")
                f.flush()
                # fsync before the rename: without it a power loss can
                # leave a truncated ticket under the final name
                os.fsync(f.fileno())
            os.replace(tmp, dst)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return dst

    def _write_state(self, state: str, kind: str, number: int, data: dict) -> Path:
        dst = self.path(state, kind, number)
        stripped = {k: v for k, v in data.items() if k != "state_changed_at"}
        current = self._load(dst)
        if current is not None and stripped == {
                k: v for k, v in current.items() if k != "state_changed_at"}:
            # identical content: no write -- no fsync churn, no dir-mtime
            # bump (viewers key cheap polls on it), and state_changed_at
            # keeps meaning "when did this actually change"
            return dst
        data["state_changed_at"] = datetime.now(timezone.utc).isoformat()
        return self._write(dst, data)

    @contextmanager
    def lock(self, kind: str, number: int):
        """Exclusive per-item transition lock (blocking)."""
        fd = self._lock_fd(kind, number, block=True)
        try:
            yield
        finally:
            os.close(fd)

    def _lock_path(self, kind: str, number, suffix: str = "lock") -> Path:
        return self.root / _LOCKS / f"{kind}-{_token(number)}.{suffix}"

    def _lock_fd(self, kind: str, number: int, *, block: bool,
                 suffix: str = "lock") -> int:
        path = self._lock_path(kind, number, suffix)
        while True:
            fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | (0 if block else fcntl.LOCK_NB))
            except OSError:
                os.close(fd)
                raise
            # prune unlinks dead items' lock files: a flock acquired on
            # an inode that no longer is the file at ``path`` excludes
            # nobody -- retry on the current file.
            try:
                if os.fstat(fd).st_ino == os.stat(path).st_ino:
                    return fd
            except FileNotFoundError:
                pass
            os.close(fd)

    # ---- basic operations (all atomic; readers lock-free) ----

    def push(self, state: str, kind: str, number: int, data: dict) -> Path:
        with self.lock(kind, number):
            return self._write_state(state, kind, number, data)

    def get(self, state: str, kind: str, number: int) -> dict | None:
        return self._load(self.path(state, kind, number))

    def pop(self, state: str, kind: str, number: int) -> dict | None:
        """Read and delete; None when absent."""
        with self.lock(kind, number):
            path = self.path(state, kind, number)
            data = self._load(path)
            if data is None:
                return None
            path.unlink()
            return data

    def move(self, src_state: str, dst_state: str, kind: str, number: int,
             mutate=None) -> bool:
        """Transition src -> dst, optionally mutating the content; False
        when the item is not in ``src_state`` (lost a race: fine)."""
        with self.lock(kind, number):
            src = self.path(src_state, kind, number)
            data = self._load(src)
            if data is None:
                return False
            if mutate is not None:
                mutate(data)
            self._write_state(dst_state, kind, number, data)
            src.unlink(missing_ok=True)
            return True

    def replace(self, state: str, kind: str, number: int, data: dict,
                *, expect: str | None) -> bool:
        """Put the item into ``state`` wherever it currently is: dst is
        written FIRST, then the old file dropped, so a crash leaves a
        remnant for the precedence rule, never a lost ticket. Non-
        blocking, and refused (False) when the item is claimed or no
        longer in ``expect`` -- any move under the caller's feet (a
        worker claim, an operator y/s/x) invalidates the decision."""
        if self._leased(kind, number):
            return False
        try:
            fd = self._lock_fd(kind, number, block=False)
        except OSError:
            return False
        try:
            prior = self.find(kind, number)
            if prior != expect:
                return False
            self._write_state(state, kind, number, data)
            if prior is not None and prior != state:
                self.path(prior, kind, number).unlink(missing_ok=True)
            return True
        finally:
            os.close(fd)

    def try_pop(self, state: str, kind: str, number: int) -> dict | None:
        """Non-blocking ``pop``: None when absent or claimed (a worker
        holds the item's lease for its whole review; blocking callers
        would stall that long). requests/ is exempt like its writer:
        a request must be consumable WHILE the review it caused runs,
        or it re-forces the item every pass and destroys each fresh
        verdict (production: pr #23914 in an endless re-review loop)."""
        if state != "requests" and self._leased(kind, number):
            return None
        try:
            fd = self._lock_fd(kind, number, block=False)
        except OSError:
            return None
        try:
            path = self.path(state, kind, number)
            data = self._load(path)
            if data is None:
                # a torn command file carries no recoverable intent:
                # consuming it beats wedging every pass on it
                path.unlink(missing_ok=True)
                return None
            path.unlink()
            return data
        finally:
            os.close(fd)

    def try_move(self, src_state: str, dst_state: str, kind: str, number: int,
                 mutate=None) -> bool:
        """Non-blocking ``move`` for interactive callers: False when the
        item is absent from ``src_state`` or its lock is held (a worker
        owns it -- blocking would stall the caller for the whole
        review). ``dst_state == src_state`` updates the content in
        place."""
        c = self.claim(src_state, src_state, kind, number)
        if c is None:
            return False
        try:
            data = c.read()
            if mutate is not None:
                mutate(data)
        except BaseException:
            c.abort()
            raise
        c.finish(dst_state, data)
        return True

    def request(self, kind: str, number: int, data: dict) -> Path:
        """Create an operator request. Deliberately lock-free: the
        per-item lock is held for the whole review while the item is
        claimed, and a request against a busy item must not block the
        operator (the write itself is atomic)."""
        return self._write_state("requests", kind, number, data)

    def list_state(self, state: str) -> list[tuple[str, int]]:
        out = []
        for p in (self.root / state).glob("*.json"):
            kind, _, num = p.stem.partition("-")
            if kind in KINDS and _TOKEN_RE.fullmatch(num):
                # plain numbers stay ints (callers sort and compare
                # them); suffixed tokens pass through as strings
                out.append((kind, int(num) if num.isdigit() else num))
        return sorted(out, key=lambda kn: (kn[0], forge_number(kn[1]),
                                           str(kn[1])))

    def find(self, kind: str, number: int) -> str | None:
        """The item's state; with crash remnants, the latest one."""
        found = None
        for state in STATES:
            if self.path(state, kind, number).exists():
                found = state
        return found

    # ---- worker claim protocol ----

    def claim(self, src_state: str, dst_state: str, kind: str, number: int) -> Claim | None:
        """Lease first, then rename under the transition lock: a claim
        that loses the rename race releases and returns None. The
        ``.claim`` flock held across the review is the worker's
        liveness signal; it is a namespace of its own so the
        micro-duration ``.lock`` transitions (push/pop/move/prune)
        never block for a review's length."""
        try:
            fd = self._lock_fd(kind, number, block=False, suffix="claim")
        except OSError:
            return None
        src = self.path(src_state, kind, number)
        dst = self.path(dst_state, kind, number)
        with self.lock(kind, number):
            if src == dst:
                # an in-place claim must not rename: the no-op rename still
                # fires a watcher event, and an agent watching the dir would
                # wake itself in a loop
                if not src.exists():
                    os.close(fd)
                    return None
            else:
                try:
                    os.rename(src, dst)
                except FileNotFoundError:
                    os.close(fd)
                    return None
        return Claim(self, fd, kind, number, src_state, dst)

    def _leased(self, kind: str, number) -> bool:
        """True while a live worker holds the item's review lease."""
        try:
            fd = self._lock_fd(kind, number, block=False, suffix="claim")
        except OSError:
            return True
        os.close(fd)
        return False

    def reap(self, state: str = "llm",
             to_state: str | None = "queued") -> list[tuple[str, int]]:
        """Recover items whose claim holder died: acquirable lock + file
        still in ``state``. A remnant whose item also exists in a later
        state is deleted instead of re-queued; ``to_state=None`` only
        deletes remnants (for states like reviewed/ where a lone file
        is simply valid)."""
        recovered = []
        for kind, number in self.list_state(state):
            try:
                fd = self._lock_fd(kind, number, block=False, suffix="claim")
            except OSError:
                continue  # live claim
            try:
                path = self.path(state, kind, number)
                later = [s for s in STATES[STATES.index(state) + 1:]
                         if self.path(s, kind, number).exists()]
                if later:
                    path.unlink(missing_ok=True)
                    logger.info("reaped crash remnant %s/%s-%s (item is in %s)",
                                state, kind, number, later[-1])
                elif to_state is not None and path.exists():
                    os.rename(path, self.path(to_state, kind, number))
                    recovered.append((kind, number))
                # a dead worker also leaves the wrapper's sidecar lock
                # and tmp next to the claimed file
                for suffix in (".lock", ".tmp"):
                    path.with_suffix(suffix).unlink(missing_ok=True)
            finally:
                os.close(fd)
        return recovered

    def prune(self, state: str, before: datetime,
              keep: set[tuple[str, int]] = frozenset(),
              key=None, kinds=None) -> int:
        """Delete ``state`` items whose last transition predates
        ``before``, except those whose ``key((kind, number))`` (default:
        identity) is in ``keep`` (e.g. still-open items whose archived
        verdict is backoff memory)."""
        removed = 0
        for kind, number in self.list_state(state):
            if kinds is not None and kind not in kinds:
                continue  # another agent's kind: its open-set, its call
            if (key((kind, number)) if key else (kind, number)) in keep:
                continue
            data = self.get(state, kind, number)
            if data is None:
                continue
            changed = data.get("state_changed_at")
            try:
                when = datetime.fromisoformat(changed)
            except (TypeError, ValueError):
                continue
            if when.tzinfo is None:
                # hand-edited naive timestamp: comparing it with the
                # aware cutoff would TypeError and kill the pass
                when = when.replace(tzinfo=timezone.utc)
            if when < before:
                with self.lock(kind, number):
                    self.path(state, kind, number).unlink(missing_ok=True)
                    removed += 1
                    logger.info("pruned %s/%s-%s (settled since %s)",
                                state, kind, number, changed)
                    if self.find(kind, number) is None:
                        # last trace gone: drop the item's lock file too,
                        # or locks/ grows one inode per item forever
                        self._lock_path(kind, number).unlink(missing_ok=True)
                        self._lock_path(kind, number, "claim").unlink(missing_ok=True)
        return removed
