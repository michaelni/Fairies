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

Exponential backoff for the retries of a failing case, shared by every
fairy process on the machine: each failure of a key is appended to the
key's file under ``~/.fairy/retry_backoff/``, and the key is not due for
another attempt until the file is older than the wait its line count
calls for.

What belongs here: recording a key's failures and answering how long
until the next attempt is due.
What does NOT belong: deciding what counts as a failure or what to do
while a key is not due.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from pathlib import Path

from common import default_cache_path

__all__ = ["RetryTooSoon", "due_in", "ensure_due", "record"]

logger = logging.getLogger(__name__)

_KEY_INVALID_RE = re.compile(r"[^A-Za-z0-9._-]")


class RetryTooSoon(Exception):
    """The key's backoff has not elapsed; ``wait_seconds`` remain.

    ``last_failure`` is the record's last line: when it was recorded, by
    which process (any fairy process on the machine, not necessarily this
    one) and what failed.
    """

    def __init__(self, key: str, failures: int, wait_seconds: float,
                 last_failure: str) -> None:
        super().__init__(f"{key}: {failures} failures, next attempt due in "
                         f"{wait_seconds:.0f}s; last failure (possibly another "
                         f"process' call): {last_failure}")
        self.wait_seconds = wait_seconds


def _failures_of(key: str) -> Path:
    return default_cache_path("retry_backoff") / (
        f"{_KEY_INVALID_RE.sub('-', key)[:80]}."
        f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:12]}"
    )


def _hold(key: str, base_seconds: float) -> tuple[float, list[str]]:
    """``(seconds until key is due, its recorded failures)``; 0 when due."""
    failures_file = _failures_of(key)
    try:
        lines = failures_file.read_text(encoding="utf-8").splitlines()
        since_last = time.time() - failures_file.stat().st_mtime
    except FileNotFoundError:
        return 0, []
    if not lines:
        return 0, []
    wait = max(min(base_seconds * 2 ** (len(lines) - 1), 7 * 24 * 3600) - since_last,
               float(lines[-1].split(" ", 1)[0]) - time.time(), 0)
    logger.debug("%s: %d failures, last %.0fs ago, %s", key, len(lines), since_last,
                 f"due in {wait:.0f}s" if wait else "due")
    return wait, lines


def due_in(key: str, base_seconds: float) -> float:
    """Seconds until ``key`` is due for another attempt, 0 when it is.

    After n consecutive failures the next attempt is due
    ``base_seconds * 2**(n-1)`` seconds after the last one, at most a week,
    and not before the ``retry_after`` the last failure was recorded with.
    """
    return _hold(key, base_seconds)[0]


def ensure_due(key: str, base_seconds: float) -> None:
    """Raise RetryTooSoon while ``key`` is not due (``due_in``)."""
    wait, lines = _hold(key, base_seconds)
    if wait:
        raise RetryTooSoon(key, len(lines), wait, lines[-1].split(" ", 1)[1])


def record(key: str, error: str, retry_after: float = 0) -> None:
    """Append ``error`` to ``key``'s failures; "" is a success and forgets them.

    retry_after: epoch seconds before which the server refuses another
                 attempt, 0 when it named none
    """
    failures_file = _failures_of(key)
    if not error:
        failures_file.unlink(missing_ok=True)
        logger.debug("%s: succeeded, failures forgotten", key)
        return
    failures_file.parent.mkdir(parents=True, exist_ok=True)
    with failures_file.open("a", encoding="utf-8") as f:
        f.write(f"{retry_after:.0f} {time.strftime('%Y-%m-%dT%H:%M:%S')} "
                f"pid={os.getpid()} {' '.join(error.split())}\n")
    logger.debug("%s: failure recorded in %s: %s", key, failures_file, error)
