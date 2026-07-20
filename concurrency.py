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

Cross-process concurrency limits, one counting semaphore per provider.

What belongs here: capping how many network calls every fairy process on
the machine has in flight against one provider at a time.

What does NOT belong: the in-process worker counts (``--llm-parallelism``)
and any provider-specific knowledge.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import logging
import random
import re
import time
from typing import Iterable, Iterator

from common import default_cache_path

__all__ = ["configure", "parse_limit", "slot"]

logger = logging.getLogger(__name__)

# A slot is an flock'd file: the kernel releases it when the holder dies, so
# a killed run never leaks capacity (a POSIX named semaphore would).
_RESCAN_DELAY_S = (0.2, 0.7)
_PROVIDER_RE = re.compile(r"[A-Za-z0-9_-]+")

_limits: dict[str, int] = {}


def parse_limit(value: str) -> tuple[str, int]:
    """argparse ``type`` for one ``provider:count`` pair."""
    provider, sep, count = value.partition(":")
    if not sep or not count.isdigit() or int(count) < 1:
        raise argparse.ArgumentTypeError(
            f"{value!r}: expected provider:count with count >= 1, e.g. codex:2"
        )
    if not _PROVIDER_RE.fullmatch(provider):
        raise argparse.ArgumentTypeError(
            f"{value!r}: provider must match {_PROVIDER_RE.pattern}"
        )
    return provider, int(count)


def configure(limits: Iterable[tuple[str, int]]) -> None:
    _limits.clear()
    _limits.update(limits)
    logger.debug("provider concurrency limits: %s", _limits or "-")


@contextmanager
def slot(provider: str) -> Iterator[None]:
    """Hold one of ``provider``'s slots for the duration of the block.

    Uncapped providers pass straight through. Waiting is unbounded; the
    caller's own timeout (``--llm-timeout``) bounds the whole review.
    Processes that disagree on the count open different numbers of slot
    files, so the effective cap is the largest count in play.
    """
    limit = _limits.get(provider)
    if limit is None:
        yield
        return

    lock_dir = default_cache_path("locks")
    lock_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    waited = False
    while True:
        for index in range(limit):
            handle = open(lock_dir / f"{provider}.slot.{index}", "a")
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close()
                continue
            except BaseException:
                handle.close()
                raise
            (logger.info if waited else logger.debug)(
                "%s slot %d/%d acquired after %.1fs",
                provider, index, limit, time.monotonic() - started,
            )
            try:
                yield
            finally:
                handle.close()
                logger.debug("%s slot %d released", provider, index)
            return
        waited = True
        logger.debug("%s: all %d slots busy, rescanning", provider, limit)
        time.sleep(random.uniform(*_RESCAN_DELAY_S))
