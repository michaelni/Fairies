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

ci_log -- fetch the tail of a Forgejo Actions job log.

What belongs here:
    Retrieving a CI job's log from its public web URL (the ``target_url``
    of a commit status) and returning a trimmed tail, so the triage and
    reviewer LLMs can see the actual error rather than only the one-line
    status description.

Public API (__all__):
    ``fetch_job_log_tail`` -- network fetch of a job log's tail, or None.

Logging: owns ``logging.getLogger("ci_log")``; entry points should list it
in their ``setup_logging(...)`` call so the fetch debug lines surface.
"""

from __future__ import annotations

import logging
import re
import urllib.error
import urllib.request

__all__ = ["fetch_job_log_tail"]

logger = logging.getLogger("ci_log")

# A commit-status ``target_url`` for a Forgejo Actions job, e.g.
# ``https://host/owner/repo/actions/runs/59352/jobs/0``. The bare
# ``/jobs/{index}`` route 307-redirects to ``/jobs/{index}/attempt/{n}``;
# the plain-text log is served at that attempt's ``/logs`` sub-route.
# (Verified against code.ffmpeg.org 2026-06-21: a plain unauthenticated GET
# returns 200 text/plain with ``Accept-Ranges: bytes`` -- no token, no
# anti-bot challenge. Expected stable across Forgejo releases.)
_JOB_URL_RE = re.compile(r"^https?://.+?/actions/runs/\d+/jobs/\d+(?:/|$)")

# Trailing bytes requested via HTTP Range; the server honours it (206 +
# Content-Range), so only this much crosses the wire in the common case.
# Comfortably larger than any sane ``max_lines`` worth of log.
_TAIL_FETCH_BYTES = 512 * 1024
# Hard cap for a non-range (200) response so a pathologically large log
# cannot exhaust memory; the true tail is preserved for logs up to here.
_FULL_READ_CAP = 8 * 1024 * 1024


def _extract_log_tail(body: bytes, *, partial: bool, max_lines: int) -> str | None:
    """Return the last ``max_lines`` lines of ``body``, or None if empty.

    ``max_lines`` must be positive (the caller gates on the disable flag).
    ``partial`` means ``body`` is a Range slice that began mid-line, so its
    first (fragment) line is dropped.
    """
    lines = body.decode("utf-8", "replace").splitlines()
    if partial and len(lines) > 1:
        lines = lines[1:]
    return "\n".join(lines[-max_lines:]) if lines else None


def _resolve_log_url(target_url: str, *, timeout: float) -> str:
    """Follow ``/jobs/{index}`` to its current attempt and return the log URL.

    The redirect target carries the right attempt number (reruns bump it),
    so this is more robust than assuming ``attempt/1``. Only the final URL
    is used; the page body is not read.
    """
    with urllib.request.urlopen(target_url, timeout=timeout) as resp:
        return resp.geturl().rstrip("/") + "/logs"


def fetch_job_log_tail(
    target_url: str,
    *,
    max_lines: int,
    timeout: float = 30.0,
) -> str | None:
    """Return the last ``max_lines`` lines of a job's log, or None.

    ``target_url`` is the commit-status link of a Forgejo Actions job
    (``https://host/owner/repo/actions/runs/{run}/jobs/{index}``). The log
    is fetched from the public web route with an unauthenticated GET (see
    the module note); a ``Range`` request keeps the transfer to the tail.

    ``max_lines`` must be positive (the caller gates on the disable flag).
    A failed fetch (log garbage-collected, network/server error) returns an
    ``error fetching "<url>": <reason>`` string rather than None, so the LLM
    sees that a fetch was attempted and failed without any extra signalling.
    None is returned only when ``target_url`` is not a job link (nothing to
    fetch). Never raises -- a missing log must not break CI triage.
    """
    if not _JOB_URL_RE.match(target_url):
        return None
    err_prefix = f'error fetching "{target_url}": '
    try:
        log_url = _resolve_log_url(target_url, timeout=timeout)
        logger.debug("fetching log tail: %s", log_url)
        req = urllib.request.Request(log_url, headers={"Range": f"bytes=-{_TAIL_FETCH_BYTES}"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_range = resp.headers.get("Content-Range", "")
            cap = _TAIL_FETCH_BYTES + 4096 if resp.status == 206 else _FULL_READ_CAP
            body = resp.read(cap)
    except urllib.error.HTTPError as exc:
        detail = exc.read(64).decode("utf-8", "replace").strip()
        logger.debug("log fetch %s -> HTTP %s (%s)", target_url, exc.code, detail or "-")
        return err_prefix + f"HTTP {exc.code}" + (f" ({detail})" if detail else "")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        logger.debug("log fetch %s -> %s", target_url, exc)
        return err_prefix + str(exc)
    logger.debug("log fetch %s -> %d bytes", log_url, len(body))
    # A Range slice that starts past byte 0 opens mid-line; drop that fragment.
    m = re.match(r"bytes\s+(\d+)-", content_range)
    return _extract_log_tail(body, partial=bool(m and int(m.group(1)) > 0), max_lines=max_lines)
