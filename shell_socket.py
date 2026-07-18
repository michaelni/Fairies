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

Wrapper-side shell dispatch for the codex backend.

The codex CLI runs inside a container on the ``--codex-host`` and spawns
``codex_bridge.py`` as its MCP shell server; the bridge connects to a
container-local socket served by ``relay.py``, which the wrapper drives
over a ``podman exec -i`` stdio channel (see ``codex_container``). This
module's ``serve_dispatch`` is the wrapper end of that channel: it reads
the bridge's shell-tool requests off the relay's stdout and dispatches
each through ``shell_tool.exec_machine_call`` into the review containers,
writing the results back down the relay's stdin. No socket of its own
crosses the host boundary.

One ``serve_dispatch`` call == one codex run == one ``shells`` dict, so
each codex pass gets its own lazily-opened per-machine review containers,
the same isolation ``run_parallel`` gives the other backends. Sessions it
opens go through the wrapper-supplied ``open_shell`` callback, which
registers them for the wrapper's own cleanup.

Wire protocol: newline-delimited JSON, one request/response pair at a
time. Request ``{"id": n, "args": {...}}`` where ``args`` is the model's
shell-tool input; response ``{"id": n, "payload": {...}}`` where
``payload`` is exec_machine_call's result (or an in-band ``{"error": ...}``
the model can read, like every other shell backend).
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Sequence

from podman_host import ContainerShellSession
from shell_tool import exec_machine_call
# ShellDispatchClient (the codex-container side) lives in the podman-free
# shell_bridge_client so the codex image can import it without this module;
# re-exported here for the tests that drive both ends together.
from shell_bridge_client import ShellDispatchClient, _recv_json_line

__all__ = ["ShellDispatchClient", "serve_dispatch"]

logger = logging.getLogger(__name__)


def _write_json(writer, obj: dict) -> None:
    writer.write(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")
    writer.flush()


def serve_dispatch(
    reader,
    writer,
    *,
    machine_labels: Sequence[str],
    open_shell: Callable[[str], tuple[ContainerShellSession, str]],
    max_timeout_s: float,
    shells: dict[str, ContainerShellSession] | None = None,
) -> None:
    """Serve one codex run's shell dispatch over a byte-stream pair.

    ``reader``/``writer`` are binary file-like objects carrying the
    newline-delimited ``{"id", "args"}`` / ``{"id", "payload"}`` protocol,
    transport-agnostic, so the wrapper drives it over a ``podman exec -i``
    channel into the codex container (the ``relay.py`` side) exactly as it
    drives the review shells, no socket of its own required.

    One call == one codex run: a private ``shells`` dict, so a run never
    shares a container working tree with a concurrent pass. Pass ``shells``
    in to observe which sessions the run opened (the relay does, to poison
    them for forensics on a crash); it defaults to a fresh private dict.
    Returns on EOF (``reader`` exhausted).
    """
    if shells is None:
        shells = {}
    while True:
        try:
            request = _recv_json_line(reader)
        except json.JSONDecodeError as exc:
            logger.warning("shell dispatch: bad request line: %s", exc)
            _write_json(writer, {"id": None, "payload": {
                "error": f"malformed request: {exc}"}})
            continue
        if request is None:
            return
        try:
            payload = exec_machine_call(
                shells, tuple(machine_labels), open_shell,
                request.get("args"), max_timeout_s=max_timeout_s,
            )
        except Exception as exc:
            logger.exception("shell dispatch failed")
            payload = {"error": f"shell dispatch failed: {exc}"}
        _write_json(writer, {"id": request.get("id"), "payload": payload})

