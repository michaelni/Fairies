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

The podman-free half of the codex shell bridge.

``codex_bridge.py`` runs INSIDE the codex container and imports only this
module, so the thin codex image needs no orchestration code -- none of
``podman_host`` / ``shell_tool`` / ``shell_socket``, which pull in the
review-side machinery. Stdlib only (``json``, ``socket``).

It holds the two pieces the in-container bridge needs: the canonical
shell-tool schema it advertises, and the dispatch socket client it uses
to reach ``relay.py`` (which the wrapper drives over ``podman exec``). The
wrapper-side counterparts (``shell_socket``, ``shell_tool``) re-export
these so their callers are unchanged.
"""

from __future__ import annotations

import json
import socket
from typing import Sequence

SHELL_TOOL_NAME = "shell"

SHELL_TOOL_DESCRIPTION = (
    "Run one shell command inside the ephemeral review container "
    "(full working-tree repos under /work/...). Command runs via "
    "``sh -c`` with an in-container timeout."
)


def build_shell_tool_schema(machine_labels: Sequence[str]) -> dict:
    """The one canonical shell-tool schema, in vendor-neutral form.

    Returns ``{"name", "description", "input_schema"}``; each backend
    wraps it in its own tool envelope (Anthropic ``input_schema``, OpenAI
    function ``parameters``, MCP ``inputSchema``). The ``machine`` enum
    exists only with two or more configured machines, mirroring
    exec_machine_call's dispatch.
    """
    properties: dict = {
        "command": {
            "type": "string",
            "description": "Shell command (``sh -c``).",
        },
        "cwd": {
            "type": "string",
            "description": "Working directory inside the container (e.g. /work/ffmpeg).",
        },
        "timeout_seconds": {
            "type": "number",
            "description": "Max wall seconds for this command (capped by the wrapper).",
        },
    }
    if len(machine_labels) >= 2:
        properties["machine"] = {
            "type": "string",
            "enum": list(machine_labels),
            "description": (
                f"Machine to run on (default {machine_labels[0]}). "
                "Each machine is a separate container with its own "
                "filesystem and checkouts; state does not carry over."
            ),
        }
    return {
        "name": SHELL_TOOL_NAME,
        "description": SHELL_TOOL_DESCRIPTION,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": ["command"],
        },
    }


def _send_json(sock: socket.socket, obj: dict) -> None:
    sock.sendall(json.dumps(obj, ensure_ascii=False).encode("utf-8") + b"\n")


def _recv_json_line(reader) -> dict | None:
    """Read one newline-delimited JSON object; ``None`` on EOF."""
    line = reader.readline()
    if not line:
        return None
    return json.loads(line)


class ShellDispatchClient:
    """Bridge-side client: one connection, sequential request/response."""

    def __init__(self, socket_path: str) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.connect(socket_path)
        self._reader = self._sock.makefile("rb")
        self._next_id = 0

    def call(self, args: dict) -> dict:
        self._next_id += 1
        _send_json(self._sock, {"id": self._next_id, "args": args})
        response = _recv_json_line(self._reader)
        if response is None:
            raise ConnectionError("shell dispatch server closed the connection")
        return response.get("payload") or {"error": "empty dispatch response"}

    def close(self) -> None:
        self._reader.close()
        self._sock.close()
