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

MCP stdio server bridging the codex CLI to the wrapper's shell dispatch.

The codex CLI spawns this script per run (``mcp_servers.shell`` in the
config the wrapper passes via ``-c``) and speaks MCP -- JSON-RPC 2.0,
newline-delimited over stdin/stdout -- to it. This bridge is a dumb pipe:
it advertises the one canonical ``shell`` tool and forwards each
``tools/call`` over the wrapper's per-run unix socket
(``shell_socket.ShellDispatchClient``), where the real dispatch,
container lifecycle and timeouts live. It holds no credentials and can
execute nothing itself.

The socket connection opens lazily on the first ``tools/call`` and then
persists: one bridge process = one connection = the codex run's private
container set on the wrapper side.

Implements the MCP subset codex needs from a tools-only server:
``initialize``, ``tools/list``, ``tools/call``, ``ping``, and ignores
notifications. Unknown methods with an id get a MethodNotFound error.
"""

from __future__ import annotations

import argparse
import json
import sys

from shell_bridge_client import ShellDispatchClient, build_shell_tool_schema

METHOD_NOT_FOUND = -32601
INTERNAL_ERROR = -32603

FALLBACK_PROTOCOL_VERSION = "2025-06-18"


def _write(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _reply(msg_id: object, result: dict) -> None:
    _write({"jsonrpc": "2.0", "id": msg_id, "result": result})


def _reply_error(msg_id: object, code: int, message: str) -> None:
    _write({"jsonrpc": "2.0", "id": msg_id,
            "error": {"code": code, "message": message}})


class Bridge:
    def __init__(self, socket_path: str, machine_labels: list[str]) -> None:
        self.socket_path = socket_path
        self.machine_labels = machine_labels
        self.client: ShellDispatchClient | None = None

    def handle(self, msg: dict) -> None:
        method = msg.get("method")
        msg_id = msg.get("id")
        if method == "initialize":
            params = msg.get("params") or {}
            _reply(msg_id, {
                "protocolVersion": params.get("protocolVersion")
                                   or FALLBACK_PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fairy-shell", "version": "1.0"},
            })
        elif method == "tools/list":
            schema = build_shell_tool_schema(self.machine_labels)
            _reply(msg_id, {"tools": [{
                "name": schema["name"],
                "description": schema["description"],
                "inputSchema": schema["input_schema"],
            }]})
        elif method == "tools/call":
            self._handle_call(msg_id, msg.get("params") or {})
        elif method == "ping":
            _reply(msg_id, {})
        elif msg_id is not None:
            _reply_error(msg_id, METHOD_NOT_FOUND, f"unknown method {method!r}")

    def _handle_call(self, msg_id: object, params: dict) -> None:
        name = params.get("name")
        if name != "shell":
            _reply(msg_id, {
                "content": [{"type": "text", "text": f"unknown tool {name!r}"}],
                "isError": True,
            })
            return
        try:
            if self.client is None:
                self.client = ShellDispatchClient(self.socket_path)
            payload = self.client.call(params.get("arguments") or {})
        except Exception as exc:
            _reply_error(msg_id, INTERNAL_ERROR, f"shell dispatch: {exc}")
            return
        _reply(msg_id, {
            "content": [{"type": "text",
                         "text": json.dumps(payload, ensure_ascii=False)}],
            "isError": False,
        })

    def run(self) -> None:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            self.handle(msg)
        if self.client is not None:
            self.client.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True,
                        help="wrapper's shell dispatch unix socket path")
    parser.add_argument("--machine", action="append", required=True,
                        help="machine label; repeatable, first is the default")
    args = parser.parse_args(argv)
    Bridge(args.socket, args.machine).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
