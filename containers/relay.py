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

In-container relay for the codex backend's shell tool.

The codex CLI runs inside a review container and spawns ``codex_bridge.py``
as its MCP shell server; that bridge connects (as a
``shell_socket.ShellDispatchClient``) to a container-local unix socket.
This relay *serves* that socket and pipes it, line for line, to its own
stdin/stdout -- which the wrapper drives from the outside via
``podman exec -i``, the same JSON-over-ssh transport the review shells
already use. Shell-tool requests thus reach the wrapper's dispatcher with
no bind mount and no reverse tunnel:

    codex -> codex_bridge --socket SOCK -> [relay's SOCK]
          -> relay stdout -> podman exec -> wrapper dispatcher
    wrapper -> podman exec stdin -> relay stdin -> SOCK -> codex_bridge

One accepted connection == one codex run. Stdlib only and no fairy
imports: the review image carries none of this, the file is ``podman
cp``'d in per run.
"""

import os
import socket
import sys
import threading


def main() -> int:
    if len(sys.argv) != 2:
        sys.stderr.write("usage: relay.py SOCKET_PATH\n")
        return 2
    sock_path = sys.argv[1]
    parent = os.path.dirname(sock_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    try:
        os.unlink(sock_path)
    except OSError:
        pass

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(1)
    # The wrapper waits for this line before launching codex, so the bridge
    # can never race an unbound socket.
    sys.stderr.write("RELAY-READY\n")
    sys.stderr.flush()

    conn, _ = srv.accept()
    out = sys.stdout.buffer

    def sock_to_stdout() -> None:
        reader = conn.makefile("rb")
        while True:
            line = reader.readline()
            if not line:
                break
            out.write(line)
            out.flush()
        try:
            out.close()
        except OSError:
            pass

    threading.Thread(target=sock_to_stdout, daemon=True).start()

    stdin = sys.stdin.buffer
    while True:
        line = stdin.readline()
        if not line:
            break
        try:
            conn.sendall(line)
        except OSError:
            break
    return 0


if __name__ == "__main__":
    sys.exit(main())
