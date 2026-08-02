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

TCP keepalive on the wrapper's OpenAI HTTP client.

Regression tests for the silent-hang fix: non-streaming
``responses.create`` calls legitimately receive zero bytes for tens of
minutes, so a connection dropped without a RST blocked ``recv()``
forever (production 2026-05-07: three concurrent reviews wedged until
``--llm-timeout``; same signature repeatedly under --simulate-past,
e.g. simpast-runs/pr23381-reworded/unrel-reworded_6). The fix routes
every OpenAI request through a client whose sockets have TCP keepalive
enabled. These tests pin that the options survive the
DefaultHttpxClient -> HTTPTransport -> httpcore plumbing and that the
kernel accepts them.
"""

from __future__ import annotations

import socket
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import openai_reviewer  # noqa: E402
from openai import OpenAI  # noqa: E402


class KeepaliveSocketOptionsTests(unittest.TestCase):
    def test_options_reach_httpcore_connection_pool(self) -> None:
        # httpcore applies pool._socket_options to every new socket; an
        # httpx/openai upgrade that drops them on the way through would
        # silently reintroduce the infinite-hang behavior.
        client = openai_reviewer.make_openai_http_client()
        pool = client._transport._pool
        self.assertEqual(
            pool._socket_options,
            openai_reviewer.OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS,
        )

    def test_openai_sdk_keeps_the_custom_http_client(self) -> None:
        http_client = openai_reviewer.make_openai_http_client()
        client = OpenAI(
            api_key="sk-test", timeout=None, max_retries=0, http_client=http_client,
        )
        self.assertIs(client._client, http_client)

    def test_kernel_accepts_the_options(self) -> None:
        # Apply the options exactly like httpcore does and read them
        # back, so an invalid constant/value fails here instead of on
        # the first real connection.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            for level, option, value in openai_reviewer.OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS:
                sock.setsockopt(level, option, value)
            self.assertEqual(sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE), 1)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE), 60)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL), 30)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT), 8)


if __name__ == "__main__":
    unittest.main()
