"""TCP keepalive on the wrapper's OpenAI HTTP client.

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

import openai_pr_review_wrapper as wrapper  # noqa: E402
from openai import OpenAI  # noqa: E402


class KeepaliveSocketOptionsTests(unittest.TestCase):
    def test_options_reach_httpcore_connection_pool(self) -> None:
        # httpcore applies pool._socket_options to every new socket; an
        # httpx/openai upgrade that drops them on the way through would
        # silently reintroduce the infinite-hang behavior.
        client = wrapper.make_openai_http_client()
        pool = client._transport._pool
        self.assertEqual(
            pool._socket_options,
            wrapper.OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS,
        )

    def test_openai_sdk_keeps_the_custom_http_client(self) -> None:
        http_client = wrapper.make_openai_http_client()
        client = OpenAI(
            api_key="sk-test", timeout=None, max_retries=0, http_client=http_client,
        )
        self.assertIs(client._client, http_client)

    def test_kernel_accepts_the_options(self) -> None:
        # Apply the options exactly like httpcore does and read them
        # back, so an invalid constant/value fails here instead of on
        # the first real connection.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            for level, option, value in wrapper.OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS:
                sock.setsockopt(level, option, value)
            self.assertEqual(sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE), 1)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE), 60)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL), 30)
            self.assertEqual(sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT), 8)


if __name__ == "__main__":
    unittest.main()
