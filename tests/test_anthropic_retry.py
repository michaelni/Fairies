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

common.call_with_retry with anthropic_common.retryable: retry transient
429s/529s, but fail fast on z.ai's "insufficient balance" and "usage
window exhausted" 429s so a dead-for-hours endpoint is not hammered.

The replies are ``httpx.Response`` objects built from the real z.ai
bodies, so the test runs without network.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import anthropic_common  # noqa: E402
import common  # noqa: E402

# Real bodies observed from z.ai's Anthropic-compatible endpoint, both as
# HTTP 429. 1113 (2026-06): account out of balance. 1308 (2026-07): the
# account's 5-hour usage window is exhausted.
_ZAI_BALANCE_BODY = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "code": "1113",
        "message": "[1113][Insufficient balance or no resource package. Please recharge.",
    },
}
_ZAI_USAGE_WINDOW_BODY = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "code": "1308",
        "message": "[1308][Usage limit reached for 5 hour. Your limit will "
                   "reset at 2026-07-03 11:11:54][20260703110251648ad94226c14a7e]",
    },
}
# Real body observed from z.ai (2026-07) during peak load, as HTTP 529.
_ZAI_OVERLOADED_BODY = {
    "type": "error",
    "error": {
        "type": "overloaded_error",
        "code": "1305",
        "message": "[1305][The service may be temporarily overloaded, please "
                   "try again later][20260703094815abf86bff2a734721]",
    },
}
# Real body observed from OpenRouter (2026-09-28), as HTTP 200.
OPENROUTER_EMPTY_BODY = {
    "type": "error",
    "error": {
        "type": "api_error",
        "error_type": "provider_unavailable",
        "message": "Provider returned an empty response",
    },
    "content": None,
    "id": None,
    "model": None,
    "role": None,
    "stop_reason": None,
    "usage": None,
}


def _response(status: int, body: dict | str) -> httpx.Response:
    request = httpx.Request("POST", "https://example.com/v1/messages")
    if isinstance(body, str):
        return httpx.Response(status, text=body, request=request)
    return httpx.Response(status, json=body, request=request)


def _status_error(status: int, body: dict | str) -> httpx.HTTPStatusError:
    response = _response(status, body)
    return httpx.HTTPStatusError(str(status), request=response.request, response=response)


class _Scripted:
    """Raises or returns the next queued reply on each call."""

    def __init__(self, replies: list) -> None:
        self._replies = list(replies)
        self.calls = 0

    def __call__(self) -> object:
        self.calls += 1
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def _retry(func: _Scripted) -> object:
    with mock.patch.object(common.time, "sleep"):
        return common.call_with_retry(func, retryable=anthropic_common.retryable, what="probe")


class QuotaExhaustedTests(unittest.TestCase):
    def test_detects_structured_1113(self) -> None:
        self.assertTrue(anthropic_common._is_quota_exhausted(_response(429, _ZAI_BALANCE_BODY)))

    def test_detects_structured_1308(self) -> None:
        self.assertTrue(anthropic_common._is_quota_exhausted(_response(429, _ZAI_USAGE_WINDOW_BODY)))

    def test_detects_message_fallback(self) -> None:
        self.assertTrue(anthropic_common._is_quota_exhausted(
            _response(429, "Insufficient balance or no resource package")))

    def test_plain_rate_limit_is_not_quota(self) -> None:
        self.assertFalse(anthropic_common._is_quota_exhausted(
            _response(429, {"type": "error", "error": {"type": "rate_limit_error"}})))


class RetryTests(unittest.TestCase):
    def test_balance_error_is_not_retried(self) -> None:
        func = _Scripted([_status_error(429, _ZAI_BALANCE_BODY)])
        with self.assertRaises(httpx.HTTPStatusError):
            _retry(func)
        self.assertEqual(1, func.calls)

    def test_usage_window_error_is_not_retried(self) -> None:
        func = _Scripted([_status_error(429, _ZAI_USAGE_WINDOW_BODY)])
        with self.assertRaises(httpx.HTTPStatusError):
            _retry(func)
        self.assertEqual(1, func.calls)

    def test_bad_request_is_not_retried(self) -> None:
        func = _Scripted([_status_error(400, "invalid_request_error")])
        with self.assertRaises(httpx.HTTPStatusError):
            _retry(func)
        self.assertEqual(1, func.calls)

    def test_transient_rate_limit_is_retried(self) -> None:
        func = _Scripted([_status_error(429, "please slow down"),
                          _status_error(429, "please slow down"), "ok"])
        self.assertEqual("ok", _retry(func))
        self.assertEqual(3, func.calls)

    def test_overloaded_529_is_retried(self) -> None:
        func = _Scripted([_status_error(529, _ZAI_OVERLOADED_BODY),
                          _status_error(529, _ZAI_OVERLOADED_BODY), "ok"])
        self.assertEqual("ok", _retry(func))
        self.assertEqual(3, func.calls)

    def test_error_body_of_transient_type_is_retried(self) -> None:
        func = _Scripted([anthropic_common.MessagesError(OPENROUTER_EMPTY_BODY["error"]), "ok"])
        self.assertEqual("ok", _retry(func))
        self.assertEqual(2, func.calls)

    def test_error_body_of_request_type_is_not_retried(self) -> None:
        func = _Scripted([anthropic_common.MessagesError(
            {"type": "invalid_request_error", "message": "max_tokens: too large"})])
        with self.assertRaises(anthropic_common.MessagesError):
            _retry(func)
        self.assertEqual(1, func.calls)


if __name__ == "__main__":
    unittest.main()
