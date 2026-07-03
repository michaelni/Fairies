"""call_with_anthropic_retry: retry transient 429s/529s, but fail fast on
z.ai's "insufficient balance" and "usage window exhausted" 429s so a
dead-for-hours endpoint is not hammered.

The ``anthropic`` SDK is mocked (like the reviewer replay tests) so the
RateLimitError class is a plain Exception we can raise with a ``.body``.
"""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "anthropic" not in sys.modules:
    fake = types.ModuleType("anthropic")

    class _E(Exception):
        pass

    class APIConnectionError(_E):
        pass

    class APITimeoutError(APIConnectionError):
        pass

    class RateLimitError(_E):
        pass

    class InternalServerError(_E):
        pass

    class OverloadedError(_E):
        pass

    fake.Anthropic = object
    fake.APIConnectionError = APIConnectionError
    fake.APITimeoutError = APITimeoutError
    fake.RateLimitError = RateLimitError
    fake.InternalServerError = InternalServerError
    fake.OverloadedError = OverloadedError
    sys.modules["anthropic"] = fake

import anthropic_common  # noqa: E402

# Real bodies observed from z.ai's Anthropic-compatible endpoint; the SDK
# surfaces both as RateLimitError. 1113 (2026-06): account out of balance.
# 1308 (2026-07): the account's 5-hour usage window is exhausted.
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
# Real body observed from z.ai (2026-07) during peak load; the SDK surfaces
# it as OverloadedError (a dedicated 529 class, NOT InternalServerError).
_ZAI_OVERLOADED_BODY = {
    "type": "error",
    "error": {
        "type": "overloaded_error",
        "code": "1305",
        "message": "[1305][The service may be temporarily overloaded, please "
                   "try again later][20260703094815abf86bff2a734721]",
    },
}


def _rate_limit(body: dict | None = None, message: str = "") -> Exception:
    exc = anthropic_common.RateLimitError(message or "429")
    exc.body = body
    exc.message = message
    return exc


class QuotaExhaustedTests(unittest.TestCase):
    def test_detects_structured_1113(self) -> None:
        self.assertTrue(anthropic_common._is_quota_exhausted(_rate_limit(_ZAI_BALANCE_BODY)))

    def test_detects_structured_1308(self) -> None:
        self.assertTrue(anthropic_common._is_quota_exhausted(_rate_limit(_ZAI_USAGE_WINDOW_BODY)))

    def test_detects_message_fallback(self) -> None:
        exc = _rate_limit(None, "Insufficient balance or no resource package")
        self.assertTrue(anthropic_common._is_quota_exhausted(exc))

    def test_plain_rate_limit_is_not_quota(self) -> None:
        exc = _rate_limit({"type": "error", "error": {"type": "rate_limit_error"}}, "slow down")
        self.assertFalse(anthropic_common._is_quota_exhausted(exc))


class RetryTests(unittest.TestCase):
    def test_balance_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def func():
            calls["n"] += 1
            raise _rate_limit(_ZAI_BALANCE_BODY)

        with mock.patch.object(anthropic_common.time, "sleep") as sleep:
            with self.assertRaises(anthropic_common.RateLimitError):
                anthropic_common.call_with_anthropic_retry(func, what="probe", verbose=False)
        self.assertEqual(1, calls["n"])
        sleep.assert_not_called()

    def test_usage_window_error_is_not_retried(self) -> None:
        calls = {"n": 0}

        def func():
            calls["n"] += 1
            raise _rate_limit(_ZAI_USAGE_WINDOW_BODY)

        with mock.patch.object(anthropic_common.time, "sleep") as sleep:
            with self.assertRaises(anthropic_common.RateLimitError):
                anthropic_common.call_with_anthropic_retry(func, what="probe", verbose=False)
        self.assertEqual(1, calls["n"])
        sleep.assert_not_called()

    def test_transient_rate_limit_is_retried(self) -> None:
        calls = {"n": 0}

        def func():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _rate_limit(None, "please slow down")
            return "ok"

        with mock.patch.object(anthropic_common.time, "sleep"):
            result = anthropic_common.call_with_anthropic_retry(func, what="probe", verbose=False)
        self.assertEqual("ok", result)
        self.assertEqual(3, calls["n"])

    def test_overloaded_529_is_retried(self) -> None:
        calls = {"n": 0}

        def func():
            calls["n"] += 1
            if calls["n"] < 3:
                exc = anthropic_common.OverloadedError("529")
                exc.body = _ZAI_OVERLOADED_BODY
                raise exc
            return "ok"

        with mock.patch.object(anthropic_common.time, "sleep"):
            result = anthropic_common.call_with_anthropic_retry(func, what="probe", verbose=False)
        self.assertEqual("ok", result)
        self.assertEqual(3, calls["n"])


if __name__ == "__main__":
    unittest.main()
