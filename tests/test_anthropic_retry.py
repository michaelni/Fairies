"""call_with_anthropic_retry: retry transient 429s, but fail fast on z.ai's
permanent "insufficient balance" 429 so an unfunded account is not hammered.

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

    fake.Anthropic = object
    fake.APIConnectionError = APIConnectionError
    fake.APITimeoutError = APITimeoutError
    fake.RateLimitError = RateLimitError
    fake.InternalServerError = InternalServerError
    sys.modules["anthropic"] = fake

import anthropic_common  # noqa: E402

# Real body observed from z.ai's Anthropic-compatible endpoint (2026-06)
# when the account is out of balance; the SDK surfaces it as RateLimitError.
_ZAI_BALANCE_BODY = {
    "type": "error",
    "error": {
        "type": "rate_limit_error",
        "code": "1113",
        "message": "[1113][Insufficient balance or no resource package. Please recharge.",
    },
}


def _rate_limit(body: dict | None = None, message: str = "") -> Exception:
    exc = anthropic_common.RateLimitError(message or "429")
    exc.body = body
    exc.message = message
    return exc


class BalanceExhaustedTests(unittest.TestCase):
    def test_detects_structured_1113(self) -> None:
        self.assertTrue(anthropic_common._is_balance_exhausted(_rate_limit(_ZAI_BALANCE_BODY)))

    def test_detects_message_fallback(self) -> None:
        exc = _rate_limit(None, "Insufficient balance or no resource package")
        self.assertTrue(anthropic_common._is_balance_exhausted(exc))

    def test_plain_rate_limit_is_not_balance(self) -> None:
        exc = _rate_limit({"type": "error", "error": {"type": "rate_limit_error"}}, "slow down")
        self.assertFalse(anthropic_common._is_balance_exhausted(exc))


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


if __name__ == "__main__":
    unittest.main()
