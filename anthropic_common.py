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

Thin Anthropic-SDK glue shared by the Anthropic / GLM reviewer: API-key
loading and a retry wrapper typed to Anthropic's exception classes.

What does NOT belong: prompt text, the review pipeline, or any
OpenAI-specific code. This mirrors the small subset of ``openai_common``
the Anthropic path needs; the two cannot share one retry helper because
each is typed to its own SDK's exception classes and reads different
rate-limit headers.

Importing this module pulls in the ``anthropic`` package, so only the
Anthropic / GLM code path imports it (lazily); OpenAI-only deployments
need not install ``anthropic``.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

from dotenv import dotenv_values
from anthropic import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OverloadedError,
    RateLimitError,
)

__all__ = [
    "DEFAULT_ANTHROPIC_RETRIES",
    "load_api_key",
    "call_with_anthropic_retry",
]

logger = logging.getLogger(__name__)

# Retry budget for rate-limit (429) / overloaded (529) / 5xx responses.
DEFAULT_ANTHROPIC_RETRIES = 30

# z.ai signals two "stop calling, waiting won't help soon" states as an
# HTTP 429 with these error codes (Anthropic proper uses neither; both
# observed on z.ai's Anthropic-compatible endpoint and stable per z.ai's
# error-code table):
#   1113 -- out of balance / no resource package: dead until re-funded
#           (observed 2026-06).
#   1308 -- 5-hour usage window exhausted: dead until the window resets,
#           which can be hours away (observed 2026-07).
# Neither must consume the retry budget; retrying just hammers a dead
# endpoint while the caller's deadline runs out.
ZAI_QUOTA_EXHAUSTED_CODES = frozenset({"1113", "1308"})

T = TypeVar("T")


def _is_quota_exhausted(exc: Exception) -> bool:
    """True for a billing/quota 429 that retrying cannot fix soon
    (``ZAI_QUOTA_EXHAUSTED_CODES``).

    ``exc.body`` is the vendor's JSON error object (attacker/vendor-shaped,
    so checked structurally at this boundary); fall back to the human
    message for shapes that don't carry the structured code.
    """
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and str(error.get("code")) in ZAI_QUOTA_EXHAUSTED_CODES:
            return True
    text = str(getattr(exc, "message", "") or exc).lower()
    return ("insufficient balance" in text or "no resource package" in text
            or "usage limit reached" in text)


def load_api_key(env_var: str) -> str | None:
    """Return the API key from the environment, falling back to ``.env``.

    Parameterized by ``env_var`` so the same loader serves Anthropic
    (``ANTHROPIC_API_KEY``) and z.ai / GLM (``ZAI_API_KEY``). Process
    environment wins over ``.env``; ``dotenv_values`` is read directly so
    we never mutate the live process environment as a side effect.
    """
    env_key = os.environ.get(env_var)
    if env_key:
        return env_key
    return dotenv_values(".env").get(env_var)


def _retry_delay(exc: Exception, attempt: int) -> float:
    # Anthropic sets ``retry-after`` on 429 / 529; honor it, else back off
    # exponentially capped at 60s.
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            retry_after = headers.get("retry-after")
        except Exception:
            retry_after = None
        if retry_after is not None:
            try:
                return max(0.5, float(retry_after))
            except (TypeError, ValueError):
                pass
    return min(60.0, max(1.0, 2.0 ** attempt))


def call_with_anthropic_retry(func: Callable[[], T], *, what: str, verbose: bool) -> T:
    """Run ``func`` retrying Anthropic rate-limit / overloaded / 5xx errors.

    ``APIConnectionError`` / ``APITimeoutError`` are propagated immediately
    (not retried): like the OpenAI main call, an unpredictable long request
    is restarted by the outer caller -- which has its own deadline --
    rather than silently re-issued here, where a duplicate would be billed.
    """
    last_exc: Exception | None = None
    for attempt in range(DEFAULT_ANTHROPIC_RETRIES + 1):
        try:
            return func()
        except (APITimeoutError, APIConnectionError):
            # APITimeoutError is a subclass of APIConnectionError; one clause
            # covers both. Propagate to the outer caller.
            raise
        # OverloadedError is the SDK's dedicated 529 class; it is NOT a
        # subclass of InternalServerError (which only covers >=500 without
        # a dedicated class), so it must be listed explicitly. Observed
        # from z.ai as code 1305 during peak load 2026-07.
        except (RateLimitError, InternalServerError, OverloadedError) as exc:
            if isinstance(exc, RateLimitError) and _is_quota_exhausted(exc):
                # Balance / usage-window exhaustion dressed as a 429: fail
                # fast rather than hammer an endpoint that stays dead for
                # hours (until re-funded or the usage window resets).
                logger.error(
                    "anthropic %s during %s: balance or usage window exhausted; "
                    "not retrying: %s",
                    type(exc).__name__, what, exc,
                )
                raise
            last_exc = exc
            if attempt >= DEFAULT_ANTHROPIC_RETRIES:
                raise
            delay = _retry_delay(exc, attempt)
            if verbose:
                logger.debug(
                    "anthropic %s during %s; retrying in %.3fs",
                    type(exc).__name__, what, delay,
                )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{what} failed unexpectedly without a captured exception")
