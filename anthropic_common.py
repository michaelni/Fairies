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

Thin Messages-API glue shared by the Anthropic / GLM / OpenRouter
reviewer: Anthropic's own endpoint and which HTTP replies
``common.call_with_retry`` retries.

What does NOT belong: prompt text, the review pipeline, or any
OpenAI-specific code.
"""

from __future__ import annotations

import logging

import httpx

__all__ = [
    "ANTHROPIC_API_URL",
    "retryable",
]

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com"

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


def _is_quota_exhausted(response: httpx.Response) -> bool:
    """True for a billing/quota 429 that retrying cannot fix soon
    (``ZAI_QUOTA_EXHAUSTED_CODES``).

    The body is the vendor's JSON error object (vendor-shaped, so checked
    structurally at this boundary); fall back to the human message for
    shapes that don't carry the structured code.
    """
    try:
        code = response.json()["error"]["code"]
    except (ValueError, KeyError, TypeError):
        code = None
    if str(code) in ZAI_QUOTA_EXHAUSTED_CODES:
        return True
    text = response.text.lower()
    return ("insufficient balance" in text or "no resource package" in text
            or "usage limit reached" in text)


def retryable(exc: Exception) -> bool:
    """Whether ``common.call_with_retry`` should retry ``exc``: a Messages
    rate-limit (429) / overloaded (529) / 5xx reply, except z.ai's quota
    exhaustion.

    Connection errors and timeouts are not retried: like the OpenAI main
    call, an unpredictable long request is restarted by the outer caller
    -- which has its own deadline -- rather than silently re-issued here,
    where a duplicate would be billed.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return False
    status = exc.response.status_code
    if status == 429 and _is_quota_exhausted(exc.response):
        logger.error("messages HTTP 429: balance or usage window exhausted; not retrying: %s",
                     exc.response.text[:2000])
        return False
    return status == 429 or status >= 500
