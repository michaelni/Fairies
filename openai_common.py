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

Shared OpenAI-SDK primitives used across the project.

Collects the thin OpenAI-SDK-facing glue that more than one module
needs: the retry wrapper for rate-limit / transient errors, file
upload / delete / exists helpers, the response-text extractor, a few
JSON type aliases, and small utilities like ``_obj_get`` and
``log_progress``.

This is a leaf module: nothing here may import from
``pr_review_wrapper`` or any of its siblings. The only allowed
in-repo dependency is ``common``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import TypeAlias

from dotenv import dotenv_values
from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

# Re-exported for callers that do ``from openai_common import JsonObject``
# (e.g. pr_review_wrapper, openai_container, openai_vector_store).
# Single source of truth lives in common.py; both forge_gcli and
# openai_common re-export so neither subsystem has to depend on the
# other.
from common import JsonPrimitive, JsonValue, JsonObject
# The debug-dump helpers are vendor-neutral and live in common; this
# module's extract_response_text uses them for its failure dumps.
from common import dump_response_debug_artifacts, response_to_debug_json


logger = logging.getLogger(__name__)


InputContentItem: TypeAlias = dict[str, object]
ResponseKwargs: TypeAlias = dict[str, object]


def load_api_key() -> str | None:
    """Return the OpenAI API key from the environment, falling back to ``.env``.

    Process environment wins over ``.env`` so operators can override on
    the command line without editing the file. ``dotenv_values(".env")``
    is used instead of ``load_dotenv()`` to avoid mutating the live
    process environment as a side effect of reading the key.
    """
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key:
        return env_key
    return dotenv_values(".env").get("OPENAI_API_KEY")


DEFAULT_RATE_LIMIT_RETRIES = 30
# Transient connection / timeout errors are retried a small number of
# times for short helper calls (file uploads, container retrieves, etc.)
# where a quick retry is almost always cheaper than asking the outer
# caller to restart the whole wrapper. The main LLM responses.create
# call opts out of this and propagates such errors to the caller.
DEFAULT_TRANSIENT_API_RETRIES = 3


def log_progress(prefix: str, done: int, total: int) -> None:
    total = max(total, 0)
    done = min(max(done, 0), total) if total else 0
    percent = (100.0 * done / total) if total else 100.0
    logger.info("%s: %d/%d (%.1f%%)", prefix, done, total, percent)


def _obj_get(obj: object, key: str, default: object = None) -> object:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def parse_retry_delay_seconds(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    m = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*ms', text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1)) / 1000.0

    m = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)\s*s', text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.fullmatch(r'([0-9]+(?:\.[0-9]+)?)', text)
    if m:
        return float(m.group(1))

    return None


def get_rate_limit_delay(exc: RateLimitError, attempt: int) -> float:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)

    if headers is not None:
        for key in ("retry-after", "x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
            try:
                value = headers.get(key)
            except Exception:
                value = None
            delay = parse_retry_delay_seconds(value)
            if delay is not None:
                return max(0.5, delay)

    m = re.search(r"Please try again in\s+([0-9]+(?:\.[0-9]+)?)s", str(exc), flags=re.IGNORECASE)
    if m:
        return max(0.5, float(m.group(1)))

    return min(60.0, max(1.0, 2.0 ** attempt))


def call_with_rate_limit_retry(
    func,
    *,
    what: str,
    verbose: bool,
    retry_transient: bool = True,
):
    """Run ``func`` with retry handling for OpenAI errors.

    Always retries ``RateLimitError``, ``AuthenticationError`` and
    ``InternalServerError`` up to ``DEFAULT_RATE_LIMIT_RETRIES`` times.

    When ``retry_transient`` is True (default), also retries
    ``APITimeoutError`` and ``APIConnectionError`` up to
    ``DEFAULT_TRANSIENT_API_RETRIES`` times with a short backoff. This
    is appropriate for short helper calls (file uploads, container
    retrieves, vector-store probes) where a quick retry is much cheaper
    than asking the outer caller to restart the whole wrapper.

    When ``retry_transient`` is False, ``APITimeoutError`` and
    ``APIConnectionError`` are propagated immediately. The main LLM
    ``responses.create`` call uses this so unpredictable long requests
    are restarted by the outer caller (which has its own deadline)
    instead of being silently re-issued here.
    """
    last_exc: Exception | None = None
    transient_attempts = 0
    for attempt in range(DEFAULT_RATE_LIMIT_RETRIES + 1):
        try:
            return func()
        except RateLimitError as exc:
            last_exc = exc
            if attempt >= DEFAULT_RATE_LIMIT_RETRIES:
                raise
            delay = get_rate_limit_delay(exc, attempt)
            if verbose:
                logger.debug("rate limited during %s; retrying in %.3fs", what, delay)
            time.sleep(delay)
        except (AuthenticationError, InternalServerError) as exc:
            last_exc = exc
            if attempt >= DEFAULT_RATE_LIMIT_RETRIES:
                raise
            delay = min(60.0, max(1.0, 2.0 ** attempt))
            if verbose:
                logger.debug(
                    "status=%s code=%s req_id=%s %s error=%s; retrying in %.3fs",
                    exc.status_code,
                    exc.code,
                    exc.request_id,
                    what,
                    str(exc).replace("\n", " "),
                    delay,
                )
            time.sleep(delay)
        except (APITimeoutError, APIConnectionError) as exc:
            # APITimeoutError is a subclass of APIConnectionError, so a
            # single except clause covers both.
            if not retry_transient:
                if verbose:
                    logger.debug(
                        "transient %s error during %s (retry_transient=False); propagating: %s",
                        type(exc).__name__,
                        what,
                        str(exc).replace("\n", " "),
                    )
                raise
            last_exc = exc
            if transient_attempts >= DEFAULT_TRANSIENT_API_RETRIES:
                if verbose:
                    logger.debug(
                        "transient %s error during %s exceeded %d retries; propagating: %s",
                        type(exc).__name__,
                        what,
                        DEFAULT_TRANSIENT_API_RETRIES,
                        str(exc).replace("\n", " "),
                    )
                raise
            transient_attempts += 1
            delay = min(30.0, max(1.0, 2.0 ** transient_attempts))
            if verbose:
                logger.debug(
                    "transient %s error during %s (attempt %d/%d); retrying in %.3fs: %s",
                    type(exc).__name__,
                    what,
                    transient_attempts,
                    DEFAULT_TRANSIENT_API_RETRIES,
                    delay,
                    str(exc).replace("\n", " "),
                )
            time.sleep(delay)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{what} failed unexpectedly without a captured exception")


def openai_file_exists(client: OpenAI, file_id: str, *, verbose: bool) -> bool:
    started = time.monotonic()
    if verbose:
        logger.debug("openai files.retrieve start file_id=%s", file_id)
    try:
        call_with_rate_limit_retry(
            lambda: client.files.retrieve(file_id),
            what=f"file retrieve {file_id}",
            verbose=verbose,
        )
        if verbose:
            logger.debug("openai files.retrieve ok file_id=%s dt=%.3fs", file_id, time.monotonic() - started)
        return True
    except Exception:
        if verbose:
            logger.debug("cached file id is not retrievable file_id=%s dt=%.3fs", file_id, time.monotonic() - started)
        return False


def upload_local_file(client: OpenAI, path: Path, *, what: str, verbose: bool) -> str:
    started = time.monotonic()
    if verbose:
        logger.debug("openai files.create start what=%s path=%s bytes=%d", what, path, path.stat().st_size)
    with path.open("rb") as f:
        uploaded = call_with_rate_limit_retry(
            lambda: client.files.create(file=f, purpose="user_data"),
            what=what,
            verbose=verbose,
        )
    file_id = getattr(uploaded, "id", None)
    if not isinstance(file_id, str) or not file_id:
        raise RuntimeError(f"{what} did not return a usable file id")
    if verbose:
        logger.debug("openai files.create ok what=%s path=%s file_id=%s dt=%.3fs", what, path, file_id, time.monotonic() - started)
    return file_id


def upload_text_file(client: OpenAI, *, filename: str, text: str, verbose: bool = False) -> str:
    suffix = Path(filename).suffix or ".txt"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=suffix, delete=False) as tf:
        tf.write(text)
        temp_path = Path(tf.name)

    try:
        file_id = upload_local_file(
            client,
            temp_path,
            what=f"file upload for {filename}",
            verbose=verbose,
        )
    finally:
        try:
            temp_path.unlink()
        except OSError:
            pass

    return file_id


def delete_uploaded_file(client: OpenAI, file_id: str, *, verbose: bool) -> None:
    try:
        if verbose:
            logger.debug("openai files.delete start %s", file_id)
        client.files.delete(file_id)
        if verbose:
            logger.debug("openai files.delete ok %s", file_id)
    except Exception as exc:
        if verbose:
            logger.warning("failed to delete uploaded file %s: %s", file_id, exc)


def extract_response_text(
    response: object,
    *,
    response_kwargs: ResponseKwargs | None = None,
    debug_dir: str | None = None,
    verbose: bool = False,
) -> str:
    text = getattr(response, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text

    dumped = response_to_debug_json(response)

    output = dumped.get("output") if isinstance(dumped, dict) else None
    if isinstance(output, list):
        parts: list[str] = []
        refusals: list[str] = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                ctype = c.get("type")
                if ctype == "output_text":
                    text_val = c.get("text")
                    if isinstance(text_val, str):
                        parts.append(text_val)
                elif ctype == "refusal":
                    refusal_text = c.get("refusal") or c.get("text")
                    if isinstance(refusal_text, str) and refusal_text.strip():
                        refusals.append(refusal_text)
        if parts:
            return "\n".join(parts)
        if refusals:
            raise RuntimeError(f"model refusal: {' | '.join(refusals)}")

    output_item_types: list[str] = []
    output_item_statuses: list[str] = []
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            item_status = item.get("status")
            if isinstance(item_type, str):
                output_item_types.append(item_type)
            if isinstance(item_status, str):
                output_item_statuses.append(item_status)

    response_status = dumped.get("status") if isinstance(dumped, dict) else None
    incomplete_details = dumped.get("incomplete_details") if isinstance(dumped, dict) else None
    dump_path = None
    if response_kwargs is not None and debug_dir:
        dump_path = dump_response_debug_artifacts(
            response,
            response_kwargs,
            debug_dir=debug_dir,
            verbose=verbose,
        )

    summary = (
        "could not extract output text from OpenAI response; "
        f"response_status={response_status!r}; "
        f"output_item_types={output_item_types!r}; "
        f"output_item_statuses={output_item_statuses!r}; "
        f"incomplete_details={json.dumps(incomplete_details, ensure_ascii=False)}"
    )
    if isinstance(incomplete_details, dict) and incomplete_details.get("reason") == "max_output_tokens":
        summary += "; hint=increase --max-output-tokens and/or reduce tool use"
    if dump_path:
        summary += f"; debug_dump={dump_path}"
    raise RuntimeError(summary)
