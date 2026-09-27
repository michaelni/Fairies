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

GeminiReviewer: one Google Gemini API review pass behind the shared
``Reviewer`` interface, driven by ``tool_loop`` over the REST API.

The model investigates via the ``shell`` function and returns its verdict
by calling a ``submit_review`` function whose parameters are the role's
output schema.
"""

from __future__ import annotations

import logging

import httpx

import concurrency
from common import JsonObject, call_with_retry, dump_response_debug_artifacts, load_api_key
from llm_prompt import REVIEWER_ROLE
from llm_review_api import ProviderTurnFailed, ReviewContext, Reviewer, RoleSpec
from shell_tool import abort_if_cancelled
from tool_loop import Conversation, ToolCall, review_prompt, review_tools, run_tool_loop

__all__ = [
    "GEMINI_EFFORTS",
    "GeminiReviewer",
]

logger = logging.getLogger(__name__)

GEMINI_EFFORTS = ("minimal", "low", "medium", "high")

_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def _daily_quota(response: httpx.Response) -> str | None:
    """The per-day quota a 429 reports exhausted, if any."""
    try:
        return next((v["quotaId"] for d in response.json()["error"]["details"]
                     for v in d.get("violations", []) if "PerDay" in v["quotaId"]), None)
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def _retryable(exc: Exception) -> bool:
    # Rate limits (https://ai.google.dev/gemini-api/docs/rate-limits) answer
    # 429 until the window passes; back off instead of failing the review,
    # unless the exhausted window is a whole day.
    if not (isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code in (429, 500, 502, 503, 504)):
        return False
    quota = _daily_quota(exc.response) if exc.response.status_code == 429 else None
    if quota:
        logger.error("generate_content: daily quota %s exhausted; not retrying", quota)
    return quota is None


class GeminiReviewer(Reviewer):
    """One Gemini API pass of a ``RoleSpec`` behind the shared interface.

    ``effort`` is a ``GEMINI_EFFORTS`` thinking level; ``None`` keeps the
    model default. ``reasoning_summary`` ``concise`` / ``detailed`` asks
    for thought summaries. ``name`` is the label recorded on the
    ``Review`` (e.g. ``"gemini:gemini-3.8-flash"``).
    """

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        api_key_env: str = "GEMINI_API_KEY",
        max_tool_rounds: int = 0,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
        reasoning_summary: str | None = None,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if effort is not None and effort not in GEMINI_EFFORTS:
            raise ValueError(f"effort {effort!r} not in {GEMINI_EFFORTS}")
        self.model = model
        self.name = name
        self.role = role
        self.api_key_env = api_key_env
        self.max_tool_rounds = max_tool_rounds
        self.exec_timeout_s = exec_timeout_s
        self.effort = effort
        self.reasoning_summary = reasoning_summary
        self.verbose = verbose
        self.debug_dir = debug_dir

    def _client(self) -> httpx.Client:
        api_key = load_api_key(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set (env or .env)")
        return httpx.Client(timeout=900.0, headers={"x-goog-api-key": api_key})

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        system, texts = review_prompt(self.role, ctx, vendor="gemini", model=self.model)
        conversation = _Conversation(self, ctx, system, texts, review_tools(self.role, ctx))
        return run_tool_loop(conversation, self.role, ctx, what="generate_content",
                             max_tool_rounds=self.max_tool_rounds,
                             exec_timeout_s=self.exec_timeout_s)


class _Conversation(Conversation):
    def __init__(self, reviewer: GeminiReviewer, ctx: ReviewContext, system: str,
                 texts: list[str], tools: list[JsonObject]) -> None:
        self.reviewer = reviewer
        self.client = reviewer._client()
        self.ctx = ctx
        thinking: JsonObject = {
            "includeThoughts": reviewer.reasoning_summary in ("concise", "detailed")}
        if reviewer.effort is not None:
            thinking["thinkingLevel"] = reviewer.effort
        self.config: JsonObject = {
            "systemInstruction": {"parts": [{"text": system}]},
            "tools": [{"functionDeclarations": [
                {"name": t["name"], "description": t["description"],
                 "parametersJsonSchema": t["input_schema"]} for t in tools]}],
            "generationConfig": {"thinkingConfig": thinking},
        }
        self.contents: list[JsonObject] = [
            {"role": "user", "parts": [{"text": t} for t in texts]}]
        self.conv_path: str | None = None

    def send(self) -> list[ToolCall]:
        reviewer = self.reviewer
        body = {**self.config, "contents": list(self.contents)}

        def post() -> JsonObject:
            response = self.client.post(_URL.format(model=reviewer.model), json=body)
            if response.is_error:
                logger.warning("generate_content HTTP %d: %s",
                               response.status_code, response.text[:2000])
            return response.raise_for_status().json()

        abort_if_cancelled()
        with concurrency.slot(reviewer.name.partition(":")[0]):
            response = call_with_retry(post, retryable=_retryable, what="generate_content")
        usage = response.get("usageMetadata", {})
        logger.info("generate_content usage prompt=%s cached=%s output=%s thoughts=%s",
                    usage.get("promptTokenCount"), usage.get("cachedContentTokenCount"),
                    usage.get("candidatesTokenCount"), usage.get("thoughtsTokenCount"))
        if reviewer.debug_dir:
            self.conv_path = dump_response_debug_artifacts(
                response, {"model": reviewer.model, **body},
                wrapper_request=self.ctx.request, debug_dir=reviewer.debug_dir,
                verbose=reviewer.verbose, conversation=self.conv_path,
            ) or self.conv_path
        candidate = (response.get("candidates") or [{}])[0]
        content = candidate.get("content") or {}
        if not content.get("parts"):
            raise ProviderTurnFailed(
                f"generate_content returned no content: finish_reason="
                f"{candidate.get('finishReason')} block_reason="
                f"{response.get('promptFeedback', {}).get('blockReason')}")
        # Thought parts and signatures must be resent exactly as received
        # (https://ai.google.dev/gemini-api/docs/thinking#signatures).
        self.contents.append(content)
        return [ToolCall(p["functionCall"].get("id", ""), p["functionCall"]["name"],
                         p["functionCall"].get("args", {}))
                for p in content["parts"] if "functionCall" in p]

    def add_user_text(self, text: str) -> None:
        self.contents.append({"role": "user", "parts": [{"text": text}]})

    def add_tool_results(self, results: list[tuple[ToolCall, JsonObject, bool]]) -> None:
        self.contents.append({"role": "user", "parts": [
            {"functionResponse": {"name": call.name, "response": payload,
                                  **({"id": call.id} if call.id else {})}}
            for call, payload, _is_error in results
        ]})
