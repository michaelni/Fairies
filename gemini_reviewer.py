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
``Reviewer`` interface, driven by ``tool_loop``.

The model investigates via the ``shell`` function and returns its verdict
by calling a ``submit_review`` function whose parameters are the role's
output schema.

Importing this module pulls in the ``google-genai`` package, so only the
Gemini code path imports it (lazily, via the reviewer factory).
"""

from __future__ import annotations

import logging

from google import genai
from google.genai import types

import concurrency
from common import JsonObject, dump_response_debug_artifacts, load_api_key
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

# Rate limits (https://ai.google.dev/gemini-api/docs/rate-limits) answer
# 429 until the window passes; back off instead of failing the review.
_HTTP_OPTIONS = types.HttpOptions(
    timeout=900_000,
    retry_options=types.HttpRetryOptions(
        attempts=30, initial_delay=2.0, max_delay=60.0,
        http_status_codes=[429, 500, 502, 503, 504],
    ),
)


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

    def _client(self) -> genai.Client:
        api_key = load_api_key(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set (env or .env)")
        return genai.Client(api_key=api_key, http_options=_HTTP_OPTIONS)

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
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            tools=[types.Tool(function_declarations=[
                types.FunctionDeclaration(name=t["name"], description=t["description"],
                                          parameters_json_schema=t["input_schema"])
                for t in tools
            ])],
            thinking_config=types.ThinkingConfig(
                thinking_level=reviewer.effort,
                include_thoughts=reviewer.reasoning_summary in ("concise", "detailed"),
            ),
        )
        self.contents: list[types.Content] = [
            types.Content(role="user", parts=[types.Part(text=t) for t in texts]),
        ]
        self.conv_path: str | None = None

    def send(self) -> list[ToolCall]:
        reviewer = self.reviewer
        contents = list(self.contents)
        abort_if_cancelled()
        with concurrency.slot(reviewer.name.partition(":")[0]):
            response = self.client.models.generate_content(
                model=reviewer.model, contents=contents, config=self.config)
        usage = response.usage_metadata or types.GenerateContentResponseUsageMetadata()
        logger.info("generate_content usage prompt=%s cached=%s output=%s thoughts=%s",
                    usage.prompt_token_count, usage.cached_content_token_count,
                    usage.candidates_token_count, usage.thoughts_token_count)
        if reviewer.debug_dir:
            self.conv_path = dump_response_debug_artifacts(
                response.model_dump(mode="json", exclude_none=True),
                {"model": reviewer.model,
                 "contents": [c.model_dump(mode="json", exclude_none=True) for c in contents],
                 "config": self.config.model_dump(mode="json", exclude_none=True)},
                wrapper_request=self.ctx.request, debug_dir=reviewer.debug_dir,
                verbose=reviewer.verbose, conversation=self.conv_path,
            ) or self.conv_path
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None or not candidate.content.parts:
            raise ProviderTurnFailed(
                f"generate_content returned no content: finish_reason="
                f"{candidate.finish_reason if candidate else None} block_reason="
                f"{response.prompt_feedback.block_reason if response.prompt_feedback else None}")
        # Thought parts and signatures must be resent exactly as received
        # (https://ai.google.dev/gemini-api/docs/thinking#signatures).
        self.contents.append(candidate.content)
        return [ToolCall(p.function_call.id or "", p.function_call.name, p.function_call.args or {})
                for p in candidate.content.parts if p.function_call]

    def add_user_text(self, text: str) -> None:
        self.contents.append(types.Content(role="user", parts=[types.Part(text=text)]))

    def add_tool_results(self, results: list[tuple[ToolCall, JsonObject, bool]]) -> None:
        self.contents.append(types.Content(role="user", parts=[
            types.Part(function_response=types.FunctionResponse(
                id=call.id or None, name=call.name, response=payload))
            for call, payload, _is_error in results
        ]))
