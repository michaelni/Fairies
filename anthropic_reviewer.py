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

AnthropicReviewer: one Anthropic Messages-API review pass behind the shared
``Reviewer`` interface. GLM is the same reviewer pointed at z.ai's
Anthropic-compatible endpoint (``base_url`` + ``ZAI_API_KEY``).

Structured output uses Anthropic's tool-use idiom: the model investigates
via the ``shell`` tool (per-machine ``ctx.open_shell`` sessions) and
returns its verdict by calling a ``submit_review`` tool whose
``input_schema`` is the role's output schema.

Importing this module pulls in the ``anthropic`` package, so only the
Anthropic / GLM code path imports it (lazily, via the reviewer factory).
"""

from __future__ import annotations

import json

from anthropic import Anthropic

import concurrency
from common import JsonObject, dump_response_debug_artifacts, load_api_key
from llm_prompt import REVIEWER_ROLE
from llm_review_api import ReviewContext, Reviewer, RoleSpec
from anthropic_common import call_with_anthropic_retry
from shell_tool import abort_if_cancelled
from tool_loop import Conversation, ToolCall, review_prompt, review_tools, run_tool_loop

__all__ = [
    "ANTHROPIC_EFFORTS",
    "DEFAULT_ANTHROPIC_MAX_TOKENS",
    "AnthropicReviewer",
]

# Conservative output-token budget that every current Anthropic / GLM model
# accepts; raise per-model via the constructor when a model allows more.
DEFAULT_ANTHROPIC_MAX_TOKENS = 16_000

# Valid ``effort`` values. ``off`` disables thinking explicitly; the named
# levels enable adaptive thinking with ``output_config.effort`` -- the same
# dialect current Claude models use (they reject manual budget_tokens) and
# which z.ai's Anthropic endpoint honors for GLM (probed 2026-07-03:
# thinking volume scales low < high < max). No effort sends no ``thinking``
# at all, keeping the provider default (on z.ai: no thinking).
ANTHROPIC_EFFORTS = ("off", "low", "medium", "high", "xhigh", "max")

# Per-request timeout, far above the longest single call observed (135s).
# Also opts out of the SDK's static pre-flight check that rejects
# non-streaming requests whose max_tokens COULD take >10 min to generate
# (raised for max_tokens > 21333; killed every GLM draft on 2026-07-03 when
# a thinking budget grew max_tokens past it). Our tool-round responses stay
# far below max_tokens, so the pessimistic estimate does not apply.
ANTHROPIC_TIMEOUT_S = 900.0

class AnthropicReviewer(Reviewer):
    """One Anthropic Messages-API pass of a ``RoleSpec`` behind the shared
    interface.

    ``run(ctx)`` builds the system/user prompt from the role, runs the
    Messages tool loop (driving per-machine ``ctx.open_shell`` sessions
    when available), and returns the result the model submits via the
    ``submit_review`` tool, validated by the role. Raises
    ``BadModelOutput`` when the model never produces a schema-valid
    verdict.

    ``base_url`` + ``api_key_env`` select the backend: defaults reach
    Anthropic; pass z.ai's Anthropic endpoint + ``ZAI_API_KEY`` for GLM.
    ``name`` is the stable label recorded on the ``Review`` (e.g.
    ``"anthropic:claude-opus-4"`` or ``"zai:glm-5.3"``).

    ``effort`` is an ``ANTHROPIC_EFFORTS`` name controlling extended
    thinking; ``None`` (default) sends no ``thinking`` parameter so the
    provider default applies.

    ``reasoning_summary`` takes --reasoning-summary's vocabulary
    (``auto``/``concise``/``detailed``): ``concise`` and ``detailed``
    request ``thinking.display=summarized`` (Anthropic's only summary
    level) when a named effort enables thinking; ``auto``/``None`` sends
    no ``display`` so the provider default applies.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        base_url: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        max_tool_rounds: int = 0,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
        reasoning_summary: str | None = None,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if effort is not None and effort not in ANTHROPIC_EFFORTS:
            raise ValueError(
                f"effort {effort!r} not in {ANTHROPIC_EFFORTS}"
            )
        self.model = model
        self.name = name
        self.role = role
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.max_tokens = max_tokens
        self.max_tool_rounds = max_tool_rounds
        self.exec_timeout_s = exec_timeout_s
        self.effort = effort
        self.reasoning_summary = reasoning_summary
        self.verbose = verbose
        self.debug_dir = debug_dir

    def _client(self) -> Anthropic:
        api_key = load_api_key(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"{self.api_key_env} is not set (env or .env)")
        kwargs: dict[str, object] = {
            "api_key": api_key, "max_retries": 0, "timeout": ANTHROPIC_TIMEOUT_S,
        }
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return Anthropic(**kwargs)

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        system, texts = review_prompt(self.role, ctx, vendor="anthropic", model=self.model)
        conversation = _Conversation(self, ctx, system, texts, review_tools(self.role, ctx))
        return run_tool_loop(conversation, self.role, ctx, what="messages.create",
                             max_tool_rounds=self.max_tool_rounds,
                             exec_timeout_s=self.exec_timeout_s)


class _Conversation(Conversation):
    def __init__(self, reviewer: AnthropicReviewer, ctx: ReviewContext, system: str,
                 texts: list[str], tools: list[JsonObject]) -> None:
        self.reviewer = reviewer
        self.client = reviewer._client()
        self.ctx = ctx
        # Anthropic caches only up to explicit cache_control breakpoints;
        # z.ai caches implicitly and ignores them.
        self.system: list[JsonObject] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
        ]
        self.tools = tools
        self.messages: list[JsonObject] = [
            {"role": "user", "content": [{"type": "text", "text": t} for t in texts]},
        ]
        self.conv_path: str | None = None

    def send(self) -> list[ToolCall]:
        reviewer = self.reviewer
        _mark_cache_breakpoint(self.messages)
        # Snapshot: ``messages`` grows across rounds and the dump
        # must record what this round actually sent.
        request_kwargs: JsonObject = {
            "model": reviewer.model,
            "system": self.system,
            "messages": list(self.messages),
            "tools": self.tools,
            "max_tokens": reviewer.max_tokens,
        }
        if reviewer.effort == "off":
            request_kwargs["thinking"] = {"type": "disabled"}
        elif reviewer.effort is not None:
            thinking: JsonObject = {"type": "adaptive"}
            # claude-opus-5 returns a thinking summary only with
            # display=summarized (probed 2026-08-09); z.ai glm-5.3
            # accepts it and still returns thinking (probed
            # 2026-08-15), so no per-backend branch.
            if reviewer.reasoning_summary in ("concise", "detailed"):
                thinking["display"] = "summarized"
            request_kwargs["thinking"] = thinking
            request_kwargs["output_config"] = {"effort": reviewer.effort}
        abort_if_cancelled()
        with concurrency.slot(reviewer.name.partition(":")[0]):
            response = call_with_anthropic_retry(
                lambda: self.client.messages.create(**request_kwargs),
                what="messages.create",
                verbose=reviewer.verbose,
            )
        if reviewer.debug_dir:
            self.conv_path = dump_response_debug_artifacts(
                response, request_kwargs, wrapper_request=self.ctx.request,
                debug_dir=reviewer.debug_dir, verbose=reviewer.verbose,
                conversation=self.conv_path,
            ) or self.conv_path
        self.messages.append({"role": "assistant", "content": _echo_content(response.content)})
        return [ToolCall(b.id, b.name, b.input) for b in response.content
                if getattr(b, "type", None) == "tool_use"]

    def add_user_text(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[tuple[ToolCall, JsonObject, bool]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id,
             "content": json.dumps(payload, ensure_ascii=False),
             **({"is_error": True} if is_error else {})}
            for call, payload, is_error in results
        ]})


def _echo_content(content: list[object]) -> list[JsonObject]:
    """Rebuild an assistant turn's content blocks as plain dicts to send back.

    The Messages API requires the prior assistant turn (including its
    ``tool_use`` blocks) to be replayed before the matching ``tool_result``s.
    Reconstructing dicts from the response blocks (rather than echoing SDK
    objects) keeps the loop independent of the SDK's input/output type
    interchangeability and trivially testable with lightweight fakes.
    """
    blocks: list[JsonObject] = []
    for block in content:
        kind = getattr(block, "type", None)
        if kind == "text":
            blocks.append({"type": "text", "text": block.text})
        elif kind == "tool_use":
            blocks.append({
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
        # With extended thinking enabled the API rejects a tool-result
        # turn unless the assistant's thinking blocks are replayed
        # unmodified (signature included); see the Messages API extended
        # thinking documentation ("Preserving thinking blocks").
        elif kind == "thinking":
            blocks.append({
                "type": "thinking",
                "thinking": block.thinking,
                "signature": block.signature,
            })
        elif kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": block.data})
    return blocks


def _mark_cache_breakpoint(messages: list[JsonObject]) -> None:
    """Move the conversation's cache_control marker to the newest block."""
    last: JsonObject | None = None
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict):
                block.pop("cache_control", None)
                last = block
    if last is not None:
        last["cache_control"] = {"type": "ephemeral"}
