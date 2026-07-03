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
via the ``shell`` tool (a ``ctx.new_shell()`` session) and returns its
verdict by calling a ``submit_review`` tool whose ``input_schema`` is the
shared ``REVIEW_SCHEMA``.

Importing this module pulls in the ``anthropic`` package, so only the
Anthropic / GLM code path imports it (lazily, via the reviewer factory).
"""

from __future__ import annotations

import json
import logging

from anthropic import Anthropic

from common import JsonObject, dump_response_debug_artifacts
from llm_prompt import generate_llm_prompt, make_combiner_user_text, make_user_text
from llm_review_api import (
    REVIEW_SCHEMA,
    BadModelOutput,
    Review,
    ReviewContext,
    Reviewer,
    validate_review,
)
from anthropic_common import call_with_anthropic_retry, load_api_key
from shell_tool import exec_shell_call

__all__ = [
    "ANTHROPIC_EFFORTS",
    "DEFAULT_ANTHROPIC_MAX_TOKENS",
    "AnthropicReviewer",
]

logger = logging.getLogger(__name__)

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

_SUBMIT_REVIEW = "submit_review"
_SHELL = "shell"


def _build_submit_review_tool() -> JsonObject:
    return {
        "name": _SUBMIT_REVIEW,
        "description": (
            "Return your final pull-request review verdict. Call this exactly "
            "once, when finished, with the classification and message."
        ),
        "input_schema": REVIEW_SCHEMA["schema"],
    }


def _build_shell_tool() -> JsonObject:
    return {
        "name": _SHELL,
        "description": (
            "Run one shell command inside the ephemeral review container "
            "(full working-tree repos under /work/...). Command runs via "
            "``sh -c`` with an in-container timeout."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command (``sh -c``)."},
                "cwd": {"type": "string", "description": "Working directory (e.g. /work/ffmpeg)."},
                "timeout_seconds": {"type": "number", "description": "Max wall seconds (capped by the wrapper)."},
            },
            "required": ["command"],
        },
    }


class AnthropicReviewer(Reviewer):
    """One Anthropic Messages-API review pass behind the shared interface.

    ``review(ctx)`` builds the system/user prompt, runs the Messages tool
    loop (driving a ``ctx.new_shell()`` session when one is available), and
    returns the verdict the model submits via the ``submit_review`` tool,
    validated against ``REVIEW_SCHEMA``. Raises ``BadModelOutput`` when the
    model never produces a schema-valid verdict.

    ``base_url`` + ``api_key_env`` select the backend: defaults reach
    Anthropic; pass z.ai's Anthropic endpoint + ``ZAI_API_KEY`` for GLM.
    ``name`` is the stable label recorded on the ``Review`` (e.g.
    ``"anthropic:claude-opus-4"`` or ``"zai:glm-4.6"``).

    ``effort`` is an ``ANTHROPIC_EFFORTS`` name controlling extended
    thinking; ``None`` (default) sends no ``thinking`` parameter so the
    provider default applies.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: str = "reviewer",
        base_url: str | None = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_tokens: int = DEFAULT_ANTHROPIC_MAX_TOKENS,
        max_tool_rounds: int = 0,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
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

    def review(self, ctx: ReviewContext) -> Review:
        client = self._client()

        features: set[str] = set()
        if ctx.source_bundle is not None:
            features.add("source_bundle")
        use_shell = ctx.new_shell is not None
        if use_shell:
            features.add("podman_shell")

        system = generate_llm_prompt(
            role=self.role,
            vendor="anthropic",
            model=self.model,
            features=features,
            repo_roots=ctx.repo_roots,
            container_repo_mounts=ctx.repo_mount_paths,
            reviewer_username=ctx.reviewer_username,
            ci_triage_mode=ctx.ci_triage_mode,
        )

        # Anthropic has no file-upload primitive, so the patch and source
        # bundle are inlined as text blocks (OpenAI uploads them as files).
        user_blocks: list[JsonObject] = [
            {"type": "text", "text": make_user_text(
                ctx.request, ctx.source_notes, ctx.source_files, ctx.patch_truncated)},
            {"type": "text", "text": ctx.patch_text},
        ]
        if ctx.source_bundle is not None:
            user_blocks.append({"type": "text", "text": ctx.source_bundle})
        if self.role == "combiner":
            user_blocks.append({"type": "text", "text": make_combiner_user_text(ctx.review_drafts())})
        user_blocks.append({
            "type": "text",
            "text": "Return your final verdict by calling the submit_review tool. "
                    "Do not put the review in plain text.",
        })

        messages: list[JsonObject] = [{"role": "user", "content": user_blocks}]
        tools: list[JsonObject] = [_build_submit_review_tool()]
        if use_shell:
            tools.append(_build_shell_tool())

        shell = ctx.new_shell() if use_shell else None
        nudged = False
        rounds = 0
        conv_path: str | None = None
        try:
            while True:
                logger.info(
                    "anthropic messages.create model=%s round=%d shell=%s",
                    self.model, rounds, use_shell,
                )
                # Snapshot: ``messages`` grows across rounds and the dump
                # must record what this round actually sent.
                request_kwargs: JsonObject = {
                    "model": self.model,
                    "system": system,
                    "messages": list(messages),
                    "tools": tools,
                    "max_tokens": self.max_tokens,
                }
                if self.effort == "off":
                    request_kwargs["thinking"] = {"type": "disabled"}
                elif self.effort is not None:
                    request_kwargs["thinking"] = {"type": "adaptive"}
                    request_kwargs["output_config"] = {"effort": self.effort}
                response = call_with_anthropic_retry(
                    lambda: client.messages.create(**request_kwargs),
                    what="messages.create",
                    verbose=self.verbose,
                )
                if self.debug_dir:
                    conv_path = dump_response_debug_artifacts(
                        response, request_kwargs, wrapper_request=ctx.request,
                        debug_dir=self.debug_dir, verbose=self.verbose,
                        conversation=conv_path,
                    ) or conv_path

                tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]
                submit = next((b for b in tool_uses if b.name == _SUBMIT_REVIEW), None)
                if submit is not None:
                    result = validate_review(submit.input)
                    if self.verbose:
                        logger.debug("anthropic verdict classification=%s", result["classification"])
                    return Review(
                        classification=result["classification"],
                        message=result["message"],
                        model=self.name,
                    )

                if not tool_uses:
                    if nudged:
                        raise BadModelOutput()
                    nudged = True
                    messages.append({"role": "assistant", "content": _echo_content(response.content)})
                    messages.append({
                        "role": "user",
                        "content": "You did not call submit_review. Return the verdict now via submit_review.",
                    })
                    continue

                rounds += 1
                if self.max_tool_rounds > 0 and rounds > self.max_tool_rounds:
                    raise RuntimeError(
                        f"messages.create: exceeded shell tool-call limit ({self.max_tool_rounds})"
                    )

                messages.append({"role": "assistant", "content": _echo_content(response.content)})
                results: list[JsonObject] = []
                for use in tool_uses:
                    if use.name == _SHELL and shell is not None:
                        payload = exec_shell_call(shell, use.input, max_timeout_s=self.exec_timeout_s)
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": use.id,
                            "content": json.dumps(payload, ensure_ascii=False),
                        })
                    else:
                        results.append({
                            "type": "tool_result",
                            "tool_use_id": use.id,
                            "content": json.dumps({"error": f"unsupported tool {use.name!r}"}),
                            "is_error": True,
                        })
                messages.append({"role": "user", "content": results})
        finally:
            if shell is not None:
                shell.close()


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
