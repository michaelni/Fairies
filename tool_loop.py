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

The submit_review / shell tool loop shared by the reviewers that drive a
chat API one turn at a time (Anthropic / GLM, Gemini).

What belongs here: the provider-neutral loop -- prompt assembly, shell
dispatch, the submit_review nudge and the tool-round cap -- and the
``Conversation`` interface each provider implements to plug into it.

What does NOT belong: any provider's request/response encoding, retries
or client setup (those stay in the ``*_reviewer`` modules).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import podman_host
from common import JsonObject
from llm_prompt import generate_llm_prompt
from llm_review_api import BadModelOutput, ReviewContext, RoleSpec
from shell_bridge_client import SHELL_TOOL_NAME
from shell_tool import build_shell_tool_schema, exec_machine_call

__all__ = [
    "SUBMIT_REVIEW",
    "Conversation",
    "ToolCall",
    "review_prompt",
    "review_tools",
    "run_tool_loop",
]

logger = logging.getLogger(__name__)

SUBMIT_REVIEW = "submit_review"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: JsonObject


class Conversation(ABC):
    """One review pass's chat history on one provider."""

    @abstractmethod
    def send(self) -> list[ToolCall]:
        """Send the history, append the model's reply to it and return the
        tool calls in that reply."""

    @abstractmethod
    def add_user_text(self, text: str) -> None:
        """Append a user turn holding ``text``."""

    @abstractmethod
    def add_tool_results(self, results: list[tuple[ToolCall, JsonObject, bool]]) -> None:
        """Append the turn answering the last reply's tool calls; each
        entry is ``(call, payload, is_error)``."""


def review_prompt(role: RoleSpec, ctx: ReviewContext, *, vendor: str,
                  model: str) -> tuple[str, list[str]]:
    """The system prompt and the ordered user texts of one review pass."""
    features: set[str] = set()
    if ctx.source_bundle is not None:
        features.add("source_bundle")
    if ctx.open_shell is not None:
        features.add("podman_shell")
    system = generate_llm_prompt(
        role=role.name,
        vendor=vendor,
        model=model,
        features=features,
        repo_roots=ctx.repo_roots,
        container_repo_mounts=ctx.repo_mount_paths,
        machines=ctx.machines,
        reviewer_username=ctx.reviewer_username,
        project_facts=ctx.project_facts,
        ci_triage_mode=ctx.ci_triage_mode,
        **role.prompt_kwargs,
    )
    user_texts = role.user_texts(ctx)
    texts = [user_texts[0]]
    if ctx.patch_text:  # empty for the issue-investigator task (no patch)
        texts.append(ctx.patch_text)
    if ctx.source_bundle is not None:
        texts.append(ctx.source_bundle)
    texts += user_texts[1:]
    texts.append("Return your final verdict by calling the submit_review tool. "
                 "Do not put the review in plain text.")
    return system, texts


def review_tools(role: RoleSpec, ctx: ReviewContext) -> list[JsonObject]:
    """The pass's tools as ``{"name", "description", "input_schema"}``:
    submit_review, plus the shell tool when ``ctx`` has a shell."""
    tools: list[JsonObject] = [{
        "name": SUBMIT_REVIEW,
        "description": (
            "Return your final verdict. Call this exactly once, when "
            "finished, filling every schema field."
        ),
        "input_schema": role.schema["schema"],
    }]
    if ctx.open_shell is not None:
        tools.append(build_shell_tool_schema([m.label for m in ctx.machines]))
    return tools


def run_tool_loop(conversation: Conversation, role: RoleSpec, ctx: ReviewContext,
                  *, what: str, max_tool_rounds: int,
                  exec_timeout_s: float) -> dict[str, object]:
    """Run ``conversation`` until the model calls submit_review and return
    the role-validated verdict.

    Shell calls run on per-machine ``ctx.open_shell`` sessions. A reply
    without tool calls is nudged once, a second one raises
    ``BadModelOutput``. More than ``max_tool_rounds`` tool rounds
    (0: unlimited) raise ``RuntimeError``; ``what`` names the API call in
    logs and errors.
    """
    shells: dict[str, podman_host.ContainerShellSession] = {}
    labels = tuple(m.label for m in ctx.machines)
    nudged = False
    rounds = 0
    while True:
        logger.info("%s role=%s round=%d shell=%s",
                    what, role.name, rounds, ctx.open_shell is not None)
        calls = conversation.send()
        submit = next((c for c in calls if c.name == SUBMIT_REVIEW), None)
        if submit is not None:
            result = role.validate(submit.input)
            ctx.collect_into(result, shells.values())
            verdict = result.get("classification") or result.get("route") or "-"
            logger.debug("%s %s verdict=%s", what, role.name, verdict)
            return result

        if not calls:
            if nudged:
                raise BadModelOutput()
            nudged = True
            conversation.add_user_text(
                "You did not call submit_review. Return the verdict now via submit_review.")
            continue

        rounds += 1
        if max_tool_rounds > 0 and rounds > max_tool_rounds:
            raise RuntimeError(f"{what}: exceeded shell tool-call limit ({max_tool_rounds})")

        results: list[tuple[ToolCall, JsonObject, bool]] = []
        for call in calls:
            if call.name == SHELL_TOOL_NAME and ctx.open_shell is not None:
                payload = exec_machine_call(shells, labels, ctx.open_shell, call.input,
                                            max_timeout_s=exec_timeout_s)
                results.append((call, payload, False))
            else:
                results.append((call, {"error": f"unsupported tool {call.name!r}"}, True))
        conversation.add_tool_results(results)
