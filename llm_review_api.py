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

The one reviewer interface the OpenAI / Anthropic / GLM reviewers and the
review pipeline all speak.

What belongs here: the provider-neutral review vocabulary (``Review``,
``ReviewContext``, the ``Reviewer`` base class), the classification
constants, and ``run_parallel`` (the single fan-out helper).

What does NOT belong: any provider/SDK-specific code (that lives in the
``*_reviewer`` modules), prompt text (``llm_prompt``), or pipeline wiring
(``review_pipeline``). This is a leaf module: it must not import any of
those.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from common import JsonObject
from podman_host import ContainerShellSession

__all__ = [
    "CLASSIFICATIONS",
    "ISSUE_CLASSIFICATIONS",
    "TERMINAL_ROUTES",
    "ENGAGE",
    "Z_AI_ANTHROPIC_URL",
    "Review",
    "ReviewContext",
    "Reviewer",
    "run_parallel",
]

logger = logging.getLogger(__name__)


# The full classification vocabulary a reviewer may emit. Single source of
# truth: the OpenAI/Anthropic output schemas and ``Review`` all read it.
CLASSIFICATIONS = (
    "ok_approve",
    "minor_issues_approve",
    "moderate_issues_comment",
    "major_request_changes",
    "helpful_reply",
    "skip",
)

# Classifications that mean "this reviewer found something worth raising".
ISSUE_CLASSIFICATIONS = ("moderate_issues_comment", "major_request_changes")

# Verdicts that are already final: when a stage returns one of these the
# pipeline stops and posts it as-is (no further models, no combine).
TERMINAL_ROUTES = ("skip", "helpful_reply")

# Non-emittable "keep going" marker a triager returns when it routes
# ``engage``. Never a valid posted classification and never reaches
# stdout; the pipeline treats it purely as "continue to the next stage".
ENGAGE = "engage"

# z.ai's Anthropic-compatible Messages endpoint. GLM is reached by pointing
# the Anthropic reviewer at this base URL with a z.ai key.
Z_AI_ANTHROPIC_URL = "https://api.z.ai/api/anthropic"


@dataclass(frozen=True)
class Review:
    """One reviewer's verdict.

    ``classification`` is a ``CLASSIFICATIONS`` member, or ``ENGAGE`` for a
    triager that wants the pipeline to continue. ``model`` records which
    reviewer produced it, for the combine stage and debug logs.
    """

    classification: str
    message: str = ""
    label_changes: tuple[JsonObject, ...] = ()
    model: str = ""

    @property
    def terminal(self) -> bool:
        return self.classification in TERMINAL_ROUTES

    @property
    def found_issue(self) -> bool:
        return self.classification in ISSUE_CLASSIFICATIONS


@dataclass
class ReviewContext:
    """The PR under review plus the drafts the pipeline accumulates.

    ``new_shell`` returns a fresh, isolated container shell (its own
    working trees) on each call; a reviewer that uses a shell owns it and
    closes it. ``None`` means no podman shell is configured (the OpenAI
    solo path uses its native tools instead). The pipeline -- not the
    reviewers -- appends to ``drafts``.
    """

    request: JsonObject
    patch_text: str
    patch_truncated: bool
    source_bundle: str | None
    source_files: list[str]
    source_notes: list[str]
    reviewer_username: str
    ci_triage_mode: bool
    repo_roots: list[Path]
    repo_mount_paths: list[str]
    new_shell: Callable[[], ContainerShellSession] | None = None
    drafts: list[Review] = field(default_factory=list)

    def review_drafts(self) -> list[Review]:
        """Prior drafts that are real review verdicts (excluding triage
        ``ENGAGE``), in the order produced -- what the combine stage merges."""
        return [d for d in self.drafts if d.classification in CLASSIFICATIONS]


class Reviewer(ABC):
    """The interface every model backend and the pipeline share.

    ``name`` is a short stable label (e.g. ``"openai:gpt-5.4"``) used in
    logs and recorded on the produced ``Review``.
    """

    name: str

    @abstractmethod
    def review(self, ctx: ReviewContext) -> Review: ...


def run_parallel(reviewers: list[Reviewer], ctx: ReviewContext) -> list[Review]:
    """Run each reviewer concurrently; return their drafts in input order.

    Each reviewer obtains its own shell via ``ctx.new_shell`` so concurrent
    runs never share a container working tree. Results are gathered only
    after every thread finishes, so the shared ``ctx`` is never mutated
    concurrently; the caller extends ``ctx.drafts`` with the returned list.
    """
    if not reviewers:
        return []
    logger.info(
        "run_parallel: launching %d reviewers: %s",
        len(reviewers),
        ", ".join(r.name for r in reviewers),
    )
    with ThreadPoolExecutor(max_workers=len(reviewers)) as executor:
        return list(executor.map(lambda r: r.review(ctx), reviewers))
