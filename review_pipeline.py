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

Review pipeline wiring: build ``Reviewer``s from ``provider:model[@effort]``
specs and run the multi-model review (parallel drafts, then an optional
combiner) over one ``ReviewContext``.

What does NOT belong here: provider internals (``openai_reviewer``,
``anthropic_reviewer``), the review vocabulary and schema
(``llm_review_api``), prompt text (``llm_prompt``), and process
orchestration / CLI (the wrapper entrypoint).
"""

from __future__ import annotations

import argparse
import logging

from llm_review_api import (
    Z_AI_ANTHROPIC_URL,
    Review,
    ReviewContext,
    Reviewer,
    run_parallel,
)
from openai_reviewer import OpenAIResources, OpenAIReviewer

__all__ = [
    "make_reviewer",
    "review_pr",
]

logger = logging.getLogger(__name__)


def make_reviewer(
    spec: str,
    *,
    args: argparse.Namespace,
    resources: OpenAIResources | None,
    role: str,
    verbose: bool,
) -> Reviewer:
    """Build a ``Reviewer`` from a ``provider:model[@effort]`` spec.

    ``openai:<m>`` (or a bare ``<m>``) -> OpenAIReviewer reusing the shared
    OpenAI resources (``resources`` must not be None for this provider).
    ``anthropic:<m>`` -> AnthropicReviewer; ``zai:<m>`` -> AnthropicReviewer
    pointed at z.ai's Anthropic endpoint (GLM). The Anthropic module (and
    its SDK) is imported only when actually requested.

    ``@effort`` sets that reviewer's effort: an OpenAI reasoning effort
    (overriding --reasoning-effort for this pass), or an Anthropic/GLM
    thinking effort (``ANTHROPIC_EFFORTS``; no suffix keeps the provider
    default).
    """
    spec_body, sep, effort = spec.partition("@")
    if not sep:
        effort = None
    provider, sep, model = spec_body.partition(":")
    if not sep:
        provider, model = "openai", spec_body
    if not model:
        raise SystemExit(f"--model {spec!r}: missing model name after {provider!r}:")

    if provider == "openai":
        return OpenAIReviewer(args, resources, model=model, role=role, effort=effort)
    if provider in ("anthropic", "zai"):
        from anthropic_reviewer import AnthropicReviewer

        base_url = Z_AI_ANTHROPIC_URL if provider == "zai" else None
        api_key_env = "ZAI_API_KEY" if provider == "zai" else "ANTHROPIC_API_KEY"
        try:
            return AnthropicReviewer(
                model,
                name=f"{provider}:{model}",
                role=role,
                base_url=base_url,
                api_key_env=api_key_env,
                max_tool_rounds=args.podman_max_tool_rounds,
                exec_timeout_s=args.podman_exec_timeout,
                effort=effort,
                verbose=verbose,
                debug_dir=(
                    args.debug_response_dir
                    if resources is not None and resources.debug_dir_specified else None
                ),
            )
        except ValueError as exc:  # invalid @effort suffix
            raise SystemExit(f"--model {spec!r}: {exc}")
    raise SystemExit(f"--model {spec!r}: unknown provider {provider!r} (use openai/anthropic/zai)")


def review_pr(
    ctx: ReviewContext,
    model_reviewers: list[Reviewer],
    combiner: Reviewer | None,
) -> Review:
    """Run the model reviewers, then optionally the combiner, over ``ctx``.

    One model reviewer runs inline; several run concurrently (each gets its
    own shell via ``ctx.new_shell``) and reviewers that fail are dropped by
    ``run_parallel``. Their drafts accumulate on ``ctx`` so the combiner can
    verify and merge them; a single (configured or surviving) draft is
    returned as-is since there is nothing to merge. Without a combiner
    exactly one model reviewer is required.
    """
    if len(model_reviewers) == 1:
        drafts = [model_reviewers[0].review(ctx)]
    else:
        drafts = run_parallel(model_reviewers, ctx)
    ctx.drafts.extend(drafts)

    if combiner is None:
        if len(drafts) != 1:
            raise SystemExit("more than one --model requires --combine-model to merge them")
        return drafts[0]
    if len(drafts) == 1:
        logger.info("only one draft available; skipping the combine stage")
        return drafts[0]

    logger.info("combine stage: %s merging %d drafts", combiner.name, len(drafts))
    return combiner.review(ctx)
