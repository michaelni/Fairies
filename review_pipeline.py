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
from typing import Callable

from llm_review_api import (
    Z_AI_ANTHROPIC_URL,
    BadModelOutput,
    ProviderTurnFailed,
    Review,
    ReviewContext,
    Reviewer,
    RoleSpec,
    run_parallel,
)
from openai_reviewer import (
    INHERIT_SERVICE_TIER,
    OpenAIContainerUnhealthy,
    OpenAIResources,
    OpenAIReviewer,
)

__all__ = [
    "make_reviewer",
    "review_pr",
    "run_triage",
]

logger = logging.getLogger(__name__)


def make_reviewer(
    spec: str,
    *,
    args: argparse.Namespace,
    resources: OpenAIResources | None,
    role: RoleSpec,
    verbose: bool,
    default_effort: str | None = None,
    max_output_tokens: int | None = None,
    service_tier: str | None = INHERIT_SERVICE_TIER,
) -> Reviewer:
    """Build a ``Reviewer`` from a ``provider:model[@effort]`` spec.

    ``openai:<m>`` -> OpenAIReviewer reusing the shared
    OpenAI resources (``resources`` must not be None for this provider).
    ``anthropic:<m>`` -> AnthropicReviewer; ``zai:<m>`` -> AnthropicReviewer
    pointed at z.ai's Anthropic endpoint (GLM). ``codex:<m>`` ->
    CodexReviewer driving the pinned codex CLI. Provider modules (and
    their SDKs) are imported only when actually requested.

    ``@effort`` sets that reviewer's effort: an OpenAI reasoning effort,
    an Anthropic/GLM thinking effort (``ANTHROPIC_EFFORTS``), or a codex
    ``model_reasoning_effort`` (``CODEX_EFFORTS``). When the spec has no
    ``@effort``, ``default_effort`` applies (``None`` keeps the provider
    default). This is the only way to set effort -- there is no shared
    main-pass effort flag; efforts live on the model specs.

    ``service_tier`` (OpenAI only) is used verbatim; ``None`` sends no
    tier. Leaving it unset inherits ``--service-tier``.
    """
    spec_body, sep, spec_effort = spec.partition("@")
    effort = spec_effort if sep else default_effort
    provider, sep, model = spec_body.partition(":")
    if not sep:
        raise SystemExit(
            f"--model {spec!r}: missing provider prefix (use openai:/anthropic:/zai:/codex:)"
        )
    if not model:
        raise SystemExit(f"--model {spec!r}: missing model name after {provider!r}:")

    if provider == "openai":
        return OpenAIReviewer(
            args, resources, model=model, role=role, effort=effort,
            max_output_tokens=max_output_tokens, service_tier=service_tier,
        )
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
    if provider == "codex":
        from codex_reviewer import CodexReviewer, resolve_web_search

        if getattr(args, "codex_host", None) is None:
            raise SystemExit(
                f"--model {spec!r}: codex requires --codex-host "
                "(codex runs only in a container on that host, never locally)"
            )
        try:
            return CodexReviewer(
                model,
                name=f"codex:{model}",
                role=role,
                codex_bin=args.codex_bin,
                codex_home=args.codex_home,
                codex_host=args.codex_host,
                codex_image=args.codex_image,
                exec_timeout_s=args.podman_exec_timeout,
                effort=effort,
                web_search=resolve_web_search(args.web_search),
                verbosity=args.verbosity,
                reasoning_summary=args.reasoning_summary,
                run_timeout_s=args.codex_timeout_seconds,
                verbose=verbose,
                debug_dir=(
                    args.debug_response_dir
                    if resources is not None and resources.debug_dir_specified else None
                ),
            )
        except ValueError as exc:  # invalid @effort suffix
            raise SystemExit(f"--model {spec!r}: {exc}")
    raise SystemExit(f"--model {spec!r}: unknown provider {provider!r} (use openai/anthropic/zai/codex)")


def run_triage(triager: Reviewer, ctx: ReviewContext) -> dict[str, object] | None:
    """Run the triager role over ``ctx``.

    Returns the validated triage dict on success. Returns ``None`` on
    failure (API error, bad output, refusal) so the caller falls through
    to the main reviewer pass. When the OpenAI container dies during
    triage, returns ``container_unhealthy=True`` so the caller can exit
    ``EXIT_CONTAINER_UNHEALTHY``.
    """
    try:
        result = triager.run(ctx)
    except OpenAIContainerUnhealthy:
        logger.warning("openai container unhealthy during triage")
        return {"route": "engage", "message": "", "reason": "", "container_unhealthy": True}
    except BadModelOutput as exc:
        logger.warning(
            "triage stage output did not match the requested schema (%s); "
            "falling through to main reviewer pass",
            str(exc).replace("\n", " "),
        )
        return None
    except Exception as exc:
        logger.warning(
            "triage stage failed with %s: %s; will fall through to main reviewer pass",
            type(exc).__name__,
            str(exc).replace("\n", " "),
        )
        return None

    logger.info(
        "triage decision route=%s message_chars=%d reason=%r",
        result.get("route"),
        len(str(result.get("message") or "")),
        result.get("reason", ""),
    )
    return result


def review_pr(
    ctx: ReviewContext,
    model_reviewers: list[Reviewer],
    combiner: Reviewer | None,
    on_drafts: Callable[[list[Review]], None] | None = None,
) -> Review:
    """Run the model reviewers, then optionally the combiner, over ``ctx``.

    One model reviewer runs inline; several run concurrently (each opens its
    own shells via ``ctx.open_shell``) and reviewers that fail are dropped by
    ``run_parallel``. Their drafts accumulate on ``ctx`` so the combiner can
    verify and merge them. ``on_drafts`` is told the surviving drafts before
    the combiner runs. A configured combiner runs even on a single
    (configured or surviving) draft: since its prompt diverged from the
    reviewer's, its verification and grading are no longer redundant.
    Without a combiner exactly one model reviewer is required.
    A provider-ended combiner turn is retried once; failing again it
    propagates, costing the caller's full-pipeline retry as before.
    """
    if len(model_reviewers) == 1:
        drafts = [model_reviewers[0].review(ctx)]
    else:
        drafts = run_parallel(model_reviewers, ctx)
    ctx.drafts.extend(drafts)
    if on_drafts is not None:
        on_drafts(drafts)

    if combiner is None:
        if len(drafts) != 1:
            raise SystemExit(
                "several model reviewers require --combine-model to merge their drafts")
        return drafts[0]

    logger.info("combine stage: %s merging %d draft(s)", combiner.name, len(drafts))
    try:
        return combiner.review(ctx)
    except ProviderTurnFailed as exc:
        logger.warning(
            "combiner %s: provider ended the turn (%s); retrying once",
            combiner.name, (str(exc).splitlines() or ["-"])[0][:160],
        )
        return combiner.review(ctx)
