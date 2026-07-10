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

Replay ONLY the combine stage of a past ensemble run, with the draft
labels or order optionally altered, to probe combiner bias.

Inputs are recovered from --debug-response-dir dumps: the original
wrapper request JSON and the two draft reviews. Everything else (user
text, source/patch bundles, developer prompt, tools, a fresh podman
container) is rebuilt through the wrapper's own code path, so the replay
differs from the original combine call only in the drafts presented.

Usage:
  tools/replay_combiner.py --request request.json --drafts drafts.json \
      --variant control|swap-labels|swap-order \
      -- <pr_review_wrapper.py flags of the original run>

``drafts.json`` maps arbitrary keys to ``{"model", "classification",
"message"}``; presentation order is file order. ``swap-labels`` keeps
the order but exchanges the model names; ``swap-order`` reverses the
order and keeps the correct names. The final combined review JSON is
printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import setup_logging  # noqa: E402
import llm_prompt  # noqa: E402
from llm_review_api import Review, ReviewContext  # noqa: E402
from openai import OpenAI  # noqa: E402
from openai_common import load_api_key, upload_text_file, delete_uploaded_file  # noqa: E402
import pr_review_wrapper as wrapper  # noqa: E402
import openai_reviewer  # noqa: E402
import review_pipeline  # noqa: E402
import podman_host  # noqa: E402
import podman_repos  # noqa: E402

logger = logging.getLogger(__name__)

VARIANTS = ("control", "swap-labels", "swap-order")


def build_drafts(drafts_path: Path, variant: str) -> list[Review]:
    raw = list(json.loads(drafts_path.read_text()).values())
    order = list(reversed(raw)) if variant == "swap-order" else raw
    labels = [d["model"] for d in order]
    if variant == "swap-labels":
        labels = labels[::-1]
    drafts = [
        Review(classification=d["classification"], message=d["message"], model=label)
        for d, label in zip(order, labels)
    ]
    for d in drafts:
        logger.info("draft as presented: model=%s classification=%s", d.model, d.classification)
    return drafts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1])
    ap.add_argument("--request", type=Path, required=True,
                    help="wrapper_request JSON recovered from a debug dump")
    ap.add_argument("--drafts", type=Path, required=True,
                    help="draft reviews JSON (see module docstring)")
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- followed by the original wrapper flags")
    tool_args = ap.parse_args()

    rest = tool_args.rest
    if rest and rest[0] == "--":
        rest = rest[1:]
    sys.argv = ["pr_review_wrapper.py", *rest]
    args = wrapper.parse_args()
    if not args.podman or not args.combine_model:
        raise SystemExit("replay requires the original run's --podman and --combine-model flags")
    debug_dir_specified = any(
        a == "--debug-response-dir" or a.startswith("--debug-response-dir=") for a in rest
    )

    setup_logging(logger, args.verbose, wrapper.logger,
                  podman_host.logger, podman_repos.logger, color=args.color)

    request = json.loads(tool_args.request.read_text())
    drafts = build_drafts(tool_args.drafts, tool_args.variant)

    client = OpenAI(api_key=load_api_key(), timeout=None, max_retries=0,
                    http_client=openai_reviewer.make_openai_http_client())
    repo_root = wrapper.find_repo_root(args.repo_root)
    repo_roots = wrapper.get_all_repo_roots(repo_root, args.extra_repo_root)
    remote_host = wrapper._build_remote_host(args)
    repo_specs = podman_repos.build_repo_specs(repo_roots, mirror_root=args.podman_mirror_root)

    patch = request.get("patch") if isinstance(request.get("patch"), str) else ""
    source_bundle, source_files, source_notes = wrapper.build_source_bundle(
        request, repo_root,
        max_source_files=args.max_source_files,
        max_file_bytes=args.max_file_bytes,
        max_header_file_bytes=args.max_header_file_bytes,
        max_bundle_bytes=args.max_bundle_bytes,
        include_direct_includes=args.include_direct_includes,
        verbose=args.verbose,
    )
    patch_bundle, patch_was_truncated = wrapper.build_patch_bundle(patch, args.max_patch_bytes)

    handle, session = wrapper.open_review_container_shell(remote_host, repo_specs, args)
    uploaded_file_ids: list[str] = []
    try:
        patch_file_id = upload_text_file(
            client, filename="pull_request.patch.txt", text=patch_bundle, verbose=args.verbose,
        )
        uploaded_file_ids.append(patch_file_id)

        ctx = ReviewContext(
            request=request,
            patch_text=patch_bundle,
            patch_truncated=patch_was_truncated,
            source_bundle=source_bundle,
            source_files=source_files,
            source_notes=source_notes,
            reviewer_username=str(request.get("reviewer_username") or ""),
            ci_triage_mode=bool(request.get("ci_triage")),
            repo_roots=repo_roots,
            repo_mount_paths=[s.container_path for s in repo_specs],
            project_facts=llm_prompt.load_project_facts(args.project_facts),
            drafts=drafts,
        )
        resources = openai_reviewer.OpenAIResources(
            client=client,
            tools=openai_reviewer.build_response_tools(
                vector_store_ids=[], file_search_max_num_results=None,
                use_web_search=False, web_search_context_size=args.web_search_context_size,
                web_search_cache_only=False, web_search_domains=[],
                use_shell=False, shell_container_id=None,
                code_interpreter_container_id=None, use_podman_shell=True,
            ),
            include=openai_reviewer.build_response_include(
                vector_store_ids=[], use_web_search=False, use_podman_shell=True,
            ),
            patch_file_id=patch_file_id,
            vector_store_ids=[],
            shared_container_id=None,
            podman_shell_session=session,
            uploaded_file_ids=uploaded_file_ids,
            debug_dir_specified=debug_dir_specified,
        )
        combiner_role = llm_prompt.role_with_labels(
            llm_prompt.COMBINER_ROLE,
            list(request.get("triage_label_allowlist") or []),
        )
        combiner = review_pipeline.make_reviewer(
            args.combine_model, args=args, resources=resources,
            role=combiner_role, verbose=args.verbose,
        )
        logger.info("replay combine stage: %s variant=%s merging %d drafts",
                    combiner.name, tool_args.variant, len(drafts))
        review = combiner.review(ctx)
        json.dump(
            {"variant": tool_args.variant,
             "classification": review.classification,
             "label_changes": list(review.label_changes),
             "message": review.message},
            sys.stdout, ensure_ascii=False, indent=1,
        )
        sys.stdout.write("\n")
        return 0
    finally:
        for file_id in uploaded_file_ids:
            delete_uploaded_file(client, file_id, verbose=args.verbose)
        session.close()
        podman_host.stop_container(handle)


if __name__ == "__main__":
    raise SystemExit(main())
