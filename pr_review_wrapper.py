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

Review a pull request with one or more LLM reviewers (OpenAI, Anthropic,
z.ai GLM, OpenRouter) and an optional combine stage.

This wrapper is meant to be used as the `--llm-review-cmd` helper described
for the PR auto-approval script. It reads one JSON object from stdin and prints
one JSON object to stdout.

Input JSON keys expected:
  - pull_request: object with metadata (number, title, body, author, html_url,
                  base_ref, head_ref, base_sha, head_sha, submodule_changes,
                  additions, deletions, changed_files, auto_merge)
  - patch: unified diff / patch text
  - patch_truncated: bool
  - discussion: prior pull-request comments and reviews
  - reviewer_username: username the tool will post as

With ``--task issue`` the request carries an ``issue`` object (number,
title, body, author, html_url, labels, created_at) instead of
``pull_request``, no patch, and ``discussion`` holds the issue comments.

Output JSON keys:
  - classification: a CLASSIFICATIONS member (--task pr) or an
    ISSUE_REPORT_CLASSIFICATIONS member (--task issue); see llm_review_api
  - message: review/analysis message, may be empty for approve; for skip
    the reason, led by the stage that chose it (``triager: ``,
    ``reviewer: `` or ``combiner: ``)

The wrapper enriches the review with source code context from a local git
checkout. When a source bundle is enabled it includes the touched files as
they would look merged into the current target-branch tip (falling back to
their pull_request.head_sha state when the merge conflicts). Direct quoted include files can also be included
optionally. In addition, the wrapper can optionally build or reuse an OpenAI
vector store containing the repository HEAD as separate file objects and enable
Responses API file_search for additional retrieval.
"""

from __future__ import annotations

import argparse
import base64
from dataclasses import replace
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

from openai import OpenAI

from common import (
    EXIT_QUOTA_EXHAUSTED,
    EXIT_REVIEW_HALTED,
    EXIT_TURN_FAILED,
    format_quota_hold,
    format_turn_failure,
    add_color_arg,
    apply_config_file_defaults,
    setup_logging,
)
from git_util import FORGE_REMOTES, git_changed_paths, git_merge_tree, git_resolve_first, git_show_file
import concurrency
import llm_review_api
from llm_review_api import (
    EXIT_BAD_MODEL_OUTPUT,
    VERBOSITY_LEVELS,
    BadModelOutput,
    ProviderTurnFailed,
    QuotaExhausted,
    Review,
    ReviewContext,
    Reviewer,
)
import branch_persist
import review_pipeline
from cli_backends import CLI_BACKENDS
from review_pipeline import (make_reviewer, review_pr, run_triage,
                             vet_review)
import workset
import podman_host
import podman_repos
import shell_tool
from llm_prompt import (
    COMBINER_ROLE,
    ISSUE_COMBINER_ROLE,
    ISSUE_INVESTIGATOR_ROLE,
    REVIEW_PROMPTS,
    REVIEWER_ROLE,
    UNGRADED_REVIEWER_ROLE,
    load_project_facts,
    make_triager_role,
    make_vetter_role,
    review_role,
    role_with_branches,
    role_with_labels,
)
import openai_common
import openai_reviewer
import openai_vector_store
from openai_common import (
    JsonObject,
    delete_uploaded_file,
    load_api_key,
    upload_text_file,
)
from openai_reviewer import (
    OpenAIResources,
    build_response_include,
    build_response_tools,
    make_openai_http_client,
)


logger = logging.getLogger(__name__)


DEFAULT_PODMAN_IMAGE = "localhost/fairy-review:latest"
# Empty: omit --network so podman uses its default (rootless
# slirp4netns/pasta). The LAN-egress block is the uid-keyed host rule
# from containers/setup_host.py --rootless-user, independent of the
# podman network.
DEFAULT_PODMAN_NETWORK = ""


# The in-container agent is shipped per review (podman cp), not baked into
# the image, so it stays versioned with this wrapper.
AGENT_LOCAL_PATH = Path(__file__).resolve().parent / "containers" / "fairy_agent.py"
AGENT_CONTAINER_DIR = "/work/.fairy"
AGENT_CONTAINER_PATH = f"{AGENT_CONTAINER_DIR}/fairy_agent.py"


# Default budget for the mini-model triage pre-check when ``--triage-model``
# is set. Triage output itself is a small JSON object, but the budget must
# also cover reasoning tokens, so keep it well above the raw schema size.
DEFAULT_TRIAGE_MAX_OUTPUT_TOKENS = 10_000
DEFAULT_TRIAGE_REASONING_EFFORT = "medium"
DEFAULT_MAX_SOURCE_FILES = 50
DEFAULT_MAX_FILE_BYTES = 500_000
DEFAULT_MAX_HEADER_FILE_BYTES = 150_000
DEFAULT_MAX_BUNDLE_BYTES = 1_000_000
DEFAULT_MAX_PATCH_BYTES = 500_000
DEFAULT_MAX_OUTPUT_TOKENS = 40_000
# 0 means "do not impose a client-side timeout"; the outer caller
# (e.g. fairy.py via its --llm-timeout) is responsible for
# bounding overall wall-clock time. The OpenAI SDK's own default (600s)
# is hidden and combined with hidden auto-retries, which makes total
# request time very hard to predict, so we override it explicitly.
DEFAULT_OPENAI_TIMEOUT_SECONDS = 0.0


def triage_label_allowlist_from_request(request: JsonObject) -> list[str]:
    raw = request.get("triage_label_allowlist")
    if not isinstance(raw, list):
        return []
    return [label for label in raw if isinstance(label, str) and label]


INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)


def parse_commit_author(value: str) -> tuple[str, str]:
    """argparse type for --commit-author: 'NAME <EMAIL>' -> (name, email)."""
    m = re.fullmatch(r"(.+?)\s*<(\S+@\S+)>", value.strip())
    if not m:
        raise argparse.ArgumentTypeError(
            f"expected 'NAME <EMAIL>', got {value!r}")
    return m.group(1), m.group(2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review a PR with one or more LLM reviewers.")
    p.add_argument(
        "--task",
        choices=("pr", "issue"),
        default="pr",
        help=(
            "What the stdin request describes: 'pr' (default) reviews a "
            "pull request; 'issue' analyzes an issue (no patch, no source "
            "bundle; the request carries an 'issue' object instead of "
            "'pull_request')."
        ),
    )
    p.add_argument(
        "--model",
        required=True,
        metavar="[PROMPT=]PROVIDER:MODEL[@EFFORT]",
        help="Reviewer for the main review pass, e.g. 'openai:gpt-5.4' or "
             "'code_review=anthropic:claude-opus-4'. 'PROMPT=' picks the prompt "
             f"it runs ({', '.join(REVIEW_PROMPTS)}, default {REVIEW_PROMPTS[0]}); "
             "--task issue takes no prompt.",
    )
    p.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="[PROMPT=]PROVIDER:MODEL[@EFFORT]",
        help=(
            "Add another reviewer to the ensemble, e.g. 'anthropic:claude-opus-4' "
            "or 'design_review=zai:glm-5.3'. Repeat for more; the same model may "
            "appear again under another prompt. All reviewers (--model plus each "
            "--extra-model) run on the same PR; with more than one "
            "you must pass --combine-model to merge their drafts. '@EFFORT' sets "
            "that reviewer's effort: an OpenAI reasoning effort, off/low/"
            "medium/high/xhigh/max as the Anthropic/GLM thinking effort, or "
            "minimal/low/medium/high as the Gemini thinking level."
        ),
    )
    p.add_argument(
        "--concurrency",
        action="append",
        default=[],
        type=concurrency.parse_limit,
        metavar="PROVIDER:COUNT",
        help="Cap in-flight calls to one provider across all fairy processes, e.g. 'codex:2'.",
    )
    p.add_argument(
        "--combine-model",
        default=None,
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Reviewer that verifies and combines the ensemble drafts into the "
            "final review (e.g. 'openai:gpt-5.4'). Required when more than one "
            "model reviewer is configured."
        ),
    )
    p.add_argument(
        "--fallback-model",
        default=None,
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Reviewer run instead when a sole main reviewer or the combiner "
            "spends its provider-ended-turn budget or is content-flagged "
            "(e.g. 'zai:glm-4.6@low'). Give it an @EFFORT; it does not "
            "inherit a requested effort, whose vocabulary is per provider."
        ),
    )
    p.add_argument(
        "--cyber-fallback-model",
        default=None,
        metavar="PROVIDER:MODEL[@EFFORT][:OPT=V]",
        help=(
            "Reviewer tried when a turn is ended by the provider's "
            "cybersecurity content flag"
        ),
    )
    p.add_argument(
        "--triage-model",
        default=None,
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Optional model to run a pre-check that classifies the PR as "
            "skip / reply_no_verdict / engage before the main review. Same "
            "``provider:model[@effort]`` form as --extra-model. "
            "When unset, triage is disabled. The "
            "triager gets podman shell and/or OpenAI file_search / "
            "web_search when configured, but not the source bundle."
        ),
    )
    p.add_argument(
        "--triage-reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        default=DEFAULT_TRIAGE_REASONING_EFFORT,
        help=(
            f"Reasoning effort for the triage model (default: {DEFAULT_TRIAGE_REASONING_EFFORT}). "
            "Ignored when --triage-model is unset."
        ),
    )
    p.add_argument(
        "--triage-max-output-tokens",
        type=int,
        default=DEFAULT_TRIAGE_MAX_OUTPUT_TOKENS,
        help=(
            f"Maximum output tokens from the triage model "
            f"(default: {DEFAULT_TRIAGE_MAX_OUTPUT_TOKENS}). "
            "Includes reasoning tokens. Ignored when --triage-model is unset."
        ),
    )
    p.add_argument(
        "--vet-model",
        default=None,
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Optional model that vets the verdict before it is posted: "
            "nothing offensive, malicious or advertising in the message, "
            "authorship and licensing of the PR's and the declared branches' "
            "commits in order; the workset ticket records the result. "
            "When unset, vetting is disabled."
        ),
    )
    p.add_argument(
        "--repo-root",
        help="Path to the primary local git checkout used to fetch changed file contents and optional vector store indexing.",
    )
    p.add_argument(
        "--extra-repo-root",
        action="append",
        default=[],
        help="Additional local git checkout to expose through optional vector store search. Can be repeated.",
    )
    p.add_argument(
        "--max-source-files",
        type=int,
        default=DEFAULT_MAX_SOURCE_FILES,
        help=f"Maximum number of changed source files to include (default: {DEFAULT_MAX_SOURCE_FILES})",
    )
    p.add_argument(
        "--max-file-bytes",
        type=int,
        default=DEFAULT_MAX_FILE_BYTES,
        help=f"Maximum bytes per non-header source file included directly (default: {DEFAULT_MAX_FILE_BYTES})",
    )
    p.add_argument(
        "--max-header-file-bytes",
        type=int,
        default=DEFAULT_MAX_HEADER_FILE_BYTES,
        help=f"Maximum bytes per header file included directly (default: {DEFAULT_MAX_HEADER_FILE_BYTES})",
    )
    p.add_argument(
        "--max-bundle-bytes",
        type=int,
        default=DEFAULT_MAX_BUNDLE_BYTES,
        help=f"Maximum total bytes for the direct source bundle (default: {DEFAULT_MAX_BUNDLE_BYTES})",
    )
    p.add_argument(
        "--max-patch-bytes",
        type=int,
        default=DEFAULT_MAX_PATCH_BYTES,
        help=f"Maximum patch bytes sent to the model (default: {DEFAULT_MAX_PATCH_BYTES})",
    )
    p.add_argument(
        "--max-output-tokens",
        type=int,
        default=DEFAULT_MAX_OUTPUT_TOKENS,
        help=f"Maximum output tokens from the model (default: {DEFAULT_MAX_OUTPUT_TOKENS})",
    )
    p.add_argument(
        "--allowed-model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL",
        help=(
            "Permit commenters to request this model for the main review "
            "pass (e.g. 'please re-review with gpt-5.5'); up to two at once "
            "review in parallel. Repeat to allow more models; same spec form "
            "as --extra-model. Reasoning effort requests (medium/high/xhigh) "
            "are always honored."
        ),
    )
    p.add_argument(
        "--service-tier",
        choices=["auto", "default", "flex", "priority"],
        default=None,
        help=(
            "Optional OpenAI ``service_tier`` parameter passed to the main "
            "responses.create call. ``flex`` trades higher latency / "
            "best-effort capacity for a substantially lower price on supported "
            "models (gpt-5 / o-series at the time of writing) and is useful as "
            "a cost-reduction measure for batch PR review runs. ``priority`` "
            "is the opposite trade-off (faster, more expensive). Omit to use "
            "the account default. Affects the main responses.create call "
            "only; the triage call has its own ``--triage-service-tier`` knob "
            "and is not implicitly coupled to this one. Note: not every model "
            "accepts every tier; if the API rejects the value the call will "
            "fail and the outer caller decides whether to retry."
        ),
    )
    p.add_argument(
        "--triage-service-tier",
        choices=["auto", "default", "flex", "priority"],
        default=None,
        help=(
            "Optional OpenAI ``service_tier`` for the triage responses.create "
            "call. Independent of ``--service-tier`` -- when unset (default) "
            "no ``service_tier`` is passed for the triage call and the account "
            "default is used, regardless of whether ``--service-tier`` is set "
            "for the main call. Useful when you want to apply ``flex`` to the "
            "cheap triage mini-pass (``--triage-service-tier flex``) without "
            "affecting the expensive main reviewer pass, or vice versa. "
            "Ignored when ``--triage-model`` is unset."
        ),
    )
    p.add_argument(
        "--reasoning-summary",
        choices=["auto", "concise", "detailed"],
        default="auto",
        help="Reasoning summary level (default: auto). OpenAI: Responses "
             "``reasoning.summary``; codex: ``model_reasoning_summary``; "
             "anthropic/zai: concise and detailed request "
             "``thinking.display=summarized`` when thinking is enabled.",
    )
    p.add_argument(
        "--top-p",
        type=float,
        help="Optional nucleus sampling parameter passed to the Responses API.",
    )
    p.add_argument(
        "--verbosity",
        default="high",
        choices=list(VERBOSITY_LEVELS),
        help="Output verbosity for every stage (default: high). Currently "
             "openai (``text.verbosity``) and codex (``model_verbosity``) "
             "only; anthropic/zai have no verbosity parameter. "
             "--final-verbosity overrides it for the user-visible stage.",
    )
    p.add_argument(
        "--final-verbosity",
        default=None,
        choices=list(VERBOSITY_LEVELS),
        help="Verbosity for the stage whose message the user sees (the "
             "combiner when one is configured, else the single main "
             "reviewer/investigator); default: --verbosity; overridden by "
             "an explicit community request the triager relays "
             "(``requested_verbosity``). Like --verbosity, openai/codex "
             "only currently.",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        help="Optional maximum number of built-in tool calls allowed in a single response.",
    )
    p.add_argument(
        "--web-search",
        choices=["off", "live", "cached"],
        default="off",
        help="Web search for reviewers (default off). 'live' reaches the "
             "current web, 'cached' uses OpenAI's index only. Drives the "
             "Responses web_search tool for openai: reviewers and codex's "
             "built-in web_search for codex:.",
    )
    p.add_argument(
        "--podman",
        action="store_true",
        help=(
            "Run shell commands in an ephemeral Podman container on the ssh "
            "podman host (function-calling ''shell'' tool). Repos are filled "
            "in from host-local bare mirrors; no OpenAI code_interpreter."
        ),
    )
    p.add_argument(
        "--podman-image",
        default=DEFAULT_PODMAN_IMAGE,
        help=f"Podman image tag for --podman (default: {DEFAULT_PODMAN_IMAGE}). Build with containers/build_image.py.",
    )
    p.add_argument(
        "--podman-network",
        default=DEFAULT_PODMAN_NETWORK,
        help="Podman network name to attach the container to. Empty "
             "(the default) omits --network and uses podman's default "
             "networking (rootless: pasta). A named network is only needed "
             "once the out-of-band egress LAN-block is in place.",
    )
    p.add_argument(
        "--shell-host",
        action="append",
        default=[],
        metavar="[LABEL=]USER@HOST[,port=N][,cpus=N][,memory=SIZE][,gpu=DEV][,device=PATH]...",
        help=(
            "Machine running review containers; repeat for more machines. "
            "The first is the default; LABEL (default x86_64) is what the "
            "model passes as the shell tool's machine parameter. All podman "
            "calls run as 'ssh DEST podman ...', repos are kept as bare "
            "mirrors on the host and filled into the container host-locally, "
            "so no full .git crosses the wire per review. Provision each "
            "host first with containers/provision_remote.py."
        ),
    )
    p.add_argument(
        "--podman-max-tool-rounds",
        type=int,
        default=0,
        help="Safety cap on shell function-call / API round-trips; 0 = unlimited (default).",
    )
    p.add_argument(
        "--podman-exec-timeout",
        type=float,
        default=600.0,
        help="Maximum seconds for a single shell function call (default: 600).",
    )
    p.add_argument(
        "--session-command",
        action="append",
        default=[],
        metavar="CMD",
        help="Run CMD in each review container before the LLM session and "
             "splice command + output into the prompt. ``{number}`` and "
             "``{base_ref}`` are replaced from the PR metadata. Repeatable.",
    )
    for cli, reviewer in CLI_BACKENDS.items():
        p.add_argument(
            f"--{cli}-host",
            metavar="[LABEL=]SSH_DEST",
            default=None,
            help=f"podman host that runs the {cli} container (same spec syntax "
                 f"as --shell-host). Required for {cli}: model specs -- {cli} "
                 "runs only in a container there, never on the wrapper host. "
                 f"Omit to disable {cli} support.",
        )
        p.add_argument(
            f"--{cli}-image",
            default=reviewer.default_image,
            metavar="TAG",
            help=f"Container image for the {cli} pass (default: %(default)s); "
                 f"build it with containers/provision_remote.py --{cli}-bin.",
        )
        p.add_argument(
            f"--{cli}-bin",
            default=cli,
            help=f"{cli} CLI path INSIDE the {cli} container (default: {cli} on "
                 "PATH, as baked into the image). Pin the deployed "
                 "version; the backend depends on its flag set.",
        )
        p.add_argument(
            f"--{cli}-timeout-seconds",
            type=float,
            default=0.0,
            help=f"Kill a {cli} pass after this many seconds; 0 = no "
                 "watchdog (default), a pass runs as long as the model works.",
        )
        p.add_argument(f"--{cli}-home", default=None, metavar="DIR",
                       help=reviewer.home_help)
    p.add_argument(
        "--podman-ssh-identity",
        metavar="KEYFILE",
        default=None,
        help=(
            "ssh identity (private key) for every --shell-host. Optional: "
            "omit to let OpenSSH pick it from the agent or ~/.ssh/config / "
            "default keys, exactly as plain 'ssh DEST' would."
        ),
    )
    p.add_argument(
        "--persist-branches",
        action="store_true",
        help=(
            "Let podman-backed reviewers keep git branches: each branch "
            "the verdict declares is packed out of its review container "
            "into the verdict's branch records and pushed only when the "
            "review is approved and sent."
        ),
    )
    p.add_argument(
        "--persist-repo",
        action="append",
        default=[],
        metavar="NAME",
        help="Container repo name whose checkout gets a fairy remote "
             "under --persist-branches; repeat per repo (default: the "
             "primary repo).",
    )
    p.add_argument(
        "--commit-author",
        type=parse_commit_author,
        default=None,
        metavar='"NAME <EMAIL>"',
        help=(
            "Default git identity (user.name/user.email) for commits "
            "created in review containers. Work taken from someone else "
            "keeps that person's authorship."
        ),
    )
    p.add_argument(
        "--podman-mirror-root",
        default=podman_repos.DEFAULT_MIRROR_ROOT,
        help=(
            "Host bare-mirror root (relative to the ssh user's home), "
            "default: %(default)s."
        ),
    )
    p.add_argument(
        "--web-search-context-size",
        choices=["low", "medium", "high"],
        default="medium",
        help="Context budget hint for the web_search tool (default: medium).",
    )
    p.add_argument(
        "--web-search-domain",
        action="append",
        default=[],
        help="Restrict web_search results to an allowed domain. Can be repeated.",
    )
    p.add_argument(
        "--file-search-max-num-results",
        type=int,
        help="Optional maximum number of file_search results to retrieve.",
    )
    p.add_argument(
        "--include-direct-includes",
        action="store_true",
        help="Also include currently resolvable quoted include files directly in the attached source bundle.",
    )
    p.add_argument(
        "--project-facts",
        type=Path,
        default=Path(__file__).resolve().parent / "project_facts" / "ffmpeg.md",
        metavar="FILE",
        help="Markdown file spliced into every role prompt as the project-facts "
             "section; per-project deployments point this at their own file "
             "(default: project_facts/ffmpeg.md next to this script).",
    )
    p.add_argument(
        "--use-vector-store-search",
        action="store_true",
        help="Index the full repository HEAD into an OpenAI vector store and enable file_search for additional retrieval.",
    )
    p.add_argument(
        "--vector-store-expiry-days",
        type=int,
        default=openai_vector_store.DEFAULT_VECTOR_STORE_EXPIRY_DAYS,
        help=f"Days after last use before an auto-created vector store expires (default: {openai_vector_store.DEFAULT_VECTOR_STORE_EXPIRY_DAYS})",
    )
    p.add_argument(
        "--vector-store-sync-max-retries",
        type=int,
        default=openai_vector_store.DEFAULT_VECTOR_STORE_SYNC_MAX_RETRIES,
        help=f"Maximum retry passes for failed vector-store files (default: {openai_vector_store.DEFAULT_VECTOR_STORE_SYNC_MAX_RETRIES})",
    )
    p.add_argument(
        "--prepare-vector-store-only",
        action="store_true",
        help="Build or refresh the cached vector store for the current repository HEAD, then exit without reading stdin or running a review.",
    )
    p.add_argument(
        "--podman-parallel-tool-calls",
        action="store_true",
        help="Allow several shell calls per follow-up round.",
    )
    p.add_argument(
        "--shared-containers",
        action="store_true",
        help="Run triage, the reviewers and the combiner in one container per "
             "machine instead of one per pass, so parallel reviewers share one "
             "working tree; the vetter always gets its own.",
    )
    p.add_argument(
        "--simulate-past-cutoff",
        metavar="ISO8601",
        help="Prune container-repo refs whose commits postdate this time "
             "(simulate-past replays; past refs incl. old force-pushes stay).",
    )
    p.add_argument(
        "--no-source-bundle",
        action="store_true",
        help="Do not attach directly included source files; review the metadata, patch, and optional vector store only.",
    )
    p.add_argument(
        "--debug-response-dir",
        default=".openai_debug",
        help="Directory where raw Responses API payloads are dumped on extraction/validation failures (default: .openai_debug)",
    )
    p.add_argument(
        "--workset-file",
        type=Path,
        help="Caller's workset item file; stage progress, the triage result "
             "and reviewer drafts are recorded there as they happen.",
    )
    p.add_argument(
        "--halt-file",
        help="Caller's halt file; once it exists the review stops at its "
             "next model turn or shell command and its containers are "
             "paused.",
    )
    p.add_argument(
        "--openai-timeout-seconds",
        type=float,
        default=DEFAULT_OPENAI_TIMEOUT_SECONDS,
        help=(
            "OpenAI HTTP client per-request timeout in seconds. "
            "0 (default) means no client-side timeout, leaving the outer caller "
            "(e.g. fairy.py --llm-timeout) in charge of bounding wall-clock time. "
            "Overrides the SDK's hidden default (600s) which combines badly with the "
            "wrapper's deliberate max_retries=0 setting and outer subprocess timeouts."
        ),
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print debug information to stderr.",
    )
    add_color_arg(p)
    apply_config_file_defaults(p)
    args = p.parse_args()
    try:
        args.machines = [
            podman_host.parse_shell_host(s, identity=args.podman_ssh_identity)
            for s in args.shell_host
        ]
    except ValueError as exc:
        p.error(str(exc))
    labels = [m.label for m in args.machines]
    if len(set(labels)) != len(labels):
        p.error(f"duplicate machine labels in --shell-host: {', '.join(labels)}")
    if args.podman and not args.machines:
        p.error("--podman requires at least one --shell-host")
    if args.shared_containers and not args.podman:
        p.error("--shared-containers requires --podman")
    if args.persist_branches and not args.podman:
        p.error("--persist-branches requires --podman (branches live in "
                "podman review containers)")
    if args.commit_author and not args.podman:
        p.error("--commit-author requires --podman (the identity is set "
                "in podman review containers)")
    if args.extra_model and not args.combine_model:
        # rejected at parse time: discovered after run_parallel it would
        # have billed every reviewer before failing
        p.error("--extra-model requires --combine-model to merge the drafts")
    for cli in CLI_BACKENDS:
        host_spec = getattr(args, f"{cli}_host")
        setattr(args, f"{cli}_host", None)
        if host_spec:
            try:
                setattr(args, f"{cli}_host", podman_host.parse_shell_host(
                    host_spec, identity=args.podman_ssh_identity))
            except ValueError as exc:
                p.error(str(exc))
    # Strip the optional PROMPT= before the provider-prefix scans below read
    # these specs. args.main_prompts stays parallel to [model, *extra_model].
    args.main_prompts = []
    bare_specs = []
    for flag, spec in (("--model", args.model),
                       *(("--extra-model", s) for s in args.extra_model)):
        prompt, sep, rest = spec.partition("=")
        if not sep:
            prompt, rest = None, spec
        elif args.task != "pr":
            p.error(f"{flag} {spec!r}: a prompt can only be chosen for --task pr")
        elif prompt not in REVIEW_PROMPTS:
            p.error(f"{flag} {spec!r}: unknown prompt {prompt!r} "
                    f"(use {', '.join(REVIEW_PROMPTS)})")
        args.main_prompts.append(prompt)
        bare_specs.append(rest)
    args.model, args.extra_model = bare_specs[0], bare_specs[1:]
    for flag, spec in (("--triage-model", args.triage_model),
                       ("--combine-model", args.combine_model),
                       ("--vet-model", args.vet_model),
                       *(("--allowed-model", s) for s in args.allowed_model)):
        if spec and "=" in spec:
            p.error(f"{flag} {spec!r}: the prompt of this pass is fixed by the flag")
    all_specs = [s for s in (args.model, *args.extra_model, args.combine_model,
                             args.triage_model, args.vet_model, *args.allowed_model,
                             args.fallback_model, args.cyber_fallback_model)
                 if s]
    for cli, reviewer in CLI_BACKENDS.items():
        specs = [s for s in all_specs if s.startswith(f"{cli}:")]
        if specs and getattr(args, f"{cli}_host") is None:
            p.error(
                f"{cli} model(s) {', '.join(specs)} require --{cli}-host "
                f"({cli} runs only in a container there, never on the wrapper host)"
            )
        if specs:
            args.max_patch_bytes = min(args.max_patch_bytes, reviewer.max_patch_bytes)
            args.max_bundle_bytes = min(args.max_bundle_bytes, reviewer.max_bundle_bytes)
    return args


def workset_note(args: argparse.Namespace, field: str, value: object) -> None:
    """Record one wrapper field in the caller's work file (if any):
    ``stage`` progress ("triage" | "review" | "combine") or a stage's
    result. The wrapper owns the file while it runs (the caller blocks
    on the subprocess); the final verdict travels on stdout and is
    persisted by the caller."""
    if args.workset_file:
        def record(data: dict) -> None:
            data[field] = value
        workset.update_json(args.workset_file, record)


def workset_note_drafts(args: argparse.Namespace, drafts, *,
                        combining: bool, failed: list[str] = ()) -> None:
    """``drafts`` are ``Review``s; the stage moves to "combine" when a
    combiner runs next. ``failed`` names the reviewers whose draft is
    missing, so the operator surface can say the verdict is partial."""
    if not args.workset_file:
        return

    def record(data: dict) -> None:
        data["drafts"] = [
            {"classification": d.classification, "message": d.message,
             "label_changes": list(d.label_changes),
             "branches": list(d.branches), "model": d.model,
             "prompt": d.prompt}
            for d in drafts
        ]
        data["failed_reviewers"] = list(failed)
        if combining:
            data["stage"] = "combine"

    workset.update_json(args.workset_file, record)


def read_request() -> JsonObject:
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"stdin does not contain valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise SystemExit("stdin JSON must be an object")
    return data


def find_repo_root(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else None

    try:
        cp = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None

    if cp.returncode != 0:
        return None

    text = cp.stdout.strip()
    if not text:
        return None
    path = Path(text)
    return path if path.exists() else None


def get_all_repo_roots(primary_root: Path | None, extra_roots: list[str]) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    def add_root(root: Path | None) -> None:
        if root is None:
            return
        root = root.resolve()
        if root in seen:
            return
        seen.add(root)
        roots.append(root)

    add_root(primary_root)
    for extra in extra_roots:
        add_root(find_repo_root(extra))

    return roots


def is_c_or_h_file(relpath: str) -> bool:
    return relpath.endswith((".c", ".h"))


def resolve_repo_relative_path(repo_root: Path, candidate: Path) -> str | None:
    repo_root_resolved = repo_root.resolve()
    resolved = (repo_root / candidate).resolve()
    try:
        rel = resolved.relative_to(repo_root_resolved)
    except ValueError as e:
        logger.warning("%s", e)
        return None
    return rel.as_posix()


def extract_repo_include_path_candidates(
    repo_root: Path, including_relpath: str, text: str
) -> list[list[str]]:
    if not is_c_or_h_file(including_relpath) or "#include" not in text:
        return []

    include_candidates: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    including_dir = Path(including_relpath).parent

    for match in INCLUDE_RE.finditer(text):
        delimiter = match.group(1)
        include_target = match.group(2).strip()

        # Ignore system includes like #include <...>.
        if delimiter != '"':
            continue

        candidates: list[str] = []
        for candidate in (including_dir / include_target, Path(include_target)):
            resolved_relpath = resolve_repo_relative_path(repo_root, candidate)
            if resolved_relpath is not None and resolved_relpath not in candidates:
                candidates.append(resolved_relpath)

        if not candidates:
            continue

        key = tuple(candidates)
        if key not in seen:
            seen.add(key)
            include_candidates.append(candidates)

    return include_candidates


def build_model_visible_repo_head_map(
    repo_roots: list[Path],
    repo_heads: dict[str, str],
    *,
    podman_repo_specs: list[podman_repos.RepoSpec] | None = None,
) -> dict[str, str]:
    if podman_repo_specs:
        visible: dict[str, str] = {}
        for spec in podman_repo_specs:
            repo_key = str(spec.repo_root.resolve())
            head_sha = repo_heads.get(repo_key)
            if isinstance(head_sha, str) and head_sha:
                visible[spec.container_path] = head_sha
        return visible

    # Without container mounts, expose simple repo labels instead of unusable host-local paths.
    visible = {}
    for repo_root in repo_roots:
        repo_key = str(repo_root.resolve())
        head_sha = repo_heads.get(repo_key)
        if not isinstance(head_sha, str) or not head_sha:
            continue
        label = re.sub(r"[^A-Za-z0-9._-]", "-", repo_root.name).strip("._-") or "repo"
        visible[label] = head_sha
    return visible


def parse_source_bundle_request_info(request: JsonObject) -> tuple[JsonObject, str, str]:
    pr = request.get("pull_request")
    if not isinstance(pr, dict):
        raise RuntimeError("missing pull_request object")

    for key in ("head_sha", "base_sha"):
        if not isinstance(pr.get(key), str) or not pr[key]:
            raise RuntimeError(f"missing pull_request.{key}")

    return pr, pr["base_sha"], pr["head_sha"]


def load_source_bundle_texts(
    repo_root: Path,
    head_sha: str,
    direct_paths: list[str],
    commit_shas: list[str],
    *,
    include_direct_includes: bool,
) -> tuple[list[str], dict[str, str], dict[str, str], list[str]]:
    ordered_paths: list[str] = []
    seen_paths: set[str] = set()
    loaded_texts: dict[str, str] = {}
    source_revisions: dict[str, str] = {}
    notes: list[str] = []

    def add_path(relpath: str) -> None:
        if relpath not in seen_paths:
            seen_paths.add(relpath)
            ordered_paths.append(relpath)

    def resolve_path(relpath: str) -> tuple[str, str] | None:
        for revision in commit_shas:
            file_content = git_show_file(repo_root, revision, relpath)
            if file_content is not None:
                return revision, file_content
        return None

    for relpath in direct_paths:
        resolved = resolve_path(relpath)
        if resolved is None:
            raise RuntimeError(f"missing source for {relpath} at {head_sha}")
        source_revisions[relpath], file_content = resolved
        loaded_texts[relpath] = file_content
        add_path(relpath)

    if not include_direct_includes:
        return ordered_paths, loaded_texts, source_revisions, notes

    for relpath in direct_paths:
        text = loaded_texts[relpath]
        for include_candidates in extract_repo_include_path_candidates(repo_root, relpath, text):
            chosen_relpath: str | None = None
            for include_relpath in include_candidates:
                if include_relpath in loaded_texts:
                    chosen_relpath = include_relpath
                    break

                resolved = resolve_path(include_relpath)
                if resolved is not None:
                    source_revisions[include_relpath], loaded_texts[include_relpath] = resolved
                    chosen_relpath = include_relpath
                    break

            if chosen_relpath is None:
                notes.append(
                    f"missing included source for {' or '.join(include_candidates)} at {head_sha}"
                )
                continue

            add_path(chosen_relpath)

    return ordered_paths, loaded_texts, source_revisions, notes


def append_source_bundle_files(
    ordered_paths: list[str],
    loaded_texts: dict[str, str],
    *,
    source_labels: dict[str, str],
    max_file_bytes: int,
    max_header_file_bytes: int,
    max_bundle_bytes: int,
    bundle_parts: list[str],
    total_bytes: int,
) -> tuple[list[str], list[str], int]:
    used_paths: list[str] = []
    truncated_paths: list[str] = []

    for relpath in ordered_paths:
        text = loaded_texts[relpath]
        source_from = source_labels[relpath]

        raw = text.encode("utf-8", errors="replace")
        truncated = False
        file_limit = max_header_file_bytes if relpath.endswith(".h") else max_file_bytes
        if len(raw) > file_limit:
            raw = raw[:file_limit]
            text = raw.decode("utf-8", errors="replace")
            truncated = True
            truncated_paths.append(relpath)
        else:
            text = raw.decode("utf-8", errors="replace")

        part = (
            f"===== BEGIN FILE: {relpath} =====\n"
            f"SOURCE: {source_from}\n"
            f"TRUNCATED: {'yes' if truncated else 'no'}\n\n"
            f"{text}\n"
            f"===== END FILE: {relpath} =====\n\n"
        )
        part_bytes = len(part.encode("utf-8"))
        if total_bytes + part_bytes > max_bundle_bytes:
            break

        bundle_parts.append(part)
        total_bytes += part_bytes
        used_paths.append(relpath)

    return used_paths, truncated_paths, total_bytes


def resolve_target_tip(repo_root: Path, base_ref: str) -> str | None:
    """The local SHA of ``base_ref``'s tip, tried through the remotes the
    deployment fetches from; None when none of them has it."""
    return git_resolve_first(
        repo_root,
        (*(f"{remote}/{base_ref}" for remote in FORGE_REMOTES),
         base_ref))


class TargetMerge(NamedTuple):
    """git_merge_tree's ``tree`` and ``conflicts`` of the PR head merged
    into ``tip``, the local tip of ``base_ref``."""
    base_ref: str
    tip: str
    tree: str
    conflicts: str


def merge_into_target(request: JsonObject, repo_root: Path | None) -> TargetMerge | None:
    """The PR head of ``request`` merged into the local tip of its
    base_ref; None, logged, when it cannot be made."""
    pr = request.get("pull_request")
    if repo_root is None or not isinstance(pr, dict):
        return None
    head_sha, base_ref = pr.get("head_sha"), pr.get("base_ref")
    if not (isinstance(head_sha, str) and head_sha and isinstance(base_ref, str) and base_ref):
        return None
    tip = resolve_target_tip(repo_root, base_ref)
    merge = None
    if tip:
        try:
            merge = TargetMerge(base_ref, tip, *git_merge_tree(repo_root, head_sha, tip))
        except RuntimeError as exc:
            logger.warning("merge into %s: %s", base_ref, exc)
    logger.info("merge of %s into %s (%s): %s%s", head_sha, base_ref, tip,
                merge.tree if merge else "unavailable",
                ", conflicts" if merge and merge.conflicts else "")
    return merge


def build_source_bundle(
    request: JsonObject,
    repo_root: Path | None,
    merge: TargetMerge | None,
    *,
    max_source_files: int,
    max_file_bytes: int,
    max_header_file_bytes: int,
    max_bundle_bytes: int,
    include_direct_includes: bool,
    verbose: bool,
) -> tuple[str, list[str], list[str]]:
    if repo_root is None:
        raise RuntimeError("no local git checkout found")

    _, patch_base_sha, head_sha = parse_source_bundle_request_info(request)
    changed_paths = git_changed_paths(repo_root, patch_base_sha, head_sha)
    logger.debug("source bundle: %s..%s changes %s", patch_base_sha, head_sha, changed_paths)

    bundle_parts: list[str] = []
    notes: list[str] = []
    total_bytes = 0

    # Contents come from the predicted merge of PR head into the current target
    # tip, so a model cannot conclude the PR "removes" unrelated target work.
    header = (
        f"Repository: {repo_root.name}\n"
        f"Head SHA: {head_sha}\n"
        + (f"File contents are the merge result of the PR head into the current "
           f"{merge.base_ref} tip {merge.tip}.\n"
           if merge else "")
        + ("The files listed in merge_conflicts carry git's conflict markers.\n"
           if merge and merge.conflicts else "")
        + f"Changed files: {len(changed_paths)}\n\n"
    )
    bundle_parts.append(header)
    total_bytes += len(header.encode("utf-8"))

    direct_paths = changed_paths[:max_source_files]
    commit_shas = [merge.tree, head_sha] if merge else [head_sha]
    ordered_paths, loaded_texts, source_revisions, include_notes = load_source_bundle_texts(
        repo_root,
        head_sha,
        direct_paths,
        commit_shas,
        include_direct_includes=include_direct_includes,
    )
    notes.extend(include_notes)

    source_labels = {
        relpath: (f"merge of PR head {head_sha} into {merge.base_ref} {merge.tip}"
                  if merge and revision == merge.tree else f"git show {revision}:{relpath}")
        for relpath, revision in source_revisions.items()
    }
    used_paths, truncated_paths, total_bytes = append_source_bundle_files(
        ordered_paths,
        loaded_texts,
        source_labels=source_labels,
        max_file_bytes=max_file_bytes,
        max_header_file_bytes=max_header_file_bytes,
        max_bundle_bytes=max_bundle_bytes,
        bundle_parts=bundle_parts,
        total_bytes=total_bytes,
    )

    if len(used_paths) < len(ordered_paths):
        next_path = ordered_paths[len(used_paths)]
        notes.append(f"stopped before {next_path}: source bundle size limit reached")

    if truncated_paths:
        preview = ", ".join(truncated_paths[:5])
        if len(truncated_paths) > 5:
            preview += f", +{len(truncated_paths) - 5} more"
        notes.append(
            f"warning: truncated {len(truncated_paths)} source file(s) due to byte limits: {preview}"
        )

    if len(changed_paths) > max_source_files:
        notes.append(
            f"only first {max_source_files} changed files were considered out of {len(changed_paths)}"
        )

    if verbose:
        logger.debug("source bundle: used %d file(s), %d byte(s)", len(used_paths), total_bytes)

    if notes:
        bundle_parts.append("===== NOTES =====\n" + "\n".join(notes) + "\n")

    return "".join(bundle_parts), used_paths, notes

def build_patch_bundle(patch: str, max_patch_bytes: int) -> tuple[str, bool]:
    raw = patch.encode("utf-8", errors="replace")
    truncated = False
    if len(raw) > max_patch_bytes:
        raw = raw[:max_patch_bytes]
        truncated = True
    text = raw.decode("utf-8", errors="replace")
    bundle = (
        "===== BEGIN PATCH =====\n"
        f"TRUNCATED: {'yes' if truncated else 'no'}\n\n"
        f"{text}\n"
        "===== END PATCH =====\n"
    )
    return bundle, truncated


def attach_turn_fallbacks(
    reviewer: Reviewer,
    *,
    args: argparse.Namespace,
    resources: OpenAIResources | None,
    verbosity: str | None,
    failure_fails_run: bool,
) -> None:
    """Set ``reviewer.fallbacks`` for provider-ended turns: the
    flag-only --cyber-fallback-model reviewer first, then, with
    ``failure_fails_run`` (a sole main reviewer or the combiner), the
    --fallback-model reviewer. Both are built with the primary's
    role."""
    specs = [args.cyber_fallback_model] + \
        [args.fallback_model] * failure_fails_run
    chain = [
        make_reviewer(spec, args=args, resources=resources,
                      role=reviewer.role, verbose=args.verbose,
                      verbosity=verbosity)
        for spec in specs if spec]
    if args.cyber_fallback_model:
        chain[0].flag_only = True
    reviewer.fallbacks = tuple(chain)


def unattended_posting_block(branches: Sequence[JsonObject]) -> str | None:
    """Why the verdict's branch records need an operator whatever the
    vetter says, or None when they may go out unattended."""
    for count, limit, what in (
            (sum(1 for record in branches if record.get("pr")), 1, "pull requests"),
            (sum(1 for record in branches if record.get("mode") != "delete"), 3, "pushed branches"),
            (sum(len(base64.b64decode(record.get("bundle") or "")) for record in branches),
             256 * 1024, "bytes of git bundles")):
        if count > limit:
            return f"the verdict has {count} {what}; more than {limit} needs an operator's approval"
    return None


def stop_containers_on_termination() -> None:
    """On SIGTERM, or when the parent process (fairy's worker) exits,
    remove every container this run started and exit with 143. Pause
    them within a second of the halt file appearing: a review stops only
    at its next boundary, and a CLI agent pass that makes no further shell
    call never reaches one."""
    parent_pid = os.getppid()
    terminated = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: terminated.set())

    def watch() -> None:
        paused = False
        while not terminated.wait(1.0) and os.getppid() == parent_pid:
            if not paused and shell_tool.halted_by_file():
                logger.warning("halt file %s exists; pausing this run's containers",
                               shell_tool.HALT_FILE)
                podman_host.pause_all_containers()
                paused = True
        logger.warning("%s; removing this run's containers",
                       "SIGTERM received" if terminated.is_set()
                       else f"parent process {parent_pid} exited")
        podman_host.stop_all_containers()
        os._exit(128 + signal.SIGTERM)

    threading.Thread(target=watch, name="termination-watch", daemon=True).start()


def emit_review_stdout(
    classification: str,
    message: str,
    *,
    label_changes: list[dict[str, object]] | None = None,
    branches: list[dict[str, object]] | None = None,
) -> None:
    out: dict[str, object] = {"classification": classification, "message": message}
    if label_changes:
        out["label_changes"] = label_changes
    if branches:
        out["branches"] = branches
    json.dump(out, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def substitute_session_command(command: str, request: JsonObject) -> str:
    """Fill ``{number}`` / ``{base_ref}`` in a --session-command from the PR."""
    pr = request.get("pull_request")
    if not isinstance(pr, dict):
        return command
    number = pr.get("number")
    if number is not None:
        command = command.replace("{number}", str(number))
    base_ref = pr.get("base_ref")
    if isinstance(base_ref, str) and base_ref:
        command = command.replace("{base_ref}", base_ref)
    return command


def open_review_container_shell(
    spec: podman_host.ShellHostSpec,
    repo_specs: list[podman_repos.RepoSpec],
    args: argparse.Namespace,
    session_commands: Sequence[str] = (),
) -> tuple[podman_host.ContainerHandle, podman_host.ContainerShellSession, str]:
    """Start a fresh ephemeral container on ``spec``'s host, fill its repos,
    and open a shell.

    One isolated container per call. ``session_commands`` run in every
    container, in the primary repo's mount, so state they create exists
    for each reviewer; the returned transcript is what the prompt splices
    in. On any provisioning failure the half-started container is stopped
    before the error propagates.
    """
    handle = podman_host.start_ephemeral_container(
        image=args.podman_image,
        host=spec.host,
        network=args.podman_network,
        memory=spec.memory,
        cpus=spec.cpus,
        extra_args=spec.podman_device_args,
    )
    try:
        podman_repos.provision_repos_into_container(
            handle, repo_specs, spec.host,
            prune_refs_after=(
                int(datetime.fromisoformat(args.simulate_past_cutoff).timestamp())
                if args.simulate_past_cutoff else None
            ),
        )
        for key, value in zip(("user.name", "user.email"),
                              args.commit_author or ()):
            cp = podman_host.run_on_remote_host(
                spec.host, "podman", "exec", handle.container_id,
                "git", "config", "--global", key, value)
            if cp.returncode != 0:
                raise podman_host.ContainerInfraError(
                    f"git config {key} in container "
                    f"{handle.container_id[:12]} failed: "
                    f"{cp.stderr_text}")
        podman_repos.start_recoll_index(handle)
        podman_host.copy_into_container(handle, AGENT_LOCAL_PATH, AGENT_CONTAINER_DIR)
        session = podman_host.open_container_shell(handle, AGENT_CONTAINER_PATH)
        transcript = shell_tool.run_session_commands(
            session, list(session_commands),
            max_timeout_s=args.podman_exec_timeout,
            cwd=repo_specs[0].container_path if repo_specs else None,
        ) if session_commands else ""
    except Exception:
        podman_host.stop_container(handle)
        raise
    return handle, session, transcript


def main() -> int:
    args = parse_args()
    shell_tool.CANCEL_FILE = args.workset_file
    shell_tool.HALT_FILE = args.halt_file
    debug_dir_specified = any(
        arg == "--debug-response-dir" or arg.startswith("--debug-response-dir=")
        for arg in sys.argv[1:]
    )
    # Pass the sibling module loggers explicitly so their DEBUG/INFO output
    # is visible under ``--verbose``; the plain ``logger`` handlers set up by
    # ``setup_logging`` for the caller do not propagate to unrelated module
    # loggers.
    setup_logging(
        logger,
        args.verbose,
        concurrency.logger,
        llm_review_api.logger,
        openai_common.logger,
        openai_reviewer.logger,
        openai_vector_store.logger,
        review_pipeline.logger,
        podman_host.logger,
        podman_repos.logger,
        # By name, not module attribute: the anthropic and gemini modules
        # are imported lazily (only when such a model is requested) and
        # fetching a logger from the registry does not import the module.
        logging.getLogger("shell_tool"),
        logging.getLogger("anthropic_common"),
        logging.getLogger("anthropic_reviewer"),
        logging.getLogger("tool_loop"),
        logging.getLogger("gemini_reviewer"),
        logging.getLogger("codex_catalog"),
        logging.getLogger("cli_container"),
        logging.getLogger("cli_reviewer"),
        *(logging.getLogger(reviewer.__module__)
          for reviewer in CLI_BACKENDS.values()),
        branch_persist.logger,
        color=args.color,
    )
    concurrency.configure(args.concurrency)
    stop_containers_on_termination()

    # OPENAI_API_KEY is only required when an OpenAI backend actually runs.
    model_specs = [
        s for s in (args.model, *args.extra_model, args.triage_model,
                    args.combine_model, args.vet_model, *args.allowed_model)
        if s
    ]
    openai_provider_requested = any(s.startswith("openai:") for s in model_specs)
    openai_needed = (
        openai_provider_requested
        or args.use_vector_store_search
        or args.prepare_vector_store_only
    )

    client: OpenAI | None = None
    if openai_needed:
        api_key = load_api_key()
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set and was not found in .env")

        openai_timeout: float | None
        if args.openai_timeout_seconds and args.openai_timeout_seconds > 0:
            openai_timeout = float(args.openai_timeout_seconds)
        else:
            openai_timeout = None
        if args.verbose:
            logger.debug(
                "openai client init timeout=%s max_retries=0 tcp_keepalive=idle60s/30sx8 (sdk auto-retry disabled)",
                f"{openai_timeout:.1f}s" if openai_timeout is not None else "none",
            )
        # max_retries=0 disables the SDK's hidden auto-retry (default: 2).
        # Combined with the outer subprocess timeout, those silent retries
        # caused the wrapper to be killed mid-retry while the server-side
        # call had already succeeded, producing duplicate billed requests
        # with no visible response. Retries (when wanted) are now handled
        # explicitly inside this wrapper, or by the caller.
        client = OpenAI(
            api_key=api_key,
            timeout=openai_timeout,
            max_retries=0,
            http_client=make_openai_http_client(),
        )
    repo_root = find_repo_root(args.repo_root)
    repo_roots = get_all_repo_roots(repo_root, args.extra_repo_root)
    if args.prepare_vector_store_only:
        if not repo_roots:
            raise RuntimeError("--prepare-vector-store-only requires at least one local git checkout")
        prepared = openai_vector_store.ensure_vector_stores_synced_for_repos(
            client,
            repo_roots,
            expiry_days=args.vector_store_expiry_days,
            sync_max_retries=args.vector_store_sync_max_retries,
            verbose=args.verbose,
        )
        json.dump(prepared, sys.stdout, ensure_ascii=False)
        sys.stdout.write("\n")
        return 0

    request = read_request()
    ci_triage_active = bool(
        isinstance(request.get("ci_triage"), dict) and request.get("ci_triage")
    )
    # Cross-process contract with fairy's forced reviews:
    # run the full reviewer pass even when triage votes ``skip``.
    ignore_triage_skip = bool(request.get("ignore_triage_skip"))
    # Cross-process contract with fairy's --force-engage: run the
    # full reviewer pass regardless of the triage route (skip / reply_no_verdict).
    force_engage = bool(request.get("force_engage"))

    patch = request.get("patch") if isinstance(request.get("patch"), str) else ""
    reviewer_username = request.get("reviewer_username")
    if not isinstance(reviewer_username, str):
        reviewer_username = ""

    logger.debug("task=%s", args.task)
    project_facts = load_project_facts(args.project_facts)
    logger.debug(
        "project facts loaded from %s (%d bytes)",
        args.project_facts, len(project_facts),
    )

    vector_store_ids: list[str] = []
    indexed_head_map: dict[str, str] = {}
    if args.use_vector_store_search:
        if not repo_roots:
            raise RuntimeError("--use-vector-store-search requires at least one local git checkout")
        vector_store_ids, indexed_head_map = openai_vector_store.get_live_cached_vector_stores_for_repos(
            client,
            repo_roots,
            verbose=args.verbose,
        )

    session_commands: list[str] = []
    session_transcript = ""
    podman_repo_specs: list[podman_repos.RepoSpec] = []
    machines: list[podman_host.ShellHostSpec] = args.machines if args.podman else []
    spec_by_label = {m.label: m for m in machines}
    ensemble_shells: list[tuple[podman_host.ContainerHandle, podman_host.ContainerShellSession]] = []
    shared_shells: dict[str, tuple[podman_host.ContainerShellSession, str]] = {}
    shared_shells_lock = threading.Lock()
    # Keyed by id() since sessions are unhashable.
    poisoned_session_ids: set[int] = set()
    session_handles: dict[int, podman_host.ContainerHandle] = {}
    session_seeds: dict[int, branch_persist.RemoteSeeds] = {}
    persist_repos: list[podman_repos.RepoSpec] = []
    persist_base_shas: dict[str, list[str]] = {}
    inherited_branches: list[JsonObject] = []

    def open_machine_shell(
        label: str,
    ) -> tuple[podman_host.ContainerShellSession, str]:
        handle, session, transcript = open_review_container_shell(
            spec_by_label[label], podman_repo_specs, args, session_commands,
        )
        ensemble_shells.append((handle, session))
        session_handles[id(session)] = handle
        if args.persist_branches:
            session_seeds[id(session)] = branch_persist.setup_container_remotes(
                handle, podman_repo_specs, [s.name for s in persist_repos])
        if inherited_branches:
            branch_persist.add_to_container_remote(
                handle, inherited_branches,
                {s.name: s.container_path for s in persist_repos})
        return session, transcript

    def close_machine_shells(
        sessions: Iterable[podman_host.ContainerShellSession],
    ) -> None:
        for session in sessions:
            handle = session_handles.pop(id(session), None)
            if handle is None:
                continue
            try:
                session.close()
                podman_host.stop_container(handle)
            except Exception as exc:
                session_handles[id(session)] = handle
                logger.warning("stopping container id=%s failed (%s: %s); left "
                               "for a later cleanup", handle.container_id[:12],
                               type(exc).__name__, exc)

    def open_shared_shell(
        label: str,
    ) -> tuple[podman_host.ContainerShellSession, str]:
        with shared_shells_lock:
            if label in shared_shells:
                logger.debug("reusing the shared container on machine %s", label)
            else:
                shared_shells[label] = open_machine_shell(label)
            return shared_shells[label]

    open_stage_shell = open_shared_shell if args.shared_containers else open_machine_shell
    close_stage_shells = None if args.shared_containers else close_machine_shells

    def hand_over_branches(records: list[JsonObject]) -> None:
        """Put a finished stage's branch records on the fairy remotes of
        the shells open now and of every shell opened later, so the
        next stage verifies them like its own work."""
        inherited_branches[:] = records
        for session, _ in shared_shells.values():
            handle = session_handles.get(id(session))
            if handle is not None:
                branch_persist.add_to_container_remote(
                    handle, records,
                    {s.name: s.container_path for s in persist_repos})

    def collect_review_branches(
        sessions: Sequence[podman_host.ContainerShellSession],
        declared: list[JsonObject],
        pull_requests: list[JsonObject],
    ) -> list[JsonObject]:
        collected = branch_persist.collect_declared_branches(
            [(session_handles[i], session_seeds.get(i) or {})
             for i in map(id, sessions) if i in session_handles],
            declared,
            pull_requests,
            repo_specs=persist_repos,
            base_shas=persist_base_shas,
            inherited=inherited_branches,
        )
        records = [cb.record() for cb in collected]
        if args.commit_author:
            for i, record in enumerate(records):
                try:
                    records[i] = branch_persist.replace_invented_fairy_identities(
                        record, args.commit_author)
                except branch_persist.BranchTransferError as exc:
                    logger.error("identity rewrite of %r failed, keeping the "
                                 "branch as pushed: %s",
                                 record.get("branch"), exc)
        return records

    def report_poisoned(session: podman_host.ContainerShellSession) -> None:
        poisoned_session_ids.add(id(session))
        logger.warning(
            "review container flagged as poisoned (CLI reviewer run crashed); "
            "pausing it for forensics"
        )
        podman_host.pause_container(session_handles[id(session)])
        shell_tool.halt()

    uploaded_file_ids: list[str] = []
    try:
        for host in {m.host for m in machines} | {
                s.host for cli in CLI_BACKENDS
                if (s := getattr(args, f"{cli}_host"))}:
            podman_host.reap_orphaned_containers(host)
        if args.podman:
            if not repo_roots:
                raise RuntimeError(
                    "--podman requires at least one local git checkout "
                    "via --repo-root / --extra-repo-root"
                )
            for m in machines:
                if not podman_host.image_tag_exists(args.podman_image, host=m.host):
                    raise RuntimeError(
                        f"podman image {args.podman_image!r} not found on "
                        f"{m.host.ssh_dest} (machine {m.label}); build with: "
                        f"python3 {(Path(__file__).resolve().parent / 'containers' / 'provision_remote.py')} "
                        f"--ssh {m.host.ssh_dest}"
                    )
                logger.info(
                    "container: ensure network=%s image=%s machine=%s host=%s",
                    args.podman_network or "(default)",
                    args.podman_image,
                    m.label,
                    m.host.ssh_dest,
                )
                if args.podman_network:
                    podman_host.ensure_isolated_network(
                        args.podman_network, host=m.host,
                    )
            podman_repo_specs = podman_repos.build_repo_specs(
                repo_roots, mirror_root=args.podman_mirror_root,
            )
            if args.persist_branches:
                # resolved before the first container opens: every open
                # seeds its fairy remotes
                known = {s.name for s in podman_repo_specs}
                for name in args.persist_repo:
                    if name not in known:
                        logger.warning("--persist-repo %r names no container "
                                       "repo (have: %s); ignored", name,
                                       ", ".join(sorted(known)))
                enabled = set(args.persist_repo) & known \
                    or {podman_repo_specs[0].name}
                persist_repos = [s for s in podman_repo_specs
                                 if s.name in enabled]
                pr_obj = request.get("pull_request")
                if not isinstance(pr_obj, dict):
                    pr_obj = {}
                base_ref = pr_obj.get("base_ref")
                # The SHAs thin the extraction bundles and ride into
                # fixed-shape ssh commands, so only hex survives.
                primary_extra = [s for s in (
                    pr_obj.get("head_sha"),
                    resolve_target_tip(repo_root, base_ref)
                    if repo_root and isinstance(base_ref, str) and base_ref
                    else None,
                ) if isinstance(s, str) and re.fullmatch(r"[0-9a-fA-F]{7,64}", s)]
                persist_base_shas = {
                    spec.name: [spec.head_sha] + (
                        primary_extra if spec is podman_repo_specs[0] else [])
                    for spec in persist_repos
                }
                logger.info("branch persistence active: repos=%s",
                            ", ".join(sorted(enabled)))
            session_commands = [
                substitute_session_command(c, request) for c in args.session_command
            ]
            transcript_session, session_transcript = open_stage_shell(machines[0].label)
            if not args.shared_containers:
                close_machine_shells([transcript_session])

        repo_mount_paths = [s.container_path for s in podman_repo_specs]

        if vector_store_ids:
            request["vector_store_repo_heads"] = build_model_visible_repo_head_map(
                repo_roots,
                indexed_head_map,
                podman_repo_specs=podman_repo_specs,
            )

        target_merge = merge_into_target(request, repo_root) if args.task == "pr" else None
        if target_merge and target_merge.conflicts:
            request["pull_request"]["merge_conflicts"] = target_merge.conflicts

        source_bundle: str | None
        source_files: list[str]
        source_notes: list[str]
        if args.no_source_bundle or args.task == "issue":
            source_bundle = None
            source_files = []
            source_notes = []
        else:
            source_bundle, source_files, source_notes = build_source_bundle(
                request,
                repo_root,
                target_merge,
                max_source_files=args.max_source_files,
                max_file_bytes=args.max_file_bytes,
                max_header_file_bytes=args.max_header_file_bytes,
                max_bundle_bytes=args.max_bundle_bytes,
                include_direct_includes=args.include_direct_includes,
                verbose=args.verbose,
            )

        if args.task == "issue":
            patch_bundle, patch_was_truncated = "", False
        else:
            patch_bundle, patch_was_truncated = build_patch_bundle(patch, args.max_patch_bytes)

        patch_file_id: str | None = None
        if args.task == "pr" and openai_provider_requested:
            patch_file_id = upload_text_file(
                client,
                filename="pull_request.patch.txt",
                text=patch_bundle,
                verbose=args.verbose,
            )
            uploaded_file_ids.append(patch_file_id)

        # Tools and include-spec are shared between the optional triage
        # pre-check and the main reviewer pass; the only per-stage
        # differences are the developer prompt, user content, schema,
        # model, effort, and max_output_tokens.
        tools = build_response_tools(
            vector_store_ids=vector_store_ids,
            file_search_max_num_results=args.file_search_max_num_results,
            use_web_search=args.web_search != "off",
            web_search_context_size=args.web_search_context_size,
            web_search_cache_only=args.web_search == "cached",
            web_search_domains=args.web_search_domain,
            use_podman_shell=args.podman,
            machines=machines,
        )
        include = build_response_include(
            vector_store_ids=vector_store_ids,
            use_web_search=args.web_search != "off",
            use_podman_shell=args.podman,
        )

        review_ctx = ReviewContext(
            request=request,
            patch_text=patch_bundle,
            patch_truncated=patch_was_truncated,
            source_bundle=source_bundle,
            source_files=source_files,
            source_notes=source_notes,
            reviewer_username=reviewer_username,
            ci_triage_mode=ci_triage_active,
            repo_roots=repo_roots,
            repo_mount_paths=repo_mount_paths,
            project_facts=project_facts,
            machines=machines,
            session_transcript=session_transcript,
            open_shell=open_stage_shell if args.podman else None,
            close_shells=close_stage_shells if args.podman else None,
            report_poisoned=report_poisoned if args.podman else None,
            collect_branches=(
                collect_review_branches if args.persist_branches else None),
        )
        openai_resources = OpenAIResources(
            client=client,
            tools=tools,
            include=include,
            patch_file_id=patch_file_id,
            vector_store_ids=vector_store_ids,
            uploaded_file_ids=uploaded_file_ids,
            debug_dir_specified=debug_dir_specified,
        )

        def vet_verdict(review: Review) -> None:
            close_machine_shells(session for session, _ in shared_shells.values())
            workset_note(args, "stage", "vet")
            inherited_branches[:] = review.branches
            vetter = make_reviewer(
                args.vet_model, args=args, resources=openai_resources,
                role=make_vetter_role(review, task=args.task,
                                      inherits_branches=args.persist_branches),
                verbose=args.verbose)
            # A stage that ran PR-derived code may have tampered with the
            # containers it used, so the vetter opens its own.
            vetting = vet_review(vetter, replace(
                review_ctx, source_bundle=None, source_files=[],
                source_notes=[], collect_branches=None,
                open_shell=open_machine_shell if args.podman else None,
                close_shells=close_machine_shells if args.podman else None))
            workset_note(args, "vetting", vetting)

        triage_label_allowlist = triage_label_allowlist_from_request(request)
        requested_models: list[str] = []
        requested_effort: str | None = None
        requested_verbosity: str | None = None

        if args.triage_model:
            triager_role = make_triager_role(
                allowed_models=args.allowed_model,
                allowed_labels=triage_label_allowlist,
                task=args.task,
            )
            triage_ctx = ReviewContext(
                request=request,
                patch_text=patch_bundle,
                patch_truncated=patch_was_truncated,
                source_bundle=None,
                source_files=[],
                source_notes=[],
                reviewer_username=reviewer_username,
                ci_triage_mode=ci_triage_active,
                repo_roots=repo_roots,
                repo_mount_paths=repo_mount_paths,
                project_facts=project_facts,
                machines=machines,
                open_shell=open_stage_shell if args.podman else None,
                close_shells=close_stage_shells if args.podman else None,
                report_poisoned=report_poisoned if args.podman else None,
            )
            triager = make_reviewer(
                args.triage_model,
                args=args,
                resources=openai_resources,
                role=triager_role,
                verbose=args.verbose,
                default_effort=args.triage_reasoning_effort,
                max_output_tokens=args.triage_max_output_tokens,
                service_tier=args.triage_service_tier,
            )
            workset_note(args, "stage", "triage")
            triage_result = run_triage(triager, triage_ctx)
            if triage_result is not None:
                workset_note(args, "triage", triage_result)
            if triage_result is None:
                logger.warning(
                    "triage stage failed; falling through to main reviewer pass"
                )
            else:
                route = triage_result["route"]
                if force_engage and route != "engage":
                    logger.info(
                        "triage route=%s overridden by --force-engage; "
                        "reason=%r; running main reviewer pass",
                        route,
                        triage_result.get("reason", ""),
                    )
                    route = "engage"
                elif route == "skip" and ignore_triage_skip:
                    logger.info(
                        "triage route=skip overridden by the forced review; "
                        "reason=%r; running main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                    route = "engage"
                triage_label_changes = list(triage_result.get("label_changes") or [])
                if triage_label_changes:
                    logger.info("triage label_changes=%r", triage_label_changes)
                if route == "skip":
                    logger.info(
                        "triage route=skip; reason=%r; skipping main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                    # The triager's reason is the only explanation this run
                    # produces; pass it along instead of an empty message so
                    # the operator sees WHY, not just "skip". Skip messages
                    # are never posted to the forge.
                    emit_review_stdout(
                        "skip", f"triager: {triage_result.get('reason') or ''}",
                        label_changes=triage_label_changes,
                    )
                    return 0
                if route == "reply_no_verdict":
                    logger.info(
                        "triage route=reply_no_verdict; reason=%r; skipping main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                    reply = Review(
                        # The issue verdict vocabulary has no verdict
                        # dimension, so its reply class is plain "reply".
                        "reply" if args.task == "issue" else "reply_no_verdict",
                        str(triage_result.get("message") or ""),
                        label_changes=tuple(triage_label_changes),
                        model=triager.name,
                    )
                    if args.vet_model:
                        vet_verdict(reply)
                    emit_review_stdout(reply.classification, reply.message,
                                       label_changes=triage_label_changes)
                    return 0
                if route == "engage":
                    if triage_label_changes:
                        # The reviewer pass owns labels on engage: it sees
                        # the source bundle and does the deep analysis the
                        # triager cannot, so shallow triage guesses are
                        # discarded rather than merged.
                        logger.info("discarding triage label_changes; reviewer pass owns labels")
                    # Apply user-requested model / effort overrides for
                    # the main pass. The triage schema has already
                    # constrained these to the allowlist + effort enum,
                    # so the values are safe to pass through.
                    requested_models = list(triage_result.get("requested_models") or [])
                    if requested_models:
                        logger.info(
                            "main pass model lineup overridden by user request: %r -> %r",
                            [args.model, *args.extra_model], requested_models,
                        )
                    requested_effort = triage_result.get("requested_effort") or None
                    if requested_effort:
                        logger.info(
                            "main pass reasoning effort set by user request: %r",
                            requested_effort,
                        )
                    requested_verbosity = triage_result.get("requested_verbosity") or None
                    if requested_verbosity:
                        logger.info(
                            "final stage verbosity set by user request: %r",
                            requested_verbosity,
                        )
                    logger.info(
                        "triage route=engage; reason=%r; running main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                else:
                    logger.warning(
                        "triage returned unexpected route=%r; falling through to main reviewer pass",
                        route,
                    )

        # The final verdict author owns the labels and --final-verbosity:
        # the combiner whenever one is configured (it runs even on a
        # single draft), else the single reviewer.
        main_specs = requested_models or [args.model, *args.extra_model]
        final_verbosity = requested_verbosity or args.final_verbosity or args.verbosity
        # A user request names models only, so each runs the task's own prompt.
        main_prompts = [None] * len(requested_models) if requested_models else args.main_prompts
        combining = len(main_specs) > 1 or bool(args.combine_model)
        reviewer_labels = [] if combining else triage_label_allowlist
        base_reviewer_role, base_combiner_role = (
            (ISSUE_INVESTIGATOR_ROLE, ISSUE_COMBINER_ROLE) if args.task == "issue"
            else (UNGRADED_REVIEWER_ROLE if combining else REVIEWER_ROLE, COMBINER_ROLE)
        )
        combiner_role = role_with_labels(base_combiner_role, triage_label_allowlist)

        def stage_role(role):
            """The stage's role, branch-enabled when this run persists
            branches."""
            if not args.persist_branches:
                return role
            return role_with_branches(role, [s.name for s in persist_repos])

        main_verbosity = args.verbosity if combining else final_verbosity
        model_reviewers = [
            make_reviewer(spec, args=args, resources=openai_resources,
                          role=stage_role(role_with_labels(review_role(base_reviewer_role, prompt), reviewer_labels)),
                          verbose=args.verbose, default_effort=requested_effort,
                          verbosity=main_verbosity)
            for spec, prompt in zip(main_specs, main_prompts, strict=True)
        ]
        combiner = (
            make_reviewer(args.combine_model, args=args, resources=openai_resources, role=stage_role(combiner_role), verbose=args.verbose, default_effort=requested_effort, verbosity=final_verbosity)
            if args.combine_model
            else None
        )
        if combiner is None and requested_models and len(model_reviewers) > 1:
            # A user may request two models on a deployment configured
            # without --combine-model; combine with the configured main
            # model rather than rejecting the request.
            logger.info(
                "user requested %d models with no --combine-model configured; "
                "combining with %s", len(model_reviewers), args.model,
            )
            combiner = make_reviewer(args.model, args=args, resources=openai_resources, role=stage_role(combiner_role), verbose=args.verbose, verbosity=final_verbosity)
        for reviewer in model_reviewers:
            attach_turn_fallbacks(
                reviewer, args=args, resources=openai_resources,
                verbosity=main_verbosity,
                failure_fails_run=len(model_reviewers) == 1)
        if combiner is not None:
            attach_turn_fallbacks(
                combiner, args=args, resources=openai_resources,
                verbosity=final_verbosity, failure_fails_run=True)
        workset_note(args, "stage", "review")

        def note_drafts(drafts) -> None:
            # the drafts' collected records preload the combiner's fairy
            # remotes AFTER those remotes' baseline seeds were taken, so
            # the combiner verifies the drafts' branches like its own
            # work and re-declares what should survive; open_machine_shell
            # applies the same list to combiner containers that open later
            workset_note_drafts(args, drafts, combining=combiner is not None,
                                failed=review_ctx.failed_reviewers)
            if combiner is not None:
                hand_over_branches(
                    [record for draft in drafts for record in draft.branches])

        try:
            review = review_pr(
                review_ctx, model_reviewers, combiner,
                on_drafts=note_drafts,
            )
        except BadModelOutput:
            return EXIT_BAD_MODEL_OUTPUT
        except ProviderTurnFailed as exc:
            logger.error("giving up, provider ended the turns: %s", exc)
            sys.stdout.write(format_turn_failure(exc) + "\n")
            return EXIT_TURN_FAILED
        except QuotaExhausted as exc:
            logger.error("giving up, every account's quota is exhausted: %s", exc)
            sys.stdout.write(format_quota_hold(str(exc), exc.due_at) + "\n")
            return EXIT_QUOTA_EXHAUSTED
        except Exception:
            if not poisoned_session_ids:
                raise
        # A suspect container taints every draft of this run, including the
        # ones that came back clean: post nothing and let the caller stop
        # rather than re-run the same PR against a fresh container.
        if poisoned_session_ids:
            logger.error(
                "review halted: %d container(s) flagged suspect; the debug "
                "dumps are written, nothing is posted and this PR must not "
                "be retried until they have been inspected",
                len(poisoned_session_ids),
            )
            return EXIT_REVIEW_HALTED

        posting_block = unattended_posting_block(review.branches)
        if posting_block:
            workset_note(args, "vetting",
                         {"hold_for_human_inspection": True, "reason": posting_block})
        elif args.vet_model and (review.classification != "skip" or review.label_changes):
            vet_verdict(review)

        message = review.message
        if review.classification == "skip":
            message = f"{'combiner' if combiner is not None else 'reviewer'}: {message}"
        emit_review_stdout(
            review.classification, message,
            label_changes=list(review.label_changes),
            branches=list(review.branches),
        )
        return 0
    except Exception:
        shell_tool.abort_if_cancelled()
        raise
    finally:
        for file_id in uploaded_file_ids:
            delete_uploaded_file(client, file_id, verbose=args.verbose)
        close_machine_shells(session for _, session in ensemble_shells)


if __name__ == "__main__":
    raise SystemExit(main())
