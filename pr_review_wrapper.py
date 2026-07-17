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
z.ai GLM) and an optional combine stage.

This wrapper is meant to be used as the `--llm-review-cmd` helper described
for the PR auto-approval script. It reads one JSON object from stdin and prints
one JSON object to stdout.

Input JSON keys expected:
  - pull_request: object with metadata (number, title, body, author, html_url,
                  base_ref, head_ref, head_sha, additions, deletions,
                  changed_files, auto_merge)
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
  - message: review/analysis message, may be empty for approve and skip

The wrapper enriches the review with source code context from a local git
checkout. It always includes touched files from pull_request.head_sha when a
source bundle is enabled. Direct quoted include files can also be included
optionally. In addition, the wrapper can optionally build or reuse an OpenAI
vector store containing the repository HEAD as separate file objects and enable
Responses API file_search for additional retrieval.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Sequence

from openai import OpenAI

from common import (
    add_color_arg,
    apply_config_file_defaults,
    setup_logging,
)
from patch_util import (
    extract_changed_paths_from_patch,
    extract_commit_shas_from_patch,
    extract_submodule_changes_from_patch,
    extract_submodule_paths_from_patch,
)
from git_util import git_show_file
import llm_review_api
from llm_review_api import (
    EXIT_BAD_MODEL_OUTPUT,
    BadModelOutput,
    ReviewContext,
)
import review_pipeline
from review_pipeline import make_reviewer, review_pr, run_triage
import podman_host
import podman_repos
import shell_socket
import shell_tool
from llm_prompt import (
    COMBINER_ROLE,
    ISSUE_COMBINER_ROLE,
    ISSUE_INVESTIGATOR_ROLE,
    REVIEWER_ROLE,
    load_project_facts,
    make_triager_role,
    role_with_labels,
)
import openai_common
import openai_container
import openai_container_pool
import openai_reviewer
import openai_vector_store
from openai_common import (
    JsonObject,
    delete_uploaded_file,
    load_api_key,
    log_progress,
    openai_file_exists,
    upload_local_file,
    upload_text_file,
)
from openai_reviewer import (
    EXIT_CONTAINER_UNHEALTHY,
    OpenAIContainerUnhealthy,
    OpenAIResources,
    build_response_include,
    build_response_tools,
    make_openai_http_client,
)


logger = logging.getLogger(__name__)


DEFAULT_PODMAN_IMAGE = "localhost/fairy-review:latest"
# Empty: omit --network so podman uses its default (rootless = pasta).
# A named isolated network is only meaningful once the out-of-band
# egress LAN-block exists; see containers/setup_host.py TODO.
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
        metavar="PROVIDER:MODEL[@EFFORT]",
        help="Reviewer for the main review pass, e.g. 'openai:gpt-5.4' or 'anthropic:claude-opus-4'.",
    )
    p.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Add another reviewer to the ensemble, e.g. 'anthropic:claude-opus-4' "
            "or 'zai:glm-5.2'. Repeat for more. All reviewers (--model plus each "
            "--extra-model) run on the same PR; with more than one "
            "you must pass --combine-model to merge their drafts. '@EFFORT' sets "
            "that reviewer's effort: an OpenAI reasoning effort, or off/low/"
            "medium/high/xhigh/max as the Anthropic/GLM thinking effort."
        ),
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
        "--reasoning-effort",
        choices=["none", "minimal", "low", "medium", "high", "xhigh"],
        help="Reasoning effort for GPT-5/o-series models. If omitted, the model/API default is used.",
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
        help="Reasoning summary level returned by the Responses API (default: auto).",
    )
    p.add_argument(
        "--top-p",
        type=float,
        help="Optional nucleus sampling parameter passed to the Responses API.",
    )
    p.add_argument(
        "--verbosity",
        default="high",
        choices=["low", "medium", "high"],
        help="Optional output verbosity parameter passed to the Responses API.",
    )
    p.add_argument(
        "--max-tool-calls",
        type=int,
        help="Optional maximum number of built-in tool calls allowed in a single response.",
    )
    p.add_argument(
        "--use-web-search",
        action="store_true",
        help="Enable the Responses API web_search tool for live/cached web retrieval.",
    )
    p.add_argument(
        "--use-shell",
        action="store_true",
        help="Enable the Responses API shell tool with an OpenAI-hosted container.",
    )
    p.add_argument(
        "--shell-container-id",
        help="Optional existing OpenAI container id to reuse for the shell tool. Defaults to container_auto.",
    )
    p.add_argument(
        "--use-openai-container-repos",
        action="store_true",
        help="Upload local git directories as plain tar archives, unpack them into a reusable OpenAI container, and point shell/python at that shared container.",
    )
    p.add_argument(
        "--container-id",
        help="Optional existing OpenAI container id to reuse for both shell and code_interpreter when --use-openai-container-repos is enabled.",
    )
    p.add_argument(
        "--container-expiry-minutes",
        type=int,
        default=openai_container.DEFAULT_CONTAINER_EXPIRY_MINUTES,
        help=f"Minutes after last activity before an auto-created OpenAI container expires (default: {openai_container.DEFAULT_CONTAINER_EXPIRY_MINUTES})",
    )
    p.add_argument(
        "--container-memory-limit",
        choices=["1g", "4g", "16g", "64g"],
        default=openai_container.DEFAULT_CONTAINER_MEMORY_LIMIT,
        help=f"Memory limit for an auto-created OpenAI container (default: {openai_container.DEFAULT_CONTAINER_MEMORY_LIMIT})",
    )
    p.add_argument(
        "--container-repos-root",
        default=openai_container.DEFAULT_CONTAINER_REPOS_ROOT,
        help=f"Directory inside the OpenAI container where repository archives are extracted (default: {openai_container.DEFAULT_CONTAINER_REPOS_ROOT})",
    )
    p.add_argument(
        "--podman",
        action="store_true",
        help=(
            "Run shell commands in an ephemeral Podman container on the ssh "
            "podman host (function-calling ''shell'' tool). Repos are filled "
            "in from host-local bare mirrors; no OpenAI code_interpreter or "
            "OpenAI shell. Conflicts with --use-openai-container-repos and "
            "--use-shell."
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
        metavar="[LABEL=]USER@HOST[,cpus=N][,memory=SIZE][,gpu=DEV]",
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
    p.add_argument(
        "--codex-bin",
        default="codex",
        help="codex CLI binary for codex: model specs (default: codex on "
             "PATH). Pin the deployed version; the backend depends on its "
             "flag set.",
    )
    p.add_argument(
        "--codex-timeout-seconds",
        type=float,
        default=0.0,
        help="Kill a codex exec pass after this many seconds; 0 = no "
             "watchdog (default), a pass runs as long as the model works.",
    )
    p.add_argument(
        "--codex-home",
        default=None,
        metavar="DIR",
        help="CODEX_HOME for codex subprocesses (auth.json location). "
             "Default: inherit the environment / codex's ~/.codex.",
    )
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
        "--web-search-cache-only",
        action="store_true",
        help="Disable live external web access and use cached/indexed web search results only.",
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
    return args


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
    container_repo_specs: list[openai_container.ContainerRepoSpec] | None = None,
    podman_repo_specs: list[podman_repos.RepoSpec] | None = None,
) -> dict[str, str]:
    if container_repo_specs:
        # 1. When container repos are available, expose the actual mounted paths the model can use.
        visible: dict[str, str] = {}
        for spec in container_repo_specs:
            head_sha = repo_heads.get(spec.repo_key)
            if isinstance(head_sha, str) and head_sha:
                visible[spec.mounted_path] = head_sha
        return visible

    if podman_repo_specs:
        visible_l: dict[str, str] = {}
        for spec in podman_repo_specs:
            repo_key = str(spec.repo_root.resolve())
            head_sha = repo_heads.get(repo_key)
            if isinstance(head_sha, str) and head_sha:
                visible_l[spec.container_path] = head_sha
        return visible_l

    # 2. Without container mounts, expose simple repo labels instead of unusable host-local paths.
    visible = {}
    for repo_root in repo_roots:
        repo_key = str(repo_root.resolve())
        head_sha = repo_heads.get(repo_key)
        if not isinstance(head_sha, str) or not head_sha:
            continue
        label = re.sub(r"[^A-Za-z0-9._-]", "-", repo_root.name).strip("._-") or "repo"
        visible[label] = head_sha
    return visible


def parse_source_bundle_request_info(request: JsonObject) -> tuple[JsonObject, str, list[str], str]:
    pr = request.get("pull_request")
    if not isinstance(pr, dict):
        raise RuntimeError("missing pull_request object")

    patch = request.get("patch")
    if not isinstance(patch, str):
        patch = ""

    changed_paths = extract_changed_paths_from_patch(patch)
    if not changed_paths:
        raise RuntimeError("no changed file paths found in patch")

    head_sha = pr.get("head_sha") if isinstance(pr.get("head_sha"), str) else None
    if not head_sha:
        raise RuntimeError("missing pull_request.head_sha")

    return pr, patch, changed_paths, head_sha


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
            raise RuntimeError(f"missing source for {relpath} in patch commit list from {head_sha}")
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
    source_revisions: dict[str, str],
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
        source_from = f"git show {source_revisions[relpath]}:{relpath}"

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


def build_source_bundle(
    request: JsonObject,
    repo_root: Path | None,
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

    _pr, patch, changed_paths, head_sha = parse_source_bundle_request_info(request)

    bundle_parts: list[str] = []
    notes: list[str] = []
    total_bytes = 0

    header = (
        f"Repository: {repo_root.name}\n"
        f"Head SHA: {head_sha}\n"
        f"Changed files seen in patch: {len(changed_paths)}\n\n"
    )
    bundle_parts.append(header)
    total_bytes += len(header.encode("utf-8"))

    direct_paths = changed_paths[:max_source_files]
    commit_shas = extract_commit_shas_from_patch(patch) or [head_sha]
    ordered_paths, loaded_texts, source_revisions, include_notes = load_source_bundle_texts(
        repo_root,
        head_sha,
        direct_paths,
        commit_shas,
        include_direct_includes=include_direct_includes,
    )
    notes.extend(include_notes)

    used_paths, truncated_paths, total_bytes = append_source_bundle_files(
        ordered_paths,
        loaded_texts,
        source_revisions=source_revisions,
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


def emit_review_stdout(
    classification: str,
    message: str,
    *,
    label_changes: list[dict[str, object]] | None = None,
) -> None:
    out: dict[str, object] = {"classification": classification, "message": message}
    if label_changes:
        out["label_changes"] = label_changes
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

    One isolated container per call, so concurrent ensemble reviewers never
    share a working tree. ``session_commands`` run in every container so
    state they create exists for each reviewer; the returned transcript is
    what the prompt splices in. On any provisioning failure the
    half-started container is stopped before the error propagates.
    """
    handle = podman_host.start_ephemeral_container(
        image=args.podman_image,
        host=spec.host,
        network=args.podman_network,
        memory=spec.memory,
        cpus=spec.cpus,
        extra_args=(f"--device={spec.gpu}",) if spec.gpu else (),
    )
    try:
        podman_repos.provision_repos_into_container(
            handle, repo_specs, spec.host,
            prune_refs_after=(
                int(datetime.fromisoformat(args.simulate_past_cutoff).timestamp())
                if args.simulate_past_cutoff else None
            ),
        )
        podman_host.copy_into_container(handle, AGENT_LOCAL_PATH, AGENT_CONTAINER_DIR)
        session = podman_host.open_container_shell(handle, AGENT_CONTAINER_PATH)
        transcript = shell_tool.run_session_commands(
            session, list(session_commands),
            max_timeout_s=args.podman_exec_timeout,
        ) if session_commands else ""
    except Exception:
        podman_host.stop_container(handle)
        raise
    return handle, session, transcript


def main() -> int:
    args = parse_args()
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
        llm_review_api.logger,
        openai_common.logger,
        openai_reviewer.logger,
        openai_vector_store.logger,
        review_pipeline.logger,
        openai_container.logger,
        openai_container_pool.logger,
        podman_host.logger,
        podman_repos.logger,
        # By name, not module attribute: the anthropic modules are imported
        # lazily (only when an anthropic/zai model is requested) and fetching
        # a logger from the registry does not import the module.
        logging.getLogger("shell_tool"),
        logging.getLogger("anthropic_common"),
        logging.getLogger("anthropic_reviewer"),
        color=args.color,
    )

    # The OpenAI client -- and therefore OPENAI_API_KEY -- is only needed
    # when an OpenAI backend actually runs: an ``openai:`` model in any slot
    # (including one the triager may pick from ``--allowed-model``), or one
    # of the OpenAI-hosted subsystems (container repos, vector-store
    # search/prepare). A codex:- or anthropic:-only run must not require the
    # key. Same spec set the codex-socket gate below scans.
    model_specs = [
        s for s in (args.model, *args.extra_model, args.triage_model,
                    args.combine_model, *args.allowed_model)
        if s
    ]
    openai_provider_requested = any(s.startswith("openai:") for s in model_specs)
    openai_needed = (
        openai_provider_requested
        or args.use_openai_container_repos
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
    if args.container_id and not args.use_openai_container_repos:
        raise RuntimeError("--container-id requires --use-openai-container-repos")
    if args.use_openai_container_repos and args.shell_container_id and args.container_id and args.shell_container_id != args.container_id:
        raise RuntimeError("--shell-container-id and --container-id must match when both are set")
    if args.podman:
        if args.use_openai_container_repos:
            raise RuntimeError("--podman cannot be combined with --use-openai-container-repos")
        if args.use_shell:
            raise RuntimeError("--podman cannot be combined with --use-shell")
        if args.container_id or args.shell_container_id:
            raise RuntimeError(
                "--podman does not use OpenAI container ids; "
                "omit --container-id and --shell-container-id"
            )

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
    # Cross-process contract with fairy's --force-review-skip:
    # run the full reviewer pass even when triage votes ``skip``.
    ignore_triage_skip = bool(request.get("ignore_triage_skip"))
    # Cross-process contract with fairy's --force-engage: run the
    # full reviewer pass regardless of the triage route (skip / reply_no_verdict)
    # and even when triage itself fails on a CI-red request.
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

    shared_container_id: str | None = None
    container_repo_specs: list[openai_container.ContainerRepoSpec] = []
    container_lease: openai_container.ContainerLease | None = None
    container_lease_healthy = True
    if args.use_openai_container_repos:
        if not repo_roots:
            raise RuntimeError("--use-openai-container-repos requires at least one local git checkout")
        container_lease = openai_container.ensure_openai_container_repos_ready(
            client,
            repo_roots,
            explicit_container_id=args.container_id or args.shell_container_id,
            container_repos_root=args.container_repos_root,
            expiry_minutes=args.container_expiry_minutes,
            memory_limit=args.container_memory_limit,
            verbose=args.verbose,
        )
        shared_container_id = container_lease.container_id
        container_repo_specs = container_lease.specs

    session_commands: list[str] = []
    session_transcript = ""
    podman_repo_specs: list[podman_repos.RepoSpec] = []
    machines: list[podman_host.ShellHostSpec] = args.machines if args.podman else []
    spec_by_label = {m.label: m for m in machines}
    # Every opened container+session, default and lazily instantiated alike
    # (one isolated container per machine per non-OpenAI reviewer);
    # released in finally.
    ensemble_shells: list[tuple[podman_host.ContainerHandle, podman_host.ContainerShellSession]] = []
    # Live sessions shared by all OpenAI role passes, keyed by machine label.
    primary_shells: dict[str, podman_host.ContainerShellSession] = {}

    def open_machine_shell(
        label: str,
    ) -> tuple[podman_host.ContainerShellSession, str]:
        handle, session, transcript = open_review_container_shell(
            spec_by_label[label], podman_repo_specs, args, session_commands,
        )
        ensemble_shells.append((handle, session))
        return session, transcript

    uploaded_file_ids: list[str] = []
    codex_shell_server: shell_socket.ShellDispatchServer | None = None
    # The eager container open below and everything after runs under
    # this try so the finally releases ensemble_shells (and uploads)
    # even when e.g. bundle building fails between open and review.
    try:
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
            session_commands = [
                substitute_session_command(c, request) for c in args.session_command
            ]
            primary_shells[machines[0].label], session_transcript = (
                open_machine_shell(machines[0].label)
            )

        repo_mount_paths = (
            [s.container_path for s in podman_repo_specs]
            if podman_repo_specs
            else [spec.mounted_path for spec in container_repo_specs]
        )

        if vector_store_ids:
            head_map_kwargs: dict[str, object] = {}
            if container_repo_specs:
                head_map_kwargs["container_repo_specs"] = container_repo_specs
            if podman_repo_specs:
                head_map_kwargs["podman_repo_specs"] = podman_repo_specs
            request["vector_store_repo_heads"] = build_model_visible_repo_head_map(
                repo_roots,
                indexed_head_map,
                **head_map_kwargs,
            )

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
            use_web_search=args.use_web_search,
            web_search_context_size=args.web_search_context_size,
            web_search_cache_only=args.web_search_cache_only,
            web_search_domains=args.web_search_domain,
            use_shell=args.use_shell,
            shell_container_id=shared_container_id if shared_container_id else args.shell_container_id,
            code_interpreter_container_id=None if args.podman else shared_container_id,
            use_podman_shell=args.podman,
            machines=machines,
        )
        include = build_response_include(
            vector_store_ids=vector_store_ids,
            use_web_search=args.use_web_search,
            use_podman_shell=args.podman,
        )

        # The codex backend's MCP bridge runs outside this process (codex
        # spawns it), so its shell tool calls come back over a local unix
        # socket. Started only when a codex: spec can actually run --
        # including via a triage model request from the allowlist.
        codex_specs = [
            s for s in (args.model, *args.extra_model, args.triage_model,
                        args.combine_model, *args.allowed_model)
            if s and s.startswith("codex:")
        ]
        if codex_specs and args.podman:
            codex_shell_server = shell_socket.ShellDispatchServer(
                machine_labels=[m.label for m in machines],
                open_shell=open_machine_shell,
                max_timeout_s=args.podman_exec_timeout,
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
            open_shell=open_machine_shell if args.podman else None,
            shell_socket_path=(
                codex_shell_server.socket_path if codex_shell_server else None
            ),
        )
        openai_resources = OpenAIResources(
            client=client,
            tools=tools,
            include=include,
            patch_file_id=patch_file_id,
            vector_store_ids=vector_store_ids,
            shared_container_id=shared_container_id,
            shells=primary_shells if args.podman else None,
            open_shell=open_machine_shell if args.podman else None,
            uploaded_file_ids=uploaded_file_ids,
            debug_dir_specified=debug_dir_specified,
        )

        triage_label_allowlist = triage_label_allowlist_from_request(request)
        requested_models: list[str] = []

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
                open_shell=open_machine_shell if args.podman else None,
                shell_socket_path=(
                    codex_shell_server.socket_path if codex_shell_server else None
                ),
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
            triage_result = run_triage(triager, triage_ctx)
            if triage_result is None:
                if ci_triage_active and not force_engage:
                    logger.warning(
                        "triage stage failed on CI triage request; skipping main "
                        "reviewer pass (unsafe to run full review when CI is red)"
                    )
                    emit_review_stdout("skip", "")
                    return 0
                logger.warning(
                    "triage stage failed; falling through to main reviewer pass"
                )
            elif triage_result.get("container_unhealthy"):
                # Container died during triage; no point retrying main on the same dead
                # container. Surface EXIT_CONTAINER_UNHEALTHY so the caller retries the
                # whole wrapper against a freshly provisioned container.
                container_lease_healthy = False
                logger.warning(
                    "openai container unhealthy during triage; "
                    "exiting with code %d so caller retries against a fresh container",
                    EXIT_CONTAINER_UNHEALTHY,
                )
                return EXIT_CONTAINER_UNHEALTHY
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
                        "triage route=skip overridden by --force-review-skip; "
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
                    emit_review_stdout("skip", "", label_changes=triage_label_changes)
                    return 0
                if route == "reply_no_verdict":
                    logger.info(
                        "triage route=reply_no_verdict; reason=%r; skipping main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                    emit_review_stdout(
                        # The issue verdict vocabulary has no verdict
                        # dimension, so its reply class is plain "reply".
                        "reply" if args.task == "issue" else "reply_no_verdict",
                        str(triage_result.get("message") or ""),
                        label_changes=triage_label_changes,
                    )
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
                    requested_effort = triage_result.get("requested_effort")
                    if requested_effort:
                        logger.info(
                            "main pass reasoning_effort overridden by user request: %r -> %r",
                            args.reasoning_effort, requested_effort,
                        )
                        args.reasoning_effort = requested_effort
                    logger.info(
                        "triage route=engage; reason=%r; running main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                else:
                    logger.warning(
                        "triage returned unexpected route=%r; falling through to main reviewer pass",
                        route,
                    )

        # The final verdict author owns the labels: the combiner whenever
        # one is configured (it runs even on a single draft), else the
        # single reviewer.
        n_reviewers = len(requested_models) if requested_models else 1 + len(args.extra_model)
        reviewer_labels = [] if n_reviewers > 1 or args.combine_model else triage_label_allowlist
        base_reviewer_role, base_combiner_role = (
            (ISSUE_INVESTIGATOR_ROLE, ISSUE_COMBINER_ROLE) if args.task == "issue"
            else (REVIEWER_ROLE, COMBINER_ROLE)
        )
        reviewer_role = role_with_labels(base_reviewer_role, reviewer_labels)
        combiner_role = role_with_labels(base_combiner_role, triage_label_allowlist)

        if requested_models:
            model_reviewers = [
                make_reviewer(spec, args=args, resources=openai_resources, role=reviewer_role, verbose=args.verbose)
                for spec in requested_models
            ]
        else:
            model_reviewers = [
                make_reviewer(args.model, args=args, resources=openai_resources, role=reviewer_role, verbose=args.verbose)
            ]
            for spec in args.extra_model:
                model_reviewers.append(
                    make_reviewer(spec, args=args, resources=openai_resources, role=reviewer_role, verbose=args.verbose)
                )
        combiner = (
            make_reviewer(args.combine_model, args=args, resources=openai_resources, role=combiner_role, verbose=args.verbose)
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
            combiner = make_reviewer(args.model, args=args, resources=openai_resources, role=combiner_role, verbose=args.verbose)
        try:
            review = review_pr(review_ctx, model_reviewers, combiner)
        except OpenAIContainerUnhealthy:
            # The outer ``finally`` still releases the lease; mark it
            # unhealthy so the dead container is dropped from the pool.
            container_lease_healthy = False
            return EXIT_CONTAINER_UNHEALTHY
        except BadModelOutput:
            return EXIT_BAD_MODEL_OUTPUT

        emit_review_stdout(
            review.classification, review.message,
            label_changes=list(review.label_changes),
        )
        return 0
    finally:
        if codex_shell_server is not None:
            codex_shell_server.close()
        for file_id in uploaded_file_ids:
            delete_uploaded_file(client, file_id, verbose=args.verbose)
        if container_lease is not None:
            container_lease.release(healthy=container_lease_healthy)
        for handle, session in ensemble_shells:
            session.close()
            podman_host.stop_container(handle)


if __name__ == "__main__":
    raise SystemExit(main())
