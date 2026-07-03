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

Review a pull request with the OpenAI API.

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

Output JSON keys:
  - classification: one of
      ok_approve
      minor_issues_approve
      moderate_issues_comment
      major_request_changes
      skip
  - message: short review message, may be empty for ok_approve

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
from dataclasses import dataclass
import json
import logging
import os
import re
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import httpx
from openai import BadRequestError, DefaultHttpxClient, OpenAI

from common import (
    add_color_arg,
    dump_response_debug_artifacts,
    response_to_debug_json,
    setup_logging,
)
from patch_util import (
    extract_changed_paths_from_patch,
    extract_commit_shas_from_patch,
    extract_submodule_changes_from_patch,
    extract_submodule_paths_from_patch,
)
from git_util import git_show_file
from llm_review_api import (
    CLASSIFICATIONS,
    ENGAGE,
    REVIEW_SCHEMA,
    TERMINAL_ROUTES,
    Z_AI_ANTHROPIC_URL,
    BadModelOutput,
    Review,
    ReviewContext,
    Reviewer,
    SchemaError,
    check_schema,
    run_parallel,
    validate_review,
)
import podman_host
import podman_repos
from shell_tool import exec_shell_call
from llm_prompt import (
    TRIAGE_REQUESTABLE_EFFORTS,
    generate_llm_prompt,
    make_combiner_user_text,
    make_developer_prompt,
    make_triage_developer_prompt,
    make_triage_user_text,
    make_user_text,
    t_prompt_triage_labels,
    t_prompt_user_request,
)
import openai_common
import openai_container
import openai_container_pool
import openai_vector_store
from openai_common import (
    InputContentItem,
    JsonObject,
    ResponseKwargs,
    _obj_get,
    call_with_rate_limit_retry,
    delete_uploaded_file,
    extract_response_text,
    load_api_key,
    log_progress,
    openai_file_exists,
    upload_local_file,
    upload_text_file,
)


logger = logging.getLogger(__name__)


# Distinct non-zero exit code the wrapper uses when it recognizes the
# attached OpenAI container as unhealthy (expired, not running, ...).
# The caller (e.g. fairy.py) just retries on any non-zero exit,
# but a dedicated code makes these routine, retryable failures
# trivially greppable in logs instead of indistinguishable from a crash.
EXIT_CONTAINER_UNHEALTHY = 2

# Distinct non-zero exit code for "the model's JSON did not match the
# schema we requested" (see SchemaError). Same retry behavior as above;
# a dedicated code keeps these model flakes greppable and distinct from
# a real crash.
EXIT_BAD_MODEL_OUTPUT = 3

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


def _build_remote_host(args: argparse.Namespace) -> podman_host.RemoteHost:
    """The ssh account where podman runs and the containers live; every
    podman call and the persistent shell channel go through it."""
    return podman_host.RemoteHost(args.podman_ssh_dest, identity=args.podman_ssh_identity)


# Match a well-formed unresolved citation marker:
#   <E200> (filecite|cite) <E202> <id-bytes...> <E201>
#
# The negated character class includes BOTH closing sentinel \uE201 AND
# opening sentinel \uE200. Including \uE200 means a truncated marker that
# is missing its own \uE201 cannot greedily reach across a subsequent
# well-formed marker's opener and consume both the intervening prose and
# the next marker as its "content". With \uE200 in the negated set, such
# a truncated marker simply fails to match and its orphan sentinel chars
# are then handled by ORPHAN_PUA_SENTINEL_RE below.
UNRESOLVED_CITATION_RE = re.compile(
    "\\uE200(?:filecite|cite)\\uE202[^\\uE200\\uE201]*\\uE201"
)

# Catches stray PUA sentinel characters left over after the well-formed
# strip pass: e.g. truncated/corrupted markers that did not match
# UNRESOLVED_CITATION_RE. Stripping these prevents PUA chars from leaking
# into the user-visible comment, and their presence is logged as a
# warning so genuine upstream corruption is visible in operator logs.
ORPHAN_PUA_SENTINEL_RE = re.compile("[\uE200-\uE202]")


DEFAULT_MODEL = "gpt-5.4-mini"
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

# TCP keepalive for every OpenAI HTTP connection. The main
# ``responses.create`` call is non-streaming, so zero bytes arriving for
# tens of minutes is normal and no read timeout can distinguish a slow
# review from a connection that died without a RST (NAT/conntrack or
# another middlebox reaping the mapping). Such dead sockets blocked
# ``recv()`` forever, hanging reviews until the outer ``--llm-timeout``
# (seen in production 2026-05-07 on three concurrent reviews, and
# repeatedly under --simulate-past). Probing after 60s idle keeps the
# mapping alive in the first place, and 8 unanswered probes 30s apart
# surface a genuinely dead peer as a connection error within ~5
# minutes, which the outer caller's existing retry handles.
OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS = [
    (socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1),
    (socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 60),
    (socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 30),
    (socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 8),
]


def make_openai_http_client() -> DefaultHttpxClient:
    """SDK-default httpx client, plus TCP keepalive on every socket."""
    return DefaultHttpxClient(
        transport=httpx.HTTPTransport(
            socket_options=OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS,
        ),
    )

TRIAGE_ROUTES = (*TERMINAL_ROUTES, ENGAGE)


def triage_label_allowlist_from_request(request: JsonObject) -> list[str]:
    raw = request.get("triage_label_allowlist")
    if not isinstance(raw, list):
        return []
    return [label for label in raw if isinstance(label, str) and label]


def sanitize_label_changes(
    raw: object,
    allowed_labels: list[str],
) -> list[dict[str, object]]:
    """Validate the triager's ``label_changes`` against the allowlist.

    Each kept item is ``{label, op, reason, post}``: ``label`` must be in
    the allowlist, ``op`` must be ``add``/``remove``, ``reason`` is a
    string (empty if missing), and ``post`` is a bool (False = log-only).
    Duplicate ``(label, op)`` pairs are dropped. ``raw`` is attacker-
    adjacent (model output crossing the process boundary), so every field
    is checked here at the boundary.
    """
    allowed = set(allowed_labels)
    if not allowed or not isinstance(raw, list):
        return []
    out: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for item in raw:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        op = item.get("op")
        if not isinstance(label, str) or label not in allowed:
            logger.warning("triage dropped label change with unknown label: %r", label)
            continue
        if op not in ("add", "remove"):
            logger.warning("triage dropped label change %r with bad op: %r", label, op)
            continue
        if (label, op) in seen:
            continue
        seen.add((label, op))
        reason = item.get("reason")
        out.append({
            "label": label,
            "op": op,
            "reason": reason if isinstance(reason, str) else "",
            "post": bool(item.get("post")),
        })
    return out


def build_triage_schema(
    allowed_models: list[str],
    allowed_labels: list[str] | None = None,
) -> dict[str, object]:
    """Triage JSON schema; the override fields appear only if enabled.

    When ``allowed_models`` is non-empty the schema gains
    ``requested_models`` (array, each entry enum-constrained to the
    allowlist; at most two are honored) and ``requested_effort``
    (nullable, enum-constrained to ``TRIAGE_REQUESTABLE_EFFORTS``).
    Strict mode enforces the enums on the wire so
    ``validate_triage_result`` does not need to re-check the values.
    """
    properties: dict[str, object] = {
        "route": {
            "type": "string",
            "description": (
                "Triage decision: skip (no new useful action now), "
                "helpful_reply (short direct reply suffices), "
                "engage (run a full reviewer pass)."
            ),
            "enum": list(TRIAGE_ROUTES),
        },
        "message": {
            "type": "string",
            "description": (
                "Markdown comment body to post to Forgejo. "
                "Must be non-empty for helpful_reply. "
                "Must be empty for skip and engage. "
                "Do not include HTML or markdown fences."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                "Short internal explanation of why this route was chosen. "
                "Used for logs only; not shown to anyone."
            ),
        },
    }
    required = ["route", "message", "reason"]
    if allowed_models:
        properties["requested_models"] = {
            "type": "array",
            "maxItems": 2,
            "items": {"type": "string", "enum": list(allowed_models)},
            "description": (
                "Models from an explicit user request, in request order "
                "(at most two), else empty."
            ),
        }
        properties["requested_effort"] = {
            "type": ["string", "null"],
            "enum": [None, *TRIAGE_REQUESTABLE_EFFORTS],
            "description": "Reasoning effort from an explicit user request, else null.",
        }
        required.extend(["requested_models", "requested_effort"])
    label_allowlist = allowed_labels or []
    if label_allowlist:
        properties["label_changes"] = {
            "type": "array",
            "description": (
                "Per-label add/remove changes for the PR; empty when no "
                "label should change."
            ),
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "label": {"type": "string", "enum": label_allowlist},
                    "op": {"type": "string", "enum": ["add", "remove"]},
                    "reason": {
                        "type": "string",
                        "description": "One concrete sentence justifying this label change.",
                    },
                    "post": {
                        "type": "boolean",
                        "description": (
                            "true if the reason is needed to understand the "
                            "label and should be posted to the PR as a comment; "
                            "false if it only serves logs."
                        ),
                    },
                },
                "required": ["label", "op", "reason", "post"],
            },
        }
        required.append("label_changes")
    return {
        "name": "pr_triage_result",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": properties,
            "required": required,
        },
    }

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*([<"])([^>"]+)[>"]', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Review a PR with the OpenAI Responses API.")
    p.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenAI model for the main review pass (default: {DEFAULT_MODEL})")
    p.add_argument(
        "--extra-model",
        action="append",
        default=[],
        metavar="PROVIDER:MODEL[@EFFORT]",
        help=(
            "Add another reviewer to the ensemble, e.g. 'anthropic:claude-opus-4' "
            "or 'zai:glm-4.6'. Repeat for more. All reviewers (the --model OpenAI "
            "pass plus each --extra-model) run on the same PR; with more than one "
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
        help=(
            "Optional smaller/cheaper model to run a pre-check that classifies the PR "
            "as skip / helpful_reply / engage before the main review model is invoked. "
            "When unset (default), triage is disabled and the main model is called directly. "
            "The triage model gets the same tool access (file_search / web_search / "
            "code_interpreter / shell) as the main model, but is NOT given the source bundle; "
            "the source bundle is only uploaded if triage routes to engage."
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
        metavar="[PROVIDER:]NAME",
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
        "--podman-memory",
        default=podman_host.CONTAINER_MEMORY,
        help="Memory limit passed to podman run (default: %(default)s). "
             "The prompt advertises the default, so prefer changing "
             "podman_host.CONTAINER_MEMORY over this flag.",
    )
    p.add_argument(
        "--podman-cpus",
        default=podman_host.CONTAINER_CPUS,
        help="CPU limit passed to podman run (default: %(default)s). "
             "The prompt advertises the default, so prefer changing "
             "podman_host.CONTAINER_CPUS over this flag.",
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
        "--podman-ssh-dest",
        metavar="USER@HOST",
        default=None,
        help=(
            "ssh destination of the podman host (anything ssh accepts, "
            "including a ~/.ssh/config alias). Required with --podman: all "
            "podman calls run as 'ssh DEST podman ...', repos are kept as "
            "bare mirrors on the host and filled into the container "
            "host-locally, so no full .git crosses the wire per review. "
            "Provision the host with containers/provision_remote.py."
        ),
    )
    p.add_argument(
        "--podman-ssh-identity",
        metavar="KEYFILE",
        default=None,
        help=(
            "ssh identity (private key) for --podman-ssh-dest. Optional: "
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
    return p.parse_args()


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



def format_response_stats(response: object, *, elapsed_seconds: float | None = None) -> str:
    dumped = response_to_debug_json(response)
    parts: list[str] = []

    if elapsed_seconds is not None:
        parts.append(f"dt={elapsed_seconds:.3f}s")

    # Echo the actual model the API reports back so the operator can
    # confirm at a glance which tier/family ran (the ``start`` line
    # logs the *requested* model, this one reflects what the server
    # returned in case of any aliasing or fallback).
    response_model = dumped.get("model") if isinstance(dumped, dict) else None
    if isinstance(response_model, str) and response_model:
        parts.append(f"model={response_model}")

    response_id = dumped.get("id") if isinstance(dumped, dict) else None
    if isinstance(response_id, str) and response_id:
        parts.append(f"id={response_id}")

    response_status = dumped.get("status") if isinstance(dumped, dict) else None
    if isinstance(response_status, str) and response_status:
        parts.append(f"status={response_status}")

    service_tier = dumped.get("service_tier") if isinstance(dumped, dict) else None
    if isinstance(service_tier, str) and service_tier:
        parts.append(f"tier={service_tier}")

    usage = dumped.get("usage") if isinstance(dumped, dict) else None
    if isinstance(usage, dict):
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        total_tokens = usage.get("total_tokens")
        input_details = usage.get("input_tokens_details")
        output_details = usage.get("output_tokens_details")
        cached_tokens = input_details.get("cached_tokens") if isinstance(input_details, dict) else None
        reasoning_tokens = output_details.get("reasoning_tokens") if isinstance(output_details, dict) else None
        if isinstance(input_tokens, int):
            parts.append(f"in={input_tokens}")
        if isinstance(cached_tokens, int):
            parts.append(f"cache={cached_tokens}")
        if isinstance(output_tokens, int):
            parts.append(f"out={output_tokens}")
        if isinstance(reasoning_tokens, int):
            parts.append(f"reason={reasoning_tokens}")
        if isinstance(total_tokens, int):
            parts.append(f"total={total_tokens}")

    output = dumped.get("output") if isinstance(dumped, dict) else None
    if isinstance(output, list):
        parts.append(f"items={len(output)}")
        item_counts: dict[str, int] = {}
        file_search_calls = 0
        file_search_results = 0
        web_search_calls = 0
        web_search_sources = 0
        code_interpreter_calls = 0
        code_interpreter_outputs = 0
        shell_calls = 0
        shell_outputs = 0
        function_calls = 0
        function_outputs = 0
        for item in output:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if isinstance(item_type, str) and item_type:
                item_counts[item_type] = item_counts.get(item_type, 0) + 1
            if item_type == "file_search_call":
                file_search_calls += 1
                results = item.get("results")
                if isinstance(results, list):
                    file_search_results += len(results)
            elif item_type == "web_search_call":
                web_search_calls += 1
                action = item.get("action")
                sources = action.get("sources") if isinstance(action, dict) else None
                if isinstance(sources, list):
                    web_search_sources += len(sources)
            elif item_type == "code_interpreter_call":
                code_interpreter_calls += 1
                outputs = item.get("outputs")
                if isinstance(outputs, list):
                    code_interpreter_outputs += len(outputs)
            elif item_type == "shell_call":
                shell_calls += 1
            elif item_type == "shell_call_output":
                shell_outputs += 1
            elif item_type == "function_call":
                function_calls += 1
            elif item_type == "function_call_output":
                function_outputs += 1
        if item_counts:
            item_counts_text = ",".join(f"{name}:{item_counts[name]}" for name in sorted(item_counts))
            parts.append(f"item_types={item_counts_text}")
        tool_parts: list[str] = []
        if file_search_calls:
            tool_parts.append(f"fs:{file_search_calls}/{file_search_results}")
        if web_search_calls:
            tool_parts.append(f"ws:{web_search_calls}/{web_search_sources}")
        if code_interpreter_calls:
            tool_parts.append(f"ci:{code_interpreter_calls}/{code_interpreter_outputs}")
        if shell_calls or shell_outputs:
            tool_parts.append(f"sh:{shell_calls}/{shell_outputs}")
        if function_calls or function_outputs:
            tool_parts.append(f"fn:{function_calls}/{function_outputs}")
        if tool_parts:
            parts.append(f"tools={','.join(tool_parts)}")

    return " ".join(parts)


def build_podman_shell_function_tool() -> JsonObject:
    return {
        "type": "function",
        "name": "shell",
        "description": (
            "Run one shell command inside the ephemeral review container "
            "(full working-tree repos under /work/...). Command runs via "
            "``sh -c`` with an in-container timeout."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command (``sh -c``).",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory inside the container (e.g. /work/ffmpeg).",
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Max wall seconds for this command (capped by the wrapper).",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        },
        "strict": False,
    }


def extract_function_calls_from_response(response: object) -> list[dict[str, str]]:
    dumped = response_to_debug_json(response)
    output = dumped.get("output")
    if not isinstance(output, list):
        return []
    calls: list[dict[str, str]] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "function_call":
            continue
        name = item.get("name")
        call_id = item.get("call_id") or item.get("id")
        arguments = item.get("arguments")
        if not isinstance(name, str) or not isinstance(call_id, str):
            continue
        if not isinstance(arguments, str):
            continue
        calls.append({"name": name, "call_id": call_id, "arguments": arguments})
    return calls


def run_responses_resolving_podman_shell(
    client: OpenAI,
    *,
    initial_kwargs: ResponseKwargs,
    podman_shell_session: podman_host.ContainerShellSession,
    max_tool_rounds: int,
    max_shell_timeout_s: float,
    what: str,
    verbose: bool,
    debug_dir: str | None = None,
    wrapper_request: JsonObject | None = None,
) -> object:
    """Drive ``responses.create`` in a loop until no pending function calls.

    Dispatches ``name=shell`` via the persistent
    :class:`podman_host.ContainerShellSession`; other function names receive
    an error ``function_call_output`` so the model can recover.

    ``max_tool_rounds <= 0`` means no cap on the number of rounds.

    With ``debug_dir`` set, EVERY round's response is dumped (paired with
    the exact kwargs that produced it), not just the final one -- the
    intermediate rounds are where the function calls and their outputs
    live, and each is a separately billed request.
    """
    def create_and_dump(kwargs: ResponseKwargs, what_label: str) -> object:
        resp = call_with_rate_limit_retry(
            lambda: client.responses.create(**kwargs),
            what=what_label,
            verbose=verbose,
            retry_transient=False,
        )
        if debug_dir:
            dump_response_debug_artifacts(
                resp, kwargs, wrapper_request=wrapper_request,
                debug_dir=debug_dir, verbose=verbose,
            )
        return resp

    response = create_and_dump(initial_kwargs, what)
    rounds = 0
    while True:
        pending = extract_function_calls_from_response(response)
        if not pending:
            return response
        rounds += 1
        if max_tool_rounds > 0 and rounds > max_tool_rounds:
            raise RuntimeError(
                f"{what}: exceeded podman shell function-call limit ({max_tool_rounds})"
            )
        output_items: list[JsonObject] = []
        for call in pending:
            if call["name"] != "shell":
                payload = json.dumps(
                    {"error": f"unsupported function {call['name']!r}"},
                    ensure_ascii=False,
                )
                output_items.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": payload,
                })
                continue
            try:
                args_obj: object = json.loads(call["arguments"])
            except json.JSONDecodeError as exc:
                output_items.append({
                    "type": "function_call_output",
                    "call_id": call["call_id"],
                    "output": json.dumps(
                        {"error": "invalid JSON in arguments", "detail": str(exc)},
                        ensure_ascii=False,
                    ),
                })
                continue
            payload_obj = exec_shell_call(
                podman_shell_session, args_obj, max_timeout_s=max_shell_timeout_s,
            )
            output_items.append({
                "type": "function_call_output",
                "call_id": call["call_id"],
                "output": json.dumps(payload_obj, ensure_ascii=False),
            })
        rid = getattr(response, "id", None)
        if not isinstance(rid, str) or not rid:
            dumped_rid = response_to_debug_json(response)
            rid2 = dumped_rid.get("id")
            rid = rid2 if isinstance(rid2, str) else None
        if not isinstance(rid, str) or not rid:
            raise RuntimeError(f"{what}: response missing id for tool follow-up")
        model = initial_kwargs.get("model")
        if not isinstance(model, str):
            raise RuntimeError(f"{what}: initial kwargs missing model string")
        follow: ResponseKwargs = {
            "model": model,
            "previous_response_id": rid,
            "input": output_items,
            "parallel_tool_calls": False,
        }
        tools = initial_kwargs.get("tools")
        if tools is not None:
            follow["tools"] = tools
        st = initial_kwargs.get("service_tier")
        if st is not None:
            follow["service_tier"] = st
        mtc = initial_kwargs.get("max_tool_calls")
        if mtc is not None:
            follow["max_tool_calls"] = mtc
        # ``text`` (the json_schema output format) is per-request and NOT
        # inherited via previous_response_id; without it a follow-up round
        # can answer off-schema (seen: {"class": ...} instead of
        # {"classification": ...}, failing the whole review attempt).
        txt = initial_kwargs.get("text")
        if txt is not None:
            follow["text"] = txt
        response = create_and_dump(follow, f"{what} (podman shell follow-up)")


def build_response_tools(
    *,
    vector_store_ids: list[str],
    file_search_max_num_results: int | None,
    use_web_search: bool,
    web_search_context_size: str,
    web_search_cache_only: bool,
    web_search_domains: list[str],
    use_shell: bool,
    shell_container_id: str | None,
    code_interpreter_container_id: str | None,
    use_podman_shell: bool = False,
) -> list[JsonObject]:
    tools: list[JsonObject] = []
    if vector_store_ids:
        file_search_tool: JsonObject = {
            "type": "file_search",
            "vector_store_ids": vector_store_ids,
        }
        if file_search_max_num_results is not None:
            file_search_tool["max_num_results"] = file_search_max_num_results
        tools.append(file_search_tool)
    if use_web_search:
        web_search_tool: JsonObject = {
            "type": "web_search",
            "search_context_size": web_search_context_size,
            "external_web_access": not web_search_cache_only,
        }
        if web_search_domains:
            web_search_tool["filters"] = {"allowed_domains": web_search_domains}
        tools.append(web_search_tool)
    if use_podman_shell:
        tools.append(build_podman_shell_function_tool())
        return tools
    if use_shell:
        shell_tool: JsonObject = {"type": "shell"}
        if shell_container_id:
            shell_tool["environment"] = {
                "type": "container_reference",
                "container_id": shell_container_id,
            }
        else:
            shell_tool["environment"] = {"type": "container_auto"}
        tools.append(shell_tool)
    code_interpreter_tool: JsonObject = {"type": "code_interpreter"}
    code_interpreter_tool["container"] = code_interpreter_container_id if code_interpreter_container_id else {"type": "auto"}
    tools.append(code_interpreter_tool)
    return tools


def build_response_include(
    *,
    vector_store_ids: list[str],
    use_web_search: bool,
    use_podman_shell: bool = False,
) -> list[str]:
    include: list[str] = []
    # if vector_store_ids:
    #     include.append("file_search_call.results")
    # if use_web_search:
    #     include.append("web_search_call.action.sources")
    if not use_podman_shell:
        include.append("code_interpreter_call.outputs")
    return include


def extract_response_annotations(response: object) -> list[object]:
    dumped = response_to_debug_json(response)
    output = dumped.get("output") if isinstance(dumped, dict) else None
    if not isinstance(output, list):
        return []

    annotations: list[object] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for c in content:
            if not isinstance(c, dict) or c.get("type") != "output_text":
                continue
            part_annotations = c.get("annotations")
            if isinstance(part_annotations, list):
                annotations.extend(part_annotations)
    return annotations


def extract_response_file_citation_metadata(response: object) -> dict[str, dict[str, str]]:
    dumped = response_to_debug_json(response)
    output = dumped.get("output") if isinstance(dumped, dict) else None
    if not isinstance(output, list):
        return {}

    metadata: dict[str, dict[str, str]] = {}
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "file_search_call":
            continue
        results = item.get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            file_id = _obj_get(result, "file_id", None)
            if not isinstance(file_id, str) or not file_id:
                continue
            attrs = _obj_get(result, "attributes", {})
            if not isinstance(attrs, dict):
                attrs = {}
            filename = _obj_get(result, "filename", None)
            path = attrs.get("path")
            title = attrs.get("title")
            blob_sha = attrs.get("blob_sha")
            metadata[file_id] = {
                "filename": filename if isinstance(filename, str) else "",
                "path": path if isinstance(path, str) else "",
                "title": title if isinstance(title, str) else "",
                "blob_sha": blob_sha if isinstance(blob_sha, str) else "",
            }
    return metadata


def format_file_citation_label(annotation: object, metadata: dict[str, dict[str, str]]) -> str:
    file_id = _obj_get(annotation, "file_id", None)
    if isinstance(file_id, str) and file_id:
        entry = metadata.get(file_id)
        if entry:
            title = entry.get("title") or ""
            path = entry.get("path") or ""

            #HACK only 2 vector stores are allowed so we have a slightly ugly directory structure, clean this up here to make it look nicer to the reader
            #we could rename these so this is more preseantable
            path = path.replace("for_ffmpeg/","").replace("forgejo_git/","")

            filename = entry.get("filename") or ""
            if title and path:
                return f"{title} (`{path}`)"
            if title:
                return title
            if path:
                return f"`{path}`"
            if filename:
                return f"`{filename}`"

    filename = _obj_get(annotation, "filename", None)
    if isinstance(filename, str) and filename:
        return f"`{filename}`"
    return ""


def detect_mid_word_citation_corruption(
    text: str, annotations: list[object],
) -> list[dict[str, object]]:
    """Flag ``file_citation`` annotations whose ``index`` lands strictly inside a word.

    The OpenAI Responses API strips inline citation tokens out of the
    model's token stream server-side and records each citation's
    position as the ``index`` field on a ``file_citation`` annotation.
    That stripping pass is occasionally destructive of adjacent
    characters — e.g. a model sentence like
    ``"in AV_TIME_BASE units, which implicitly accepts..."``
    with a citation marker inside ``implicitly`` has been observed to
    collapse into ``"in AV_TIME_BASE unitscitly accepts..."``, with
    the annotation's ``index`` pointing exactly at the ``s|c``
    junction of the fused word.

    A well-formed citation always sits at a word boundary, so any
    annotation whose ``index`` lies between two alphanumeric
    characters is suspicious.
    """
    suspects: list[dict[str, object]] = []
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        if ann.get("type") not in ("file_citation", "container_file_citation"):
            continue
        idx = ann.get("index")
        if not isinstance(idx, int):
            continue
        if not (0 < idx < len(text)):
            continue
        if text[idx - 1].isalnum() and text[idx].isalnum():
            lo = max(0, idx - 20)
            hi = min(len(text), idx + 20)
            suspects.append(
                {
                    "index": idx,
                    "filename": ann.get("filename") or ann.get("file_id"),
                    "context": text[lo:idx] + "|" + text[idx:hi],
                }
            )
    return suspects


def render_file_citations_for_markdown(
    text: str,
    annotations: list[object],
    metadata: dict[str, dict[str, str]] | None = None,
) -> str:
    for suspect in detect_mid_word_citation_corruption(text, annotations):
        logger.warning(
            "suspected OpenAI citation-token stripping corruption: "
            "file_citation at text index=%d (filename=%s) lands mid-word; "
            "context=%r (pipe marks citation index). The posted message "
            "likely has fused/truncated words near this point.",
            suspect["index"],
            suspect["filename"],
            suspect["context"],
        )

    text = UNRESOLVED_CITATION_RE.sub("", text)

    orphan_count = len(ORPHAN_PUA_SENTINEL_RE.findall(text))
    if orphan_count:
        logger.warning(
            "stripped %d orphan OpenAI citation sentinel char(s) from "
            "rendered message; this usually indicates the upstream "
            "Responses API emitted a malformed or truncated citation "
            "marker. Text after well-formed strip (PUA chars shown as "
            "<E200>/<E201>/<E202>): %r",
            orphan_count,
            text.replace("\uE200", "<E200>")
                .replace("\uE201", "<E201>")
                .replace("\uE202", "<E202>"),
        )
        text = ORPHAN_PUA_SENTINEL_RE.sub("", text)
    text = text.rstrip()

    labels: list[str] = []
    seen: set[str] = set()
    metadata = metadata or {}
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        if annotation.get("type") not in ("file_citation", "container_file_citation"):
            continue
        label = format_file_citation_label(annotation, metadata)
        if label and label not in seen:
            seen.add(label)
            labels.append(label)

    if not labels:
        return text

    sources = "\n".join(f"- {label}" for label in labels)
    return f"{text}\n\nSources:\n{sources}" if text else f"Sources:\n{sources}"


def validate_result(
    obj: object,
    annotations: list[object] | None = None,
    file_citation_metadata: dict[str, dict[str, str]] | None = None,
) -> dict[str, str]:
    """``validate_review`` plus rendering of OpenAI file citations."""
    result = validate_review(obj)
    result["message"] = render_file_citations_for_markdown(
        result["message"], annotations or [], file_citation_metadata,
    )
    return result


def validate_triage_result(
    obj: object,
    annotations: list[object] | None = None,
    file_citation_metadata: dict[str, dict[str, str]] | None = None,
    allowed_labels: list[str] | None = None,
) -> dict[str, object]:
    """Validate + normalize the triage model's JSON response.

    Enforces the safety rails documented in ``T_PROMPT_TRIAGE_TASK``:
    - ``route`` must be one of ``TRIAGE_ROUTES``.
    - ``message`` must be a string.
    - ``skip`` / ``engage`` with non-empty ``message``: ``message`` is
      force-cleared to the empty string and a warning is logged.
    - ``helpful_reply`` with empty ``message``: treated as ``engage``
      with empty message (caller will fall through to the main reviewer
      pass). A warning is logged.

    The optional ``requested_models`` / ``requested_effort`` fields are
    constrained by the schema (see ``build_triage_schema``); we just pass
    them through, deduplicated (requesting the same model twice means one
    run of it). Only the engage path consumes them.
    """
    if not isinstance(obj, dict):
        raise RuntimeError("triage model output is not a JSON object")

    route = obj.get("route")
    message = obj.get("message")
    reason = obj.get("reason")

    if route not in TRIAGE_ROUTES:
        raise RuntimeError(f"invalid triage route: {route!r}")
    if not isinstance(message, str):
        raise RuntimeError("triage message is not a string")
    if not isinstance(reason, str):
        reason = ""

    requested_models = list(dict.fromkeys(obj.get("requested_models") or []))
    requested_effort = obj.get("requested_effort")
    label_changes = sanitize_label_changes(obj.get("label_changes"), allowed_labels or [])

    rendered_message = render_file_citations_for_markdown(
        message, annotations or [], file_citation_metadata,
    )

    if route == "helpful_reply" and not rendered_message.strip():
        logger.warning(
            "triage returned route=helpful_reply with empty message; "
            "falling back to engage so the main reviewer pass runs; reason=%r",
            reason,
        )
        return {
            "route": "engage", "message": "", "reason": reason,
            "requested_models": requested_models,
            "requested_effort": requested_effort,
            "label_changes": label_changes,
        }

    if route in ("skip", "engage") and rendered_message.strip():
        logger.warning(
            "triage returned route=%s with non-empty message; "
            "clearing message (route=%s must have empty message); reason=%r; dropped_message=%r",
            route, route, reason, rendered_message[:200],
        )
        rendered_message = ""

    return {
        "route": route,
        "message": rendered_message,
        "reason": reason,
        "requested_models": requested_models,
        "requested_effort": requested_effort,
        "label_changes": label_changes,
    }


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


def run_triage_stage(
    client: OpenAI,
    *,
    args: argparse.Namespace,
    request: JsonObject,
    patch_file_id: str,
    patch_was_truncated: bool,
    reviewer_username: str,
    repo_roots: list[Path],
    vector_store_ids: list[str],
    repo_mount_paths: list[str],
    use_podman_shell: bool,
    podman_shell_session: podman_host.ContainerShellSession | None,
    tools: list[JsonObject],
    include: list[str],
    debug_dir_specified: bool,
) -> dict[str, object] | None:
    """Run the optional mini-model triage pre-check.

    Returns a dict with ``route`` / ``message`` / ``reason`` keys on
    success. Returns ``None`` on triage failure (API error, refusal,
    parse error), signalling that the caller should fall through to
    the main reviewer pass.

    When the container went unhealthy during triage, the returned dict
    also contains ``container_unhealthy=True`` so the caller can surface
    ``EXIT_CONTAINER_UNHEALTHY`` without running the main pass on the
    dead container.
    """
    triage_content: list[InputContentItem] = [
        {"type": "input_text", "text": make_triage_user_text(request, patch_was_truncated)},
        {"type": "input_file", "file_id": patch_file_id},
    ]

    ci_triage_mode = bool(
        isinstance(request.get("ci_triage"), dict) and request.get("ci_triage")
    )
    triage_label_allowlist = triage_label_allowlist_from_request(request)

    triage_features: set[str] = set()
    if vector_store_ids:
        triage_features.add("vector_store_search")
    if args.use_web_search:
        triage_features.add("web_search")
    if use_podman_shell:
        triage_features.add("podman_shell")
    else:
        triage_features.add("code_interpreter")

    triage_kwargs: ResponseKwargs = {
        "model": args.triage_model,
        "input": [
            {
                "role": "developer",
                "content": generate_llm_prompt(
                    role="triager",
                    vendor="openai",
                    model=args.triage_model,
                    features=triage_features,
                    repo_roots=repo_roots,
                    container_repo_mounts=repo_mount_paths,
                    reviewer_username=reviewer_username,
                    ci_triage_mode=ci_triage_mode,
                    allowed_models=args.allowed_model,
                    allowed_labels=triage_label_allowlist,
                ),
            },
            {"role": "user", "content": triage_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                **build_triage_schema(args.allowed_model, triage_label_allowlist),
            },
        },
        "max_output_tokens": args.triage_max_output_tokens,
    }
    if args.verbosity is not None:
        triage_kwargs["text"]["verbosity"] = args.verbosity
    if args.top_p is not None:
        triage_kwargs["top_p"] = args.top_p
    triage_reasoning: JsonObject = {}
    if args.triage_reasoning_effort:
        triage_reasoning["effort"] = args.triage_reasoning_effort
    if args.reasoning_summary:
        triage_reasoning["summary"] = args.reasoning_summary
    if triage_reasoning:
        triage_kwargs["reasoning"] = triage_reasoning
    if args.max_tool_calls is not None:
        triage_kwargs["max_tool_calls"] = args.max_tool_calls
    # The triage tier is independent of ``--service-tier`` -- it is only
    # applied when the user explicitly asks for it via
    # ``--triage-service-tier``. We do NOT inherit from ``--service-tier``
    # here because the two stages have very different cost/latency
    # profiles and silently coupling them surprises users who only
    # wanted the main pass on a non-default tier.
    if args.triage_service_tier is not None:
        triage_kwargs["service_tier"] = args.triage_service_tier
    if tools:
        triage_kwargs["tools"] = tools
    if include:
        triage_kwargs["include"] = include

    if args.verbose and ci_triage_mode:
        logger.debug("triage developer prompt: CI failure mode (ci_triage payload present)")

    if args.verbose:
        tool_names = [
            tool.get("type") if tool.get("type") != "function" else f"function:{tool.get('name')}"
            for tool in tools
            if isinstance(tool, dict)
        ]
        logger.debug(
            "triage responses.create start model=%s effort=%s tier=%s tools=%s vector_stores=%d max_output_tokens=%d",
            args.triage_model,
            args.triage_reasoning_effort or "-",
            args.triage_service_tier or "-",
            ",".join(tool_names) if tool_names else "-",
            len(vector_store_ids),
            args.triage_max_output_tokens,
        )

    triage_started = time.monotonic()
    try:
        if use_podman_shell:
            if podman_shell_session is None:
                raise RuntimeError("podman shell triage requires a container session")
            triage_response = run_responses_resolving_podman_shell(
                client,
                initial_kwargs=triage_kwargs,
                podman_shell_session=podman_shell_session,
                max_tool_rounds=args.podman_max_tool_rounds,
                max_shell_timeout_s=args.podman_exec_timeout,
                what="triage responses.create",
                verbose=args.verbose,
                debug_dir=args.debug_response_dir if debug_dir_specified else None,
                wrapper_request=request,
            )
        else:
            triage_response = call_with_rate_limit_retry(
                lambda: client.responses.create(**triage_kwargs),
                what="triage responses.create",
                verbose=args.verbose,
                # Like the main request, let the outer caller decide whether
                # to retry on transient timeouts rather than silently piling
                # up duplicate billed runs here.
                retry_transient=False,
            )
    except Exception as exc:
        if args.verbose:
            logger.debug(
                "triage responses.create failed dt=%.3fs error=%s %s",
                time.monotonic() - triage_started,
                type(exc).__name__,
                str(exc).replace("\n", " "),
            )
        if (
            args.use_openai_container_repos
            and repo_roots
            and openai_container.is_container_unhealthy_error(exc)
        ):
            logger.warning(
                "openai container unhealthy during triage responses.create (%s)",
                str(exc).replace("\n", " "),
            )
            return {"route": "engage", "message": "", "reason": "", "container_unhealthy": True}
        logger.warning(
            "triage stage failed with %s: %s; will fall through to main reviewer pass",
            type(exc).__name__,
            str(exc).replace("\n", " "),
        )
        return None

    if args.verbose:
        logger.debug(
            "triage responses.create ok %s",
            format_response_stats(triage_response, elapsed_seconds=time.monotonic() - triage_started),
        )
    # The podman tool loop already dumps every round (including the final
    # response) with the kwargs that actually produced it.
    if debug_dir_specified and not use_podman_shell:
        dump_response_debug_artifacts(
            triage_response,
            triage_kwargs,
            wrapper_request=request,
            debug_dir=args.debug_response_dir,
            verbose=args.verbose,
        )

    try:
        triage_annotations = extract_response_annotations(triage_response)
        triage_file_citation_metadata = extract_response_file_citation_metadata(triage_response)
        triage_raw_text = extract_response_text(
            triage_response,
            response_kwargs=triage_kwargs,
            debug_dir=args.debug_response_dir,
            verbose=args.verbose,
        )
        parsed = json.loads(triage_raw_text)
        validated = validate_triage_result(
            parsed, triage_annotations, triage_file_citation_metadata,
            allowed_labels=triage_label_allowlist,
        )
    except Exception as exc:
        logger.warning(
            "triage stage output could not be parsed (%s: %s); falling through to main reviewer pass",
            type(exc).__name__,
            str(exc).replace("\n", " "),
        )
        return None

    logger.info(
        "triage decision route=%s message_chars=%d reason=%r",
        validated["route"],
        len(validated["message"]),
        validated.get("reason", ""),
    )
    return dict(validated)


class OpenAIContainerUnhealthy(Exception):
    """The attached OpenAI container is unhealthy (expired / stopped).

    Raised by ``OpenAIReviewer`` so the entrypoint releases the lease and
    exits ``EXIT_CONTAINER_UNHEALTHY``, letting the caller retry against a
    freshly provisioned container.
    """


@dataclass
class OpenAIResources:
    """OpenAI-specific per-run resources shared across reviewer calls.

    Built once by the entrypoint after vector-store / container / podman
    setup, so the single uploaded patch file, tool wiring and container ids
    are reused rather than rebuilt per call. ``uploaded_file_ids`` is the
    entrypoint's own cleanup list; the reviewer appends any file it uploads
    (e.g. the source bundle) so the outer ``finally`` deletes it.
    """

    client: OpenAI
    tools: list[JsonObject]
    include: list[str]
    patch_file_id: str
    vector_store_ids: list[str]
    shared_container_id: str | None
    podman_shell_session: podman_host.ContainerShellSession | None
    uploaded_file_ids: list[str]
    debug_dir_specified: bool


class OpenAIReviewer(Reviewer):
    """One OpenAI Responses-API review pass behind the shared interface.

    ``review(ctx)`` builds the developer/user input, runs the model
    (driving the podman shell tool loop when ``--podman`` is set, else the
    direct Responses call), validates the JSON against ``REVIEW_SCHEMA`` and
    renders file citations. Raises ``OpenAIContainerUnhealthy`` /
    ``BadModelOutput`` for the routine, retryable failure modes.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        resources: OpenAIResources,
        *,
        model: str | None = None,
        role: str = "reviewer",
        effort: str | None = None,
    ) -> None:
        self.args = args
        self.res = resources
        self.model = model or args.model
        self.role = role
        self.effort = effort or args.reasoning_effort
        self.name = f"openai:{self.model}"

    def review(self, ctx: ReviewContext) -> Review:
        args = self.args
        res = self.res
        client = res.client

        content: list[InputContentItem] = [
            {
                "type": "input_text",
                "text": make_user_text(
                    ctx.request, ctx.source_notes, ctx.source_files, ctx.patch_truncated
                ),
            },
            {"type": "input_file", "file_id": res.patch_file_id},
        ]
        if self.role == "combiner":
            content.append({
                "type": "input_text",
                "text": make_combiner_user_text(ctx.review_drafts()),
            })
        if ctx.source_bundle is not None:
            source_file_id = upload_text_file(
                client,
                filename="source_bundle.txt",
                text=ctx.source_bundle,
                verbose=args.verbose,
            )
            res.uploaded_file_ids.append(source_file_id)
            content.append({"type": "input_file", "file_id": source_file_id})

        reviewer_features: set[str] = set()
        if ctx.source_bundle is not None:
            reviewer_features.add("source_bundle")
        if res.vector_store_ids:
            reviewer_features.add("vector_store_search")
        if args.use_web_search:
            reviewer_features.add("web_search")
        if args.podman:
            reviewer_features.add("podman_shell")
        else:
            reviewer_features.add("code_interpreter")

        response_kwargs: ResponseKwargs = {
            "model": self.model,
            "input": [
                {
                    "role": "developer",
                    "content": generate_llm_prompt(
                        role=self.role,
                        vendor="openai",
                        model=self.model,
                        features=reviewer_features,
                        repo_roots=ctx.repo_roots,
                        container_repo_mounts=ctx.repo_mount_paths,
                        reviewer_username=ctx.reviewer_username,
                        ci_triage_mode=ctx.ci_triage_mode,
                    ),
                },
                {"role": "user", "content": content},
            ],
            "text": {"format": {"type": "json_schema", **REVIEW_SCHEMA}},
            "max_output_tokens": args.max_output_tokens,
        }
        if args.verbosity is not None:
            response_kwargs["text"]["verbosity"] = args.verbosity
        if args.top_p is not None:
            response_kwargs["top_p"] = args.top_p
        reasoning: JsonObject = {}
        if self.effort:
            reasoning["effort"] = self.effort
        if args.reasoning_summary:
            reasoning["summary"] = args.reasoning_summary
        if reasoning:
            response_kwargs["reasoning"] = reasoning
        if args.max_tool_calls is not None:
            response_kwargs["max_tool_calls"] = args.max_tool_calls
        if args.service_tier is not None:
            response_kwargs["service_tier"] = args.service_tier
        if res.tools:
            response_kwargs["tools"] = res.tools
        if res.include:
            response_kwargs["include"] = res.include

        if args.verbose:
            tool_names = [
                tool.get("type") if tool.get("type") != "function" else f"function:{tool.get('name')}"
                for tool in res.tools
                if isinstance(tool, dict)
            ]
            source_bundle_bytes = len(ctx.source_bundle.encode("utf-8")) if ctx.source_bundle is not None else 0
            logger.debug("responses.create start model=%s effort=%s verbosity=%s tier=%s tools=%s vector_stores=%d source_files=%d source_bytes=%d max_output_tokens=%d", self.model, self.effort or "-", args.verbosity or "-", args.service_tier or "-", ",".join(tool_names) if tool_names else "-", len(res.vector_store_ids), len(ctx.source_files), source_bundle_bytes, args.max_output_tokens)

        create_started = time.monotonic()
        try:
            if args.podman:
                if res.podman_shell_session is None:
                    raise RuntimeError("container shell session missing for main review pass")
                response = run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs=response_kwargs,
                    podman_shell_session=res.podman_shell_session,
                    max_tool_rounds=args.podman_max_tool_rounds,
                    max_shell_timeout_s=args.podman_exec_timeout,
                    what="responses.create",
                    verbose=args.verbose,
                    debug_dir=args.debug_response_dir if res.debug_dir_specified else None,
                    wrapper_request=ctx.request,
                )
            else:
                response = call_with_rate_limit_retry(
                    lambda: client.responses.create(**response_kwargs),
                    what="responses.create",
                    verbose=args.verbose,
                    # Main LLM request: APITimeoutError / APIConnectionError
                    # are propagated to the outer caller (e.g. fairy.py)
                    # which decides whether to retry the entire wrapper. These
                    # requests are long and unpredictable, so silently re-issuing
                    # them here would risk piling up duplicate billed runs.
                    retry_transient=False,
                )
        except Exception as exc:
            if args.verbose:
                logger.debug("responses.create failed dt=%.3fs error=%s %s", time.monotonic() - create_started, type(exc).__name__, str(exc).replace("\n", " "))
            if (
                args.use_openai_container_repos
                and ctx.repo_roots
                and openai_container.is_container_unhealthy_error(exc)
            ):
                # Known, routine, retryable failure: surface a typed error
                # so the entrypoint marks the lease unhealthy and exits with
                # EXIT_CONTAINER_UNHEALTHY rather than dumping the full SDK
                # traceback. The caller retries against a fresh container.
                logger.warning(
                    "openai container unhealthy during responses.create (%s); "
                    "exiting with code %d so caller retries against a fresh container",
                    str(exc).replace("\n", " "),
                    EXIT_CONTAINER_UNHEALTHY,
                )
                raise OpenAIContainerUnhealthy() from exc
            raise

        if args.verbose:
            logger.debug("responses.create ok %s", format_response_stats(response, elapsed_seconds=time.monotonic() - create_started))
        # The podman tool loop already dumps every round (including the
        # final response) with the kwargs that actually produced it.
        if res.debug_dir_specified and not args.podman:
            dump_response_debug_artifacts(
                response,
                response_kwargs,
                wrapper_request=ctx.request,
                debug_dir=args.debug_response_dir,
                verbose=args.verbose,
            )

        annotations = extract_response_annotations(response)
        file_citation_metadata = extract_response_file_citation_metadata(response)
        raw_text = extract_response_text(
            response,
            response_kwargs=response_kwargs,
            debug_dir=args.debug_response_dir,
            verbose=args.verbose,
        )
        try:
            result = validate_result(json.loads(raw_text), annotations, file_citation_metadata)
        except (json.JSONDecodeError, SchemaError, RecursionError) as exc:
            # The model returned text that isn't the JSON shape we asked for
            # (OpenAI ``strict`` output is best-effort, not a guarantee).
            # ``RecursionError`` covers a pathologically nested JSON payload
            # (the model output is attacker-influenceable via prompt
            # injection); json's own recursion guard turns that into a clean
            # exception, not a crash. Discard the run with a typed error so
            # the entrypoint exits EXIT_BAD_MODEL_OUTPUT and the caller
            # retries.
            logger.error(
                "reviewer output did not match the requested schema (%s: %s); "
                "discarding run, exiting %d so caller retries",
                type(exc).__name__, str(exc).replace("\n", " "),
                EXIT_BAD_MODEL_OUTPUT,
            )
            raise BadModelOutput() from exc

        if args.verbose:
            extra = f" vector_stores={','.join(res.vector_store_ids)}" if res.vector_store_ids else ""
            logger.debug("classification=%s source_files=%d%s", result["classification"], len(ctx.source_files), extra)

        return Review(
            classification=result["classification"],
            message=result["message"],
            model=self.name,
        )


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


def open_review_container_shell(
    remote_host: podman_host.RemoteHost,
    repo_specs: list[podman_repos.RepoSpec],
    args: argparse.Namespace,
) -> tuple[podman_host.ContainerHandle, podman_host.ContainerShellSession]:
    """Start a fresh ephemeral container, fill its repos, and open a shell.

    One isolated container per call, so concurrent ensemble reviewers never
    share a working tree. On any provisioning failure the half-started
    container is stopped before the error propagates.
    """
    handle = podman_host.start_ephemeral_container(
        image=args.podman_image,
        host=remote_host,
        network=args.podman_network,
        memory=args.podman_memory,
        cpus=args.podman_cpus,
    )
    try:
        podman_repos.provision_repos_into_container(handle, repo_specs, remote_host)
        podman_host.copy_into_container(handle, AGENT_LOCAL_PATH, AGENT_CONTAINER_DIR)
        session = podman_host.open_container_shell(handle, AGENT_CONTAINER_PATH)
    except Exception:
        podman_host.stop_container(handle)
        raise
    return handle, session


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
        openai_common.logger,
        openai_vector_store.logger,
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
    # full reviewer pass regardless of the triage route (skip / helpful_reply)
    # and even when triage itself fails on a CI-red request.
    force_engage = bool(request.get("force_engage"))

    patch = request.get("patch") if isinstance(request.get("patch"), str) else ""
    reviewer_username = request.get("reviewer_username")
    if not isinstance(reviewer_username, str):
        reviewer_username = ""

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

    podman_container_handle: podman_host.ContainerHandle | None = None
    podman_shell_session: podman_host.ContainerShellSession | None = None
    podman_repo_specs: list[podman_repos.RepoSpec] = []
    remote_host: podman_host.RemoteHost | None = None
    # Extra containers spun up on demand by the ensemble's new_shell factory
    # (one isolated container per non-OpenAI reviewer); released in finally.
    ensemble_shells: list[tuple[podman_host.ContainerHandle, podman_host.ContainerShellSession]] = []
    if args.podman:
        if not repo_roots:
            raise RuntimeError(
                "--podman requires at least one local git checkout "
                "via --repo-root / --extra-repo-root"
            )
        if not args.podman_ssh_dest:
            raise RuntimeError(
                "--podman requires --podman-ssh-dest; provision the host with "
                "containers/provision_remote.py first"
            )
        remote_host = _build_remote_host(args)
        if not podman_host.image_tag_exists(args.podman_image, host=remote_host):
            raise RuntimeError(
                f"podman image {args.podman_image!r} not found; build with: "
                f"python3 {(Path(__file__).resolve().parent / 'containers' / 'build_image.py')} "
                f"--tag {args.podman_image!r}"
            )
        logger.info(
            "container: ensure network=%s image=%s host=%s",
            args.podman_network or "(default)",
            args.podman_image,
            remote_host.ssh_dest,
        )
        if args.podman_network:
            podman_host.ensure_isolated_network(
                args.podman_network, host=remote_host,
            )
        podman_repo_specs = podman_repos.build_repo_specs(
            repo_roots, mirror_root=args.podman_mirror_root,
        )
        podman_container_handle, podman_shell_session = open_review_container_shell(
            remote_host, podman_repo_specs, args,
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
    if args.no_source_bundle:
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

    patch_bundle, patch_was_truncated = build_patch_bundle(patch, args.max_patch_bytes)

    uploaded_file_ids: list[str] = []

    try:
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
        )
        include = build_response_include(
            vector_store_ids=vector_store_ids,
            use_web_search=args.use_web_search,
            use_podman_shell=args.podman,
        )

        def new_shell() -> podman_host.ContainerShellSession:
            assert remote_host is not None  # only wired in when --podman
            handle, session = open_review_container_shell(remote_host, podman_repo_specs, args)
            ensemble_shells.append((handle, session))
            return session

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
            new_shell=new_shell if args.podman else None,
        )
        openai_resources = OpenAIResources(
            client=client,
            tools=tools,
            include=include,
            patch_file_id=patch_file_id,
            vector_store_ids=vector_store_ids,
            shared_container_id=shared_container_id,
            podman_shell_session=podman_shell_session,
            uploaded_file_ids=uploaded_file_ids,
            debug_dir_specified=debug_dir_specified,
        )

        stashed_triage_label_changes: list[dict[str, object]] = []
        requested_models: list[str] = []

        if args.triage_model:
            triage_result = run_triage_stage(
                client,
                args=args,
                request=request,
                patch_file_id=patch_file_id,
                patch_was_truncated=patch_was_truncated,
                reviewer_username=reviewer_username,
                repo_roots=repo_roots,
                vector_store_ids=vector_store_ids,
                repo_mount_paths=repo_mount_paths,
                use_podman_shell=args.podman,
                podman_shell_session=podman_shell_session,
                tools=tools,
                include=include,
                debug_dir_specified=debug_dir_specified,
            )
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
                if route == "helpful_reply":
                    logger.info(
                        "triage route=helpful_reply; reason=%r; skipping main reviewer pass",
                        triage_result.get("reason", ""),
                    )
                    emit_review_stdout(
                        "helpful_reply",
                        str(triage_result.get("message") or ""),
                        label_changes=triage_label_changes,
                    )
                    return 0
                if route == "engage":
                    stashed_triage_label_changes = triage_label_changes
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

        if requested_models:
            model_reviewers = [
                make_reviewer(spec, args=args, resources=openai_resources, role="reviewer", verbose=args.verbose)
                for spec in requested_models
            ]
        else:
            model_reviewers = [
                OpenAIReviewer(args, openai_resources, model=args.model, role="reviewer")
            ]
            for spec in args.extra_model:
                model_reviewers.append(
                    make_reviewer(spec, args=args, resources=openai_resources, role="reviewer", verbose=args.verbose)
                )
        combiner = (
            make_reviewer(args.combine_model, args=args, resources=openai_resources, role="combiner", verbose=args.verbose)
            if args.combine_model
            else None
        )
        if combiner is None and requested_models and len(model_reviewers) > 1:
            # A user may request two models on a deployment configured
            # without --combine-model; combine with the configured main
            # model rather than rejecting the request.
            logger.info(
                "user requested %d models with no --combine-model configured; "
                "combining with openai:%s", len(model_reviewers), args.model,
            )
            combiner = OpenAIReviewer(args, openai_resources, model=args.model, role="combiner")
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
            label_changes=stashed_triage_label_changes,
        )
        return 0
    finally:
        for file_id in uploaded_file_ids:
            delete_uploaded_file(client, file_id, verbose=args.verbose)
        if container_lease is not None:
            container_lease.release(healthy=container_lease_healthy)
        if podman_shell_session is not None:
            podman_shell_session.close()
        if podman_container_handle is not None:
            podman_host.stop_container(podman_container_handle)
        for handle, session in ensemble_shells:
            session.close()
            podman_host.stop_container(handle)


if __name__ == "__main__":
    raise SystemExit(main())
