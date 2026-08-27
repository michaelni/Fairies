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

OpenAI reviewer: ``OpenAIReviewer`` (the Responses-API implementation of
``llm_review_api.Reviewer``) and the OpenAI-specific plumbing one review
pass needs -- HTTP client tuning, tool/include builders, the podman shell
function-call loop, response stats and file-citation rendering.

What does NOT belong here: triage, argument parsing, source-bundle
building and process orchestration (the wrapper entrypoint), SDK-generic
helpers (``openai_common``), vector stores and OpenAI containers
(``openai_vector_store``, ``openai_container*``), and any other
provider's code (``anthropic_reviewer``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
import re
import socket
import threading
import time
from typing import Callable, Sequence

import httpx
from openai import DefaultHttpxClient, OpenAI

import concurrency
from common import dump_response_debug_artifacts, response_to_debug_json
from llm_prompt import REVIEWER_ROLE, generate_llm_prompt
from llm_review_api import (
    EXIT_BAD_MODEL_OUTPUT,
    BadModelOutput,
    ReviewContext,
    Reviewer,
    RoleSpec,
    SchemaError,
)
import openai_container
from openai_common import (
    InputContentItem,
    JsonObject,
    ResponseKwargs,
    _obj_get,
    call_with_rate_limit_retry,
    extract_response_text,
    upload_text_file,
)
import podman_host
from shell_tool import build_shell_tool_schema, exec_machine_call

__all__ = [
    "EXIT_CONTAINER_UNHEALTHY",
    "INHERIT_SERVICE_TIER",
    "OPENAI_TCP_KEEPALIVE_SOCKET_OPTIONS",
    "OpenAIContainerUnhealthy",
    "OpenAIResources",
    "OpenAIReviewer",
    "build_response_include",
    "build_response_tools",
    "extract_function_calls_from_response",
    "extract_response_annotations",
    "extract_response_file_citation_metadata",
    "format_response_stats",
    "make_openai_http_client",
    "prompt_cache_key_for",
    "render_file_citations_for_markdown",
    "run_responses_resolving_podman_shell",
]

logger = logging.getLogger(__name__)


# Default for the ``service_tier`` constructor parameter: inherit
# ``--service-tier``. Distinct from an explicit ``None``, which sends no
# tier at all (account default).
INHERIT_SERVICE_TIER = "__inherit__"

# Distinct non-zero exit code the wrapper uses when it recognizes the
# attached OpenAI container as unhealthy (expired, not running, ...).
# The caller (e.g. fairy.py) just retries on any non-zero exit,
# but a dedicated code makes these routine, retryable failures
# trivially greppable in logs instead of indistinguishable from a crash.
EXIT_CONTAINER_UNHEALTHY = 2

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


def build_podman_shell_function_tool(
    machines: Sequence[podman_host.ShellHostSpec],
) -> JsonObject:
    schema = build_shell_tool_schema([m.label for m in machines])
    return {
        "type": "function",
        "name": schema["name"],
        "description": schema["description"],
        "parameters": {**schema["input_schema"], "additionalProperties": False},
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
    shells: dict[str, podman_host.ContainerShellSession],
    machine_labels: Sequence[str],
    open_shell: Callable[[str], tuple[podman_host.ContainerShellSession, str]],
    open_lock: threading.Lock | None = None,
    max_tool_rounds: int,
    max_shell_timeout_s: float,
    what: str,
    verbose: bool,
    debug_dir: str | None = None,
    wrapper_request: JsonObject | None = None,
    parallel_tool_calls: bool = False,
) -> object:
    """Drive ``responses.create`` in a loop until no pending function calls.

    Dispatches ``name=shell`` via :func:`shell_tool.exec_machine_call` over
    the persistent per-machine sessions in ``shells`` (lazily opened through
    ``open_shell``); other function names receive an error
    ``function_call_output`` so the model can recover.

    ``max_tool_rounds <= 0`` means no cap on the number of rounds.
    ``parallel_tool_calls=False`` forces one shell call per follow-up round
    (round 1 is never forced).

    With ``debug_dir`` set, EVERY round's response is dumped (paired with
    the exact kwargs that produced it), not just the final one -- the
    intermediate rounds are where the function calls and their outputs
    live, and each is a separately billed request. All rounds append to
    the same conversation file.
    """
    conv_path: str | None = None

    def create_and_dump(kwargs: ResponseKwargs, what_label: str) -> object:
        nonlocal conv_path
        with concurrency.slot("openai"):
            resp = call_with_rate_limit_retry(
                lambda: client.responses.create(**kwargs),
                what=what_label,
                verbose=verbose,
                retry_transient=False,
            )
        if debug_dir:
            conv_path = dump_response_debug_artifacts(
                resp, kwargs, wrapper_request=wrapper_request,
                debug_dir=debug_dir, verbose=verbose, conversation=conv_path,
            ) or conv_path
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
            payload_obj = exec_machine_call(
                shells, machine_labels, open_shell, args_obj,
                max_timeout_s=max_shell_timeout_s, open_lock=open_lock,
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
        }
        if not parallel_tool_calls:
            follow["parallel_tool_calls"] = False
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
        # gpt-5.6 keys its prompt cache on ``reasoning``: omitting it made
        # every round 2 a full-prefix miss.
        rsn = initial_kwargs.get("reasoning")
        if rsn is not None:
            follow["reasoning"] = rsn
        pck = initial_kwargs.get("prompt_cache_key")
        if pck is not None:
            follow["prompt_cache_key"] = pck
        response = create_and_dump(follow, f"{what} (podman shell follow-up)")


def prompt_cache_key_for(role_name: str, request: JsonObject) -> str | None:
    """Cache-routing key: one per role+subject+head, so a conversation's
    rounds and retries share a cache machine without funneling unrelated
    reviews into one key (OpenAI guidance: ~15 requests/minute per key).
    Deliberately built from forge data only -- upload file_ids change on
    every retry and must not enter the key."""
    subject = request.get("pull_request") or request.get("issue")
    if not isinstance(subject, dict) or subject.get("number") is None:
        return None
    key = f"fairy:{role_name}:{subject['number']}"
    head = str(subject.get("head_sha") or "")[:12]
    return f"{key}:{head}" if head else key


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
    machines: Sequence[podman_host.ShellHostSpec] = (),
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
        tools.append(build_podman_shell_function_tool(machines))
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

    client: OpenAI | None
    tools: list[JsonObject]
    include: list[str]
    # None when the run has no patch (the issue-investigator task).
    patch_file_id: str | None
    vector_store_ids: list[str]
    shared_container_id: str | None
    shells: dict[str, podman_host.ContainerShellSession] | None
    open_shell: Callable[[str], tuple[podman_host.ContainerShellSession, str]] | None
    uploaded_file_ids: list[str]
    debug_dir_specified: bool
    # Serializes lazy opens into ``shells``: role passes run in parallel
    # threads (llm_review_api.run_parallel) but share the one dict.
    shells_lock: threading.Lock = field(default_factory=threading.Lock)


class OpenAIReviewer(Reviewer):
    """One OpenAI Responses-API pass of a ``RoleSpec`` behind the shared
    interface.

    ``run(ctx)`` builds the developer/user input from the role, runs the
    model (driving the podman shell tool loop when ``--podman`` is set,
    else the direct Responses call), renders file citations into the
    message and validates against the role's schema. Raises
    ``OpenAIContainerUnhealthy`` / ``BadModelOutput`` for the routine,
    retryable failure modes.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        resources: OpenAIResources,
        *,
        model: str | None = None,
        role: RoleSpec = REVIEWER_ROLE,
        effort: str | None = None,
        max_output_tokens: int | None = None,
        service_tier: str | None = INHERIT_SERVICE_TIER,
        verbosity: str | None = None,
    ) -> None:
        self.args = args
        self.res = resources
        self.model = model or args.model
        self.role = role
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self.verbosity = verbosity if verbosity is not None else args.verbosity
        # An explicit ``service_tier=None`` sends no tier: the triage call
        # is documented (--triage-service-tier) as independent of
        # --service-tier, so it must not inherit it.
        self.service_tier = args.service_tier if service_tier is INHERIT_SERVICE_TIER else service_tier
        self.name = f"openai:{self.model}"

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        args = self.args
        res = self.res
        client = res.client

        user_texts = self.role.user_texts(ctx)
        content: list[InputContentItem] = [
            {"type": "input_text", "text": user_texts[0]},
        ]
        if res.patch_file_id is not None:
            content.append({"type": "input_file", "file_id": res.patch_file_id})
        content.extend({"type": "input_text", "text": text} for text in user_texts[1:])
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
        if args.web_search != "off":
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
                        role=self.role.name,
                        vendor="openai",
                        model=self.model,
                        features=reviewer_features,
                        repo_roots=ctx.repo_roots,
                        container_repo_mounts=ctx.repo_mount_paths,
                        machines=ctx.machines,
                        reviewer_username=ctx.reviewer_username,
                        project_facts=ctx.project_facts,
                        ci_triage_mode=ctx.ci_triage_mode,
                        **self.role.prompt_kwargs,
                    ),
                },
                {"role": "user", "content": content},
            ],
            "text": {"format": {"type": "json_schema", **self.role.schema}},
            "max_output_tokens": (
                self.max_output_tokens
                if self.max_output_tokens is not None
                else args.max_output_tokens
            ),
        }
        if self.verbosity is not None:
            response_kwargs["text"]["verbosity"] = self.verbosity
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
        if self.service_tier is not None:
            response_kwargs["service_tier"] = self.service_tier
        pck = prompt_cache_key_for(self.role.name, ctx.request)
        if pck is not None:
            response_kwargs["prompt_cache_key"] = pck
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
            logger.debug("responses.create start role=%s model=%s effort=%s verbosity=%s tier=%s tools=%s vector_stores=%d source_files=%d source_bytes=%d max_output_tokens=%d", self.role.name, self.model, self.effort or "-", self.verbosity or "-", self.service_tier or "-", ",".join(tool_names) if tool_names else "-", len(res.vector_store_ids), len(ctx.source_files), source_bundle_bytes, response_kwargs["max_output_tokens"])

        create_started = time.monotonic()
        try:
            if args.podman:
                if res.shells is None or res.open_shell is None:
                    raise RuntimeError("container shell session missing for main review pass")
                response = run_responses_resolving_podman_shell(
                    client,
                    initial_kwargs=response_kwargs,
                    shells=res.shells,
                    machine_labels=tuple(m.label for m in ctx.machines),
                    open_shell=res.open_shell,
                    open_lock=res.shells_lock,
                    max_tool_rounds=args.podman_max_tool_rounds,
                    max_shell_timeout_s=args.podman_exec_timeout,
                    what="responses.create",
                    verbose=args.verbose,
                    debug_dir=args.debug_response_dir if res.debug_dir_specified else None,
                    wrapper_request=ctx.request,
                    parallel_tool_calls=args.podman_parallel_tool_calls,
                )
            else:
                with concurrency.slot("openai"):
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
            parsed = json.loads(raw_text)
            # Model output at the process boundary: render citation markup
            # into the message only when the shape is plausible; the role
            # validator rejects anything else right after.
            if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
                parsed["message"] = render_file_citations_for_markdown(
                    parsed["message"], annotations, file_citation_metadata,
                )
            result = self.role.validate(parsed)
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
                "%s output did not match the requested schema (%s: %s); "
                "discarding run, exiting %d so caller retries",
                self.role.name, type(exc).__name__, str(exc).replace("\n", " "),
                EXIT_BAD_MODEL_OUTPUT,
            )
            raise BadModelOutput() from exc

        ctx.collect_into(result, list(res.shells.values()) if res.shells else [])

        if args.verbose:
            extra = f" vector_stores={','.join(res.vector_store_ids)}" if res.vector_store_ids else ""
            verdict = result.get("classification") or result.get("route") or "-"
            logger.debug("classification=%s source_files=%d%s", verdict, len(ctx.source_files), extra)

        return result
