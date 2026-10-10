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

Grok-Build-CLI (``grok``) pass of a ``RoleSpec``: the headless ``grok``
command line, the agent definition that carries the prompt, the per-run
``$GROK_HOME`` grok reads its credential and MCP server from, the
streaming-messages-json output and grok's error strings, on top of
``cli_reviewer.CliReviewer`` which runs the pinned grok binary in a
container on the ``--grok-host``.

The prompt goes in as the body of an agent definition, which grok takes
as its whole system prompt, and the user turn is a fixed sentence: grok
reads files for a user turn, not for the system prompt (see
``build_grok_command``). Of grok's tools ``search_tool`` and
``use_tool`` stay, which call MCP tools, and a deny rule keeps
``use_tool`` from loading files as their arguments. The shell into the
review containers is the one MCP server configured.

A result that names a missing or rejected credential before the first
turn is a credential failure: the host credential is set aside and a
human restores it. A result whose errors name a 402 is an exhausted
balance: a ``QuotaExhausted`` that holds the account. Every other run
without a schema-forced answer is a ``ProviderTurnFailed`` that takes
the retry and fallback path.

``--concurrency grok:N`` caps how many passes run at once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess

from cli_container import CliContainer
from cli_reviewer import MCP_TOOL_TIMEOUT_S, CliReviewer, Staged, last_event
from common import JsonObject
from llm_review_api import ProviderTurnFailed, quota_exhausted

__all__ = [
    "CREDENTIAL_NAME",
    "GrokReviewer",
    "GrokTurnFailed",
    "build_grok_command",
    "result_event",
]

logger = logging.getLogger(__name__)

# The reasoning_efforts of the model catalog grok 1.0.44 fetched on
# 2026-10-01, united across its models; that day grok rejected xhigh for
# grok-4.5, whose menu lacks it.
GROK_EFFORTS = ("low", "medium", "high", "xhigh")

CONTAINER_GROK_HOME = "/grok-home"
CREDENTIAL_NAME = "auth.json"
AGENT_NAME = "agent.md"

# The body of an agent definition is a template whose three delimiters
# start with "${"
# (https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-tools/src/types/description.rs);
# this is the expression that renders a "${".
DELIMITER_START = '${{ "${" }}'

# Read-back cap for the untrusted grok container: an oversized file
# arrives truncated, so its comparison fails closed.
MAX_PROMPT_BYTES = 64 * 1024 * 1024


class GrokTurnFailed(ProviderTurnFailed):
    """grok ended the run without the schema-forced answer."""


def build_grok_command(
    *,
    grok_bin: str,
    model: str,
    effort: str | None,
    schema: JsonObject,
    agent_path: str,
) -> list[str]:
    """The headless ``grok`` argv for one pass; the prompt is the agent
    definition at ``agent_path`` (see ``GrokReviewer._stage``).

    Kept a pure function of its inputs so tests can pin the exact
    security-relevant flag set without spawning anything.
    """
    cmd = [
        grok_bin,
        "--agent", agent_path,
        # No PR text may reach the user turn: there grok attaches the
        # file behind each "@path" and replaces a long text by an excerpt
        # and a file for the model to read (prompt_parser.rs and
        # acp_session_impl/prompt_offload.rs in
        # https://github.com/xai-org/grok-build/tree/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-shell/src/session).
        "--single", "Do the task your system prompt gives you.",
        "--output-format", "streaming-messages-json",
        "--json-schema", json.dumps(schema),
        "--model", model,
        "--tools", "search_tool,use_tool",
        # use_tool can load a file as an MCP tool's arguments
        # (https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-tools/src/implementations/use_tool/input.rs);
        # this rule refused all 7 spellings of a path tried on grok 1.0.44,
        # 2026-10-01.
        "--deny", "Read(/**)",
        # Safe despite the name: what is left to approve are calls of MCP
        # tools, and a call that waited for approval cancelled the run in
        # 2 of 2 runs on grok 1.0.44, 2026-10-01.
        "--always-approve",
    ]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def result_event(jsonl: str) -> JsonObject | None:
    """The ``result`` line of a streaming-messages-json run, ``None`` when
    the run never produced one."""
    return last_event(jsonl, lambda event: event.get("type") == "result")


class GrokReviewer(CliReviewer):
    """One headless ``grok`` pass of a ``RoleSpec`` (see ``CliReviewer``).

    grok specifics: ``auth.json`` and a ``config.toml`` naming the MCP
    shell are copied into a private ``$GROK_HOME``; the prompt is the
    system prompt of an agent definition; the schema-forced answer is
    the ``structured_output`` of the ``result`` line.
    """

    backend = "grok"
    efforts = GROK_EFFORTS
    # Under /work = the image WORKDIR.
    run_dir = "/work/.grok-run"
    credential_dir = CONTAINER_GROK_HOME
    credential_name = CREDENTIAL_NAME
    default_home = ".grok"
    image_binaries = ("grok",)
    home_help = (
        f"Wrapper-side directory holding grok's {CREDENTIAL_NAME}, copied "
        f"into the grok container. Default: grok's own ~/{default_home}.")
    login_hint = f"run `grok login` and copy its {CREDENTIAL_NAME} to --grok-home"
    # The model catalog grok 1.0.44 fetched on 2026-10-01 gives grok-4.7 a
    # context window of 256000 tokens, and 100 KB of C source cost about
    # 32000 of them that day; these are our choice, together about 96000
    # tokens at that rate
    max_patch_bytes = 150_000
    max_bundle_bytes = 150_000

    def _stage(self, scratch: str, prompt: str, *, use_shell: bool,
               machine_labels: tuple[str, ...]) -> Staged:
        # A file: grok lost a definition piped to it as /dev/stdin
        # (observed on grok 1.0.44, 2026-10-01).
        agent = Path(scratch) / AGENT_NAME
        agent.write_text(
            "---\n"
            "name: fairy\n"
            "description: one pass of fairy\n"
            "promptMode: full\n"
            "discoverSkills: false\n"
            "agentsMd: false\n"
            "---\n"
            + prompt.replace("${", DELIMITER_START),
            encoding="utf-8")
        staged = Staged(
            build_grok_command(
                grok_bin=self.cli_bin, model=self.model, effort=self.effort,
                schema=self.role.schema["schema"],
                agent_path=f"{self.run_dir}/{AGENT_NAME}"),
            {"GROK_HOME": CONTAINER_GROK_HOME,
             "GROK_DISABLE_AUTOUPDATER": "1",
             "GROK_TELEMETRY_ENABLED": "false",
             "GROK_MANAGED_MCPS_ENABLED": "false"},
            "", [(agent, self.run_dir)])
        if not use_shell:
            return staged
        config = (
            # grok's default cap on an MCP result is 20000 bytes
            # (https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-pager/docs/user-guide/07-mcp-servers.md),
            # and a cut result pointed the model at a copy it has no tool
            # to read (grok 1.0.44, 2026-10-01); the wrapper bounds the
            # shell output.
            "[mcp]\n"
            "max_output_bytes = 1073741824\n"
            "[mcp_servers.shell]\n"
            'command = "python3"\n'
            # A JSON array of strings is a TOML one, but for a raw DEL
            # (https://toml.io/en/v1.0.0#string).
            "args = " + json.dumps(
                self._bridge_args(machine_labels),
                ensure_ascii=False).replace("\x7f", "\\u007f") + "\n"
            f"tool_timeout_sec = {MCP_TOOL_TIMEOUT_S}\n")
        logger.debug("grok config.toml:\n%s", config)
        config_local = Path(scratch) / "config.toml"
        config_local.write_text(config, encoding="utf-8")
        staged.files.append((config_local, CONTAINER_GROK_HOME))
        return staged

    def _run_summary(self, proc: subprocess.CompletedProcess
                     ) -> tuple[JsonObject, str]:
        result = result_event(proc.stdout) or {}
        usage, errors = result.get("usage"), result.get("errors")
        return (usage if isinstance(usage, dict) else {},
                "\n".join(map(str, errors)) if isinstance(errors, list) else "")

    def _final_object(self, container: CliContainer,
                      proc: subprocess.CompletedProcess,
                      error_text: str) -> object:
        result = result_event(proc.stdout)
        if result is None:
            raise RuntimeError(
                f"{self.name}: grok produced no result line "
                f"(rc={proc.returncode}); stderr: "
                f"{proc.stderr.strip()[-2000:] or '-'}"
            )
        # What grok 1.0.44 reported, before any turn, without a credential
        # and with tokens the server rejected (observed 2026-10-01); a
        # revoked login has not been observed. After a turn the errors
        # can hold the schema validator's message about the model's answer
        # (https://github.com/xai-org/grok-build/blob/2bdd1d6a6369de0e8c68132ea4539e9abd9e14a8/crates/codegen/xai-grok-shell/src/session/acp_session_impl/turn.rs).
        if not result.get("num_turns") and any(
                mark in error_text for mark in
                ("Not signed in", "Invalid or expired credentials")):
            raise self._invalidate_credential(error_text)
        if "402 Payment Required" in error_text:
            raise quota_exhausted(
                self.quota_key,
                f"{self.name}: grok balance exhausted; errors: {error_text}")
        if "structured_output" not in result:
            raise GrokTurnFailed(
                f"{self.name}: grok ended with {result.get('subtype')!r} and "
                f"no structured_output; errors: {error_text or '-'}"
            )
        # grok 1.0.44 answered in schema to a system prompt of its own
        # when it could not read the agent definition (2026-10-01), so
        # what its session recorded, under the working directory's
        # percent-encoded name, has to be the staged prompt.
        recorded = container.read_file(
            f"{CONTAINER_GROK_HOME}/sessions/%2Fwork/"
            f"{result.get('session_id')}/system_prompt.txt",
            max_bytes=MAX_PROMPT_BYTES)
        staged = container.read_file(
            f"{self.run_dir}/{AGENT_NAME}", max_bytes=MAX_PROMPT_BYTES)
        if recorded is None or staged is None or recorded.strip() != \
                staged.partition("\n---\n")[2].replace(DELIMITER_START, "${").strip():
            raise RuntimeError(
                f"{self.name}: grok did not run on the staged system prompt")
        return result["structured_output"]
