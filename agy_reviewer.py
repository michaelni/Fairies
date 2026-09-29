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

Antigravity-CLI (``agy``) pass of a ``RoleSpec``: the headless ``agy``
command line, the per-run ``$HOME`` it reads its token, settings and MCP
servers from, the stream-json output and agy's error strings, on top of
``cli_reviewer.CliReviewer`` which runs the pinned agy binary in a
container on the ``--agy-host``.

agy's own tools (shell, file access, web) stay soft-denied: headless
mode auto-denies every tool without an allow rule, and the only rule
written grants the MCP shell into the review containers.

An authentication prompt is a credential failure: the token is set
aside and a human restores it. agy shows the same prompt with a valid
token while Google is unreachable, so a network outage takes that path
too. A run that ends without a final object otherwise is checked
against ``agy -p /usage``: an exhausted quota is a ``CliUsageLimit``,
anything else a ``ProviderTurnFailed`` that takes the retry and
fallback path.

``--concurrency agy:N`` caps how many passes run at once.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import subprocess

from cli_container import CliContainer
from cli_reviewer import (BRIDGE_PATH, CliReviewer, CliUsageLimit, Staged,
                          parse_event_line)
from common import JsonObject
from llm_prompt import REVIEWER_ROLE
from llm_review_api import BadModelOutput, ProviderTurnFailed, RoleSpec
from podman_host import ShellHostSpec

__all__ = [
    "DEFAULT_AGY_IMAGE",
    "TOKEN_NAME",
    "AgyReviewer",
    "AgyTurnFailed",
    "build_agy_command",
    "exhausted_bucket",
    "summarize_agy_events",
]

logger = logging.getLogger(__name__)

DEFAULT_AGY_IMAGE = "localhost/fairy-agy:latest"

# agy 1.2.12 --help
AGY_EFFORTS = ("low", "medium", "high", "max")

# Under /work = the image WORKDIR.
CONTAINER_RUN_DIR = "/work/.agy-run"
# Outside the workspace: agy's default permissions let the model read
# workspace files, and this is where its token lives.
CONTAINER_HOME = "/agy-home"
# agy 1.2.13 spawns a self-update in the background of every run unless
# this variable is exactly "true" ("1" does not disable it; observed
# 2026-09-29); the image pins the binary, so no run may replace it.
CONTAINER_ENV = {"HOME": CONTAINER_HOME, "AGY_CLI_DISABLE_AUTO_UPDATE": "true"}

# agy 1.2.12 on Linux keeps the OAuth token as a plain file next to its
# settings under $HOME/.gemini/antigravity-cli and reads MCP servers
# from $HOME/.gemini/config/mcp_config.json.
TOKEN_NAME = "antigravity-oauth-token"
SETTINGS_RELATIVE_DIR = ".gemini/antigravity-cli"
MCP_CONFIG_RELATIVE_DIR = ".gemini/config"
MCP_SERVER = "shell"

# agy 1.2.12 prints this, waits 60 s for a login code and exits 1 with
# the result error "authentication failed or timed out" both without a
# token and with a valid token while Google is unreachable (observed
# 2026-09-28 and 2026-09-29); a revoked token has not been observed.
AUTH_REQUIRED_MARK = "Authentication required"

USAGE_TIMEOUT_S = 120.0


class AgyTurnFailed(ProviderTurnFailed):
    """agy ended the run without a final object: a non-SUCCESS status, or a
    tool call it denied left the model with nothing to answer from."""


def build_agy_command(
    *,
    agy_bin: str,
    model: str,
    effort: str | None,
    schema_path: str,
) -> list[str]:
    """The headless ``agy`` argv for one pass; the prompt arrives on stdin
    as one stream-json user message (see ``AgyReviewer._stdin_text``).

    Kept a pure function of its inputs so tests can pin the exact flag
    set without spawning anything.
    """
    cmd = [
        agy_bin,
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--json-schema", schema_path,
        "--model", model,
        # PR text in the prompt must not expand as a slash command or skill.
        "--disable-slash-commands",
    ]
    if effort:
        cmd += ["--effort", effort]
    return cmd


def summarize_agy_events(jsonl: str) -> tuple[JsonObject, JsonObject | None]:
    """``(usage, result)`` of an ``--output-format stream-json`` run: the
    token usage and the payload of the last ``result`` event, ``None``
    when the run never produced one. Tolerant of unknown event shapes."""
    for line in reversed(jsonl.splitlines()):
        event = parse_event_line(line)
        if event is not None and event.get("event") == "result" \
                and isinstance(event.get("result"), dict):
            result = event["result"]
            usage = result.get("usage")
            return (usage if isinstance(usage, dict) else {}), result
    return {}, None


def exhausted_bucket(usage_json: str, model: str) -> JsonObject | None:
    """The usage bucket of ``model``'s group that has nothing left, from
    ``agy -p /usage --output-format json`` output; ``None`` when every
    bucket of that group has quota left or the output names no group for
    the model.

    A group is the model's when the slug's first word is in the group
    name: agy 1.2.12 groups ``gemini-*`` under "Gemini Models" and
    ``claude-*`` / ``gpt-*`` under "Claude and GPT models".
    """
    family = model.split("-", 1)[0].lower()
    for line in usage_json.splitlines():
        envelope = parse_event_line(line)
        if envelope is None:
            continue
        command = envelope.get("command")
        groups = command.get("data", {}).get("groups") if isinstance(command, dict) else None
        for group in groups if isinstance(groups, list) else ():
            if not isinstance(group, dict) or family not in str(group.get("name", "")).lower():
                continue
            for bucket in group.get("buckets") or ():
                if isinstance(bucket, dict) and bucket.get("remaining_fraction") == 0:
                    return bucket
    return None


class AgyReviewer(CliReviewer):
    """One headless ``agy`` pass of a ``RoleSpec`` (see ``CliReviewer``).

    agy specifics: the token, ``settings.json`` (the MCP allow rule) and
    ``mcp_config.json`` are copied into a private ``$HOME``; the prompt
    goes in as a stream-json user message; the schema-forced answer is
    the ``structured_output`` of the ``result`` event.
    """

    backend = "agy"
    efforts = AGY_EFFORTS
    run_dir = CONTAINER_RUN_DIR
    credential_dir = f"{CONTAINER_HOME}/{SETTINGS_RELATIVE_DIR}"
    login_hint = f"log in with an interactive `agy` and copy its {TOKEN_NAME} to --agy-home"

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        agy_bin: str = "agy",
        agy_home: str | None = None,
        agy_host: ShellHostSpec | None = None,
        agy_image: str = DEFAULT_AGY_IMAGE,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
        run_timeout_s: float = 0.0,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        super().__init__(
            model, name=name, role=role, host=agy_host, image=agy_image,
            credential_local=(Path(agy_home) if agy_home
                              else Path.home() / SETTINGS_RELATIVE_DIR) / TOKEN_NAME,
            exec_timeout_s=exec_timeout_s, effort=effort,
            run_timeout_s=run_timeout_s, verbose=verbose, debug_dir=debug_dir,
        )
        self.agy_bin = agy_bin
        self.agy_home = agy_home

    def _stage(self, scratch: str, *, use_shell: bool,
               machine_labels: tuple[str, ...]) -> Staged:
        cmd = build_agy_command(
            agy_bin=self.agy_bin, model=self.model, effort=self.effort,
            schema_path=f"{CONTAINER_RUN_DIR}/output_schema.json")
        files = []
        if use_shell:
            bridge_args = [f"{CONTAINER_RUN_DIR}/{BRIDGE_PATH.name}",
                           "--socket", self.socket_path]
            for label in machine_labels:
                bridge_args += ["--machine", label]
            mcp_config = Path(scratch) / "mcp_config.json"
            mcp_config.write_text(json.dumps({"mcpServers": {MCP_SERVER: {
                "command": "python3", "args": bridge_args}}}), encoding="utf-8")
            # The grant must name server and tool (or "*"): a bare server
            # name is accepted and never matches
            # (https://github.com/google-antigravity/antigravity-cli/issues/916).
            settings = Path(scratch) / "settings.json"
            settings.write_text(json.dumps({"permissions": {
                "allow": [f"mcp({MCP_SERVER}/*)"]}}), encoding="utf-8")
            files = [(mcp_config, f"{CONTAINER_HOME}/{MCP_CONFIG_RELATIVE_DIR}"),
                     (settings, self.credential_dir)]
        return Staged(cmd, CONTAINER_ENV, files)

    def _stdin_text(self, prompt: str) -> str:
        return json.dumps({"event": "user",
                           "message": {"content": prompt}}) + "\n"

    def _run_summary(self, proc: subprocess.CompletedProcess
                     ) -> tuple[JsonObject, str]:
        usage, result = summarize_agy_events(proc.stdout)
        return usage, str((result or {}).get("error") or "")

    def _final_object(self, container: CliContainer,
                      proc: subprocess.CompletedProcess,
                      error_text: str) -> object:
        if AUTH_REQUIRED_MARK in proc.stderr:
            raise self._invalidate_credential(
                error_text or proc.stderr.strip()[-2000:])
        _, result = summarize_agy_events(proc.stdout)
        if result is None or result.get("status") != "SUCCESS":
            bucket = self._exhausted_bucket(container)
            if bucket is not None:
                raise CliUsageLimit(
                    f"{self.name}: agy quota exhausted for {self.model} "
                    f"({bucket.get('name')}, resets {bucket.get('reset_time')})"
                )
        if result is None:
            raise RuntimeError(
                f"{self.name}: agy produced no result event "
                f"(rc={proc.returncode}); stderr: "
                f"{proc.stderr.strip()[-2000:] or '-'}"
            )
        denied = result.get("denied_actions")
        if result.get("status") != "SUCCESS" or \
                (denied and "structured_output" not in result):
            raise AgyTurnFailed(
                f"{self.name}: agy ended with status {result.get('status')!r}"
                f"{' after denying ' + json.dumps(denied) if denied else ''}"
                f"; error: {error_text or '-'}"
            )
        if denied:
            logger.warning("%s: agy denied %s; the review was produced "
                           "without them", self.name, json.dumps(denied))
        if "structured_output" not in result:
            raise BadModelOutput(
                f"{self.name}: no structured_output despite --json-schema; "
                f"response starts: {str(result.get('response'))[:200]!r}"
            )
        return result["structured_output"]

    def _exhausted_bucket(self, container: CliContainer) -> JsonObject | None:
        """``exhausted_bucket`` of a ``/usage`` query in the run's container;
        ``None`` when the query does not answer in time."""
        cmd = [self.agy_bin, "-p", "/usage", "--output-format", "json"]
        logger.info("agy usage query after a failed run: %s", cmd)
        try:
            proc = container.run(cmd, env=CONTAINER_ENV, timeout_s=USAGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("agy usage query exceeded %.0fs", USAGE_TIMEOUT_S)
            return None
        return exhausted_bucket(proc.stdout, self.model)
