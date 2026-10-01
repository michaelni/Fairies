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

agy's own tools (shell, file access, web) are denied by a pre-tool
hook whose reason points the model at the MCP shell into the review
containers, the only tool granted. The hook matters because headless
mode's own auto-denial of an unapproved tool ends the run with no
output, while a hook denial reaches the model as a tool error it can
recover from (observed on agy 1.2.13, 2026-09-30).

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
from cli_reviewer import (CliReviewer, CliUsageLimit, Staged, last_event,
                          parse_event_line)
from common import JsonObject
from llm_review_api import BadModelOutput, ProviderTurnFailed

__all__ = [
    "TOKEN_NAME",
    "AgyReviewer",
    "AgyTurnFailed",
    "build_agy_command",
    "exhausted_bucket",
    "summarize_agy_events",
]

logger = logging.getLogger(__name__)

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
# Tool names the pre-tool hook denies: run_command and write_to_file were
# observed on agy 1.2.13; the others are the terminal, file, web and
# browser step types the binary names. The hook matcher takes plain "|"
# alternation; the "\|" the docs show never matched (2026-09-30).
NATIVE_TOOLS_MATCHER = "|".join((
    "run_command", "send_command_input", "shell_exec", "write_to_file",
    "delete_directory", "read_url_content", "search_web", "open_browser_url",
    "browser_.*", "execute_browser_javascript", "run_extension_code",
))
HOOK_SCRIPT = "deny_native_tools.sh"

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
    as one stream-json user message (see ``AgyReviewer._stage``).

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
    event = last_event(jsonl, lambda e: e.get("event") == "result"
                       and isinstance(e.get("result"), dict))
    if event is None:
        return {}, None
    usage = event["result"].get("usage")
    return (usage if isinstance(usage, dict) else {}), event["result"]


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
    credential_name = TOKEN_NAME
    default_home = SETTINGS_RELATIVE_DIR
    image_binaries = ("agy",)
    home_help = (
        f"Wrapper-side directory holding agy's {TOKEN_NAME}, copied into "
        f"the agy container. Default: agy's own ~/{SETTINGS_RELATIVE_DIR}.")
    login_hint = f"log in with an interactive `agy` and copy its {TOKEN_NAME} to --agy-home"
    # agy's own input limit is unknown; these are our choice until its
    # behaviour and cost are understood
    max_patch_bytes = 300_000
    max_bundle_bytes = 450_000

    def _stage(self, scratch: str, prompt: str, *, use_shell: bool,
               machine_labels: tuple[str, ...]) -> Staged:
        schema = self._schema_file(scratch)
        cmd = build_agy_command(
            agy_bin=self.cli_bin, model=self.model, effort=self.effort,
            schema_path=f"{CONTAINER_RUN_DIR}/{schema.name}")
        config_dir = f"{CONTAINER_HOME}/{MCP_CONFIG_RELATIVE_DIR}"
        reason = (
            "This terminal is not the review machine and is disabled. Run "
            f"every command through the MCP tool {MCP_SERVER} (server "
            f"{MCP_SERVER}), which executes on the review machine."
            if use_shell else
            "No commands can be run in this review; answer from the text given."
        )
        hook_script = Path(scratch) / HOOK_SCRIPT
        hook_script.write_text(
            "#!/bin/sh\ncat >/dev/null\nprintf '%s' "
            + json.dumps(json.dumps({"decision": "deny", "reason": reason}))
            + "\n", encoding="utf-8")
        hooks = Path(scratch) / "hooks.json"
        hooks.write_text(json.dumps({"fairy-native-tools": {"PreToolUse": [{
            "matcher": NATIVE_TOOLS_MATCHER,
            "hooks": [{"type": "command",
                       "command": f"sh ~/{MCP_CONFIG_RELATIVE_DIR}/{HOOK_SCRIPT}",
                       "timeout": 10}]}]}}), encoding="utf-8")
        files = [(schema, CONTAINER_RUN_DIR),
                 (hook_script, config_dir), (hooks, config_dir)]
        if use_shell:
            mcp_config = Path(scratch) / "mcp_config.json"
            mcp_config.write_text(json.dumps({"mcpServers": {MCP_SERVER: {
                "command": "python3",
                "args": self._bridge_args(machine_labels)}}}), encoding="utf-8")
            # The grant must name server and tool (or "*"): a bare server
            # name is accepted and never matches
            # (https://github.com/google-antigravity/antigravity-cli/issues/916).
            settings = Path(scratch) / "settings.json"
            settings.write_text(json.dumps({"permissions": {
                "allow": [f"mcp({MCP_SERVER}/*)"]}}), encoding="utf-8")
            files += [(mcp_config, config_dir), (settings, self.credential_dir)]
        return Staged(
            cmd, CONTAINER_ENV,
            json.dumps({"event": "user", "message": {"content": prompt}}) + "\n",
            files)

    def _prompt_tail(self, use_shell: bool) -> str:
        return (
            f"Your own terminal and file tools are not the review machine; the "
            f"only way to run commands there is the MCP tool {MCP_SERVER}."
        ) if use_shell else ""

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
        cmd = [self.cli_bin, "-p", "/usage", "--output-format", "json"]
        logger.info("agy usage query after a failed run: %s", cmd)
        try:
            proc = container.run(cmd, env=CONTAINER_ENV, timeout_s=USAGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            logger.warning("agy usage query exceeded %.0fs", USAGE_TIMEOUT_S)
            return None
        return exhausted_bucket(proc.stdout, self.model)
