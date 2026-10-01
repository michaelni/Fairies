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

Codex-CLI pass of a ``RoleSpec``: the ``codex exec`` command line, the
model-catalog hardening, the ``--json`` event stream and codex's error
strings, on top of ``cli_reviewer.CliReviewer`` which runs the pinned
codex binary in a container on the ``--codex-host``.

``usage_limit_reached`` raises ``CliUsageLimit`` immediately and
run_parallel drops the pass, keeping surviving drafts.

``--concurrency codex:N`` caps how many passes run at once.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import subprocess
import sys

from cli_container import CliContainer
from cli_reviewer import (BRIDGE_PATH, MCP_TOOL_TIMEOUT_S, CliReviewer,
                          CliUsageLimit, Staged, parse_event_line)
from codex_catalog import harden_codex_catalog, resolve_codex_home
from common import JsonObject
from llm_prompt import REVIEWER_ROLE
from llm_review_api import (BadModelOutput, ProviderContentFlagged,
                            ProviderTurnFailed, RoleSpec)
from podman_host import ShellHostSpec

__all__ = [
    "CODEX_EFFORTS",
    "CODEX_WEB_SEARCH_MODES",
    "DEFAULT_CODEX_IMAGE",
    "CodexReviewer",
    "CodexTurnFailed",
    "build_codex_exec_command",
    "resolve_web_search",
]

logger = logging.getLogger(__name__)

DEFAULT_CODEX_IMAGE = "localhost/fairy-codex:latest"

# Under /work = the image WORKDIR.
CONTAINER_RUN_DIR = "/work/.codex-run"
CONTAINER_CODEX_HOME = "/work/.codex-home"

# The union across models; an unsupported pairing still fails server-side.
# 2026-07-17: gpt-5.6 rejected codex's documented "minimal" with
# "Supported values are: 'none', 'low', 'medium', 'high', and 'xhigh'".
CODEX_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max", "ultra")

# "cached"/"indexed" use OpenAI's maintained index rather than a live
# fetch. All run backend-side, no egress from the codex container.
CODEX_WEB_SEARCH_MODES = ("disabled", "cached", "indexed", "live")


def resolve_web_search(mode: str) -> str:
    """Map the wrapper's ``--web-search`` value to a codex ``web_search`` mode.

    The CLI spells "off"; codex's config spells "disabled". "live" and
    "cached" pass through unchanged.
    """
    return "disabled" if mode == "off" else mode

# Read-back caps for the untrusted codex container: an oversized file
# arrives truncated, so its JSON parse fails closed.
MAX_LAST_MESSAGE_BYTES = 8 * 1024 * 1024


class CodexTurnFailed(ProviderTurnFailed):
    """The provider ended the turn itself, so no final message exists."""


def _load_codex_catalog(codex_home: str | None) -> JsonObject | None:
    """Read ``<CODEX_HOME>/models_cache.json``, or ``None`` if absent or
    unparseable."""
    cache = os.path.join(resolve_codex_home(codex_home), "models_cache.json")
    try:
        with open(cache, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("codex: cannot read model catalog %s: %s", cache, exc)
        return None
    return data if isinstance(data, dict) else None


def build_codex_exec_command(
    *,
    codex_bin: str,
    model: str,
    effort: str | None,
    scratch_dir: str,
    schema_path: str,
    last_message_path: str,
    socket_path: str | None,
    machine_labels: tuple[str, ...],
    web_search: str = "cached",
    verbosity: str | None = None,
    reasoning_summary: str | None = None,
    catalog_override_path: str | None = None,
    bridge_python: str = sys.executable,
    bridge_path: str = str(BRIDGE_PATH),
) -> list[str]:
    """The full ``codex exec`` argv for one pass; prompt arrives on stdin.

    Kept a pure function of its inputs so tests can pin the exact
    security-relevant flag set without spawning anything.
    """
    cmd = [
        codex_bin, "exec",
        "--json",
        "--output-schema", schema_path,
        "--output-last-message", last_message_path,
        "--cd", scratch_dir,
        "--skip-git-repo-check",
        "--ignore-user-config",
        # Safe despite the name: the exec tools are removed below, leaving
        # the MCP shell into the review container as codex's only shell.
        "--sandbox", "danger-full-access",
        "--model", model,
        "-c", "features.shell_tool=false",
        "-c", "features.unified_exec=false",
        "-c", "features.apps=false",
        "-c", f'web_search="{web_search}"',
        "-c", "analytics.enabled=false",
    ]
    if verbosity:
        cmd += ["-c", f'model_verbosity="{verbosity}"']
    if reasoning_summary:
        cmd += ["-c", f'model_reasoning_summary="{reasoning_summary}"']
    if catalog_override_path:
        # A path, not inline JSON, and it replaces the built-in catalog
        # wholesale. See harden_codex_catalog.
        cmd += ["-c", f"model_catalog_json={catalog_override_path}"]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    if socket_path:
        bridge_args = ["--socket", socket_path]
        for label in machine_labels:
            bridge_args += ["--machine", label]
        cmd += [
            "-c", f'mcp_servers.shell.command="{bridge_python}"',
            "-c", "mcp_servers.shell.args="
                  + json.dumps([bridge_path, *bridge_args]),
            # "approve" generates no approval request; exec mode auto-cancels
            # any that are generated.
            "-c", 'mcp_servers.shell.default_tools_approval_mode="approve"',
            "-c", f"mcp_servers.shell.tool_timeout_sec={MCP_TOOL_TIMEOUT_S}",
        ]
    cmd.append("-")  # prompt from stdin; argv cannot hold a patch bundle
    return cmd


def summarize_codex_events(jsonl: str) -> tuple[JsonObject, str]:
    """Extract ``(usage, error_text)`` from a ``codex exec --json`` stream.

    ``usage`` is the last token-usage object seen (codex reports it on
    turn completion); ``error_text`` collects failure/error event text so
    quota exhaustion is detectable. Tolerant of unknown event shapes: the
    stream is only telemetry here, the review result travels via
    ``--output-last-message``.
    """
    usage: JsonObject = {}
    errors: list[str] = []
    for line in jsonl.splitlines():
        event = parse_event_line(line)
        if event is None:
            continue
        found = event.get("usage")
        if not isinstance(found, dict):
            found = (event.get("turn") or {}).get("usage") \
                if isinstance(event.get("turn"), dict) else None
        if isinstance(found, dict):
            usage = found
        event_type = str(event.get("type", ""))
        if "fail" in event_type or "error" in event_type:
            errors.append(json.dumps(event, ensure_ascii=False))
    return usage, "\n".join(errors)


def _is_usage_limit(error_text: str) -> bool:
    lowered = error_text.lower()
    return "usage_limit_reached" in lowered or "usage limit" in lowered


def _is_auth_failure(error_text: str) -> bool:
    # codex-cli 0.147.0's terminal credential failures (strings(1) over the
    # binary): dead refresh tokens end in "Please log out and sign in
    # again.", absent or incomplete credentials point at "codex login".
    # Its "Failed to refresh token:" can wrap a transient transport error,
    # so it deliberately does not match.
    lowered = error_text.lower()
    return "log out and sign in again" in lowered or "codex login" in lowered


class CodexReviewer(CliReviewer):
    """One ``codex exec`` pass of a ``RoleSpec`` (see ``CliReviewer``).

    Codex specifics: the model catalog is hardened host-side before
    every run (``harden_codex_catalog``) and the schema-forced final
    message is read from ``--output-last-message``; ``--json`` events on
    stdout are telemetry and the source of the error classification.
    """

    backend = "codex"
    efforts = CODEX_EFFORTS
    run_dir = CONTAINER_RUN_DIR
    credential_dir = CONTAINER_CODEX_HOME
    login_hint = "run `codex login` for the deployment's CODEX_HOME"

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        codex_bin: str = "codex",
        codex_home: str | None = None,
        codex_host: ShellHostSpec | None = None,
        codex_image: str = DEFAULT_CODEX_IMAGE,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
        web_search: str = "cached",
        verbosity: str | None = None,
        reasoning_summary: str | None = None,
        run_timeout_s: float = 0.0,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if web_search not in CODEX_WEB_SEARCH_MODES:
            raise ValueError(
                f"web_search {web_search!r} not in {CODEX_WEB_SEARCH_MODES}")
        super().__init__(
            model, name=name, role=role, host=codex_host, image=codex_image,
            credential_local=Path(resolve_codex_home(codex_home)) / "auth.json",
            exec_timeout_s=exec_timeout_s, effort=effort,
            run_timeout_s=run_timeout_s, verbose=verbose, debug_dir=debug_dir,
        )
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        self.web_search = web_search
        self.verbosity = verbosity
        self.reasoning_summary = reasoning_summary

    def _write_hardened_catalog(self, scratch: str) -> Path:
        """Write a host-hardened copy of codex's model catalog into scratch
        and return its path for ``-c model_catalog_json=``.

        Raises when there is no catalog or it does not contain this
        pass's model: codex never runs without the view_image /
        apply_patch hardening. See ``harden_codex_catalog``.
        """
        catalog = _load_codex_catalog(self.codex_home)
        if not isinstance(catalog, dict):
            raise RuntimeError(
                f"{self.name}: no codex model catalog cache; refusing to "
                "run codex without the view_image/apply_patch hardening "
                "(refresh: tools/refresh_codex_catalog.sh)"
            )
        models = catalog.get("models")
        entry = next(
            (m for m in models
             if isinstance(m, dict) and m.get("slug") == self.model),
            None,
        ) if isinstance(models, list) else None
        if entry is None:
            raise RuntimeError(
                f"{self.name}: model {self.model!r} absent from the codex "
                "catalog; refusing to run codex without the "
                "view_image/apply_patch hardening "
                "(refresh: tools/refresh_codex_catalog.sh)"
            )
        path = Path(scratch) / "hardened_catalog.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(harden_codex_catalog(catalog), f)
        os.chmod(path, 0o600)
        logger.info(
            "codex: hardened model catalog (view_image neutered; apply_patch "
            "removed) for %s", self.name,
        )
        return path

    def _prompt_features(self) -> set[str]:
        return {"web_search"} if self.web_search != "disabled" else set()

    def _stage(self, scratch: str, *, use_shell: bool,
               machine_labels: tuple[str, ...]) -> Staged:
        catalog_local = self._write_hardened_catalog(scratch)
        cmd = build_codex_exec_command(
            codex_bin=self.codex_bin,
            model=self.model,
            effort=self.effort,
            scratch_dir=CONTAINER_RUN_DIR,
            schema_path=f"{CONTAINER_RUN_DIR}/output_schema.json",
            last_message_path=f"{CONTAINER_RUN_DIR}/last_message.json",
            socket_path=self.socket_path if use_shell else None,
            machine_labels=machine_labels,
            catalog_override_path=f"{CONTAINER_RUN_DIR}/{catalog_local.name}",
            web_search=self.web_search,
            verbosity=self.verbosity,
            reasoning_summary=self.reasoning_summary,
            bridge_python="python3",
            bridge_path=f"{CONTAINER_RUN_DIR}/{BRIDGE_PATH.name}",
        )
        # The container env carries only CODEX_HOME; the wrapper's
        # OPENAI_API_KEY/CODEX_API_KEY are not forwarded by podman
        # exec, so codex cannot rebind to another credential.
        return Staged(cmd, {"CODEX_HOME": CONTAINER_CODEX_HOME},
                      [(catalog_local, CONTAINER_RUN_DIR)])

    def _run_summary(self, proc: subprocess.CompletedProcess
                     ) -> tuple[JsonObject, str]:
        return summarize_codex_events(proc.stdout)

    def _final_object(self, container: CliContainer,
                      proc: subprocess.CompletedProcess,
                      error_text: str) -> object:
        if _is_usage_limit(error_text) or _is_usage_limit(proc.stderr):
            raise CliUsageLimit(
                f"{self.name}: plan usage limit reached; details: "
                f"{error_text or proc.stderr.strip()[-500:]}"
            )
        # Treat the last-message file, not the exit code, as the
        # success signal: codex has exited 0 with no output.
        last_message = (container.read_file(
            f"{CONTAINER_RUN_DIR}/last_message.json",
            max_bytes=MAX_LAST_MESSAGE_BYTES) or "").strip()
        if not last_message:
            if _is_auth_failure(error_text) or _is_auth_failure(proc.stderr):
                raise self._invalidate_credential(
                    error_text or proc.stderr.strip()[-2000:])
            # A ``turn.failed`` event means the provider ended the turn
            # itself, which says nothing about the review containers.
            # Observed 2026-07-28: gpt-5.6-sol was refused mid-review of
            # PR #23750 with "flagged for possible cybersecurity risk".
            raise (
                ProviderContentFlagged
                if "flagged for possible cybersecurity risk" in error_text
                else CodexTurnFailed if '"turn.failed"' in error_text
                else RuntimeError)(
                f"{self.name}: codex exec produced no final message "
                f"(rc={proc.returncode}); errors: "
                f"{error_text or proc.stderr.strip()[-2000:] or '-'}"
            )
        try:
            return json.loads(last_message)
        except json.JSONDecodeError as exc:
            raise BadModelOutput(
                f"final message is not JSON despite --output-schema: {exc}"
            )
