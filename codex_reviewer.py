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

Codex-CLI pass of a ``RoleSpec`` behind the shared ``Reviewer`` interface.

Unlike the API backends this one runs no tool loop of its own: it spawns
one pinned ``codex exec`` subprocess per pass and lets codex drive the
model, with the container shell reaching codex as an MCP tool via
``codex_bridge.py`` and the wrapper's shell dispatch socket.

``usage_limit_reached`` raises ``CodexUsageLimit`` immediately and
run_parallel drops the pass, keeping surviving drafts.

Passes are serialized on ``_CODEX_RUN_LOCK``.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

from common import JsonObject
from llm_prompt import REVIEWER_ROLE, generate_llm_prompt
from llm_review_api import BadModelOutput, ReviewContext, Reviewer, RoleSpec

__all__ = [
    "CODEX_EFFORTS",
    "CodexReviewer",
    "CodexUsageLimit",
    "build_codex_exec_command",
]

logger = logging.getLogger(__name__)

# codex -c model_reasoning_effort values (codex config reference).
CODEX_EFFORTS = ("minimal", "low", "medium", "high", "xhigh")

# codex-side per-MCP-tool-call watchdog. The real per-command cap is
# enforced wrapper-side (exec_shell_call clamps timeout_seconds to
# --podman-exec-timeout) and a first call may additionally pay for a lazy
# container open (podman/ssh bounded); this only needs to never be the
# binding constraint, so: one day.
MCP_TOOL_TIMEOUT_S = 86_400

_BRIDGE_PATH = str(Path(__file__).resolve().parent / "codex_bridge.py")

# queue here while other providers run in parallel around them.
_CODEX_RUN_LOCK = threading.Lock()


class CodexUsageLimit(RuntimeError):
    """The usage window is exhausted; do not retry."""


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
        "--sandbox", "read-only",
        "--model", model,
        # No model-chosen execution on this machine: the local shell tool
        # is removed from the tool list outright (see module docstring).
        "-c", "features.shell_tool=false",
        # OpenAI-side search only; also pins against a default change.
        "-c", f'web_search="{web_search}"',
        # The Statsig OTLP metrics ping is codex's only egress besides
        # the model API and token refresh; turn it off.
        "-c", "analytics.enabled=false",
    ]
    if effort:
        cmd += ["-c", f'model_reasoning_effort="{effort}"']
    if socket_path:
        bridge_args = ["--socket", socket_path]
        for label in machine_labels:
            bridge_args += ["--machine", label]
        cmd += [
            "-c", f'mcp_servers.shell.command="{sys.executable}"',
            "-c", "mcp_servers.shell.args="
                  + json.dumps([_BRIDGE_PATH, *bridge_args]),
            # "approve" never generates an approval request, which exec
            # mode would auto-cancel (openai/codex#24135).
            "-c", 'mcp_servers.shell.default_tools_approval_mode="approve"',
            "-c", f"mcp_servers.shell.tool_timeout_sec={MCP_TOOL_TIMEOUT_S}",
        ]
    cmd.append("-")  # prompt from stdin; argv cannot hold a patch bundle
    return cmd


def _parse_event_line(line: str) -> JsonObject | None:
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


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
        event = _parse_event_line(line)
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


class CodexReviewer(Reviewer):
    """One ``codex exec`` pass of a ``RoleSpec`` behind the shared interface.

    ``run(ctx)`` builds the same role prompt as the API backends (inlining
    patch and source bundle as text, like Anthropic), runs the pinned
    codex binary against it and validates the schema-forced last message
    with the role. Raises ``CodexUsageLimit`` on an exhausted plan window
    and ``BadModelOutput`` when the final message fails validation.
    """

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        codex_bin: str = "codex",
        codex_home: str | None = None,
        effort: str | None = None,
        run_timeout_s: float = 0.0,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if effort is not None and effort not in CODEX_EFFORTS:
            raise ValueError(f"effort {effort!r} not in {CODEX_EFFORTS}")
        self.model = model
        self.name = name
        self.role = role
        self.codex_bin = codex_bin
        # Where codex keeps auth.json etc.; None inherits the process env
        # (a set CODEX_HOME or codex's ~/.codex default).
        self.codex_home = codex_home
        self.effort = effort
        # 0 disables the whole-subprocess watchdog (a pass legitimately
        # runs for however long the model reasons and builds).
        self.run_timeout_s = run_timeout_s
        self.verbose = verbose
        self.debug_dir = debug_dir

    def _build_prompt(self, ctx: ReviewContext, use_shell: bool) -> str:
        features: set[str] = {"web_search"}
        if ctx.source_bundle is not None:
            features.add("source_bundle")
        if use_shell:
            features.add("podman_shell")
        developer = generate_llm_prompt(
            role=self.role.name,
            vendor="codex",
            model=self.model,
            features=features,
            repo_roots=ctx.repo_roots,
            container_repo_mounts=ctx.repo_mount_paths,
            machines=ctx.machines,
            reviewer_username=ctx.reviewer_username,
            project_facts=ctx.project_facts,
            ci_triage_mode=ctx.ci_triage_mode,
            **self.role.prompt_kwargs,
        )
        user_texts = self.role.user_texts(ctx)
        parts = [developer, user_texts[0]]
        if ctx.patch_text:
            parts.append(ctx.patch_text)
        if ctx.source_bundle is not None:
            parts.append(ctx.source_bundle)
        parts.extend(user_texts[1:])
        parts.append(
            "Your final message must be exactly one JSON object matching "
            "the configured output schema -- no surrounding text."
        )
        return "\n\n".join(parts)

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        use_shell = bool(ctx.shell_socket_path) and bool(ctx.machines)
        prompt = self._build_prompt(ctx, use_shell)
        scratch = tempfile.mkdtemp(prefix="fairy-codex-")
        try:
            schema_path = os.path.join(scratch, "output_schema.json")
            last_message_path = os.path.join(scratch, "last_message.json")
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(self.role.schema["schema"], f)
            cmd = build_codex_exec_command(
                codex_bin=self.codex_bin,
                model=self.model,
                effort=self.effort,
                scratch_dir=scratch,
                schema_path=schema_path,
                last_message_path=last_message_path,
                socket_path=ctx.shell_socket_path if use_shell else None,
                machine_labels=tuple(m.label for m in ctx.machines),
            )
            with _CODEX_RUN_LOCK:
                logger.info(
                    "codex exec start role=%s model=%s effort=%s shell=%s",
                    self.role.name, self.model, self.effort or "-", use_shell,
                )
                started = time.monotonic()
                env = None
                if self.codex_home:
                    env = dict(os.environ, CODEX_HOME=self.codex_home)
                try:
                    proc = subprocess.run(
                        cmd, input=prompt, capture_output=True, text=True,
                        timeout=self.run_timeout_s or None, env=env,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"{self.name}: codex exec exceeded "
                        f"--codex-timeout-seconds ({self.run_timeout_s:.0f}s)"
                    )
                elapsed = time.monotonic() - started

            usage, error_text = summarize_codex_events(proc.stdout)
            logger.info(
                "codex exec done rc=%d dt=%.1fs usage=%s",
                proc.returncode, elapsed,
                json.dumps(usage, ensure_ascii=False) if usage else "-",
            )
            self._dump_debug_artifacts(prompt, proc)
            if _is_usage_limit(error_text) or _is_usage_limit(proc.stderr):
                raise CodexUsageLimit(
                    f"{self.name}: plan usage limit reached; details: "
                    f"{error_text or proc.stderr.strip()[-500:]}"
                )
            # The known codex bug of exiting 0 with no output makes the
            # last-message file, not the exit code, the success signal.
            last_message = ""
            if os.path.exists(last_message_path):
                with open(last_message_path, encoding="utf-8") as f:
                    last_message = f.read().strip()
            if not last_message:
                raise RuntimeError(
                    f"{self.name}: codex exec produced no final message "
                    f"(rc={proc.returncode}); errors: "
                    f"{error_text or proc.stderr.strip()[-2000:] or '-'}"
                )
            try:
                result_obj = json.loads(last_message)
            except json.JSONDecodeError as exc:
                raise BadModelOutput(
                    f"final message is not JSON despite --output-schema: {exc}"
                )
            result = self.role.validate(result_obj)
            if self.verbose:
                verdict = result.get("classification") or result.get("route") or "-"
                logger.debug("codex %s verdict=%s", self.role.name, verdict)
            return result
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _dump_debug_artifacts(
        self, prompt: str, proc: subprocess.CompletedProcess,
    ) -> None:
        if not self.debug_dir:
            return
        try:
            os.makedirs(self.debug_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            base = os.path.join(
                self.debug_dir, f"codex-{stamp}-{self.role.name}",
            )
            for suffix, text in (
                ("prompt.txt", prompt),
                ("events.jsonl", proc.stdout),
                ("stderr.txt", proc.stderr),
            ):
                with open(f"{base}-{suffix}", "w", encoding="utf-8") as f:
                    f.write(text)
        except OSError:
            logger.exception("codex debug artifact dump failed")
