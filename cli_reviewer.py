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

The ``Reviewer`` shared by the CLI coding-agent backends (codex, agy).

One pass of a ``RoleSpec`` is one CLI process in an ephemeral container
on the backend's podman host (see ``cli_container``): the role prompt
goes in on stdin with the patch and source bundle inlined, the CLI
drives the model, the review-container shell reaches it as an MCP tool
through ``mcp_shell_bridge.py`` -> ``relay.py`` -> ``serve_dispatch``,
and the schema-forced final object comes back for ``role.validate``.

What belongs here: everything the backends do alike -- the prompt, the
container and relay lifecycle, the credential copy-in and refresh
persistence, the concurrency slot, the debug dump, the poison and
cleanup paths, and the exceptions the pipeline treats as clean
provider-side stops. What does NOT belong here: a CLI's command line,
its output format and its error strings -- a backend supplies those
through ``_stage``, ``_stdin_text``, ``_run_summary`` and
``_final_object``.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from typing import Callable

from cli_container import CliContainer, CliShellRelay, persist_refreshed_auth
import concurrency
from common import JsonObject, dump_response_debug_artifacts
from llm_prompt import REVIEWER_ROLE, generate_llm_prompt
from llm_review_api import (BadModelOutput, BranchCollectionFailed,
                            ProviderTurnFailed, ReviewContext,
                            Reviewer, RoleSpec)
from podman_host import ShellHostSpec
import shell_tool

__all__ = [
    "BRIDGE_PATH",
    "MCP_TOOL_TIMEOUT_S",
    "CliAuthFailed",
    "CliReviewer",
    "CliUsageLimit",
    "Staged",
    "last_event",
    "parse_event_line",
]

logger = logging.getLogger(__name__)

_REPO_DIR = Path(__file__).resolve().parent
BRIDGE_PATH = _REPO_DIR / "mcp_shell_bridge.py"
_BRIDGE_CLIENT_PATH = _REPO_DIR / "shell_bridge_client.py"
_RELAY_PATH = _REPO_DIR / "containers" / "relay.py"

# Deliberately far above the real per-command cap, which is enforced
# wrapper-side (exec_shell_call clamps to --podman-exec-timeout).
MCP_TOOL_TIMEOUT_S = 86_400


class CliUsageLimit(RuntimeError):
    """The subscription's usage window is exhausted; do not retry."""


class CliAuthFailed(RuntimeError):
    """The CLI rejected the stored credentials; only a human login can
    restore service, so do not retry."""


@dataclass(frozen=True)
class Staged:
    """What ``_stage`` prepared for one run: the CLI argv, the container
    env of that process, and local files to copy in as ``(file,
    container directory)`` before it starts."""

    cmd: list[str]
    env: dict[str, str]
    files: list[tuple[Path, str]] = field(default_factory=list)


def parse_event_line(line: str) -> JsonObject | None:
    """One JSONL event, or ``None`` for a blank, non-JSON or non-object line."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        event = json.loads(line)
    except json.JSONDecodeError:
        return None
    return event if isinstance(event, dict) else None


def last_event(jsonl: str,
               wanted: Callable[[JsonObject], bool]) -> JsonObject | None:
    """The last JSONL event ``wanted`` accepts, or ``None``."""
    for line in reversed(jsonl.splitlines()):
        event = parse_event_line(line)
        if event is not None and wanted(event):
            return event
    return None


class CliReviewer(Reviewer):
    """One CLI pass of a ``RoleSpec`` behind the shared interface.

    Class attributes a backend sets:

    ``backend``
        the provider prefix (``"codex"``); also the concurrency slot,
        the prompt vendor and the debug-dump id prefix
    ``efforts``
        the ``@effort`` values the CLI accepts
    ``run_dir``
        the container directory for per-run files (schema, bridge, relay)
    ``credential_dir``
        the container directory the credential file is copied into
    ``login_hint``
        how a human restores a rejected credential

    ``run(ctx)`` raises ``CliUsageLimit`` on an exhausted plan window,
    ``CliAuthFailed`` after setting the host credential aside as
    ``<name>.invalid`` so later passes fail fast, ``ProviderTurnFailed``
    when the provider ended the turn, and ``BadModelOutput`` when the
    final object fails validation.
    """

    backend: str
    efforts: tuple[str, ...]
    run_dir: str
    credential_dir: str
    login_hint: str

    def __init__(
        self,
        model: str,
        *,
        name: str,
        role: RoleSpec = REVIEWER_ROLE,
        host: ShellHostSpec | None,
        image: str,
        credential_local: Path,
        exec_timeout_s: float = 600.0,
        effort: str | None = None,
        run_timeout_s: float = 0.0,
        verbose: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if effort is not None and effort not in self.efforts:
            raise ValueError(f"effort {effort!r} not in {self.efforts}")
        self.model = model
        self.name = name
        self.role = role
        # None means the CLI is unavailable (there is no local backend);
        # run() rejects it.
        self.host = host
        self.image = image
        self.credential_local = credential_local
        self.exec_timeout_s = exec_timeout_s
        self.effort = effort
        self.run_timeout_s = run_timeout_s  # 0 disables the watchdog
        self.verbose = verbose
        self.debug_dir = debug_dir

    @property
    def socket_path(self) -> str:
        return f"{self.run_dir}/shell.sock"

    @property
    def credential_container_path(self) -> str:
        return f"{self.credential_dir}/{self.credential_local.name}"

    def _prompt_features(self) -> set[str]:
        """Prompt features beyond ``source_bundle`` / ``podman_shell``."""
        return set()

    def _prompt_tail(self, use_shell: bool) -> str:
        """Backend-specific lines after the role's user texts; "" for none."""
        return ""

    @abstractmethod
    def _stage(self, scratch: str, *, use_shell: bool,
               machine_labels: tuple[str, ...]) -> Staged:
        """Prepare the run before the container starts.

        ``scratch`` is a local directory for files to copy in. With
        ``use_shell`` the CLI must be configured to spawn
        ``{run_dir}/mcp_shell_bridge.py --socket {socket_path}
        --machine L...`` as its MCP server; the base copies the bridge,
        its client and the relay into ``run_dir``.
        """

    def _stdin_text(self, prompt: str) -> str:
        """What the CLI reads on stdin for ``prompt``."""
        return prompt

    @abstractmethod
    def _run_summary(self, proc: subprocess.CompletedProcess
                     ) -> tuple[JsonObject, str]:
        """``(usage, error_text)`` of the finished process, for the log and
        the debug dump; must not raise."""

    @abstractmethod
    def _final_object(self, container: CliContainer,
                      proc: subprocess.CompletedProcess,
                      error_text: str) -> object:
        """The CLI's final object for ``role.validate``, or a raised
        ``CliUsageLimit`` / ``CliAuthFailed`` / ``ProviderTurnFailed`` /
        ``RuntimeError`` (``BadModelOutput`` when the CLI ran but its
        final message is not JSON)."""

    def _invalidate_credential(self, errors: str) -> CliAuthFailed:
        """Set the host credential aside as ``<name>.invalid`` and return
        the ``CliAuthFailed`` to raise."""
        invalidated = self.credential_local.with_name(
            self.credential_local.name + ".invalid")
        try:
            self.credential_local.replace(invalidated)
        except FileNotFoundError:
            pass  # a parallel pass already set it aside
        logger.critical(
            "%s: credentials rejected; %s set aside as %s so no further "
            "logins are attempted until a human logs in again (%s)", self.backend,
            self.credential_local, invalidated.name, self.login_hint)
        return CliAuthFailed(
            f"{self.name}: {self.backend} rejected the credentials; "
            f"{self.credential_local} set aside as {invalidated.name}, "
            f"{self.login_hint} to restore service; errors: {errors or '-'}"
        )

    def _build_prompt(self, ctx: ReviewContext, use_shell: bool) -> str:
        features = self._prompt_features()
        if ctx.source_bundle is not None:
            features.add("source_bundle")
        if use_shell:
            features.add("podman_shell")
        developer = generate_llm_prompt(
            role=self.role.name,
            vendor=self.backend,
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
        if tail := self._prompt_tail(use_shell):
            parts.append(tail)
        parts.append(
            "Your final message must be exactly one JSON object matching "
            "the configured output schema -- no surrounding text."
        )
        return "\n\n".join(parts)

    def run(self, ctx: ReviewContext) -> dict[str, object]:
        if self.host is None:
            raise RuntimeError(
                f"{self.name}: {self.backend} requires --{self.backend}-host; "
                f"there is no local {self.backend} backend."
            )
        use_shell = bool(ctx.machines) and ctx.open_shell is not None
        prompt = self._build_prompt(ctx, use_shell)
        scratch = tempfile.mkdtemp(prefix=f"fairy-{self.backend}-")  # local staging only
        container = CliContainer(
            image=self.image, host=self.host.host,
            memory=self.host.memory, cpus=self.host.cpus,
        )
        relay: CliShellRelay | None = None
        machine_labels = tuple(m.label for m in ctx.machines)
        try:
            schema_local = Path(scratch) / "output_schema.json"
            schema_local.write_text(
                json.dumps(self.role.schema["schema"]), encoding="utf-8")
            if not self.credential_local.is_file():
                raise RuntimeError(
                    f"{self.name}: {self.backend} credential "
                    f"{self.credential_local.name} not found at "
                    f"{self.credential_local} ({self.login_hint})"
                )
            staged = self._stage(scratch, use_shell=use_shell,
                                 machine_labels=machine_labels)

            container.start()
            container.put_file(self.credential_local, self.credential_dir)
            container.put_file(schema_local, self.run_dir)
            for local, dest_dir in staged.files:
                container.put_file(local, dest_dir)
            if use_shell:
                for local in (BRIDGE_PATH, _BRIDGE_CLIENT_PATH, _RELAY_PATH):
                    container.put_file(local, self.run_dir)
                relay = CliShellRelay(
                    container,
                    relay_container_path=f"{self.run_dir}/relay.py",
                    machine_labels=machine_labels,
                    open_shell=ctx.open_shell,
                    max_timeout_s=self.exec_timeout_s,
                    socket_path=self.socket_path,
                ).start()

            with concurrency.slot(self.backend):
                logger.info(
                    "%s start role=%s model=%s effort=%s shell=%s host=%s",
                    self.backend, self.role.name, self.model,
                    self.effort or "-", use_shell, self.host.host.ssh_dest,
                )
                started = time.monotonic()
                try:
                    proc = container.run(
                        staged.cmd, input_text=self._stdin_text(prompt),
                        env=staged.env,
                        timeout_s=self.run_timeout_s or None,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"{self.name}: {self.backend} exceeded "
                        f"--{self.backend}-timeout-seconds "
                        f"({self.run_timeout_s:.0f}s)"
                    )
                elapsed = time.monotonic() - started
                # The CLI may have rotated the OAuth tokens mid-run, and
                # --rm is about to discard the container's copy.
                persist_refreshed_auth(
                    container, self.credential_container_path,
                    self.credential_local)

            usage, error_text = self._run_summary(proc)
            logger.info(
                "%s done rc=%d dt=%.1fs usage=%s", self.backend,
                proc.returncode, elapsed,
                json.dumps(usage, ensure_ascii=False) if usage else "-",
            )
            self._dump_debug_artifacts(ctx, prompt, staged.cmd, proc, usage)
            result = self.role.validate(
                self._final_object(container, proc, error_text))
            ctx.collect_into(
                result, relay.opened_sessions() if relay is not None else [])
            if self.verbose:
                verdict = result.get("classification") or result.get("route") or "-"
                logger.debug("%s %s verdict=%s", self.backend,
                             self.role.name, verdict)
            return result
        except (CliAuthFailed, CliUsageLimit, ProviderTurnFailed):
            raise  # clean provider-side stop; the containers are not suspect
        except BadModelOutput:
            raise  # the CLI ran fine, only the final JSON was malformed
        except BranchCollectionFailed:
            raise  # the CLI ran fine; fairy's own branch packing failed
        except Exception:
            shell_tool.abort_if_cancelled()
            if shell_tool.halted():
                raise  # a sibling's halt tore this run down; not evidence

            # An abnormal exit may mean a PR-derived command tampered with the
            # review containers, so mark them for forensics rather than removal.
            if ctx.report_poisoned is not None and relay is not None:
                for session in relay.opened_sessions():
                    ctx.report_poisoned(session)
            raise
        finally:
            if relay is not None:
                relay.stop()
            container.stop()
            shutil.rmtree(scratch, ignore_errors=True)

    def _dump_debug_artifacts(
        self, ctx: ReviewContext, prompt: str, cmd: list[str],
        proc: subprocess.CompletedProcess, usage: JsonObject,
    ) -> None:
        """One dump file per run in the API backends' request/response
        format (``dump_response_debug_artifacts``), so CLI runs read the
        same way as openai:/anthropic: ones in the debug dir."""
        if not self.debug_dir:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dump_response_debug_artifacts(
            {
                "id": f"{self.backend}-{stamp}-{self.role.name}",
                "model": self.model,
                "events": [parse_event_line(line) or {"raw": line}
                           for line in proc.stdout.splitlines()
                           if line.strip()],
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "usage": usage,
            },
            {"model": self.model, "effort": self.effort, "cmd": cmd,
             "input": prompt},
            wrapper_request=ctx.request,
            debug_dir=self.debug_dir, verbose=self.verbose,
        )
