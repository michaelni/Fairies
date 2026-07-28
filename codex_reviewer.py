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
one pinned ``codex exec`` per pass inside a container on the
``--codex-host`` (see ``codex_container``) and lets codex drive the model,
with the review-container shell reaching codex as an MCP tool via
``codex_bridge.py`` -> ``relay.py`` -> the wrapper's ``serve_dispatch``,
over a ``podman exec -i`` channel (no host-crossing socket).

``usage_limit_reached`` raises ``CodexUsageLimit`` immediately and
run_parallel drops the pass, keeping surviving drafts.

``--concurrency codex:N`` caps how many passes run at once.
"""

from __future__ import annotations

import copy
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from codex_container import (
    CONTAINER_CODEX_HOME,
    CONTAINER_RUN_DIR,
    RELAY_SOCKET_PATH,
    CodexContainer,
    CodexShellRelay,
)
import concurrency
from common import JsonObject, dump_response_debug_artifacts
from llm_prompt import REVIEWER_ROLE, generate_llm_prompt
from llm_review_api import BadModelOutput, ReviewContext, Reviewer, RoleSpec
from podman_host import ShellHostSpec
import shell_tool

__all__ = [
    "CODEX_EFFORTS",
    "CODEX_WEB_SEARCH_MODES",
    "CodexReviewer",
    "CodexTurnFailed",
    "CodexUsageLimit",
    "build_codex_exec_command",
    "harden_codex_catalog",
    "resolve_web_search",
]

logger = logging.getLogger(__name__)

DEFAULT_CODEX_IMAGE = "localhost/fairy-codex:latest"

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

# Deliberately far above the real per-command cap, which is enforced
# wrapper-side (exec_shell_call clamps to --podman-exec-timeout).
MCP_TOOL_TIMEOUT_S = 86_400

# Read-back caps for the untrusted codex container: an oversized file
# arrives truncated, so its JSON parse fails closed.
MAX_LAST_MESSAGE_BYTES = 8 * 1024 * 1024
MAX_AUTH_BYTES = 256 * 1024

_REPO_DIR = Path(__file__).resolve().parent
_BRIDGE_PATH = str(_REPO_DIR / "codex_bridge.py")
_BRIDGE_CLIENT_PATH = _REPO_DIR / "shell_bridge_client.py"
_RELAY_PATH = _REPO_DIR / "containers" / "relay.py"


def _resolve_codex_home(codex_home: str | None) -> str:
    return codex_home or os.environ.get("CODEX_HOME") \
        or os.path.expanduser("~/.codex")

class CodexUsageLimit(RuntimeError):
    """The usage window is exhausted; do not retry."""


class CodexTurnFailed(RuntimeError):
    """The provider ended the turn itself, so no final message exists."""


def _is_code_mode(tool_mode: object) -> bool:
    """Whether a catalog ``tool_mode`` puts the model in a code_mode runtime."""
    return isinstance(tool_mode, str) and tool_mode.startswith("code_mode")


def harden_codex_catalog(catalog: JsonObject) -> JsonObject:
    """Return a copy of a codex model catalog with the two direct host-file
    tools closed on every model entry.

    Fairy's codex only ever needs to drive the review container via the MCP
    shell tool; codex's own host-side tools are pure attack surface on a
    PR-derived (untrusted) prompt. The model catalog is the only lever codex
    exposes for them:

    * ``input_modalities`` loses ``image`` -- the ``view_image`` handler then
      rejects every call ("view_image is not allowed because you do not
      support image inputs"), so no local file is base64'd into the
      conversation. The tool stays listed but is inert.
    * ``apply_patch_tool_type`` -> ``None`` -- the ``apply_patch`` tool
      (which reads and writes host files) is not offered at all.

    ``tool_mode`` is deliberately left untouched: forcing a ``code_mode``
    model (gpt-5.6-*) to standard tool calling does not shrink its surface,
    it *explodes* it (``run``, ``spawn_agent``, multi-agent + plugin tools
    that code_mode otherwise consolidates). code_mode models are instead
    flagged by ``_write_hardened_catalog`` -- their JS-exec path is not
    lockable at the catalog layer and belongs behind the container boundary.
    """
    hardened = copy.deepcopy(catalog)
    models = hardened.get("models")
    if isinstance(models, list):
        for entry in models:
            if not isinstance(entry, dict):
                continue
            mods = entry.get("input_modalities")
            if isinstance(mods, list):
                entry["input_modalities"] = [m for m in mods if m != "image"]
            entry["apply_patch_tool_type"] = None
    return hardened


def _load_codex_catalog(codex_home: str | None) -> JsonObject | None:
    """Read ``<CODEX_HOME>/models_cache.json``, or ``None`` if absent or
    unparseable -- availability must not hinge on this inner lock.
    """
    cache = os.path.join(_resolve_codex_home(codex_home), "models_cache.json")
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
    bridge_path: str = _BRIDGE_PATH,
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
    with the role. Raises ``CodexUsageLimit`` on an exhausted plan window,
    ``CodexTurnFailed`` when the provider ended the turn, and
    ``BadModelOutput`` when the final message fails validation.
    """

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
        if effort is not None and effort not in CODEX_EFFORTS:
            raise ValueError(f"effort {effort!r} not in {CODEX_EFFORTS}")
        if web_search not in CODEX_WEB_SEARCH_MODES:
            raise ValueError(
                f"web_search {web_search!r} not in {CODEX_WEB_SEARCH_MODES}")
        self.model = model
        self.name = name
        self.role = role
        self.codex_bin = codex_bin
        self.codex_home = codex_home
        # None means codex is unavailable (there is no local codex); run()
        # rejects it.
        self.codex_host = codex_host
        self.codex_image = codex_image
        self.exec_timeout_s = exec_timeout_s
        self.effort = effort
        self.web_search = web_search
        self.verbosity = verbosity
        self.reasoning_summary = reasoning_summary
        self.run_timeout_s = run_timeout_s  # 0 disables the watchdog
        self.verbose = verbose
        self.debug_dir = debug_dir

    def _write_hardened_catalog(self, scratch: str) -> str | None:
        """Write a host-hardened copy of codex's model catalog into scratch.

        Returns its path for ``-c model_catalog_json=``, or ``None`` (logged)
        when the catalog cannot be sourced or does not contain this pass's
        model -- in which case the pass runs without the override rather than
        failing. See ``harden_codex_catalog``.
        """
        catalog = _load_codex_catalog(self.codex_home)
        if not isinstance(catalog, dict):
            logger.warning(
                "codex: no model catalog to harden; running without the "
                "view_image/apply_patch lock -- rely on container isolation "
                "for %s", self.name,
            )
            return None
        models = catalog.get("models")
        entry = next(
            (m for m in models
             if isinstance(m, dict) and m.get("slug") == self.model),
            None,
        ) if isinstance(models, list) else None
        if entry is None:
            logger.warning(
                "codex: model %r absent from cached catalog; skipping tool "
                "hardening override for %s", self.model, self.name,
            )
            return None
        if _is_code_mode(entry.get("tool_mode")):
            logger.warning(
                "codex: model %r uses code_mode (tool_mode=%r) -- a JS-exec "
                "path not containable by config; run it behind the podman "
                "boundary or switch to a tool_mode=None model. Hardening "
                "view_image/apply_patch only for %s.",
                self.model, entry.get("tool_mode"), self.name,
            )
        path = os.path.join(scratch, "hardened_catalog.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(harden_codex_catalog(catalog), f)
        os.chmod(path, 0o600)
        logger.info(
            "codex: hardened model catalog (view_image neutered; apply_patch "
            "removed) for %s", self.name,
        )
        return path

    def _build_prompt(self, ctx: ReviewContext, use_shell: bool) -> str:
        features: set[str] = set()
        if self.web_search != "disabled":
            features.add("web_search")
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
        if self.codex_host is None:
            raise RuntimeError(
                f"{self.name}: codex requires --codex-host; there is no "
                "local codex backend."
            )
        use_shell = bool(ctx.machines) and ctx.open_shell is not None
        prompt = self._build_prompt(ctx, use_shell)
        scratch = tempfile.mkdtemp(prefix="fairy-codex-")  # local staging only
        container = CodexContainer(
            image=self.codex_image, host=self.codex_host.host,
            memory=self.codex_host.memory, cpus=self.codex_host.cpus,
        )
        relay: CodexShellRelay | None = None
        machine_labels = tuple(m.label for m in ctx.machines)
        try:
            schema_local = Path(scratch) / "output_schema.json"
            schema_local.write_text(
                json.dumps(self.role.schema["schema"]), encoding="utf-8")
            catalog_local = self._write_hardened_catalog(scratch)
            auth_local = Path(_resolve_codex_home(self.codex_home)) / "auth.json"
            if not auth_local.is_file():
                raise RuntimeError(
                    f"{self.name}: codex auth.json not found at {auth_local} "
                    "(run `codex login` for the deployment's CODEX_HOME)"
                )

            container.start()
            container.put_file(auth_local, CONTAINER_CODEX_HOME)
            container.put_file(schema_local, CONTAINER_RUN_DIR)
            catalog_container: str | None = None
            if catalog_local:
                container.put_file(Path(catalog_local), CONTAINER_RUN_DIR)
                catalog_container = \
                    f"{CONTAINER_RUN_DIR}/{Path(catalog_local).name}"
            if use_shell:
                for local in (Path(_BRIDGE_PATH), _BRIDGE_CLIENT_PATH,
                              _RELAY_PATH):
                    container.put_file(local, CONTAINER_RUN_DIR)
                relay = CodexShellRelay(
                    container,
                    relay_container_path=f"{CONTAINER_RUN_DIR}/relay.py",
                    machine_labels=machine_labels,
                    open_shell=ctx.open_shell,
                    max_timeout_s=self.exec_timeout_s,
                ).start()

            cmd = build_codex_exec_command(
                codex_bin=self.codex_bin,
                model=self.model,
                effort=self.effort,
                scratch_dir=CONTAINER_RUN_DIR,
                schema_path=f"{CONTAINER_RUN_DIR}/output_schema.json",
                last_message_path=f"{CONTAINER_RUN_DIR}/last_message.json",
                socket_path=RELAY_SOCKET_PATH if use_shell else None,
                machine_labels=machine_labels,
                catalog_override_path=catalog_container,
                web_search=self.web_search,
                verbosity=self.verbosity,
                reasoning_summary=self.reasoning_summary,
                bridge_python="python3",
                bridge_path=f"{CONTAINER_RUN_DIR}/codex_bridge.py",
            )

            with concurrency.slot("codex"):
                logger.info(
                    "codex exec start role=%s model=%s effort=%s shell=%s host=%s",
                    self.role.name, self.model, self.effort or "-", use_shell,
                    self.codex_host.host.ssh_dest,
                )
                started = time.monotonic()
                # The container env carries only CODEX_HOME; the wrapper's
                # OPENAI_API_KEY/CODEX_API_KEY are not forwarded by podman
                # exec, so codex cannot rebind to another credential.
                try:
                    proc = container.run(
                        cmd, input_text=prompt,
                        env={"CODEX_HOME": CONTAINER_CODEX_HOME},
                        timeout_s=self.run_timeout_s or None,
                    )
                except subprocess.TimeoutExpired:
                    raise RuntimeError(
                        f"{self.name}: codex exec exceeded "
                        f"--codex-timeout-seconds ({self.run_timeout_s:.0f}s)"
                    )
                elapsed = time.monotonic() - started
                # Codex may have rotated the OAuth tokens mid-run, and --rm is
                # about to discard the container's copy.
                self._persist_refreshed_auth(container, auth_local)

            usage, error_text = summarize_codex_events(proc.stdout)
            logger.info(
                "codex exec done rc=%d dt=%.1fs usage=%s",
                proc.returncode, elapsed,
                json.dumps(usage, ensure_ascii=False) if usage else "-",
            )
            self._dump_debug_artifacts(ctx, prompt, cmd, proc, usage)
            if _is_usage_limit(error_text) or _is_usage_limit(proc.stderr):
                raise CodexUsageLimit(
                    f"{self.name}: plan usage limit reached; details: "
                    f"{error_text or proc.stderr.strip()[-500:]}"
                )
            # Treat the last-message file, not the exit code, as the
            # success signal: codex has exited 0 with no output.
            last_message = (container.read_file(
                f"{CONTAINER_RUN_DIR}/last_message.json",
                max_bytes=MAX_LAST_MESSAGE_BYTES) or "").strip()
            if not last_message:
                # A ``turn.failed`` event means the provider ended the turn
                # itself, which says nothing about the review containers.
                # Observed 2026-07-28: gpt-5.6-sol was refused mid-review of
                # PR #23750 with "flagged for possible cybersecurity risk".
                raise (CodexTurnFailed if '"turn.failed"' in error_text
                       else RuntimeError)(
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
        except (CodexUsageLimit, CodexTurnFailed):
            raise  # clean provider-side stop; the containers are not suspect
        except BadModelOutput:
            raise  # codex ran fine, only the final JSON was malformed
        except Exception:
            if shell_tool.cancelled():
                raise SystemExit("operator cancelled")
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

    def _persist_refreshed_auth(
        self, container: CodexContainer, auth_local: Path,
    ) -> None:
        """Copy a mid-run token refresh back to the host auth.json.

        Best-effort: reads the container's copy, and if it is valid JSON
        that differs from the host file, atomically replaces it (0600).
        A read/parse failure just logs -- the run already succeeded, so a
        stale host token surfaces as an auth error on a later run rather
        than failing this one.
        """
        try:
            refreshed = container.read_file(
                f"{CONTAINER_CODEX_HOME}/auth.json", max_bytes=MAX_AUTH_BYTES)
        except Exception:
            logger.warning("codex: could not read back auth.json for refresh "
                           "persistence", exc_info=True)
            return
        if not refreshed:
            return
        try:
            parsed = json.loads(refreshed)
        except ValueError:
            logger.warning("codex: refreshed auth.json is not valid JSON; "
                           "not persisting")
            return
        if not isinstance(parsed, dict):
            logger.warning("codex: refreshed auth.json is not a JSON object; "
                           "not persisting")
            return
        current = auth_local.read_text(encoding="utf-8") \
            if auth_local.is_file() else ""
        if refreshed == current:
            return
        tmp = auth_local.with_name(auth_local.name + ".tmp")
        fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, refreshed.encode("utf-8"))
        finally:
            os.close(fd)
        os.replace(tmp, auth_local)
        logger.info("codex: persisted refreshed auth.json from container")

    def _dump_debug_artifacts(
        self, ctx: ReviewContext, prompt: str, cmd: list[str],
        proc: subprocess.CompletedProcess, usage: JsonObject,
    ) -> None:
        """One dump file per run in the API backends' request/response
        format (``dump_response_debug_artifacts``), so codex runs read
        the same way as openai:/anthropic: ones in the debug dir."""
        if not self.debug_dir:
            return
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dump_response_debug_artifacts(
            {
                "id": f"codex-{stamp}-{self.role.name}",
                "model": self.model,
                "events": [_parse_event_line(line) or {"raw": line}
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
