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

The model-facing shell tool's vendor-neutral core: run one parsed
shell-tool call on a ``ContainerShellSession`` and return the JSON-able
result payload.

The OpenAI Responses loop and the Anthropic Messages loop share this; they
differ only in the wire envelope (``function_call`` / ``tool_use``) they
wrap around it, which stays in each per-vendor reviewer. The per-vendor
tool *schema* also stays with its vendor, since each SDK spells the schema
differently; only the ``{command, cwd?, timeout_seconds?}`` input shape and
the command->result step are common, and that is what lives here.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Callable, Sequence

from common import JsonObject
from podman_host import ContainerShellSession

__all__ = [
    "DEFAULT_SHELL_TIMEOUT_S",
    "exec_machine_call",
    "exec_shell_call",
    "run_session_commands",
]

logger = logging.getLogger(__name__)

DEFAULT_SHELL_TIMEOUT_S = 120.0


def exec_shell_call(
    session: ContainerShellSession,
    args: object,
    *,
    max_timeout_s: float,
    default_timeout_s: float = DEFAULT_SHELL_TIMEOUT_S,
) -> JsonObject:
    """Run one model shell-tool call on ``session``; return its result payload.

    ``args`` is the parsed tool input (``{command, cwd?, timeout_seconds?}``).
    It is model-supplied and thus attacker-influenceable, so its shape is
    validated here at the boundary: a malformed call yields an
    ``{"error": ...}`` payload (handed back to the model so it can recover)
    rather than raising. ``timeout_seconds`` is clamped to
    ``[1.0, max_timeout_s]``.
    """
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}
    command = args.get("command")
    if not isinstance(command, str) or not command.strip():
        return {"error": "missing or empty command"}
    cwd_raw = args.get("cwd")
    cwd = cwd_raw if isinstance(cwd_raw, str) and cwd_raw.strip() else None
    timeout_raw = args.get("timeout_seconds")
    try:
        timeout_s = float(timeout_raw) if timeout_raw is not None else default_timeout_s
    except (TypeError, ValueError):
        timeout_s = default_timeout_s
    timeout_s = max(1.0, min(timeout_s, max_timeout_s))

    logger.info("shell call timeout=%.1fs cwd=%s cmd=%s", timeout_s, cwd or "-", command[:500])
    result = session.exec(command, cwd=cwd, timeout_s=timeout_s)
    logger.info(
        "shell done rc=%d dt=%.3fs out_trunc=%s err_trunc=%s",
        result.exit_code, result.duration_s,
        result.stdout_truncated, result.stderr_truncated,
    )
    return {
        "exit_code": result.exit_code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_s": result.duration_s,
        "stdout_truncated": result.stdout_truncated,
        "stderr_truncated": result.stderr_truncated,
    }


def exec_machine_call(
    shells: dict[str, ContainerShellSession],
    machine_labels: Sequence[str],
    open_shell: Callable[[str], tuple[ContainerShellSession, str]],
    args: object,
    *,
    max_timeout_s: float,
    default_timeout_s: float = DEFAULT_SHELL_TIMEOUT_S,
    open_lock: threading.Lock | None = None,
) -> JsonObject:
    """Route one shell-tool call to the machine named by its ``machine`` arg.

    ``shells`` maps machine label -> live session; a missing label is
    instantiated via ``open_shell`` on first use (lifecycle/cleanup stays
    with the caller who supplied the callback). ``machine_labels[0]`` is
    the default machine. A non-default machine's ``--session-command``
    transcript rides its first result as ``setup_transcript`` (the default
    machine's transcript reaches the model via the prompt instead).
    ``machine`` is model-supplied: an unknown label yields an in-band
    ``{"error": ...}`` payload like exec_shell_call's own validation.

    Callers that share ``shells`` across threads (parallel OpenAI role
    passes) must pass ``open_lock`` so a racing lazy open cannot start two
    containers for one label and strand one of them.
    """
    machine = None
    if isinstance(args, dict):
        # Copy before popping: the Anthropic loop holds this dict by
        # reference in the conversation history it replays.
        args = dict(args)
        machine = args.pop("machine", None)
    if machine is None:
        machine = machine_labels[0]
    if machine not in machine_labels:
        return {
            "error": f"unknown machine {machine!r}; "
                     f"available: {', '.join(machine_labels)}"
        }
    first = False
    transcript = ""
    if machine not in shells:
        with open_lock if open_lock is not None else contextlib.nullcontext():
            if machine not in shells:
                shells[machine], transcript = open_shell(machine)
                first = True
    payload = exec_shell_call(
        shells[machine], args,
        max_timeout_s=max_timeout_s, default_timeout_s=default_timeout_s,
    )
    if first and machine != machine_labels[0] and transcript:
        payload["setup_transcript"] = transcript
    return payload


def run_session_commands(
    session: ContainerShellSession,
    commands: list[str],
    *,
    max_timeout_s: float,
) -> str:
    """Run operator-configured commands on ``session``; return a transcript.

    The transcript shows each command and its output the way a terminal
    would, for splicing into the model's prompt. Failures are recorded
    (``(exit N)``), not raised: the session stays usable either way.
    """
    parts: list[str] = []
    for command in commands:
        payload = exec_shell_call(
            session, {"command": command}, max_timeout_s=max_timeout_s,
            default_timeout_s=max_timeout_s,
        )
        block = f"$ {command}\n"
        block += str(payload.get("stdout") or "")
        stderr = str(payload.get("stderr") or "")
        if stderr:
            block += ("" if block.endswith("\n") else "\n") + stderr
        if not block.endswith("\n"):
            block += "\n"
        if payload.get("stdout_truncated") or payload.get("stderr_truncated"):
            block += "(output truncated)\n"
        exit_code = payload.get("exit_code")
        if exit_code not in (0, None):
            block += f"(exit {exit_code})\n"
        parts.append(block)
    return "\n".join(parts)
