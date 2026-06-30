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

import logging

from common import JsonObject
from podman_host import ContainerShellSession

__all__ = ["DEFAULT_SHELL_TIMEOUT_S", "exec_shell_call"]

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
