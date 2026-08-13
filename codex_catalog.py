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

The codex model catalog of one codex login (CODEX_HOME): resolving the
home directory and hardening a catalog into the override a review pass
may run with.

What does NOT belong here: container lifecycle (``codex_container``)
and the review pass itself (``codex_reviewer``).
"""

from __future__ import annotations

import copy
import os

from common import JsonObject

__all__ = [
    "harden_codex_catalog",
    "resolve_codex_home",
]


def resolve_codex_home(codex_home: str | None) -> str:
    return codex_home or os.environ.get("CODEX_HOME") \
        or os.path.expanduser("~/.codex")


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
    that code_mode otherwise consolidates). A code_mode model's JS-exec
    path is not lockable at the catalog layer; the podman container every
    codex pass runs in contains it.
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
