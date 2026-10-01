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

The CLI coding-agent backends, as the one table of them.

What belongs here: which backends exist. What does NOT belong here:
anything about one of them -- its flags' help, its limits, its image
are attributes of its reviewer class, which the table's readers use.
"""

from __future__ import annotations

from agy_reviewer import AgyReviewer
from cli_reviewer import CliReviewer
from codex_reviewer import CodexReviewer
from grok_reviewer import GrokReviewer

__all__ = ["CLI_BACKENDS"]

# The providers that run a CLI coding agent in a container on their own
# --<provider>-host, with --<provider>-image/-bin/-home/-timeout-seconds.
CLI_BACKENDS: dict[str, type[CliReviewer]] = {
    reviewer.backend: reviewer
    for reviewer in (CodexReviewer, AgyReviewer, GrokReviewer)
}
