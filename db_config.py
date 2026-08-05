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

The db root's config.toml: the repo facts configurator.py records for
the other components -- agent and worker configure their sides from
it, the TUI its repo labels and log tails.

What belongs here: the config.toml format, its writer and its reader.
What does NOT belong: the ticket files (filedb), parsing the side
argument strings (fairy / issue_fairy).
"""

from __future__ import annotations

import json
import logging
import tomllib
from pathlib import Path

from common import atomic_write_text

__all__ = ["CONFIG_NAME", "read_config", "read_side_strings", "write_config"]

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.toml"


def write_config(root: Path, label: str, log_files: set[Path],
                 pr_args: str | None, issue_args: str | None) -> None:
    """Record the repo label, the sides' log files and the verbatim
    side argument strings as <root>/config.toml.

    A non-ASCII-escaping JSON-encoded str or list of str is also a
    valid TOML basic string / array (ASCII-escaping is not: JSON
    spells non-BMP characters as surrogate pairs, which TOML rejects),
    so json.dumps does the value quoting; a None-valued side is an
    omitted key (TOML has no null)."""
    pairs = {"label": label,
             "log_files": sorted(str(f.resolve()) for f in log_files),
             "pr_args": pr_args,
             "issue_args": issue_args}
    atomic_write_text(root / CONFIG_NAME, "".join(
        f"{key} = {json.dumps(value, ensure_ascii=False)}\n"
        for key, value in pairs.items() if value is not None))
    logger.info("wrote %s", root / CONFIG_NAME)


def read_config(root: Path) -> dict:
    """The dict write_config stored under ``root``."""
    try:
        return tomllib.loads(
            (root / CONFIG_NAME).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        hint = "".join(
            f" ({sibling.name} exists -- check the case)"
            for sibling in root.parent.glob("*")
            if sibling.name.casefold() == root.name.casefold()
        ) if not root.exists() else ""
        raise SystemExit(
            f"{exc}: run ./configurator.py for this repo{hint}") from exc
    except (OSError, ValueError) as exc:
        raise SystemExit(f"{root / CONFIG_NAME}: {exc!r}") from exc


def read_side_strings(root: Path) -> tuple[str | None, str | None]:
    """The (pr_args, issue_args) side strings from ``root``'s config;
    at least one is present."""
    cfg = read_config(root)
    pr_args, issue_args = cfg.get("pr_args"), cfg.get("issue_args")
    if not (pr_args or issue_args):
        raise SystemExit(f"{root / CONFIG_NAME}: no side argument strings")
    return pr_args, issue_args
