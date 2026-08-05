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
import shlex
import tomllib
from pathlib import Path

from common import atomic_write_text

__all__ = ["CONFIG_NAME", "log_side_argv", "read_config", "read_side_argv",
           "write_config"]

logger = logging.getLogger(__name__)

CONFIG_NAME = "config.toml"


def _toml_value(value: bool | str | list[str]) -> str:
    """``value`` is a side option as write_config documents it: True
    for a bare flag, a list for a repeated option, a token string
    otherwise. A non-ASCII-escaping JSON-encoded str is also a valid
    TOML basic string (ASCII-escaping is not: JSON spells non-BMP
    characters as surrogate pairs, which TOML rejects), so json.dumps
    does the quoting; a multi-line token becomes a TOML multi-line
    literal string when its content permits one."""
    if value is True:
        return "true"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(v) for v in value) + "]"
    if "\n" in value and "'''" not in value and not value.endswith("'") \
            and all(c == "\n" or c == "\t" or c.isprintable() for c in value):
        return f"'''\n{value}'''"
    return json.dumps(value, ensure_ascii=False)


def write_config(root: Path, label: str, log_files: set[Path],
                 pr_options: dict | None,
                 issue_options: dict | None) -> None:
    """Record the repo label, the sides' log files and the sides'
    options as <root>/config.toml, one ``key = value`` line per CLI
    option under a [pr] / [issue] table: the key is the option name
    without the leading dashes, the value True for a bare flag, a
    list for a repeated option, the token string otherwise. A None
    side is an omitted table (TOML has no null)."""
    lines = [f"label = {json.dumps(label, ensure_ascii=False)}\n",
             "log_files = " +
             json.dumps(sorted(str(f.resolve()) for f in log_files),
                        ensure_ascii=False) + "\n"]
    for side, options in (("pr", pr_options), ("issue", issue_options)):
        if options is None:
            continue
        lines.append(f"\n[{side}]\n")
        lines += [f"{key} = {_toml_value(value)}\n"
                  for key, value in options.items()]
    atomic_write_text(root / CONFIG_NAME, "".join(lines))
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


def _argv(options: dict) -> list[str]:
    """The [pr]/[issue] table is a hand-editable boundary: a list is a
    repeated option, True a bare flag, false an absent one, any other
    scalar one option value -- spelled ``--key=value`` in one token,
    since argparse takes a leading-dash value only in that form."""
    argv: list[str] = []
    for key, value in options.items():
        for v in (value if isinstance(value, list) else [value]):
            if v is False:
                continue
            argv.append(f"--{key}" if v is True else f"--{key}={v}")
    return argv


def read_side_argv(root: Path) -> tuple[list[str] | None, list[str] | None]:
    """The (pr, issue) argv lists rebuilt from ``root``'s config
    tables; at least one side is present, an absent side is None."""
    cfg = read_config(root)
    pr, issue = (_argv(cfg[side]) if side in cfg else None
                 for side in ("pr", "issue"))
    if not (pr or issue):
        raise SystemExit(f"{root / CONFIG_NAME}: no side options")
    return pr, issue


def log_side_argv(pr_argv: list[str] | None, issue_argv: list[str] | None,
                  pr_overrides: list[str] | None = None,
                  issue_overrides: list[str] | None = None) -> None:
    """One provenance line per configured side and one per side's CLI
    overrides, for the shared log; overrides for a side the config
    lacks warn instead."""
    for kind, argv, over in (("pr", pr_argv, pr_overrides),
                             ("issue", issue_argv, issue_overrides)):
        if argv:
            logger.info("%s side from %s: %s", kind, CONFIG_NAME,
                        shlex.join(argv))
        if not over:
            continue
        if argv:
            logger.info("%s side CLI overrides: %s", kind, shlex.join(over))
        else:
            logger.warning("%s side CLI overrides ignored: the config has "
                           "no such side", kind)
