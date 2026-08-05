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

The db root's config.json: the repo facts the agent records at startup
for the other components -- the TUI reads its repo labels and log
tails from it.

What belongs here: the config.json format, its writer and its waiting
reader.
What does NOT belong: the ticket files (filedb), parsing the side
argument strings (fairy / issue_fairy).
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from common import atomic_write_text

__all__ = ["read_config", "write_config"]

logger = logging.getLogger(__name__)

CONFIG_WAIT = 5.0


def write_config(root: Path, label: str, log_files: set[Path]) -> None:
    """Record the repo label and the sides' log files as
    <root>/config.json."""
    atomic_write_text(root / "config.json", json.dumps({
        "label": label,
        "log_files": sorted(str(f.resolve()) for f in log_files)}))
    logger.info("wrote %s", root / "config.json")


def read_config(root: Path) -> dict:
    """The dict write_config stored under ``root``, waiting up to
    CONFIG_WAIT seconds for it: launchers like fairy-ui-ref.sh
    background the agent moments before the readers start, and the
    agent writes the file at startup."""
    deadline = time.monotonic() + CONFIG_WAIT
    waiting = False
    while True:
        try:
            return json.loads(
                (root / "config.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            if time.monotonic() >= deadline:
                hint = "".join(
                    f" ({sibling.name} exists -- check the case)"
                    for sibling in root.parent.glob("*")
                    if sibling.name.casefold() == root.name.casefold()
                ) if not root.exists() else ""
                raise SystemExit(
                    f"{exc}: the repo's agent writes config.json at "
                    f"startup{hint}") from exc
            if not waiting:
                logger.warning("waiting for %s", root / "config.json")
                waiting = True
            time.sleep(0.1)
        except (OSError, ValueError) as exc:
            raise SystemExit(f"{root / 'config.json'}: {exc!r}") from exc
