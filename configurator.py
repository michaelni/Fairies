#!/usr/bin/env python3
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

The configurator: validate a repo's side argument strings and write
them, with the repo label and log files, as the db root's config.toml
that agent, worker and TUI configure themselves from. Run it before
starting the daemons; rerun it to change a repo's configuration.

What belongs here: the side-string validation and everything else that
decides what goes into a db root's config.
What does NOT belong: the config.toml format itself (db_config),
running the configured processes (agent / worker / fairy_tui).
"""

from __future__ import annotations

import argparse
import logging
import shlex
from pathlib import Path

import db_config
import fairy
import issue_fairy
import workset
from common import setup_logging

__all__ = ["main"]

logger = logging.getLogger(__name__)


def db_root_for(ns: argparse.Namespace) -> Path:
    return workset.repo_dir(
        Path.home() / ".fairy" / "db",
        forge_type=ns.forge_type, account=ns.gcli_account or "",
        owner=ns.owner, repo=ns.repo)


def _config_error(message: str) -> None:
    logger.error(message)
    raise SystemExit(2)


def validate_sides(pr_ns: argparse.Namespace | None,
                   issue_ns: argparse.Namespace | None) -> None:
    """Reject broken side configs with rc=2: discovered per-item they
    would burn error retries for days."""
    if pr_ns and pr_ns.llm_review_cmd and pr_ns.patch_repo is None:
        _config_error("--llm-review-cmd requires --patch-repo PATH")
    for ns, forced in ((pr_ns, "force_review_prs"),
                       (issue_ns, "force_review_issues")):
        if ns is None:
            continue
        if getattr(ns, "simulate_past", None) is not None \
                and "{number}" not in (getattr(ns, "patch_pr_ref_template", None) or ""):
            _config_error(
                "--simulate-past requires --patch-pr-ref-template TEMPLATE "
                "containing {number} (e.g. fforge/pr/{number})")
        if ns.forced_only and not getattr(ns, forced):
            _config_error(
                "--forced-only requires at least one --force-review-*")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate a repo's side argument strings and write them "
                    "into the db root's config.toml, which agent, worker and "
                    "TUI configure themselves from.",
    )
    p.add_argument("--pr-args", metavar="ARGS",
                   help="the PR side's argument string; ./fairy.py --help "
                        "documents its contents")
    p.add_argument("--issue-args", metavar="ARGS",
                   help="the issue side's argument string; ./issue_fairy.py "
                        "--help documents its contents")
    p.add_argument("--db-root", type=Path,
                   help="filedb root (default: ~/.fairy/db/<forge~account~owner~repo>)")
    args = p.parse_args(argv)
    if not args.pr_args and not args.issue_args:
        p.error("at least one of --pr-args / --issue-args is required")
    return args


def main() -> int:
    args = parse_args()
    pr_ns = fairy.parse_args(shlex.split(args.pr_args)) if args.pr_args else None
    issue_ns = issue_fairy.parse_args(shlex.split(args.issue_args)) if args.issue_args else None
    lead = pr_ns or issue_ns
    setup_logging(fairy.logger, max(ns.verbose for ns in (pr_ns, issue_ns) if ns),
                  logger, db_config.logger, workset.logger, color=lead.color)
    validate_sides(pr_ns, issue_ns)
    root = args.db_root or db_root_for(lead)
    db_config.write_config(
        root, f"{lead.owner}/{lead.repo}",
        {ns.log_file for ns in (pr_ns, issue_ns) if ns and ns.log_file},
        args.pr_args, args.issue_args)
    logger.info("configured %s/%s, db %s", lead.owner, lead.repo, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
