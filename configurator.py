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

The configurator: validate a repo's side options -- given directly on
its command line, shared ones first, then a --prs and/or --issues
section per side -- and write them, with the repo label and log files,
as the db root's config.toml that agent, worker and TUI configure
themselves from. Run it before starting the daemons; rerun it to
change a repo's configuration.

What belongs here: everything that decides what goes into a db root's
config.
What does NOT belong: the config.toml format itself (db_config),
running the configured processes (agent / worker / fairy_tui).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import db_config
import fairy
import issue_fairy
import workset
from common import (add_grouped_help, apply_config_file_defaults,
                    config_option_groups, grouped_help, setup_logging,
                    side_options, split_sections)

__all__ = ["main"]

logger = logging.getLogger(__name__)


def db_root_for(ns: argparse.Namespace) -> Path:
    return workset.repo_dir(
        Path.home() / ".fairy" / "db",
        forge_type=ns.forge_type, account=ns.gcli_account or "",
        owner=ns.owner, repo=ns.repo)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        add_help=False,
        usage="%(prog)s [--db-root DIR] [shared side options] "
              "--prs [pr options] --issues [issue options]",
        description="Validate a repo's side options and write them into the "
                    "db root's config.toml, which agent, worker and TUI "
                    "configure themselves from. Options before the first "
                    "--prs / --issues marker are shared by both sides; each "
                    "marker starts that side's own options, which append "
                    "after (and thereby override) the shared ones.",
    )
    p.add_argument("--db-root", type=Path,
                   help="filedb root (default: ~/.fairy/db/<forge~account~owner~repo>)")
    add_grouped_help(p, _help)
    return p


def _help() -> str:
    return grouped_help(make_parser(), config_option_groups(
        fairy.make_parser(), issue_fairy.make_parser()))


def main() -> int:
    shared, sections = split_sections(sys.argv[1:])
    own_parser = make_parser()
    args, shared = own_parser.parse_known_args(shared)
    if not sections:
        own_parser.error("at least one of --prs / --issues is required")
    pr_tokens = shared + sections["--prs"] if "--prs" in sections else None
    issue_tokens = shared + sections["--issues"] if "--issues" in sections \
        else None
    pr_ns = fairy.parse_args(pr_tokens) if pr_tokens is not None else None
    issue_ns = issue_fairy.parse_args(issue_tokens) \
        if issue_tokens is not None else None
    lead = pr_ns or issue_ns
    setup_logging(fairy.logger, max(ns.verbose for ns in (pr_ns, issue_ns) if ns),
                  logger, db_config.logger, workset.logger, color=lead.color)
    fairy.validate_sides(pr_ns, issue_ns)
    root = args.db_root or db_root_for(lead)
    pr_options = issue_options = None
    if pr_tokens is not None:
        pr_parser = fairy.make_parser()
        apply_config_file_defaults(pr_parser, pr_tokens)
        pr_options = side_options(pr_parser, pr_tokens)
    if issue_tokens is not None:
        issue_parser = issue_fairy.make_parser()
        apply_config_file_defaults(issue_parser, issue_tokens)
        issue_options = side_options(issue_parser, issue_tokens)
    db_config.write_config(
        root, f"{lead.owner}/{lead.repo}",
        {ns.log_file for ns in (pr_ns, issue_ns) if ns and ns.log_file},
        pr_options, issue_options)
    logger.info("configured %s/%s, db %s", lead.owner, lead.repo, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
