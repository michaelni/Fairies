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

What belongs here: the side validation and everything else that
decides what goes into a db root's config.
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
from common import apply_config_file_defaults, setup_logging

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


def side_options(parser: argparse.ArgumentParser,
                 tokens: list[str]) -> dict[str, bool | str | list[str]]:
    """One dict entry per CLI option in ``tokens``, keyed by the
    option name without the leading dashes: True for a bare flag, a
    list for a repeated (argparse append) option, the token string
    otherwise -- the shape db_config.write_config stores. The parser
    must carry every option the tokens use -- for the PR side that
    includes --config, which apply_config_file_defaults registers.
    parse_args has already accepted ``tokens``, so the only rejection
    left here is an abbreviated option name, which parse_args resolves
    but a config key must not carry."""
    actions = {opt: a for a in parser._actions for opt in a.option_strings}
    options: dict[str, bool | str | list[str]] = {}
    i = 0
    while i < len(tokens):
        name, eq, inline = tokens[i].partition("=")
        action = actions.get(name)
        if action is None:
            raise SystemExit(f"{name}: unknown or abbreviated option; "
                             "the config stores full option names")
        key = name.removeprefix("--")
        if action.nargs == 0:
            options[key] = True
            i += 1
            continue
        value = inline if eq else tokens[i + 1]
        i += 1 if eq else 2
        if isinstance(action, argparse._AppendAction):
            existing = options.setdefault(key, [])
            existing.append(value)
        else:
            options[key] = value
    return options


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
    return p


def split_sections(argv: list[str]) -> tuple[list[str], dict[str, list[str]]]:
    """The tokens before the first ``--prs``/``--issues`` marker, and
    one token list per marker present."""
    shared: list[str] = []
    sections: dict[str, list[str]] = {}
    current = shared
    for token in argv:
        if token in ("--prs", "--issues"):
            if token in sections:
                raise SystemExit(f"{token} given twice")
            current = sections[token] = []
        else:
            current.append(token)
    return shared, sections


def main() -> int:
    argv = sys.argv[1:]
    if "-h" in argv or "--help" in argv:
        print(make_parser().format_help()
              + "\nPR side options (after --prs, or shared):\n\n"
              + fairy.make_parser().format_help()
              + "\nIssue side options (after --issues, or shared):\n\n"
              + issue_fairy.make_parser().format_help())
        return 0
    shared, sections = split_sections(argv)
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
    validate_sides(pr_ns, issue_ns)
    root = args.db_root or db_root_for(lead)
    pr_options = None
    if pr_tokens is not None:
        pr_parser = fairy.make_parser()
        apply_config_file_defaults(pr_parser, pr_tokens)
        pr_options = side_options(pr_parser, pr_tokens)
    db_config.write_config(
        root, f"{lead.owner}/{lead.repo}",
        {ns.log_file for ns in (pr_ns, issue_ns) if ns and ns.log_file},
        pr_options,
        side_options(issue_fairy.make_parser(), issue_tokens)
        if issue_tokens is not None else None)
    logger.info("configured %s/%s, db %s", lead.owner, lead.repo, root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
