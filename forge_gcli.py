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

Shared gcli/subprocess helpers for the forgejo_fairy bots.

Extracted from ``fairy.py`` so additional entry points (in
particular ``mail_fairy.py``) can reuse the same plumbing without
importing the orchestrator module. The helpers are deliberately
CLI-agnostic: they only consult ``args.gcli_account``,
``args.forge_type`` and ``args.verbose``, which every entry point
already provides. ``args.gcli_account`` is allowed to be falsy (None
or empty string), in which case ``gcli_prefix`` omits the ``-a``
flag and gcli falls through to its own configured default account.

What belongs here: everything that knows a forge exists -- building
the gcli command line, running it, parsing its JSON, and fetching
each endpoint. This is the only module that branches on
``args.forge_type`` to reach an API; ``mail_fairy`` owns the parallel
notion for notification mail (``--forge-flavor``).

What does NOT belong: review policy, LLM plumbing, rendering. A
caller that needs another field asks for it here rather than reaching
past this module to ``gcli_api``.

Returned payloads
-----------------
The keys below are what the tree relies on and what this module
undertakes to keep supplying. The Forgejo/GitHub wire spellings
behind them are private -- callers must not reach for a key that is
not listed. Consumers read them with ``.get``, so a forge that omits
one degrades that feature rather than raising.

``list_issue_comments``     id, body, user, created_at, updated_at,
                            html_url
``list_pr_reviews``         id, state, body, user, submitted_at,
                            updated_at, dismissed_at, commit_id,
                            comments_count, stale, dismissed
``list_pr_review_comments`` id, body, user, created_at, updated_at,
                            path
``list_issue_timeline``     id, type, body, user, created_at
``list_pr_commits``         sha, commit.message, author.{login,id},
                            commit.author.{name,email,date},
                            commit.committer.date
``list_pr_files``           filename, status, additions, deletions,
                            changes, previous_filename

A ``user`` sub-object carries login, id, full_name, html_url.

Logging:
This module owns its own ``logging.getLogger("forge_gcli")`` instance.
Entry points must list it in their ``setup_logging(...)`` call (the
``common.setup_logging`` accepts variadic extra loggers exactly for
this purpose) so the per-command ``+ <cmd>`` debug lines and any
warnings reach the same handlers as the rest of the bot's output.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from threading import Thread
from urllib.parse import quote

from common import JsonValue

__all__ = [
    "KIND_ISSUE",
    "KIND_PR",
    "add_forge_repo_args",
    "apply_issue_label_changes",
    "build_repo_path",
    "gcli_api",
    "gcli_prefix",
    "list_issue_comments",
    "list_issue_timeline",
    "list_pr_commits",
    "list_pr_files",
    "list_pr_review_comments",
    "list_pr_reviews",
    "load_json",
    "post_issue_comment",
    "run_cmd",
]


logger = logging.getLogger("forge_gcli")


def gcli_prefix(args: argparse.Namespace) -> list[str]:
    cmd = ["gcli"]
    if args.gcli_account:
        cmd += ["-a", args.gcli_account]
    if args.forge_type:
        cmd += ["-t", args.forge_type]
    return cmd


def add_forge_repo_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard --owner/--repo/--gcli-account/--forge-type arguments.

    These four flags are required by every CLI entry point in the
    project (fairy, mail_fairy, forgejo_export). Adding
    them through a single helper keeps the operator-facing surface
    consistent: --owner and --repo are required, --gcli-account is
    optional (falls through to gcli's configured default when
    omitted, mirroring what ``gcli_prefix`` does at runtime), and
    the forge type accepts both --forge-type (preferred) and
    --gcli-type (legacy alias) under ``dest='forge_type'``.
    """
    parser.add_argument(
        "--owner",
        required=True,
        help="Repository owner / namespace.",
    )
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository name.",
    )
    parser.add_argument(
        "--gcli-account",
        help=(
            "Optional gcli account override (passed as top-level -a/--account). "
            "When omitted gcli falls through to its own configured default."
        ),
    )
    parser.add_argument(
        "--forge-type",
        "--gcli-type",
        default="gitea",
        dest="forge_type",
        help=(
            "Forge backend type passed to gcli's -t flag (default: gitea). "
            "The deprecated alias --gcli-type is accepted for backwards "
            "compatibility with older operator scripts."
        ),
    )


_WIRE_LINE = re.compile(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d) ([DIWEC]) (.*)$")
_WIRE_LEVELS = {"D": logging.DEBUG, "I": logging.INFO, "W": logging.WARNING,
                "E": logging.ERROR, "C": logging.CRITICAL}


def _relay_line(prefix: str, line: str) -> None:
    """Re-log one relayed child line; a FAIRY_LOG_WIRE-shaped line keeps
    its own level (the pane colors by it) and its own time (the record's
    time IS the child's stamp, so no second timestamp appears)."""
    match = _WIRE_LINE.match(line)
    if match is None:
        logger.info("%s%s", prefix, line)
        return
    stamp, letter, message = match.groups()
    record = logger.makeRecord(logger.name, _WIRE_LEVELS[letter], __file__, 0,
                               "%s%s", (prefix, message), None)
    record.created = datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S").timestamp()
    record.msecs = 0.0
    logger.handle(record)


def run_cmd(
    cmd: list[str],
    *,
    verbose: int = 0,
    verbose_threshold: int = 1,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
    timeout: int | None = None,
    capture_stderr: bool = True,
    stderr_line_prefix: str | None = None,
) -> subprocess.CompletedProcess[str]:
    if verbose >= verbose_threshold:
        logger.debug("+ %s", shlex.join(cmd))
    if stderr_line_prefix is None:
        return subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=(subprocess.PIPE if capture_stderr else None),
            env=env,
            input=input_text,
            timeout=timeout,
        )

    # stderr_line_prefix mode: live-stream the child's stderr through
    # our logger with ``stderr_line_prefix`` prepended on every line so
    # that concurrent wrapper invocations (several workers, one log)
    # remain attributable -- and so the lines reach the --log-file
    # handlers the fairy-ui pane tails; a raw sys.stderr write reached
    # only the process console. A child in FAIRY_LOG_WIRE format keeps
    # its own level and time through the relay (_relay_line).
    #
    # Implementation note: we hand the child a raw pipe fd for stderr
    # (NOT ``subprocess.PIPE``). With a raw fd, ``Popen.stderr`` is
    # ``None``, so ``subprocess.run``/``communicate()`` does not spawn
    # its internal stderr-draining thread -- our single pump below is
    # therefore the only reader of the pipe and there is no race over
    # ``os.read(stderr_fd, ...)`` chunks. (When we previously used
    # ``stderr=subprocess.PIPE`` together with our own pump,
    # ``communicate()``'s drain thread silently swallowed ~45% of the
    # lines, hiding most ``responses.create start/ok`` and triage
    # decision logs from ``fairy.log``.)
    read_fd, write_fd = os.pipe()

    def _pump_stderr() -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as src:
                for line in src:
                    _relay_line(stderr_line_prefix, line.rstrip("\n"))
        except Exception:  # best-effort: never let the pump crash the review
            logger.exception("stderr pump failed for %s", shlex.join(cmd))

    pump = Thread(target=_pump_stderr, name="stderr-pump", daemon=True)
    pump.start()
    try:
        cp = subprocess.run(
            cmd,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=write_fd,
            env=env,
            input=input_text,
            timeout=timeout,
        )
    finally:
        # Close our copy of the write end so the pump's read sees EOF
        # once the child (and any grandchildren that inherited fd 2) also
        # close their copies. ``pump.join(timeout=...)`` below provides a
        # bounded fallback if a grandchild keeps fd 2 open past timeout.
        try:
            os.close(write_fd)
        except OSError:
            pass
    pump.join(timeout=2)
    return cp


def load_json(stdout: str, *, context: str) -> JsonValue:
    text = stdout.strip()
    if not text:
        raise RuntimeError(f"{context}: gcli returned empty output")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    pos = 0
    values: list[JsonValue] = []
    n = len(text)

    while True:
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            break
        try:
            value, end = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            starts = [i for i in (text.find('{', pos), text.find('[', pos)) if i != -1]
            if starts:
                start = min(starts)
                if start > pos:
                    try:
                        value, end = decoder.raw_decode(text, start)
                    except json.JSONDecodeError:
                        snippet = text[:500]
                        raise RuntimeError(
                            f"{context}: expected JSON from gcli, got: {snippet!r}"
                        ) from exc
                    pos = end
                    values.append(value)
                    continue
            snippet = text[:500]
            raise RuntimeError(f"{context}: expected JSON from gcli, got: {snippet!r}") from exc
        values.append(value)
        pos = end

    if not values:
        raise RuntimeError(f"{context}: expected JSON from gcli, got: {text[:500]!r}")
    if len(values) == 1:
        return values[0]

    if all(isinstance(v, list) for v in values):
        merged: list[JsonValue] = []
        for v in values:
            merged.extend(v)
        return merged

    if all(isinstance(v, dict) for v in values):
        return values

    return values


def gcli_api(
    args: argparse.Namespace,
    path: str,
    *,
    all_pages: bool = False,
    verbose_threshold: int = 2,
) -> JsonValue:
    cmd = gcli_prefix(args) + ["api"]
    if all_pages:
        cmd.append("-a")
    cmd.append(path)
    cp = run_cmd(cmd, verbose=args.verbose, verbose_threshold=verbose_threshold)
    if cp.returncode != 0:
        raise RuntimeError(
            f"gcli api failed for {path!r} with exit code {cp.returncode}:\n{cp.stderr.strip()}"
        )
    return load_json(cp.stdout, context=f"gcli api {path}")


def run_gcli_editor_submission(
    cmd: list[str],
    *,
    message: str,
    verbose: int,
    timeout: int | None = None,
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory(prefix="gcli-review-") as tmpdir:
        tmp = Path(tmpdir)
        body_path = tmp / "body.txt"
        editor_path = tmp / "editor.sh"
        body_path.write_text(message, encoding="utf-8")
        editor_path.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "dest=${1:?missing-target}\n"
            f"cat {shlex.quote(str(body_path))} > \"$dest\"\n",
            encoding="utf-8",
        )
        editor_path.chmod(0o700)

        env = os.environ.copy()
        env["GIT_EDITOR"] = str(editor_path)
        env["VISUAL"] = str(editor_path)
        env["EDITOR"] = str(editor_path)

        return run_cmd(cmd, verbose=verbose, env=env, timeout=timeout)


# Canonical kinds, kept in sync with mail_fairy.KIND_PR / KIND_ISSUE.
# Duplicated here (rather than imported from mail_fairy) so forge_gcli
# stays free of cycles -- mail_fairy already imports forge_gcli.
KIND_PR = "pr"
KIND_ISSUE = "issue"


def _comments_api_path(
    forge_type: str, owner: str, repo: str, number: int, kind: str,
) -> str:
    """Build the gcli-api path for fetching comments on PR/issue.

    Backend dispatch:
    - forgejo / gitea / github: serve PR and Issue comments on the
      same ``/repos/{owner}/{repo}/issues/{n}/comments`` endpoint.
    - gitlab: comments are "notes" and live under separate top-level
      paths for MRs and Issues. The project is identified by a URL-
      encoded ``owner/repo`` pair.

    Other ``forge_type`` values raise ``NotImplementedError`` with a
    pointer to this function so a downstream operator/developer
    knows where to add their backend.
    """
    ft = (forge_type or "").lower()
    if ft in ("forgejo", "gitea", "github"):
        return (
            f"/repos/{quote(owner, safe='')}/{quote(repo, safe='')}"
            f"/issues/{number}/comments"
        )
    if ft == "gitlab":
        project = quote(f"{owner}/{repo}", safe="")
        if kind == KIND_PR:
            return f"/projects/{project}/merge_requests/{number}/notes"
        if kind == KIND_ISSUE:
            return f"/projects/{project}/issues/{number}/notes"
        raise NotImplementedError(
            f"gitlab comment listing requires kind in "
            f"{{{KIND_PR!r}, {KIND_ISSUE!r}}}, got {kind!r}. "
            f"Patch forge_gcli._comments_api_path to add the "
            f"missing branch."
        )
    raise NotImplementedError(
        f"forge_gcli has no comment-API path mapping for "
        f"forge-type={forge_type!r}. "
        f"Supported: forgejo, gitea, github, gitlab. "
        f"To add a new backend, extend "
        f"forge_gcli._comments_api_path with the right path shape "
        f"and please report or patch."
    )


def list_issue_comments(
    args: argparse.Namespace,
    owner: str,
    repo: str,
    number: int,
    *,
    kind: str = KIND_PR,
) -> list[dict]:
    """Return all comments on PR/issue ``number`` in ``owner/repo``.

    The exact API path depends on the active gcli backend (see
    ``_comments_api_path``). For forgejo/gitea/github the request
    goes to ``/repos/{owner}/{repo}/issues/{n}/comments`` regardless
    of ``kind``. For gitlab the path differs for MRs vs Issues and
    is selected by ``kind`` (``KIND_PR`` / ``KIND_ISSUE``).

    Raises ``RuntimeError`` if the response is not a JSON list --
    we prefer to fail loud rather than silently return ``[]``,
    because this list is consumed by duplicate-detection code
    paths where a silent miss would cause a duplicate post.

    Raises ``NotImplementedError`` for unsupported backends with a
    message pointing the caller at where to add support.
    """
    forge_type = (getattr(args, "forge_type", None) or "gitea")
    path = _comments_api_path(forge_type, owner, repo, number, kind)
    logger.debug(
        "list_issue_comments forge_type=%s kind=%s path=%s",
        forge_type, kind, path,
    )
    raw = gcli_api(args, path, all_pages=True)
    if not isinstance(raw, list):
        raise RuntimeError(
            f"unexpected gcli response for {path} "
            f"(forge_type={forge_type!r}, kind={kind!r}): "
            f"{type(raw).__name__}"
        )
    comments = [c for c in raw if isinstance(c, dict)]
    logger.debug(
        "list_issue_comments forge_type=%s path=%s -> %d comments",
        forge_type, path, len(comments),
    )
    return comments


def post_issue_comment(
    args: argparse.Namespace,
    owner: str,
    repo: str,
    number: int,
    body: str,
    *,
    kind: str = KIND_PR,
) -> None:
    """Post a comment on PR/issue ``number`` in ``owner/repo``.

    Wraps ``gcli comment -o <owner> -r <repo> -p <number> -y`` and
    feeds the body through the editor stub helper (same path as
    ``gcli pulls ... approve`` reviews). ``gcli comment`` is meant
    to be backend-abstracted by gcli itself, so the same command
    line is sent regardless of forge_type. ``kind`` is accepted for
    parity with ``list_issue_comments`` and surfaces in the failure
    message so an operator can quickly distinguish a PR-comment
    failure from an Issue-comment failure on a backend where gcli's
    abstraction may differ.

    Raises ``RuntimeError`` on non-zero gcli exit. The error
    message names the backend so failures on untested backends are
    immediately attributable.
    """
    forge_type = (getattr(args, "forge_type", None) or "gitea")
    cmd = gcli_prefix(args) + [
        "comment",
        "-o", owner,
        "-r", repo,
        "-p", str(number),
        "-y",
    ]
    logger.info(
        "+ %s   # body bytes=%d  forge_type=%s kind=%s",
        shlex.join(cmd), len(body), forge_type, kind,
    )
    cp = run_gcli_editor_submission(
        cmd, message=body, verbose=getattr(args, "verbose", 0),
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"gcli comment failed for {owner}/{repo}#{number} "
            f"(forge_type={forge_type!r}, kind={kind!r}, "
            f"rc={cp.returncode}): {cp.stderr.strip()}\n"
            f"If forge_type is github or gitlab, the gcli "
            f"abstraction for ``gcli comment`` may not yet support "
            f"that backend in the way mail_fairy expects. Please "
            f"report or patch forge_gcli.post_issue_comment."
        )


def build_repo_path(owner: str, repo: str, suffix: str) -> str:
    """Build a ``/repos/{owner}/{repo}{suffix}`` API path.

    Owner and repo are URL-quoted; the suffix is appended verbatim.
    A leading slash on ``suffix`` is optional. Used for every repo-
    scoped Forgejo / Gitea / GitHub endpoint the bot speaks to.
    """
    owner_q = quote(owner, safe="")
    repo_q = quote(repo, safe="")
    if not suffix.startswith("/"):
        suffix = "/" + suffix
    return f"/repos/{owner_q}/{repo_q}{suffix}"


def _list_repo_endpoint(
    args: argparse.Namespace,
    owner: str,
    repo: str,
    suffix: str,
    *,
    what: str,
) -> list[dict]:
    """Fetch all pages of a repo-scoped list endpoint, raising on bad data.

    Wraps ``gcli_api(.., all_pages=True)``, validates the response is
    a JSON list, and drops non-dict items defensively (gcli should
    never emit them but the projection at the consumer is robust to
    it). ``what`` is a short human label included in the error
    message so a failure points the operator at the right endpoint.
    """
    path = build_repo_path(owner, repo, suffix)
    raw = gcli_api(args, path, all_pages=True)
    if not isinstance(raw, list):
        raise RuntimeError(
            f"unexpected gcli response for {what} ({path}): "
            f"{type(raw).__name__}"
        )
    return [item for item in raw if isinstance(item, dict)]


def list_pr_reviews(
    args: argparse.Namespace, owner: str, repo: str, pr_number: int,
) -> list[dict]:
    """Return all reviews on ``owner/repo`` PR ``pr_number``."""
    return _list_repo_endpoint(
        args, owner, repo, f"/pulls/{pr_number}/reviews",
        what=f"reviews for PR #{pr_number}",
    )


def list_pr_review_comments(
    args: argparse.Namespace,
    owner: str,
    repo: str,
    pr_number: int,
    reviews: list[dict],
) -> list[dict]:
    """Return all inline (file/line-pinned) review comments for a PR.

    Iterates the already-fetched ``reviews`` list and pulls per-review
    inline comments. Reviews with explicit ``comments_count == 0``
    are skipped so we don't fire an API call to confirm emptiness.
    Per-review fetch errors are logged and skipped, not raised: one
    bad review id should not block the rest of the discussion.
    """
    out: list[dict] = []
    for review in reviews:
        review_id = review.get("id")
        if not isinstance(review_id, int) or review_id <= 0:
            continue
        comments_count = review.get("comments_count")
        if isinstance(comments_count, int) and comments_count <= 0:
            continue
        try:
            out.extend(_list_repo_endpoint(
                args, owner, repo,
                f"/pulls/{pr_number}/reviews/{review_id}/comments",
                what=f"review comments for PR #{pr_number} review {review_id}",
            ))
        except Exception as exc:
            logger.warning(
                "failed to fetch inline review comments for "
                "PR #%d review %d: %s",
                pr_number, review_id, exc,
            )
    return out


def list_issue_timeline(
    args: argparse.Namespace, owner: str, repo: str, number: int,
) -> list[dict]:
    """Return the typed timeline events for ``owner/repo`` issue/PR ``number``.

    The endpoint is ``/repos/{owner}/{repo}/issues/{n}/timeline`` for
    both issues and PRs (PRs are issues in the data model). PR
    timelines additionally include typed events like ``pull_push``,
    ``pull_scheduled_merge`` and ``pull_cancel_scheduled_merge`` that
    the plain comments endpoint omits, which is why the bot's
    auto-merge detection and push-event tracking source from this
    feed rather than from ``/issues/{n}/comments``.
    """
    return _list_repo_endpoint(
        args, owner, repo, f"/issues/{number}/timeline",
        what=f"timeline for #{number}",
    )


def list_pr_commits(
    args: argparse.Namespace, owner: str, repo: str, pr_number: int,
) -> list[dict]:
    """Return the commits on ``owner/repo`` PR ``pr_number``'s head branch."""
    return _list_repo_endpoint(
        args, owner, repo, f"/pulls/{pr_number}/commits",
        what=f"commits for PR #{pr_number}",
    )


def list_pr_files(
    args: argparse.Namespace, owner: str, repo: str, pr_number: int,
) -> list[dict]:
    """Return the per-file change entries for ``owner/repo`` PR ``pr_number``."""
    return _list_repo_endpoint(
        args, owner, repo, f"/pulls/{pr_number}/files",
        what=f"files for PR #{pr_number}",
    )


def apply_issue_label_changes(
    args: argparse.Namespace,
    owner: str,
    repo: str,
    number: int,
    labels_add: list[str],
    labels_remove: list[str],
    current_label_names: set[str],
    kind: str = KIND_PR,
) -> None:
    """Add/remove labels by name via ``gcli pulls/issues ... labels``.

    ``kind`` selects the gcli subcommand: ``KIND_PR`` -> ``pulls``,
    ``KIND_ISSUE`` -> ``issues`` (both expose the same
    ``labels add/remove`` action).

    Skips ``remove`` when the label is not currently attached so gcli
    is not asked to delete something that is already absent.

    Stock gcli (<= 2.12.0) cannot attach labels reliably on
    Forgejo/Gitea: ``pulls labels add NAME`` resolves NAME to an id,
    sends the id *as a JSON string*, and newer Gitea/Forgejo
    releases then treat that string as another name lookup, don't
    find it, and return 200 with no labels attached. Older servers
    reject strings outright via ``[]int64`` binding. Org-level
    labels are also unreachable because gcli only probes the repo's
    labels for the lookup.

    The bot requires a gcli built from a tree that includes both
    ``gitea: send issue label ids as JSON numbers, not strings`` and
    ``gitea: also look up org-level labels by name`` (see ``~/gcli``
    on the deployment host; submitted upstream).
    """
    if not labels_add and not labels_remove:
        return

    label_args: list[str] = []
    for name in labels_add:
        label_args.extend(["add", name])
    for name in labels_remove:
        if name not in current_label_names:
            logger.warning(
                "label remove skipped: label=%r not on %s #%d",
                name, kind, number,
            )
            continue
        label_args.extend(["remove", name])
    if not label_args:
        return

    subcommand = "pulls" if kind == KIND_PR else "issues"
    cmd = gcli_prefix(args) + [
        subcommand,
        "-o", owner,
        "-r", repo,
        "-i", str(number),
        "labels",
        *label_args,
    ]
    logger.info("+ %s", shlex.join(cmd))
    cp = run_cmd(cmd, verbose=args.verbose)
    if cp.returncode != 0:
        raise RuntimeError(
            f"gcli {subcommand} labels failed for {owner}/{repo}#{number} "
            f"(rc={cp.returncode}): {cp.stderr.strip()}"
        )
