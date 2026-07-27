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
``list_issue_timeline``     id, type, body, user, created_at; a
                            ``pull_push`` event whose payload was
                            readable also carries commit_ids, and
                            is_force_push where the forge says
``list_pr_commits``         sha, commit.message, author.{login,id},
                            commit.author.{name,email,date},
                            commit.committer.date
``list_pr_files``           filename, status, additions, deletions,
                            changes, previous_filename
``list_commit_statuses``    context, state, description, target_url,
                            created_at, updated_at

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
    "AUTO_MERGE_CANCEL_EVENT",
    "AUTO_MERGE_SCHEDULE_EVENT",
    "KIND_ISSUE",
    "KIND_PR",
    "PUSH_EVENT",
    "add_forge_repo_args",
    "auto_merge_state",
    "apply_issue_label_changes",
    "build_repo_path",
    "gcli_api",
    "gcli_prefix",
    "list_commit_statuses",
    "list_issue_comments",
    "list_issue_timeline",
    "list_pr_commits",
    "list_pr_files",
    "list_pr_review_comments",
    "list_pr_reviews",
    "load_json",
    "norm_user",
    "post_issue_comment",
    "project_timeline_event",
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
    forge_type = _forge_type(args)
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
    forge_type = _forge_type(args)
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


def _forge_type(args: argparse.Namespace) -> str:
    """The backend gcli will talk to; empty means gcli's own default."""
    return (getattr(args, "forge_type", None) or "gitea").lower()


def norm_user(user: dict | None) -> dict | None:
    """Thin a forge user object down to the keys this module hands out."""
    if not user:
        return None
    return {k: user.get(k) for k in ("login", "id", "full_name", "html_url")}


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


# Forgejo names the typed timeline comments for pushes and the
# auto-merge actions. Source:
# https://codeberg.org/forgejo/forgejo/src/branch/forgejo/models/issues/comment.go
#   * ``CommentTypePRScheduledToAutoMerge`` (= 34, ``pull_scheduled_merge``)
#   * ``CommentTypePRUnScheduledToAutoMerge`` (= 35,
#     ``pull_cancel_scheduled_merge``)
# First observed on Forgejo 15.0.x; expected to be stable across
# releases since the constants are part of the public API enum and
# Gitea uses the same names. The full set of event types is
# forge-defined and open, so these are named constants rather than an
# enum: an event this module does not name still passes through.
PUSH_EVENT = "pull_push"
AUTO_MERGE_SCHEDULE_EVENT = "pull_scheduled_merge"
AUTO_MERGE_CANCEL_EVENT = "pull_cancel_scheduled_merge"


_AUTO_MERGE_EVENTS = frozenset({AUTO_MERGE_SCHEDULE_EVENT, AUTO_MERGE_CANCEL_EVENT})


def auto_merge_state(args: argparse.Namespace, pr: dict,
                     timeline: list[dict]) -> str:
    """Whether ``pr`` is queued to merge itself: merge / no / ? (unknown).

    Forgejo publishes no endpoint for the schedule, so the answer is
    read off the typed timeline entries, latest wins.

    GitHub states it on the pull request as ``auto_merge``, null when
    nothing is queued (observed 2026-07-28 on michaelni/testrepo #2),
    and emits its own timeline entries which are NOT the Forgejo ones.
    Deriving it from the timeline there would answer a confident "no"
    to every GitHub PR -- and a wrong "no" is the dangerous direction:
    the caller uses it to decide that approving is only a comment,
    when on a queued PR an approval is what merges it.
    """
    if _forge_type(args) == "github":
        return "merge" if pr.get("auto_merge") else "no"
    latest_ts: str | None = None
    latest_type: str | None = None
    for entry in timeline:
        if entry.get("type") not in _AUTO_MERGE_EVENTS:
            continue
        ts = entry.get("created_at")
        if not isinstance(ts, str):
            continue
        if latest_ts is None or ts > latest_ts:
            latest_ts, latest_type = ts, entry.get("type")
    return "merge" if latest_type == AUTO_MERGE_SCHEDULE_EVENT else "no"


def _push_fields(body: str) -> dict:
    """Decode a push payload, which Forgejo ships as JSON text in ``body``.

    Returns ``{}`` for a payload this module cannot read, which is how a
    consumer tells "a push whose commits are unknown" (both keys absent)
    from "a push that touched no listed commit" (``commit_ids == []``).
    """
    try:
        decoded = json.loads(body)
    except (ValueError, TypeError):
        return {}
    if not isinstance(decoded, dict):
        return {}
    commit_ids = decoded.get("commit_ids")
    return {
        "is_force_push": bool(decoded.get("is_force_push")),
        "commit_ids": [c for c in commit_ids if isinstance(c, str) and c]
                      if isinstance(commit_ids, list) else [],
    }


def project_timeline_event(event: dict) -> dict:
    """Project one raw timeline event to the keys this module hands out."""
    projected = {
        "type": event.get("type"),
        "id": event.get("id"),
        "user": norm_user(event.get("user")),
        "created_at": event.get("created_at"),
        "body": event.get("body") or "",
    }
    if projected["type"] == PUSH_EVENT:
        projected.update(_push_fields(projected["body"]))
    return projected


# GitHub spells the entry kind ``event`` where Forgejo says ``type``,
# and puts the actor under ``user`` (comments, reviews), ``actor``
# (state changes) or ``author`` (a ``committed`` entry, which carries a
# git identity with no forge login and dates the entry under
# ``author.date`` instead of ``created_at``). Captured 2026-07-28 from
_GITHUB_COMMIT_EVENT = "committed"


def _github_commit_author(event: dict) -> dict | None:
    author = event.get("author")
    if not isinstance(author, dict):
        return None
    return {"login": None, "id": None,
            "full_name": author.get("name"), "html_url": None}


def _git_date(event: dict, key: str) -> str | None:
    stamp = event.get(key)
    return stamp.get("date") if isinstance(stamp, dict) else None


def _project_github_event(event: dict) -> dict:
    return {
        "type": event.get("event"),
        "id": event.get("id"),
        "user": norm_user(event.get("user") or event.get("actor"))
                or _github_commit_author(event),
        # A commit entry carries two dates: when the change was written
        # and when it was last applied to the branch. The second is the
        # one near the push -- they can be days apart -- and a push is what the activity gate measures.
        "created_at": event.get("created_at") or event.get("submitted_at")
                      or _git_date(event, "committer") or _git_date(event, "author"),
        "body": event.get("body") or "",
    }


def _project_github_timeline(events: list[dict]) -> list[dict]:
    """Fold GitHub's timeline into the events this module hands out.

    GitHub lists one ``committed`` entry per commit and does not mark
    where one push ended and the next began, so a run of them with
    nothing in between is reported as a single push. That grouping is a
    reading of the order GitHub returned, not something GitHub states.

    ``is_force_push`` is left absent: the run of commits looks the same
    either way, and ``head_ref_force_pushed`` -- the entry that would
    say so -- is not covered by a capture, so claiming ``False`` here
    would tell the reviewing model something unverified.
    """
    out: list[dict] = []
    run: list[dict] = []

    def flush() -> None:
        if not run:
            return
        last = run[-1]
        out.append({**_project_github_event(last), "type": PUSH_EVENT,
                    "id": None, "body": "",
                    "commit_ids": [c["sha"] for c in run
                                   if isinstance(c.get("sha"), str)]})
        run.clear()

    for event in events:
        if event.get("event") == _GITHUB_COMMIT_EVENT:
            run.append(event)
            continue
        flush()
        out.append(_project_github_event(event))
    flush()
    return out


def list_issue_timeline(
    args: argparse.Namespace, owner: str, repo: str, number: int,
) -> list[dict]:
    """Return the timeline events for ``owner/repo`` issue/PR ``number``.

    The endpoint is ``/repos/{owner}/{repo}/issues/{n}/timeline`` for
    both issues and PRs (PRs are issues in the data model). PR
    timelines additionally carry the typed events named above, which
    the plain comments endpoint omits -- which is why auto-merge
    detection and push tracking source from this feed rather than from
    ``/issues/{n}/comments``.

    Events are projected to ``type``, ``id``, ``user``, ``created_at``
    and ``body``; ``pull_push`` events also carry ``is_force_push`` and
    ``commit_ids``. Projecting here rather than at each consumer keeps
    the decode in one place and keeps the per-event user object out of
    the cache, which stores whatever this returns.
    """
    raw = _list_repo_endpoint(
        args, owner, repo, f"/issues/{number}/timeline",
        what=f"timeline for #{number}",
    )
    if _forge_type(args) == "github":
        return _project_github_timeline(raw)
    return [project_timeline_event(e) for e in raw]


def _project_status_row(row: dict) -> dict:
    """Project one CI status row to the keys this module hands out.

    The ``name``/``status`` spellings are alternatives seen for the same
    two fields, so the choice is made here once instead of at every
    consumer.
    """
    return {
        "context": row.get("context") or row.get("name") or row.get("target_url"),
        "state": row.get("state") or row.get("status"),
        "description": row.get("description") or "",
        "target_url": row.get("target_url") or "",
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }


def _check_run_as_status_row(run: dict) -> dict:
    """Project a GitHub check run onto a commit-status row.

    ``conclusion`` is null until a run completes, so an unfinished run
    is reported by its ``status`` (``queued`` / ``in_progress``), which
    the state vocabulary already reads as pending.
    """
    output = run.get("output") or {}
    completed = run.get("status") == "completed"
    return {
        "context": run.get("name"),
        "state": run.get("conclusion") if completed else run.get("status"),
        "description": output.get("title") or "",
        "target_url": run.get("html_url") or "",
        "created_at": run.get("started_at"),
        "updated_at": run.get("completed_at") or run.get("started_at"),
    }


def _list_check_runs(
    args: argparse.Namespace, owner: str, repo: str, ref: str,
) -> list[dict]:
    """Fetch the check runs for ``ref``; the endpoint wraps them in an object."""
    path = build_repo_path(owner, repo, f"/commits/{quote(ref, safe='')}/check-runs")
    data = gcli_api(args, path, all_pages=True)
    pages = data if isinstance(data, list) else [data]
    return [run for page in pages if isinstance(page, dict)
            for run in (page.get("check_runs") or []) if isinstance(run, dict)]


def list_commit_statuses(
    args: argparse.Namespace, owner: str, repo: str, ref: str,
) -> list[dict]:
    """Return the CI status rows for ``owner/repo`` commit ``ref``.

    GitHub Actions reports through the Checks API and posts nothing to
    the commit-status endpoint, so on GitHub both are read and merged:
    ``/commits/{sha}/statuses`` can return 0 rows while
    ``/commits/{sha}/check-runs`` returns them all. Third-party CI still
    posts to the status endpoint, so dropping it would lose those.
    Forgejo and GitLab report everything through statuses and are left
    at the single request.

    A response that is not a list yields no rows rather than raising:
    a PR whose CI cannot be read is skipped for want of CI, and that
    beats failing the review outright.
    """
    path = build_repo_path(owner, repo, f"/commits/{quote(ref, safe='')}/statuses")
    data = gcli_api(args, path, all_pages=True)
    rows = ([_project_status_row(r) for r in data if isinstance(r, dict)]
            if isinstance(data, list) else [])
    if _forge_type(args) == "github":
        rows += [_check_run_as_status_row(r)
                 for r in _list_check_runs(args, owner, repo, ref)]
    return rows


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
