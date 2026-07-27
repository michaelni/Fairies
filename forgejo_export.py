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
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import forge_gcli
import gcli_cache
from common import (
    add_color_arg,
    attachment_urls,
    default_cache_path,
    iso_to_dt,
    setup_logging,
)
from forge_gcli import add_forge_repo_args, norm_user

LOG = logging.getLogger("forgejo_export")
MARKER = ".forgejo-exporter.json"
ALLOWED_TOP = {".git", MARKER, "issues", "pulls", ".openai_vector_store_cache.json", ".gitignore"}


class CommandError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export Forgejo/Gitea issues and pull requests to a managed git repository via GCLI."
    )
    p.add_argument("target_dir", type=Path)
    add_forge_repo_args(p)
    p.add_argument("--git", default="git")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument(
        "--cache",
        type=Path,
        default=default_cache_path("pr_data_cache.pkl"),
        help="Shared gcli_cache pickle path (default: ~/.fairy/pr_data_cache.pkl). "
             "Shared with fairy so each PR's gcli data is fetched once.",
    )
    p.add_argument(
        "--discussion-cache-max-age-hours",
        type=float,
        default=24.0,
        help="Maximum age of cached PR/issue comments + reviews in hours before "
             "refetching (default: 24). Backstops the three edit-prone fields "
             "(issue_comments, reviews, review_comments) which the forge does "
             "not bump updated_at for on edit/delete.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    add_color_arg(p)
    return p.parse_args()


def q(s: str) -> str:
    if not s:
        return "''"
    safe = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._/:=,+"
    if all(c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def cmd_str(cmd: list[str]) -> str:
    return " ".join(q(x) for x in cmd)


def run(cmd: list[str], *, cwd: Path | None = None) -> str:
    LOG.debug("run: %s", cmd_str(cmd))
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode != 0:
        raise CommandError(
            "command failed\n"
            f"cwd: {cwd or Path.cwd()}\n"
            f"cmd: {cmd_str(cmd)}\n"
            f"exit_code: {p.returncode}\n"
            f"stdout:\n{p.stdout}\n"
            f"stderr:\n{p.stderr}"
        )
    return p.stdout


def compact_ws(text: str) -> str:
    return " ".join(text.split())


def need_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"required executable not found in PATH: {name}")


def utc_str(value: str | None) -> str | None:
    if not value:
        return None
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    dt = datetime.fromisoformat(value).astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M:%S UTC")


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def user_str(user: dict[str, Any] | None) -> str:
    if not user:
        return "unknown-user"
    login = user.get("login") or "unknown-login"
    user_id = user.get("id")
    return f"{login} (id={user_id})" if user_id is not None else login


def stable_write(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def dump_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


PR_FIELDS = ("timeline", "issue_comments", "reviews", "review_comments", "commits", "files")
ISSUE_FIELDS = ("timeline", "issue_comments")


def list_repo_endpoint(args: argparse.Namespace, suffix: str, **params: Any) -> list[dict[str, Any]]:
    """Top-level enumeration helper for ``/repos/{owner}/{repo}{suffix}``.

    Used for the two listing calls (``/issues``, ``/pulls``) that do not
    have a single ``updated_at`` to gate on and therefore live outside
    gcli_cache. Per-item fetches go through ``gcli_cache.get``.
    """
    query = urlencode({"limit": args.limit, **params}, doseq=True)
    path = forge_gcli.build_repo_path(args.owner, args.repo, f"{suffix}?{query}")
    data = forge_gcli.gcli_api(args, path, all_pages=True)
    if not isinstance(data, list):
        raise TypeError(f"expected list from {suffix}, got {type(data).__name__}")
    LOG.debug("fetched %d items from %s", len(data), suffix)
    return data


def get_item_fields(
    args: argparse.Namespace,
    cache: gcli_cache.Cache,
    item: dict[str, Any],
    kind: str,
    field_names: tuple[str, ...],
    *,
    now: datetime,
    cache_max_age: timedelta,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    """Fetch ``field_names`` for one PR/issue through gcli_cache.

    Returns ``(fields, warnings)``. ``fields`` maps each name to a
    list of dicts; empty lists when ``gcli_cache`` raised. Warnings
    are collapsed (one entry per failed item) since the cache fetches
    the whole field set atomically.
    """
    num = int(item["number"])
    live_updated_at = iso_to_dt(item.get("updated_at"))
    if live_updated_at is None:
        msg = f"{kind} #{num}: missing updated_at, cannot gate cache"
        LOG.warning("%s", msg)
        return {name: [] for name in field_names}, [msg]
    try:
        result = gcli_cache.get(
            cache, args, kind, args.owner, args.repo, num, live_updated_at,
            *field_names, max_age=cache_max_age, now=now,
        )
    except Exception as exc:
        msg = f"{kind} #{num}: {compact_ws(str(exc))[:400]}"
        LOG.warning("%s", msg)
        return {name: [] for name in field_names}, [msg]
    return {name: list(result[name]) for name in field_names}, []


def ensure_clean_target(target: Path, *, gcli_account: str | None, forge_type: str, owner: str, repo: str, git: str) -> None:
    target.mkdir(parents=True, exist_ok=True)
    marker_path = target / MARKER

    if not marker_path.exists():
        if any(target.iterdir()):
            raise RuntimeError(
                f"refusing to use non-empty unmanaged directory: {target}\n"
                f"expected an empty directory or an existing {MARKER}"
            )
        run([git, "init"], cwd=target)
        stable_write(
            marker_path,
            dump_json(
                {
                    "format_version": 1,
                    "gcli_account": gcli_account,
                    "forge_type": forge_type,
                    "owner": owner,
                    "repo": repo,
                }
            ),
        )
        (target / "issues").mkdir(exist_ok=True)
        (target / "pulls").mkdir(exist_ok=True)
        return

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    expected = {
        "gcli_account": gcli_account,
        "forge_type": forge_type,
        "owner": owner,
        "repo": repo,
    }
    actual = {k: marker.get(k) for k in expected}
    if actual != expected:
        raise RuntimeError(f"target directory is managed for a different repo: expected {expected}, got {actual}")

    top = {p.name for p in target.iterdir()}
    extra = sorted(top - ALLOWED_TOP)
    if extra:
        raise RuntimeError(f"refusing to touch managed directory with unexpected top-level entries: {extra}")

    if not (target / ".git").exists():
        raise RuntimeError(f"managed target is missing .git: {target}")

    for sub in ("issues", "pulls"):
        d = target / sub
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if p.is_dir():
                continue
            if p.suffix not in {".json", ".md"} or not p.stem.isdigit():
                raise RuntimeError(f"refusing to touch unexpected file in managed tree: {p}")


def labels(obj: dict[str, Any]) -> list[str]:
    return [x.get("name") for x in obj.get("labels") or [] if x.get("name")]


def assignees(obj: dict[str, Any]) -> list[dict[str, Any]]:
    return [norm_user(x) for x in obj.get("assignees") or []]


def milestone(obj: dict[str, Any]) -> str | None:
    ms = obj.get("milestone")
    return ms.get("title") if ms else None


def norm_issue(issue: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "issue",
        "number": issue["number"],
        "title": issue.get("title"),
        "state": issue.get("state"),
        "locked": issue.get("is_locked"),
        "author": norm_user(issue.get("user")),
        "assignees": assignees(issue),
        "labels": labels(issue),
        "milestone": milestone(issue),
        "created_at": utc_str(issue.get("created_at")),
        "updated_at": utc_str(issue.get("updated_at")),
        "closed_at": utc_str(issue.get("closed_at")),
        "url": issue.get("html_url"),
        **({"attachment_urls": urls} if (urls := attachment_urls(issue)) else {}),
        "body": issue.get("body") or "",
    }


def norm_comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": comment.get("id"),
        "author": norm_user(comment.get("user")),
        "created_at": utc_str(comment.get("created_at")),
        "updated_at": utc_str(comment.get("updated_at")),
        "url": comment.get("html_url"),
        **({"attachment_urls": urls} if (urls := attachment_urls(comment)) else {}),
        "body": comment.get("body") or "",
    }


def norm_timeline_event(event: dict[str, Any]) -> dict[str, Any]:
    """Project a Forgejo/Gitea ``/issues/{n}/timeline`` event to a stable shape.

    Forge backends emit many event types with type-specific fields
    (``pull_push`` carries a JSON ``body``; ``label`` carries a
    ``label`` sub-object; etc.). We capture the type, when, who, and
    the raw ``body`` string verbatim -- consumers (humans and LLMs)
    can read the body for typed events, and the type+when give enough
    chronology for events without one.
    """
    return {
        "type": event.get("type"),
        "id": event.get("id"),
        "author": norm_user(event.get("user")),
        "created_at": utc_str(event.get("created_at")),
        "body": event.get("body") or "",
    }


def pr_state(pr: dict[str, Any]) -> str:
    return "merged" if pr.get("merged_at") else (pr.get("state") or "unknown")


def norm_pr(pr: dict[str, Any]) -> dict[str, Any]:
    head = pr.get("head") or {}
    base = pr.get("base") or {}
    return {
        "kind": "pull_request",
        "number": pr["number"],
        "title": pr.get("title"),
        "state": pr_state(pr),
        "draft": pr.get("draft"),
        "locked": pr.get("is_locked"),
        "author": norm_user(pr.get("user")),
        "assignees": assignees(pr),
        "labels": labels(pr),
        "milestone": milestone(pr),
        "created_at": utc_str(pr.get("created_at")),
        "updated_at": utc_str(pr.get("updated_at")),
        "closed_at": utc_str(pr.get("closed_at")),
        "merged_at": utc_str(pr.get("merged_at")),
        "url": pr.get("html_url"),
        "body": pr.get("body") or "",
        "base": {
            "repo": ((base.get("repo") or {}).get("full_name")),
            "ref": base.get("ref"),
        },
        "head": {
            "repo": ((head.get("repo") or {}).get("full_name")),
            "ref": head.get("ref"),
            "sha": (head.get("sha") or "")[:12] or None,
        },
        "merge_commit_sha": (pr.get("merge_commit_sha") or "")[:12] or None,
    }


def norm_review(review: dict[str, Any]) -> dict[str, Any]:
    when = review.get("submitted_at") or review.get("dismissed_at") or review.get("updated_at")
    return {
        "id": review.get("id"),
        "state": review.get("state"),
        "author": norm_user(review.get("user")),
        "submitted_at": utc_str(when),
        "commit_id": (review.get("commit_id") or "")[:12] or None,
        "body": review.get("body") or "",
    }


def norm_review_comment(comment: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": comment.get("id"),
        "review_id": comment.get("pull_request_review_id") or comment.get("review_id"),
        "author": norm_user(comment.get("user")),
        "created_at": utc_str(comment.get("created_at")),
        "updated_at": utc_str(comment.get("updated_at")),
        "path": comment.get("path"),
        "line": comment.get("line") if comment.get("line") is not None else comment.get("position"),
        "old_line": comment.get("old_line") if comment.get("old_line") is not None else comment.get("original_position"),
        "side": comment.get("side"),
        "commit_id": (comment.get("commit_id") or "")[:12] or None,
        "original_commit_id": (comment.get("original_commit_id") or "")[:12] or None,
        "diff_hunk": comment.get("diff_hunk"),
        "body": comment.get("body") or "",
    }


def norm_commit(commit: dict[str, Any]) -> dict[str, Any]:
    c = commit.get("commit") or {}
    a = c.get("author") or {}
    m = c.get("committer") or {}
    return {
        "sha": (commit.get("sha") or "")[:12] or None,
        "author": {
            "login": (commit.get("author") or {}).get("login"),
            "id": (commit.get("author") or {}).get("id"),
            "name": a.get("name"),
            "email": a.get("email"),
        },
        "authored_at": utc_str(a.get("date")),
        "committed_at": utc_str(m.get("date")),
        "message": c.get("message") or "",
    }


def norm_file(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "filename": item.get("filename"),
        "status": item.get("status"),
        "additions": item.get("additions"),
        "deletions": item.get("deletions"),
        "changes": item.get("changes"),
        "previous_filename": item.get("previous_filename"),
    }


def by_time(items: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    return sorted(items, key=lambda x: (x.get(field) or "", int(x.get("id") or 0)))


def attachment_lines(item: dict[str, Any]) -> list[str]:
    return [
        f"Attachment: {a['name']} ({a['size']} bytes) {a['url']}"
        for a in item.get("attachment_urls") or []
    ]


def meta_lines(item: dict[str, Any]) -> list[str]:
    out = [
        f"- Kind: {item['kind']}",
        f"- Number: {item['number']}",
        f"- State: {item['state']}",
        f"- Author: {user_str(item.get('author'))}",
        f"- Created: {item.get('created_at') or '-'}",
        f"- Updated: {item.get('updated_at') or '-'}",
        f"- Closed: {item.get('closed_at') or '-'}",
        f"- Locked: {bool(item.get('locked'))}",
        f"- Labels: {', '.join(item.get('labels') or []) or '-'}",
        f"- Assignees: {', '.join(user_str(x) for x in item.get('assignees') or []) or '-'}",
        f"- Milestone: {item.get('milestone') or '-'}",
        f"- URL: {item.get('url') or '-'}",
    ]
    out += [f"- {line}" for line in attachment_lines(item)]
    if item["kind"] == "pull_request":
        out += [
            f"- Draft: {bool(item.get('draft'))}",
            f"- Merged: {item.get('merged_at') or '-'}",
            f"- Base: {(item.get('base') or {}).get('repo') or '-'}:{(item.get('base') or {}).get('ref') or '-'}",
            f"- Head: {(item.get('head') or {}).get('repo') or '-'}:{(item.get('head') or {}).get('ref') or '-'} @ {(item.get('head') or {}).get('sha') or '-'}",
            f"- Merge commit: {item.get('merge_commit_sha') or '-'}",
        ]
    return out


def add_message_block(
    lines: list[str],
    title: str,
    author: dict[str, Any] | None,
    when: str | None,
    body: str,
    extra: list[str] | None = None,
) -> None:
    lines += [title, "", f"From: {user_str(author)}", f"When: {when or '-'}"]
    if extra:
        lines += extra
    lines += ["", body.rstrip(), ""]


def add_timeline_section(lines: list[str], events: list[dict[str, Any]]) -> None:
    lines += ["## Timeline", ""]
    if not events:
        lines += ["_No timeline events._", ""]
        return
    for i, event in enumerate(events, 1):
        add_message_block(
            lines,
            f"### Event {i}: {event.get('type') or '-'}",
            event["author"],
            event["created_at"],
            event["body"],
        )


def render_issue_md(data: dict[str, Any]) -> str:
    lines = [f"# Issue #{data['number']}: {data['title']}", ""]
    lines += meta_lines(data["issue"])
    if data.get("fetch_warnings"):
        lines += ["", "## Fetch warnings", ""]
        lines += [f"- {warning}" for warning in data["fetch_warnings"]]
    lines += ["", "## Description", "", data["issue"]["body"], "", "## Comments", ""]
    if not data["comments"]:
        lines += ["_No comments._", ""]
    else:
        for i, comment in enumerate(data["comments"], 1):
            add_message_block(lines, f"### Comment {i}", comment["author"], comment["created_at"], comment["body"],
                              attachment_lines(comment) or None)
    add_timeline_section(lines, data["timeline"])
    return "\n".join(lines).rstrip() + "\n"


def render_pr_md(data: dict[str, Any]) -> str:
    lines = [f"# Pull Request #{data['number']}: {data['title']}", ""]
    lines += meta_lines(data["pull_request"])
    if data.get("fetch_warnings"):
        lines += ["", "## Fetch warnings", ""]
        lines += [f"- {warning}" for warning in data["fetch_warnings"]]
    lines += ["", "## Description", "", data["pull_request"]["body"], "", "## Conversation", ""]
    if not data["issue_comments"]:
        lines += ["_No issue comments._", ""]
    else:
        for i, comment in enumerate(data["issue_comments"], 1):
            add_message_block(lines, f"### Comment {i}", comment["author"], comment["created_at"], comment["body"],
                              attachment_lines(comment) or None)

    lines += ["## Reviews", ""]
    if not data["reviews"]:
        lines += ["_No reviews._", ""]
    else:
        for i, review in enumerate(data["reviews"], 1):
            add_message_block(
                lines,
                f"### Review {i}",
                review["author"],
                review["submitted_at"],
                review["body"],
                [f"State: {review.get('state') or '-'}", f"Commit: {review.get('commit_id') or '-'}"],
            )

    lines += ["## Review comments", ""]
    if not data["review_comments"]:
        lines += ["_No review comments._", ""]
    else:
        for i, comment in enumerate(data["review_comments"], 1):
            add_message_block(
                lines,
                f"### Review comment {i}",
                comment["author"],
                comment["created_at"],
                comment["body"],
                [
                    f"Review: {comment.get('review_id') or '-'}",
                    f"Path: {comment.get('path') or '-'}",
                    f"Line: {comment.get('line') if comment.get('line') is not None else '-'}",
                    f"Old line: {comment.get('old_line') if comment.get('old_line') is not None else '-'}",
                    f"Side: {comment.get('side') or '-'}",
                    f"Commit: {comment.get('commit_id') or '-'}",
                    f"Original commit: {comment.get('original_commit_id') or '-'}",
                ],
            )

    lines += ["## Commits", ""]
    if not data["commits"]:
        lines += ["_No commits found._", ""]
    else:
        for i, commit in enumerate(data["commits"], 1):
            author = commit["author"]
            who = " ".join(
                x for x in [
                    author.get("name"),
                    f"{author.get('login')} (id={author.get('id')})" if author.get("login") else None,
                    f"<{author.get('email')}>" if author.get("email") else None,
                ] if x
            )
            lines += [
                f"### Commit {i}: {commit.get('sha') or '-'}",
                "",
                f"Author: {who or '-'}",
                f"Authored: {commit.get('authored_at') or '-'}",
                f"Committed: {commit.get('committed_at') or '-'}",
                "",
                commit["message"].rstrip(),
                "",
            ]

    lines += ["## Files changed", ""]
    if not data["files"]:
        lines += ["_No file list found._", ""]
    else:
        for i, item in enumerate(data["files"], 1):
            lines += [
                f"### File {i}: {item.get('filename') or '-'}",
                "",
                f"Status: {item.get('status') or '-'}",
                f"Additions: {item.get('additions') if item.get('additions') is not None else '-'}",
                f"Deletions: {item.get('deletions') if item.get('deletions') is not None else '-'}",
                f"Changes: {item.get('changes') if item.get('changes') is not None else '-'}",
                f"Previous filename: {item.get('previous_filename') or '-'}",
                "",
            ]

    add_timeline_section(lines, data["timeline"])
    return "\n".join(lines).rstrip() + "\n"


def export_issue(
    target: Path,
    issue: dict[str, Any],
    comments: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    fetch_warnings: list[str],
) -> set[str]:
    num = int(issue["number"])
    base = target / "issues" / f"{num:06d}"
    data = {
        "number": num,
        "title": issue.get("title"),
        "issue": norm_issue(issue),
        "comments": by_time([norm_comment(x) for x in comments], "created_at"),
        "timeline": by_time([norm_timeline_event(x) for x in timeline], "created_at"),
        "fetch_warnings": fetch_warnings,
    }
    stable_write(base.with_suffix(".json"), dump_json(data))
    stable_write(base.with_suffix(".md"), render_issue_md(data))
    return {
        base.with_suffix(".json").relative_to(target).as_posix(),
        base.with_suffix(".md").relative_to(target).as_posix(),
    }


def export_pr(
    target: Path,
    pr: dict[str, Any],
    issue_comments: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    review_comments: list[dict[str, Any]],
    commits: list[dict[str, Any]],
    files: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    fetch_warnings: list[str],
) -> set[str]:
    num = int(pr["number"])
    base = target / "pulls" / f"{num:06d}"
    data = {
        "number": num,
        "title": pr.get("title"),
        "pull_request": norm_pr(pr),
        "issue_comments": by_time([norm_comment(x) for x in issue_comments], "created_at"),
        "reviews": by_time([norm_review(x) for x in reviews], "submitted_at"),
        "review_comments": by_time([norm_review_comment(x) for x in review_comments], "created_at"),
        "commits": [norm_commit(x) for x in commits],
        "files": [norm_file(x) for x in files],
        "timeline": by_time([norm_timeline_event(x) for x in timeline], "created_at"),
        "fetch_warnings": fetch_warnings,
    }
    stable_write(base.with_suffix(".json"), dump_json(data))
    stable_write(base.with_suffix(".md"), render_pr_md(data))
    return {
        base.with_suffix(".json").relative_to(target).as_posix(),
        base.with_suffix(".md").relative_to(target).as_posix(),
    }


def remove_stale_files(target: Path, keep: set[str]) -> None:
    for sub in ("issues", "pulls"):
        d = target / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*"), reverse=True):
            if p.is_dir():
                if not any(p.iterdir()):
                    p.rmdir()
                continue
            if p.relative_to(target).as_posix() not in keep:
                LOG.info("remove stale file: %s", p.relative_to(target))
                p.unlink()


def git_commit_if_needed(git: str, target: Path, message: str) -> bool:
    run([git, "add", "-A"], cwd=target)
    p = subprocess.run(
        [git, "diff", "--cached", "--quiet"],
        cwd=str(target),
        text=True,
        capture_output=True,
        check=False,
    )
    if p.returncode == 0:
        return False
    if p.returncode != 1:
        raise CommandError(
            "git diff --cached --quiet failed\n"
            f"exit_code: {p.returncode}\nstdout:\n{p.stdout}\nstderr:\n{p.stderr}"
        )
    run([git, "commit", "-m", message], cwd=target)
    return True


def main() -> int:
    args = parse_args()
    setup_logging(LOG, bool(args.verbose), forge_gcli.logger, gcli_cache.logger, color=args.color)
    need_binary("gcli")
    need_binary(args.git)

    target = args.target_dir.resolve()
    ensure_clean_target(
        target,
        gcli_account=args.gcli_account,
        forge_type=args.forge_type,
        owner=args.owner,
        repo=args.repo,
        git=args.git,
    )

    LOG.info("exporting %s/%s", args.owner, args.repo)

    cache = gcli_cache.load_cache(args.cache)
    cache_max_age = timedelta(hours=args.discussion_cache_max_age_hours)
    now = datetime.now(timezone.utc)

    issues = list_repo_endpoint(args, "/issues", state="all")
    pulls = list_repo_endpoint(args, "/pulls", state="all")

    pr_numbers = {int(pr["number"]) for pr in pulls}
    real_issues = [issue for issue in issues if int(issue["number"]) not in pr_numbers]
    LOG.info("found %d issues and %d pull requests", len(real_issues), len(pulls))

    fetched_issues: list[tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]] = []
    for issue in real_issues:
        fields, warnings = get_item_fields(
            args, cache, issue, "issues", ISSUE_FIELDS,
            now=now, cache_max_age=cache_max_age,
        )
        fetched_issues.append((issue, fields, warnings))

    fetched_pulls: list[
        tuple[dict[str, Any], dict[str, list[dict[str, Any]]], list[str]]
    ] = []
    for pr in pulls:
        fields, warnings = get_item_fields(
            args, cache, pr, "pulls", PR_FIELDS,
            now=now, cache_max_age=cache_max_age,
        )
        fetched_pulls.append((pr, fields, warnings))

    gcli_cache.save_cache(args.cache, cache)

    keep = {MARKER}
    for issue, fields, fetch_warnings in fetched_issues:
        keep |= export_issue(
            target, issue, fields["issue_comments"], fields["timeline"], fetch_warnings,
        )
    for pr, fields, fetch_warnings in fetched_pulls:
        keep |= export_pr(
            target, pr,
            fields["issue_comments"], fields["reviews"], fields["review_comments"],
            fields["commits"], fields["files"], fields["timeline"], fetch_warnings,
        )

    remove_stale_files(target, keep)

    msg = f"forgejo export: {args.owner}/{args.repo} {now_utc()}"
    if git_commit_if_needed(args.git, target, msg):
        LOG.info("committed changes")
    else:
        LOG.info("no changes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
