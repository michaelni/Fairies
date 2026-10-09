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

A forge PR or issue as fairy and forgejo_export present it: its metadata
and its discussion. Fetching it belongs in forge_gcli.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timezone

import forge_gcli
from common import attachment_urls, iso_to_dt
from forge_gcli import (
    AUTO_MERGE_CANCEL_EVENT,
    AUTO_MERGE_SCHEDULE_EVENT,
    CLOSE_EVENT,
    COMMIT_REFERENCE_EVENTS,
    CROSS_REFERENCE_EVENTS,
    DEPENDENCY_EVENTS,
    MERGE_EVENT,
    PUSH_EVENT,
    REOPEN_EVENT,
    REVIEW_EVENT,
)

__all__ = [
    "FIELDLESS_KINDS",
    "STATE_EVENTS",
    "TIMELINE_ITEMS",
    "author_login",
    "authorship",
    "build_llm_discussion",
    "discussion_time",
    "first_dt",
    "item_standing",
    "labels",
    "logins",
    "milestone",
    "normalize_review_state",
    "reactions_by_content",
    "timeline_events",
]


def labels(obj: dict) -> list[str]:
    return [x.get("name") for x in obj.get("labels") or [] if x.get("name")]


def milestone(obj: dict) -> str | None:
    ms = obj.get("milestone")
    return ms.get("title") if ms else None


def reactions_by_content(reactions: Iterable[dict]) -> dict[str, list[str]]:
    """``{"+1": ["michaelni"], "eyes": ["dowu"]}`` from the forge
    layer's ``{"user", "content"}`` reactions."""
    grouped: dict[str, list[str]] = {}
    for reaction in reactions:
        grouped.setdefault(str(reaction["content"]), []).append(reaction["user"] or "?")
    return grouped


def item_standing(item: dict) -> dict:
    """Who a PR or issue is assigned to, its milestone and whether its
    discussion is locked: ``assignees`` by login, ``milestone`` title or
    None, ``locked``."""
    return {
        "assignees": logins(item.get("assignees")),
        "milestone": milestone(item),
        "locked": forge_gcli.item_locked(item),
    }


def logins(users: object) -> list[str]:
    """The logins in a forge user list, skipping the null a forge puts
    where it could not load the user."""
    return [user["login"] for user in users or () if isinstance(user, dict) and user.get("login")]


def author_login(user: dict | None) -> str:
    """The login of the forge ``user``, "?" when the forge names none."""
    user = user or {}
    for key in ("login", "username", "full_name"):
        value = user.get(key)
        if isinstance(value, str) and value:
            return value
    return "?"


def authorship(user: dict | None) -> dict:
    """``author``, the login of the forge ``user``, and
    ``author_full_name`` where the forge has one other than the login."""
    author, full_name = author_login(user), (user or {}).get("full_name")
    return {"author": author, **({"author_full_name": full_name} if full_name and full_name != author else {})}


def _push_fields(entry: dict) -> dict | None:
    """None for a push whose payload the forge layer could not read:
    suppressing it beats blocking the whole review on a hard failure."""
    if "commit_ids" not in entry:
        return None
    commit_ids = entry["commit_ids"]
    return {"head_sha": commit_ids[-1] if commit_ids else None,
            "is_force_push": entry.get("is_force_push"),
            "commit_ids": commit_ids}


STATE_EVENTS = {CLOSE_EVENT: "closed", REOPEN_EVENT: "reopened",
                MERGE_EVENT: "merged"}


def _assignee_name(entry: dict) -> str:
    return entry.get("assignee_team") or author_login(entry.get("assignee"))


def _either(flag: str, when_set: str, when_clear: str,
            fields: Callable[[dict], dict]) -> Callable[[dict], tuple[str, dict]]:
    """An entry whose ``flag`` tells a removal from an addition."""
    return lambda entry: (when_set if entry.get(flag) else when_clear, fields(entry))


def _milestone(entry: dict) -> tuple[str, dict]:
    old, new = entry.get("old_milestone"), entry.get("milestone")
    if old and new:
        return "milestone_changed", {"old_milestone": old, "new_milestone": new}
    return ("milestone_set", {"milestone": new}) if new else ("milestone_removed", {"milestone": old})


def _ref_change(kind: str) -> Callable[[dict], tuple[str, dict]]:
    """An entry whose ``old_ref`` became ``new_ref``, either absent when
    the forge names none."""
    return lambda entry: (kind, {key: entry[key] for key in ("old_ref", "new_ref") if entry.get(key)})


_MENTION_KINDS = {"description": "mentioned_in_description",
                  "comment": "mentioned_in_comment", None: "mentioned"}


def _mention(entry: dict) -> tuple[str, dict]:
    return _MENTION_KINDS[entry["ref_origin"]], {
        **entry["ref_issue"],
        **({"action": entry["ref_action"]} if entry["ref_action"] is not None else {}),
        **({"body": entry["ref_comment"]} if entry["ref_origin"] == "comment" else {})}


# The forge's name for each timeline entry that carries nothing beyond
# who and when, and the discussion kind it becomes.
FIELDLESS_KINDS = {
    forge_gcli.UNLOCK_EVENT: "unlocked", forge_gcli.PIN_EVENT: "pinned",
    forge_gcli.UNPIN_EVENT: "unpinned",
    AUTO_MERGE_SCHEDULE_EVENT: "auto_merge_scheduled",
    AUTO_MERGE_CANCEL_EVENT: "auto_merge_cancelled",
    "convert_to_draft": "converted_to_draft", "ready_for_review": "ready_for_review",
    "connected": "linked", "disconnected": "unlinked", "transferred": "transferred",
    "marked_as_duplicate": "marked_duplicate", "unmarked_as_duplicate": "unmarked_duplicate",
    "head_ref_restored": "head_branch_restored",
    "start_tracking": "time_tracking_started", "cancel_tracking": "time_tracking_cancelled",
}

# Forge timeline entry -> the discussion kind and fields it becomes,
# None for an entry that cannot be read.
TIMELINE_ITEMS: dict[str, Callable[[dict], tuple[str, dict] | None]] = {
    PUSH_EVENT: lambda entry: (
        ("push", fields) if (fields := _push_fields(entry)) is not None else None),
    forge_gcli.REVIEW_REQUEST_EVENT: _either(
        "removed_assignee", "review_request_withdrawn", "review_request",
        lambda entry: {"reviewer": _assignee_name(entry)}),
    forge_gcli.ASSIGNEES_EVENT: _either(
        "removed_assignee", "unassigned", "assigned",
        lambda entry: {"assignee": _assignee_name(entry)}),
    forge_gcli.LABEL_EVENT: _either(
        "removed_label", "label_removed", "label_added",
        lambda entry: {"label": entry["label"]}),
    forge_gcli.MILESTONE_EVENT: _milestone,
    forge_gcli.TITLE_EVENT: lambda entry: ("title_changed", {
        "old_title": entry["old_title"], "new_title": entry["new_title"]}),
    forge_gcli.TARGET_BRANCH_EVENT: _ref_change("target_branch_changed"),
    forge_gcli.ISSUE_REF_EVENT: _ref_change("issue_ref_changed"),
    forge_gcli.DELETE_BRANCH_EVENT: lambda entry: ("head_branch_deleted", {
        "branch": entry["old_ref"]} if entry.get("old_ref") else {}),
    **dict.fromkeys(DEPENDENCY_EVENTS, _either(
        "removed_dependency", "dependency_removed", "dependency_added",
        lambda entry: entry["dependent_issue"])),
    forge_gcli.LOCK_EVENT: lambda entry: ("locked", {"reason": entry["body"]}),
    forge_gcli.DISMISS_REVIEW_EVENT: lambda entry: ("review_dismissed", {"reason": entry["body"]}),
    "added_deadline": lambda entry: ("due_date_set", {"date": entry["body"]}),
    "modified_deadline": lambda entry: ("due_date_changed", dict(zip(
        ("new_date", "old_date"), entry["body"].split("|", 1)))),
    "removed_deadline": lambda entry: ("due_date_removed", {"date": entry["body"]}),
    "stop_tracking": lambda entry: ("time_tracking_stopped", {"duration": entry["body"]}),
    **{event: (lambda entry, kind=kind: (kind, {})) for event, kind in FIELDLESS_KINDS.items()},
    **dict.fromkeys(CROSS_REFERENCE_EVENTS, _mention),
    **dict.fromkeys(COMMIT_REFERENCE_EVENTS, lambda entry: ("mentioned_in_commit", {
        "sha": entry["ref_commit_sha"], "body": entry["body"]})),
    **dict.fromkeys(STATE_EVENTS, lambda entry: ("state", {
        "state": STATE_EVENTS[entry["type"]],
        **{key: entry[key] for key in ("commit_id", "state_reason") if entry.get(key)}})),
}


def timeline_events(timeline: list[dict]) -> list[dict]:
    """The typed entries of an issue timeline as discussion items: the
    kind, who did it, when, and the kind's own fields as the triage
    prompt in ``llm_prompt`` describes them to the model."""
    items: list[dict] = []
    for entry in timeline:
        item = TIMELINE_ITEMS.get(entry.get("type"), lambda entry: None)(entry)
        if item is not None:
            kind, fields = item
            items.append({"kind": kind, **authorship(entry.get("user")),
                          **fields, "created_at": entry.get("created_at")})
    return items


def first_dt(obj: dict, *keys: str) -> datetime | None:
    for key in keys:
        dt = iso_to_dt(obj.get(key))
        if dt is not None:
            return dt
    return None


def normalize_review_state(state: str | None) -> str | None:
    if not state:
        return None
    s = state.strip().upper().replace("-", "_").replace(" ", "_")
    mapping = {
        "APPROVE": "APPROVED",
        "APPROVED": "APPROVED",
        "REQUEST_CHANGES": "CHANGES_REQUESTED",
        "REQUESTED_CHANGES": "CHANGES_REQUESTED",
        "CHANGES_REQUESTED": "CHANGES_REQUESTED",
        "REJECTED": "CHANGES_REQUESTED",
        "COMMENT": "COMMENTED",
        "COMMENTED": "COMMENTED",
        "PENDING": "PENDING",
    }
    return mapping.get(s, s)


def build_llm_discussion(
    reviews: list[dict],
    comments: list[dict],
    review_comments: list[dict],
    timeline: list[dict] | None = None,
) -> list[dict]:
    """Merge the PR's comments, reviews, review comments and (optionally)
    push events into a single chronologically-sorted list for the LLM.

    ``timeline`` is the raw ``/issues/{n}/timeline`` payload; its typed
    entries are consumed as ``timeline_events`` describes.
    Passing ``None`` (or omitting it) yields a comments-only discussion.
    The push items carry ``kind="push"`` plus ``head_sha`` /
    ``is_force_push`` so the triage LLM can tell unambiguously that
    new code arrived after its prior comment instead of guessing from
    a SHA mentioned in the comment body. A review without a body is
    kept when it carries a verdict (APPROVED / CHANGES_REQUESTED) or
    inline comments: a bare approve click is discussion-worthy for the
    operator and the LLM alike, and the inline comments name their
    review. A review's text is the one its timeline entry
    carries: Forgejo's review listing kept the text as first submitted
    after an edit (observed 2026-10-09 on Forgejo 16.0.5).
    """
    items: list[dict] = []
    if timeline:
        items.extend(timeline_events(timeline))

    for comment in comments:
        body = comment.get("body")
        if not isinstance(body, str):
            body = ""
        urls = attachment_urls(comment)
        # a Forgejo comment can be an attachment with no text at all
        if not body.strip() and not urls:
            continue
        item: dict = {
            "kind": "comment",
            "id": comment.get("id"),
            **authorship(comment.get("user")),
            "created_at": comment.get("created_at"),
            "updated_at": comment.get("updated_at"),
            "url": comment.get("html_url"),
            "body": body,
        }
        if urls:
            item["attachment_urls"] = urls
        if comment.get("reactions"):
            item["reactions"] = reactions_by_content(comment["reactions"])
        items.append(item)

    edited = {entry["review_id"]: entry["body"]
              for entry in timeline or () if entry["type"] == REVIEW_EVENT}
    commented = {_review_of(comment) for comment in review_comments}
    for review in reviews:
        body = edited.get(review.get("id"), review.get("body"))
        if not isinstance(body, str) or not body.strip():
            if normalize_review_state(review.get("state")) \
                    not in ("APPROVED", "CHANGES_REQUESTED") and review.get("id") not in commented:
                continue
            body = ""
        items.append(
            {
                "kind": "review",
                "id": review.get("id"),
                **authorship(review.get("user")),
                "state": review.get("state"),
                "submitted_at": review.get("submitted_at"),
                "updated_at": review.get("updated_at"),
                "commit_id": review.get("commit_id"),
                "body": body,
            }
        )

    for comment in review_comments:
        body = comment.get("body")
        if not isinstance(body, str) or not body.strip():
            continue
        items.append(
            {
                "kind": "review_comment",
                "id": comment.get("id"),
                "review_id": _review_of(comment),
                **authorship(comment.get("user")),
                "created_at": comment.get("created_at"),
                "updated_at": comment.get("updated_at"),
                "path": comment.get("path"),
                "line": comment.get("line") if comment.get("line") is not None else comment.get("position"),
                "old_line": comment.get("old_line") if comment.get("old_line") is not None
                else comment.get("original_position"),
                "side": comment.get("side"),
                "commit_id": comment.get("commit_id"),
                "original_commit_id": comment.get("original_commit_id"),
                "diff_hunk": comment.get("diff_hunk"),
                "body": body,
                **({"reactions": reactions_by_content(comment["reactions"])}
                   if comment.get("reactions") else {}),
            }
        )

    items.sort(key=discussion_time)
    return items


def _review_of(review_comment: dict) -> int | None:
    return review_comment.get("pull_request_review_id") or review_comment.get("review_id")


def discussion_time(item: dict) -> datetime:
    """A discussion entry's arrival time: forges bump the item's
    updated_at watermark on arrivals, never on edits (see gcli_cache),
    so an entry compares by creation, not by when it was last edited.
    The epoch for an entry without a stamp."""
    return first_dt(item, "submitted_at", "created_at", "updated_at") \
        or datetime.min.replace(tzinfo=timezone.utc)
