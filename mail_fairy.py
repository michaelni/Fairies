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

mail_fairy: forward mailing-list replies into Forgejo PR/Issue comments.

The setup this is built for: a public mailing list is CC'd on every
PR/Issue notification the forge generates. Humans reply to that mail
on the list. Those replies are invisible to Forgejo. mail_fairy walks
a Maildir, threads replies via ``In-Reply-To``, locates the ones whose
thread root is a Forgejo notification (a mail that the forge itself
sent to the list), and posts the reply text back to the corresponding
PR/Issue as a comment via ``gcli``.

The script is project-agnostic. Every project-specific value -- the
repository owner/name, the forge base URL, the ``gcli`` account, the
regex matching the forge's outgoing bot address, the loop-guard
regexes -- is supplied via command-line arguments. Nothing about any
specific mailing list or any specific project is hard-coded.

Threading-based classification:
- A "Forgejo notification" is detected by ``Subject:`` matching
  ``(PR #N)`` or ``(Issue #N)`` AND the body containing a URL of the
  form ``<host>/<owner>/<repo>/(pulls|issues)/N`` for the same N. Both
  signals must agree.
- A non-root mail is classified by walking ``In-Reply-To`` upward
  through the maildir until that detection hits. Every mail in that
  subtree belongs to the corresponding PR/Issue. Mails whose chain
  cannot be resolved to such an ancestor are skipped (they are not
  posted under any guess).

Identity (we cannot post as the original author):
The Forgejo API does NOT let an account post on behalf of an arbitrary
non-registered user. mail_fairy therefore posts as its own ``gcli``
account, and prefixes the body with a single attribution line whose
entire text is a clickable link to the original mail in the list's
public archive. The ``Archived-At:`` header that Mailman 3 inserts
gives us this URL directly (lore- and public-inbox-style mirrors
provide it out of the box) -- no Mailman 2 SHA1+base32 hashing
required.

Dedup (defence in depth):
1. A local state pickle of forwarded ``Message-ID``s for fast restart.
2. Every posted comment carries a stable HTML-comment marker
   ``<!-- mail-fairy:msgid:<id> -->``. Before posting, mail_fairy
   fetches the target's existing comments through ``gcli api`` and
   skips if any already carries that marker. This survives state-file
   loss.

Body handling:
- Mailman trailing footer (the ``___...___\n<list-name> mailing list
  -- ...`` block) is always stripped.
- Inline replies and patch hunks are kept verbatim. Lists with an
  inline-reply convention (interleaved reply text with quoted
  context, common on technical mailing lists) get the structure
  preserved on the forge.
- A trailing block of consecutive quoted lines (possibly preceded by
  an ``On ... wrote:`` attribution) at the very end of the body is
  dropped. This catches the rare bottom-full-quote case without
  touching well-trimmed inline replies.
"""

from __future__ import annotations

import argparse
import email
import email.policy
import email.utils
import logging
import os
import pickle
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Iterable

# Shared bot plumbing lives in forge_gcli (gcli_prefix / gcli_api /
# run_gcli_editor_submission etc.) and common (logging + pickle
# helpers). Keeping those imports module-level lets us route
# forge_gcli's debug log through our setup_logging call.
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402
from common import atomic_write_pickle, setup_logging  # noqa: E402


logger = logging.getLogger("mail_fairy")


# ---------------------------------------------------------------------------
# Regexes and constants
# ---------------------------------------------------------------------------

_MARKER_FMT = "<!-- mail-fairy:msgid:{msgid} -->"
_MARKER_RE = re.compile(r"<!--\s*mail-fairy:msgid:([^\s>]+)\s*-->")

# Mailman list footer always begins with a long underscore line.
_MAILMAN_FOOTER_RE = re.compile(r"^_{30,}\s*$")

_QUOTE_LINE_RE = re.compile(r"^\s*>")

# Attribution lines that immediately precede a bottom full-quote,
# e.g. ``On 2026-04-21, X wrote:`` or ``<Name> via <list> (HE12026-04-21):``.
# We err on the conservative side: only strip a line that ends in
# ``wrote:`` (or its German/Spanish equivalents) or ``):`` -- the
# parenthesised-date form some mail clients produce.
_ATTRIBUTION_LINE_RE = re.compile(
    r"(.+\b(wrote|schrieb|escribió)\s*:|.+\)\s*:)\s*$",
    re.IGNORECASE,
)

# Detect a hard-wrapped attribution closer by structure rather
# than language: "On <date>, X <addr@host> wrote:" can wrap onto
# two physical lines because mail clients (notably Gmail) hard-wrap
# inside the ``<email>`` envelope. The resulting closing line
# starts with an email-like envelope tail ("addr@host>" or
# "<addr@host>") followed by the wrote/schrieb/escribió verb that
# _ATTRIBUTION_LINE_RE already matches. When the closer line
# starts that way, the "On <date>, X <" head is on the line above.
# This is language-agnostic: we never look at the head's leading
# word, which would otherwise need a per-language enumeration.
_WRAPPED_ATTRIBUTION_ENVELOPE_HEAD_RE = re.compile(r"^\s*<?\S*@\S+>?")

# ``<Name> via <list-name>`` -- the From: display-name rewrite Mailman
# 3 applies to every list mail. The original address is typically left
# in Cc:.
_VIA_LIST_RE = re.compile(r"\s+via\s+[\w\-.]+\s*$", re.IGNORECASE)

# Maildir filename convention: ``<unix-time>.<id>.<host>``. The leading
# integer is what we use for cheap age pre-filtering before opening
# any mail file.
_MAILDIR_FILENAME_TS_RE = re.compile(r"^(\d{9,12})\.")

DEFAULT_MAX_AGE_DAYS = 14
DEFAULT_MAX_MAIL_BYTES = 256 * 1024
DEFAULT_FORGE_BOT_SENDER_RE = r"^code@"
DEFAULT_STATE_PATH = Path.home() / ".fairy" / "mail_fairy_state.pkl"

_STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Forge flavors
# ---------------------------------------------------------------------------
#
# All forge-specific shape (notification subject regex, user-visible
# URL path segments) lives in a ForgeFlavor instance. mail_fairy
# itself is otherwise backend-agnostic: the rest of the pipeline
# accepts a flavor and asks it to translate canonical (kind, number)
# pairs back into URLs. Adding a new backend means adding one
# ForgeFlavor instance and registering it in FORGE_FLAVORS below.
#
# This commit only registers the forgejo flavor; github/gitlab follow
# in a separate change.

# Canonical kinds. Stored in ForgejoTarget.kind. The active flavor
# knows how to map them to a forge-specific URL segment ("pulls" on
# Forgejo, "pull" on GitHub, "merge_requests" on GitLab, etc.).
KIND_PR = "pr"
KIND_ISSUE = "issue"


@dataclass
class ForgeFlavor:
    """Per-backend tuning for mail subject parsing and URL building."""

    name: str
    # List of (subject_regex, kind). Each regex must capture the
    # PR/Issue number as group 1. ``kind`` is KIND_PR, KIND_ISSUE, or
    # None. ``None`` means the subject alone cannot disambiguate PR
    # vs Issue (e.g. GitHub's ``(#N)``); the body URL match in
    # ``build_thread_index`` then picks the kind.
    subject_patterns: tuple[tuple[re.Pattern[str], str | None], ...]
    # Canonical kind -> URL path segment used in user-visible URLs.
    kind_url_segment: dict[str, str]
    # GitLab convention: ``{host}/{owner}/{repo}/-/{segment}/{n}``
    # (note the ``/-/`` separator). Set True for gitlab.
    url_dash_prefix: bool = False
    # If True, this flavor's parsing/posting paths are not exercised
    # against real notification mails. main() warns once at startup.
    untested: bool = False

    def html_url(
        self, host: str, owner: str, repo: str, kind: str, number: int,
    ) -> str:
        try:
            seg = self.kind_url_segment[kind]
        except KeyError as exc:
            raise ValueError(
                f"forge flavor {self.name!r} does not declare a URL "
                f"segment for canonical kind {kind!r}; declared kinds: "
                f"{sorted(self.kind_url_segment)}"
            ) from exc
        sep = "/-/" if self.url_dash_prefix else "/"
        return f"{host.rstrip('/')}/{owner}/{repo}{sep}{seg}/{number}"

    def classify_subject(
        self, subject: str,
    ) -> list[tuple[str | None, int]]:
        """Return ``[(kind|None, number), ...]`` candidates from a subject.

        Multiple candidates are only returned if more than one
        configured pattern matches the same subject.
        """
        subject = subject or ""
        out: list[tuple[str | None, int]] = []
        seen: set[tuple[str | None, int]] = set()
        for regex, kind in self.subject_patterns:
            for m in regex.finditer(subject):
                try:
                    n = int(m.group(1))
                except (IndexError, ValueError):
                    continue
                key = (kind, n)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
        return out

    def candidate_targets(
        self, host: str, owner: str, repo: str, subject: str,
    ) -> list["ForgejoTarget"]:
        """Build candidate ForgejoTarget(s) from a subject line.

        For subjects whose kind is unambiguous from the regex
        (forgejo) this returns one target per matched ``(kind, N)``.
        For subjects where the regex captures only a number
        (github-style ``(#N)``), this returns a target for every
        declared kind for that number; the body URL match in
        ``build_thread_index`` picks the right one.
        """
        out: list["ForgejoTarget"] = []
        for kind, number in self.classify_subject(subject):
            kinds = (
                [kind] if kind is not None
                else list(self.kind_url_segment.keys())
            )
            for k in kinds:
                out.append(
                    ForgejoTarget(
                        host=host, owner=owner, repo=repo,
                        kind=k, number=number, flavor_name=self.name,
                    )
                )
        return out


FORGEJO_FLAVOR = ForgeFlavor(
    name="forgejo",
    subject_patterns=(
        (re.compile(r"\(PR\s*#(\d+)\)\s*$"), KIND_PR),
        (re.compile(r"\(Issue\s*#(\d+)\)\s*$"), KIND_ISSUE),
    ),
    kind_url_segment={KIND_PR: "pulls", KIND_ISSUE: "issues"},
)

# UNTESTED. GitHub notification mail (from notifications@github.com)
# typically ends the Subject in ``(#N)`` for both PRs and Issues, so
# the regex captures only the number; build_thread_index then tries
# both a PR-shaped and an Issue-shaped body URL and the matching one
# wins. URL segments are GitHub's: ``/owner/repo/pull/N`` (singular)
# and ``/owner/repo/issues/N``.
GITHUB_FLAVOR = ForgeFlavor(
    name="github",
    subject_patterns=(
        (re.compile(r"\(#(\d+)\)\s*$"), None),
    ),
    kind_url_segment={KIND_PR: "pull", KIND_ISSUE: "issues"},
    untested=True,
)

# UNTESTED. GitLab merge-request notification subjects typically end
# in ``(!N)``; issue subjects in ``(#N)``. URLs use the ``/-/``
# separator, e.g. ``/group/project/-/merge_requests/N``.
GITLAB_FLAVOR = ForgeFlavor(
    name="gitlab",
    subject_patterns=(
        (re.compile(r"\(!(\d+)\)\s*$"), KIND_PR),
        (re.compile(r"\(#(\d+)\)\s*$"), KIND_ISSUE),
    ),
    kind_url_segment={KIND_PR: "merge_requests", KIND_ISSUE: "issues"},
    url_dash_prefix=True,
    untested=True,
)

# Registry of accepted ``--forge-flavor`` (and fall-through
# ``--forge-type``) values. ``gitea`` aliases ``forgejo`` because
# both speak the same notification mail shape.
FORGE_FLAVORS: dict[str, ForgeFlavor] = {
    "forgejo": FORGEJO_FLAVOR,
    "gitea": FORGEJO_FLAVOR,
    "github": GITHUB_FLAVOR,
    "gitlab": GITLAB_FLAVOR,
}


def select_forge_flavor(name: str) -> ForgeFlavor:
    """Resolve a flavor by registry key, raising on unknown names."""
    key = (name or "").lower().strip()
    if key in FORGE_FLAVORS:
        return FORGE_FLAVORS[key]
    raise ValueError(
        f"unknown forge flavor: {name!r}. "
        f"Supported: {', '.join(sorted(set(FORGE_FLAVORS)))}. "
        f"Pass one of those via --forge-flavor (or --forge-type to "
        f"also propagate to gcli)."
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForgejoTarget:
    host: str
    owner: str
    repo: str
    kind: str            # canonical KIND_PR or KIND_ISSUE
    number: int
    flavor_name: str = "forgejo"

    @property
    def flavor(self) -> ForgeFlavor:
        return FORGE_FLAVORS[self.flavor_name]

    @property
    def html_url(self) -> str:
        return self.flavor.html_url(
            self.host, self.owner, self.repo, self.kind, self.number,
        )


@dataclass
class MailHeaders:
    path: Path
    file_ts: float
    file_size: int
    message_id: str = ""
    in_reply_to: str = ""
    references: tuple[str, ...] = ()
    subject: str = ""
    from_raw: str = ""
    cc_raw: str = ""
    x_mailfrom: str = ""
    date: datetime | None = None
    lore_url: str = ""
    list_id: str = ""


@dataclass
class MailDecision:
    headers: MailHeaders
    action: str
    reason: str
    target: ForgejoTarget | None = None
    body: str = ""           # composed comment body, ready to post
    author_name: str = ""
    author_email: str = ""
    date_iso: str = ""
    raw_body_excerpt: str = ""  # first ~200 chars of original body for log


# Action vocabulary used in MailDecision.action and surfaced in summary.
ACTIONABLE = "post"
SKIP_TOO_OLD = "skip:too-old"
SKIP_TOO_LARGE = "skip:too-large"
SKIP_NO_MSGID = "skip:no-message-id"
SKIP_NO_PARENT = "skip:no-parent-in-thread"
SKIP_NOT_FORGE_THREAD = "skip:thread-root-not-forge"
SKIP_BOT_MAIL = "skip:forge-bot-or-self"
SKIP_DEDUP_LOCAL = "skip:already-forwarded-local"
SKIP_DEDUP_REMOTE = "skip:already-forwarded-remote"
SKIP_EMPTY_BODY = "skip:empty-body-after-cleanup"
SKIP_PARSE_ERROR = "skip:parse-error"
SKIP_FULL_QUOTE_WITH_FOOTER = "skip:full-quote-with-footer"


# ---------------------------------------------------------------------------
# Pure helpers (covered by tests/test_mail_fairy.py)
# ---------------------------------------------------------------------------


def normalize_msgid(value: str | None) -> str:
    """Strip surrounding whitespace and angle brackets from a Message-ID."""
    if not value:
        return ""
    value = value.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1]
    return value.strip()


def parse_maildir_filename_ts(name: str) -> float | None:
    """Return the unix timestamp encoded in a Maildir filename.

    Maildir convention is ``<unix-time>.<id>.<host>``. If the filename
    does not begin with a plausible unix timestamp (9-12 digits) we
    return None and let the caller fall back to ``stat()``.
    """
    m = _MAILDIR_FILENAME_TS_RE.match(name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_addresses(raw: str) -> list[tuple[str, str]]:
    """Parse a header value containing one or more RFC 5322 addresses.

    Returns ``[(display_name, email_addr), ...]``. ``display_name``
    has any ``via <list>`` Mailman rewrite stripped so it is safe to
    match against a Cc: entry.
    """
    if not raw:
        return []
    out: list[tuple[str, str]] = []
    for name, addr in email.utils.getaddresses([raw]):
        name = (name or "").strip()
        addr = (addr or "").strip()
        out.append((name, addr))
    return out


def strip_via_list(display_name: str) -> str:
    return _VIA_LIST_RE.sub("", display_name).strip()


def extract_lore_url(archived_at_values: Iterable[str]) -> str:
    """Pick the lore-style ``Archived-At:`` URL from one or more values.

    Mailman 3 typically inserts both a hashed mailman3 archive URL and
    a public-inbox/lore-style URL when one is configured. We prefer
    the lore one because it is the human-readable archive most lists
    publish; the mailman3-hashed URL is opaque and resolves only via
    the list's web UI.
    """
    for raw in archived_at_values:
        if not raw:
            continue
        # ``Archived-At:`` values are angle-bracketed URLs.
        for piece in raw.split(","):
            piece = piece.strip()
            if piece.startswith("<") and piece.endswith(">"):
                piece = piece[1:-1]
            if "/lore/" in piece:
                return piece
    return ""


def body_confirms_target(body: str, target: ForgejoTarget) -> bool:
    """True if the mail body contains the canonical URL of ``target``.

    The URL is computed by the active forge flavor. We require BOTH
    the subject tag and a body URL to agree before we accept a mail
    as a forge notification root, so a list mail that happened to
    put ``(PR #99)`` in its subject for unrelated reasons cannot
    trick us into mis-classifying.
    """
    return target.html_url in body


def strip_mailman_footer(body: str) -> str:
    """Remove the trailing Mailman list footer if present.

    Mailman footers always begin with a line of 30+ underscores
    followed by ``<list-name> mailing list -- ...``. We cut from the
    underscore line onward; the underscore line alone is a strong
    enough signal that we do not need to inspect the list-name to
    confirm the match.
    """
    lines = body.splitlines()
    for i, line in enumerate(lines):
        if _MAILMAN_FOOTER_RE.match(line):
            return "\n".join(lines[:i]).rstrip() + "\n"
    return body


# Match a QUOTED Mailman v3 list-footer continuation line: one or more
# stacked "> " quote prefixes followed by the canonical
# "<list-name> mailing list -- <list-addr>" shape. Presence of this
# pattern anywhere in the body means the sender quoted a previous
# list mail in full (footer included). build_decision either skips
# such mails or drops just the offending quote block, depending on
# ``--full-quote-action``.
_QUOTED_LIST_FOOTER_RE = re.compile(
    r"^\s*(?:>\s*)+\S+\s+mailing list\s*--\s*\S+@\S+",
    re.MULTILINE,
)


def strip_quoted_list_footer_blocks(body: str) -> str:
    """Drop every ``>``-quoted block that contains a quoted Mailman
    list-footer line.

    For each line matching ``_QUOTED_LIST_FOOTER_RE``, walk outward
    through contiguous ``>``-prefixed lines (in both directions) to
    find the surrounding quote block and remove it. If an
    ``On <date>, X wrote:`` attribution line (allowing one blank
    line in between) immediately precedes the dropped block, that
    line is dropped too -- otherwise it would be left dangling
    above an empty space.

    Bodies with no quoted full-mail block are returned unchanged.
    Returns a body that always ends in exactly one ``\\n`` (or is
    empty when everything was dropped).
    """
    if not _QUOTED_LIST_FOOTER_RE.search(body):
        return body
    lines = body.splitlines()
    n = len(lines)
    drop = [False] * n
    for i, line in enumerate(lines):
        if not _QUOTED_LIST_FOOTER_RE.match(line):
            continue
        start = i
        while start > 0 and _QUOTE_LINE_RE.match(lines[start - 1]):
            start -= 1
        end = i
        while end + 1 < n and _QUOTE_LINE_RE.match(lines[end + 1]):
            end += 1
        for k in range(start, end + 1):
            drop[k] = True
        attr_idx = start - 1
        while attr_idx >= 0 and lines[attr_idx].strip() == "":
            attr_idx -= 1
        if attr_idx >= 0 and _ATTRIBUTION_LINE_RE.match(lines[attr_idx]):
            drop[attr_idx] = True
            # Hard-wrapped attribution: closer starts with an
            # envelope tail ("addr@host>" / "<addr@host>"), meaning
            # the "On <date>, X <" head wrapped onto the previous
            # line. Drop wrapped continuation lines that contain
            # envelope/email characters (``<`` or ``@``). Bail at
            # blanks or quoted lines so paragraph boundaries are
            # respected. The data-character requirement is what
            # keeps us from eating unrelated prose that happens to
            # sit directly above the wrap.
            if _WRAPPED_ATTRIBUTION_ENVELOPE_HEAD_RE.match(lines[attr_idx]):
                for k in range(attr_idx - 1, max(-1, attr_idx - 5), -1):
                    line_k = lines[k]
                    if not line_k.strip() or _QUOTE_LINE_RE.match(line_k):
                        break
                    if "<" not in line_k and "@" not in line_k:
                        break
                    drop[k] = True
    out = "\n".join(line for line, d in zip(lines, drop) if not d).rstrip("\n")
    return out + "\n" if out else ""


def strip_bottom_quote(body: str) -> str:
    """Drop a trailing block of quoted lines if it sits at the very end.

    Walks from the end of the body backward through blank/quote lines
    until it hits a non-quote non-blank line. If there is real content
    above and the trailing block is non-empty, we drop the trailing
    block (and an optional ``On ... wrote:`` attribution line right
    above it). Otherwise the body is returned unchanged.

    This intentionally preserves inline replies (text/quote/text) and
    patch hunks (which look like quotes only at first glance but are
    not preceded by an attribution line and never sit at the end of a
    list reply that has actual reply content).
    """
    lines = body.rstrip("\n").split("\n")
    if not lines:
        return body

    i = len(lines) - 1
    # Skip trailing blank lines.
    while i >= 0 and lines[i].strip() == "":
        i -= 1
    end_of_quote_block = i

    # We need at least one non-blank line at the end and it must be a
    # quote for the bottom-quote pattern to apply.
    if end_of_quote_block < 0 or not _QUOTE_LINE_RE.match(lines[end_of_quote_block]):
        return body

    # Walk up through contiguous quote/blank lines.
    while i >= 0 and (
        _QUOTE_LINE_RE.match(lines[i]) or lines[i].strip() == ""
    ):
        i -= 1
    start_of_quote_block = i + 1

    # If the entire body is a quote, leave it alone -- there is no
    # "new content" to keep, so dropping would yield an empty post.
    if start_of_quote_block == 0:
        return body

    # Optionally also drop an "On ... wrote:" attribution line that
    # sits immediately above the quote block.
    cutoff = start_of_quote_block
    while cutoff > 0 and lines[cutoff - 1].strip() == "":
        cutoff -= 1
    if cutoff > 0 and _ATTRIBUTION_LINE_RE.match(lines[cutoff - 1]):
        cutoff -= 1

    kept = lines[:cutoff]
    # Trim trailing blanks on the kept portion.
    while kept and kept[-1].strip() == "":
        kept.pop()
    return ("\n".join(kept) + "\n") if kept else body


def extract_text_body(msg: EmailMessage) -> str:
    """Return the best-effort text/plain body of an email message.

    Falls back to walking parts if ``get_body`` can't find a clean
    text/plain part (e.g. multipart/signed wrappers).
    """
    try:
        body_part = msg.get_body(preferencelist=("plain",))
    except Exception:
        body_part = None
    if body_part is not None:
        try:
            return str(body_part.get_content())
        except Exception:
            pass
    # Manual walk fallback.
    for part in msg.walk():
        if part.get_content_type() == "text/plain":
            try:
                return str(part.get_content())
            except Exception:
                continue
    return ""


def extract_author(headers: MailHeaders) -> tuple[str, str]:
    """Extract a (display_name, email_addr) pair for the human author.

    Mailman 3 rewrites the From: header on every list mail to
    ``<name> via <list-name> <list-addr>`` while leaving the original
    sender's address in the Cc:. We strip the via-list suffix off the
    From display name and look for a matching display name in the Cc:;
    if found, we use that Cc: address. Otherwise we fall back to the
    (rewritten) From address, which still gives the reader the
    display name even if the email is the list's.
    """
    from_pairs = parse_addresses(headers.from_raw)
    if not from_pairs:
        return ("", "")
    from_name_raw, from_addr = from_pairs[0]
    from_name = strip_via_list(from_name_raw) or from_name_raw

    cc_pairs = parse_addresses(headers.cc_raw)
    for cc_name, cc_addr in cc_pairs:
        if not cc_addr:
            continue
        if cc_name and cc_name.strip() == from_name:
            return (from_name, cc_addr)

    # Fallback: prefer any Cc: address that is not the forge bot's
    # outgoing address (X-MailFrom of the *root*; we don't have that
    # cheaply here, so this branch just takes the first Cc: that does
    # not look like the rewritten list address).
    list_addr = from_addr.lower()
    for _cc_name, cc_addr in cc_pairs:
        if cc_addr and cc_addr.lower() != list_addr:
            return (from_name, cc_addr)

    return (from_name, from_addr)


def format_attribution_link(
    author_name: str,
    author_email: str,
    date_iso: str,
    lore_url: str,
) -> str:
    """Single-line markdown link of the form requested by the user.

    Example output::

        [Fw by mail-fairy From: Jane Doe \\<jane@example.org\\> 2026-04-21 23:45 UTC](https://lists.example.org/lore/devlist/abc123@example.org/)

    The angle brackets around the email are escaped so markdown does
    not parse them as an autolink. If we have no archive URL the line
    falls back to plain (non-link) text so the attribution is still
    visible on the forge.
    """
    parts: list[str] = ["Fw by mail-fairy"]
    if author_name or author_email:
        if author_name and author_email:
            parts.append(f"From: {author_name} \\<{author_email}\\>")
        elif author_name:
            parts.append(f"From: {author_name}")
        else:
            parts.append(f"From: \\<{author_email}\\>")
    if date_iso:
        parts.append(date_iso)
    label = " ".join(parts)
    if lore_url:
        return f"[{label}]({lore_url})"
    return label


def find_marker_msgids(comments: Iterable[dict]) -> set[str]:
    """Return the set of mail Message-IDs already forwarded to a target.

    ``comments`` is the list returned by Forgejo's
    ``GET /repos/.../issues/<n>/comments``. Each entry has a ``body``
    string; we scan it for the mail-fairy marker.
    """
    out: set[str] = set()
    for c in comments:
        body = c.get("body") if isinstance(c, dict) else None
        if not isinstance(body, str):
            continue
        for m in _MARKER_RE.finditer(body):
            out.add(m.group(1))
    return out


_BACKTICK_RUN_RE = re.compile(r"`+")


def _fence_for(body: str) -> str:
    longest = 0
    for m in _BACKTICK_RUN_RE.finditer(body):
        if len(m.group()) > longest:
            longest = len(m.group())
    return "`" * max(3, longest + 1)


def compose_comment_body(
    cleaned_body: str,
    attribution_line: str,
    msgid: str,
) -> str:
    """Glue the attribution line, fenced original body and dedup marker.

    Layout::

        <attribution-line>

        ```text
        <cleaned mail body>
        ```

        <!-- mail-fairy:msgid:<id> -->

    The body is wrapped in a fenced code block so Forgejo's Markdown
    renderer does not interpret the content

    The fence is sized dynamically by ``_fence_for`` so any
    backtick runs inside the body cannot terminate it.
    """
    fence = _fence_for(cleaned_body)
    parts = [
        attribution_line.rstrip(),
        "",
        f"{fence}text",
        cleaned_body.strip("\n"),
        fence,
        "",
        _MARKER_FMT.format(msgid=msgid),
    ]
    return "\n".join(parts).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Maildir scanning
# ---------------------------------------------------------------------------


def iter_maildir_files(maildirs: Iterable[Path]) -> Iterable[Path]:
    """Yield ``Path`` for every mail file under ``new/`` and ``cur/``.

    ``tmp/`` is intentionally excluded -- those files are mid-delivery
    and may be incomplete.
    """
    for root in maildirs:
        for sub in ("new", "cur"):
            d = root / sub
            if not d.is_dir():
                logger.debug("maildir subfolder missing path=%s", d)
                continue
            count = 0
            with os.scandir(d) as it:
                for entry in it:
                    if entry.is_file():
                        count += 1
                        yield Path(entry.path)
            logger.debug("maildir scan path=%s entries=%d", d, count)


def read_headers(path: Path, *, max_bytes: int = 64 * 1024) -> MailHeaders | None:
    """Cheap header-only parse.

    We read at most ``max_bytes`` to find the end of the header block;
    that is enough for any real-world mail header (the longest field
    we care about is ``References:``, which is rarely > 4 KB). Body
    bytes are not read here.
    """
    try:
        st = path.stat()
    except OSError as exc:
        logger.warning("stat failed path=%s err=%s", path, exc)
        return None

    file_ts = parse_maildir_filename_ts(path.name)
    if file_ts is None:
        file_ts = st.st_mtime

    try:
        with path.open("rb") as f:
            head = f.read(max_bytes)
    except OSError as exc:
        logger.warning("read failed path=%s err=%s", path, exc)
        return None

    # Truncate at the first blank line so the parser does not try to
    # interpret body bytes as headers.
    sep = head.find(b"\r\n\r\n")
    if sep < 0:
        sep = head.find(b"\n\n")
    if sep >= 0:
        head = head[:sep]

    try:
        msg = email.message_from_bytes(head, policy=email.policy.default)
    except Exception as exc:  # noqa: BLE001
        logger.warning("header parse failed path=%s err=%s", path, exc)
        return None

    h = MailHeaders(path=path, file_ts=file_ts, file_size=st.st_size)
    h.message_id = normalize_msgid(msg.get("Message-ID"))
    h.in_reply_to = normalize_msgid(msg.get("In-Reply-To"))
    refs_raw = msg.get("References", "")
    h.references = tuple(
        normalize_msgid(r) for r in refs_raw.split() if r.strip()
    )
    h.subject = (msg.get("Subject") or "").strip()
    h.from_raw = msg.get("From", "") or ""
    h.cc_raw = msg.get("Cc", "") or ""
    h.x_mailfrom = (msg.get("X-MailFrom") or "").strip()
    h.list_id = (msg.get("List-Id") or "").strip()
    date_raw = msg.get("Date")
    if date_raw:
        try:
            h.date = email.utils.parsedate_to_datetime(date_raw)
        except (TypeError, ValueError):
            h.date = None
    h.lore_url = extract_lore_url(msg.get_all("Archived-At") or [])
    return h


def read_body(path: Path) -> str:
    try:
        with path.open("rb") as f:
            data = f.read()
    except OSError as exc:
        logger.warning("body read failed path=%s err=%s", path, exc)
        return ""
    try:
        msg = email.message_from_bytes(data, policy=email.policy.default)
    except Exception as exc:  # noqa: BLE001
        logger.warning("body parse failed path=%s err=%s", path, exc)
        return ""
    return extract_text_body(msg)


# ---------------------------------------------------------------------------
# Threading and root identification
# ---------------------------------------------------------------------------


@dataclass
class ThreadIndex:
    by_msgid: dict[str, MailHeaders] = field(default_factory=dict)
    forge_root_target: dict[str, ForgejoTarget] = field(default_factory=dict)


def build_thread_index(
    headers: list[MailHeaders],
    *,
    forge_host: str,
    forge_owner: str,
    forge_repo: str,
    flavor: ForgeFlavor = FORGEJO_FLAVOR,
) -> ThreadIndex:
    """Index headers by Message-ID and pre-compute root→target mapping.

    A "forge root" is a mail whose subject yields one or more
    candidate targets via ``flavor.candidate_targets`` AND whose
    body contains the canonical URL of one of those candidates.
    Reading body for every mail is expensive on a 285k-mail
    maildir, so we only crack open the body for headers whose
    subject matches the flavor's regex.

    For unambiguous flavors (forgejo) there is exactly one
    candidate per matching subject. For flavors whose subject regex
    cannot disambiguate PR vs Issue (github), every declared kind
    is a candidate and the body URL match decides which one wins.
    """
    idx = ThreadIndex()
    for h in headers:
        if h.message_id and h.message_id not in idx.by_msgid:
            idx.by_msgid[h.message_id] = h

    candidate_count = 0
    confirmed_count = 0
    for h in headers:
        candidates = flavor.candidate_targets(
            forge_host, forge_owner, forge_repo, h.subject,
        )
        if not candidates:
            continue
        candidate_count += 1
        body = read_body(h.path)
        if not body:
            logger.debug(
                "forge-root candidate has empty body path=%s subject=%r",
                h.path, h.subject,
            )
            continue
        matched: ForgejoTarget | None = None
        for target in candidates:
            if body_confirms_target(body, target):
                matched = target
                break
        if matched is None:
            logger.debug(
                "forge-root candidate body does not confirm any target "
                "path=%s subject=%r flavor=%s tried_urls=%s",
                h.path, h.subject, flavor.name,
                [t.html_url for t in candidates],
            )
            continue
        if h.message_id:
            idx.forge_root_target[h.message_id] = matched
            confirmed_count += 1
    logger.info(
        "thread index: flavor=%s msgids=%d forge-root candidates=%d confirmed=%d",
        flavor.name, len(idx.by_msgid), candidate_count, confirmed_count,
    )
    return idx


def classify_via_threading(
    h: MailHeaders, idx: ThreadIndex, *, max_depth: int = 50,
) -> ForgejoTarget | None:
    """Walk ``In-Reply-To`` upward looking for a forge-root ancestor.

    Cycle protection via a visited-set; depth cap as a safety net.
    """
    if h.message_id and h.message_id in idx.forge_root_target:
        return idx.forge_root_target[h.message_id]

    visited: set[str] = set()
    cur_msgid = h.in_reply_to
    depth = 0
    while cur_msgid and depth < max_depth:
        if cur_msgid in visited:
            logger.debug(
                "thread cycle path=%s msgid=%s",
                h.path, cur_msgid,
            )
            return None
        visited.add(cur_msgid)
        if cur_msgid in idx.forge_root_target:
            return idx.forge_root_target[cur_msgid]
        parent = idx.by_msgid.get(cur_msgid)
        if parent is None:
            return None
        cur_msgid = parent.in_reply_to
        depth += 1
    return None


# ---------------------------------------------------------------------------
# Decision-building (per mail)
# ---------------------------------------------------------------------------


def _is_forge_or_self(headers: MailHeaders, *, bot_re: re.Pattern[str]) -> bool:
    """True if the mail looks like a forge-bot or self-loop notification.

    We check ``X-MailFrom`` first (Mailman 3 sets this to the original
    envelope sender, untouched by the From: rewrite). If absent we
    fall back to scanning the From: addresses.
    """
    if headers.x_mailfrom and bot_re.search(headers.x_mailfrom):
        return True
    for _name, addr in parse_addresses(headers.from_raw):
        if addr and bot_re.search(addr):
            return True
    return False


def build_decision(
    headers: MailHeaders,
    idx: ThreadIndex,
    *,
    now_ts: float,
    max_age_seconds: float,
    max_mail_bytes: int,
    forge_bot_re: re.Pattern[str],
    extra_skip_re: re.Pattern[str] | None,
    forwarded_msgids: set[str],
    full_quote_action: str = "strip",
) -> MailDecision:
    """Decide what to do with a single mail.

    Returns a MailDecision with ``action == ACTIONABLE`` if mail-fairy
    should post it, or one of the ``skip:*`` actions with a reason.
    The composed comment body is filled in only for the actionable
    case.
    """
    if not headers.message_id:
        return MailDecision(
            headers=headers, action=SKIP_NO_MSGID,
            reason="missing Message-ID header",
        )

    if (now_ts - headers.file_ts) > max_age_seconds:
        return MailDecision(
            headers=headers, action=SKIP_TOO_OLD,
            reason=(
                f"file_ts={headers.file_ts:.0f} older than "
                f"{max_age_seconds:.0f}s"
            ),
        )

    if headers.file_size > max_mail_bytes:
        return MailDecision(
            headers=headers, action=SKIP_TOO_LARGE,
            reason=(
                f"size={headers.file_size} > limit={max_mail_bytes}"
            ),
        )

    if _is_forge_or_self(headers, bot_re=forge_bot_re):
        return MailDecision(
            headers=headers, action=SKIP_BOT_MAIL,
            reason="from forge bot or self (loop guard)",
        )
    if extra_skip_re is not None and (
        extra_skip_re.search(headers.from_raw)
        or extra_skip_re.search(headers.x_mailfrom)
    ):
        return MailDecision(
            headers=headers, action=SKIP_BOT_MAIL,
            reason="from --skip-from regex",
        )

    if not headers.in_reply_to:
        return MailDecision(
            headers=headers, action=SKIP_NO_PARENT,
            reason="no In-Reply-To header (thread root)",
        )

    target = classify_via_threading(headers, idx)
    if target is None:
        return MailDecision(
            headers=headers, action=SKIP_NOT_FORGE_THREAD,
            reason="ancestor chain does not reach a forge notification",
        )

    if headers.message_id in forwarded_msgids:
        return MailDecision(
            headers=headers, action=SKIP_DEDUP_LOCAL,
            reason="Message-ID present in local state file",
            target=target,
        )

    raw_body = read_body(headers.path)
    if not raw_body.strip():
        return MailDecision(
            headers=headers, action=SKIP_EMPTY_BODY,
            reason="empty text/plain body",
            target=target,
        )

    # If the body contains a previous list mail's footer in QUOTED
    # text, the sender included an entire prior mail unsnipped --
    # typical "huge quote with terse reply" pattern. Two responses
    # are configurable via ``full_quote_action``:
    #   "strip" (default): drop the offending quote block(s) and
    #     forward the rest. Cheap and lets a real reply through if
    #     one exists.
    #   "skip": do not post anything. Safer fallback if the strip
    #     turns out to remove genuine content for a particular
    #     list's quoting style.
    if _QUOTED_LIST_FOOTER_RE.search(raw_body):
        if full_quote_action == "skip":
            logger.warning(
                "full-quote-with-footer skip msgid=%s path=%s",
                headers.message_id, headers.path,
            )
            return MailDecision(
                headers=headers, action=SKIP_FULL_QUOTE_WITH_FOOTER,
                reason=(
                    "body quotes a previous list mail in full "
                    "(Mailman footer present in quoted text); "
                    "skipping per --full-quote-action=skip"
                ),
                target=target,
            )
        # strip mode: drop the offending block(s) and continue.
        stripped = strip_quoted_list_footer_blocks(raw_body)
        logger.warning(
            "full-quote-with-footer strip msgid=%s path=%s "
            "raw_bytes=%d stripped_bytes=%d",
            headers.message_id, headers.path,
            len(raw_body), len(stripped),
        )
        raw_body = stripped
        if not raw_body.strip():
            return MailDecision(
                headers=headers, action=SKIP_EMPTY_BODY,
                reason=(
                    "body became empty after stripping "
                    "full-quote-with-footer block"
                ),
                target=target,
            )

    body_no_footer = strip_mailman_footer(raw_body)
    body_clean = strip_bottom_quote(body_no_footer)
    if not body_clean.strip():
        return MailDecision(
            headers=headers, action=SKIP_EMPTY_BODY,
            reason="body empty after stripping footer/bottom-quote",
            target=target,
        )

    author_name, author_email = extract_author(headers)
    if headers.date is not None:
        date_utc = headers.date.astimezone(timezone.utc)
        date_iso = date_utc.strftime("%Y-%m-%d %H:%M UTC")
    else:
        date_iso = ""

    attribution = format_attribution_link(
        author_name=author_name,
        author_email=author_email,
        date_iso=date_iso,
        lore_url=headers.lore_url,
    )
    body = compose_comment_body(body_clean, attribution, headers.message_id)

    excerpt = body_clean.strip().splitlines()[0:2]
    return MailDecision(
        headers=headers, action=ACTIONABLE,
        reason="ready to forward",
        target=target,
        body=body,
        author_name=author_name,
        author_email=author_email,
        date_iso=date_iso,
        raw_body_excerpt=" / ".join(excerpt)[:200],
    )


# ---------------------------------------------------------------------------
# State file (forwarded Message-IDs)
# ---------------------------------------------------------------------------


def _empty_state() -> dict:
    return {"version": _STATE_VERSION, "forwarded": {}}


def load_state(path: Path) -> dict:
    """Read the mail-fairy state file, falling back to an empty state.

    We deliberately do NOT use ``common.load_pickle_cache`` here even
    though the on-disk shape is similar: that helper checks against
    ``common._CACHE_VERSION`` (the version of pr_auto_approve's
    ``repo_discussion_cache.pkl`` schema), which is unrelated to
    mail-fairy's state schema. Owning our own version constant means
    a future bump of either schema cannot silently reset the other.
    """
    try:
        with path.open("rb") as f:
            raw = pickle.load(f)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError, OSError):
        return _empty_state()
    if not isinstance(raw, dict) or raw.get("version") != _STATE_VERSION:
        return _empty_state()
    if not isinstance(raw.get("forwarded"), dict):
        raw["forwarded"] = {}
    return raw


def state_msgids(state: dict) -> set[str]:
    return set(state.get("forwarded", {}).keys())


def record_forwarded(
    state: dict, *, msgid: str, target_url: str, posted_at: float,
) -> None:
    state.setdefault("forwarded", {})[msgid] = {
        "target_url": target_url,
        "posted_at": posted_at,
    }


# ---------------------------------------------------------------------------
# Manual prompt (matches pr_auto_approve.py's [yes/skip/defer/quit/retry])
# ---------------------------------------------------------------------------


def prompt_manual(decision: MailDecision) -> str:
    while True:
        try:
            answer = input(
                f"Forward {decision.headers.message_id} -> "
                f"{decision.target.html_url if decision.target else '?'}? "
                f"[yes/skip/defer/quit] "
            ).strip().lower()
        except EOFError:
            return "quit"
        if answer in ("y", "yes"):
            return "apply"
        if answer in ("", "s", "skip"):
            return "skip"
        if answer in ("d", "defer"):
            return "defer"
        if answer in ("q", "quit"):
            return "quit"
        logger.warning("Please answer yes, skip, defer or quit.")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def _scan_phase(
    maildirs: list[Path], *, max_age_seconds: float, now_ts: float,
) -> list[MailHeaders]:
    """First pass: cheap header-only parse + age pre-filter by filename."""
    kept: list[MailHeaders] = []
    seen = 0
    aged_out_by_filename = 0
    parse_errors = 0
    for path in iter_maildir_files(maildirs):
        seen += 1
        # Cheap pre-filter using the filename's unix timestamp -- avoids
        # opening the file for the bulk of an aged-out maildir.
        ts = parse_maildir_filename_ts(path.name)
        if ts is not None and (now_ts - ts) > max_age_seconds:
            aged_out_by_filename += 1
            continue
        h = read_headers(path)
        if h is None:
            parse_errors += 1
            continue
        kept.append(h)
    logger.info(
        "scan: total=%d aged-out-by-filename=%d header-parse-errors=%d kept=%d",
        seen, aged_out_by_filename, parse_errors, len(kept),
    )
    return kept


def _build_summary(decisions: list[MailDecision]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in decisions:
        counts[d.action] = counts.get(d.action, 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    setup_logging(logger, bool(args.verbose), forge_gcli.logger)

    maildirs = [Path(p).expanduser().resolve() for p in args.maildir]
    for d in maildirs:
        if not d.is_dir():
            logger.error("maildir does not exist: %s", d)
            return 2

    forge_bot_re = re.compile(args.forge_bot_sender)
    extra_skip_re = (
        re.compile("|".join(f"(?:{p})" for p in args.skip_from))
        if args.skip_from
        else None
    )

    flavor_name = args.forge_flavor or args.forge_type
    try:
        flavor = select_forge_flavor(flavor_name)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    logger.info(
        "active forge flavor: name=%s (resolved from %s=%r)",
        flavor.name,
        "--forge-flavor" if args.forge_flavor else "--forge-type",
        flavor_name,
    )
    if flavor.untested:
        logger.warning(
            "forge flavor %r is UNTESTED in this codebase. Subject "
            "patterns and URL shapes were written from public docs, "
            "not from running mails. If a real notification fails to "
            "classify, the operator log will say which subject regex "
            "matched and which body URLs the bot tried (look for "
            "'forge-root candidate body does not confirm any target' "
            "at debug level). Please report or patch with the failing "
            "subject and a sample body URL.",
            flavor.name,
        )

    state_path = Path(args.state_file).expanduser()
    state = load_state(state_path)
    forwarded_msgids = state_msgids(state)
    logger.info(
        "state file path=%s already-forwarded=%d",
        state_path, len(forwarded_msgids),
    )

    now_ts = time.time()
    max_age_seconds = args.max_age_days * 86400.0

    headers_list = _scan_phase(
        maildirs, max_age_seconds=max_age_seconds, now_ts=now_ts,
    )
    idx = build_thread_index(
        headers_list,
        forge_host=args.forge_base_url,
        forge_owner=args.owner,
        forge_repo=args.repo,
        flavor=flavor,
    )

    decisions: list[MailDecision] = []
    for h in headers_list:
        d = build_decision(
            h, idx,
            now_ts=now_ts,
            max_age_seconds=max_age_seconds,
            max_mail_bytes=args.max_mail_bytes,
            forge_bot_re=forge_bot_re,
            extra_skip_re=extra_skip_re,
            forwarded_msgids=forwarded_msgids,
            full_quote_action=args.full_quote_action,
        )
        decisions.append(d)

    summary = _build_summary(decisions)
    logger.info(
        "decisions: %s",
        ", ".join(f"{k}={v}" for k, v in sorted(summary.items())),
    )

    actionable = [d for d in decisions if d.action == ACTIONABLE]
    logger.info("actionable mails: %d", len(actionable))

    # Group by target so we make at most one comments-fetch per PR.
    by_target: dict[str, list[MailDecision]] = {}
    for d in actionable:
        assert d.target is not None
        by_target.setdefault(d.target.html_url, []).append(d)

    posted = 0
    skipped_remote = 0
    quit_early = False
    for url, group in by_target.items():
        target = group[0].target
        assert target is not None
        try:
            existing = forge_gcli.list_issue_comments(
                args, target.owner, target.repo, target.number,
                kind=target.kind,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "skipping target=%s: cannot fetch existing comments: %s",
                url, exc,
            )
            continue
        existing_msgids = find_marker_msgids(existing)
        logger.info(
            "target=%s existing_marker_msgids=%d candidate_mails=%d",
            url, len(existing_msgids), len(group),
        )

        for d in group:
            if d.headers.message_id in existing_msgids:
                d.action = SKIP_DEDUP_REMOTE
                d.reason = "marker already present in existing comments"
                skipped_remote += 1
                logger.info(
                    "SKIP-REMOTE-DEDUP msgid=%s target=%s",
                    d.headers.message_id, url,
                )
                continue

            # Show the full body whenever the operator might act on it:
            # ``--manual`` because the prompt is meaningless without seeing
            # what gets posted, ``--dry-run`` because there is no real post
            # to inspect afterwards, and ``-vv`` as the explicit override.
            # Otherwise a 400-char head is enough for the post-mortem log
            # because the actual comment is visible on the forge.
            preview_full = args.dry_run or args.manual or args.verbose >= 2
            preview = d.body if preview_full else (
                d.body[:400] + ("..." if len(d.body) > 400 else "")
            )
            logger.info(
                "POST candidate target=%s msgid=%s author=%s date=%s",
                url, d.headers.message_id, d.author_name, d.date_iso,
            )
            logger.info("body preview:\n%s", preview)

            if args.manual:
                choice = prompt_manual(d)
                if choice == "quit":
                    quit_early = True
                    break
                if choice in ("skip", "defer"):
                    # ``defer`` and ``skip`` behave the same here for
                    # v1: there is no producer/consumer pipeline that
                    # could re-deliver this mail later in the run.
                    # Operator can re-run mail-fairy to revisit.
                    logger.info("manual %s msgid=%s", choice, d.headers.message_id)
                    continue

            if args.dry_run:
                logger.info(
                    "DRY-RUN would post msgid=%s target=%s",
                    d.headers.message_id, url,
                )
                continue

            try:
                forge_gcli.post_issue_comment(
                    args, target.owner, target.repo, target.number, d.body,
                    kind=target.kind,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "post failed msgid=%s target=%s err=%s",
                    d.headers.message_id, url, exc,
                )
                continue
            posted += 1
            record_forwarded(
                state,
                msgid=d.headers.message_id,
                target_url=url,
                posted_at=time.time(),
            )
            atomic_write_pickle(state_path, state)
            logger.info(
                "POSTED msgid=%s target=%s",
                d.headers.message_id, url,
            )

        if quit_early:
            break

    logger.info(
        "summary: posted=%d remote-dedup-skipped=%d quit_early=%s",
        posted, skipped_remote, quit_early,
    )
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mail_fairy",
        description=(
            "Forward mailing-list replies to Forgejo PRs/Issues as "
            "comments via gcli."
        ),
    )
    p.add_argument(
        "--maildir", action="append", required=True,
        help=(
            "Maildir root containing new/ and cur/ subfolders. May be "
            "specified multiple times to scan several lists."
        ),
    )
    forge_gcli.add_forge_repo_args(p)
    p.add_argument(
        "--forge-base-url", required=True,
        help=(
            "Forge base URL, e.g. https://forge.example.org or "
            "https://github.com or https://gitlab.com."
        ),
    )
    p.add_argument(
        "--forge-flavor", default=None,
        help=(
            "Override the mail-parsing flavor (forgejo/gitea/github/"
            "gitlab). When omitted, mail_fairy derives the flavor "
            "from --forge-type. Use this if you point gcli at a "
            "different backend than your mail notifications come "
            "from. NOTE: github and gitlab parsing/posting are "
            "UNTESTED in this codebase; mail_fairy will log a "
            "warning at startup so failures are easy to spot."
        ),
    )
    p.add_argument(
        "--max-age-days", type=int, default=DEFAULT_MAX_AGE_DAYS,
        help=(
            "Skip mails older than this. Pre-filtering uses the "
            "Maildir filename's unix timestamp (cheap) before opening "
            "any file. Default: %(default)s."
        ),
    )
    p.add_argument(
        "--max-mail-bytes", type=int, default=DEFAULT_MAX_MAIL_BYTES,
        help="Skip mails larger than this many bytes. Default: %(default)s.",
    )
    p.add_argument(
        "--state-file", default=str(DEFAULT_STATE_PATH),
        help=(
            "Pickle of forwarded Message-IDs. Used as the fast dedup "
            "layer; the comment-marker scan is the authoritative one. "
            "Default: %(default)s."
        ),
    )
    p.add_argument(
        "--forge-bot-sender", default=DEFAULT_FORGE_BOT_SENDER_RE,
        help=(
            "Regex matched against ``X-MailFrom`` and the From: address. "
            "Mails matching are NOT forwarded -- they are notifications "
            "Forgejo itself sent to the list. Default: %(default)r."
        ),
    )
    p.add_argument(
        "--skip-from", action="append", default=[],
        help=(
            "Additional From-regex to skip (loop-guard for our own "
            "bot account or other automation). Repeatable."
        ),
    )
    p.add_argument(
        "--full-quote-action", choices=("strip", "skip"), default="strip",
        help=(
            "How to handle mails whose body quotes a previous list "
            "mail in full (Mailman footer present in quoted text). "
            "'strip' (default) drops just the offending quote "
            "block(s) and forwards the rest. 'skip' skips the entire "
            "mail. Use 'skip' as a fallback if 'strip' turns out to "
            "drop content that should have been forwarded for your "
            "particular list's quoting style."
        ),
    )
    p.add_argument(
        "--manual", action="store_true",
        help="Prompt before each post.",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Do not post; just log what would happen.",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    return p


if __name__ == "__main__":
    sys.exit(main())
