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

Replace the values in a captured fixture with values from fixed pools.

The pools are generated from a fixed seed, so they are the same on every
run and a reader can enumerate them. Which pool member replaces a given
value is decided by the system CSPRNG and by nothing else: not the value,
not its length, not the file name. Two runs over the same untouched input
therefore differ, and no replacement can be predicted from what it
replaced.

A value already drawn from a pool, or named in tools/redact_exceptions.txt,
is left alone. That makes a redacted file a fixed point, and makes the
check exact: after redacting, every token in scope must be a pool member
or an exception, and anything else is a fatal internal failure.

    tools/redact.py tests/fixtures/issue_fairy/*.json
    tools/redact.py --check tests/fixtures/issue_fairy/*.json
    tools/redact.py --names $(find tests -type f)
    tools/redact.py --lines 12,40-52 tests/fixtures/mail_fairy/a.eml
"""
from __future__ import annotations

import argparse
import json
import random
import re
import secrets
import sys
from pathlib import Path

__all__ = ["Pools", "Redactor", "redact_json", "redact_text", "check_tokens",
           "check_names", "classify", "Kind", "FatalInternalFailure"]

POOL_SEED = 20260730
POOL_SIZE = 4096
EXCEPTIONS_FILE = Path(__file__).with_name("redact_exceptions.txt")

RESERVED_DOMAINS = ("example.com", "example.org", "example.net")
RESERVED_NETS = ("192.0.2.", "198.51.100.", "203.0.113.")

CONSONANT = "bcdfghjklmnprstvwz"
VOWEL = "aeiou"
COMMON = ("hi", "hello", "thanks", "lgtm", "ok", "ping", "nit", "oops",
          "yes", "no", "please", "sure", "done", "fixed", "agreed")

KEPT_FIELDS = frozenset({"sha", "merge_base", "ref", "label", "commit_ids",
                         "pronouns", "language"})

ISO_DATE = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:Z|[+-]\d\d:\d\d)\Z")
PHONE_ANY = re.compile(r"(?<![\w.])[+(]\d[\d ()./-]{5,17}\d(?![\w.])")
RFC2822_ANY = re.compile(
    r"[A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} "
    r"\d\d:\d\d:\d\d [+-]\d{4}(?: \([A-Z]{2,5}\))?")
RFC2822_DATE = re.compile(
    r"[A-Z][a-z]{2}, \d{1,2} [A-Z][a-z]{2} \d{4} "
    r"\d\d:\d\d:\d\d [+-]\d{4}(?: \([A-Z]{2,5}\))?\Z")
MSGID_RE = re.compile(r"[\w.+%-]+@[\w.-]+\Z")
DATE_ONLY = re.compile(r"\d{4}-\d\d-\d\d\Z")
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                     r"[0-9a-f]{4}-[0-9a-f]{12}\Z", re.I)
HEX_RE = re.compile(r"(?=[0-9a-f]*[a-f])[0-9a-f]{7,64}\Z")
BASE32_RE = re.compile(r"[A-Z2-7]{26,}\Z")
ADDRESS_PATTERN = r"[\w.+%-]+@[\w.-]+\.[a-z]{2,}"
EMAIL_RE = re.compile(ADDRESS_PATTERN + r"\Z", re.I)
URL_RE = re.compile(r"[a-z][a-z0-9+.-]*://[^\s<>()\[\]\"']+\Z", re.I)
SSH_RE = re.compile(r"[\w.-]+@[\w.-]+:[\w./-]+\Z")
IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}\Z")
FILE_RE = re.compile(r"[\w.+ -]+\.(?:c|h|py|json|txt|md|eml|zip|png|jpg|patch|diff|git|log|sh|xml|html)\Z", re.I)
DOMAIN_RE = re.compile(r"[a-z\d-]+(?:\.[a-z\d-]+)*\.[a-z]{2,}\Z", re.I)
PATH_RE = re.compile(r"/?(?:[\w.+-]+/)+[\w.+-]*\Z")
NUMBER_RE = re.compile(r"-?\d+\Z")
FLOAT_RE = re.compile(r"-?\d+\.\d+\Z")
PHONE_RE = re.compile(r"[+(]?\d[\d ()./-]{5,17}\d\Z")
COLOUR_RE = re.compile(r"[0-9a-fA-F]{6}\Z")
WORD_RE = re.compile(r"[^\W\d_]+")
WORDISH_RE = re.compile(r"[\w.'-]+")
TOKEN_RE = re.compile(r"[^\s]+")

NAME_KEY_ALT = (r"(?:\w+_)?(login|username|user_name|full_name|realname|"
                 r"display_name|nickname|author|committer|email)")
NAME_KEY_TEST_RE = re.compile(NAME_KEY_ALT + r"\Z", re.I)
NAME_KEY_RE = re.compile(
    r"[\"']?\b" + NAME_KEY_ALT + r"\b[\"']?\]?[ \t]*(?:==|!=|[:=])[ \t]*"
    r"([\"'])((?:(?!\2)[^\n]){1,120})\2", re.I)
NAME_ATTR_RE = re.compile(
    r"\.[ \t]*" + NAME_KEY_ALT + r"[ \t]*(?:==|!=|,)[ \t]*"
    r"([\"'])((?:(?!\2)[^\n]){1,120})\2")
NAME_REV_RE = re.compile(
    r"assert\w*\([ \t]*"
    r"([\"'])(?P<value>(?:(?!\1)[^\n]){1,120})\1[ \t]*,[ \t]*"
    r"[^,()\n]*(?:\.|\[[\"'])[ \t]*(?P<key>" + NAME_KEY_ALT
    + r")\b[\"']?\]?[ \t]*\)")
NAME_GET_RE = re.compile(
    r"\.get\([ \t]*[\"']" + NAME_KEY_ALT + r"[\"'][ \t]*"
    r"\)[ \t]*(?:==|!=)[ \t]*"
    r"([\"'])((?:(?!\2)[^\n]){1,120})\2")
NAME_NEAR_RES = (
    re.compile(r"[\"'](?:login|username|email)[\"'][^{}]{0,200}?"
               r"[\"']name[\"'][ \t]*:[ \t]*"
               r"([\"'])((?:(?!\1)[^\n]){1,120})\1", re.S),
    re.compile(r"[\"']name[\"'][ \t]*:[ \t]*"
               r"([\"'])((?:(?!\1)[^\n]){1,120})\1[^{}]{0,200}?"
               r"[\"'](?:login|username|email)[\"']", re.S),
)
ADDRESS_RE = re.compile(ADDRESS_PATTERN, re.I)
NAME_ADDR_RE = re.compile(
    r"(?:\"([^\"\n]{1,80})\""
    r"|((?:[\w.'-]{1,40},?[ \t]+){0,2}[\w.'-]{1,40}))"
    r"[ \t]*<[^<>\s]+@[^<>\s]+>")

CAST = frozenset(
    "alice bob carol dave erin eve mallory trent oscar peggy victor walter "
    "jane john doe roe dev bot".split())

CAPTURE_SUFFIXES = (".json", ".eml", ".txt")

class FatalInternalFailure(RuntimeError):
    """The check found a token the redaction should have replaced."""

class Kind:
    DATE = "date"
    DATE2822 = "date2822"
    DATEONLY = "dateonly"
    MSGID = "msgid"
    MAILTO = "mailto"
    UUID = "uuid"
    HEX = "hex"
    BASE32 = "base32"
    EMAIL = "email"
    URL = "url"
    SSH = "ssh"
    IPV4 = "ipv4"
    FILE = "file"
    DOMAIN = "domain"
    PATH = "path"
    NUMBER = "number"
    PHONE = "phone"
    NAME = "name"
    FLOAT = "float"
    COLOUR = "colour"
    WORD = "word"
    TEXT = "text"

def _is_phone(value: str) -> bool:
    if not PHONE_RE.match(value):
        return False
    digits = sum(c.isdigit() for c in value)
    groups = sum(not c.isdigit() for c in value)
    return 7 <= digits <= 15 and (value[0] in "+(" or groups >= 2)

def classify(value: str) -> str:
    """The syntax a replacement has to keep."""
    if ISO_DATE.match(value):
        return Kind.DATE
    if RFC2822_DATE.match(value):
        return Kind.DATE2822
    if DATE_ONLY.match(value):
        return Kind.DATEONLY
    if UUID_RE.match(value):
        return Kind.UUID
    if EMAIL_RE.match(value):
        return Kind.EMAIL
    if URL_RE.match(value):
        return Kind.URL
    if SSH_RE.match(value):
        return Kind.SSH
    if IPV4_RE.match(value):
        return Kind.IPV4
    if PATH_RE.match(value):
        return Kind.PATH
    if FILE_RE.match(value):
        return Kind.FILE
    if DOMAIN_RE.match(value):
        return Kind.DOMAIN
    if value.lower().startswith("mailto:"):
        return Kind.MAILTO
    if MSGID_RE.match(value):
        return Kind.MSGID
    if HEX_RE.match(value):
        return Kind.HEX
    if BASE32_RE.match(value):
        return Kind.BASE32
    if _is_phone(value):
        return Kind.PHONE
    if NUMBER_RE.match(value):
        return Kind.NUMBER
    if FLOAT_RE.match(value):
        return Kind.FLOAT
    if COLOUR_RE.match(value):
        return Kind.COLOUR
    if WORD_RE.fullmatch(value):
        return Kind.WORD
    return Kind.TEXT

class Pools:
    """Fixed seed, so every run offers the same members to choose from."""

    def __init__(self, seed: int = POOL_SEED, size: int = POOL_SIZE) -> None:
        rng = random.Random(seed)
        self.words = tuple(dict.fromkeys(
            list(COMMON) + [self._word(rng) for _ in range(size)]))
        self.numbers = tuple(dict.fromkeys(
            str(rng.randrange(10 ** 6, 10 ** 12)) for _ in range(size)))
        self.hex = {n: tuple(dict.fromkeys(
            self._digest(rng, n) for _ in range(size // 8)))
            for n in range(7, 65)}
        self.base32 = tuple(dict.fromkeys(
            "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567")
                    for _ in range(32)) for _ in range(size // 8)))
        self.uuids = tuple(dict.fromkeys(
            "%s-%s-%s-%s-%s" % tuple(
                "".join(rng.choice("0123456789abcdef") for _ in range(n))
                for n in (8, 4, 4, 4, 12)) for _ in range(size // 8)))
        self.domains = tuple(dict.fromkeys(
            f"{self._word(rng)}.{rng.choice(RESERVED_DOMAINS)}"
            for _ in range(size // 8))) + RESERVED_DOMAINS
        self.emails = tuple(dict.fromkeys(
            f"{self._word(rng)}@{rng.choice(RESERVED_DOMAINS)}"
            for _ in range(size)))
        self.urls = tuple(dict.fromkeys(
            "https://%s/%s/%s" % (rng.choice(RESERVED_DOMAINS),
                                  self._word(rng), self._word(rng))
            for _ in range(size)))
        self.ssh = tuple(dict.fromkeys(
            "git@%s:%s/%s.git" % (rng.choice(RESERVED_DOMAINS),
                                  self._word(rng), self._word(rng))
            for _ in range(size // 8)))
        self.ipv4 = tuple(dict.fromkeys(
            f"{rng.choice(RESERVED_NETS)}{rng.randrange(1, 255)}"
            for _ in range(size // 8)))
        self.paths = tuple(dict.fromkeys(
            "/".join(self._word(rng) for _ in range(rng.randrange(2, 5)))
            + rng.choice(("", ".c", ".h", ".py", ".json", ".txt"))
            for _ in range(size)))
        self.floats = tuple(dict.fromkeys(
            repr(float("%d.%d" % (rng.randrange(10 ** 6, 10 ** 9),
                                  rng.randrange(1, 10 ** 5))))
            for _ in range(size)))
        self.phones = tuple(dict.fromkeys(
            [f"+1 555 {100 + n:04d}" for n in range(100)]
            + [f"+44 7700 900{n:03d}" for n in range(1000)]))
        self.colours = tuple(dict.fromkeys(
            self._colour(rng) for _ in range(size // 8)))
        self.files = tuple(dict.fromkeys(
            self._word(rng) + rng.choice(
                (".c", ".h", ".py", ".json", ".txt", ".md", ".eml", ".zip",
                 ".png", ".patch", ".log"))
            for _ in range(size)))
        self.msgids = tuple(dict.fromkeys(
            "%s.%d@%s" % (self._word(rng), rng.randrange(10 ** 9),
                          self._digest(rng, 12)) for _ in range(size)))

    @staticmethod
    def _colour(rng: random.Random) -> str:
        while True:
            out = "".join(rng.choice("0123456789abcdef") for _ in range(6))
            if classify(out) == Kind.COLOUR:
                return out

    @staticmethod
    def _digest(rng: random.Random, length: int) -> str:
        """At least one a-f, so it cannot be read back as a plain number."""
        while True:
            out = "".join(rng.choice("0123456789abcdef")
                          for _ in range(length))
            if classify(out) == Kind.HEX:
                return out

    @staticmethod
    def _word(rng: random.Random) -> str:
        """Never all hex letters: such a word would classify as a digest
        and no longer be recognisable as a member of this pool."""
        while True:
            word = "".join(rng.choice(CONSONANT) + rng.choice(VOWEL)
                           for _ in range(rng.randrange(2, 5)))
            if classify(word) == Kind.WORD:
                return word

    def of(self, kind: str, sample: str = "") -> tuple[str, ...]:
        if kind == Kind.HEX:
            return self.hex.get(len(sample), ())
        return {Kind.WORD: self.words, Kind.NUMBER: self.numbers,
                Kind.BASE32: self.base32, Kind.UUID: self.uuids,
                Kind.EMAIL: self.emails, Kind.URL: self.urls,
                Kind.SSH: self.ssh, Kind.IPV4: self.ipv4,
                Kind.PATH: self.paths,
                Kind.FILE: self.files,
                Kind.DOMAIN: self.domains, Kind.MSGID: self.msgids,
                Kind.FLOAT: self.floats, Kind.PHONE: self.phones,
                Kind.COLOUR: self.colours}.get(kind, ())

    def holds(self, kind: str, value: str) -> bool:
        return value in self.of(kind, value)

    def misclassified(self) -> list[tuple[str, str]]:
        """Every member has to be recognisable as its own kind, or the
        check would reject what this tool itself produced."""
        kinds = (Kind.WORD, Kind.NUMBER, Kind.BASE32, Kind.UUID, Kind.EMAIL,
                 Kind.URL, Kind.SSH, Kind.IPV4, Kind.PATH,
                 Kind.FILE, Kind.DOMAIN, Kind.MSGID, Kind.FLOAT,
                 Kind.COLOUR, Kind.PHONE)
        wrong = [(kind, member) for kind in kinds
                 for member in self.of(kind) if classify(member) != kind]
        wrong += [(Kind.HEX, member) for members in self.hex.values()
                  for member in members if classify(member) != Kind.HEX]
        return wrong

def load_exceptions(path: Path = EXCEPTIONS_FILE) -> tuple[frozenset, tuple]:
    """Tokens shipped verbatim, and whole paths left alone. Explicit and
    enumerable on purpose: this is the only way something survives."""
    tokens, paths, whole = set(), [], set()
    if not path.exists():
        return frozenset(), (), frozenset()
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("path:"):
            paths.append(line[5:].strip())
        elif line.startswith("line:"):
            whole.add(line[5:])
        else:
            tokens.add(line)
    return frozenset(tokens), tuple(paths), frozenset(whole)

STRUCTURED = (Kind.URL, Kind.SSH, Kind.EMAIL, Kind.PATH, Kind.FILE,
              Kind.DOMAIN, Kind.MAILTO)

LITERAL, DRAWN, NAMED = 0, 1, 2

def _decompose(kind: str, value: str) -> list[tuple[int, str]]:
    """A structured value as literal and replaceable pieces. A number is a
    literal: inside a URL or a path it is an identifier, and a caller
    parses it back out; the part before an address's @ is NAMED, and
    the number rule does not reach it."""
    def segments(text: str) -> list[tuple[bool, str]]:
        out = []
        for index, piece in enumerate(text.split("/")):
            if index:
                out.append((False, "/"))
            route, dash, number = piece.rpartition("-")
            if route and number.isdigit():
                out.extend([(True, route), (False, dash + number)])
            else:
                out.append((bool(piece), piece))
        return out

    if kind == Kind.URL:
        scheme, sep, rest = value.partition("://")
        host, slash, rest = rest.partition("/")
        pieces = [(True, scheme), (False, sep), (True, host)]
        if slash:
            pieces.append((False, slash))
            for index, part in enumerate(re.split(r"([#?])", rest)):
                if index % 2:
                    pieces.append((False, part))
                else:
                    pieces.extend(segments(part))
        return pieces
    if kind == Kind.SSH:
        user, at, rest = value.partition("@")
        host, colon, path = rest.partition(":")
        return ([(NAMED, user), (LITERAL, at), (DRAWN, host),
                 (LITERAL, colon)] + segments(path))
    if kind == Kind.MAILTO:
        scheme, sep, rest = value.partition(":")
        address, mark, query = rest.partition("?")
        pieces = [(True, scheme), (False, sep)]
        pieces += _decompose(Kind.EMAIL, address)
        if mark:
            pieces += [(False, mark), (True, query)]
        return pieces
    if kind == Kind.EMAIL:
        local, at, domain = value.rpartition("@")
        pieces = []
        for index, part in enumerate(local.split(".")):
            if index:
                pieces.append((LITERAL, "."))
            pieces.append((NAMED, part))
        return pieces + [(LITERAL, at), (DRAWN, domain)]
    if kind == Kind.FILE:
        stem, dot, ext = value.rpartition(".")
        return [(True, stem), (False, dot + ext)]
    if kind == Kind.PATH:
        return segments(value)
    return [(True, value)]

class Redactor:
    """One run. A value seen twice gets the same replacement, so what the
    fixture says about itself stays true; the replacement itself comes
    from the CSPRNG and from nothing about the value."""

    def __init__(self, pools: Pools, exceptions: frozenset = frozenset(),
                 whole_lines: frozenset = frozenset()):
        self.pools = pools
        self.exceptions = exceptions
        self.whole_lines = whole_lines
        self.rng = secrets.SystemRandom()
        self.memo: dict[tuple[str, str], str] = {}
        self.runs = 0
        self.taken: dict[str, set] = {}

    KEPT = (Kind.DATE, Kind.DATE2822, Kind.DATEONLY, Kind.NUMBER,
            Kind.FLOAT)

    def keeps(self, kind: str, value: str, named: bool = False) -> bool:
        if not value or value in self.exceptions:
            return True
        if kind in self.KEPT and not named:
            return True
        nested = _as_json(value)
        if nested is not None:
            return all(self.keeps(classify(inner), inner)
                       for inner in _json_strings(nested, []))
        if kind in STRUCTURED:
            pieces = _decompose(kind, value)
            if len(pieces) == 1 and pieces[0][0] and pieces[0][1] == value:
                return self.pools.holds(kind, value)
            return all(self.keeps(classify(p), p, slot == NAMED)
                       for slot, p in pieces if slot)
        return self.pools.holds(kind, value)

    def rebuild(self, kind: str, value: str) -> str:
        """A structured value keeps its structure: only the parts that are
        not already a pool member or an exception move, each inside its own
        kind, so what a caller parses out of it still parses."""
        pieces = _decompose(kind, value)
        if len(pieces) == 1 and pieces[0] == (DRAWN, value):
            return self.draw(kind, value)
        out = []
        for slot, piece in pieces:
            kind_p = classify(piece)
            if not slot or self.keeps(kind_p, piece, slot == NAMED):
                self.spend(kind_p, piece)
                out.append(piece)
            elif kind_p == Kind.TEXT:
                out.append(self.draw(Kind.WORD, piece))
            elif slot == NAMED:
                if kind_p == Kind.PHONE or not self.pools.of(kind_p, piece):
                    kind_p = Kind.NUMBER if piece.isdigit() else Kind.WORD
                out.append(self.draw(kind_p, piece))
            else:
                out.append(self.token(piece))
        return "".join(out)

    def draw(self, kind: str, value: str) -> str:
        key = (kind, value)
        if key in self.memo:
            return self.memo[key]
        pool = self.pools.of(kind, value)
        if not pool:
            raise FatalInternalFailure(f"no pool for {kind} {value!r}")
        used = self.taken.setdefault(kind, set())
        free = [p for p in pool if p not in used]
        if not free:
            raise FatalInternalFailure(f"pool for {kind} exhausted")
        pick = self.rng.choice(free)
        used.add(pick)
        self.memo[key] = pick
        return pick

    def hex_like(self, value: str) -> str:
        """An index operand names an object: hex of its own length, even
        when the abbreviation happens to be all digits."""
        if self.pools.holds(Kind.HEX, value):
            self.spend(Kind.HEX, value)
            return value
        return self.draw(Kind.HEX, value)

    def spend(self, kind: str, value: str) -> None:
        """A member that survives verbatim is spent as surely as a drawn
        one, or a later input could be handed the same value."""
        if self.pools.holds(kind, value):
            self.taken.setdefault(kind, set()).add(value)

    def name_value(self, value: str) -> str:
        if classify(value) == Kind.NUMBER and value not in self.exceptions:
            if self.pools.holds(Kind.NUMBER, value):
                self.spend(Kind.NUMBER, value)
                return value
            return self.draw(Kind.NUMBER, value)
        return self.token(value)

    def token(self, value: str) -> str:
        if value[:1] == "-" and classify(value) in (Kind.NUMBER, Kind.FLOAT):
            return "-" + self.token(value[1:])
        nested = _as_json(value)
        if nested is not None:
            return json.dumps(
                _walk_kept(nested, self.token,
                           lambda v: (int(self.token(str(v)))
                                      if isinstance(v, int)
                                      else _float_token(self, v)),
                           KEPT_FIELDS, self.name_value),
                separators=(",", ":"))
        kind = classify(value)
        if self.keeps(kind, value):
            self.spend(kind, value)
            return value
        if kind == Kind.TEXT:
            return self.text(value)
        if kind in STRUCTURED:
            return self.rebuild(kind, value)
        return self.draw(kind, value)

    def field(self, value: str) -> str:
        call = PHONE_ANY.search(value)
        if call and _is_phone(call.group()):
            return (self.field(value[:call.start()])
                    + self.token(call.group())
                    + self.field(value[call.end():]))
        stamp = RFC2822_ANY.search(value)
        if stamp:
            return (self.field(value[:stamp.start()])
                    + (stamp.group() if self.keeps(Kind.DATE2822, stamp.group())
                       else self.draw(Kind.DATE2822, stamp.group()))
                    + self.field(value[stamp.end():]))
        return self._field(value)

    def _field(self, value: str) -> str:
        """Token by token with every space kept: a typed value is replaced
        inside its kind, and a run of plain words becomes at most three."""
        parts = re.split(r"(\s+)", value)
        out, index = [], 0
        while index < len(parts):
            piece = parts[index]
            if not piece or piece.isspace():
                out.append(piece)
                index += 1
                continue
            last, run = index, []
            probe = index
            while probe < len(parts):
                candidate = parts[probe]
                if not candidate or candidate.isspace():
                    probe += 1
                    continue
                prefix, core, suffix = _peel(candidate)
                if (classify(core) != Kind.WORD or prefix or suffix
                        or self.keeps(Kind.WORD, core)):
                    break
                run.append(core)
                last = probe
                probe += 1
            if run:
                out.append(self.text(" ".join(run)))
                index = last + 1
                continue
            prefix, core, suffix = _peel(piece)
            kind = classify(core)
            if self.keeps(kind, core):
                self.spend(kind, core)
                out.append(prefix + core + suffix)
            else:
                out.append(prefix + self.token(core) + suffix)
            index += 1
        return "".join(out)

    def name(self, value: str) -> str:
        """A display name and the address beside it: the name draws once
        so every header naming the same person agrees."""
        parts = DISPLAY_RE.match(value)
        if not parts:
            return self.field(value)
        lead, display, rest = parts.groups()
        via = re.search(r"\s+via\s+\S+\s*\Z", display)
        stem = display[:via.start()] if via else display
        if not stem.strip() or all(self.keeps(Kind.WORD, w)
                                   for w in WORD_RE.findall(stem)):
            return lead + display + self.field(rest)
        words = " ".join(sorted(WORD_RE.findall(stem.lower())))
        key = (Kind.NAME, words or stem)
        if key not in self.memo:
            self.memo[key] = self.text(stem)
        return (lead + self.memo[key] + (via.group() if via else "")
                + self.field(rest))

    def text(self, value: str) -> str:
        self.runs += 1
        """A run of prose becomes at most three pool words. The result is
        remembered, so the same run reads the same wherever it appears --
        how many words and which ones is still the CSPRNG's alone."""
        if all(self.keeps(classify(t), t) for t in TOKEN_RE.findall(value)):
            for token in TOKEN_RE.findall(value):
                self.spend(classify(token), token)
            return value
        return " ".join(self.draw(Kind.WORD, f"{value}\0{self.runs}\0{i}")
                        for i in range(self.rng.randrange(1, 4)))

def _float_token(red, value: float) -> float:
    text = repr(float(value))
    if classify(text) != Kind.FLOAT:
        raise FatalInternalFailure(f"no pool for the float {text}")
    return float(red.token(text))

def _as_json(value: str):
    """A field can carry a payload of its own; it is structure, not prose."""
    if value[:1] not in "{[":
        return None
    try:
        return json.loads(value)
    except ValueError:
        return None

def _walk_json(tree, visit, numbers=None):
    if isinstance(tree, dict):
        return {k: _walk_json(v, visit, numbers) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_walk_json(v, visit, numbers) for v in tree]
    if isinstance(tree, str):
        return visit(tree)
    if numbers is not None and isinstance(tree, (int, float)) \
            and not isinstance(tree, bool):
        return numbers(tree)
    return tree

def _json_strings(tree, out, with_numbers=False, nested=False, keys=False,
                  keys_only=False):
    """Every leaf the check has to see. It must reach exactly what the
    redaction reaches, or a value hides in whatever one of them skips."""
    if isinstance(tree, dict):
        for k, v in tree.items():
            if keys or keys_only:
                out.append(k)
            if k in KEPT_FIELDS and not isinstance(v, dict):
                continue
            _json_strings(v, out, with_numbers, nested, keys, keys_only)
    elif isinstance(tree, list):
        for v in tree:
            _json_strings(v, out, with_numbers, nested, keys, keys_only)
    elif keys_only:
        return out
    elif isinstance(tree, str):
        out.append(tree)
        if nested:
            inner = _as_json(tree)
            if inner is not None:
                _json_strings(inner, out, with_numbers, nested, keys,
                              keys_only)
    elif (with_numbers and isinstance(tree, (int, float))
          and not isinstance(tree, bool)):
        out.append(str(tree))
    return out

def _walk_kept(tree, visit, numbers, kept, name_visit=None):
    if isinstance(tree, dict):
        def one(k, v):
            if k in kept and not isinstance(v, dict):
                return v
            if name_visit and NAME_KEY_TEST_RE.match(k):
                if isinstance(v, str):
                    return name_visit(v)
                if isinstance(v, int) and not isinstance(v, bool):
                    return int(name_visit(str(v)))
            return _walk_kept(v, visit, numbers, kept, name_visit)

        return {k: one(k, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_walk_kept(v, visit, numbers, kept, name_visit)
                for v in tree]
    if isinstance(tree, str):
        return visit(tree)
    if isinstance(tree, (int, float)) and not isinstance(tree, bool):
        return numbers(tree)
    return tree

def redact_json(tree, red: Redactor):
    """A number in a payload names an item as surely as a string does, so
    it is drawn from the number pool and stays the kind of number it
    was."""
    def number(value):
        if isinstance(value, int):
            drawn = red.token(str(value))
            return int(drawn)
        text = repr(float(value))
        if classify(text) != Kind.FLOAT:
            raise FatalInternalFailure(f"no pool for the float {text}")
        return float(red.token(text))

    return _walk_kept(tree, red.token, number, KEPT_FIELDS,
                      red.name_value)

def redact_text(text: str, red: Redactor, lines: set | None = None) -> str:
    out = []
    for number, line in enumerate(text.split("\n"), 1):
        if lines is not None and number not in lines:
            out.append(line)
            continue
        out.append(_redact_line(line, red))
    return "\n".join(out)

QUOTE_RE = re.compile(r"[>\s]*")
HEADER_RE = re.compile(r"([A-Za-z][A-Za-z-]*:)(\s*)(.*)\Z", re.S)
HUNK_RE = re.compile(r"(@@ -)(\d+)(,\d+)?( \+)(\d+)(,\d+)?( @@)(.*)\Z", re.S)
INDEX_RE = re.compile(r"(index )([0-9a-f]+)(\.\.)([0-9a-f]+)(.*)\Z", re.S)
GIT_PATH_RE = re.compile(r"(diff --git |--- |\+\+\+ )(.*)\Z", re.S)
FORMAT_PATCH_RE = re.compile(
    r"(>?From )([0-9a-f]{7,40})( Mon Sep 17 00:00:00 2001)\Z")
DIFFSTAT_RE = re.compile(r"(\s*)(\S+)(\s+\|\s+\d+ [-+]*\s*)\Z")
SUMMARY_RE = re.compile(
    r"(\s*\d+ files? changed(?:, \d+ insertions?\(\+\))?"
    r"(?:, \d+ deletions?\(-\))?\s*)\Z")
SIDE_RE = re.compile(r"([ab]/)(\S+)")
NAME_HEADERS = ("From", "Cc", "To", "Reply-To", "Sender",
                "X-Original-From")
ADDR_SPLIT_RE = re.compile(r",(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)")
DISPLAY_RE = re.compile(r"(\s*)(.*?)(\s*<[^>]*>\s*)\Z", re.S)
PEEL = "<>()[]{}\"',;:!?#+-"

def _peel(token: str) -> tuple[str, str, str]:
    core = token.strip(PEEL)
    if not core:
        return "", token, ""
    start = token.index(core)
    return token[:start], core, token[start + len(core):]

def line_tokens(line: str, whole: frozenset = frozenset()) -> list[str]:
    """What both the redaction and the check see, so the two cannot drift:
    a quote prefix is structure, and a stamp is one token because its
    spelling only makes sense whole."""
    if line in whole:
        return []
    body = line[QUOTE_RE.match(line).end():]
    hunk = HUNK_RE.match(body)
    if hunk:
        return ([hunk.group(2), hunk.group(5)]
                + TOKEN_RE.findall(hunk.group(8)))
    index = INDEX_RE.match(body)
    if index:
        return []
    sides = GIT_PATH_RE.match(body)
    if sides:
        return [SIDE_RE.sub(r"\2", t) for t in TOKEN_RE.findall(
            sides.group(2)) if t != "/dev/null"]
    patch = FORMAT_PATCH_RE.match(body)
    if patch:
        return []
    stat = DIFFSTAT_RE.match(body)
    if stat:
        return [stat.group(2)]
    if SUMMARY_RE.match(body):
        return []
    header = HEADER_RE.match(body)
    if header and header.group(1).rstrip(":") == "Date":
        return [header.group(3)] if header.group(3) else []
    calls = [c for c in PHONE_ANY.findall(body) if _is_phone(c)]
    body = PHONE_ANY.sub(lambda m: " " if _is_phone(m.group()) else m.group(),
                         body)
    stamps = RFC2822_ANY.findall(body)
    return calls + stamps + TOKEN_RE.findall(RFC2822_ANY.sub(" ", body))

def _redact_line(line: str, red: Redactor) -> str:
    """A header keeps its name and its punctuation; only the values in it
    move, and a run of plain words in it collapses."""
    lead = line[:QUOTE_RE.match(line).end()]
    body = line[len(lead):]
    if not body:
        return line
    hunk = HUNK_RE.match(body)
    if hunk:
        return (lead + hunk.group(1) + red.token(hunk.group(2))
                + (hunk.group(3) or "") + hunk.group(4)
                + red.token(hunk.group(5)) + (hunk.group(6) or "")
                + hunk.group(7) + red.field(hunk.group(8)))
    index = INDEX_RE.match(body)
    if index:
        return line
    if line in red.whole_lines:
        return line
    sides = GIT_PATH_RE.match(body)
    if sides:
        return lead + sides.group(1) + SIDE_RE.sub(
            lambda m: m.group(1) + red.token(m.group(2)), sides.group(2))
    patch = FORMAT_PATCH_RE.match(body)
    if patch:
        return line
    stat = DIFFSTAT_RE.match(body)
    if stat:
        return (lead + stat.group(1) + red.token(stat.group(2))
                + stat.group(3))
    if SUMMARY_RE.match(body):
        return line
    header = HEADER_RE.match(body)
    if header and header.group(1).rstrip(":") in NAME_HEADERS:
        return lead + header.group(1) + header.group(2) + ",".join(
            red.name(part) for part in ADDR_SPLIT_RE.split(header.group(3)))
    if header and header.group(1).rstrip(":") == "Date":
        value = header.group(3)
        return (lead + header.group(1) + header.group(2)
                + (value if red.keeps(classify(value), value)
                   else red.token(value)))
    if header and header.group(1).rstrip(":") in red.exceptions:
        return lead + header.group(1) + header.group(2) + red.field(
            header.group(3))
    return lead + red.field(body)

def check_tokens(text: str, red: Redactor, is_json: bool) -> list[str]:
    """Every token in scope must be a pool member or an exception."""
    problems: list[str] = []
    if is_json:
        values = []
        for value in _json_strings(json.loads(text), [], True):
            nested = _as_json(value)
            values.extend(_json_strings(nested, [], True)
                          if nested is not None else [value])
        problems.extend(
            f"address in a key: {k}" for k in _json_strings(
                json.loads(text), [], keys_only=True)
            if EMAIL_RE.search(k) or URL_RE.search(k) or IPV4_RE.search(k))
        tokens = [t for v in values for t in TOKEN_RE.findall(v)] + [
            v for v in values if not TOKEN_RE.findall(v)]
    else:
        tokens = [t for line in text.split("\n")
                  for t in line_tokens(line, red.whole_lines)]
    cores = [t if red.keeps(classify(t), t) else _peel(t)[1] for t in tokens]
    return problems + sorted(
        {c for c in cores if not red.keeps(classify(c), c)})

def _dump_like(original: str, tree) -> str:
    for kw in ({"indent": 1}, {"indent": 2, "sort_keys": True},
               {"separators": (",", ":")}):
        for post in (lambda s: s, lambda s: s + "\n",
                     lambda s: _go_escape(s) + "\n"):
            if post(json.dumps(json.loads(original), **kw)) == original:
                return post(json.dumps(tree, **kw))
    raise FatalInternalFailure("unrecognised JSON serialisation")

def _go_escape(text: str) -> str:
    for ch in "<>&":
        text = text.replace(ch, "\\u%04x" % ord(ch))
    return text

def parse_lines(spec: str) -> set:
    numbers = set()
    for part in spec.split(","):
        if "-" in part:
            first, last = part.split("-")
            numbers.update(range(int(first), int(last) + 1))
        elif part.strip():
            numbers.add(int(part))
    return numbers

def check_names(text: str, red: Redactor) -> list[str]:
    problems: list[str] = []
    ok = CAST | {e.lower() for e in red.exceptions}
    dates = (Kind.DATE, Kind.DATE2822, Kind.DATEONLY)

    def address_ok(address: str) -> bool:
        local, _, domain = address.rpartition("@")
        if (all(p.lower() in ok for p in local.split("."))
                and red.keeps(classify(domain), domain)):
            return True
        return red.keeps(Kind.EMAIL, address)

    def undrawn(value: str, floor: int = 1) -> list[str]:
        if classify(value.strip()) in dates:
            return []
        out = set()
        for token in TOKEN_RE.findall(value):
            core = _peel(token)[1]
            kind = classify(core)
            if kind in dates:
                continue
            if "@" in core:
                local, _, host = core.rpartition("@")
                if "." not in host:
                    if len(local) <= 2 or all(
                            p.lower() in ok for p in local.split(".")):
                        continue
                    out.add(core)
                elif not address_ok(core):
                    out.add(core)
                continue
            if (len(core) > floor and WORDISH_RE.fullmatch(core)
                    and core.lower() not in ok
                    and token.lower() not in ok
                    and not red.keeps(kind, core, True)):
                out.add(core)
        return sorted(out)

    for address in ADDRESS_RE.findall(text):
        if not address_ok(address):
            problems.append(f"address: {address}")
    for quoted, bare in NAME_ADDR_RE.findall(text):
        left = undrawn(quoted or bare, floor=2)
        if left:
            problems.append(f"name beside an address: {' '.join(left)}")
    for regex in (NAME_KEY_RE, NAME_ATTR_RE, NAME_GET_RE):
        for key, _, value in regex.findall(text):
            left = undrawn(value)
            if left:
                problems.append(f"{key.lower()}: {' '.join(left)}")
    for m in NAME_REV_RE.finditer(text):
        left = undrawn(m.group("value"))
        if left:
            problems.append(f"{m.group('key').lower()}: {' '.join(left)}")
    for regex in NAME_NEAR_RES:
        for _, value in regex.findall(text):
            left = undrawn(value)
            if left:
                problems.append(f"name: {' '.join(left)}")

    def walk(node) -> None:
        if isinstance(node, dict):
            sibling = any(k.lower() in ("email", "login", "username",
                                        "user_name") for k in node)
            for key, value in node.items():
                if isinstance(value, str):
                    nested = _as_json(value)
                    if nested is not None:
                        walk(nested)
                    elif (NAME_KEY_TEST_RE.match(key)
                          or (sibling and key.lower() == "name")):
                        left = undrawn(value)
                        if left:
                            problems.append(
                                f"{key.lower()}: {' '.join(left)}")
                elif (isinstance(value, int) and not isinstance(value, bool)
                        and NAME_KEY_TEST_RE.match(key)):
                    left = undrawn(str(value))
                    if left:
                        problems.append(f"{key.lower()}: {value}")
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    tree = _as_json(text.lstrip())
    if tree is not None:
        walk(tree)
    return sorted(set(problems))

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="+", type=Path)
    ap.add_argument("--check", action="store_true",
                    help="report tokens that are not pool members")
    ap.add_argument("--names", action="store_true",
                    help="report names, addresses and name-carrying fields "
                         "that are not drawn, listed or conventional")
    ap.add_argument("--lines", help="only these 1-based lines (a,b-c)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    pools = Pools()
    tokens, exempt_paths, whole_lines = load_exceptions()
    red = Redactor(pools, tokens, whole_lines)
    lines = parse_lines(args.lines) if args.lines else None
    given = [p for p in args.paths if p.is_file()]
    files = [p for p in given
             if not any(str(p).endswith(e) or e in str(p)
                        for e in exempt_paths)]

    if args.names:
        bad = 0
        for path in given:
            data = path.read_bytes()
            try:
                text = data.decode()
            except UnicodeDecodeError:
                text = data.decode("latin-1")
            left = check_names(text, red)
            for item in left[:20]:
                print(f"{path}: {item!r}", file=sys.stderr)
            bad += len(left)
        print(f"checked {len(given)} files, {bad} undrawn name(s)")
        if not args.check:
            return 1 if bad else 0
        names_bad = bad

    alien = [p for p in files if p.suffix not in CAPTURE_SUFFIXES]
    if alien:
        for path in alien:
            print(f"{path}: not a capture ({' '.join(CAPTURE_SUFFIXES)}); "
                  "only --names reads other files", file=sys.stderr)
        return 3

    if args.check:
        bad = 0
        for path in files:
            try:
                text = path.read_text()
            except UnicodeDecodeError:
                print(f"{path}: 'not decodable as UTF-8'", file=sys.stderr)
                bad += 1
                continue
            left = check_tokens(text, red, path.suffix == ".json")
            for token in left[:20]:
                print(f"{path}: {token!r}", file=sys.stderr)
            bad += len(left)
        print(f"checked {len(files)} files, {bad} tokens not from a pool")
        return 1 if bad or (args.names and names_bad) else 0

    produced: list[tuple[Path, str, str, str]] = []
    for path in files:
        data = path.read_bytes()
        try:
            text, codec = data.decode(), "utf-8"
        except UnicodeDecodeError:
            text, codec = data.decode("latin-1"), "latin-1"
        if path.suffix == ".json":
            out = _dump_like(text, redact_json(json.loads(text), red))
            if lines is not None:
                before, after = text.split("\n"), out.split("\n")
                out = "\n".join(
                    after[i] if i + 1 in lines and i < len(after) else old
                    for i, old in enumerate(before))
        else:
            out = redact_text(text, red, lines)
        left = check_tokens(out, red, path.suffix == ".json")
        if left:
            raise FatalInternalFailure(
                f"{path}: still not from a pool after redacting: {left[:10]}")
        produced.append((path, text, out, codec))
    for path, text, out, codec in produced:
        path.write_bytes(out.encode(codec))
        if args.verbose:
            print(f"{path}: {len(text)} -> {len(out)} bytes")
        for item in check_names(out, red):
            print(f"{path}: left for review: {item!r}", file=sys.stderr)
    print(f"redacted {len(files)} files")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except FatalInternalFailure as failure:
        print(f"FATAL INTERNAL FAILURE: {failure}", file=sys.stderr)
        sys.exit(2)
