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

Tests for ``gcli_cache``.

Coverage map (one test per pinned property):

* schema constants pinned (version, EDIT_PRONE, FETCH membership)
* persistence: missing / corrupt / wrong-version cold-warm, round-trip
* freshness: updated_at mismatch / immutable ignores TTL / edit-prone
  within TTL fresh / edit-prone past TTL stale / absent stale / None
  stale
* ``get`` end-to-end: hit serves no fetch, miss fetches only missing
  immutable, edit-prone miss expands to trio, siblings carried across
  same updated_at, siblings dropped across advance, TTL expiry
  refetches trio only, unknown field rejected on both kinds
* stable-fetch loop: settles in one attempt, retries on advance,
  exhausts attempts, reviews fetched before review_comments
* bot scratch round-trips through pickle
"""

from __future__ import annotations

import pickle
import sys
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import forge_gcli  # noqa: E402
import gcli_cache  # noqa: E402
from gcli_cache import (  # noqa: E402
    Cache,
    EDIT_PRONE,
    Entry,
    EntryKey,
    FETCH,
    MAX_REFETCH_ATTEMPTS,
    SCHEMA_VERSION,
    _list_pr_files_or_empty_on_500,
    entry_key,
    get,
    load_cache,
    save_cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts).astimezone(timezone.utc)


PR_FIELDS = frozenset(f for (k, f) in FETCH if k == "pulls")
ISSUE_FIELDS = frozenset(f for (k, f) in FETCH if k == "issues")
NOW = utc("2026-05-26T00:00:00Z")
LIVE = utc("2026-05-21T10:00:00Z")
MAX_AGE = timedelta(hours=24)


class CountingFetcher:
    """Mock for one field's fetcher; returns scripted payloads, records calls."""

    def __init__(self, payloads: list[list[dict]] | None = None):
        self.payloads = payloads or [[{"id": 1}]]
        self.calls: list[tuple] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.payloads[min(len(self.calls) - 1, len(self.payloads) - 1)]


def make_refetch(stream: list[datetime]):
    """Build a ``refetch`` mock that yields ``stream`` then repeats last."""
    state = {"i": 0}
    calls: list[int] = []

    def reader(args, kind, owner, repo, n):
        calls.append(state["i"])
        i = min(state["i"], len(stream) - 1)
        state["i"] += 1
        return stream[i]
    reader.calls = calls  # type: ignore[attr-defined]
    return reader


def patch_fetch(testcase, **overrides):
    """Patch FETCH entries via ``unittest.mock`` so ``get`` uses mocks.

    Returns a dict of the installed CountingFetcher mocks for the keys
    provided in ``overrides`` (each value being either a CountingFetcher
    or a list-of-payloads shortcut).
    """
    from unittest.mock import patch
    installed: dict[tuple, CountingFetcher] = {}
    new_table = dict(FETCH)
    for key, val in overrides.items():
        kind, name = key.split(".", 1)
        f = val if isinstance(val, CountingFetcher) else CountingFetcher(val)
        installed[(kind, name)] = f
        new_table[(kind, name)] = f
    # Also stub the rest with no-op fetchers so accidental fetches are loud.
    for key in list(new_table):
        if key not in installed:
            new_table[key] = CountingFetcher()  # records calls
            installed[key] = new_table[key]
    p = patch.dict(gcli_cache.FETCH, new_table, clear=True)
    p.start()
    testcase.addCleanup(p.stop)
    return installed


ARGS = SimpleNamespace(forge_type="gitea", gcli_account="")


def call_get(cache, kind, fetchers, refetch, names, *,
             n=1, owner="o", repo="r", live=LIVE, now=NOW, max_age=MAX_AGE,
             args=ARGS):
    return get(
        cache, args, kind, owner, repo, n, live,
        *names, max_age=max_age, now=now, refetch=refetch,
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class FetchKindBindingTests(unittest.TestCase):
    """``FETCH`` binds the ``kind`` arg of ``list_issue_comments``.

    Forgejo/Gitea/GitHub serve PR and Issue comments on the same
    ``/issues/{n}/comments`` endpoint; GitLab uses two different
    paths. Either way, the kind shows up in the debug log line --
    fetching an issue's comments with ``kind=pr`` is wrong on both
    counts (silent path bug on GitLab, lying log line everywhere
    else).
    """

    def test_pulls_kind_binds_to_pr(self) -> None:
        fn = FETCH[("pulls", "issue_comments")]
        self.assertIs(fn.func, forge_gcli.list_issue_comments)
        self.assertEqual(fn.keywords, {"kind": forge_gcli.KIND_PR})

    def test_issues_kind_binds_to_issue(self) -> None:
        fn = FETCH[("issues", "issue_comments")]
        self.assertIs(fn.func, forge_gcli.list_issue_comments)
        self.assertEqual(fn.keywords, {"kind": forge_gcli.KIND_ISSUE})


class FilesEndpoint500WorkaroundTests(unittest.TestCase):
    """``_list_pr_files_or_empty_on_500`` masks the deleted-fork 500.

    See the function's docstring for the upstream bug. The wrapper is
    wired in via ``FETCH[("pulls", "files")]``; both pieces are pinned
    so removing one without the other can't slip through review.
    """

    def test_wrapper_is_wired_into_fetch_table(self) -> None:
        self.assertIs(FETCH[("pulls", "files")], _list_pr_files_or_empty_on_500)

    def test_500_is_silenced_to_empty_tuple(self) -> None:
        def raising(args, owner, repo, n):
            raise RuntimeError(
                "gcli api failed for '/repos/o/r/pulls/1/files' with exit "
                "code 1:\ngcli: error: failed to fetch data: request to "
                "https://example/.../files failed with code 500: API error"
            )
        with unittest.mock.patch.object(forge_gcli, "list_pr_files", raising):
            result = _list_pr_files_or_empty_on_500(SimpleNamespace(), "o", "r", 1)
        self.assertEqual(result, ())

    def test_non_500_errors_propagate(self) -> None:
        # Anything that isn't the deleted-fork wart should still surface
        # so the caller's atomic-or-raise contract triggers and the user
        # sees a warning instead of a silently empty file list.
        for code in (404, 403, 502, 503):
            def raising(args, owner, repo, n, code=code):
                raise RuntimeError(
                    f"gcli api failed: request ... failed with code {code}: ..."
                )
            with unittest.mock.patch.object(forge_gcli, "list_pr_files", raising):
                with self.assertRaises(RuntimeError):
                    _list_pr_files_or_empty_on_500(
                        SimpleNamespace(), "o", "r", 1,
                    )


class ConstantsTests(unittest.TestCase):

    def test_schema_version_pinned(self) -> None:
        # Pinned so a bump is deliberate: mismatched version on disk
        # -> empty cache -> stable-fetch loop refills.
        self.assertEqual(SCHEMA_VERSION, 4)

    def test_edit_prone_is_the_trio(self) -> None:
        # Adding here forces TTL refetches on data that doesn't need
        # them; removing resurrects the edit/delete blindspot for
        # that field.
        self.assertEqual(
            EDIT_PRONE,
            frozenset({"issue_comments", "reviews", "review_comments"}),
        )

    def test_issue_fields_subset_of_pr_fields(self) -> None:
        # Issues share a subset of PR fields (timeline + comments);
        # FETCH membership encodes this without separate sets.
        self.assertTrue(ISSUE_FIELDS.issubset(PR_FIELDS))


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class PersistenceTests(unittest.TestCase):

    def setUp(self) -> None:
        self.tmp = REPO_ROOT / "tests" / "_tmp_gcli_cache.pkl"
        if self.tmp.exists():
            self.tmp.unlink()

    def tearDown(self) -> None:
        if self.tmp.exists():
            self.tmp.unlink()

    def test_missing_file_yields_empty_cache(self) -> None:
        self.assertEqual(load_cache(self.tmp), Cache())

    def test_corrupt_pickle_yields_empty_cache(self) -> None:
        self.tmp.write_bytes(b"\xff\xff not a pickle \xff\xff")
        self.assertEqual(load_cache(self.tmp), Cache())

    def test_wrong_shape_yields_empty_cache(self) -> None:
        with self.tmp.open("wb") as f:
            pickle.dump({"some": "dict"}, f)
        self.assertEqual(load_cache(self.tmp), Cache())

    def test_wrong_version_yields_empty_cache(self) -> None:
        save_cache(self.tmp, Cache(version=SCHEMA_VERSION + 1))
        self.assertEqual(load_cache(self.tmp), Cache())

    def test_roundtrip_preserves_entries(self) -> None:
        cache = Cache()
        cache.entries[EntryKey("gitea", "", "pulls", "ffmpeg", "FFmpeg", 42)] = Entry(
            updated_at=LIVE,
            fetched_at=LIVE,
            comments_fetched_at=LIVE,
            fields={"timeline": ({"x": 1},), "commits": ({"sha": "abc"},)},
        )
        cache.entries[EntryKey("gitea", "", "issues", "ffmpeg", "FFmpeg", 7)] = Entry(
            updated_at=utc("2026-05-20T00:00:00Z"),
            fetched_at=utc("2026-05-20T00:00:01Z"),
            comments_fetched_at=None,
            fields={"timeline": ({"k": "v"},)},
        )
        save_cache(self.tmp, cache)
        loaded = load_cache(self.tmp)

        self.assertEqual(
            loaded.entries[EntryKey("gitea", "", "pulls", "ffmpeg", "FFmpeg", 42)].fields["commits"],
            ({"sha": "abc"},),
        )
        self.assertIsNone(
            loaded.entries[EntryKey("gitea", "", "issues", "ffmpeg", "FFmpeg", 7)].comments_fetched_at,
        )


# ---------------------------------------------------------------------------
# Freshness (exercised via ``get`` rather than touching internals)
# ---------------------------------------------------------------------------


def _seed_entry(cache: Cache, fields: dict, *,
                updated_at: datetime = LIVE,
                comments_fetched_at: datetime | None = LIVE,
                kind: str = "pulls", n: int = 1) -> None:
    cache.entries[entry_key(ARGS, kind, "o", "r", n)] = Entry(
        updated_at=updated_at,
        fetched_at=updated_at,
        comments_fetched_at=comments_fetched_at,
        fields=fields,
    )


class ForgeIdentityKeyTests(unittest.TestCase):
    """A cache slot belongs to one forge endpoint, not to a bare name.

    gcli's ``-t``/``-a`` decide which instance ``owner/repo`` resolves
    on, so the same pair can name two unrelated repositories. Serving
    one forge's payload for the other would hand fairy a foreign PR's
    comments and reviews.
    """

    def test_other_forge_type_is_a_miss(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"timeline": ({"t": 1},)})
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("timeline",),
                 args=SimpleNamespace(forge_type="github", gcli_account=""))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 1)

    def test_other_account_is_a_miss(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"timeline": ({"t": 1},)})
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("timeline",),
                 args=SimpleNamespace(forge_type="gitea", gcli_account="other"))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 1)

    def test_forge_type_case_does_not_split_the_slot(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"timeline": ({"t": 1},)})
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("timeline",),
                 args=SimpleNamespace(forge_type="GITEA", gcli_account=""))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 0)


class GetEndToEndTests(unittest.TestCase):

    def test_initial_miss_fetches_only_requested_immutable(self) -> None:
        cache = Cache()
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("timeline",))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 1)
        for k in PR_FIELDS - {"timeline"}:
            self.assertEqual(len(fetchers[("pulls", k)].calls), 0,
                             f"{k} fetched unexpectedly")

    def test_edit_prone_request_expands_to_trio(self) -> None:
        # Asking for ``reviews`` alone fetches the whole trio so they
        # share a single comments_fetched_at stamp.
        cache = Cache()
        fetchers = patch_fetch(self)
        result = call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("reviews",))
        self.assertEqual(set(result), {"reviews"})
        for k in EDIT_PRONE:
            self.assertEqual(len(fetchers[("pulls", k)].calls), 1,
                             f"{k} should be fetched")
        for k in PR_FIELDS - EDIT_PRONE:
            self.assertEqual(len(fetchers[("pulls", k)].calls), 0,
                             f"{k} fetched unexpectedly")

    def test_cache_hit_calls_no_fetcher(self) -> None:
        # ``comments_fetched_at`` must be within ``MAX_AGE`` of NOW for
        # the edit-prone trio to count as fresh.
        cache = Cache()
        _seed_entry(cache, {
            "timeline": ({"t": 1},),
            "issue_comments": ({"c": 1},),
            "reviews": ({"r": 1},),
            "review_comments": (),
        }, comments_fetched_at=NOW - timedelta(hours=1))
        fetchers = patch_fetch(self)
        result = call_get(cache, "pulls", fetchers, make_refetch([LIVE]),
                          ("timeline", "issue_comments", "reviews"))
        self.assertEqual(set(result), {"timeline", "issue_comments", "reviews"})
        for key, f in fetchers.items():
            self.assertEqual(len(f.calls), 0, f"{key} hit a fetcher")

    def test_updated_at_advance_drops_sibling_fields(self) -> None:
        new = utc("2026-05-25T21:46:48Z")
        cache = Cache()
        _seed_entry(cache, {
            "timeline": ({"old": True},),
            "commits": ({"sha": "old"},),
        }, updated_at=LIVE)
        fetchers = patch_fetch(self, **{"pulls.timeline": [[{"new": True}]]})
        call_get(cache, "pulls", fetchers, make_refetch([new]), ("timeline",), live=new)
        entry = cache.entries[entry_key(ARGS, "pulls", "o", "r", 1)]
        self.assertEqual(entry.updated_at, new)
        self.assertEqual(entry.fields["timeline"], ({"new": True},))
        self.assertNotIn("commits", entry.fields,
                         "stale sibling under old updated_at must be dropped")

    def test_updated_at_unchanged_preserves_sibling_fields(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"commits": ({"sha": "kept"},)},
                    comments_fetched_at=None)
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("timeline",))
        entry = cache.entries[entry_key(ARGS, "pulls", "o", "r", 1)]
        self.assertEqual(entry.fields["commits"], ({"sha": "kept"},))
        self.assertIn("timeline", entry.fields)
        self.assertEqual(len(fetchers[("pulls", "commits")].calls), 0)

    def test_ttl_expiry_refetches_trio_only(self) -> None:
        stale = NOW - timedelta(hours=48)
        cache = Cache()
        _seed_entry(cache, {
            "timeline": ({"x": 1},),
            "issue_comments": ({"id": 1, "body": "old"},),
            "reviews": (),
            "review_comments": (),
            "commits": ({"sha": "abc"},),
        }, comments_fetched_at=stale)
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("issue_comments",))
        for k in EDIT_PRONE:
            self.assertEqual(len(fetchers[("pulls", k)].calls), 1,
                             f"{k} should be refetched")
        for k in PR_FIELDS - EDIT_PRONE:
            self.assertEqual(len(fetchers[("pulls", k)].calls), 0,
                             f"{k} refetched unexpectedly")
        entry = cache.entries[entry_key(ARGS, "pulls", "o", "r", 1)]
        self.assertEqual(entry.fields["commits"], ({"sha": "abc"},))
        self.assertEqual(entry.fields["timeline"], ({"x": 1},))
        self.assertEqual(entry.comments_fetched_at, NOW)

    def test_updated_at_mismatch_is_a_miss(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"timeline": ({"x": 1},)},
                    updated_at=utc("2026-05-21T10:00:00Z"))
        fetchers = patch_fetch(self)
        call_get(cache, "pulls", fetchers, make_refetch([utc("2026-05-22T00:00:00Z")]),
                 ("timeline",), live=utc("2026-05-22T00:00:00Z"))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 1)

    def test_immutable_field_ignores_ttl(self) -> None:
        cache = Cache()
        _seed_entry(cache, {"timeline": ({"x": 1},)},
                    comments_fetched_at=None)
        fetchers = patch_fetch(self)
        result = call_get(cache, "pulls", fetchers, make_refetch([LIVE]),
                          ("timeline",), now=NOW + timedelta(days=100))
        self.assertEqual(result["timeline"], ({"x": 1},))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 0)

    def test_unknown_field_raises(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            call_get(Cache(), "pulls", patch_fetch(self), make_refetch([LIVE]),
                     ("no_such_field",))
        self.assertIn("no_such_field", str(ctx.exception))

    def test_pr_only_field_rejected_on_issue_path(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            call_get(Cache(), "issues", patch_fetch(self), make_refetch([LIVE]),
                     ("reviews",), n=7)
        self.assertIn("reviews", str(ctx.exception))


class GetIssuePathTests(unittest.TestCase):

    def test_issue_fetch_then_hit(self) -> None:
        cache = Cache()
        fetchers = patch_fetch(self)
        live = utc("2026-05-20T00:00:00Z")
        call_get(cache, "issues", fetchers, make_refetch([live]),
                 ("timeline",), n=7, live=live)
        self.assertEqual(len(fetchers[("issues", "timeline")].calls), 1)
        call_get(cache, "issues", fetchers, make_refetch([live]),
                 ("timeline",), n=7, live=live)
        self.assertEqual(len(fetchers[("issues", "timeline")].calls), 1)


# ---------------------------------------------------------------------------
# Stable-fetch loop
# ---------------------------------------------------------------------------


class StableFetchTests(unittest.TestCase):

    def test_consistent_read_settles_in_one_attempt(self) -> None:
        cache = Cache()
        fetchers = patch_fetch(self)
        refetch = make_refetch([LIVE])
        call_get(cache, "pulls", fetchers, refetch, ("timeline",))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 1)
        self.assertEqual(len(refetch.calls), 1)

    def test_one_advance_then_settle_retries_once(self) -> None:
        moved = utc("2026-05-21T10:05:00Z")
        cache = Cache()
        fetchers = patch_fetch(self, **{
            "pulls.timeline": [[{"attempt": 1}], [{"attempt": 2}]],
        })
        call_get(cache, "pulls", fetchers, make_refetch([moved, moved]),
                 ("timeline",))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), 2)
        entry = cache.entries[entry_key(ARGS, "pulls", "o", "r", 1)]
        self.assertEqual(entry.updated_at, moved)
        self.assertEqual(entry.fields["timeline"], ({"attempt": 2},))

    def test_continuous_advance_exhausts_attempts(self) -> None:
        moving = [utc(f"2026-05-21T10:0{i}:00Z")
                  for i in range(1, MAX_REFETCH_ATTEMPTS + 2)]
        cache = Cache()
        fetchers = patch_fetch(self)
        refetch = make_refetch(moving)
        call_get(cache, "pulls", fetchers, refetch, ("timeline",))
        self.assertEqual(len(fetchers[("pulls", "timeline")].calls), MAX_REFETCH_ATTEMPTS)
        self.assertEqual(len(refetch.calls), MAX_REFETCH_ATTEMPTS)

    def test_reviews_fetched_before_review_comments(self) -> None:
        # review_comments consumes the freshly-fetched reviews list;
        # if they were fetched out of order the review_comments call
        # would see an empty list and silently miss inline comments.
        seen_reviews_arg: list[list[dict]] = []

        class CaptureReviewComments(CountingFetcher):
            def __call__(self, args, owner, repo, n, reviews):
                seen_reviews_arg.append(list(reviews))
                return super().__call__(args, owner, repo, n, reviews)

        cache = Cache()
        fetchers = patch_fetch(self, **{
            "pulls.reviews": [[
                {"id": 100, "comments_count": 2},
                {"id": 200, "comments_count": 5},
            ]],
            "pulls.review_comments": CaptureReviewComments(),
        })
        call_get(cache, "pulls", fetchers, make_refetch([LIVE]), ("reviews",))
        self.assertEqual(seen_reviews_arg, [[
            {"id": 100, "comments_count": 2},
            {"id": 200, "comments_count": 5},
        ]])


if __name__ == "__main__":
    unittest.main()
