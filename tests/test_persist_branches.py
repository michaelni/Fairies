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

Branch persistence through the containers' fairy remotes: the
``branches``/``pull_requests`` declaration vocabulary and its boundary
sanitizers, the remote setup and declared-branch collection into
bundle-carrying records, the publication (push, force, delete, PR
creation) at send time, and the prompt/role plumbing.
"""

from __future__ import annotations

import base64
import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Minimal fakes so ``openai_reviewer`` imports without the SDK (same
# pattern as test_wrapper_functional_replay).
try:
    import openai  # noqa: F401
except ModuleNotFoundError:
    _fake_openai = types.ModuleType("openai")

    class _FakeOpenAIError(Exception):
        pass

    _fake_openai.OpenAI = object
    _fake_openai.RateLimitError = _FakeOpenAIError
    _fake_openai.AuthenticationError = _FakeOpenAIError
    _fake_openai.InternalServerError = _FakeOpenAIError
    sys.modules["openai"] = _fake_openai
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    _fake_dotenv = types.ModuleType("dotenv")
    _fake_dotenv.dotenv_values = lambda *_args, **_kwargs: {}
    sys.modules["dotenv"] = _fake_dotenv

try:
    import openai_reviewer  # noqa: E402
except ModuleNotFoundError:  # a transitive dependency (e.g. httpx) is absent
    openai_reviewer = None

import branch_persist  # noqa: E402
import llm_prompt  # noqa: E402
import llm_review_api  # noqa: E402
import agent  # noqa: E402
import fairy  # noqa: E402
import issue_fairy  # noqa: E402
import podman_host  # noqa: E402
import podman_repos  # noqa: E402
import worker  # noqa: E402
from llm_review_api import (  # noqa: E402
    MAX_PERSIST_BRANCHES,
    sanitize_branch_declarations,
    sanitize_pull_requests,
    schema_with_branches,
    validate_result_with_branches,
)

REPOS = ["ffmpeg", "ffmpeg-web"]
VALID_PR_REQUEST = {"repo": "ffmpeg", "branch": "pr7-fix-overflow",
                    "title": "Fix overflow", "body": "details",
                    "target": "master"}
VALID_DECLARATION = {"repo": "ffmpeg", "branch": "pr7-fix-overflow",
                     "action": "push"}


class SanitizePullRequestsTests(unittest.TestCase):
    def test_valid_requests_survive(self) -> None:
        self.assertEqual(sanitize_pull_requests([VALID_PR_REQUEST], REPOS),
                         [VALID_PR_REQUEST])

    def test_unknown_repo_is_dropped(self) -> None:
        self.assertEqual(
            sanitize_pull_requests(
                [dict(VALID_PR_REQUEST, repo="all_ffmpeg")], REPOS), [])

    def test_unsafe_branch_names_are_dropped(self) -> None:
        for name in ("a b", "../evil", "refs/heads/x", "-fast", "", "a" * 65,
                     "ünïcode", "x;rm", None, 7):
            with self.subTest(name=name):
                self.assertEqual(
                    sanitize_pull_requests(
                        [dict(VALID_PR_REQUEST, branch=name)], REPOS), [])

    def test_bad_title_or_target_is_dropped(self) -> None:
        for bad in (dict(VALID_PR_REQUEST, title=""),
                    dict(VALID_PR_REQUEST, target="-evil"),
                    dict(VALID_PR_REQUEST, target="a b"),
                    "open one please"):
            with self.subTest(bad=bad):
                self.assertEqual(sanitize_pull_requests([bad], REPOS), [])

    def test_duplicates_and_cap(self) -> None:
        many = [dict(VALID_PR_REQUEST, branch=f"b{i}")
                for i in range(MAX_PERSIST_BRANCHES + 2)]
        self.assertEqual(len(sanitize_pull_requests(many, REPOS)),
                         MAX_PERSIST_BRANCHES)
        self.assertEqual(
            len(sanitize_pull_requests([VALID_PR_REQUEST] * 2, REPOS)), 1)

    def test_non_list_is_empty(self) -> None:
        for raw in (None, "x", {"branch": "b"}):
            self.assertEqual(sanitize_pull_requests(raw, REPOS), [])


class SanitizeBranchDeclarationsTests(unittest.TestCase):
    def test_valid_declarations_survive(self) -> None:
        self.assertEqual(
            sanitize_branch_declarations([VALID_DECLARATION], REPOS),
            [VALID_DECLARATION])

    def test_bad_action_repo_or_name_is_dropped(self) -> None:
        for bad in (dict(VALID_DECLARATION, action="merge"),
                    dict(VALID_DECLARATION, repo="all_ffmpeg"),
                    dict(VALID_DECLARATION, branch="a b"),
                    {"branch": "x"}, "keep it"):
            with self.subTest(bad=bad):
                self.assertEqual(sanitize_branch_declarations([bad], REPOS), [])

    def test_duplicates_and_cap(self) -> None:
        many = [dict(VALID_DECLARATION, branch=f"b{i}")
                for i in range(MAX_PERSIST_BRANCHES + 2)]
        self.assertEqual(len(sanitize_branch_declarations(many, REPOS)),
                         MAX_PERSIST_BRANCHES)
        self.assertEqual(
            len(sanitize_branch_declarations([VALID_DECLARATION] * 2, REPOS)), 1)


class PersistenceSchemaTests(unittest.TestCase):
    def test_schema_accepts_valid_output(self) -> None:
        schema = schema_with_branches(llm_review_api.REVIEW_SCHEMA, REPOS)
        llm_review_api.check_schema(
            {"classification": "approve", "message": "",
             "head_vs_branch_diff_evidence": False,
             "branches": [VALID_DECLARATION],
             "pull_requests": [VALID_PR_REQUEST]},
            schema["schema"])

    def test_schema_rejects_unknown_repo_and_keys(self) -> None:
        schema = schema_with_branches(llm_review_api.REVIEW_SCHEMA, REPOS)
        for bad in (dict(VALID_PR_REQUEST, repo="all_ffmpeg"),
                    dict(VALID_PR_REQUEST, remote="origin")):
            with self.subTest(bad=bad):
                with self.assertRaises(llm_review_api.SchemaError):
                    llm_review_api.check_schema(
                        {"classification": "approve", "message": "",
                         "head_vs_branch_diff_evidence": False,
                         "branches": [],
                         "pull_requests": [bad]},
                        schema["schema"])

    def test_base_schema_still_rejects_pull_requests(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            llm_review_api.check_schema(
                {"classification": "approve", "message": "",
                 "head_vs_branch_diff_evidence": False, "pull_requests": []},
                llm_review_api.REVIEW_SCHEMA["schema"])


class ValidateWithBranchesTests(unittest.TestCase):
    def test_composes_over_labels_validator(self) -> None:
        result = validate_result_with_branches(
            {"classification": "minor_issues_approve", "message": "m",
             "head_vs_branch_diff_evidence": False,
             "label_changes": [{"label": "important", "op": "add",
                                "reason": "r", "post": False}],
             "branches": [dict(VALID_DECLARATION, branch="cleanup",
                               action="delete"),
                          dict(VALID_DECLARATION, action="merge")],
             "pull_requests": [VALID_PR_REQUEST,
                               dict(VALID_PR_REQUEST, branch="bad name")]},
            REPOS,
            lambda obj: llm_review_api.validate_review_result(obj, ["important"]))
        self.assertEqual(result["classification"], "minor_issues_approve")
        self.assertEqual(len(result["label_changes"]), 1)
        self.assertEqual(result["branches"],
                         [dict(VALID_DECLARATION, branch="cleanup",
                               action="delete")])
        self.assertEqual(result["pull_requests"], [VALID_PR_REQUEST])

    def test_verdict_fields_stay_strict(self) -> None:
        with self.assertRaises(llm_review_api.SchemaError):
            validate_result_with_branches(
                {"message": "m", "pull_requests": []}, REPOS,
                llm_review_api.validate_review)


GOOD_RECORD = {
    "branch": "pr7-fix-overflow", "mode": "ff", "pr": None, "repo": "ffmpeg",
    "sha": "a" * 40, "old_sha": None,
    "bundle": base64.b64encode(b"BUNDLE").decode(),
    "objects_repo": "/client/ffmpeg", "diff_base_sha": "b" * 40,
}
FORCE_RECORD = dict(GOOD_RECORD, mode="force", old_sha="d" * 40)
DELETE_RECORD = {
    "branch": "pr7-stale", "mode": "delete", "pr": None, "repo": "ffmpeg",
    "sha": "c" * 40, "old_sha": "c" * 40, "bundle": "",
    "objects_repo": "/client/ffmpeg", "diff_base_sha": None,
}


def _cmd(rc: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> SimpleNamespace:
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _repo_spec(name: str = "ffmpeg") -> podman_repos.RepoSpec:
    return podman_repos.RepoSpec(
        repo_root=Path(f"/client/{name}"), name=name, head_sha="c" * 40,
        container_path=f"/work/{name}",
        mirror_path=f"fairy-mirrors/{name}.git")


HANDLE = podman_host.ContainerHandle(
    "cid123", "img", None, podman_host.RemoteHost("fairy@h"))


class SetupContainerRemotesTests(unittest.TestCase):
    def test_seeds_come_from_the_fake_remote_listing(self) -> None:
        listing = (f"good {'d' * 40}\n"
                   f"bad name {'d' * 40}\n"
                   f"odd {'zz'}\n")

        def fake_run(host, *argv, **kwargs):
            if "for-each-ref" in argv:
                return _cmd(stdout=listing.encode())
            return _cmd()

        with mock.patch.object(branch_persist, "run_on_remote_host",
                               side_effect=fake_run):
            seeds = branch_persist.setup_container_remotes(
                HANDLE, [_repo_spec()], ["ffmpeg"])
        self.assertEqual(seeds, {"ffmpeg": {"good": "d" * 40}})

    def test_disabled_repos_are_skipped(self) -> None:
        with mock.patch.object(branch_persist, "run_on_remote_host",
                               side_effect=AssertionError("touched")):
            seeds = branch_persist.setup_container_remotes(
                HANDLE, [_repo_spec()], [])
        self.assertEqual(seeds, {})

    def test_the_ref_listing_is_read_under_the_output_cap(self) -> None:
        caps: list[object] = []

        def fake_run(host, *argv, **kwargs):
            if "for-each-ref" in argv:
                caps.append(kwargs.get("max_output_bytes"))
            return _cmd()

        with mock.patch.object(branch_persist, "run_on_remote_host",
                               side_effect=fake_run):
            branch_persist.setup_container_remotes(
                HANDLE, [_repo_spec()], ["ffmpeg"])
        self.assertEqual(caps, [branch_persist.MAX_BUNDLE_BYTES])

    def test_seeds_come_from_the_fork_and_forge_remotes(self) -> None:
        refspecs: list[str] = []

        def fake_run(host, *argv, **kwargs):
            if "push" in argv:
                refspecs.append(argv[-1])
            return _cmd()

        with mock.patch.object(branch_persist, "run_on_remote_host",
                               side_effect=fake_run):
            branch_persist.setup_container_remotes(
                HANDLE, [_repo_spec()], ["ffmpeg"])
        # the priority remote pushes last, so it wins name collisions
        self.assertEqual(
            refspecs,
            [f"+refs/remotes/{remote}/fairy/*:refs/heads/*"
             for remote in reversed(branch_persist.FAIRY_BRANCH_REMOTES)])
        self.assertEqual(branch_persist.FAIRY_BRANCH_REMOTES[0], "fairy")

    def test_a_failing_setup_raises(self) -> None:
        with mock.patch.object(branch_persist, "run_on_remote_host",
                               return_value=_cmd(rc=1, stderr=b"no space")):
            with self.assertRaisesRegex(branch_persist.BranchTransferError,
                                        "no space"):
                branch_persist.setup_container_remotes(
                    HANDLE, [_repo_spec()], ["ffmpeg"])


class CollectDeclaredBranchesTests(unittest.TestCase):
    """collect_declared_branches packs the verdict's declared branches
    out of the fairy remotes: a push becomes a record carrying its thin
    bundle, a delete records the seeded tip, and everything the verdict
    did not declare stays behind."""

    def _collect(self, seeds, current, declared, pull_requests=(), *,
                 ancestor_rc=0, bundle=_cmd(stdout=b"BUNDLE")):
        def fake_cgit(handle, repo_path, *args, **kwargs):
            if args[0] == "for-each-ref":
                return _cmd(stdout=f"fforge/master {'c' * 40}\n".encode())
            if args[:2] == ("merge-base", "--is-ancestor"):
                return _cmd(rc=ancestor_rc)
            if args[0] == "merge-base":
                return _cmd(stdout=("b" * 40).encode() + b"\n")
            if args[:2] == ("bundle", "create"):
                return bundle
            raise AssertionError(f"unexpected container git {args}")

        with mock.patch.object(branch_persist, "_list_fake_refs",
                               return_value=dict(current)), \
                mock.patch.object(branch_persist, "_container_git",
                                  side_effect=fake_cgit):
            return branch_persist.collect_declared_branches(
                [(HANDLE, {"ffmpeg": dict(seeds)})], list(declared),
                list(pull_requests), repo_specs=[_repo_spec()],
                base_shas={"ffmpeg": ["c" * 40]})

    def test_a_declared_push_carries_its_bundle(self) -> None:
        records = self._collect({}, {"new": "f" * 40},
                                [{"repo": "ffmpeg", "branch": "new",
                                  "action": "push"}])
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual((record.branch, record.mode, record.sha,
                          record.old_sha, record.objects_repo),
                         ("new", "ff", "f" * 40, None, "/client/ffmpeg"))
        self.assertEqual(base64.b64decode(record.bundle), b"BUNDLE")

    def test_undeclared_pushed_branches_stay_behind(self) -> None:
        records = self._collect({}, {"new": "f" * 40, "scratch": "e" * 40},
                                [{"repo": "ffmpeg", "branch": "new",
                                  "action": "push"}])
        self.assertEqual([r.branch for r in records], ["new"])

    def test_a_pull_request_implies_the_push_and_attaches(self) -> None:
        request = dict(VALID_PR_REQUEST, branch="new")
        records = self._collect({}, {"new": "f" * 40}, [], [request])
        self.assertEqual(records[0].pr, request)
        self.assertEqual(records[0].mode, "ff")

    def test_a_declared_branch_no_remote_holds_is_dropped(self) -> None:
        records = self._collect({}, {}, [{"repo": "ffmpeg", "branch": "gone",
                                          "action": "push"}])
        self.assertEqual(records, [])

    def test_a_non_ancestor_seed_derives_force(self) -> None:
        records = self._collect({"moved": "d" * 40}, {"moved": "e" * 40},
                                [{"repo": "ffmpeg", "branch": "moved",
                                  "action": "push"}], ancestor_rc=1)
        self.assertEqual((records[0].mode, records[0].old_sha),
                         ("force", "d" * 40))

    def test_an_unchanged_branch_yields_no_record(self) -> None:
        records = self._collect({"keep": "d" * 40}, {"keep": "d" * 40},
                                [{"repo": "ffmpeg", "branch": "keep",
                                  "action": "push"}])
        self.assertEqual(records, [])

    def test_a_declared_deletion_records_the_seeded_tip(self) -> None:
        records = self._collect({"stale": "d" * 40}, {"stale": "d" * 40},
                                [{"repo": "ffmpeg", "branch": "stale",
                                  "action": "delete"}])
        self.assertEqual(len(records), 1)
        self.assertEqual((records[0].mode, records[0].sha, records[0].bundle),
                         ("delete", "d" * 40, ""))

    def test_deleting_an_unpublished_branch_is_dropped(self) -> None:
        records = self._collect({}, {}, [{"repo": "ffmpeg", "branch": "x",
                                          "action": "delete"}])
        self.assertEqual(records, [])

    def test_cap_limits_records_per_repo(self) -> None:
        current = {f"b{i}": ("%02d" % i) * 20
                   for i in range(MAX_PERSIST_BRANCHES + 2)}
        declared = [{"repo": "ffmpeg", "branch": b, "action": "push"}
                    for b in current]
        records = self._collect({}, current, declared)
        self.assertEqual(len(records), MAX_PERSIST_BRANCHES)

    def test_the_found_handles_own_seeds_drive_mode_and_negatives(self) -> None:
        handle2 = podman_host.ContainerHandle(
            "cid456", "img", None, podman_host.RemoteHost("fairy@h"))
        bundle_negatives: list[tuple] = []

        def fake_cgit(handle, repo_path, *args, **kwargs):
            if args[0] == "for-each-ref":
                return _cmd(stdout=b"")
            if args[:2] == ("merge-base", "--is-ancestor"):
                return _cmd(rc=1)
            if args[0] == "merge-base":
                return _cmd(rc=1)
            if args[:2] == ("bundle", "create"):
                bundle_negatives.append(args[args.index("--not") + 1:])
                return _cmd(stdout=b"BUNDLE")
            raise AssertionError(args)

        def fake_list(handle, repo):
            return {"moved": "f" * 40} if handle is handle2 else {}

        with mock.patch.object(branch_persist, "_list_fake_refs",
                               side_effect=fake_list), \
                mock.patch.object(branch_persist, "_container_git",
                                  side_effect=fake_cgit):
            records = branch_persist.collect_declared_branches(
                [(HANDLE, {"ffmpeg": {"moved": "1" * 40}}),
                 (handle2, {"ffmpeg": {"moved": "2" * 40}})],
                [{"repo": "ffmpeg", "branch": "moved", "action": "push"}],
                [], repo_specs=[_repo_spec()], base_shas={})
        self.assertEqual(records[0].old_sha, "2" * 40)
        self.assertEqual(bundle_negatives, [("2" * 40,)])

    def test_a_pr_target_thins_the_bundle_and_anchors_the_diff(self) -> None:
        target_tip = "9" * 40
        bundle_args: list[tuple] = []

        def fake_cgit(handle, repo_path, *args, **kwargs):
            if args[0] == "for-each-ref":
                return _cmd(stdout=f"fforge/master {target_tip}\n".encode())
            if args[0] == "merge-base":
                return _cmd(stdout=(args[2] + "\n").encode())
            if args[:2] == ("bundle", "create"):
                bundle_args.append(args)
                return _cmd(stdout=b"BUNDLE")
            raise AssertionError(f"unexpected container git {args}")

        request = dict(VALID_PR_REQUEST, branch="new")
        with mock.patch.object(branch_persist, "_list_fake_refs",
                               return_value={"new": "f" * 40}), \
                mock.patch.object(branch_persist, "_container_git",
                                  side_effect=fake_cgit):
            records = branch_persist.collect_declared_branches(
                [(HANDLE, {"ffmpeg": {}})], [], [request],
                repo_specs=[_repo_spec()], base_shas={"ffmpeg": ["c" * 40]})
        self.assertIn(target_tip, bundle_args[0])
        self.assertEqual(records[0].diff_base_sha, target_tip)

    def test_an_unresolved_pr_target_leaves_the_preview_baseless(self) -> None:
        def fake_cgit(handle, repo_path, *args, **kwargs):
            if args[0] == "for-each-ref":
                return _cmd(stdout=f"fforge/release/8.1 {'8' * 40}\n".encode())
            if args[:2] == ("bundle", "create"):
                return _cmd(stdout=b"BUNDLE")
            raise AssertionError(f"unexpected container git {args}")

        request = dict(VALID_PR_REQUEST, branch="new")
        with mock.patch.object(branch_persist, "_list_fake_refs",
                               return_value={"new": "f" * 40}), \
                mock.patch.object(branch_persist, "_container_git",
                                  side_effect=fake_cgit):
            records = branch_persist.collect_declared_branches(
                [(HANDLE, {"ffmpeg": {}})], [], [request],
                repo_specs=[_repo_spec()], base_shas={"ffmpeg": ["c" * 40]})
        self.assertEqual(len(records), 1)
        self.assertIsNone(records[0].diff_base_sha)

    def test_a_plain_push_diffs_from_its_single_prerequisite(self) -> None:
        fork_point = "a1" * 20
        header = (b"# v2 git bundle\n-" + fork_point.encode()
                  + b" base subject\n" + b"f" * 40
                  + b" refs/heads/new\n\nPACKDATA")
        records = self._collect({}, {"new": "f" * 40},
                                [{"repo": "ffmpeg", "branch": "new",
                                  "action": "push"}],
                                bundle=_cmd(stdout=header))
        self.assertEqual(records[0].diff_base_sha, fork_point)

    def test_two_prerequisites_leave_the_preview_baseless(self) -> None:
        header = (b"# v2 git bundle\n-" + b"a" * 40 + b" one\n-"
                  + b"b" * 40 + b" two\n" + b"f" * 40
                  + b" refs/heads/new\n\nPACK")
        records = self._collect({}, {"new": "f" * 40},
                                [{"repo": "ffmpeg", "branch": "new",
                                  "action": "push"}],
                                bundle=_cmd(stdout=header))
        self.assertIsNone(records[0].diff_base_sha)

    def test_a_tip_the_forge_knows_bundles_empty(self) -> None:
        # observed on git 2.43: a bundle whose every object the negatives
        # cover is refused with "fatal: Refusing to create empty bundle."
        records = self._collect(
            {}, {"alias": "f" * 40},
            [{"repo": "ffmpeg", "branch": "alias", "action": "push"}],
            bundle=_cmd(rc=128, stderr=b"fatal: Refusing to create empty bundle."))
        self.assertEqual(records[0].bundle, "")

    def test_a_failing_bundle_raises(self) -> None:
        with self.assertRaisesRegex(branch_persist.BranchTransferError,
                                    "bundling"):
            self._collect({}, {"new": "f" * 40},
                          [{"repo": "ffmpeg", "branch": "new",
                            "action": "push"}],
                          bundle=_cmd(rc=128, stderr=b"fatal: bad object"))


def _fake_host_git(calls: list[tuple] | None = None,
                   ls_remote: bytes | str = ""):
    """A ``branch_persist._git`` stand-in: ``init`` creates the bare
    layout ``materialized_record`` writes into, everything else records
    and succeeds."""
    def fake_git(repo, *args, check=True, **kwargs):
        if calls is not None:
            calls.append(args)
        if args[0] == "init":
            (Path(args[-1]) / "objects" / "info").mkdir(parents=True)
        if args[0] == "ls-remote":
            out = ls_remote if isinstance(ls_remote, str) else ls_remote.decode()
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        if args[:2] == ("rev-parse", "--path-format=absolute"):
            return SimpleNamespace(returncode=0, stdout="/client/objects\n",
                                   stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")
    return fake_git


class PublishBranchRecordTests(unittest.TestCase):
    URL = "https://forge.example.com/r.git"

    def test_bad_records_are_refused_before_any_git(self) -> None:
        for bad in (dict(GOOD_RECORD, branch="a b"),
                    dict(GOOD_RECORD, mode="push"),
                    dict(GOOD_RECORD, sha="not-hex"),
                    dict(GOOD_RECORD, mode="force"),
                    dict(GOOD_RECORD, old_sha="xyz"),
                    dict(GOOD_RECORD, bundle=None),
                    {}):
            with self.subTest(bad=bad), \
                    mock.patch.object(branch_persist, "_git",
                                      side_effect=AssertionError("git ran")), \
                    mock.patch.object(branch_persist, "git_push_refspecs",
                                      side_effect=AssertionError("pushed")):
                with self.assertRaises(branch_persist.BranchTransferError):
                    branch_persist.publish_branch_record(bad, remote_url=self.URL)

    def test_ff_pushes_plainly_so_a_moved_forge_branch_refuses(self) -> None:
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git()), \
                mock.patch.object(branch_persist, "git_rev_parse",
                                  return_value=GOOD_RECORD["sha"]), \
                mock.patch.object(branch_persist, "git_push_refspecs") as push:
            name = branch_persist.publish_branch_record(
                GOOD_RECORD, remote_url=self.URL)
        self.assertEqual(name, "fairy/pr7-fix-overflow")
        self.assertEqual(
            push.call_args.args[2],
            [f"{GOOD_RECORD['sha']}:refs/heads/fairy/pr7-fix-overflow"])
        self.assertFalse(push.call_args.kwargs["force"])
        self.assertIsNone(push.call_args.kwargs["force_with_lease"])

    def test_force_pushes_under_the_recorded_old_tips_lease(self) -> None:
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git()), \
                mock.patch.object(branch_persist, "git_rev_parse",
                                  return_value=FORCE_RECORD["sha"]), \
                mock.patch.object(branch_persist, "git_push_refspecs") as push:
            branch_persist.publish_branch_record(FORCE_RECORD,
                                                 remote_url=self.URL)
        self.assertEqual(
            push.call_args.kwargs["force_with_lease"],
            f"refs/heads/fairy/pr7-fix-overflow:{FORCE_RECORD['old_sha']}")

    def test_a_bundle_with_the_wrong_tip_refuses_the_push(self) -> None:
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git()), \
                mock.patch.object(branch_persist, "git_rev_parse",
                                  return_value="f" * 40), \
                mock.patch.object(branch_persist, "git_push_refspecs",
                                  side_effect=AssertionError("pushed")):
            with self.assertRaises(branch_persist.BranchTransferError):
                branch_persist.publish_branch_record(GOOD_RECORD,
                                                     remote_url=self.URL)

    def test_delete_pushes_under_the_recorded_tips_lease(self) -> None:
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git()), \
                mock.patch.object(branch_persist, "git_push_refspecs") as push:
            branch_persist.publish_branch_record(DELETE_RECORD,
                                                 remote_url=self.URL)
        self.assertEqual(push.call_args.args[2],
                         [":refs/heads/fairy/pr7-stale"])
        self.assertEqual(
            push.call_args.kwargs["force_with_lease"],
            f"refs/heads/fairy/pr7-stale:{DELETE_RECORD['sha']}")

    def test_delete_of_an_already_gone_branch_passes(self) -> None:
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git(ls_remote="")), \
                mock.patch.object(branch_persist, "git_push_refspecs",
                                  side_effect=RuntimeError("stale info")):
            branch_persist.publish_branch_record(DELETE_RECORD,
                                                 remote_url=self.URL)

    def test_delete_of_a_moved_tip_is_refused(self) -> None:
        listed = f"{'e' * 40}\trefs/heads/fairy/pr7-stale\n"
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git(ls_remote=listed)), \
                mock.patch.object(branch_persist, "git_push_refspecs",
                                  side_effect=RuntimeError("stale info")):
            with self.assertRaisesRegex(branch_persist.BranchTransferError,
                                        "moved"):
                branch_persist.publish_branch_record(DELETE_RECORD,
                                                     remote_url=self.URL)

    def test_a_bundleless_record_publishes_from_the_checkout(self) -> None:
        calls: list[tuple] = []
        record = dict(GOOD_RECORD, bundle="")
        with mock.patch.object(branch_persist, "_git",
                               side_effect=_fake_host_git(calls)), \
                mock.patch.object(branch_persist, "git_rev_parse",
                                  return_value=record["sha"]), \
                mock.patch.object(branch_persist, "git_push_refspecs"):
            branch_persist.publish_branch_record(record, remote_url=self.URL)
        self.assertIn(("update-ref", "refs/heads/pr7-fix-overflow",
                       record["sha"]), calls)


class ParseBranchRecordsTests(unittest.TestCase):
    def test_keeps_string_complete_records_and_deletions(self) -> None:
        self.assertEqual(
            fairy.parse_branch_records([GOOD_RECORD, DELETE_RECORD]),
            (GOOD_RECORD, DELETE_RECORD))
        for bad in (dict(GOOD_RECORD, sha=""), dict(GOOD_RECORD, bundle=7),
                    {"branch": "x"}, "x"):
            with self.subTest(bad=bad):
                self.assertEqual(fairy.parse_branch_records([bad]), ())


def _push_spec(repo: str = "ffmpeg") -> fairy.BranchPushSpec:
    return fairy.BranchPushSpec(
        repo=repo, owner="mm", forge_repo=repo,
        url=f"https://forge.example.com/mm/{repo}.git")


def _send_args(**overrides: object) -> SimpleNamespace:
    return SimpleNamespace(**{"owner": "o", "repo": "r",
                              "branch_push": [_push_spec()],
                              "branch_push_head_owner": [], **overrides})


def _branch_decision(*records: dict) -> fairy.Decision:
    return fairy.Decision(1, "t", "a", "-", "comment", "llm", None,
                          llm_message="LLM review",
                          branches=records or (GOOD_RECORD,))


class BranchPushSpecTests(unittest.TestCase):
    def test_parse_roundtrip(self) -> None:
        spec = fairy.parse_branch_push_spec(
            "ffmpeg=mm/ffmpeg=https://x/y.git?a=b")
        self.assertEqual(spec, fairy.BranchPushSpec(
            "ffmpeg", "mm", "ffmpeg", "https://x/y.git?a=b"))

    def test_malformed_specs_are_rejected(self) -> None:
        import argparse
        for bad in ("ffmpeg", "ffmpeg=mm=https://x", "=mm/r=u", "a=mm/=u"):
            with self.subTest(bad=bad):
                with self.assertRaises(argparse.ArgumentTypeError):
                    fairy.parse_branch_push_spec(bad)


class SendPathBranchTests(unittest.TestCase):
    def test_missing_push_config_blocks_before_posting(self) -> None:
        decision = _branch_decision()
        with mock.patch.object(fairy, "post_issue_comment") as post:
            reason = fairy.submit_decision_action(
                _send_args(branch_push=[]), decision, decision,
                skip_guard=True)
        self.assertIn("--branch-push", reason)
        post.assert_not_called()

    def test_unconfigured_repo_blocks_the_send(self) -> None:
        decision = _branch_decision(dict(GOOD_RECORD, repo="ffmpeg-web"))
        with mock.patch.object(fairy, "post_issue_comment") as post:
            reason = fairy.submit_decision_action(
                _send_args(), decision, decision, skip_guard=True)
        self.assertIn("branch publication failed", reason)
        self.assertIn("ffmpeg-web", reason)
        post.assert_not_called()

    def test_an_unconfigured_repo_blocks_before_any_push(self) -> None:
        decision = _branch_decision(
            GOOD_RECORD, dict(GOOD_RECORD, repo="ffmpeg-web"))
        with mock.patch.object(branch_persist, "publish_branch_record") as pub:
            with self.assertRaisesRegex(branch_persist.BranchTransferError,
                                        "ffmpeg-web"):
                fairy.publish_decision_branches(_send_args(), decision)
        pub.assert_not_called()

    def test_publication_failure_blocks_before_posting(self) -> None:
        decision = _branch_decision()
        with mock.patch.object(fairy, "post_issue_comment") as post, \
                mock.patch.object(
                    fairy, "publish_decision_branches",
                    side_effect=branch_persist.BranchTransferError("ref moved")):
            reason = fairy.submit_decision_action(
                _send_args(), decision, decision, skip_guard=True)
        self.assertEqual(reason, "branch publication failed: ref moved")
        post.assert_not_called()

    def test_a_publication_timeout_blocks_the_send(self) -> None:
        import subprocess
        decision = _branch_decision()
        with mock.patch.object(fairy, "post_issue_comment") as post, \
                mock.patch.object(
                    fairy, "publish_decision_branches",
                    side_effect=subprocess.TimeoutExpired("git push", 300)):
            reason = fairy.submit_decision_action(
                _send_args(), decision, decision, skip_guard=True)
        self.assertIn("branch publication failed", reason)
        post.assert_not_called()

    def test_branches_publish_then_review_posts(self) -> None:
        decision = _branch_decision()
        with mock.patch.object(fairy, "post_issue_comment") as post, \
                mock.patch.object(fairy, "publish_decision_branches") as publish:
            reason = fairy.submit_decision_action(
                _send_args(), decision, decision, skip_guard=True)
        self.assertIsNone(reason)
        publish.assert_called_once()
        post.assert_called_once()

    def test_publish_routes_each_repo_to_its_spec(self) -> None:
        web_record = dict(GOOD_RECORD, repo="ffmpeg-web", branch="site-fix")
        decision = _branch_decision(GOOD_RECORD, web_record)
        urls: list[str] = []
        with mock.patch.object(
                branch_persist, "publish_branch_record",
                side_effect=lambda record, **kw: (
                    urls.append(kw["remote_url"]),
                    f"fairy/{record['branch']}")[1]):
            fairy.publish_decision_branches(
                _send_args(branch_push=[_push_spec(), _push_spec("ffmpeg-web")]),
                decision)
        self.assertEqual(urls, ["https://forge.example.com/mm/ffmpeg.git",
                                "https://forge.example.com/mm/ffmpeg-web.git"])

    def test_publish_pushes_all_branches_before_opening_prs(self) -> None:
        pr_record = dict(GOOD_RECORD, branch="add-test",
                         pr={"title": "Add test", "body": "b", "target": "master"})
        decision = _branch_decision(pr_record, GOOD_RECORD)
        order: list[str] = []
        with mock.patch.object(
                branch_persist, "publish_branch_record",
                side_effect=lambda record, **kw: (
                    order.append(f"publish {record['branch']}"),
                    f"fairy/{record['branch']}")[1]), \
                mock.patch.object(
                    fairy, "gcli_create_pr",
                    side_effect=lambda *a, **kw: order.append(f"pr {a[4]}")):
            fairy.publish_decision_branches(_send_args(), decision)
        self.assertEqual(order, ["publish add-test", "publish pr7-fix-overflow",
                                 "pr fairy/add-test"])

    def test_a_refused_create_of_an_open_pr_passes(self) -> None:
        pr_record = dict(GOOD_RECORD, branch="add-test",
                         pr={"title": "Add test", "body": "b", "target": "master"})
        decision = _branch_decision(pr_record)
        with mock.patch.object(branch_persist, "publish_branch_record",
                               return_value="fairy/add-test"), \
                mock.patch.object(
                    fairy, "list_open_prs",
                    return_value=[{"head": {"ref": "fairy/add-test"}}]), \
                mock.patch.object(fairy, "gcli_create_pr",
                                  side_effect=RuntimeError("already exists")):
            fairy.publish_decision_branches(_send_args(), decision)

    def test_a_refused_create_without_an_open_pr_raises(self) -> None:
        pr_record = dict(GOOD_RECORD, branch="add-test",
                         pr={"title": "Add test", "body": "b", "target": "master"})
        decision = _branch_decision(pr_record)
        with mock.patch.object(branch_persist, "publish_branch_record",
                               return_value="fairy/add-test"), \
                mock.patch.object(fairy, "list_open_prs", return_value=[]), \
                mock.patch.object(fairy, "gcli_create_pr",
                                  side_effect=RuntimeError("bad target")):
            with self.assertRaisesRegex(RuntimeError, "bad target"):
                fairy.publish_decision_branches(_send_args(), decision)

    def test_pr_creation_names_the_specs_repo_and_head_owner(self) -> None:
        pr_record = dict(GOOD_RECORD,
                         pr={"title": "T", "body": "", "target": "master"})
        decision = _branch_decision(pr_record)
        with mock.patch.object(branch_persist, "publish_branch_record",
                               return_value="fairy/pr7-fix-overflow"), \
                mock.patch.object(fairy, "list_open_prs", return_value=[]), \
                mock.patch.object(fairy, "gcli_create_pr") as create:
            fairy.publish_decision_branches(
                _send_args(branch_push_head_owner=["ffmpeg=forkfairy"]),
                decision)
        self.assertEqual(create.call_args.args[1:5],
                         ("mm", "ffmpeg", "forkfairy", "fairy/pr7-fix-overflow"))

    def test_skip_verdicts_drop_their_branches(self) -> None:
        decision = fairy.decision_from_review(
            fairy.LLMReview("skip", "nothing to add",
                            branches=(GOOD_RECORD,)),
            number=1, title="t", author="a", auto_merge="-",
            last_activity=None, base_reason="llm")
        self.assertEqual(decision.branches, ())
        actionable = fairy.decision_from_review(
            fairy.LLMReview("moderate_issues", "m", branches=(GOOD_RECORD,)),
            number=1, title="t", author="a", auto_merge="-",
            last_activity=None, base_reason="llm")
        self.assertEqual(actionable.branches, (GOOD_RECORD,))


class GcliCreatePrArgvTests(unittest.TestCase):
    def test_the_create_rides_the_editor_stub(self) -> None:
        # gcli 2.12 pulls create composes its message in $EDITOR
        # unconditionally and getopt-parses a title like "-Wformat fix"
        # as options unless "--" pins it as the positional
        import forge_gcli
        calls: list[tuple] = []

        def fake_stub(cmd, *, message, **kwargs):
            calls.append((cmd, message))
            return SimpleNamespace(returncode=0, stderr="")

        with mock.patch.object(forge_gcli, "run_gcli_editor_submission",
                               side_effect=fake_stub):
            forge_gcli.gcli_create_pr(
                SimpleNamespace(gcli_account=None, forge_type=None),
                "mm", "ffmpeg", "forkfairy", "fairy/x", "master",
                "-Wformat fix", "the body")
        cmd, message = calls[0]
        self.assertEqual(cmd[-2:], ["--", "-Wformat fix"])
        self.assertNotIn("-T", cmd)
        self.assertEqual(message, "the body")


class EditorStubBangLineTests(unittest.TestCase):
    def test_a_leading_bang_line_survives_gcli(self) -> None:
        import forge_gcli
        seen: dict = {}

        def fake_run_cmd(cmd, *, env, **kwargs):
            body = Path(env["EDITOR"]).parent / "body.txt"
            seen["body"] = body.read_text(encoding="utf-8")
            return SimpleNamespace(returncode=0, stderr="")

        with mock.patch.object(forge_gcli, "run_cmd",
                               side_effect=fake_run_cmd):
            forge_gcli.run_gcli_editor_submission(
                ["gcli"], message="See:\n![img](u)\n!literal\nend",
                verbose=0)
        self.assertEqual(seen["body"], "See:\n ![img](u)\n !literal\nend")


class RoleWithBranchesTests(unittest.TestCase):
    def test_composes_over_role_with_labels(self) -> None:
        role = llm_prompt.role_with_branches(
            llm_prompt.role_with_labels(llm_prompt.REVIEWER_ROLE, ["important"]),
            REPOS)
        props = role.schema["schema"]["properties"]
        self.assertIn("branches", props)
        self.assertIn("pull_requests", props)
        self.assertIn("label_changes", props)
        self.assertTrue(role.prompt_kwargs["persist_branches"])
        result = role.validate(
            {"classification": "approve", "message": "",
             "head_vs_branch_diff_evidence": False,
             "label_changes": [], "branches": [VALID_DECLARATION],
             "pull_requests": [VALID_PR_REQUEST]})
        self.assertEqual(result["branches"], [VALID_DECLARATION])
        self.assertEqual(result["pull_requests"], [VALID_PR_REQUEST])

    def test_prompt_section_appears_only_when_enabled(self) -> None:
        kwargs = dict(
            vendor="anthropic", model="m", features=set(),
            repo_roots=[], container_repo_mounts=["/work/ffmpeg"],
            reviewer_username="fairy")
        for role in ("review", "combiner", "issue_investigator", "issue_combiner"):
            with self.subTest(role=role):
                self.assertNotIn("##Persisting branches",
                                 llm_prompt.generate_llm_prompt(role=role, **kwargs))
                enabled = llm_prompt.generate_llm_prompt(
                    role=role, persist_branches=True, **kwargs)
                self.assertIn("##Persisting branches", enabled)
                self.assertIn('remote "fairy"', enabled)
                self.assertIn("fairy/<name>", enabled)
                self.assertIn("Assisted-by: Fairy", enabled)

    def test_combiner_prompt_says_only_declared_branches_survive(self) -> None:
        prompt = llm_prompt.prompt_persist_branches(
            llm_prompt.PromptFor("combiner", "m"))
        self.assertIn("on your fairy remotes", prompt)
        self.assertIn("only your own declarations count", prompt)
        self.assertIn("Re-declare", prompt)


class AddToContainerRemoteTests(unittest.TestCase):
    """The drafts' collected branches reach the combiner by preloading
    its fairy remotes -- unpublished reviewer work must not be lost on
    the way to the combine stage."""

    def test_bundles_load_the_remote_and_deletions_apply(self) -> None:
        remote_ops: list[tuple] = []
        copied: list[str] = []

        def fake_cgit(handle, repo_path, *args, **kwargs):
            remote_ops.append(args)
            return _cmd()

        with mock.patch.object(branch_persist, "_container_git",
                               side_effect=fake_cgit), \
                mock.patch.object(
                    branch_persist, "copy_into_container",
                    side_effect=lambda handle, path, dest: copied.append(
                        path.name)):
            branch_persist.add_to_container_remote(
                HANDLE, [GOOD_RECORD, dict(FORCE_RECORD, bundle=""),
                         DELETE_RECORD],
                {"ffmpeg": "/work/ffmpeg"})
        self.assertEqual(copied, ["ffmpeg-pr7-fix-overflow.bundle"])
        fetches = [op for op in remote_ops if op[0] == "fetch"]
        self.assertEqual(len(fetches), 1)
        self.assertIn("/quarantine/ffmpeg-pr7-fix-overflow.bundle", fetches[0])
        self.assertIn(
            "+refs/heads/pr7-fix-overflow:refs/heads/pr7-fix-overflow",
            fetches[0])
        self.assertIn(("update-ref", "refs/heads/pr7-fix-overflow",
                       FORCE_RECORD["sha"]), remote_ops)
        self.assertIn(("update-ref", "-d", "refs/heads/pr7-stale"), remote_ops)


@unittest.skipUnless(openai_reviewer is not None,
                     "openai_reviewer dependencies missing")
class OpenAIPersistBranchesTests(unittest.TestCase):
    """The OpenAI reviewer hands its shared per-machine sessions and its
    validated declarations and pull_requests to ctx.collect_branches,
    whose records land on the Review."""

    def test_collects_via_shared_sessions(self) -> None:
        session = object()
        resources = openai_reviewer.OpenAIResources(
            client=None, tools=[], include=[], patch_file_id=None,
            vector_store_ids=[], shared_container_id=None,
            shells={"x86_64": session},
            open_shell=lambda label: (session, ""),
            uploaded_file_ids=[], debug_dir_specified=False)
        args = SimpleNamespace(
            podman=True, verbose=False, verbosity=None, service_tier=None,
            top_p=None, reasoning_summary=None, max_tool_calls=None,
            max_output_tokens=1000, web_search="off",
            podman_max_tool_rounds=0, podman_exec_timeout=60.0,
            podman_parallel_tool_calls=False, debug_response_dir=".dbg",
            use_openai_container_repos=False, model="gpt-5.4")
        reviewer = openai_reviewer.OpenAIReviewer(
            args, resources, model="gpt-5.4",
            role=llm_prompt.role_with_branches(llm_prompt.REVIEWER_ROLE, REPOS),
            service_tier=None)
        calls: list[tuple] = []

        def collect(sessions, declared, pull_requests):
            calls.append((list(sessions), declared, pull_requests))
            return [GOOD_RECORD]

        ctx = llm_review_api.ReviewContext(
            request={"pull_request": {"number": 7}},
            patch_text="", patch_truncated=False, source_bundle=None,
            source_files=[], source_notes=[], reviewer_username="fairy",
            ci_triage_mode=False, repo_roots=[], repo_mount_paths=["/work/ffmpeg"],
            machines=[podman_host.ShellHostSpec(
                "x86_64", podman_host.RemoteHost("fairy@h"))],
            collect_branches=collect)
        payload = json.dumps({
            "classification": "minor_issues_approve", "message": "LLM review",
            "head_vs_branch_diff_evidence": False,
            "branches": [VALID_DECLARATION],
            "pull_requests": [VALID_PR_REQUEST]})
        with mock.patch.object(openai_reviewer,
                               "run_responses_resolving_podman_shell",
                               return_value={}), \
                mock.patch.object(openai_reviewer, "extract_response_annotations",
                                  return_value=[]), \
                mock.patch.object(openai_reviewer,
                                  "extract_response_file_citation_metadata",
                                  return_value={}), \
                mock.patch.object(openai_reviewer, "extract_response_text",
                                  return_value=payload):
            review = reviewer.review(ctx)
        self.assertEqual(review.branches, (GOOD_RECORD,))
        self.assertEqual(calls, [([session], [VALID_DECLARATION],
                                  [VALID_PR_REQUEST])])


class CrossItemBranchTests(unittest.TestCase):
    """Working one item does not scope what fairy may touch: during
    issue 1234 she can read PR 4321 -- every pull request's head is a
    revision in the checkout -- and persist a modified version of it as
    a fairy branch that rides the issue's own verdict to the forge."""

    def test_issue_prompt_offers_pr_heads_and_the_fairy_remote(self) -> None:
        prompt = llm_prompt.generate_llm_prompt(
            role="issue_investigator", vendor="anthropic", model="m",
            features={"podman_shell"}, repo_roots=[Path("/client/ffmpeg")],
            container_repo_mounts=["/work/ffmpeg"], reviewer_username="fairy",
            persist_branches=True,
            machines=[podman_host.ShellHostSpec(
                "x86_64", podman_host.RemoteHost("fairy@h"))])
        self.assertIn("fforge/pr/", prompt)
        self.assertIn('remote "fairy"', prompt)

    def test_issue_session_persists_a_branch_modifying_another_pr(self) -> None:
        def fake_cgit(handle, repo_path, *args, **kwargs):
            if args[0] == "for-each-ref":
                return _cmd(stdout=b"")
            if args[:2] == ("bundle", "create"):
                return _cmd(stdout=b"BUNDLE")
            return _cmd(rc=1)

        with mock.patch.object(branch_persist, "_list_fake_refs",
                               return_value={"pr4321-fix": "e" * 40}), \
                mock.patch.object(branch_persist, "_container_git",
                                  side_effect=fake_cgit):
            records = branch_persist.collect_declared_branches(
                [(HANDLE, {"ffmpeg": {}})],
                [{"repo": "ffmpeg", "branch": "pr4321-fix", "action": "push"}],
                [], repo_specs=[_repo_spec()], base_shas={"ffmpeg": []})
        self.assertEqual([(r.branch, r.mode) for r in records],
                         [("pr4321-fix", "ff")])

        decision = issue_fairy.issue_review_decision(
            fairy.LLMReview("reply", "fixed PR 4321 while investigating",
                            branches=(records[0].record(),)),
            number=1234, title="t", author="a", reason="llm",
            last_activity=None)
        self.assertEqual(decision.branches[0]["branch"], "pr4321-fix")

        published: list[str] = []
        cache = SimpleNamespace(entries={})
        args = _send_args(cache="/nonexistent")
        with mock.patch.object(
                fairy, "publish_decision_branches",
                side_effect=lambda a, d: published.extend(
                    r["branch"] for r in d.branches)), \
                mock.patch.object(issue_fairy, "post_issue_comment") as post, \
                mock.patch.object(issue_fairy.gcli_cache, "entry_key",
                                  return_value="k"), \
                mock.patch.object(issue_fairy.gcli_cache, "save_cache"):
            reason = issue_fairy.submit_issue_decision(
                args, decision, cache=cache,
                submitted_counts={"comment": 0}, skip_guard=True)
        self.assertIsNone(reason)
        self.assertEqual(published, ["pr4321-fix"])
        post.assert_called_once()


class TicketRoundTripTests(unittest.TestCase):
    def test_branches_survive_verdict_fields_and_ticket_decision(self) -> None:
        decision = fairy.Decision(
            5, "t", "a", "-", "comment", "llm", None,
            llm_classification="moderate_issues", llm_message="msg",
            branches=(GOOD_RECORD, DELETE_RECORD))
        prepared = SimpleNamespace(
            pr={"updated_at": "2026-08-25T00:00:00Z", "head": {"ref": "b"}},
            discussion=[])
        fields = worker.verdict_fields(decision, prepared)
        self.assertEqual(fields["review"]["branches"],
                         [GOOD_RECORD, DELETE_RECORD])
        rebuilt = agent.ticket_decision("pr", "5", {"review": fields["review"]})
        self.assertEqual(rebuilt.branches, (GOOD_RECORD, DELETE_RECORD))


if __name__ == "__main__":
    unittest.main()
