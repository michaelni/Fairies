"""Tests for ``mail_fairy.py`` -- the maildir-to-Forgejo bridge.

The code under test is project-agnostic: it accepts any forge host,
owner, repo, and gcli account via CLI arguments. The test fixtures
below happen to be real list mails captured from one specific
project's mailing list, used as concrete inputs to anchor the most
subtle invariants. Nothing in ``mail_fairy.py`` itself depends on
those values.

Two fixture mails live in ``tests/fixtures/mail_fairy/``:

- ``forge_pr_root.eml`` is a Forgejo PR-#22883 notification as it
  arrived on the list: subject ending with ``(PR #22883)``, body
  line 2 carrying a forge URL of the form
  ``<host>/<owner>/<repo>/pulls/22883``, an ``X-MailFrom:`` header
  giving the forge's outgoing address, and two ``Archived-At:``
  headers (the second being the lore-style URL we want to link to).
- ``human_reply.eml`` is a human reply to that root, with
  ``In-Reply-To:`` chained to the root's ``Message-ID``, ``From:``
  rewritten by Mailman 3 to the ``<name> via <list> <list-addr>``
  form, the original sender's address in ``Cc:``, and an inline
  reply (quote on top, ``LGTM.`` body, signature, Mailman footer).

The pure helpers (subject parsing, address extraction, body cleanup,
attribution-line composition, marker scanning, threading) all run
without touching the filesystem or any network. Only ``read_headers``
and ``read_body`` open the fixture files.
"""

from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mail_fairy  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "mail_fairy"
FORGE_ROOT = FIX / "forge_pr_root.eml"
HUMAN_REPLY = FIX / "human_reply.eml"


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormalizeMsgid(unittest.TestCase):
    def test_strips_angle_brackets_and_whitespace(self):
        self.assertEqual(
            mail_fairy.normalize_msgid("  <abc@host>  "),
            "abc@host",
        )

    def test_returns_empty_for_none_or_empty(self):
        self.assertEqual(mail_fairy.normalize_msgid(None), "")
        self.assertEqual(mail_fairy.normalize_msgid(""), "")

    def test_leaves_unbracketed_value_alone(self):
        self.assertEqual(
            mail_fairy.normalize_msgid("abc@host"),
            "abc@host",
        )


class TestMaildirFilenameTs(unittest.TestCase):
    def test_extracts_timestamp(self):
        ts = mail_fairy.parse_maildir_filename_ts("1776807809.541154_0.neo")
        self.assertEqual(ts, 1776807809.0)

    def test_returns_none_for_non_maildir_name(self):
        self.assertIsNone(mail_fairy.parse_maildir_filename_ts("hello.txt"))


class TestForgejoFlavorSubjectParsing(unittest.TestCase):
    def test_pr_subject(self):
        self.assertEqual(
            mail_fairy.FORGEJO_FLAVOR.classify_subject(
                "[FFmpeg-devel] [PR] Fix drawtext error handling (PR #22883)"
            ),
            [(mail_fairy.KIND_PR, 22883)],
        )

    def test_issue_subject(self):
        self.assertEqual(
            mail_fairy.FORGEJO_FLAVOR.classify_subject(
                "[FFmpeg-devel] [Issue] some title (Issue #42)"
            ),
            [(mail_fairy.KIND_ISSUE, 42)],
        )

    def test_non_forge_subject(self):
        self.assertEqual(
            mail_fairy.FORGEJO_FLAVOR.classify_subject(
                "[FFmpeg-devel] [PATCH] avformat/wavenc: ...",
            ),
            [],
        )

    def test_empty(self):
        self.assertEqual(mail_fairy.FORGEJO_FLAVOR.classify_subject(""), [])


class TestSelectForgeFlavor(unittest.TestCase):
    def test_known_keys_resolve(self):
        for key in ("forgejo", "gitea", "github", "gitlab"):
            self.assertEqual(mail_fairy.select_forge_flavor(key).name,
                             {"gitea": "forgejo"}.get(key, key))

    def test_case_and_whitespace_insensitive(self):
        self.assertEqual(
            mail_fairy.select_forge_flavor("  GitHub ").name, "github",
        )

    def test_unknown_raises_with_helpful_message(self):
        with self.assertRaises(ValueError) as cm:
            mail_fairy.select_forge_flavor("bitbucket")
        msg = str(cm.exception)
        self.assertIn("bitbucket", msg)
        self.assertIn("Supported", msg)
        self.assertIn("forgejo", msg)


class TestGitHubFlavorParsing(unittest.TestCase):
    flavor = mail_fairy.GITHUB_FLAVOR

    def test_subject_captures_number_with_unknown_kind(self):
        # GitHub mails end "(#N)" without distinguishing PR vs Issue.
        # The flavor returns kind=None so build_thread_index will try
        # both PR-shaped and Issue-shaped body URLs.
        self.assertEqual(
            self.flavor.classify_subject(
                "[octocat/spoon-knife] Sample title (#1234)"
            ),
            [(None, 1234)],
        )

    def test_body_url_uses_singular_pull_for_pr(self):
        candidates = self.flavor.candidate_targets(
            "https://github.com", "octocat", "spoon-knife",
            "[octocat/spoon-knife] Title (#9)",
        )
        urls = sorted(c.html_url for c in candidates)
        self.assertEqual(urls, [
            "https://github.com/octocat/spoon-knife/issues/9",
            "https://github.com/octocat/spoon-knife/pull/9",
        ])

    def test_body_confirmation_against_pr_url(self):
        candidates = self.flavor.candidate_targets(
            "https://github.com", "octocat", "spoon-knife",
            "[octocat/spoon-knife] Title (#9)",
        )
        body = "View it on GitHub:\nhttps://github.com/octocat/spoon-knife/pull/9\n"
        matches = [
            c for c in candidates if mail_fairy.body_confirms_target(body, c)
        ]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].kind, mail_fairy.KIND_PR)

    def test_non_github_subject_returns_no_candidates(self):
        self.assertEqual(
            self.flavor.classify_subject("Plain subject without number"),
            [],
        )


class TestGitLabFlavorParsing(unittest.TestCase):
    flavor = mail_fairy.GITLAB_FLAVOR

    def test_mr_subject_resolves_to_pr_kind(self):
        self.assertEqual(
            self.flavor.classify_subject(
                "Project | feature: New merge request (!42)"
            ),
            [(mail_fairy.KIND_PR, 42)],
        )

    def test_issue_subject_resolves_to_issue_kind(self):
        self.assertEqual(
            self.flavor.classify_subject(
                "Project | bug: Crash on startup (#7)"
            ),
            [(mail_fairy.KIND_ISSUE, 7)],
        )

    def test_url_uses_dash_separator_and_merge_requests_segment(self):
        candidates = self.flavor.candidate_targets(
            "https://gitlab.com", "group/sub", "myproject",
            "Project | title (!42)",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].html_url,
            "https://gitlab.com/group/sub/myproject/-/merge_requests/42",
        )

    def test_url_uses_dash_separator_and_issues_segment_for_issues(self):
        candidates = self.flavor.candidate_targets(
            "https://gitlab.com", "group/sub", "myproject",
            "Project | bug (#7)",
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            candidates[0].html_url,
            "https://gitlab.com/group/sub/myproject/-/issues/7",
        )

    def test_body_confirmation_with_dash_segment(self):
        candidates = self.flavor.candidate_targets(
            "https://gitlab.com", "g", "p", "Title (!11)",
        )
        body = "View MR: https://gitlab.com/g/p/-/merge_requests/11\n"
        self.assertTrue(
            any(mail_fairy.body_confirms_target(body, c) for c in candidates)
        )

    def test_untested_flag_is_set(self):
        # The CLI uses this to print a startup warning. If anyone
        # graduates the flavor to "tested", they should flip the flag
        # explicitly (and remove this assertion).
        self.assertTrue(self.flavor.untested)


class TestForgeFlavorRegression(unittest.TestCase):
    def test_forgejo_is_not_marked_untested(self):
        self.assertFalse(mail_fairy.FORGEJO_FLAVOR.untested)


class TestBodyConfirmsTarget(unittest.TestCase):
    def setUp(self):
        self.target = mail_fairy.ForgejoTarget(
            host="https://code.ffmpeg.org",
            owner="FFmpeg", repo="FFmpeg",
            kind=mail_fairy.KIND_PR, number=22883,
        )

    def test_matches(self):
        body = "URL: https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/22883\n"
        self.assertTrue(mail_fairy.body_confirms_target(body, self.target))

    def test_different_number(self):
        body = "URL: https://code.ffmpeg.org/FFmpeg/FFmpeg/pulls/22884\n"
        self.assertFalse(mail_fairy.body_confirms_target(body, self.target))

    def test_different_kind(self):
        body = "URL: https://code.ffmpeg.org/FFmpeg/FFmpeg/issues/22883\n"
        self.assertFalse(mail_fairy.body_confirms_target(body, self.target))

    def test_different_host(self):
        body = "URL: https://other.example.org/FFmpeg/FFmpeg/pulls/22883\n"
        self.assertFalse(mail_fairy.body_confirms_target(body, self.target))


class TestStripMailmanFooter(unittest.TestCase):
    def test_strips_underscore_block(self):
        body = (
            "Hello.\n"
            "\n"
            "_______________________________________________\n"
            "ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
            "To unsubscribe send an email to ffmpeg-devel-leave@ffmpeg.org\n"
        )
        out = mail_fairy.strip_mailman_footer(body)
        self.assertIn("Hello.", out)
        self.assertNotIn("mailing list", out)
        self.assertNotIn("_______", out)

    def test_no_footer_passthrough(self):
        body = "Hello.\nworld.\n"
        self.assertEqual(mail_fairy.strip_mailman_footer(body), body)


class TestStripBottomQuote(unittest.TestCase):
    def test_drops_trailing_full_quote(self):
        body = (
            "Short reply text.\n"
            "\n"
            "On Tue, X wrote:\n"
            "> long quote line 1\n"
            "> long quote line 2\n"
            ">\n"
            "> long quote line 3\n"
        )
        out = mail_fairy.strip_bottom_quote(body)
        self.assertIn("Short reply text.", out)
        self.assertNotIn("long quote line", out)
        self.assertNotIn("On Tue, X wrote:", out)

    def test_keeps_inline_reply(self):
        body = (
            "> quoted text\n"
            "\n"
            "LGTM.\n"
            "\n"
            "Regards,\n"
        )
        out = mail_fairy.strip_bottom_quote(body)
        self.assertIn("LGTM.", out)
        self.assertIn("quoted text", out)

    def test_keeps_quote_only_body(self):
        body = "> all of this is quoted\n> and this too\n"
        out = mail_fairy.strip_bottom_quote(body)
        self.assertIn("all of this", out)

    def test_empty_body(self):
        self.assertEqual(mail_fairy.strip_bottom_quote(""), "")


class TestExtractAuthor(unittest.TestCase):
    def test_picks_cc_match_for_via_list(self):
        h = mail_fairy.MailHeaders(
            path=Path("/tmp/x"), file_ts=0.0, file_size=0,
            from_raw="Nicolas George via ffmpeg-devel <ffmpeg-devel@ffmpeg.org>",
            cc_raw=(
                "Marvin Scholz <code@ffmpeg.org>, "
                "Nicolas George <george@nsup.org>"
            ),
        )
        name, addr = mail_fairy.extract_author(h)
        self.assertEqual(name, "Nicolas George")
        self.assertEqual(addr, "george@nsup.org")

    def test_falls_back_to_from_addr_without_cc_match(self):
        h = mail_fairy.MailHeaders(
            path=Path("/tmp/x"), file_ts=0.0, file_size=0,
            from_raw='"Some Name" <name@example.org>',
            cc_raw="",
        )
        name, addr = mail_fairy.extract_author(h)
        self.assertEqual(name, "Some Name")
        self.assertEqual(addr, "name@example.org")


class TestFormatAttributionLink(unittest.TestCase):
    def test_full_form(self):
        link = mail_fairy.format_attribution_link(
            "Nicolas George",
            "george@nsup.org",
            "2026-04-21 21:45 UTC",
            "https://lists.ffmpeg.org/lore/ffmpeg-devel/aefv4SA0WeymGdZA@phare.normalesup.org/",
        )
        self.assertTrue(link.startswith("[Fw by mail-fairy"))
        self.assertIn("Nicolas George", link)
        self.assertIn(r"\<george@nsup.org\>", link)
        self.assertIn("2026-04-21 21:45 UTC", link)
        self.assertTrue(link.endswith(")"))

    def test_no_url_falls_back_to_plain_text(self):
        link = mail_fairy.format_attribution_link(
            "X", "x@y", "DATE", "",
        )
        self.assertNotIn("[", link)
        self.assertNotIn("](", link)
        self.assertIn("From: X", link)


class TestFindMarkerMsgids(unittest.TestCase):
    def test_finds_marker(self):
        comments = [
            {"body": "hello\n<!-- mail-fairy:msgid:abc@host -->\nworld"},
            {"body": "no marker here"},
            {"body": "<!-- mail-fairy:msgid:def@x -->"},
        ]
        out = mail_fairy.find_marker_msgids(comments)
        self.assertEqual(out, {"abc@host", "def@x"})

    def test_robust_to_non_dict_entries(self):
        out = mail_fairy.find_marker_msgids(
            [None, 42, {"body": "<!-- mail-fairy:msgid:zzz -->"}]
        )
        self.assertEqual(out, {"zzz"})


class TestComposeCommentBody(unittest.TestCase):
    def test_marker_present_at_end(self):
        out = mail_fairy.compose_comment_body(
            "the body", "[Fw by mail-fairy ...](https://example/x)", "abc@host",
        )
        self.assertTrue(out.endswith("<!-- mail-fairy:msgid:abc@host -->\n"))
        self.assertIn("the body", out)
        self.assertIn("[Fw by mail-fairy", out)

    def test_marker_grep_roundtrip(self):
        msgid = "abc.def@host"
        out = mail_fairy.compose_comment_body("body", "attr", msgid)
        self.assertEqual(
            mail_fairy.find_marker_msgids([{"body": out}]),
            {msgid},
        )


# ---------------------------------------------------------------------------
# Fixture-driven integration of the parsing layer
# ---------------------------------------------------------------------------


class TestForgejoRootFixture(unittest.TestCase):
    def test_parse_headers(self):
        h = mail_fairy.read_headers(FORGE_ROOT)
        self.assertIsNotNone(h)
        self.assertEqual(
            h.message_id,
            "177680775885.45.8264619795826170142@29965ddac10e",
        )
        self.assertEqual(h.in_reply_to, "")
        self.assertIn("(PR #22883)", h.subject)
        self.assertEqual(h.x_mailfrom, "code@ffmpeg.org")
        self.assertEqual(
            h.lore_url,
            "https://lists.ffmpeg.org/lore/ffmpeg-devel/"
            "177680775885.45.8264619795826170142@29965ddac10e/",
        )

    def test_classify_root_subject(self):
        h = mail_fairy.read_headers(FORGE_ROOT)
        self.assertEqual(
            mail_fairy.FORGEJO_FLAVOR.classify_subject(h.subject),
            [(mail_fairy.KIND_PR, 22883)],
        )

    def test_body_confirms_target(self):
        h = mail_fairy.read_headers(FORGE_ROOT)
        body = mail_fairy.read_body(FORGE_ROOT)
        target = mail_fairy.ForgejoTarget(
            host="https://code.ffmpeg.org",
            owner="FFmpeg", repo="FFmpeg",
            kind=mail_fairy.KIND_PR, number=22883,
        )
        self.assertTrue(mail_fairy.body_confirms_target(body, target))


class TestHumanReplyFixture(unittest.TestCase):
    def test_in_reply_to_chains_to_root(self):
        root = mail_fairy.read_headers(FORGE_ROOT)
        reply = mail_fairy.read_headers(HUMAN_REPLY)
        self.assertEqual(reply.in_reply_to, root.message_id)
        self.assertIn(root.message_id, reply.references)

    def test_extract_author_picks_cc(self):
        reply = mail_fairy.read_headers(HUMAN_REPLY)
        name, addr = mail_fairy.extract_author(reply)
        self.assertEqual(name, "Nicolas George")
        self.assertEqual(addr, "george@nsup.org")

    def test_body_cleanup_keeps_inline_lgtm(self):
        body = mail_fairy.read_body(HUMAN_REPLY)
        cleaned = mail_fairy.strip_bottom_quote(
            mail_fairy.strip_mailman_footer(body)
        )
        self.assertIn("LGTM.", cleaned)
        self.assertNotIn("ffmpeg-devel mailing list --", cleaned)
        # The reply is inline (quote on top, text on bottom): the
        # quote MUST be preserved because it is not a trailing-only
        # block.
        self.assertIn(">", cleaned)


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------


class TestThreadIndex(unittest.TestCase):
    def test_root_resolves_and_reply_walks_up(self):
        root_h = mail_fairy.read_headers(FORGE_ROOT)
        reply_h = mail_fairy.read_headers(HUMAN_REPLY)
        idx = mail_fairy.build_thread_index(
            [root_h, reply_h],
            forge_host="https://code.ffmpeg.org",
            forge_owner="FFmpeg",
            forge_repo="FFmpeg",
        )
        self.assertIn(root_h.message_id, idx.forge_root_target)
        # Root is itself classified directly (no walk needed).
        target_root = mail_fairy.classify_via_threading(root_h, idx)
        self.assertIsNotNone(target_root)
        self.assertEqual(target_root.number, 22883)
        # Reply walks up via In-Reply-To to the root.
        target_reply = mail_fairy.classify_via_threading(reply_h, idx)
        self.assertIsNotNone(target_reply)
        self.assertEqual(target_reply.number, 22883)
        self.assertEqual(target_reply.kind, mail_fairy.KIND_PR)

    def test_orphan_reply_is_skipped(self):
        # A reply whose In-Reply-To target is NOT in the index must
        # not be classified.
        reply_h = mail_fairy.read_headers(HUMAN_REPLY)
        idx = mail_fairy.build_thread_index(
            [reply_h],
            forge_host="https://code.ffmpeg.org",
            forge_owner="FFmpeg",
            forge_repo="FFmpeg",
        )
        self.assertIsNone(mail_fairy.classify_via_threading(reply_h, idx))


# ---------------------------------------------------------------------------
# build_decision skip-paths
# ---------------------------------------------------------------------------


class TestBuildDecision(unittest.TestCase):
    def setUp(self):
        self.idx = mail_fairy.build_thread_index(
            [
                mail_fairy.read_headers(FORGE_ROOT),
                mail_fairy.read_headers(HUMAN_REPLY),
            ],
            forge_host="https://code.ffmpeg.org",
            forge_owner="FFmpeg",
            forge_repo="FFmpeg",
        )
        self.now = time.time()
        self.bot_re = mail_fairy.re.compile(r"^code@")

    def test_human_reply_is_actionable(self):
        h = mail_fairy.read_headers(HUMAN_REPLY)
        # Pretend the reply is recent enough for the age filter to
        # accept it (the fixture mail is from 2026-04-21 in real
        # time, so we override file_ts).
        h.file_ts = self.now - 60
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.ACTIONABLE, msg=d.reason)
        self.assertIsNotNone(d.target)
        self.assertEqual(d.target.number, 22883)
        self.assertIn("LGTM.", d.body)
        self.assertIn("[Fw by mail-fairy", d.body)
        self.assertIn(
            "<!-- mail-fairy:msgid:" + h.message_id + " -->",
            d.body,
        )

    def test_root_is_skipped_as_bot_mail(self):
        h = mail_fairy.read_headers(FORGE_ROOT)
        h.file_ts = self.now - 60
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        # X-MailFrom: code@ffmpeg.org matches the bot regex.
        self.assertEqual(d.action, mail_fairy.SKIP_BOT_MAIL)

    def test_too_old_skip(self):
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 86400 * 365
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 14,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.SKIP_TOO_OLD)

    def test_too_large_skip(self):
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        h.file_size = 10_000_000
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=256_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.SKIP_TOO_LARGE)

    def test_local_dedup_skip(self):
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids={h.message_id},
        )
        self.assertEqual(d.action, mail_fairy.SKIP_DEDUP_LOCAL)


class TestForgeGcliCommentsApiPathDispatch(unittest.TestCase):
    """Lock in the per-backend path shape used by list_issue_comments."""

    def setUp(self):
        import forge_gcli  # local import keeps the test self-contained
        self.fg = forge_gcli

    def test_forgejo(self):
        self.assertEqual(
            self.fg._comments_api_path("forgejo", "o", "r", 5, "pr"),
            "/repos/o/r/issues/5/comments",
        )

    def test_gitea_alias(self):
        self.assertEqual(
            self.fg._comments_api_path("gitea", "o", "r", 5, "pr"),
            "/repos/o/r/issues/5/comments",
        )

    def test_github_uses_same_issues_path_for_both_kinds(self):
        # GitHub's API serves PR comments through the issues endpoint,
        # mirroring forgejo/gitea.
        self.assertEqual(
            self.fg._comments_api_path("github", "octocat", "spoon-knife",
                                       9, "pr"),
            "/repos/octocat/spoon-knife/issues/9/comments",
        )
        self.assertEqual(
            self.fg._comments_api_path("github", "octocat", "spoon-knife",
                                       9, "issue"),
            "/repos/octocat/spoon-knife/issues/9/comments",
        )

    def test_gitlab_mr_uses_merge_requests_notes(self):
        path = self.fg._comments_api_path(
            "gitlab", "group/sub", "p", 42, "pr",
        )
        # Owner ``group/sub`` MUST be URL-encoded as a single segment
        # so gitlab parses it as one project identifier.
        self.assertEqual(
            path, "/projects/group%2Fsub%2Fp/merge_requests/42/notes",
        )

    def test_gitlab_issue_uses_issues_notes(self):
        path = self.fg._comments_api_path(
            "gitlab", "group", "p", 7, "issue",
        )
        self.assertEqual(path, "/projects/group%2Fp/issues/7/notes")

    def test_gitlab_unknown_kind_raises_with_pointer(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.fg._comments_api_path("gitlab", "g", "p", 1, "comment")
        self.assertIn("forge_gcli._comments_api_path", str(cm.exception))

    def test_unsupported_backend_raises_with_pointer(self):
        with self.assertRaises(NotImplementedError) as cm:
            self.fg._comments_api_path("bitbucket", "g", "p", 1, "pr")
        msg = str(cm.exception)
        self.assertIn("bitbucket", msg)
        self.assertIn("Supported", msg)
        self.assertIn("forge_gcli._comments_api_path", msg)


if __name__ == "__main__":
    unittest.main()
