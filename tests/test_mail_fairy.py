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

import re
import sys
import time
import email.utils
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mail_fairy  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "mail_fairy"


def _header(path, name: str) -> str:
    """Names and message-ids are drawn by tools/redact.py, so a test reads
    what it needs out of the fixture rather than naming it."""
    found = re.search(rf"^{name}:[ \t]*(.*)$", path.read_text(), re.M)
    return found.group(1).strip() if found else ""
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


class TestQuotedListFooterRe(unittest.TestCase):
    """Detect "full quote of prior list mail (footer included)" pattern."""

    def _matches(self, body: str) -> bool:
        return bool(mail_fairy._QUOTED_LIST_FOOTER_RE.search(body))

    def test_clean_reply_does_not_match(self):
        self.assertFalse(self._matches("Just a reply.\n"))

    def test_quoted_text_without_footer_does_not_match(self):
        body = (
            "> some quoted line\n"
            "> another quoted line\n"
            "\n"
            "My reply.\n"
        )
        self.assertFalse(self._matches(body))

    def test_unquoted_real_footer_does_not_match(self):
        # The actual Mailman footer of THIS mail (unquoted) must not
        # trigger -- this regex is specifically about a PREVIOUS
        # mail's footer included in the quote.
        body = (
            "Reply text.\n"
            "_______________________________________________\n"
            "ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
        )
        self.assertFalse(self._matches(body))

    def test_bottom_posted_full_quote_matches(self):
        body = (
            "OK.\n"
            "\n"
            "On Thu, X wrote:\n"
            "> some quoted body\n"
            "> _______________________________________________\n"
            "> ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
            "> To unsubscribe send an email to ffmpeg-devel-leave@ffmpeg.org\n"
        )
        self.assertTrue(self._matches(body))

    def test_top_posted_full_quote_matches(self):
        # Top-posting: the one-word reply is ABOVE the quote rather
        # than below. Detection must fire either way.
        body = (
            "Acknowledged.\n"
            "\n"
            "> some quoted body\n"
            "> _______________________________________________\n"
            "> ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
        )
        self.assertTrue(self._matches(body))

    def test_double_quote_levels_match(self):
        body = (
            "> > some text\n"
            "> > _______________________________________________\n"
            "> > a-list mailing list -- list@example.org\n"
            "\n"
            "Acknowledged.\n"
        )
        self.assertTrue(self._matches(body))

    def test_prose_mention_of_mailing_list_does_not_match(self):
        # Belt-and-suspenders: a prose line that happens to include
        # "mailing list --" in the middle (no quote prefix, no
        # email after the dashes) must not trigger.
        body = (
            "As you mentioned, the mailing list -- the devel one --\n"
            "is great.\n"
        )
        self.assertFalse(self._matches(body))


class TestStripQuotedListFooterBlocks(unittest.TestCase):
    """Drop quote blocks that contain a quoted Mailman list footer."""

    def test_no_match_returns_unchanged(self):
        body = "Plain reply.\n> snippet\n> more\n"
        self.assertEqual(
            mail_fairy.strip_quoted_list_footer_blocks(body), body,
        )

    def test_strips_block_above_reply(self):
        # Bottom-posted reply: quote on top, real content below.
        body = (
            "On Tue, X wrote:\n"
            "> long body\n"
            "> _______________________________________________\n"
            "> a-list mailing list -- a-list@example.org\n"
            "\n"
            "LGTM.\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertNotIn("mailing list", out)
        self.assertNotIn("On Tue, X wrote:", out)
        self.assertNotIn(">", out)
        self.assertIn("LGTM.", out)

    def test_strips_block_below_reply(self):
        # Top-posted reply: real content above, quote below.
        body = (
            "OK.\n"
            "\n"
            "On Tue, X wrote:\n"
            "> long body\n"
            "> _______________________________________________\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertIn("OK.", out)
        self.assertNotIn("mailing list", out)
        self.assertNotIn("On Tue, X wrote:", out)
        self.assertNotIn(">", out)

    def test_preserves_other_quote_blocks_without_footer(self):
        # Two quote blocks: only the one with the footer is dropped.
        body = (
            "Reply.\n"
            "\n"
            "On Tue, A wrote:\n"
            "> innocuous quote without footer\n"
            "\n"
            "On Mon, B wrote:\n"
            "> quoted full mail\n"
            "> _______________________________________________\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertIn("innocuous quote without footer", out)
        self.assertNotIn("mailing list", out)
        self.assertNotIn("quoted full mail", out)
        # Attribution for the surviving quote is preserved.
        self.assertIn("On Tue, A wrote:", out)
        # Attribution for the dropped quote is removed.
        self.assertNotIn("On Mon, B wrote:", out)

    def test_strips_to_empty_when_body_was_only_quote(self):
        body = (
            "On Tue, X wrote:\n"
            "> only this and footer\n"
            "> _______________________________________________\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertEqual(out, "")

    def test_strips_gmail_wrapped_multiline_attribution(self):
        # Gmail hard-wraps the ``<email>`` envelope of "On <date>,
        # X <addr> wrote:" -- the closing ``wrote:`` ends up on a
        # different physical line from "On". Both lines must be
        # dropped, otherwise the leading "On Thu, ... <" leaks into
        # the forwarded comment. Regression for the bug report
        # against b3fb740.
        body = (
            "On Thu, 30 Apr 2026, 03:13 michaelni via ffmpeg-devel, <\n"
            "ffmpeg-devel@ffmpeg.org> wrote:\n"
            "\n"
            "> quoted full body\n"
            "> _______________________________________________\n"
            "> ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
            "\n"
            "Lgtm\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertNotIn("On Thu", out)
        self.assertNotIn("ffmpeg-devel@ffmpeg.org>", out)
        self.assertNotIn("wrote:", out)
        self.assertNotIn("mailing list", out)
        self.assertIn("Lgtm", out)

    def test_envelope_tail_is_language_agnostic(self):
        # The wrap detector is keyed on the envelope-tail SHAPE of
        # the closer ("addr@host> wrote:"), not on the head word.
        # A French ("Le ... a écrit:")-style head wraps just like
        # English does and must be dropped even though the
        # _ATTRIBUTION_LINE_RE language list does not include
        # French. We verify the structural detector itself; the
        # closer is forced into a shape _ATTRIBUTION_LINE_RE
        # already accepts so the test exercises the new logic, not
        # _ATTRIBUTION_LINE_RE's language list.
        body = (
            "Le jeudi 30 avril 2026, Some Long Name, <\n"
            "addr@example.org> wrote:\n"
            "> body\n"
            "> a-list mailing list -- a-list@example.org\n"
            "\n"
            "merci\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertNotIn("Le jeudi", out)
        self.assertNotIn("addr@example.org", out)
        self.assertIn("merci", out)

    def test_envelope_tail_does_not_eat_prose_above(self):
        # If the closer starts with an envelope tail BUT the line
        # directly above is plain prose (no ``<`` or ``@``), the
        # walkback bails out and only the closer is dropped. This
        # is the core safety property that lets us be language-
        # agnostic without a head-word allowlist.
        body = (
            "Real prose ending with comma,\n"
            "addr@example.org> wrote:\n"
            "> quoted body\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertIn("Real prose ending with comma,", out)
        self.assertNotIn("addr@example.org", out)
        self.assertNotIn("mailing list", out)

    def test_attribution_without_envelope_tail_falls_back_to_one_line(self):
        # If the closer matches _ATTRIBUTION_LINE_RE but does NOT
        # start with an envelope tail, we never attempt a walkback
        # -- only the closer is dropped. This protects unrelated
        # prose above a single-line attribution.
        body = (
            "Some real prose above\n"
            "that wraps onto two lines.\n"
            "2026-04-30 X) wrote:\n"  # closer matches, no envelope
            "> quoted body\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertIn("Some real prose above", out)
        self.assertIn("that wraps onto two lines.", out)
        self.assertNotIn("wrote:", out)
        self.assertNotIn("mailing list", out)

    def test_attribution_blank_gap_stops_walkback(self):
        # Even with an envelope-tail closer, a blank line above
        # must stop the walkback -- otherwise we'd cross paragraph
        # boundaries.
        body = (
            "On a side note about addr@example.com.\n"
            "\n"
            "addr@example.org> wrote:\n"
            "> quoted body\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertIn("On a side note", out)
        self.assertNotIn("addr@example.org>", out)
        self.assertNotIn("mailing list", out)

    def test_does_not_strip_unquoted_mailing_list_line(self):
        # The actual Mailman footer of THIS mail must not be touched
        # by the quote-block stripper -- strip_mailman_footer handles
        # that elsewhere.
        body = (
            "Reply.\n"
            "_______________________________________________\n"
            "ffmpeg-devel mailing list -- ffmpeg-devel@ffmpeg.org\n"
        )
        out = mail_fairy.strip_quoted_list_footer_blocks(body)
        self.assertEqual(out, body)


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

    def test_body_is_wrapped_in_fenced_code_block(self):
        # Without the fence, Forgejo's markdown renderer turns
        # ``> ``-quoted patch text into blockquotes and mangles
        # ``<email>`` envelopes. Regression for the unfenced
        # output at ffmpeg PR #23205 comment 42267.
        out = mail_fairy.compose_comment_body(
            "> diff --git a/x b/y\n>  some content\n",
            "[Fw](https://x/y)",
            "abc@host",
        )
        self.assertIn("```text", out)
        # Find the opening fence and the matching closing fence.
        self.assertEqual(out.count("```"), 2)
        # Body content lives between the two fences.
        before, _, after = out.partition("```text\n")
        body_in_fence, _, _ = after.partition("\n```")
        self.assertIn("> diff --git", body_in_fence)
        # Attribution sits ABOVE the fence so its markdown link
        # actually renders.
        self.assertIn("[Fw](https://x/y)", before)
        # Marker sits BELOW the fence so it remains an HTML
        # comment and stays grep-able for dedup.
        self.assertTrue(out.endswith("<!-- mail-fairy:msgid:abc@host -->\n"))

    def test_marker_grep_roundtrip_with_backticks_in_body(self):
        # The grep-based dedup invariant must hold even when the
        # body contained backtick runs that forced a longer fence.
        msgid = "id@host"
        out = mail_fairy.compose_comment_body(
            "use ```bash``` blocks", "attr", msgid,
        )
        self.assertEqual(
            mail_fairy.find_marker_msgids([{"body": out}]),
            {msgid},
        )

    def test_fence_grows_around_embedded_triple_backticks(self):
        # If the body itself contains ```, a 3-backtick fence
        # would terminate at the first embedded run. The fence
        # must be at least one backtick longer than the longest
        # embedded run.
        body = "before\n```sh\nls\n```\nafter"
        out = mail_fairy.compose_comment_body(body, "attr", "id@host")
        # 4-backtick fence is the minimum that survives the
        # embedded 3-backtick run.
        self.assertIn("````text", out)
        self.assertNotIn("`````", out)
        # The embedded triple backticks survive verbatim.
        self.assertIn("```sh", out)
        self.assertIn("```\nafter", out)

    def test_fence_grows_around_quadruple_backticks(self):
        # Defense in depth: arbitrary-length runs work.
        body = "stuff ````````\nmore"
        out = mail_fairy.compose_comment_body(body, "attr", "id@host")
        # 8 embedded -> need at least 9.
        self.assertIn("`" * 9 + "text", out)
        self.assertNotIn("`" * 10, out)


# ---------------------------------------------------------------------------
# Fixture-driven integration of the parsing layer
# ---------------------------------------------------------------------------


class TestForgejoRootFixture(unittest.TestCase):
    def test_parse_headers(self):
        h = mail_fairy.read_headers(FORGE_ROOT)
        self.assertIsNotNone(h)
        self.assertEqual(
            h.message_id,
            _header(FORGE_ROOT, "Message-ID").strip("<>"),
        )
        self.assertEqual(h.in_reply_to, "")
        self.assertIn("(PR #22883)", h.subject)
        self.assertEqual(h.x_mailfrom, "code@ffmpeg.org")
        self.assertEqual(
            h.lore_url,
            "https://lists.ffmpeg.org/lore/ffmpeg-devel/"
            + _header(FORGE_ROOT, "Message-ID").strip("<>") + "/",
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
        self.assertEqual(name, reply.from_raw.split(" via ")[0].strip())
        self.assertIn(addr, [pair[1] for pair in
                             email.utils.getaddresses([reply.cc_raw])])

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

    def test_missing_msgid_skip(self):
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        h.message_id = ""
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.SKIP_NO_MSGID)

    def test_extra_skip_re_matches_from(self):
        # --skip-from regex hit on From: counts as a bot mail and short-
        # circuits before threading; the reason must differ from the
        # default forge_bot_re path so operators can tell them apart.
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=mail_fairy.re.compile(r"ffmpeg-devel"),
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.SKIP_BOT_MAIL)
        self.assertIn("--skip-from", d.reason)

    def test_no_in_reply_to_skip(self):
        # A thread root that is not itself a forge notification has no
        # In-Reply-To and no way to reach a forge ancestor; skip rather
        # than walk into a NoneType comparison.
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        h.in_reply_to = ""
        d = mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )
        self.assertEqual(d.action, mail_fairy.SKIP_NO_PARENT)

    FULL_QUOTE_BODY = (
        "OK.\n"
        "\n"
        "On Tue, X wrote:\n"
        "> a long quoted mail body\n"
        "> _______________________________________________\n"
        "> some-list mailing list -- list@example.org\n"
    )

    def test_full_quote_with_footer_skip_action(self):
        # full_quote_action="skip" : the mail is skipped entirely.
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        with mock.patch.object(
            mail_fairy, "read_body", return_value=self.FULL_QUOTE_BODY,
        ):
            d = mail_fairy.build_decision(
                h, self.idx,
                now_ts=self.now,
                max_age_seconds=86400 * 30,
                max_mail_bytes=1_000_000,
                forge_bot_re=self.bot_re,
                extra_skip_re=None,
                forwarded_msgids=set(),
                full_quote_action="skip",
            )
        self.assertEqual(d.action, mail_fairy.SKIP_FULL_QUOTE_WITH_FOOTER)
        # The target is retained so an operator investigating the
        # skip can still see which PR/Issue the mail targeted.
        self.assertIsNotNone(d.target)

    def test_full_quote_with_footer_strip_action(self):
        # full_quote_action="strip" (the default): drop the quote
        # block + attribution and forward what remains, here just
        # "OK." -- the actual reply that was sitting above the quote.
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        with mock.patch.object(
            mail_fairy, "read_body", return_value=self.FULL_QUOTE_BODY,
        ):
            d = mail_fairy.build_decision(
                h, self.idx,
                now_ts=self.now,
                max_age_seconds=86400 * 30,
                max_mail_bytes=1_000_000,
                forge_bot_re=self.bot_re,
                extra_skip_re=None,
                forwarded_msgids=set(),
                full_quote_action="strip",
            )
        self.assertEqual(d.action, mail_fairy.ACTIONABLE, msg=d.reason)
        self.assertIn("OK.", d.body)
        self.assertNotIn("mailing list", d.body)
        self.assertNotIn("On Tue, X wrote", d.body)
        self.assertNotIn("a long quoted mail body", d.body)
        self.assertNotIn("_____________", d.body)

    def test_full_quote_with_footer_strip_to_empty(self):
        # If stripping the offending block leaves nothing behind
        # (the reply WAS the quote), the mail falls through to
        # the existing empty-body skip rather than posting an
        # attribution-only stub.
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        body = (
            "On Tue, X wrote:\n"
            "> only the quote here\n"
            "> _______________________________________________\n"
            "> a-list mailing list -- a-list@example.org\n"
        )
        with mock.patch.object(mail_fairy, "read_body", return_value=body):
            d = mail_fairy.build_decision(
                h, self.idx,
                now_ts=self.now,
                max_age_seconds=86400 * 30,
                max_mail_bytes=1_000_000,
                forge_bot_re=self.bot_re,
                extra_skip_re=None,
                forwarded_msgids=set(),
                full_quote_action="strip",
            )
        self.assertEqual(d.action, mail_fairy.SKIP_EMPTY_BODY)

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

    def _patch_decision(self, subject: str) -> mail_fairy.MailDecision:
        """Run build_decision against a HUMAN_REPLY clone with a faked subject.

        The HUMAN_REPLY fixture is reused for everything except the
        subject so we don't need a second .eml file -- the patch-series
        check looks only at headers.subject and runs before any of the
        body-touching steps.
        """
        h = mail_fairy.read_headers(HUMAN_REPLY)
        h.file_ts = self.now - 60
        h.subject = subject
        return mail_fairy.build_decision(
            h, self.idx,
            now_ts=self.now,
            max_age_seconds=86400 * 30,
            max_mail_bytes=1_000_000,
            forge_bot_re=self.bot_re,
            extra_skip_re=None,
            forwarded_msgids=set(),
        )

    def test_patch_series_cover_letter_skip(self):
        d = self._patch_decision("[PATCH 0/3] lavfi: vf_drawtext: cover letter")
        self.assertEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_numbered_skip(self):
        d = self._patch_decision("[PATCH 1/3] lavfi: vf_drawtext: foo")
        self.assertEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_v2_skip(self):
        d = self._patch_decision("[PATCH v2 1/3] lavfi: vf_drawtext: foo")
        self.assertEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_rfc_skip(self):
        d = self._patch_decision("[RFC PATCH] lavfi: experimental thing")
        self.assertEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_with_list_prefix_skip(self):
        d = self._patch_decision(
            "[FFmpeg-devel] [PATCH 1/3] lavfi: vf_drawtext: foo",
        )
        self.assertEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_re_reply_not_caught(self):
        # ``Re: [PATCH ...]`` is a discussion reply; the guard must
        # not match it. Falls through to threading, which here finds
        # no forge ancestor and returns SKIP_NOT_FORGE_THREAD.
        d = self._patch_decision("Re: [PATCH 1/3] lavfi: vf_drawtext: foo")
        self.assertNotEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)

    def test_patch_series_forge_pr_subject_not_caught(self):
        d = self._patch_decision(
            "[FFmpeg-devel] [PR] Fix drawtext error handling (PR #22883)",
        )
        self.assertNotEqual(d.action, mail_fairy.SKIP_PATCH_SERIES)


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
