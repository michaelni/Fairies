"""mail_fairy CLI options, driven through ``main()`` end to end.

Each test runs the real pipeline over a temporary Maildir holding the
two fixture mails (a Forgejo PR notification and a human reply to it),
with only the two gcli seams -- reading existing comments and posting
one -- mocked. The observable effect of every flag is therefore what
actually gets posted where, not an attribute value.

The expectations come from each option's ``--help`` text: --maildir
scans ``new/`` and ``cur/`` and may be repeated, --max-age-days
pre-filters on the Maildir filename timestamp, --forge-base-url is the
host the thread index confirms against, --forge-bot-sender is the
loop guard, --forge-flavor overrides the parsing flavor derived from
--forge-type, and --state-file is the fast dedup layer.
"""

from __future__ import annotations

import pickle
import shutil
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mail_fairy  # noqa: E402

FIX = REPO_ROOT / "tests" / "fixtures" / "mail_fairy"
MAIL_TS = 1776807758
def _msgid(path) -> str:
    """Drawn by tools/redact.py, so read it out of the fixture."""
    return re.search(r"^Message-ID:[ \t]*<(.+?)>", path.read_text(),
                     re.M).group(1)


REPLY_MSGID = _msgid(FIX / "human_reply.eml")
FORGE_URL = "https://code.ffmpeg.org"
NEVER_TOO_OLD = "100000"


class MailFairyMainCase(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.tmp = Path(tmp.name)
        self.state_file = self.tmp / "state.pkl"
        self.maildir = self.make_maildir("list", ts=MAIL_TS)

    def make_maildir(self, name: str, *, ts: int,
                     mails: tuple[str, ...] = ("forge_pr_root.eml",
                                               "human_reply.eml")) -> Path:
        root = self.tmp / name
        (root / "new").mkdir(parents=True)
        (root / "cur").mkdir(parents=True)
        for index, mail in enumerate(mails):
            sub = "new" if index == 0 else "cur"
            shutil.copy(FIX / mail, root / sub / f"{ts}.{index}.testhost")
        return root

    def run_main(self, *flags: str, maildirs: tuple[Path, ...] = ()) -> tuple:
        argv: list[str] = []
        for path in (maildirs or (self.maildir,)):
            argv += ["--maildir", str(path)]
        argv += ["--owner", "FFmpeg", "--repo", "FFmpeg",
                 "--forge-base-url", FORGE_URL,
                 "--state-file", str(self.state_file),
                 "--max-age-days", NEVER_TOO_OLD, *flags]
        with mock.patch.object(mail_fairy, "setup_logging"), \
                mock.patch.object(mail_fairy.forge_gcli, "list_issue_comments",
                                  return_value=[]), \
                mock.patch.object(mail_fairy.forge_gcli,
                                  "post_issue_comment") as post:
            rc = mail_fairy.main(argv)
        return rc, post

    def decisions(self, *flags: str, maildirs: tuple[Path, ...] = ()) -> str:
        with self.assertLogs(mail_fairy.logger, level="INFO") as logs:
            self.run_main(*flags, maildirs=maildirs)
        return "\n".join(line for line in logs.output if "decisions:" in line)

    def posted_numbers(self, post) -> list[int]:
        return [call.args[3] for call in post.call_args_list]


class MaildirOptionTests(MailFairyMainCase):
    def test_the_reply_in_the_scanned_maildir_is_forwarded(self) -> None:
        rc, post = self.run_main()
        self.assertEqual(rc, 0)
        self.assertEqual(self.posted_numbers(post), [22883])

    def test_both_new_and_cur_are_scanned(self) -> None:
        """The root sits in new/ and the reply in cur/: dropping
        either subfolder loses the thread and nothing is forwarded."""
        for missing in ("new", "cur"):
            with self.subTest(missing=missing):
                root = self.make_maildir(f"only-{missing}", ts=MAIL_TS)
                shutil.rmtree(root / missing)
                _, post = self.run_main(maildirs=(root,))
                self.assertEqual(self.posted_numbers(post), [])

    def test_several_maildirs_are_all_scanned(self) -> None:
        second = self.make_maildir("second", ts=MAIL_TS)
        empty = self.make_maildir("empty", ts=MAIL_TS, mails=())
        _, post = self.run_main(maildirs=(empty, second))
        self.assertEqual(self.posted_numbers(post), [22883])

    def test_a_missing_maildir_is_a_config_error(self) -> None:
        rc, post = self.run_main(maildirs=(self.tmp / "nope",))
        self.assertEqual(rc, 2)
        post.assert_not_called()


class MaxAgeDaysOptionTests(MailFairyMainCase):
    def test_a_mail_older_than_the_window_is_never_opened(self) -> None:
        with mock.patch.object(mail_fairy, "read_headers") as read:
            rc, post = self.run_main("--max-age-days", "1")
        self.assertEqual(rc, 0)
        read.assert_not_called()
        post.assert_not_called()

    def test_a_mail_inside_the_window_is_forwarded(self) -> None:
        _, post = self.run_main()
        self.assertEqual(self.posted_numbers(post), [22883])


class ForgeBaseUrlOptionTests(MailFairyMainCase):
    def test_the_url_is_what_the_body_link_is_confirmed_against(self) -> None:
        argv_flags = ("--forge-base-url", "https://forge.example.org")
        rc, post = self.run_main(*argv_flags)
        self.assertEqual(rc, 0)
        self.assertEqual(self.posted_numbers(post), [])

    def test_the_matching_url_forwards_to_that_targets_number(self) -> None:
        _, post = self.run_main()
        self.assertEqual(self.posted_numbers(post), [22883])

    def test_a_trailing_slash_is_not_a_silent_no_op(self) -> None:
        _, post = self.run_main("--forge-base-url", FORGE_URL + "/")
        self.assertEqual(self.posted_numbers(post), [22883])


class ForgeBotSenderOptionTests(MailFairyMainCase):
    def test_the_default_regex_keeps_the_forge_notification_out(self) -> None:
        """The root mail's X-MailFrom is code@ffmpeg.org; forwarding
        it would echo the forge's own notification back onto the PR."""
        _, post = self.run_main()
        self.assertEqual(len(post.call_args_list), 1)
        self.assertIn(
            re.search(r"^From:[ \t]*(.+?)\s+via", (FIX / "human_reply.eml")
                      .read_text(), re.M).group(1),
            post.call_args.args[4])

    def test_a_widened_regex_also_silences_the_human_reply(self) -> None:
        _, post = self.run_main("--forge-bot-sender", "ffmpeg-devel@")
        self.assertEqual(self.posted_numbers(post), [])

    def test_a_narrow_regex_stops_guarding_against_the_loop(self) -> None:
        """The forge's own notification is only held back by this regex:
        with a regex that does not match it, the loop guard no longer
        fires on it (it is then a thread root with no parent)."""
        self.assertIn("skip:forge-bot-or-self=1", self.decisions())
        self.assertNotIn("skip:forge-bot-or-self",
                         self.decisions("--forge-bot-sender", "^nobody@"))


class ForgeFlavorOptionTests(MailFairyMainCase):
    def test_an_unknown_flavor_is_a_config_error(self) -> None:
        rc, post = self.run_main("--forge-flavor", "bitbucket")
        self.assertEqual(rc, 2)
        post.assert_not_called()

    def test_the_override_replaces_the_flavor_derived_from_forge_type(self) -> None:
        """github subject/URL shapes do not match a Forgejo
        notification, so the override shows up as an unresolved thread."""
        _, post = self.run_main("--forge-flavor", "github")
        self.assertEqual(self.posted_numbers(post), [])

    def test_an_untested_flavor_warns(self) -> None:
        with self.assertLogs(mail_fairy.logger, level="WARNING") as logs:
            self.run_main("--forge-flavor", "gitlab")
        self.assertIn("UNTESTED", "\n".join(logs.output))


class StateFileOptionTests(MailFairyMainCase):
    def seed_state(self, msgid: str) -> None:
        state = mail_fairy._empty_state()
        mail_fairy.record_forwarded(state, msgid=msgid,
                                    target_url="https://forge/pr/22883",
                                    posted_at=0.0)
        self.state_file.write_bytes(pickle.dumps(state))

    def test_a_known_message_id_is_not_forwarded_twice(self) -> None:
        self.seed_state(REPLY_MSGID)
        _, post = self.run_main()
        self.assertEqual(self.posted_numbers(post), [])

    def test_a_post_records_the_message_id_in_the_named_file(self) -> None:
        _, post = self.run_main()
        self.assertEqual(self.posted_numbers(post), [22883])
        state = pickle.loads(self.state_file.read_bytes())
        self.assertIn(REPLY_MSGID, state["forwarded"])

    def test_another_state_file_does_not_dedup_the_same_mail(self) -> None:
        self.seed_state(REPLY_MSGID)
        self.state_file = self.tmp / "other.pkl"
        _, post = self.run_main()
        self.assertEqual(self.posted_numbers(post), [22883])

    def test_dry_run_records_nothing(self) -> None:
        _, post = self.run_main("--dry-run")
        post.assert_not_called()
        self.assertFalse(self.state_file.exists())


if __name__ == "__main__":
    unittest.main()
