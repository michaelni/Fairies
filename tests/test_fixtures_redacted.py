"""
Every tracked fixture carries values tools/redact.py drew.

The check is the tool's own ``--check``, over the whole of
tests/fixtures: a token has to be a pool member or sit on the exception
list, so a fixture added as captured fails here. Held back only by a
``path:`` line in tools/redact_exceptions.txt, which is where a
hand-written fixture goes -- its values were never drawn from a pool and
never will be.

Passing the files rather than the directory is what makes a new fixture
directory fail closed: nothing is in scope by being somewhere the tool
was not pointed at.
"""
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
import redact  # noqa: E402


def _files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and "__pycache__" not in p.parts)

class FixtureValuesTests(unittest.TestCase):
    def test_fixtures_carry_drawn_values(self) -> None:
        names = _files(REPO_ROOT / "tests" / "fixtures")
        self.assertTrue(names, "no fixtures found to check")

        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            status = redact.main([str(n) for n in names] + ["--check"])
        self.assertEqual(
            status, 0,
            "fixture value(s) not drawn by tools/redact.py; run it over the "
            "file, or add a path: line to tools/redact_exceptions.txt if the "
            "fixture is hand-written:\n" + said.getvalue(),
        )

    def test_names_are_drawn_or_conventional(self) -> None:
        names = _files(REPO_ROOT / "tests")
        self.assertTrue(names)
        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            status = redact.main([str(n) for n in names] + ["--names"])
        self.assertEqual(
            status, 0,
            "name(s) not drawn, listed or conventional. Use a pool word "
            "or the Alice-and-Bob cast (alice, bob, jane doe, ...):\n"
            + said.getvalue(),
        )
    def test_redaction_is_a_fixed_point_on_its_output(self) -> None:
        names = _files(REPO_ROOT / "tests" / "fixtures")
        self.assertTrue(names)
        with tempfile.TemporaryDirectory() as tmp:
            copies = []
            for name in names:
                dst = Path(tmp) / name.relative_to(REPO_ROOT)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(name, dst)
                copies.append(dst)
            before = {c: c.read_bytes() for c in copies}
            said = io.StringIO()
            with contextlib.redirect_stdout(said), \
                    contextlib.redirect_stderr(said):
                status = redact.main([str(c) for c in copies])
            self.assertEqual(0, status, said.getvalue())
            changed = [str(c) for c in copies
                       if c.read_bytes() != before[c]]
            self.assertEqual([], changed, said.getvalue())

    def test_the_pools_classify_as_themselves(self) -> None:
        self.assertEqual([], redact.Pools().misclassified())


class DetectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokens, _, whole = redact.load_exceptions()
        cls.red = redact.Redactor(redact.Pools(), tokens, whole)

    def found(self, text: str) -> list[str]:
        return redact.check_names(text, self.red)

    def test_digits_before_the_at_are_not_an_id(self) -> None:
        self.assertFalse(
            self.red.keeps(redact.Kind.EMAIL, "12345678@" "example.org"))
        self.assertTrue(self.found("see %s" % ("12345678@" "example.org")))

    def test_digits_elsewhere_still_stand(self) -> None:
        self.assertTrue(self.red.keeps(redact.Kind.NUMBER, "12345678"))
        self.assertTrue(self.red.keeps(
            redact.Kind.URL, "https://code.ffmpeg.org/api/v1/issues/12345678"))

    def test_an_address_written_into_source(self) -> None:
        self.assertTrue(self.found("x = %r" % ("zbqwx@" "nowhere.example")))

    def test_a_display_name_in_front_of_an_address(self) -> None:
        self.assertTrue(
            self.found("From: Zbqwx Vfrpl <%s>" % ("jane@" "example.org")))
        self.assertTrue(
            self.found("From: Zbqwxson, Jane <%s>" % ("jane@" "example.org")))

    def test_a_value_under_a_name_key(self) -> None:
        self.assertTrue(self.found('{"login": %s}' % '"zbqwxvfrpl"'))

    def test_a_keyed_digit_login_is_flagged_and_drawn(self) -> None:
        digits = "31415926"
        self.assertTrue(self.found('{"login": %s}' % f'"{digits}"'))
        tree = redact.redact_json({"log" "in": digits, "id": 31415926},
                                  self.red)
        drawn = tree["log" "in"]
        self.assertNotEqual(digits, drawn)
        again = redact.redact_json({"log" "in": drawn}, self.red)
        self.assertEqual(drawn, again["log" "in"])

    def test_the_shapes_tests_write_logins_in(self) -> None:
        for probe in ('assert data["login"] == %s' % '"zbqwx"',
                      "self.assertEqual(user.login, %s)" % '"zbqwx"',
                      "self.assertEqual(%s, user.login)" % '"zbqwx"',
                      'self.assertEqual(%s, data["login"])' % '"zbqwx"',
                      "if user.login != %s: pass" % '"zbqwx"',
                      'data.get("login") == %s' % '"zbqwx"'):
            self.assertTrue(self.found(probe), probe)
        self.assertEqual([], self.found(
            'login = user.get("login", "unknown-login")'))
        self.assertEqual([], self.found(
            'ap.add_argument("--author", default=cfg.author)'))
        self.assertEqual([], self.found('(Kind.EMAIL, "zbqwx")'))
        self.assertEqual([], self.found('render(x.author, "%Y-%m-%d")'))

    def test_a_name_key_beside_a_login_in_source(self) -> None:
        self.assertTrue(self.found(
            'U = {"login": "alice", "name": %s}' % '"Zbqwx Vfrpl"'))
        self.assertEqual([], self.found(
            '{"name": "never-runs", "status": "queued"}'))

    def test_an_integer_login_is_seen_and_drawn(self) -> None:
        found = self.found('{"login": 271828459, "id": 271828459}')
        self.assertEqual(["log" "in: 271828459"], found)
        tree = redact.redact_json({"log" "in": 271828459, "id": 271828459},
                                  self.red)
        self.assertNotEqual(271828459, tree["log" "in"])
        self.assertEqual(271828459, tree["id"])

    def test_an_unroutable_login_is_still_a_login(self) -> None:
        self.assertTrue(self.found('{"login": %s}' % '"zbqwx@internal"'))
        self.assertEqual([], self.found('{"email": "t@x"}'))

    def test_a_name_beside_an_email_key(self) -> None:
        self.assertTrue(self.found(
            '{"author": {"name": %s, "email": %s}}'
            % ('"Zbqwx Vfrpl"', '"alice@' 'example.org"')))

    def test_a_cast_address_passes_under_an_email_key(self) -> None:
        self.assertEqual([], self.found('{"email": "alice@example.org"}'))

    def test_a_timestamp_under_a_name_key_passes(self) -> None:
        self.assertEqual([], self.found(
            '{"last_login": "0001-01-01T00:00:00Z"}'))

    def test_the_cast_passes_and_is_still_drawn_in_captures(self) -> None:
        self.assertEqual([], self.found(
            '{"login": "alice"}\nFrom: Jane Doe <jane@example.org>'))
        self.assertNotEqual("alice", self.red.token("alice"))

    def test_what_the_pools_drew_passes(self) -> None:
        drawn = self.red.token("zbqwx@" "nowhere.example")
        self.assertEqual([], self.found(drawn))

    def test_only_the_tokens_before_the_address_are_read(self) -> None:
        self.assertEqual([], self.found(
            "maintained since 2019 by Jane Doe <%s>" % ("jane@" "example.org")))

    def test_source_code_is_not_a_capture(self) -> None:
        said = io.StringIO()
        with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
            status = redact.main([__file__, "--check"])
        self.assertEqual(2, status)
        self.assertIn("not a capture", said.getvalue())

if __name__ == "__main__":
    unittest.main()
