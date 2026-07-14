"""--config: TOML defaults with @argsfile semantics (explicit CLI wins)."""
import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common import apply_config_file_defaults


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--alpha")
    p.add_argument("--num-things", type=float, default=1.0)
    p.add_argument("--facts-path", type=Path)
    p.add_argument("--flag", action="store_true")
    p.add_argument("--item", action="append", default=[])
    return p


def _parse(toml: str, argv: list[str]) -> argparse.Namespace:
    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as fh:
        fh.write(toml)
    p = _parser()
    full_argv = ["--config", fh.name, *argv]
    apply_config_file_defaults(p, full_argv)
    return p.parse_args(full_argv)


class ConfigFileTests(unittest.TestCase):
    def test_config_fills_unset_cli_wins(self) -> None:
        args = _parse(
            '# comment\nalpha = "from-config"\nnum-things = 3.5\n',
            ["--alpha", "from-cli"],
        )
        self.assertEqual("from-cli", args.alpha)
        self.assertEqual(3.5, args.num_things)

    def test_underscore_keys_and_bool(self) -> None:
        args = _parse('num_things = 2.0\nflag = true\n', [])
        self.assertEqual(2.0, args.num_things)
        self.assertTrue(args.flag)

    def test_string_values_go_through_type(self) -> None:
        args = _parse('facts-path = "facts/x.md"\nnum-things = "4"\n', [])
        self.assertEqual(Path("facts/x.md"), args.facts_path)
        self.assertEqual(4.0, args.num_things)

    def test_append_option_accumulates_cli_onto_config(self) -> None:
        args = _parse('item = ["a", "b"]\n', ["--item", "c"])
        self.assertEqual(["a", "b", "c"], args.item)

    def test_unknown_key_errors(self) -> None:
        with self.assertRaises(SystemExit), \
                contextlib.redirect_stderr(io.StringIO()):
            _parse('no-such-option = 1\n', [])

    def test_no_config_flag_is_noop(self) -> None:
        p = _parser()
        apply_config_file_defaults(p, ["--alpha", "x"])
        self.assertEqual("x", p.parse_args(["--alpha", "x"]).alpha)


if __name__ == "__main__":
    unittest.main()
