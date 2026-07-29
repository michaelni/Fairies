"""Pin local PR-patch synthesis (replaces the Forgejo .patch URL fetch).

Two things can regress here:

* ``git_util.git_format_patch_series`` actually shells out to git and
  returns the canonical mbox shape (``From <sha>`` per commit) the
  wrapper parses.
* ``fetch_patch_for_llm`` honors the ``max_bytes`` cap and reports
  truncation in its second tuple element.

The merge-base / sim-past SHA selection is pinned in
``test_simulate_past_pr_shas.py``. The wrapper's own commit/path
parsers are tested elsewhere; we do not re-test their grammar here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import git_util  # noqa: E402
import fairy  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@x",
        "GIT_AUTHOR_DATE": "2026-05-01T12:00:00+00:00",
        "GIT_COMMITTER_DATE": "2026-05-01T12:00:00+00:00",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True, text=True, env=env,
    ).stdout


@unittest.skipUnless(shutil.which("git"), "git required")
class SynthesizePRPatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pr-fairy-patch-")
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "r"
        self.repo.mkdir()
        _git(self.repo, "init", "--initial-branch=master", "--quiet")
        (self.repo / "f.c").write_text("int f(void){return 1;}\n")
        _git(self.repo, "add", "f.c")
        _git(self.repo, "commit", "-m", "base", "--quiet")
        self.base_sha = _git(self.repo, "rev-parse", "HEAD").strip()
        (self.repo / "f.c").write_text("int f(void){return 2;}\n")
        _git(self.repo, "commit", "-am", "bump", "--quiet")
        self.head_sha = _git(self.repo, "rev-parse", "HEAD").strip()

    def test_format_patch_series_has_From_header_and_diff(self) -> None:
        out = git_util.git_format_patch_series(
            self.repo, self.base_sha, self.head_sha,
        )
        text = out.decode("utf-8")
        self.assertIn(f"From {self.head_sha}", text)
        self.assertIn("+++ b/f.c", text)

    def test_format_patch_unknown_sha_raises_with_repo_in_message(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            git_util.git_format_patch_series(
                self.repo, "deadbeef" * 5, self.head_sha,
            )
        self.assertIn(str(self.repo), str(ctx.exception))

    def test_a_head_pushed_after_the_last_mirror_fetch_self_heals(self) -> None:
        """A retry right after a push must not stay broken until the
        next scheduled mirror fetch: the missing head triggers one
        fetch and a second format-patch try."""
        mirror = Path(self._tmp.name) / "mirror"
        _git(Path(self._tmp.name), "clone", "--quiet",
             str(self.repo), str(mirror))
        (self.repo / "f.c").write_text("int f(void){return 3;}\n")
        _git(self.repo, "commit", "-am", "pushed after fetch", "--quiet")
        new_head = _git(self.repo, "rev-parse", "HEAD").strip()
        args = SimpleNamespace(patch_repo=mirror)
        text, truncated = fairy.fetch_patch_for_llm(
            args, self.head_sha, new_head, 10_000,
        )
        self.assertIn(f"From {new_head}", text)
        self.assertFalse(truncated)

    def test_fetch_patch_for_llm_truncates_and_flags(self) -> None:
        args = SimpleNamespace(patch_repo=self.repo)
        text, truncated = fairy.fetch_patch_for_llm(
            args, self.base_sha, self.head_sha, 50,
        )
        self.assertTrue(truncated)
        self.assertLessEqual(len(text.encode("utf-8")), 50)


if __name__ == "__main__":
    unittest.main()
