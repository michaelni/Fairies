"""Source-bundle files are the PR's predicted merge result.

Sending PR-head file contents while the vector store / container
checkout sit at the target tip made models (gpt-5.4, glm-5.2,
gpt-5.6-terra on PR #23720) conclude the PR "removes" newer target
work. The bundle must therefore contain the merge of the PR head into
the current target tip, labeled as such, and fall back to PR-commit
contents with a note when the merge conflicts.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pr_review_wrapper as wrapper  # noqa: E402


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


BASE = "".join(f"line{i}\n" for i in range(20))


@unittest.skipUnless(shutil.which("git"), "git required")
class SourceBundleMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="pr-fairy-bundle-")
        self.addCleanup(self._tmp.cleanup)
        self.repo = Path(self._tmp.name) / "r"
        self.repo.mkdir()
        _git(self.repo, "init", "--initial-branch=master", "--quiet")
        (self.repo / "f.c").write_text(BASE)
        _git(self.repo, "add", "f.c")
        _git(self.repo, "commit", "-m", "base", "--quiet")
        _git(self.repo, "checkout", "-b", "pr", "--quiet")
        (self.repo / "f.c").write_text(BASE.replace("line2\n", "pr_change\n"))
        _git(self.repo, "commit", "-am", "pr work", "--quiet")
        self.head_sha = _git(self.repo, "rev-parse", "HEAD").strip()
        self.patch = _git(self.repo, "format-patch", "--stdout", "master..pr")
        _git(self.repo, "checkout", "master", "--quiet")

    def _bundle(self) -> tuple[str, list[str], list[str]]:
        request = {
            "pull_request": {"head_sha": self.head_sha, "base_ref": "master"},
            "patch": self.patch,
        }
        return wrapper.build_source_bundle(
            request, self.repo,
            max_source_files=5, max_file_bytes=10000,
            max_header_file_bytes=10000, max_bundle_bytes=100000,
            include_direct_includes=False, verbose=False,
        )

    def test_bundle_contains_merge_of_head_and_target_tip(self) -> None:
        # master moved on after the merge base, touching a distant line
        (self.repo / "f.c").write_text(BASE.replace("line17\n", "master_change\n"))
        _git(self.repo, "commit", "-am", "master work", "--quiet")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)
        self.assertIn("pr_change", bundle)
        self.assertIn("master_change", bundle)
        self.assertIn("merge result of the PR head", bundle)
        self.assertIn(f"merge of PR head {self.head_sha} into master", bundle)

    def test_conflicting_merge_falls_back_to_pr_commits(self) -> None:
        # master rewrote the same line the PR changes: merge conflicts
        (self.repo / "f.c").write_text(BASE.replace("line2\n", "master_conflict\n"))
        _git(self.repo, "commit", "-am", "master work", "--quiet")

        bundle, used, notes = self._bundle()
        self.assertEqual(["f.c"], used)
        self.assertIn("pr_change", bundle)
        self.assertNotIn("master_conflict", bundle)
        self.assertIn(f"git show {self.head_sha}:f.c", bundle)
        self.assertTrue(any("does not merge cleanly" in n for n in notes))


if __name__ == "__main__":
    unittest.main()
