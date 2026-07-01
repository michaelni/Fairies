"""``--simulate-past`` reads PR head/base from --patch-repo refs.

In sim mode the operator preps a separate cutoff-aware mirror with
PR refs pointing at historical heads. The bot must trust those
refs verbatim --- a ``git rev-list --before`` walk on the live
mirror would silently land on the wrong commit once the historical
head leaves the reachable set (force-push, ``git gc``).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fairy  # noqa: E402


def _git(repo: Path, *args: str) -> str:
    env = {
        "GIT_AUTHOR_NAME": "T", "GIT_AUTHOR_EMAIL": "t@x",
        "GIT_COMMITTER_NAME": "T", "GIT_COMMITTER_EMAIL": "t@x",
        "GIT_AUTHOR_DATE": "2026-05-01T12:00:00+0000",
        "GIT_COMMITTER_DATE": "2026-05-01T12:00:00+0000",
        "PATH": "/usr/bin:/bin:/usr/local/bin",
    }
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, check=True, text=True, env=env,
    ).stdout


@unittest.skipUnless(shutil.which("git"), "git required")
class SimulatePastPrepRefTests(unittest.TestCase):
    def test_uses_prepped_ref_after_force_push(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pr-fairy-sim-") as tmp:
            repo = Path(tmp) / "r"
            repo.mkdir()
            _git(repo, "init", "--initial-branch=master", "--quiet")
            (repo / "f").write_text("0\n")
            _git(repo, "add", "f")
            _git(repo, "commit", "-m", "M0", "--quiet")
            m0 = _git(repo, "rev-parse", "HEAD").strip()
            _git(repo, "checkout", "-b", "topic", "--quiet")
            (repo / "f").write_text("p_old\n")
            _git(repo, "commit", "-am", "P_old", "--quiet")
            p_old = _git(repo, "rev-parse", "HEAD").strip()
            # Operator prep: pin the historical head at the ref the
            # bot will read from, then force-push topic to a wholly
            # different commit to model what may happen upstream.
            _git(repo, "update-ref", "refs/heads/fforge/pr/42", p_old)
            _git(repo, "checkout", "master", "--quiet")
            _git(repo, "branch", "-D", "topic")
            _git(repo, "checkout", "-b", "topic", "--quiet")
            (repo / "f").write_text("p_new\n")
            _git(repo, "commit", "-am", "P_new", "--quiet")
            self.assertNotEqual(_git(repo, "rev-parse", "HEAD").strip(), p_old)

            args = SimpleNamespace(
                simulate_past=datetime(2026, 5, 1, tzinfo=timezone.utc),
                patch_repo=repo,
                patch_pr_ref_template="fforge/pr/{number}",
            )
            base, head = fairy.patch_shas_for_run(
                args, {"number": 42, "base": {"ref": "master"}},
            )
            self.assertEqual((base, head), (m0, p_old))


class PatchShasForRunLiveTests(unittest.TestCase):
    def test_live_mode_uses_pr_object_unchanged(self) -> None:
        args = SimpleNamespace(simulate_past=None)
        pr = {
            "merge_base": "aaa", "base": {"sha": "bbb"}, "head": {"sha": "ccc"},
        }
        self.assertEqual(
            fairy.patch_shas_for_run(args, pr), ("aaa", "ccc"),
        )


if __name__ == "__main__":
    unittest.main()
