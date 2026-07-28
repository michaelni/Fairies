"""Ratchet: forge knowledge belongs to the modules that own it.

``forge_gcli`` decides which forge is being talked to and what its
payloads mean; everything downstream is handed one documented shape and
should not know a forge exists. This test counts the modules that name a
forge and lets the number only shrink, so a new ``if forge_type ==`` or a
new "Forgejo does X" aside in general code has to be argued for rather
than merged quietly.

Exempt, because naming a forge IS their job:

* ``forge_gcli.py``    -- owns the branch and the payload projections
* ``mail_fairy.py``    -- owns the per-forge notification-mail flavors
* ``forgejo_export.py``-- the exporter, named for what it exports
* ``ci_log.py``        -- fetches a Forgejo Actions job log by URL shape
* ``tools/capture_github_fixtures.py`` -- records the GitHub fixtures
* ``llm_prompt.py``    -- prompt text, exempt under project-agnostic.mdc

The count is a ceiling, not a target: the remaining mentions are mostly
comments recording forge behavior, which are worth keeping. Lower the
baseline when you remove one; never raise it.
"""

import re
import shutil
import subprocess
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

FORGE_NAME = re.compile(r"forgejo|gitea|github|gitlab", re.IGNORECASE)
OWNS_FORGE_KNOWLEDGE = {
    "forge_gcli.py",
    "mail_fairy.py",
    "forgejo_export.py",
    "ci_log.py",
    "llm_prompt.py",
    "tools/capture_github_fixtures.py",
}
BASELINE = 62  # lower this when cleaning existing mentions up


def _mentions() -> list[str]:
    files = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
        check=True, capture_output=True, text=True,
    ).stdout.split()
    return [
        f"{name}:{i}: {line.strip()}"
        for name in files
        if not name.startswith("tests/") and name not in OWNS_FORGE_KNOWLEDGE
        for i, line in enumerate(
            (REPO_ROOT / name).read_text(encoding="utf-8").splitlines(), 1)
        if FORGE_NAME.search(line)
    ]


@unittest.skipUnless(shutil.which("git"), "git required")
class ForgeNeutralityRatchetTests(unittest.TestCase):

    def test_forge_names_in_general_code_only_shrink(self) -> None:
        hits = _mentions()
        self.assertLessEqual(
            len(hits), BASELINE,
            f"{len(hits)} forge mentions in general code, baseline "
            f"{BASELINE}. Forge knowledge belongs in forge_gcli, which "
            f"hands the rest of the tree one documented shape.\n"
            + "\n".join(hits),
        )

    def test_no_forge_dispatch_outside_the_modules_that_own_it(self) -> None:
        # A comment naming a forge is a fact about the world; a branch on
        # forge_type is a second implementation of forge_gcli's job.
        dispatch = [h for h in _mentions()
                    if re.search(r"forge_type\s*==|==\s*[\"']"
                                 r"(github|gitlab|gitea|forgejo)", h)]
        self.assertEqual([], dispatch)


if __name__ == "__main__":
    unittest.main()
