"""Opt-in end-to-end check: a real model, given the production shell
cookbook, finds an issue and a spec PDF through the recoll index
without struggling (bounded tool rounds, at least one successful
recollq).

Costs real API money and starts a real review container, so it skips
unless configured:

    FAIRY_RECOLL_LIVE_SSH=fairy@host \\
    FAIRY_RECOLL_LIVE_REPOS=/path/ffmpeg:/path/all_ffmpeg \\
    python3 -m unittest tests.test_recoll_llm_live -v

Optional: FAIRY_RECOLL_LIVE_MODEL (default gpt-5.6),
FAIRY_RECOLL_LIVE_IMAGE (default localhost/fairy-review:recoll-live).
The image is (re)built on the podman host from the checked-out
Containerfile; the OpenAI key comes from the environment or .env.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTAINERS_DIR = REPO_ROOT / "containers"
for p in (str(REPO_ROOT), str(CONTAINERS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

import llm_prompt  # noqa: E402
import podman_host  # noqa: E402
import podman_repos  # noqa: E402
import pr_review_wrapper  # noqa: E402
import provision_remote  # noqa: E402
from openai_common import extract_response_text, load_api_key  # noqa: E402
import openai_reviewer  # noqa: E402

SSH = os.environ.get("FAIRY_RECOLL_LIVE_SSH")
REPOS = os.environ.get("FAIRY_RECOLL_LIVE_REPOS")
MODEL = os.environ.get("FAIRY_RECOLL_LIVE_MODEL", "gpt-5.6")
IMAGE = os.environ.get("FAIRY_RECOLL_LIVE_IMAGE", "localhost/fairy-review:recoll-live")
INDEX_WAIT_S = 900.0
MAX_TOOL_ROUNDS = 8


def _issue_ground_truth(all_ffmpeg_root: Path) -> tuple[str, str]:
    """Pick the largest exported issue and a distinctive phrase from it."""
    issues = sorted((all_ffmpeg_root / "forgejo_git" / "issues").glob("[0-9]*.md"),
                    key=lambda p: p.stat().st_size)
    body = issues[-1].read_text(errors="replace")
    phrase = max((line.strip() for line in body.splitlines()), key=len)
    return issues[-1].stem, phrase[:200]


@unittest.skipUnless(SSH and REPOS and load_api_key(),
                     "live test: set FAIRY_RECOLL_LIVE_SSH, "
                     "FAIRY_RECOLL_LIVE_REPOS and an OpenAI key")
class RecollLiveLLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.machine = podman_host.ShellHostSpec("x86_64", podman_host.RemoteHost(SSH))
        provision_remote.ensure_image(
            host=cls.machine.host, tag=IMAGE,
            dockerfile=CONTAINERS_DIR / "Containerfile",
            context_dir=CONTAINERS_DIR, rebuild=False,
        )
        cls.repo_roots = [Path(p).expanduser() for p in REPOS.split(":")]
        specs = podman_repos.build_repo_specs(cls.repo_roots)
        args = argparse.Namespace(
            podman_image=IMAGE, podman_network=None,
            simulate_past_cutoff=None, podman_exec_timeout=120.0,
        )
        cls.handle, cls.session, _ = pr_review_wrapper.open_review_container_shell(
            cls.machine, specs, args)
        cls.mounts = [s.container_path for s in specs]
        deadline = time.monotonic() + INDEX_WAIT_S
        while cls.session.exec("test -f /root/.recoll/index.done").exit_code:
            if time.monotonic() > deadline:
                podman_host.stop_container(cls.handle)
                raise AssertionError(f"index not done within {INDEX_WAIT_S}s")
            time.sleep(10)

    @classmethod
    def tearDownClass(cls) -> None:
        if hasattr(cls, "handle"):
            podman_host.stop_container(cls.handle)

    def _ask(self, question: str) -> tuple[str, list]:
        """Run one tool-loop conversation; return (final text, shell calls)."""
        cookbook = llm_prompt.generate_llm_prompt(
            role="reviewer", vendor="openai", model=MODEL,
            features={"podman_shell"}, repo_roots=self.repo_roots,
            container_repo_mounts=self.mounts, reviewer_username="fairy",
            machines=[self.machine],
        )
        calls: list[tuple[str, object]] = []
        session, orig_exec = type(self).session, type(self).session.exec

        def spy(command: str, **kw: object) -> object:
            result = orig_exec(command, **kw)
            calls.append((command, result))
            return result

        client = openai_reviewer.OpenAI(
            api_key=load_api_key(), timeout=None, max_retries=0,
            http_client=openai_reviewer.make_openai_http_client())
        session.exec = spy
        try:
            response = openai_reviewer.run_responses_resolving_podman_shell(
                client,
                initial_kwargs={
                    "model": MODEL,
                    "input": [{"role": "developer", "content": cookbook},
                              {"role": "user", "content": question}],
                    "tools": [openai_reviewer.build_podman_shell_function_tool(
                        [self.machine])],
                },
                shells={self.machine.label: session},
                machine_labels=[self.machine.label],
                open_shell=lambda label: (session, ""),
                max_tool_rounds=MAX_TOOL_ROUNDS,
                max_shell_timeout_s=120.0,
                what="recoll live test", verbose=True,
            )
        finally:
            session.exec = orig_exec
        return extract_response_text(response), calls

    def _assert_recollq_used_well(self, calls: list) -> None:
        self.assertLessEqual(len(calls), MAX_TOOL_ROUNDS, "model struggled")
        self.assertTrue(
            any("recollq" in cmd and not res.exit_code and res.stdout.strip()
                for cmd, res in calls),
            "no successful recollq call in: "
            + " | ".join(cmd for cmd, _ in calls))

    def test_finds_issue_by_phrase(self) -> None:
        all_ffmpeg = next(r for r in self.repo_roots if r.name == "all_ffmpeg")
        number, phrase = _issue_ground_truth(all_ffmpeg)
        text, calls = self._ask(
            "Using the recoll full-text index (recollq), find the FFmpeg "
            f'Forgejo issue whose discussion contains: "{phrase}". '
            "Reply with only the 6-digit issue number.")
        self._assert_recollq_used_well(calls)
        self.assertIn(number, text)

    def test_finds_spec_pdf(self) -> None:
        text, calls = self._ask(
            "Which specification document available in the container "
            'contains the phrase "chromaticity coordinates of the source '
            'colour primaries"? Reply with the document file name(s).')
        self._assert_recollq_used_well(calls)
        self.assertTrue(
            any(t in text for t in ("H.273", "H.274", "23001")),
            f"unexpected answer: {text!r}")


if __name__ == "__main__":
    unittest.main()
