#!/usr/bin/env python3
"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License version 2 for more details.
 *
 * Additional permission:
 *
 * Michael Niedermayer is permitted to relicense this file, in whole or
 * in part, under any version of the GNU General Public License, the GNU
 * Affero General Public License, or the GNU Lesser General Public License
 * published by the Free Software Foundation.
 *
 * This additional permission is personal to Michael Niedermayer.  It is
 * not transferable and does not grant any other person permission to
 * relicense this file under a different license.
 *
 * This additional permission may be removed from modified copies of this
 * file.  Removal of this additional permission does not affect the
 * licensing of the file under the GNU General Public License version 2.
 */

Diff the assembled LLM prompts of two git revisions.

Each revision is exported with ``git archive`` into a temp dir, and this
script re-invokes itself with that checkout first on sys.path, so the
revision's own code generates every prompt into one text file per
prompt: the developer prompts of all roles (review, code_review,
design_review, combiner, triager, issue_investigator, issue_combiner,
issue_triager, plus the CI-failure variants) and the user-text builders.
The two directories are then compared with ``git diff --no-index``.
All generation inputs (model, features, repos, machines, fixtures) are
fixed by the invoking script, so the diff shows exactly what the code
change changed.  The deployment's repo names come from the ``--inputs``
JSON file (default: prompt_diff_inputs.json next to this script); the
shipped one names the repo pair that turns on every conditional prompt
section (spec store, forge export, FATE, recollq).

Usage:
  tools/prompt_diff.py [--color[=WHEN]] [--word-diff] [--keep] [-v] REV1 [REV2]

With a single revision the generated prompts are printed instead of
diffed.

Exit status is git diff's: 0 when all prompts are identical, 1 when they
differ; with a single revision it is 0.
"""

from __future__ import annotations

import argparse
import inspect
import io
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import traceback
from pathlib import Path

logger = logging.getLogger(__name__)

ROLES = ("review", "code_review", "design_review", "combiner", "triager",
         "issue_investigator", "issue_combiner", "issue_triager")
CI_ROLES = ("review", "combiner", "triager")

MODEL = "openai:gpt-5.5"
FEATURES = frozenset({"source_bundle", "vector_store_search", "web_search",
                      "code_interpreter", "podman_shell"})
MACHINE_SPECS = ("x86=fairy@192.0.2.1,gpu=nvidia.com/gpu=0",
                 "arm64=fairy@192.0.2.2")
ALLOWED_MODELS = ["gpt-5.5", "claude-opus-5"]
REVIEWER_USERNAME = "fairy"

PR_REQUEST = {
    "reviewer_username": REVIEWER_USERNAME,
    "pull_request": {
        "number": 12345,
        "title": "avcodec/example: fix overflow",
        "author": "alice",
        "html_url": "https://example.com/pr/12345",
        "base_ref": "master",
        "head_ref": "fix-overflow",
        "head_sha": "0123456789abcdef0123456789abcdef01234567",
        "additions": 10,
        "deletions": 2,
        "changed_files": 1,
        "labels": ["fix/bug"],
    },
    "discussion": [
        {"kind": "comment", "author": "bob", "body": "please rebase"},
    ],
}

ISSUE_REQUEST = {
    "reviewer_username": REVIEWER_USERNAME,
    "issue": {
        "number": 4242,
        "title": "crash decoding example.mkv",
        "author": "alice",
        "html_url": "https://example.com/issues/4242",
        "created_at": "2026-01-01T00:00:00Z",
        "labels": ["bug"],
        "attachment_urls": [],
    },
    "discussion": [],
}


def dump_prompts(checkout: Path, outdir: Path, project_facts: str,
                 inputs_path: Path) -> int:
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))

    def write(name: str, text: str) -> None:
        (outdir / (name + ".txt")).write_text(
            text.replace(str(checkout), "<checkout>"),
            encoding="utf-8", newline="\n")

    try:
        import llm_prompt
    except Exception:
        write("llm_prompt_error", "ERROR importing llm_prompt:\n"
              + traceback.format_exc())
        return 0

    def generate(name: str, thunk) -> None:
        try:
            text = thunk()
        except Exception:
            logger.warning("generating %s failed", name)
            text = f"ERROR generating {name}:\n{traceback.format_exc()}"
        write(name, text)

    try:
        import podman_host
        machines = [podman_host.parse_shell_host(s) for s in MACHINE_SPECS]
    except Exception:
        machines = []
        write("machines_error", "ERROR parsing machine specs:\n"
              + traceback.format_exc())

    facts = ""
    if project_facts:
        try:
            facts = llm_prompt.load_project_facts(checkout / project_facts)
        except Exception:
            write("project_facts_error", f"ERROR loading {project_facts}:\n"
                  + traceback.format_exc())

    common_kwargs = dict(
        vendor="openai", model=MODEL, features=FEATURES,
        repo_roots=[Path(p) for p in inputs["repo_roots"]],
        container_repo_mounts=list(inputs["container_repo_mounts"]),
        reviewer_username=REVIEWER_USERNAME, project_facts=facts,
        allowed_models=ALLOWED_MODELS,
        allowed_labels=sorted(getattr(llm_prompt, "TRIAGE_LABEL_DEFINITIONS", {})),
        machines=machines,
    )
    params = inspect.signature(llm_prompt.generate_llm_prompt).parameters
    dropped = (set(common_kwargs) | {"role", "ci_triage_mode"}) - set(params)
    if dropped:
        logger.warning("this revision's generate_llm_prompt does not accept:"
                       " %s", sorted(dropped))
    for role in ROLES:
        for ci in (False, True) if role in CI_ROLES else (False,):
            kwargs = {k: v for k, v in
                      {**common_kwargs, "role": role, "ci_triage_mode": ci}.items()
                      if k in params}
            generate(role + "+ci" * ci,
                     lambda kw=kwargs: llm_prompt.generate_llm_prompt(**kw))

    generate("user_review", lambda: llm_prompt.make_user_text(
        PR_REQUEST, ["2 of 3 changed files attached"],
        ["libavcodec/example.c"], False))
    generate("user_triage",
             lambda: llm_prompt.make_triage_user_text(PR_REQUEST, False))
    generate("user_issue", lambda: llm_prompt.make_issue_user_text(ISSUE_REQUEST))

    def combiner_drafts() -> str:
        from llm_review_api import Review
        return llm_prompt.make_combiner_user_text([
            Review(classification="moderate_issues", message="Draft body A",
                   model="openai:gpt-5.5", prompt="code_review"),
            Review(classification="approve", message="Draft body B",
                   model="anthropic:claude-opus-5", prompt="design_review"),
        ])
    generate("user_combiner_drafts", combiner_drafts)
    return 0


def export_revision(repo: Path, rev: str, dest: Path) -> str:
    sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--verify", rev + "^{commit}"],
        check=True, stdout=subprocess.PIPE, text=True).stdout.strip()
    cmd = ["git", "-C", str(repo), "archive", "--format=tar", sha]
    logger.info("exporting %s (%s): %s", rev, sha[:12], shlex.join(cmd))
    archive = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    dest.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(dest, filter="data")
    return sha


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Diff the assembled LLM prompts of two git revisions,"
                    " or print one revision's prompts.")
    ap.add_argument("revisions", nargs="*", metavar="REV")
    ap.add_argument("--color", nargs="?", const="always", default="auto",
                    choices=("auto", "always", "never"))
    ap.add_argument("--word-diff", action="store_true")
    ap.add_argument("--repo", type=Path,
                    default=Path(__file__).resolve().parent.parent,
                    help="repository to export the revisions from")
    ap.add_argument("--project-facts", default="",
                    help="repo-relative facts file to splice into the prompts")
    ap.add_argument("--inputs", type=Path,
                    default=Path(__file__).resolve().with_name("prompt_diff_inputs.json"),
                    help="JSON file with the deployment's repo names")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp dir with the generated prompt files")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--dump", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--checkout", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if not args.dump and len(args.revisions) not in (1, 2):
        ap.error("one or two revisions required")

    root = args.checkout if args.dump else Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root))
    from common import setup_logging
    if "color" in inspect.signature(setup_logging).parameters:
        setup_logging(logger, args.verbose, color=args.color)
    else:
        setup_logging(logger, args.verbose)

    if args.dump:
        return dump_prompts(args.checkout, args.dump, args.project_facts,
                            args.inputs)

    tmp = Path(tempfile.mkdtemp(prefix="prompt_diff_"))
    try:
        labels = [re.sub(r'[\\/:*?"<>|\s]+', "_", rev) for rev in args.revisions]
        if len(labels) == 2 and labels[0] == labels[1]:
            labels = [labels[0] + "-a", labels[1] + "-b"]
        prompt_dirs = []
        for rev, label in zip(args.revisions, labels):
            src = tmp / ("src-" + label)
            sha = export_revision(args.repo, rev, src)
            out = tmp / label
            out.mkdir()
            cmd = ([sys.executable, str(Path(__file__).resolve()),
                    "--dump", str(out), "--checkout", str(src),
                    "--inputs", str(args.inputs)]
                   + ["--project-facts", args.project_facts] * bool(args.project_facts)
                   + ["--verbose"] * args.verbose)
            logger.info("generating prompts for %s (%s): %s",
                        rev, sha[:12], shlex.join(cmd))
            subprocess.run(cmd, check=True)
            prompt_dirs.append(out)

        if len(prompt_dirs) == 1:
            for prompt_file in sorted(prompt_dirs[0].iterdir()):
                print(f"======== {prompt_file.stem} ========")
                print(prompt_file.read_text(encoding="utf-8"))
            return 0

        diff_cmd = (["git", "diff", "--no-index", f"--color={args.color}"]
                    + ["--word-diff"] * args.word_diff
                    + [d.name for d in prompt_dirs])
        logger.info("diff: %s", shlex.join(diff_cmd))
        rc = subprocess.run(diff_cmd, cwd=tmp).returncode
        if rc > 1:
            logger.error("git diff failed with status %d", rc)
        return rc
    except subprocess.CalledProcessError as e:
        logger.error("%s failed with status %d", shlex.join(map(str, e.cmd)), e.returncode)
        return 2
    finally:
        if args.keep:
            logger.info("kept %s", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
