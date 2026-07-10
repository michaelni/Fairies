#!/bin/bash

set -euo pipefail

cd ~/forgejo_fairy

cd ffmpeg
git checkout master
git fetch fforge
git pull --rebase
cd ..

# Issue helper: repro / bisect happens inside the podman container (add
# --podman-host USER@HOST), duplicate search uses the exported-issue
# vector store. Triage runs on the cheap mini model, the full pass on
# GPT-5.5@high; both on the flex tier. Dry-run by default; add
# --approve or --manual to submit.
./issue_fairy.py --owner FFmpeg --repo FFmpeg --gcli-account ff --issue-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/fixed' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --extra-repo-root all_ffmpeg --triage-model openai:gpt-5.4-mini --triage-service-tier flex --model openai:gpt-5.5 --reasoning-effort high --service-tier flex --use-vector-store-search --verbose --debug-response-dir openaidebug --use-web-search --max-tool-calls 100 ' --llm-parallelism 3 --verbose 2 $*
