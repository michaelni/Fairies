#!/bin/bash

set -euo pipefail

cd ~/forgejo_fairy

cd ffmpeg
git checkout master
git fetch fforge
git pull --rebase
cd ..

# Issue helper: repro / bisect happens inside the podman container (add
# --podman-host 'x86_64=USER@HOST,gpu=nvidia.com/gpu=0'), duplicate search
# uses the exported-issue
# vector store. Triage runs on GPT-5.6-luna@medium, the full pass on
# GPT-5.6@high; both on the flex tier. Dry-run by default; add
# --approve or --manual to submit.
./issue_fairy.py --owner FFmpeg --repo FFmpeg --gcli-account ff --issue-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --extra-repo-root all_ffmpeg --triage-model openai:gpt-5.6-luna --triage-service-tier flex --model openai:gpt-5.6@high --service-tier flex --use-vector-store-search --verbose --debug-response-dir openaidebug --web-search live --max-tool-calls 100 ' --llm-parallelism 3 --verbose 2 $*
