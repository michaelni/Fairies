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
# vector store. Dry-run by default; add --approve or --manual.
./issue_fairy.py --owner FFmpeg --repo FFmpeg --gcli-account ff --issue-label 'duplicate,invalid,needs sample,fix/bug,fix/regression,important' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.4 --extra-repo-root all_ffmpeg --use-vector-store-search --reasoning-effort high --verbose --debug-response-dir openaidebug --use-web-search --max-tool-calls 100 ' --verbose 2 --min-age-days 1 $*
