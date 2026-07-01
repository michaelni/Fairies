#!/bin/bash

set -euo pipefail

cd ~/forgejo_fairy

cd ffmpeg
git checkout master
git fetch fforge
git pull --rebase
cd ..

#--include-direct-includes --use-vector-store-search
./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --patch-repo ffmpeg --triage-label 'important,enhancement,fix/bug,fix/regression,invalid,API,API major,needs sample,needs docs,needs testing,duplicate' --llm-review-cmd './openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4 --extra-repo-root all_ffmpeg --use-vector-store-search --reasoning-effort high --verbose --debug-response-dir openaidebug --use-web-search --max-tool-calls 100 ' --verbose 2 --min-age-days 56 $*

#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --llm-review-cmd './openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4 --extra-repo-root for_ffmpeg --extra-repo-root forgejo_git --extra-repo-root ffmpeg-web --extra-repo-root fateserver --use-vector-store-search --reasoning-effort high --verbose --debug-response-dir openaidebug --use-web-search' --verbose --min-age-days 56 $*

#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --llm-review-cmd './openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4-mini --verbose' --verbose --min-age-days 56 $*
#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --verbose --min-age-days 56 $*
