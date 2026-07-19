#!/bin/bash

set -euo pipefail

cd ~/forgejo_fairy

cd ffmpeg
git checkout master
git fetch fforge
git pull --rebase
cd ..

# Both fairy-ref.sh and issue-fairy-ref.sh in one 4-pane TUI (README
# "Interactive TUI"). Same per-side arguments as those launchers; the
# issue side gets its own cache pickles because both sides run
# concurrently in one process and would otherwise race on the shared
# defaults. Extra arguments ($*) go to fairy_tui.py itself.
./fairy_tui.py --log-file fairy_tui.log \
    --pr-args "--owner FFmpeg --repo FFmpeg --gcli-account ff --patch-repo ffmpeg --triage-label 'important,enhancement,fix/bug,fix/regression,resolution/invalid,API,API major,needs sample,needs docs,needs testing,resolution/duplicate' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.6@high --extra-repo-root all_ffmpeg --use-vector-store-search --verbose --debug-response-dir openaidebug --web-search live --max-tool-calls 100 ' --verbose 2 --min-age-days 56" \
    --issue-args "--owner FFmpeg --repo FFmpeg --gcli-account ff --issue-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --extra-repo-root all_ffmpeg --triage-model openai:gpt-5.6-luna --triage-service-tier flex --model openai:gpt-5.6@high --service-tier flex --use-vector-store-search --verbose --debug-response-dir openaidebug --web-search live --max-tool-calls 100 ' --cache $HOME/.fairy/issue_data_cache.pkl --llm-parallelism 3 --verbose 2" \
    $*
