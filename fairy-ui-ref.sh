#!/bin/bash

set -euo pipefail

cd ~/forgejo_fairy

# Clone-if-missing with the layout --patch-repo needs: an fforge
# remote whose refspec also fetches refs/pull/*/head as fforge/pr/*.
# PR head SHAs must resolve locally for git format-patch, and a
# default clone never fetches pull refs (README "Using fairy with
# your project", step 3).
ensure_repo() {
    [ -d "$1" ] || git clone --origin fforge "$2" "$1"
    git -C "$1" remote get-url fforge >/dev/null 2>&1 || \
        git -C "$1" remote add fforge "$2"
    git -C "$1" config --get-all remote.fforge.fetch | grep -qF refs/pull || \
        git -C "$1" config --add remote.fforge.fetch \
            '+refs/pull/*/head:refs/remotes/fforge/pr/*'
    (cd "$1" && git checkout master && git fetch fforge && git pull --rebase)
}

ensure_repo ffmpeg     https://code.ffmpeg.org/FFmpeg/FFmpeg.git
ensure_repo ffmpeg-web https://code.ffmpeg.org/FFmpeg/web.git
ensure_repo fateserver https://code.ffmpeg.org/FFmpeg/fateserver.git
ensure_repo fairies    https://code.ffmpeg.org/michaelni/Fairies.git

# All repos' fairies in one 4-pane TUI (README "Interactive TUI"): one
# --pr-args/--issue-args per side, same per-side arguments as the
# standalone launchers. Every side beyond the first gets its own
# --cache pickle and --debug-response-dir because the sides run
# concurrently in one process and would otherwise race on the shared
# defaults. FFmpeg/web, FFmpeg/fateserver and michaelni/Fairies define
# no FFmpeg-style label sets, hence no --triage-label/--issue-label
# there. Extra arguments ($*) go to fairy_tui.py itself.
COMMON="--gcli-account ff --verbose 2"
MODEL="openai:gpt-5.6@high"
WRAP_TAIL="--use-vector-store-search --verbose --web-search live --max-tool-calls 100"
./fairy_tui.py --log-file fairy_tui.log \
    --pr-args "--owner FFmpeg --repo FFmpeg $COMMON --patch-repo ffmpeg --triage-label 'important,enhancement,fix/bug,fix/regression,resolution/invalid,API,API major,needs sample,needs docs,needs testing,resolution/duplicate' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --model $MODEL --extra-repo-root all_ffmpeg --debug-response-dir openaidebug $WRAP_TAIL ' --min-age-days 56" \
    --pr-args "--owner FFmpeg --repo web $COMMON --patch-repo ffmpeg-web --cache $HOME/.fairy/web_pr_data_cache.pkl --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg-web --project-facts project_facts/ffmpeg-web.md --model $MODEL --extra-repo-root all_ffmpeg --debug-response-dir openaidebug-web $WRAP_TAIL ' --min-age-days 13" \
    --pr-args "--owner FFmpeg --repo fateserver $COMMON --patch-repo fateserver --cache $HOME/.fairy/fateserver_pr_data_cache.pkl --llm-review-cmd './pr_review_wrapper.py --repo-root fateserver --project-facts project_facts/fateserver.md --model $MODEL --extra-repo-root all_ffmpeg --debug-response-dir openaidebug-fateserver $WRAP_TAIL ' --min-age-days 13" \
    --pr-args "--owner michaelni --repo Fairies $COMMON --patch-repo fairies --cache $HOME/.fairy/fairies_pr_data_cache.pkl --llm-review-cmd './pr_review_wrapper.py --repo-root fairies --model $MODEL --debug-response-dir openaidebug-fairies $WRAP_TAIL '" \
    --issue-args "--owner FFmpeg --repo FFmpeg $COMMON --issue-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed' --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --extra-repo-root all_ffmpeg --triage-model openai:gpt-5.6-luna --triage-service-tier flex --model $MODEL --service-tier flex --debug-response-dir openaidebug-issues $WRAP_TAIL ' --cache $HOME/.fairy/issue_data_cache.pkl --llm-parallelism 3" \
    --issue-args "--owner michaelni --repo Fairies $COMMON --llm-review-cmd './pr_review_wrapper.py --repo-root fairies --model $MODEL --debug-response-dir openaidebug-fairies-issues $WRAP_TAIL ' --cache $HOME/.fairy/fairies_issue_data_cache.pkl" \
    $*
