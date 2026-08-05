#!/bin/bash
# Copyright (C) 2026 Michael Niedermayer
#
# This file is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 2 as
# published by the Free Software Foundation.
#
# This file is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License version 2 for more details.
#
# Additional permission:
#
# Michael Niedermayer is permitted to relicense this file, in whole or
# in part, under any version of the GNU General Public License, the GNU
# Affero General Public License, or the GNU Lesser General Public License
# published by the Free Software Foundation.
#
# This additional permission is personal to Michael Niedermayer.  It is
# not transferable and does not grant any other person permission to
# relicense this file under a different license.
#
# This additional permission may be removed from modified copies of this
# file.  Removal of this additional permission does not affect the
# licensing of the file under the GNU General Public License version 2.

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

# One agent + one worker per repository (both kinds of a repo share the
# process and its gcli cache), the TUI as a pure view over the same
# filedbs: one --db-root variable per repo ties the three processes
# together (the names match what the agent would derive by default).
# Only the agent takes the side strings; it records them, with each
# side's --log-file, in the db root's config.json, where the worker
# reads its configuration and the TUI what to tail. Every repo gets its
# own --debug-response-dir because the processes run concurrently and
# would otherwise race on the shared default; the gcli cache pickle
# needs no such flag, its default is already derived per side.
# FFmpeg/web, FFmpeg/fateserver and michaelni/Fairies define no
# FFmpeg-style label sets, hence no --triage-label/--issue-label there.
# Extra arguments ($*) go to fairy_tui.py itself.
COMMON="--gcli-account ff --verbose 2"
MODEL="openai:gpt-5.6@high"
FFMPEG_DB="$HOME/.fairy/db/gitea~ff~FFmpeg~FFmpeg"
WEB_DB="$HOME/.fairy/db/gitea~ff~FFmpeg~web"
FATE_DB="$HOME/.fairy/db/gitea~ff~FFmpeg~fateserver"
FAIRIES_DB="$HOME/.fairy/db/gitea~ff~michaelni~Fairies"
WRAP_TAIL="--use-vector-store-search --verbose --web-search live --max-tool-calls 100"

FFMPEG_PR="
    --owner FFmpeg --repo FFmpeg $COMMON
    --log-file logs/ffmpeg.log
    --patch-repo ffmpeg
    --triage-label 'important,enhancement,fix/bug,fix/regression,resolution/invalid,API,API major,needs sample,needs docs,needs testing,resolution/duplicate'
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root ffmpeg
        --model $MODEL
        --extra-repo-root all_ffmpeg
        --debug-response-dir openaidebug
        $WRAP_TAIL'
    --min-age-days 56
"
FFMPEG_ISSUES="
    --owner FFmpeg --repo FFmpeg $COMMON
    --log-file logs/ffmpeg.log
    --issue-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed'
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root ffmpeg
        --extra-repo-root all_ffmpeg
        --triage-model openai:gpt-5.6-luna
        --triage-service-tier flex
        --model $MODEL
        --service-tier flex
        --debug-response-dir openaidebug-issues
        $WRAP_TAIL'
"
WEB_PR="
    --owner FFmpeg --repo web $COMMON
    --log-file logs/web.log
    --patch-repo ffmpeg-web
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root ffmpeg-web
        --project-facts project_facts/ffmpeg-web.md
        --model $MODEL
        --extra-repo-root all_ffmpeg
        --debug-response-dir openaidebug-web
        $WRAP_TAIL'
    --min-age-days 13
"
FATE_PR="
    --owner FFmpeg --repo fateserver $COMMON
    --log-file logs/fateserver.log
    --patch-repo fateserver
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root fateserver
        --project-facts project_facts/fateserver.md
        --model $MODEL
        --extra-repo-root all_ffmpeg
        --debug-response-dir openaidebug-fateserver
        $WRAP_TAIL'
    --min-age-days 13
"
FAIRIES_PR="
    --owner michaelni --repo Fairies $COMMON
    --log-file logs/fairies.log
    --patch-repo fairies
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root fairies
        --model $MODEL
        --debug-response-dir openaidebug-fairies
        $WRAP_TAIL'
"
FAIRIES_ISSUES="
    --owner michaelni --repo Fairies $COMMON
    --log-file logs/fairies.log
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root fairies
        --model $MODEL
        --debug-response-dir openaidebug-fairies-issues
        $WRAP_TAIL'
"

mkdir -p logs
# console output goes to .console files: a backgrounded process's
# stderr handlers would otherwise scribble over the blessed screen
./agent.py  --pr-args "$FFMPEG_PR" --issue-args "$FFMPEG_ISSUES" --db-root "$FFMPEG_DB" --loop 600 >"logs/agent-ffmpeg.console" 2>&1 &
./worker.py --db-root "$FFMPEG_DB" --loop 600 >"logs/worker-ffmpeg.console" 2>&1 &
./agent.py  --pr-args "$WEB_PR" --db-root "$WEB_DB" --loop 600 >"logs/agent-web.console" 2>&1 &
./worker.py --db-root "$WEB_DB" --loop 600 >"logs/worker-web.console" 2>&1 &
./agent.py  --pr-args "$FATE_PR" --db-root "$FATE_DB" --loop 600 >"logs/agent-fateserver.console" 2>&1 &
./worker.py --db-root "$FATE_DB" --loop 600 >"logs/worker-fateserver.console" 2>&1 &
./agent.py  --pr-args "$FAIRIES_PR" --issue-args "$FAIRIES_ISSUES" --db-root "$FAIRIES_DB" --loop 600 >"logs/agent-fairies.console" 2>&1 &
./worker.py --db-root "$FAIRIES_DB" --loop 600 >"logs/worker-fairies.console" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT

./fairy_tui.py --log-file fairy_tui.log \
    --db-root "$FFMPEG_DB" \
    --db-root "$WEB_DB" \
    --db-root "$FATE_DB" \
    --db-root "$FAIRIES_DB" \
    $*
