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

cd ffmpeg
git checkout master
git fetch fforge
git pull --rebase
cd ..

# One-shot cron cycle: configurator writes the db root's config.toml,
# then the agent runs scan -> inline worker -> send from it. Extra
# arguments ($*) go to agent.py itself (e.g. --dry-run, --loop 600).
# ./configurator.py --help documents the side options.
# One db root is one config: the configurator writes the whole file,
# so this PR-only deployment has its own root and issue-fairy-ref.sh
# uses another. fairy-ui-ref.sh is the alternative both-sides
# deployment of the same repos; run one style or the other, not both.
#--include-direct-includes --use-vector-store-search
FFMPEG_DB="$HOME/.fairy/db/gitea~ff~FFmpeg~FFmpeg"
./configurator.py --db-root "$FFMPEG_DB" \
    --owner FFmpeg --repo FFmpeg \
    --gcli-account ff \
    --verbose 2 \
    --prs \
    --patch-repo ffmpeg \
    --triage-label 'important,enhancement,fix/bug,fix/regression,resolution/invalid,API,API major,needs sample,needs docs,needs testing,resolution/duplicate' \
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root ffmpeg
        --model openai:gpt-5.6@high
        --extra-repo-root all_ffmpeg
        --use-vector-store-search
        --verbose
        --debug-response-dir openaidebug
        --web-search live
        --max-tool-calls 100' \
    --min-age-days 56
./agent.py --drain --db-root "$FFMPEG_DB" $*

#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.6@high --extra-repo-root for_ffmpeg --extra-repo-root forgejo_git --extra-repo-root ffmpeg-web --extra-repo-root fateserver --use-vector-store-search --verbose --debug-response-dir openaidebug --web-search live' --verbose --min-age-days 56 $*

#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --llm-review-cmd './pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.6-luna --verbose' --verbose --min-age-days 56 $*
#./fairy.py --owner FFmpeg --repo FFmpeg  --gcli-account ff --verbose --min-age-days 56 $*

# Ensemble: GPT + Opus + GLM review the same PR, GPT-5.6-Terra verifies and combines (needs --podman-host for the per-reviewer shells, ANTHROPIC_API_KEY + ZAI_API_KEY in .env):
#./fairy.py --owner FFmpeg --repo FFmpeg --gcli-account ff --patch-repo ffmpeg --podman-host 'x86_64=fairy@HOST,gpu=nvidia.com/gpu=0' --llm-review-cmd './pr_review_wrapper.py --podman --repo-root ffmpeg --extra-repo-root all_ffmpeg --triage-model openai:gpt-5.6-luna --model openai:gpt-5.6 --extra-model anthropic:claude-opus-4 --extra-model zai:glm-5.2 --combine-model openai:gpt-5.6-terra --verbose' --verbose --min-age-days 56 $*
