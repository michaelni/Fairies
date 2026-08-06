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

# Issue helper: repro / bisect happens inside the podman container (add
# --podman-host 'x86_64=USER@HOST,gpu=nvidia.com/gpu=0'), duplicate search
# uses the exported-issue
# vector store. Triage runs on GPT-5.6-luna@medium, the full pass on
# GPT-5.6@high; both on the flex tier. Verdicts wait in reviewed/ by
# default; put --auto-mode in the --issues section to let the send pass
# post them. Extra arguments ($*) go to agent.py itself.
# configurator.py writes the db root's config.toml the agent then runs
# from. One db root is one config: the configurator writes the whole
# file, so this issue-only deployment has its own root, separate from
# fairy-ref.sh's PR root.
FFMPEG_DB="$HOME/.fairy/db/gitea~ff~FFmpeg~FFmpeg-issues"
./configurator.py --db-root "$FFMPEG_DB" \
    --owner FFmpeg --repo FFmpeg \
    --gcli-account ff \
    --verbose 2 \
    --issues \
    --triage-label 'repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed' \
    --llm-review-cmd './pr_review_wrapper.py
        --repo-root ffmpeg
        --extra-repo-root all_ffmpeg
        --triage-model openai:gpt-5.6-luna
        --triage-service-tier flex
        --model openai:gpt-5.6@high
        --service-tier flex
        --use-vector-store-search
        --verbose
        --debug-response-dir openaidebug
        --web-search live
        --max-tool-calls 100'
./agent.py --drain --db-root "$FFMPEG_DB" $*
