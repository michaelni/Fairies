#!/bin/bash

set -e

#if we had more than 2 vector stores support in the API
#./openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root for_ffmpeg --extra-repo-root forgejo_git --extra-repo-root ffmpeg-web --extra-repo-root fateserver

./openai_pr_review_wrapper.py --repo-root ffmpeg --model gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root all_ffmpeg
