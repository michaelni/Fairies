#!/bin/bash

set -e

#if we had more than 2 vector stores support in the API
#./pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root for_ffmpeg --extra-repo-root forgejo_git --extra-repo-root ffmpeg-web --extra-repo-root fateserver

./pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root all_ffmpeg
./pr_review_wrapper.py --repo-root ffmpeg-web --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose
./pr_review_wrapper.py --repo-root fateserver --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose
