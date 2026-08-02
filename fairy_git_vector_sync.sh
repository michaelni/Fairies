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

set -e

#if we had more than 2 vector stores support in the API
#./pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root for_ffmpeg --extra-repo-root forgejo_git --extra-repo-root ffmpeg-web --extra-repo-root fateserver

./pr_review_wrapper.py --repo-root ffmpeg --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose --extra-repo-root all_ffmpeg
./pr_review_wrapper.py --repo-root ffmpeg-web --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose
./pr_review_wrapper.py --repo-root fateserver --model openai:gpt-5.4-mini --prepare-vector-store-only --verbose
