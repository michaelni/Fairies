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

set -e -o pipefail

fetch(){
    cd $1
    git checkout master
    git pull --rebase
    git fetch --all
    cd ..
}

fetch ffmpeg
fetch ffmpeg-web
fetch fateserver
#fetch for_ffmpeg (we are upstream so not needed)

# CLICOLOR_FORCE keeps the live terminal colored across the ``tee`` pipe
# (auto-mode disables color whenever stderr isn't a TTY). The trade-off
# is ANSI escapes in forgejo_export.log -- view with ``less -R``.
CLICOLOR_FORCE=1 ./forgejo_export.py --gcli-account ff --owner FFmpeg --verbose --repo FFmpeg forgejo_git 2>&1 | tee -a forgejo_export.log

cd all_ffmpeg
git subtree pull --prefix=for_ffmpeg  ../for_ffmpeg master
git subtree pull --prefix=forgejo_git ../forgejo_git master
git subtree pull --prefix=ffmpeg-web  ../ffmpeg-web master
git subtree pull --prefix=fateserver  ../fateserver master
