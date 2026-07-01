#!/bin/bash

set -e

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
