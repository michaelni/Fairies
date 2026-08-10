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

# Refresh one codex login's model catalog through the pinned container
# (codex never runs on the host). The catalog carries the models'
# prompts: review the diff, then confirm. The old cache is kept as
# models_cache.json~. codex may rotate the OAuth tokens during the
# fetch, so the container's auth.json is echoed back after a __AUTH__
# marker line and persisted (backup auth.json~) before the question --
# losing a rotated refresh token would break the login.
# usage: refresh_codex_catalog.sh CODEX_HOST CODEX_HOME [IMAGE]
set -eu
host=$1; home=$2; image=${3:-localhost/fairy-codex:latest}
cache=$home/models_cache.json
out=$(mktemp); new=$(mktemp); auth=$(mktemp)
trap 'rm -f "$out" "$new" "$auth"' EXIT
ssh "$host" "podman run --rm -i -e CODEX_HOME=/work/.codex-home $image \
    sh -c 'mkdir -p /work/.codex-home && cat >/work/.codex-home/auth.json \
           && codex debug models && echo __AUTH__ \
           && cat /work/.codex-home/auth.json'" <"$home/auth.json" >"$out"
awk '/^__AUTH__$/{exit} {print}' "$out" >"$new"
awk 'f{print} /^__AUTH__$/{f=1}' "$out" >"$auth"
python3 -m json.tool "$new" >/dev/null
if python3 -m json.tool "$auth" >/dev/null 2>&1 \
        && ! cmp -s "$auth" "$home/auth.json"; then
    cp "$home/auth.json" "$home/auth.json~"
    install -m 600 "$auth" "$home/auth.json"
    echo "auth.json was rotated by codex; persisted (backup auth.json~)"
fi
diff -u <(python3 -m json.tool "$cache" 2>/dev/null) \
        <(python3 -m json.tool "$new") || true
printf 'write %s? [y/N] ' "$cache"; read -r a; [ "$a" = y ]
[ ! -e "$cache" ] || cp "$cache" "$cache~"
mv "$new" "$cache"
