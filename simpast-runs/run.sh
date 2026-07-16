#!/bin/bash
#
# Simulate-past prompt A/B harness.
#
# Reviews one or more historical PRs under --simulate-past with two (or
# more) prompt "arms", N samples per arm, in a single session so the
# model snapshot stays constant across arms. Arms are git refs (branch
# or commit) passed as positional args; only llm_prompt.py is swapped
# into the working tree per arm, so the prompt is the sole variable
# (harness and wrapper stay at the working-tree version).
#
# Arms run sequentially (they share the working-tree prompt). Samples
# within an arm run in parallel up to PAR, each with an isolated cache +
# bot_state so they don't race. Each cell streams live to the terminal
# and to its run.log; a cell is only marked complete (.done) after a
# clean finish, so an interrupted run resumes without skipping a cell
# that never finished.
#
# Config via env (defaults = the standard 4-PR FFmpeg suite):
#   PRS         space-separated PR numbers     (default "22290 22337 22624 20997")
#   CUTOFF      --simulate-past timestamp      (default 2026-04-23T20:00:00+0000)
#   PATCH_REPO  cutoff-prepped mirror          (default simpast-mirror/ffmpeg)
#   EXTRA_REPO  extra container repo, "" omits (default simpast-mirror/all_ffmpeg)
#   OUTROOT     output dir, one per experiment (default simpast-runs/out)
#   SAMPLES     samples per arm                (default 3)
#   PAR         concurrent samples per arm     (default SAMPLES: one wave)
#   TIER        OpenAI service tier            (default flex)
#   MODE        podman (default) | container   (OpenAI-hosted, A/B reference)
#   MODEL       main reviewer model            (default openai:gpt-5.4);
#               reasoning effort comes ONLY from the @suffix (bare = API default)
#   PODMAN_SSH  ssh dest for the podman host   (e.g. fairy@podman-host)
#   TREE        checkout to run fairy from     (default this repo)
#   WRAPPER_EXTRA  extra wrapper args, e.g. "--extra-model zai:glm-5.2 --combine-model openai:gpt-5.4"
#
# The LLM shell always runs in an ephemeral Podman container on
# PODMAN_SSH, matching production; OpenAI-hosted containers are
# deprecated and produce non-comparable results, so this harness
# refuses to use them. Provision the host first
# (containers/provision_remote.py --ssh PODMAN_SSH <repo> ...). Keep PAR
# modest so one host is not swamped by concurrent containers.
#
# Examples:
#   # 4-PR suite, baseline vs a prompt branch
#   OUTROOT=simpast-runs/mytest \
#     bash simpast-runs/run.sh experiment/test-base experiment/unrel-v2
#
#   # single PR, 4 parallel samples per arm, different mirror
#   PRS=23381 PATCH_REPO=simpast-mirror-23381/ffmpeg EXTRA_REPO= \
#   SAMPLES=4 PAR=2 OUTROOT=simpast-runs/pr23381 \
#     bash simpast-runs/run.sh experiment/test-base experiment/unrel-v2
#
#   # one PR, prompt taken from specific commits (bisect)
#   PRS=22624 OUTROOT=simpast-runs/bisect SAMPLES=3 PAR=3 \
#     bash simpast-runs/run.sh 8fee6d5 experiment/test-base

set -euo pipefail
cd "${TREE:-$(dirname "$0")/..}"
ROOT=$(pwd)
WRAPPER_EXTRA=${WRAPPER_EXTRA:-}

PRS=(${PRS:-22290 22337 22624 20997})
CUTOFF=${CUTOFF:-2026-04-23T20:00:00+0000}
PATCH_REPO=$(cd "${PATCH_REPO:-simpast-mirror/ffmpeg}" && pwd)
EXTRA_REPO=${EXTRA_REPO-simpast-mirror/all_ffmpeg}
[[ -n "$EXTRA_REPO" ]] && EXTRA_REPO=$(cd "$EXTRA_REPO" && pwd)
OUTROOT=${OUTROOT:-simpast-runs/out}; [[ "$OUTROOT" = /* ]] || OUTROOT="$ROOT/$OUTROOT"
SAMPLES=${SAMPLES:-3}
PAR=${PAR:-$SAMPLES}
TIER=${TIER:-flex}
MODEL=${MODEL:-openai:gpt-5.4}
PODMAN_SSH=${PODMAN_SSH:-}
MODE=${MODE:-podman}
if [[ "$MODE" = container ]]; then
    # OpenAI-hosted containers: deprecated for production but still the
    # reference arm for backend/billing A/Bs (they bill context once).
    CONTAINER_ARGS="--use-openai-container-repos --max-tool-calls 100"
else
    [[ -n "$PODMAN_SSH" ]] || { echo "PODMAN_SSH=user@host required for MODE=podman" >&2; exit 2; }
    CONTAINER_ARGS="--podman --podman-ssh-dest $PODMAN_SSH --podman-max-tool-rounds 100"
fi
(($# >= 1)) || { echo "usage: $0 <arm-ref> [arm-ref ...]" >&2; exit 2; }

# Write the prompt to the working tree only (never the index), so a
# concurrent `git commit` can't accidentally capture an arm's prompt.
restore() { git show HEAD:llm_prompt.py > llm_prompt.py 2>/dev/null || true; }
trap restore EXIT

run_one() {
    local label=$1 i=$2
    local outdir="$OUTROOT/${label}_${i}"
    # .done is written only after a clean finish, so an interrupted cell
    # (which still left a partial run.log) is correctly re-run on resume.
    [[ -f "$outdir/.done" ]] && { echo "skip ${label}_${i} (done)"; return; }
    mkdir -p "$outdir/openaidebug"
    local force=() pr; for pr in "${PRS[@]}"; do force+=(--force-review-pr "$pr"); done
    local extra=""; [[ -n "$EXTRA_REPO" ]] && extra="--extra-repo-root $EXTRA_REPO"
    # Live output: tee the raw stream to run.log (what the analysis tools
    # read) and to the terminal. When samples run concurrently (PAR>1),
    # tag each terminal line with the cell so the interleaving is readable.
    local pfx="s/^//"; ((PAR > 1)) && pfx="s/^/${label}_${i}| /"
    echo ">>> start ${label}_${i}  PRs=${PRS[*]}"
    # --simulate-past rewinds git but reads live forge state: a replayed PR
    # that has since merged/closed would skip as "not open". Replaying it at
    # the cutoff is the whole point, so force review regardless of live state.
    if CLICOLOR_FORCE=1 ./fairy.py \
        --owner FFmpeg --repo FFmpeg --gcli-account ff \
        --simulate-past "$CUTOFF" \
        --patch-repo "$PATCH_REPO" \
        --patch-pr-ref-template "fforge/pr/{number}" \
        --cache "$outdir/cache.pkl" \
        --fairy-state-cache "$outdir/bot_state.pkl" \
        --forced-only --force-review-non-open "${force[@]}" \
        --llm-parallelism "${#PRS[@]}" \
        --llm-review-cmd "./pr_review_wrapper.py \
            --repo-root $PATCH_REPO $extra \
            $CONTAINER_ARGS $WRAPPER_EXTRA \
            --model $MODEL --triage-model openai:gpt-5.4-mini \
            --service-tier $TIER --reasoning-summary detailed \
            --allowed-model openai:gpt-5.5 --allowed-model openai:gpt-5.4 \
            --allowed-model openai:gpt-5.6 --allowed-model openai:gpt-5.6-sol \
            --allowed-model openai:gpt-5.6-terra \
            --debug-response-dir $outdir/openaidebug --verbose" \
        --verbose 2 2>&1 | tee "$outdir/run.log" | sed -u "$pfx"
    then
        touch "$outdir/.done"; echo "<<< done ${label}_${i}"
    else
        echo "!!! FAILED ${label}_${i} (see $outdir/run.log)"
    fi
}

for arm in "$@"; do
    label=${arm##*/}
    echo "=== arm $label ($arm)  PRs=${PRS[*]}  samples=$SAMPLES ==="
    git show "$arm:llm_prompt.py" > llm_prompt.py
    for i in $(seq 1 "$SAMPLES"); do
        run_one "$label" "$i" &
        while (( $(jobs -rp | wc -l) >= PAR )); do wait -n; done
    done
    wait
done
echo "all arms done"
