#!/bin/bash
#
# Cutover A/B: the old pipeline (arm A, your deployed master tree) and
# the filedb agent (arm B, this branch) decide over the same forge
# snapshot; ab_diff.py reports every item where they disagree.
#
# Neither arm posts anything: no --approve, and the LLM is
# tools/ab_stub_llm.py (captures the payload, verdicts a deterministic
# skip -- zero model spend). Production caches and workset are only
# ever COPIED; the arms work on the copies under $AB_DIR.
#
# Run (defaults match the FFmpeg/FFmpeg production setup):
#
#     ~/omp/forgejo_fairy/tools/ab_run.sh
#
# Repeat daily for a few days; benign, understood differences go into
# an allowlist file (one regex per line):
#
#     ALLOWLIST=~/fairy-ab/allow.txt ~/omp/forgejo_fairy/tools/ab_run.sh
#
# Defaults mirror the production fairy-ui.sh / fairy-sides.sh side
# definitions (account ff-production, PR min-age 10 + CI triage +
# forced skips, issue min-age 255). --limit is deliberately NOT set on
# either arm: the two pipelines order candidates differently, so a
# limit would compare two different top-N cuts instead of eligibility;
# the stub makes the unlimited run free anyway.
#
# Overridables: OLD_TREE NEW_TREE AB_DIR OWNER REPO ACCOUNT PATCH_REPO
# MIN_AGE_DAYS ISSUE_MIN_AGE_DAYS PR_GATE_FLAGS PROD_CACHE
# PROD_ISSUE_CACHE PROD_WORKSET TRIAGE_LABELS ISSUE_LABELS.
# Throwaway tooling: delete after the cutover proves out.

set -euo pipefail

OLD_TREE=${OLD_TREE:-$HOME/forgejo_fairy}
NEW_TREE=${NEW_TREE:-$(cd "$(dirname "$0")/.." && pwd)}
AB_DIR=${AB_DIR:-$HOME/fairy-ab/$(date +%F-%H%M)}
OWNER=${OWNER:-FFmpeg}
REPO=${REPO:-FFmpeg}
ACCOUNT=${ACCOUNT:-ff-production}
PATCH_REPO=${PATCH_REPO:-$OLD_TREE/ffmpeg}
MIN_AGE_DAYS=${MIN_AGE_DAYS:-10}
ISSUE_MIN_AGE_DAYS=${ISSUE_MIN_AGE_DAYS:-255}
PR_GATE_FLAGS=${PR_GATE_FLAGS:---triage-on-ci-failure --force-skip-pr 22287,22447}
PROD_CACHE=${PROD_CACHE:-$HOME/.fairy/pr_data_cache.pkl}
PROD_ISSUE_CACHE=${PROD_ISSUE_CACHE:-$HOME/.fairy/issue_data_cache.pkl}
PROD_WORKSET=${PROD_WORKSET:-$HOME/.fairy/workset/gitea~ff-production~FFmpeg~FFmpeg}
TRIAGE_LABELS=${TRIAGE_LABELS:-important,enhancement,fix/bug,fix/regression,resolution/invalid,API,API major,needs sample,needs docs,resolution/duplicate}
ISSUE_LABELS=${ISSUE_LABELS:-repro/yes,repro/no,repro/no(env),repro/flaky,needs info,needs sample,bug,enhancement,regression,resolution/duplicate,resolution/invalid,resolution/external,resolution/fixed}
STUB="$NEW_TREE/tools/ab_stub_llm.py"

mkdir -p "$AB_DIR/payloads-a" "$AB_DIR/payloads-b" "$AB_DIR/worksetA"
echo "=== A/B snapshot -> $AB_DIR"
cp "$PROD_CACHE" "$AB_DIR/cacheA.pkl"
cp "$PROD_CACHE" "$AB_DIR/cacheB.pkl"
cp "$PROD_ISSUE_CACHE" "$AB_DIR/icacheA.pkl"
cp "$PROD_ISSUE_CACHE" "$AB_DIR/icacheB.pkl"
cp -a "$PROD_WORKSET" "$AB_DIR/worksetA/$(basename "$PROD_WORKSET")"
"$NEW_TREE/tools/workset_to_filedb.py" \
    "$AB_DIR/worksetA/$(basename "$PROD_WORKSET")" "$AB_DIR/db"

echo "=== arm A: old PR pipeline (dry, stub LLM)"
(cd "$OLD_TREE" && AB_PAYLOAD_DIR="$AB_DIR/payloads-a" ./fairy.py \
    --owner "$OWNER" --repo "$REPO" --gcli-account "$ACCOUNT" \
    --patch-repo "$PATCH_REPO" \
    --cache "$AB_DIR/cacheA.pkl" \
    --workset-dir "$AB_DIR/worksetA" \
    --triage-label "$TRIAGE_LABELS" \
    --llm-review-cmd "$STUB" \
    --min-age-days "$MIN_AGE_DAYS" \
    $PR_GATE_FLAGS \
    --verbose 2) 2>&1 | tee "$AB_DIR/logA-pr.txt"

echo "=== arm A: old issue pipeline (dry, stub LLM)"
(cd "$OLD_TREE" && AB_PAYLOAD_DIR="$AB_DIR/payloads-a" ./issue_fairy.py \
    --owner "$OWNER" --repo "$REPO" --gcli-account "$ACCOUNT" \
    --cache "$AB_DIR/icacheA.pkl" \
    --workset-dir "$AB_DIR/worksetA" \
    --issue-label "$ISSUE_LABELS" \
    --llm-review-cmd "$STUB" \
    --min-age-days "$ISSUE_MIN_AGE_DAYS" \
    --verbose 2) 2>&1 | tee "$AB_DIR/logA-issue.txt"

echo "=== arm B: filedb agent (scan + stub drain, no send)"
(cd "$NEW_TREE" && AB_PAYLOAD_DIR="$AB_DIR/payloads-b" ./agent.py \
    --drain --db-root "$AB_DIR/db" \
    --pr-args "--owner $OWNER --repo $REPO --gcli-account $ACCOUNT
        --patch-repo $PATCH_REPO
        --cache $AB_DIR/cacheB.pkl
        --triage-label '$TRIAGE_LABELS'
        --llm-review-cmd '$STUB'
        --min-age-days $MIN_AGE_DAYS
        $PR_GATE_FLAGS
        --verbose 2" \
    --issue-args "--owner $OWNER --repo $REPO --gcli-account $ACCOUNT
        --cache $AB_DIR/icacheB.pkl
        --issue-label '$ISSUE_LABELS'
        --llm-review-cmd '$STUB'
        --min-age-days $ISSUE_MIN_AGE_DAYS
        --verbose 2") 2>&1 | tee "$AB_DIR/logB.txt"

echo "=== diff"
"$NEW_TREE/tools/ab_diff.py" \
    --payloads-a "$AB_DIR/payloads-a" --payloads-b "$AB_DIR/payloads-b" \
    --new-db "$AB_DIR/db" \
    --log-a "$AB_DIR/logA-pr.txt" --log-a "$AB_DIR/logA-issue.txt" \
    ${ALLOWLIST:+--allowlist "$ALLOWLIST"} \
    | tee "$AB_DIR/report.txt"
