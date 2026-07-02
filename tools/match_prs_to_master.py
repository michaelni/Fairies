#!/usr/bin/env python3
"""
/*
 * Copyright (C) 2026 Michael Niedermayer
 *
 * This file is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 *
 * This file is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License version 2 for more details.
 *
 * Additional permission:
 *
 * Michael Niedermayer is permitted to relicense this file, in whole or
 * in part, under any version of the GNU General Public License, the GNU
 * Affero General Public License, or the GNU Lesser General Public License
 * published by the Free Software Foundation.
 *
 * This additional permission is personal to Michael Niedermayer.  It is
 * not transferable and does not grant any other person permission to
 * relicense this file under a different license.
 *
 * This additional permission may be removed from modified copies of this
 * file.  Removal of this additional permission does not affect the
 * licensing of the file under the GNU General Public License version 2.
 */

Match commits in a base branch (origin/master) to forge PR head refs.

Each PR commit is scored against candidate master commits by four tests:

  P  equal git patch-id (the diff hash, stable across a rebase)   2 points
  S  equal subject (first line)                                   1 point
  A  equal author date (preserved by a rebase/cherry-pick)        1 point
  C  master commit date no earlier than the PR commit's           2 points
     (within a day, to absorb timezone slop)

Match tiers, strongest first, are an exact SHA (the commit itself is on master)
then total score 6, 5 and 4. The four tests are shown as a signature, e.g.
"PSAC" (all four), "P.AC" (subject differs) or "SHA" for an exact SHA; a "."
marks a failed test, so a mistakenly opened PR stands out as, say, "PSA." (its
patch is on master but predates it).

A master commit matches only one PR commit: once a stronger tier claims it, it
is no longer offered to weaker ones, so a superseded copy of a merged patch is
left unmatched rather than mis-paired. When a master commit is claimed by
several PR commits within a tier none can be singled out and those matches are
flagged uncertain with "?". Every master commit reaching the same score is
listed, so a patch that landed more than once is shown in full.

A PR's status is decided by its tip, the patch that lands last:
  merged - the tip was found on master
  maybe  - the tip matched, but the master commit is contested by another PR
  part   - the tip was not found, but (with --verbose) some other commit was
  open   - nothing was found on master

--verbose matches and lists every PR commit (capped by --max-commits), which
helps spot partially merged PRs.

Only PRs targeting the base are in scope; a PR built on a release branch is
detected by the branch its oldest commit sits on and skipped.

Matching is by content using only git, so on its own it cannot tell an open PR
from one closed without merging. With --gcli the forge is queried (one batched
gcli call) for each PR's real state, which overrides that guess; the forge's
target branch is only used to note a disagreement, as the git-based guess is
more reliable. It is off by default as its an order of magnitude slower.

Usage:
  ./match_prs_to_master.py [--base origin/master]
                           [--pr-glob 'refs/remotes/fforge/pr/*']
                           [--release-glob REF] [--since DATE] [--limit N]
                           [--verbose] [--max-commits N] [--show-oos]
                           [--show-alt] [--color auto|always|never]
                           [--gcli [--gcli-repo OWNER/REPO] [--gcli-account A]]
                           [--format tsv|report]
"""

import argparse
import json
import os
import subprocess
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

STATUS_COLOR = {"merged": "32", "maybe": "33", "part": "36",
                "open": "31", "closed": "35", "oos": "90"}


def git(*args, check=True):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          check=check).stdout


def git_lines(*args):
    out = git(*args)
    return out.splitlines() if out else []


PATCH_GEN = ["git", "diff-tree", "--stdin", "-C", "-M", "--root", "-p", "-r",
             "--no-ext-diff", "--no-color"]


def patch_id_chunk(shas):
    """{sha: patch-id} for shas, via `git diff-tree --stdin | git patch-id`.

    diff-tree reads the commits from stdin and emits each diff in one process,
    so patches are generated in a single batched stream rather than per commit;
    -C -M give the same copy/rename detection a porcelain diff would. Each rev
    needs a trailing newline or diff-tree drops the last one. Commits without a
    diff (e.g. merges) are simply absent from the result.
    """
    gen = subprocess.Popen(PATCH_GEN, text=True, stdin=subprocess.PIPE,
                           stdout=subprocess.PIPE)
    pid = subprocess.Popen(["git", "patch-id", "--stable"],
                           stdin=gen.stdout, stdout=subprocess.PIPE, text=True)
    gen.stdout.close()
    threading.Thread(target=lambda: (gen.stdin.write("".join(s + "\n" for s in shas)),
                                     gen.stdin.close()), daemon=True).start()
    out = {}
    for line in pid.stdout:
        parts = line.split()
        if len(parts) == 2:
            out[parts[1]] = parts[0]
    pid.wait()
    gen.wait()
    return out


def patch_ids(shas, jobs=None):
    """{sha: patch-id} for many commits, computed in parallel.

    Every patch-id the program needs is produced here through one command, so
    generating the diff (which dominates patch-id) is batched instead of forked
    per commit, and master and PR ids are always the same kind and comparable.
    """
    shas = list(shas)
    jobs = max(1, jobs or os.cpu_count() or 1)
    out = {}
    with ThreadPoolExecutor(jobs) as ex:
        for part in ex.map(patch_id_chunk, [shas[i::jobs] for i in range(jobs)]):
            out.update(part)
    return out


def build_master_index(base, since):
    """Return (sha set, subject -> [sha], sha -> meta), meta in git log order.

    meta is (subject, author_date, commit_date). The SHA set spans all of base
    (merges and full history included) so the exact-SHA match is a true
    reachability test; subject and meta are windowed by --since to keep indexing
    cheap. Patch-ids are computed later, in one batch over every needed commit.
    """
    master_shas = set(git_lines("rev-list", base))
    rng_args = ["--no-merges"]
    if since:
        rng_args += ["--since", since]

    meta = {}
    subject_to_shas = defaultdict(list)
    for line in git_lines("log", *rng_args, "--format=%H\x1f%at\x1f%ct\x1f%s", base):
        sha, at, ct, subj = line.split("\x1f", 3)
        meta[sha] = (subj, int(at), int(ct))
        subject_to_shas[subj].append(sha)

    return master_shas, subject_to_shas, meta


def pr_pool(heads, base):
    """Walk every PR's commits at once: return {sha: (parents, subject,
    author_date, commit_date)} and a sha -> output-order index, for all commits
    reachable from heads but not base.

    One union walk replaces a per-PR log; a single PR's commits are a
    subsequence of this order, recovered from the parent graph in Python.
    """
    pool, order = {}, {}
    gen = subprocess.Popen(["git", "log", "--stdin", "--not", base,
                            "--format=%H\x1f%P\x1f%at\x1f%ct\x1f%s"],
                           text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    threading.Thread(target=lambda: (gen.stdin.write("\n".join(heads)),
                                     gen.stdin.close()), daemon=True).start()
    for line in gen.stdout:
        sha, parents, at, ct, subj = line.rstrip("\n").split("\x1f", 4)
        order[sha] = len(order)
        pool[sha] = (parents.split(), subj, int(at), int(ct))
    gen.wait()
    return pool, order


def pr_divergent(head, pool, order):
    """A PR's own commits as [(sha, subject, author_date, commit_date)], newest
    first (the pool's git log order); its last, oldest entry is the fork point.

    Merges are kept: an exact-SHA match can still place one on the base branch.
    """
    seen, stack = set(), [head]
    while stack:
        sha = stack.pop()
        if sha in pool and sha not in seen:
            seen.add(sha)
            stack.extend(pool[sha][0])
    return [(sha, *pool[sha][1:]) for sha in sorted(seen, key=order.get)]


SHA_SCORE = 7              # exact-SHA match, ranked above any scored match
TIERS = (SHA_SCORE, 6, 5, 4)
CDATE_TOL = 24 * 3600      # commit-date slack absorbing timezone errors


def candidate_masters(commit, pid, master_shas, patchid_to_shas,
                      subject_to_shas, meta):
    """Master commits a PR commit could be, as {master_sha: (score, signature)}.

    An exact SHA wins outright. Otherwise candidates (sharing the patch-id or
    subject) are scored: patch-id 2, subject 1, author date 1, and 2 when the
    master commit date is not earlier than the PR commit's (within CDATE_TOL).
    The signature spells the four tests as P/S/A/C, '.' where one failed. Only
    candidates scoring at least 4 (the weakest tier) are returned.
    """
    sha, _subj, at, ct = commit
    if sha in master_shas:
        return {sha: (SHA_SCORE, "SHA")}
    pool = set(patchid_to_shas.get(pid, ())) | set(subject_to_shas.get(_subj, ()))
    out = {}
    for m in pool:
        msubj, mat, mct = meta[m]
        tests = (pid is not None and m in patchid_to_shas.get(pid, ()),
                 _subj == msubj, at == mat, mct >= ct - CDATE_TOL)
        score = 2 * tests[0] + tests[1] + tests[2] + 2 * tests[3]
        if score >= 4:
            out[m] = (score, "".join(c if t else "." for c, t in zip("PSAC", tests)))
    return out


def auto_since(pr_glob, margin_days=30):
    """Earliest PR head commit date minus a margin, to bound master indexing."""
    dates = git_lines("for-each-ref", "--format=%(committerdate:unix)", pr_glob)
    dates = [int(d) for d in dates if d.strip()]
    if not dates:
        return None
    return f"@{min(dates) - margin_days * 86400}"


def base_branch(divergent, base, release_sets):
    """Integration branch the PR targets, or base if it builds on base.

    The oldest commit a PR adds on top of base (the last of divergent) is its
    fork point. When the PR was cut from a release branch that commit is a
    release-only commit, so the release ref still contains it; a master PR's
    oldest commit lives nowhere but the PR ref. This holds no matter how little
    the release had diverged when the PR was opened.
    """
    if not divergent or not release_sets:
        return base
    fork = divergent[-1][0]
    return next((ref for ref, shas in release_sets.items() if fork in shas), base)


def short_branch(ref):
    """Forge branch name of a git ref: origin/master -> master,
    refs/remotes/fforge/release/8.1 -> release/8.1."""
    if ref.startswith("refs/remotes/"):
        return ref.split("/", 3)[-1]
    return ref.rsplit("/", 1)[-1]


def remote_repo(pr_glob):
    """OWNER/REPO from the remote named in a refs/remotes/<remote>/... glob."""
    parts = pr_glob.split("/")
    remote = parts[2] if parts[:2] == ["refs", "remotes"] else "origin"
    url = git("remote", "get-url", remote).strip()
    path = url.split("/", 3)[-1] if "://" in url else url.split(":", 1)[-1]
    return path.removesuffix(".git").strip("/")


def gcli_pulls(repo, account=None, jobs=8):
    """{pr_number: (state, target_branch)} for every PR, in one batched fetch.

    The forge caps a listing at 50 PRs, so pages are pulled in parallel until a
    short page marks the end. state is merged/closed/open (merged wins).
    """
    base_cmd = ["gcli"] + (["-a", account] if account else []) + ["api"]

    def page(n):
        out = subprocess.run(base_cmd + [
            f"repos/{repo}/pulls?state=all&limit=50&page={n}"],
            capture_output=True, text=True, check=True).stdout
        return json.loads(out)

    info, page_no = {}, 1
    with ThreadPoolExecutor(jobs) as ex:
        while True:
            batch = list(ex.map(page, range(page_no, page_no + jobs)))
            for pulls in batch:
                for p in pulls:
                    state = "merged" if p.get("merged") else p.get("state", "open")
                    info[p["number"]] = (state, p["base"]["ref"])
            if any(len(pulls) < 50 for pulls in batch):
                return info
            page_no += jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="origin/master",
                    help="base branch PRs are matched against (default: %(default)s)")
    ap.add_argument("--pr-glob", default="refs/remotes/fforge/pr/*",
                    help="git ref glob naming the PR head refs to scan "
                         "(default: %(default)s)")
    ap.add_argument("--release-glob", default=None,
                    help="out-of-scope base branches "
                         "(default: derive 'release/*' from --pr-glob)")
    ap.add_argument("--since", default=None,
                    help="master window start (default: earliest PR date - 30d)")
    ap.add_argument("--limit", type=int, default=0,
                    help="process only the first N PRs by number (default: 0 = all)")
    ap.add_argument("--format", choices=["tsv", "report"], default="report",
                    help="output as a human report or machine-readable tsv")
    ap.add_argument("--verbose", action="store_true",
                    help="match every PR commit, not just the tip")
    ap.add_argument("--max-commits", type=int, default=50,
                    help="cap on PR commits inspected in --verbose mode")
    ap.add_argument("--show-oos", action="store_true",
                    help="list PRs whose target branch is out of scope")
    ap.add_argument("--show-alt", action="store_true",
                    help="also show matches dominated by a better one (marked +)")
    ap.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                    help="colorize the report (default: auto, on when a terminal)")
    ap.add_argument("--gcli", action="store_true",
                    help="ask the forge (via gcli) for PR state and target branch; "
                         "state overrides the git guess, a differing target is "
                         "only noted (default: off, to spare the server)")
    ap.add_argument("--gcli-repo", default=None,
                    help="OWNER/REPO for gcli (default: derive from --pr-glob remote)")
    ap.add_argument("--gcli-account", default=None,
                    help="gcli account (default: gcli's configured default)")
    args = ap.parse_args()

    pr_heads = dict(line.split() for line in git_lines(
        "for-each-ref", "--format=%(refname:short) %(objectname)", args.pr_glob))
    if not pr_heads:
        parts = args.pr_glob.split("/")
        remote = parts[2] if parts[:2] == ["refs", "remotes"] else "<remote>"
        sys.exit(f"error: no PR refs match {args.pr_glob!r}\nfetch them with: "
                 f"git config --add remote.{remote}.fetch "
                 f"'+refs/pull/*/head:{args.pr_glob}' && git fetch {remote}")

    since = args.since or auto_since(args.pr_glob)
    print(f"# indexing {args.base} since {since} ...", file=sys.stderr)
    master_shas, subject_to_shas, meta = build_master_index(args.base, since)
    print(f"# indexed {len(meta)} master commits", file=sys.stderr)

    # sort numerically by trailing PR number when possible
    def prnum(r):
        tail = r.rsplit("/", 1)[-1]
        return int(tail) if tail.isdigit() else -1
    pr_refs = sorted(pr_heads, key=prnum)
    if args.limit:
        pr_refs = pr_refs[:args.limit]

    if args.format == "tsv":
        print("pr_ref\tstatus\ttip_commit\tmaster_commit\tsignature\tsubject")

    use_color = (args.color == "always"
                 or (args.color == "auto" and sys.stdout.isatty()))

    def paint(s, code):
        return f"\033[{code}m{s}\033[0m" if use_color and code else s

    def commit_meta(sha):
        if sha in meta:
            return meta[sha]
        at, ct, subj = git("show", "-s", "--format=%at\x1f%ct\x1f%s",
                           sha).strip().split("\x1f", 2)
        return subj, int(at), int(ct)

    def subject_of(sha):
        return commit_meta(sha)[0]

    # Drop PRs that target a branch the base never absorbs (e.g. release/*):
    # their head trails a long release-only history that must not be read as
    # unmerged PR commits.
    release_glob = args.release_glob
    if release_glob is None:
        prefix, sep, _ = args.pr_glob.partition("/pr/")
        release_glob = prefix + "/release/*" if sep else None
    release_refs = (git_lines("for-each-ref", "--format=%(refname)", release_glob)
                    if release_glob else [])
    if release_glob and not release_refs:
        print(f"# warning: no refs match {release_glob!r}; release-branch PRs "
              "cannot be detected as out of scope", file=sys.stderr)
    release_sets = {ref: set(git_lines("rev-list", ref)) for ref in release_refs}

    # One union walk of every PR's commits, split per PR in Python and reused
    # for both scope and matching.
    pool, order = pr_pool([pr_heads[r] for r in pr_refs], args.base)
    divergent = {r: pr_divergent(pr_heads[r], pool, order) for r in pr_refs}
    pr_base = {pr_ref: base_branch(divergent[pr_ref], args.base, release_sets)
               for pr_ref in pr_refs}
    oos_refs = [r for r in pr_refs if pr_base[r] != args.base]
    pr_refs = [r for r in pr_refs if pr_base[r] == args.base]

    # Optional forge lookup: its state overrides our content-based guess, while
    # its target branch only annotates a mismatch (our git guess stays in force).
    pulls = {}
    if args.gcli:
        repo = args.gcli_repo or remote_repo(args.pr_glob)
        print(f"# fetching PR metadata from {repo} via gcli ...", file=sys.stderr)
        pulls = gcli_pulls(repo, args.gcli_account)
        print(f"# gcli returned {len(pulls)} pulls", file=sys.stderr)
        for r in pr_refs + oos_refs:
            forge = pulls.get(prnum(r))
            if forge and short_branch(pr_base[r]) != forge[1]:
                print(f"# note: {r} targets {short_branch(pr_base[r])} per git "
                      f"but {forge[1]} per gcli", file=sys.stderr)

    if args.show_oos:
        for pr_ref in oos_refs:
            head, base = pr_heads[pr_ref], pr_base[pr_ref]
            if args.format == "tsv":
                print(f"{pr_ref}\toos\t{head[:12]}\t-\t-\t{subject_of(head)}")
            else:
                print(f"{pr_ref}  [{paint(' oos  ', STATUS_COLOR['oos'])}]"
                      f"  base {base}  {subject_of(head)}")

    # Commits to match per PR: the tip always, plus its own commits with
    # --verbose (capped). Each is (sha, subject, author_date, commit_date).
    inspected = {}
    for pr_ref in pr_refs:
        head = pr_heads[pr_ref]
        div = divergent[pr_ref]
        tip = div[0] if div else (head, *commit_meta(head))
        commits = [tip]
        if args.verbose:
            capped = div[:args.max_commits] if args.max_commits else div
            commits += [c for c in capped if c[0] != head]
        inspected[pr_ref] = commits

    # One batched patch-id pass over every commit the run needs: the windowed
    # master commits and every inspected PR commit. master and PR ids thus come
    # from the same command and are directly comparable.
    need = set(meta).union(c[0] for cs in inspected.values() for c in cs)
    pids = patch_ids(need)
    print(f"# patch-ids for {len(pids)} of {len(need)} commits", file=sys.stderr)
    patchid_to_shas = defaultdict(list)
    for sha in meta:  # keep lists newest-first (meta follows git log order)
        if sha in pids:
            patchid_to_shas[pids[sha]].append(sha)

    # Resolve matches in score tiers so a master commit taken by a stronger
    # match is no longer offered to a weaker one. Within a tier a master commit
    # claimed by several PR commits cannot be pinned to one of them, so those
    # matches are flagged uncertain; the master is still consumed for the next.
    entries = [(pr_ref, commit[0], commit[1],
                candidate_masters(commit, pids.get(commit[0]), master_shas,
                                  patchid_to_shas, subject_to_shas, meta))
               for pr_ref in pr_refs for commit in inspected[pr_ref]]

    matched = {}
    consumed = set()
    for tier in TIERS:
        won = {}
        claimant_shas = defaultdict(set)
        for i, (pr_ref, sha, subj, cand) in enumerate(entries):
            if i in matched:
                continue
            avail = {m: sig for m, (sc, sig) in cand.items()
                     if sc == tier and m not in consumed}
            if avail:
                won[i] = avail
                for m in avail:
                    claimant_shas[m].add(sha)
        for i, avail in won.items():
            certain = all(len(claimant_shas[m]) == 1 for m in avail)
            matched[i] = (avail, certain)
        consumed.update(m for avail in won.values() for m in avail)

    def sig_field(sig):  # green when every test passed, yellow when some failed
        return paint(f"{sig:^4}", "32" if "." not in sig else "33")

    # Alternatives: candidates a commit could have matched but did not, because
    # a better (or equally good, contested) match claimed the master commit.
    rows_of = defaultdict(list)
    for i, (pr_ref, sha, subj, cand) in enumerate(entries):
        avail, certain = matched.get(i, ({}, True))
        alts = ({m: sig for m, (_sc, sig) in cand.items() if m not in avail}
                if args.show_alt else {})
        rows_of[pr_ref].append((sha, avail, certain, alts, subj))

    def cell(avail, certain):
        arrow = "->?" if avail and not certain else "-> "
        if not avail:
            return f"{arrow} " + paint("(unmatched)".ljust(19), "90")
        return f"{arrow} " + " ".join(f"{m[:12]} ({sig_field(sig)})"
                                      for m, sig in avail.items())

    def alt_str(alts):
        return "".join(f"  +{m[:12]} ({sig_field(sig)})" for m, sig in alts.items())

    counts = defaultdict(int)
    counts["oos"] = len(oos_refs)
    for pr_ref in pr_refs:
        rows = rows_of[pr_ref]
        head, head_avail, head_certain, head_alts, head_subj = rows[0]

        # Status is decided by the PR tip: when a PR is merged, its last patch
        # lands on master. A contested match is only plausible, so a matched
        # tip is "maybe" rather than "merged".
        if head_avail:
            status = "merged" if head_certain else "maybe"
        elif any(r[1] for r in rows):
            status = "part"
        else:
            status = "open"
        forge = pulls.get(prnum(pr_ref))
        if forge:
            status = forge[0]
        counts[status] += 1

        if args.format == "tsv":
            for sha, avail, certain, alts, subj in rows:
                q = "" if certain else "?"
                if not avail:
                    print(f"{pr_ref}\t{status}\t{sha[:12]}\t-\t-\t{subj}")
                for m, sig in avail.items():
                    print(f"{pr_ref}\t{status}\t{sha[:12]}\t{m[:12]}\t{sig}{q}\t{subj}")
                for m, asig in alts.items():
                    print(f"{pr_ref}\t{status}\t{sha[:12]}\t{m[:12]}\t{asig}+\t{subj}")
        else:
            print(f"{pr_ref}  [{paint(f'{status:^6}', STATUS_COLOR.get(status))}]"
                  f"  tip {head[:12]} "
                  f"{cell(head_avail, head_certain)}{alt_str(head_alts)}  {head_subj}")
            if args.verbose:
                for sha, avail, certain, alts, subj in rows[1:]:
                    print(f"{'':31}{sha[:12]} {cell(avail, certain)}"
                          f"{alt_str(alts)}  {subj}")

    print("# summary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
          file=sys.stderr)


if __name__ == "__main__":
    main()
