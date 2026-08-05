# Fairy

LLM tools for development forges, talking to the forge through
[gcli](https://sr.ht/~herrhotzenplotz/gcli/):

* **Forgejo Fairy** (`fairy.py`) reviews and assists pull requests.
* **Mail Fairy** (`mail_fairy.py`) mirrors mailing-list replies onto forge threads.

Send a pull request if something doesn't work or looks ugly or is aisloppy.
Make sure it's readable, reviewable and tested!

## Self tests

From the repository root, run the whole suite:

    python -m unittest discover -s tests -b

or a single module:

    python -m unittest -b tests.test_mail_fairy

Many tests exercise error paths on purpose; `-b` hides the stderr
diagnostics they provoke (unittest replays them for failing tests).
The tests are offline: no forge, API keys or podman host needed.

## Forgejo Fairy

Note: Forgejo Fairy is under heavy development and this codebase has not been cleaned up yet!

The state of every PR and issue is one JSON ticket file whose state is
the *directory* it sits in (`~/.fairy/db/<forge~account~owner~repo>/`:
`requests/ queued/ llm/ reviewed/ outgoing/ posted/ skipped/ ...` --
`mv` is a state change, `ls | wc -l` is a statistic; `filedb.py` is the
thin atomic API). `configurator.py` takes the per-side configuration
directly as options -- shared ones first, then a `--prs` and/or
`--issues` section per side; `./configurator.py --help` documents them
all, and a side's `--log-file` is the shared agent+worker log the UI
tails. It validates them and records them in the db root's
`config.toml`; the three processes cooperating over the directories
all take only `--db-root` and configure themselves from that file.
One db root is one config: each configurator run replaces the whole
file, so exactly one invocation owns a root -- a deployment that
configures the sides of a repo separately uses a separate root per
side. With `pip install
watchdog` the processes react to new files within 100ms; without it
they fall back to their poll intervals:

- `agent.py` (one per repository, PRs and issues together, one shared
  gcli cache): lists the forge, runs the gates, creates tickets in
  `queued/` (attention outcomes get their own dirs: `ci-blocked/`,
  `merge-ready/`, `awaiting-approver/`), applies the skip backoff and
  `--limit`, posts `outgoing/` verdicts (guard-checked), reaps dead
  workers and prunes. `--loop N` to daemonize, default is one pass
  (cron style); `--drain N` runs the worker inline for a
  self-contained one-shot, N tickets concurrently; `--dry-run` logs
  what would be posted. Side options on its command line override the
  config.toml values like the worker's do.
- `worker.py`: claims `queued/` tickets (flock + rename; the held lock
  is its liveness signal), runs the LLM wrapper, writes the verdict to
  `reviewed/` / `skipped/` / `error/`. `--parallel N` reviews N
  tickets concurrently; running several workers composes too. Side
  options on its command line override the config.toml values for
  this run (e.g. a different `--llm-review-cmd`; before a `--prs` /
  `--issues` marker they apply to both sides, after one to that side).
- `fairy_tui.py` (optional): a pure view; every key is a file
  operation on the same db.

A human moves any reviewed verdict out whenever they choose (`y` in the
TUI, `agent.py --ask` for the classic per-verdict terminal prompt, or
plain `mv reviewed/pr-N.json outgoing/`); with `--approve` in a side's
options the agent promotes actionable verdicts itself.

The reviewer (`pr_review_wrapper.py`) receives the PR data and returns one
structured JSON review. Inside it runs a pipeline: an optional cheap triage
pass (skip / helpful reply / engage; it can also honor the PR author's model
requests within `--allowed-model`), then one or more model reviewers (run
concurrently), then -- when there is more than one -- a combiner model that
verifies and merges the drafts. Every backend (OpenAI, Anthropic, z.ai GLM)
implements the same small `Reviewer` interface (`llm_review_api.py`):
`review(ctx) -> Review`. The stages are plain Python in `review_pr`, so
adding, removing or reordering them is an edit, not a restructuring.

### Using fairy with your project

1. Configure a gcli account for your fairy user.
2. Put API keys in the environment or `.env`: `OPENAI_API_KEY`, plus
   `ANTHROPIC_API_KEY` / `ZAI_API_KEY` when those providers are used
   (`pip install anthropic` only then; OpenAI-only deployments don't need it).
3. Clone your repository next to fairy and add the PR head refs
   (used for patch generation and available inside the review container):

       git config --add remote.fforge.fetch '+refs/pull/*/head:refs/remotes/fforge/pr/*'
       git fetch fforge

4. Write a project-facts file: what your build/test system is and the
   project-specific review rules the models must know. See
   `project_facts/*.md` for examples; pass yours with `--project-facts`.
5. Provision a podman host (below) so the models get a shell.
6. Copy `fairy-ref.sh` to your own launcher and adjust `--owner`, `--repo`,
   `--forge-type`, `--gcli-account`, `--patch-repo`, the models, and your
   forge's labels (`--triage-label`, repeatable). To run fairies for several
   repositories concurrently, give each launcher its own `--cache` and
   `--debug-response-dir`.

### Using fairy with GitHub

Pass `--forge-type github` and pick one of two credentials:

* **A token.** A classic or fine-grained PAT in a gcli account, selected
  with `--gcli-account`. Fairy posts as that account, so it cannot approve
  pull requests the account opened itself (GitHub refuses self-approval).
* **A GitHub App.** `--github-app-id` and `--github-app-key <pem>`.
  Installation tokens are minted and renewed automatically;
  `--github-app-installation` is only needed when the app has more than one
  installation. `--self-login 'your-app[bot]'` is required -- installation
  tokens cannot read `/user`, and without its own login fairy re-posts its
  comments on every run. The app is its own identity and can approve
  anyone's pull requests. Permissions: Issues and Pull requests read+write,
  Contents read, Checks read; newly added permissions count only once the
  installation owner accepts them.

CI is read from both the Checks API (GitHub Actions reports only there) and
the commit-status endpoint (third-party CI) and merged.

### Interactive TUI

`fairy_tui.py` (requires `pip install blessed watchdog`) shows the
filedb of one or more repositories in a 4-pane terminal UI: statistics
(the per-state file counts), the ticket list, a merged tail of the
agent/worker log files, and the rendered review message with its label
changes. It is a pure view -- start the agent and worker processes
separately and point the TUI at the same db roots (`configurator.py`
writes a `config.toml` into each root naming the repo and the log
files, which are tailed automatically; `--tail FILE` adds extras):

    ./fairy_tui.py --db-root '<a repo's filedb root>' \
        --db-root '<another repo's filedb root>' \
        --log-file fairy_tui.log

The list is a table over the ticket files: select any row and act on
it at any time; every action is a rename or a `requests/` file, so the
UI can quit and restart freely. The full pane, key, lens and mouse
reference lives in [README-FAIRY-UI.md](README-FAIRY-UI.md); see
`fairy-ui-ref.sh` for a launcher example.

### Self-hosted Podman container

The LLM's shell tool runs in an ephemeral Podman container on an ssh host
(local, a VM, or in the cloud). Each review gets a fresh container (no reuse)
with a full dev toolchain and internet access (restrictable via iptables).
Repos are filled from host-local mirrors, so no multi-hundred-MB `.git`
crosses the wire per review.

Prerequisite, once, as root on the host: install `podman`. Nothing else --
no `podman.socket`, no lingering; fairy uses rootless podman purely over ssh
(`ssh DEST podman ...`).

Provision the host (build the image + seed one bare mirror per repo);
idempotent, safe to re-run, and best called at the top of your launcher so
the image and mirrors stay current:

    ./containers/provision_remote.py --ssh fairy@HOST \
        ~/forgejo_fairy/ffmpeg ~/forgejo_fairy/all_ffmpeg

A review run never builds the image itself: it errors if the image is missing
and otherwise reuses it as-is. Repo commits need no re-provision; each review
syncs the checkout's current HEAD into the mirror automatically.

Then point the reviewer at the host, either directly:

    ./pr_review_wrapper.py --podman --shell-host fairy@HOST \
        --repo-root ~/forgejo_fairy/ffmpeg \
        --extra-repo-root ~/forgejo_fairy/all_ffmpeg ...

or via the pipeline, which injects `--podman --shell-host` for you:

    ./configurator.py ... --prs ... --podman-host fairy@HOST

Add `--podman-ssh-identity KEY` if the key is not offered by your ssh agent /
`~/.ssh/config`. A non-standard ssh port goes in the host spec, e.g.
`--podman-host fairy@HOST,port=17022` (same for `--shell-host` and
`--codex-host`); provision such a host with `provision_remote.py --port`.
The LLM's shell runs over a single persistent
`ssh DEST podman exec -i` pipe into an in-container agent
(`containers/fairy_agent.py`) speaking a small JSON protocol, so there is no
per-command ssh handshake and no shell-quoting of model output.

### Ensemble (multiple models + verify/combine)

`--model` is the main pass; add more reviewers with
`--extra-model PROVIDER:MODEL` (repeatable) and merge with
`--combine-model PROVIDER:MODEL` (required once there is more than one
reviewer). Provider prefixes: `openai:`, `anthropic:`,
`zai:`, `codex:`. Each reviewer gets its own isolated container shell; the
model reviewers run concurrently (`--concurrency PROVIDER:COUNT` caps how
many calls one provider gets at a time, across every fairy process on the
machine). A local-GPU backend is TODO -- PRs very welcome.

    ./pr_review_wrapper.py \
        --podman --shell-host fairy@HOST \
        --repo-root ~/forgejo_fairy/ffmpeg \
        --triage-model openai:gpt-5.4-mini \
        --model openai:gpt-5.4 \
        --extra-model anthropic:claude-opus-4 \
        --extra-model zai:glm-5.2 \
        --combine-model openai:gpt-5.4

### Codex backend

`codex:MODEL[@EFFORT]` (e.g. `codex:gpt-5.6-sol@high`; efforts `none`, `low`,
`medium`, `high`, `xhigh`, `max`, `ultra`) runs the pass through the codex CLI.
Codex runs in an ephemeral container on `--codex-host` (a podman host, same
spec syntax as `--shell-host`), never on the wrapper host. A `codex:` spec
without `--codex-host` is a hard error at startup; there is no local codex.
Through the pipeline, pass `--codex-host` and `--codex-home` in the
configurator's `--prs` section; they are forwarded to the wrapper
alongside `--podman-host`/`--shell-host`.

Build the thin codex image once from `containers/Containerfile.codex` (bakes a
pinned codex binary; `--codex-bin` is its in-container path, `--codex-image`
its tag) and `codex login` once as the bot's own account. `--codex-home DIR`
is the wrapper-side login: its `auth.json` is `podman cp`'d into the container
per run and its `models_cache.json` is used to harden the tool catalog.

The security model matches the API backends -- the model can execute only
inside the review containers, never on the wrapper host: codex runs with
as much disabled as possible in a separate container. Codex's own egress
is limited to the OpenAI API + token refresh. There is no direct connection
between the codex and review containers. Cap concurrent passes with
`--concurrency codex:N`.

### Static data and vector stores

Each set of static data given to the models is a git repository: source
repos, manually maintained ones, or generated ones (e.g. a tool dumping every
bug-tracker issue as a JSON file). With `--use-vector-store-search` each
repo's HEAD is indexed into an OpenAI vector store for `file_search`.

OpenAI allows max 2 vector stores per request, so you cannot attach one store
per repository; aggregate the rest as subtrees of one repo (that is what
`all_ffmpeg` is). Be warned that OpenAI vector stores are fragile: uploading
the wrong file can poison them. That is an OpenAI bug, not something we can
fix.

### OpenAI cloud containers (deprecated)

The OpenAI-hosted container backend still works but Podman is preferred.
OpenAI containers allow max 1000 files, restrict network egress and CPU time,
and thus need bare (no checkout) repos with compressed git structures; a
model checking out a large tree may never be heard from again.

## Simulating past PRs (offline replay / A-B)

`simpast-runs/run.sh` replays historical PRs through the real reviewer under
`--simulate-past`, so prompt / model / container-backend changes can be
compared without touching the live forge. It reviews a set of PRs at a fixed
cutoff against a cutoff-prepped mirror (a clone with master rewound to
`CUTOFF` and each PR's head pinned at the `--patch-pr-ref-template` ref),
runs N samples per "arm" (a git ref whose `llm_prompt.py` is swapped in), and
writes each cell's stream to `OUTROOT/<arm>_<i>/run.log` plus every OpenAI
call's JSON to `openaidebug/`.

`--simulate-past` rewinds git history but cannot time-travel the live forge:
PR open/closed state and CI status are read as they are now. Most replayed
PRs have since merged, so the harness passes `--force-review-non-open`;
without it they would skip as `not open`. Swap in currently relevant PRs
(with a matching mirror and `CUTOFF`) as old ones age out.

Config is via env (the script header lists all knobs); defaults are the
standard 4-PR FFmpeg suite:

    PRS, CUTOFF, PATCH_REPO, EXTRA_REPO   what to replay
    SAMPLES, PAR                          repeats and concurrency
    BACKEND, PODMAN_SSH                   openai container (default) or a
                                          Podman host (see above)

    # 4-PR suite, 3 samples, OpenAI container, current prompt
    OUTROOT=simpast-runs/openai SAMPLES=3 bash simpast-runs/run.sh HEAD

    # same in the self-hosted Podman container; PAR=1 keeps a single
    # host from being swamped
    OUTROOT=simpast-runs/podman SAMPLES=3 PAR=1 \
      BACKEND=podman PODMAN_SSH=fairy@HOST \
      bash simpast-runs/run.sh HEAD

## Mail Fairy

Scans a maildir from a mailing list and posts replies onto the matching pull
request and issue threads via gcli. It detects replies from In-Reply-To
header trees (so it needs more than one mail), checks and caches previously
posted comments to avoid duplicates, can strip or skip full quotes, strips
mailman footers, and prefixes each message with author, date and a clickable
link to the mailman3 / lore archive.

    ./mail_fairy.py \
        --maildir ~/mail/ffmpeg/dev \
        --owner FFmpeg --repo FFmpeg \
        --forge-base-url https://code.ffmpeg.org \
        --gcli-account mf \
        --max-age-days 14 \
        --manual -v

## Tools

`tools/match_prs_to_master.py` tells you which commits and/or PRs have
already landed on a base branch, using hash, patch-id, subject and dates, so
it recognizes slightly amended patches while favoring better matches. It only
needs git (and the fforge PR refs above); with `--gcli` it also asks the
forge for each PR's real open/closed/merged state (slower).

    match_prs_to_master.py \
        --base origin/master \
        --pr-glob 'refs/remotes/fforge/pr/*' \
        --release-glob 'refs/remotes/fforge/release/*' \
        --since '6 months ago' \
        --verbose

## Supported forges

* Forgejo
* GitHub
* GitLab (untested)
* Gitea (untested)

## License

GPL v2, with removable clause that allows us to change to a different version
of the *GPL in case that's where the community wants to go.
