## Mail Fairy

Scans a maildir from a mailing list. She will then with GCLI post replies to pull request and issue
threads to a forge (Forgejo, Gitlab, Github, Gittea).
She detects replies by analyzing in-reply-to header trees and thus needs more than just 1 mail
She will check and cache all previously posted comments to avoid duplicates.
She will also check for "full quotes" and can be configured to remove these quotes or skip affected mail
mailman footers will be stripped, messages will be prefixed by author and date and should be a clickable
link to a mailman3 mailinglist (lore supported too)

Send pull request if something doesnt work or looks ugly or is aisloppy. Make sure its read/reviewable and tested!
(Github/Gitlab untested but GCLI supports them so they should work)

#### Example:

./mail_fairy.py \
    --maildir ~/mail/ffmpeg/dev \
    --owner FFmpeg --repo FFmpeg \
    --forge-base-url https://code.ffmpeg.org \
    --gcli-account mf \
    --max-age-days 14 \
    --manual  -v



## Forgejo Fairy

Note: Forgejo Fairy is under heavy development and this codebase has not been cleaned up yet!

Forgejo Fairy, is a Reviewer and Assistent for Forgejo and other development forges.
Future plans includes helping with bugs too.

High level structure of teh tools

### The main tool

    It goes through Pull requests, using GCLI and a cache, performs a series of checks to prefilter
    than passes PRs to the LLM input que
    LLM workers take PRs out of the que and pass them to the reviewer
    the result is passed to the UI que
    The UI thread allows a human to defer, accept, skip, redo PRs or to automatically accept all
    It uses GCLI to communicate with teh forge

### The reviewer

    receies data to review and output structured json representing the review
    It can be implemented using cloud LLMs, local LLMs, or even communicating with snail mail and humans
    The reviewer is simply a wrapper called by the main script

    Every backend (OpenAI, Anthropic, z.ai GLM) implements the same small
    `Reviewer` interface (`llm_review_api.py`): `review(ctx) -> Review`.
    The wrapper composes them imperatively in `review_pr`: optional triage,
    then one or more model reviewers (run concurrently when there is more
    than one), then an optional combiner that verifies and merges their
    drafts into the final review. Stages are plain Python, so adding,
    removing, or reordering them is an edit to `review_pr`, not a
    restructuring of the codebase.


#### OpenAI Reviewer

    Supports openAI (deprecated) and podman containers.
    A triage pass is done to judge if the PR needs a review, or something else or its best to skip
    If the triage determines a review is needed then the GPT model choosen performs a review.
    The review model is choosen by the Triager

##### openAI Vector stores

    If you want to use the vector stores, keep in mind that openAI allows max 2 vector stores
    so you cannot do 1 vector store per git repository if you have more than 2. Also be warned
    the openai vector stores are fragile, uploading the wrong file can poision them. This is a
    openai bug, not something we can fix.

##### openAI Containers

    The openAI reviewer sets up a openai container with all the repositories (with all the caching and retrying needed)
    openAI containers have many limitations (like max 1000 files in them, network egress and cpu time are restricted by openAI)
    If you intend to use openAI containers with a moderate to large git repository then you must setup a bare git (no checkout)
    and compress the internal git structures to stay below 1000 files. Also if the GPT model tries a checkout exceeding 1000 files
    you may or may not hear from it again.


##### Self-hosted Podman container

    Short docs:
    Simple provide a empty ssh account and have podman packages installed on
    the host. Fairy uses rootless padman. It will setup everything else.

    More details:
The LLM's shell tool can run in an ephemeral Podman container on an ssh
host (local, a VM, or in the cloud). Each review gets a fresh container (no reuse) with a full dev
toolchain and internet access. (You can restrict network through iptables) Repos are filled from
host-local mirrors, so no multi-hundred-MB `.git` crosses the
wire per review.

Prerequisite, once, as root on the host: install `podman` and the dev
packages. Nothing else -- no `podman.socket`, no lingering. The bot
talks to podman purely over ssh (`ssh DEST podman ...`).

Provision the host (build the image + seed one bare mirror per repo);
idempotent, safe to re-run:

    ./containers/provision_remote.py --ssh fairy@HOST \
        ~/forgejo_fairy/ffmpeg ~/forgejo_fairy/all_ffmpeg

Rebuild just the image: `./containers/build_image.py --ssh fairy@HOST
--force` (without `--force` an existing tag is left as-is).

Re-run provisioning whenever `containers/Containerfile` changes: a review
run never builds or rebuilds the image -- it errors if the image is
missing and otherwise reuses it as-is. Repo commits need no re-provision;
each review syncs the checkout's current HEAD into the mirror
automatically. Tip: call `provision_remote.py` at the top of your
launcher script so the image and mirrors stay current with one command.

Then point the reviewer at the host, either directly:

    ./openai_pr_review_wrapper.py --podman --podman-ssh-dest fairy@HOST \
        --repo-root ~/forgejo_fairy/ffmpeg \
        --extra-repo-root ~/forgejo_fairy/all_ffmpeg ...

or via the main tool, which injects `--podman --podman-ssh-dest` for you:

    ./fairy.py ... --podman-host fairy@HOST

Add `--podman-ssh-identity KEY` if the key is not offered by your ssh
agent / `~/.ssh/config`. Lifecycle steps (image/network/run/cp/rm) are
one `ssh DEST podman ...` each; the LLM's shell runs over a single
persistent `ssh DEST podman exec -i` pipe into an in-container agent
(`containers/fairy_agent.py`) speaking a small JSON protocol, so there is
no per-command ssh handshake and no shell-quoting of model output.

#### Anthropic / GLM Reviewer

The Anthropic reviewer (`anthropic_reviewer.py`) speaks the Messages API:
the model investigates through the same shell tool (a fresh ephemeral
Podman container per reviewer) and returns its verdict by calling a
`submit_review` tool whose schema is the shared `REVIEW_SCHEMA`. z.ai's
GLM is the same reviewer pointed at z.ai's Anthropic-compatible endpoint.

Keys are read from the environment or `.env`: `ANTHROPIC_API_KEY` for
Anthropic, `ZAI_API_KEY` for GLM. The `anthropic` package is only needed
when an Anthropic/GLM model is actually used (`pip install anthropic`);
OpenAI-only deployments do not need it.

#### Ensemble (multiple models + verify/combine)

Run several models on the same PR and have a final model verify and merge
their reviews. `--model` is the OpenAI main pass; add more with
`--extra-model PROVIDER:MODEL` (repeatable), and merge with
`--combine-model PROVIDER:MODEL` (required once there is more than one
reviewer). Each reviewer gets its own isolated container shell, and the
non-triage model reviewers run concurrently.

    ./openai_pr_review_wrapper.py \
        --podman --podman-ssh-dest fairy@HOST \
        --repo-root ~/forgejo_fairy/ffmpeg \
        --triage-model gpt-5.4-mini \
        --model gpt-5.4 \
        --extra-model anthropic:claude-opus-4 \
        --extra-model zai:glm-4.6 \
        --combine-model openai:gpt-5.4

Provider prefixes: `openai:` (or a bare model name), `anthropic:`, `zai:`.

#### Local GPU Reviewer
    TODO / PR VERY welcome

## High level structure of teh data

    Whats provided to teh reviewer
    A Prompt
    A data bundle
    Static data
        Each set of static data is in a git repository, in case of openai, HEAD is loaded into a vector store
        the full hostory is provided in the openai container
        These git repository can be
        * source repositories,
        * manually maintained repos
        * created by some tool that turns all issues from a bug tracker into a git repo of json files for example

    all_ffmpeg
        This is to workaround the current openai API limitzation of max 2 vector stores
        this simply contains all repositories except main ffmpeg as subtrees


## Simulating past PRs (offline replay / A-B)

`simpast-runs/run.sh` replays historical PRs through the real reviewer
under `--simulate-past`, so prompt / model / container-backend changes
can be compared without touching the live forge. It reviews a set of PRs
at a fixed cutoff against a cutoff-prepped mirror, runs N samples per
"arm" (a git ref whose `llm_prompt.py` is swapped into the working tree),
and writes each cell's live stream to `OUTROOT/<arm>_<i>/run.log` plus
every OpenAI call's JSON to `openaidebug/`.

A cutoff-prepped mirror (`PATCH_REPO`) is a clone with master rewound to
`CUTOFF` and each replayed PR's head pinned at the
`--patch-pr-ref-template` ref.

`--simulate-past` rewinds git history but cannot time-travel the live
forge: a PR's open/closed state and CI status are read as they are now,
not as of `CUTOFF`. Most replayed PRs have since merged or closed, so the
harness passes `--force-review-non-open`; without it they would skip as
`not open` (the symptom is a suite that returns mostly `SKIP`). The
default 4-PR suite is also not eternal -- swap in currently relevant PRs
(with a matching mirror and `CUTOFF`) as old ones age out.

Config is via env (the script header lists all knobs); defaults are the
standard 4-PR FFmpeg suite. Key knobs:

    PRS, CUTOFF, PATCH_REPO, EXTRA_REPO   what to replay
    SAMPLES, PAR                          repeats and concurrency
    BACKEND, PODMAN_SSH                   openai container (default) or a
                                          Podman host (see above)

To speed a run up, lean on concurrency: each sample's PRs review in
parallel, and samples within an arm run in parallel up to `PAR` (default
`PAR=SAMPLES`, one wave), so on the OpenAI backend a whole arm finishes in
roughly one review's wall-clock -- raise `SAMPLES`/`PAR` freely there. For
`BACKEND=podman` keep `PAR` small so a single host is not swamped.

Examples:

    # 4-PR suite, 3 samples, OpenAI container, current prompt
    OUTROOT=simpast-runs/openai SAMPLES=3 bash simpast-runs/run.sh HEAD

    # same suite in the self-hosted Podman container (provision the host
    # first); PAR=1 keeps a single host from being swamped
    OUTROOT=simpast-runs/podman SAMPLES=3 PAR=1 \
      BACKEND=podman PODMAN_SSH=fairy@HOST \
      bash simpast-runs/run.sh HEAD


## Tools

### Matching PRs to a branch

`tools/match_prs_to_master.py` tells you which Commits and or PRs have already landed
on a base branch. It uses hash, patchid, subject and commit and author dates.
So it can still recognize slightly amended patches but favors better matches.
It only needs git; with `--gcli` it also asks the
forge for each PR's real open/closed/merged state (slower).

First fetch the PR head refs the tool scans, e.g. for FFmpeg's forge:

    git config --add remote.fforge.fetch \
        '+refs/pull/*/head:refs/remotes/fforge/pr/*'
    git fetch fforge

Example:

    match_prs_to_master.py \
        --base origin/master \
        --pr-glob 'refs/remotes/fforge/pr/*' \
        --release-glob 'refs/remotes/fforge/release/*' \
        --since '6 months ago' \
        --verbose

## Supporte Forges
* GitHub (untested)
* GitLab (untested)
* Gitea  (untested)
* Forgejo

## License
GPL v2, with removable clause that allows us to change to a different version of the *GPL in case thats where the community wants to go
