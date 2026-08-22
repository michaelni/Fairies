# Fairy without an LLM -- the forge viewer

`agent.py` and `fairy_tui.py` need no model and no API key: leave
`--llm-review-cmd` unset and what is left is a multi-repo terminal view
of your forge -- every open PR and issue with its thread, labels,
approvals, CI and merge state -- that writes nothing back.

    ./configurator.py --db-root ~/.fairy/db/view \
        --owner OWNER --repo REPO --gcli-account ACCOUNT \
        --log-file logs/view.log \
        --prs --min-age-days 0 --scan-closed-days 7 \
        --issues --min-age-days 0 --scan-closed-days 7
    ./agent.py --db-root ~/.fairy/db/view --loop 6000 &
    ./fairy_tui.py --db-root ~/.fairy/db/view --log-file fairy_tui.log

A gcli account and `pip install blessed watchdog pygments`, nothing
else: no API key, no podman host; `worker.py` is the LLM half and
stays off. Steps 2, 4 and 5 of the README's "Using fairy with your
project" are the reviewer's and do not apply; step 3's `--patch-repo`
clone is optional here -- with it, `m` and `d` show every PR's patches
and merge diff. Everything in [README-TUI.md](README-TUI.md) works
unchanged, several repos included -- one `--db-root` each.

## What the view shows

Every open item, refreshed each agent pass:

- the table with the status letters (approvals, change requests,
  auto-merge, the issue's bug/repro/resolution labels,
  open/closed/merged), age, the lenses, sorts and `/search`;
- per item the whole thread -- description, comments, reviews, inline
  review comments, pushes and force-pushes, each with author, date and
  age -- and, with a `--patch-repo` clone, the `m`/`d` diff views;
- `e`/`E` to export any pane.

A few rows sort themselves into attention classes: `merge-ready` --
PRs the configured account has approved and nobody merged, with how
long the approval has waited (`appr=131d`); `merge-ready*` -- approved
by someone else; `awaiting-approver`; `ci-blocked` (one CI status
fetch per open PR -- `--min-age-days` exists to delay a review, which
a viewer does not have, so the example sets 0); `error` -- the item's
forge reads failed, retried after a doubling backoff of at least 24h.

`--scan-closed-days 7` keeps items closed or merged within the last
week in the view, their thread and status refreshed whenever they
still change; without it the view covers open items only.

Every other row shows state `skipped`: the state names come from the
review pipeline, and for a viewer "skipped" just means "nothing to
act on" -- most rows. The default `relevant` lens hides them, so
press `a` until the title bar says `all`: that is the viewer's
working lens.

## It cannot post

With no reviewer there is never a verdict: `y` refuses on every row
and nothing is ever written to the forge. `r`/`R`/`f` request a
review no worker exists to run; the row just returns to `skipped`.

## What a pass costs

One agent pass over FFmpeg/FFmpeg (429 open PRs, 289 open issues) on
2026-08-22, `--verbose 2`, one measurement each: 7m17s and 3328 `gcli
api` GETs against a cold cache, 2m09s and 593 GETs on the following
pass. Reads only; the pickle cache keyed on each item's `updated_at`
covers the difference. `--loop 6000` is a comfortable interval at that
size.
