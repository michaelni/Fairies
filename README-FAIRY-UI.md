# fairy_tui.py — the operator's table over the filedb

The TUI is a pure view. The agent scans the forge and sends verdicts,
the worker runs the LLM reviews; both write ticket files into the
filedb, and every row here is one of those files — the state IS the
directory name. The TUI never talks to the forge or the LLM: each key
is a file rename or a small `requests/` file, so it can be quit and
restarted at any time without losing anything, and it can run beside
any number of agents and workers or none at all.

    ./fairy_tui.py --db-root '<a repo's filedb root>' \
        --db-root '<another repo's filedb root>' \
        --log-file fairy_tui.log

One `--db-root` per repo; the PR and issue sides of one repo share a
filedb. `configurator.py` writes a `config.toml` into the root naming
the repo and the sides' `--log-file`s, which are tailed into the log
pane automatically;
`--tail FILE` adds extra files, and
`--save-dir DIR` sets where `e`/`E` exports land (default: cwd).
`fairy-ui-ref.sh` is a complete three-process launcher example.
Requires `pip install blessed watchdog`.

## Panes

Four panes, dividers draggable with the mouse, `Tab` or a click moves
focus, arrows / `PgUp` / `PgDn` scroll the focused pane, `Home` /
`End` jump to its top / bottom (on the list: cursor to the first /
last row).

- **Σ stats** — elapsed time, how many reviewed verdicts await you,
  and per-repo counts per state. `merge-ready=N/M` folds
  `awaiting-approver` in: `M` PRs need a human to press Merge, of
  which `N` were approved by fairy (the rest, shown as `merge-ready*`
  in the list, were approved by someone else). A red `PAUSED` appears
  while `p` holds the daemons stopped.
- **☰ list** — the table. Columns: `▶` marks reviewed rows, kind,
  repo (only when sides span several), `#number`, state, llm, age,
  title. The title bar shows the active lens, the bottom bar the keys.
- **≣ logs** — merged tail of every side's agent/worker log plus the
  TUI's own, one padded source tag per line, level-colored.
- **¶ message** — the selected ticket: its review and the action `y`
  would perform, then the discussion. The thread is kept current by
  the agent's scan; a `── sampled … ──` line marks how far the review
  saw. The byline's branch is the PR's head branch — in the author's
  fork unless it lives in the reviewed repo.

## The list

**States** are the filedb directories: `requests`, `queued`, `llm`
(the llm column names the wrapper stage: triage/review/combine),
`reviewed` (awaiting you), `outgoing` (you pressed y; the send pass
posts it), `posted`, `skipped`, `cancelled`, `ci-blocked`,
`merge-ready`, `awaiting-approver` (displayed `merge-ready*`),
`error`, plus `invalid` for an unparsable file. A cancelled row whose
PR turned out merged displays as `merged`. A state suffixed `?` means
the last poll found the file in no directory — almost always a poll
racing a rename; the row dies only after 10 consecutive misses (both
events are logged).

**The llm column** doubles as a status column: the wrapper stage while
in `llm/`, `requested` while an `r`/`R`/`f` request is pending,
`appr=<age>` on merge-ready rows (how long the merge has waited),
`err=<age>` on error rows (how old the failure is), otherwise the
verdict classification.

**Lenses** (`a` cycles): `relevant` (default — everything except
settled rows you never interacted with), `review` (the y-session:
reviewed plus queued/llm/outgoing on their way in and out), `merge`,
`ci`, `actionable`, `all`.

**Sorts** (`t` cycles): `arrival`, `status` (actionable first, then
the live pipeline, attention, settled), `repo`, `number`.

**The cursor is a key, not an index.** It stays on its PR/issue no
matter how the list reorders, and the highlight is only ever drawn on
that row. If the current lens hides the cursor's ticket (or it is
mid-rename for a tick), no row is highlighted, actions refuse with a
log line, and the message pane keeps showing the ticket so you can
see where it went; the first arrow press summons the cursor back at
its last screen spot. With no selection at all the first listed
ticket is adopted.

## Keys

| key | action |
|-----|--------|
| `y` | apply: hand the reviewed verdict to the agent's send pass (`reviewed/` → `outgoing/`). The send re-checks that the PR is unchanged since the review; a mismatch returns the ticket with a `send blocked:` note and, in manual mode, parks it for you — `r`, `s` or `Y` are the answers. |
| `Y` | post anyway: like `y` but waives the staleness guard once. For a verdict you have read and judged still valid. |
| `s` | skip now: one-shot; the next scan reconsiders the item afresh (earned backoff history is kept). |
| `S` | snooze: like `s` but the item waits out a doubling backoff (min 24h) before it is reconsidered; new PR activity bypasses the wait. |
| `r` | request a fresh, gate-bypassing review of the item. `2r`..`9r` request that many parallel sample evaluations (slots `s1..sN` — refilling slots clobbers earlier samples in them). Refused while the item is in flight. |
| `R` | one more evaluation: takes the next FREE sample slot, never touching the base verdict or earlier samples. Each press adds one; capped at `s9`. |
| `f` | currently identical to `r` (historic: force). |
| `x` | drop: ticket → `cancelled/`; sticks until new PR activity. On an `llm/` row it cancels the running review: the wrapper stops at the next shell call and the container is torn down. |
| `o` | edit the review message in `$EDITOR` (markdown round-trip; refused while a worker holds the ticket). |
| `p` | pause: SIGSTOP every agent/worker the launcher started, with their whole subprocess trees; `p` again resumes. Remote containers keep computing — only local processing freezes. Quitting while paused thaws first. |
| `a` / `t` | cycle lens / sort. |
| `/text` | search number, title and state; `Enter` jumps, `Esc` cancels, `n` repeats. |
| `e` / `E` | export the focused pane as painted / in full to `--save-dir`. |
| `0-9` | count prefix for `r` and for arrow scrolling. |
| `q` | quit. |

## Mouse

- **Click a row** to select it; click a pane to focus it.
- **The wheel scrolls the view**, never the cursor; the window snaps
  back to the cursor on the next cursor key.
- **Click-to-copy:** a URL, git hash, `#number`, or the byline's
  author / branch value copies to the primary selection (X11
  middle-click), falling back to the OSC 52 clipboard over a plain
  ssh session — in tmux turn `set-clipboard` on. `#123` copies the
  bare number.
- **⧉ in the message pane's title:** a click there copies the whole
  raw markdown review message (what `o` edits) — ready to paste into
  a mail or forge comment.
- **Dividers** drag.

## What the TUI never does

Post, review, or fetch. `y`/`Y` only stage a ticket for the agent's
send pass, which re-validates before posting; `r`/`R`/`f` only write a
request file the agent answers with a fresh ticket. If the TUI dies
mid-anything, the files are exactly where they were.
