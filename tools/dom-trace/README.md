# dom-trace — minesweeper.online recorder + fixture packer (dev tool)

A small **development-only** instrument. It is **not** part of the shipped
extension (`extension/`) and is never bundled or distributed — it exists only to
learn how the live site's DOM behaves and to capture session fixtures
(`fixtures/recordings/`).

It **launches its own throwaway private Edge**, opens minesweeper.online, and
saves the **full-page HTML** every time the board or surrounding UI changes (the
ticking timer / flag counter are ignored, so it doesn't fire every tick).
Recordings are then packed into a single, queryable **SQLite** file.

> **It never clicks anything.** The tool only opens a window and *observes* — you
> do all the playing. No input is ever synthesized.

## Setup

```sh
cd tools/dom-trace
npm install      # playwright-core (uses your installed Edge — no browser download) + tsx
```

Python helpers (pack / query) run via [`uv`](https://docs.astral.sh/uv/) —
standard library only, no Python dependencies.

## 1. Record

```sh
npm run record   # launches a private Edge at minesweeper.online; ~150 ms poll
```

With options — pass them to `tsx` directly (`npm run` swallows `--flags`):

```sh
npx tsx record.ts --interval 50 --label mixed
#   --interval <ms>  poll period (default 150)
#   --label <name>   output-folder prefix (default "expert")
#   --edge <path>    msedge.exe (default C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe)
#   --out <dir>      output root (default %TEMP%\msm-traces)
#   --profile <dir>  Edge user-data-dir (default %TEMP%\msm-rec-profile)
#   --url <url>      start URL (default https://minesweeper.online/)
```

Play normally. **Stop by closing the Edge window** (or Ctrl-C) — it flushes and
exits, and closes nothing but the window it opened. Each run writes
`<label>-<timestamp>/` under `%TEMP%\msm-traces\` (outside the repo / Dropbox):
`frame-NNNNN-<hash>.html` (full page, one per change) + `index.ndjson`.

## 2. Pack into a committed fixture

```sh
uv run python pack_sqlite.py --src "%TEMP%\msm-traces\<label>-<ts>" \
                             --out ../../fixtures/recordings/<name>/recording.sqlite
```

Each frame is pretty-printed (one tag per line → granular diffs), gzip-compressed,
and stored keyed by frame number in one SQLite file.

## 3. Query / bookmark

```sh
DB=../../fixtures/recordings/<name>/recording.sqlite
uv run python query.py $DB info                        # frame count / range
uv run python query.py $DB show 180 --out /tmp/f.html  # one frame's HTML
uv run python query.py $DB diff 179 180                # per-cell unified diff
uv run python query.py $DB bookmarks                   # frame -> description

# bookmarks: edit a `frame<TAB>description` TSV, then load it:
uv run python load_bookmarks.py --db $DB --tsv ../../fixtures/recordings/<name>/bookmarks.tsv
```

## Files

| file | what |
|---|---|
| `record.ts` | launch private Edge; save full-page HTML on each board/UI change |
| `pack_sqlite.py` | a recording dir → one queryable `recording.sqlite` |
| `query.py` | `show` / `diff` / `bookmarks` / `info` |
| `load_bookmarks.py` | load a `frame → description` TSV into the DB |

## Notes

- Self-contained; `node_modules/` is gitignored — only source here is committed.
- Raw recordings are written to `%TEMP%` (not the repo); only the packed
  `recording.sqlite` (+ `bookmarks.tsv`) is promoted into `fixtures/recordings/`.
- Distilled findings from captured sessions live in `docs/dom-behavior.md`.
