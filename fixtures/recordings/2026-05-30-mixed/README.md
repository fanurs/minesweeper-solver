# Recording: 2026-05-30-mixed

A live session captured with `tools/dom-trace` (full-page HTML, ~50 ms polling,
private Edge). **893 frames** (`f0`–`f892`). A frame is saved whenever the board
**or surrounding UI** changes; the ticking timer / flag counter (`#top_area`) are
excluded, so frames are meaningful board/UI states, not clock ticks.

Contents: **24 game attempts** (Standard mode, then No-guessing) followed by a full
**website browsing tour** (every section + a 13-article Help read-through).
Distilled behavioral findings: [docs/dom-behavior.md](../../../docs/dom-behavior.md).

- **Standard mode** (`f16–628`): 5 wins, 6 losses, 1 abandoned.
- **No-guessing mode** (`f629–797`): 3 wins, 2 losses, 4 abandoned. *(No-guess
  begins at `f629` — detectable by the NG-only Hint button `#hint_btn` + active
  `link_new_game_ng`, **not** the ever-present "No guessing mode" menu item.)*
- Flagging: wins were typically fully flagged (peak flags == mine count); the fast
  losses used **no flags**.

## Files

- `recording.sqlite` — all 893 frames in one SQLite file, keyed by frame number;
  each frame's full-page HTML is pretty-printed and gzip-compressed. Fetch any
  frame instantly by number — no full extraction.
- `bookmarks.tsv` — `frame → description` (the table below); loaded into the DB's
  `bookmarks` table.
- `index.ndjson` — one line per frame: `{ n, t, hash, file }` (`t` = epoch ms).

## How to query it

Helpers in `tools/dom-trace/` (run with `uv`; standard library only):

```sh
DB=fixtures/recordings/2026-05-30-mixed/recording.sqlite
uv run python tools/dom-trace/query.py $DB show 180 --out /tmp/f180.html  # one frame's HTML
uv run python tools/dom-trace/query.py $DB diff 179 180                   # per-cell unified diff
uv run python tools/dom-trace/query.py $DB bookmarks                      # the list below
```

Or plain SQL: `SELECT html FROM frames WHERE n=180` (the blob is gzipped).

## Bookmarks — gameplay

| frames | board | mode | result | notes |
|---|---|---|---|---|
| 0–15 | Homepage | — | — | difficulty-select landing |
| 16–41 | Beginner | Standard | **WIN** @39 | flagged 10/10 |
| 42–78 | Beginner | Standard | LOSE @77 | no flags |
| 79–118 | Beginner | Standard | WIN @117 | flagged |
| 119–148 | Beginner | Standard | WIN @147 | flagged |
| 149–154 | Beginner | Standard | LOSE @153 | no flags |
| 155–184 | Custom 9×9, 12 mines | Standard | LOSE @180 | ★ **chord → 3 detonated** (`t12`×3) @180–184 |
| 185–187 | **Settings page** | — | — | "Start new game on middle-click" option on screen |
| 188–192 | Beginner | Standard | _abandoned_ | 0 clicks |
| 193–287 | Intermediate | Standard | LOSE @286 | ★ single detonation (`t12`×1) |
| 288–425 | Intermediate | Standard | WIN @424 | fully flagged 40/40 |
| 426–575 | Intermediate | Standard | WIN @574 | fully flagged |
| 576–587 | Intermediate | Standard | LOSE @586 | no flags |
| 588–601 | Intermediate | Standard | LOSE @600 | no flags |
| 602–628 | Expert | Standard | LOSE @626 | last Standard game |
| 629–632 | Custom 30×30, 150 mines | **No-guess** | _abandoned_ | **NG begins here** |
| 633–656 | Custom 5×5, 5 mines | No-guess | **WIN** | fully cleared |
| 658–667 | Beginner | No-guess | LOSE @666 | no flags |
| 668–702 | Beginner | No-guess | WIN @701 | flagged |
| 703–751 | Beginner | No-guess | WIN @750 | flagged |
| 752–753 | Beginner | No-guess | _abandoned_ | |
| 754–790 | Intermediate | No-guess | LOSE @788 | |
| 791–797 | Expert / Custom 600 / 5×5 | No-guess | _abandoned_ | rapid cycling; end of play |

## Bookmarks — browsing tour (`f798–892`, no gameplay)

Site sections, in order: **PvP** (798) · Ranking (802) · My games (810) · Best
players (813) · Season leaders (817) · Quests (821) · Arena (823) · Equipment
(825) · Marketplace (831) · Events (834) · Championship (838) · Players online
(841) · News (844) · Statistics (847) · My profile (853) · Chat (855).

**Help read-through** (`f856–885`), 13 articles in sequence: Gameplay (856) ·
Patterns (858) · Efficiency (860) · Trophies (862) · Arena (865) · Gems (867) ·
Equipment (869) · Events (871) · Quests (873) · Achievements (875) · Ranks (878) ·
Guides (881) · Website-rules (884). Then **Premium** (886) · **Shop** (888, end).

## Corrections vs the first (board-only) scan
- `f185–187` is the **Settings page**, not a "Custom-40 game" (the 40-cell/win
  reading was a stale board lingering behind the Settings overlay).
- **No-guessing is `f629–797` only**; `f16–628` are **Standard**. (You started in
  Standard — confirmed via `#hint_btn` presence, not the menu.)
- `f188–192 / f629–632 / f752–753 / f791–797` are **abandoned** boards (0 clicks).
- The custom 5×5 at `f633–656` is a **WIN**.
- The whole `f798–892` **browsing tour + 13-article Help read-through** was missing.

## Caveats
- **Settings toggle state** isn't serialized in the static HTML (JS-managed) — we
  can confirm the Settings page was open and the middle-click-reset option shown,
  but not whether it was actually enabled.
- **No-guess win faces** weren't captured at the win instant (face stayed blank);
  those wins are inferred from "all safe cells opened, 0 mines revealed."
- Game-boundary frames are ±1 where a board cleared between 50 ms polls.
