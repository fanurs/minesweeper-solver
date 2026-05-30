# Observed live DOM behavior (minesweeper.online)

Ground-truth behaviors observed by recording real play with `tools/dom-trace` and
inspecting the frames. These are **empirical** (what the site actually does) and
supersede earlier guesses. Primary source:
`fixtures/recordings/2026-05-30-mixed/` (893 frames, full-page, ~50 ms polling).

## Cell classes (complete taxonomy)

Only these state classes occur on cells:

| classes | meaning | `.msm` code |
|---|---|---|
| `hd_closed` | covered | `0x09` |
| `hd_closed hd_flag` | flag (a flag is a *closed* cell **plus** the `hd_flag` overlay) | `0x0A` |
| `hd_opened hd_type0` … `hd_type8` | opened number (0 = blank) | `0x00`–`0x08` |
| `hd_type10` | revealed mine (loss; one **not** clicked) | `0x0B` |
| `hd_type12` | **detonated** mine (a mine that was clicked / chorded into) | `0x0C` |
| `hd_type11` | wrong flag (flagged a non-mine; X on loss) | `0x0D` |

- **A flag is `hd_closed hd_flag`** — flags are a *subset* of closed cells, so the
  parser must test `hd_flag` before `hd_closed`, and any raw "closed" count
  includes flags.
- No `hd_type9` or `hd_type13+` ever observed. `hd_type5/6/7/8` are valid but rare
  (didn't occur this session).
- **Question marks (`?`) do not occur** — the site's "Question marks" setting is
  disabled and stays off; there is no `?` cell class to handle (out of scope).
- Decorative overlay classes may be appended to a cell (e.g. `cell-ticket-flower`,
  see `snapshots/expert_loss_01.html`). Read state from the known tokens only;
  ignore extras (never `0xFF`).

## Loss reveal

- **Atomic.** On a loss, *all* unflagged mines reveal in a single frame as
  `hd_type10`; correctly-flagged mines stay `hd_closed hd_flag`. No partial /
  animated reveal was caught even at 50 ms. Invariant on a loss frame:
  `count(hd_type10) + count(hd_flag) + count(hd_type12) == total mines`
  (verified: a 40-mine Intermediate loss summed to exactly 40; a 12-mine board to 12).
- **The detonated mine(s) are `hd_type12`, and there can be MORE THAN ONE.**
  - A single-click loss detonates **1** cell.
  - A **chord** (left-click on a number) that opens several mines at once
    **detonates all of them** → `hd_type12` × N. Observed ×3 in one 9×9 loss
    (recording frames 180–184). **⇒ the `.msm` `0x0C` code must allow N detonated
    cells per loss, not exactly one.**
- **`hd_type12` persists for the whole loss screen** (seen across 5 consecutive
  50 ms frames ≈ 250 ms, until the next game) — *not* a sub-frame flash. It was
  missed in an earlier 150 ms capture only because the player restarted before a
  poll landed. ⇒ capture the reveal **event-driven at the loss instant**, not on a
  slow timer.
- **Wrong flags** become `hd_type11` (one X per misplaced flag); they are not mines
  and are excluded from the mine total above.
- The complete mine map is always derivable:
  `mines = hd_type10 ∪ hd_flag(correct) ∪ hd_type12`.

## Win

- The site **auto-flags every remaining mine** (`hd_closed hd_flag`), even ones the
  player never flagged, and opens all non-mines. On a won board
  `count(hd_closed) == count(hd_flag)` (every covered cell is a flag) and there are
  no `hd_type10/11/12`. Mine map = the flagged cells.

## Game reset

- Starting a new game is a single-frame transition: the finished board is replaced
  by an all-`hd_closed` fresh board in one frame (no intermediate state).

## No-guessing mode

- It is a **game-creation mode**, not a board state — started via the menu item
  (`<i class="fa fa-graduation-cap"></i> No guessing mode` → `executeUrl('new-game/ng')`).
- **The board cells are byte-identical to a normal game**, but the mode *is*
  detectable from the surrounding **page UI**: in no-guess mode the page shows a
  **Hint button (`#hint_btn`)** and the nav link `link_new_game_ng` is `active`.
  (The "No guessing mode" *menu item* is on every page and is **not** an
  indicator.) So to record the mode, read `#hint_btn` / the active nav — not the
  cells. Verified in `fixtures/recordings/2026-05-30-mixed/` (Standard f16–628,
  no-guess f629–797).

## Top bar (counter / timer / face)

- Mine counter and timer are 7-segment digit groups using `hd_top-area-num{0-9|-}`
  inside `#top_area`. The timer increments ~1 Hz; exclude `#top_area` from any
  change-detection that should not fire on the clock.
- Face: `#top_area_face` with `hd_top-area-face-unpressed | -win | -lose`.

## Click cadence (this player)

- Median ~125 ms between board-changing actions (~8/sec), bursting to ~33 ms.
  (Counts board changes, so raw click rate is a little higher.) Consistent with the
  cursor-pilot's ~6 CPS sustained / higher peaks.
