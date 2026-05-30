# Fixtures

HTML snapshots from minesweeper.online used to develop and test the DOM parser.

## How to capture a fixture

1. Go to https://minesweeper.online and start a game
2. Play until you reach an interesting state (mid-game, win, loss, etc.)
3. Open DevTools → Elements tab
4. Find the outermost element that contains the full board
   - Right-click any cell → Inspect, then navigate UP until you see the entire board wrapper
   - You want the element that includes: the mine counter, timer, smiley button, AND the cell grid
5. Right-click that element → **Copy → Copy outerHTML**
6. Paste into a new `.html` file in this directory

## Naming convention

```
{difficulty}_{state}_{index}.html
```

| Part | Values |
|------|--------|
| difficulty | `beginner`, `intermediate`, `expert`, `custom` |
| state | `fresh` (no cells opened), `midgame`, `win`, `loss`, `overflagged`, `midgame_pressed` |
| index | `01`, `02`, … (if multiple of same type) |

Examples:
- `expert_fresh_01.html`
- `expert_midgame_01.html`
- `expert_win_01.html`
- `expert_loss_01.html`
- `expert_overflagged_01.html` — more flags placed than total mines (counter shows negative)
- `intermediate_midgame_pressed_01.html` — synthetic: one closed cell modified to also carry `hd_pressed`

## What states to capture

For good test coverage, please provide at minimum:
- [x] One board at game start (all cells closed) — any difficulty
- [x] One mid-game expert board (mix of closed, open numbers, flags)
- [x] One expert win (post-game stats visible: time, 3BV, 3BV/s, IOE, clicks)
- [x] One expert loss (mine revealed, post-game stats visible)
- [x] One beginner or intermediate board (to verify different grid sizes)
- [x] One over-flagged board (counter goes negative) — exposes `hd_top-area-num-` minus-sign class
- [x] One loss with an incorrectly-flagged cell (yields a `hd_opened hd_type11` cell)
- [x] One synthetic mid-game with a pressed cell (`hd_closed hd_pressed`)
- [x] A cell with a decorative gamification overlay class — `expert_loss_01.html`
      has `cell-ticket-flower` appended to a revealed cell (e.g. `cell size26
      hd_opened hd_type1 cell-ticket-flower`). The parser must ignore such extra
      classes and read state from the known tokens only (never `0xFF`).

High-number cells (6, 7, 8) are rare — include them if you happen to have a board with them.

## Synthetic fixtures

Some states are hard to capture from a live game (e.g., the transient `hd_pressed`
class only exists while a mouse button is held). For those, we hand-edit an existing
fixture and name it with the modification in the state slot. Synthetic files contain
a comment near the top documenting what was changed.

## What the parser will extract

From the cell grid:
- Board dimensions (rows × cols, from max `data-x`/`data-y` + 1)
- Mine count (from the three `#top_area_mines_*` slots)
- State of every cell: closed / open0–8 / flag / mine / wrong-flag

From the top bar:
- Face state (`hd_top-area-face-unpressed` / `-win` / `-lose`)
- Timer digits (`#top_area_time_*` carrying `hd_top-area-num{0-9}`)

From the post-game overlay (win/loss screens, both):
- Time (seconds, ms-precision)
- 3BV (denominator on loss, full value on win)
- 3BV/s, IOE, Ops, ThrP, Corr, ZNE, ZNT, RQP, IOS
- Left clicks, right clicks (right may be absent), CPS, Efficiency %
- Estimated time (loss-only)
