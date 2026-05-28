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
| state | `start` (no cells opened), `midgame`, `win`, `loss` |
| index | `01`, `02`, … (if multiple of same type) |

Examples:
- `expert_start_01.html`
- `expert_midgame_01.html`
- `expert_win_01.html`
- `expert_loss_01.html`
- `intermediate_midgame_01.html`
- `beginner_win_01.html`

## What states to capture

For good test coverage, please provide at minimum:
- [ ] One board at game start (all cells closed) — any difficulty
- [ ] One mid-game expert board (mix of closed, open numbers, flags)
- [ ] One expert win (post-game stats visible: time, 3BV, 3BV/s, IOE, clicks)
- [ ] One expert loss (mine revealed, post-game stats visible)
- [ ] One beginner or intermediate board (to verify different grid sizes)

High-number cells (6, 7, 8) are rare — include them if you happen to have a board with them.

## What the parser will extract

From the cell grid:
- Board dimensions (rows × cols)
- Mine count (from the counter display)
- State of every cell: closed / open0–8 / flag / mine / mine_hit

From the post-game overlay (win/loss screens):
- Time (seconds)
- 3BV
- 3BV/s
- IOE (efficiency)
- Left clicks, right clicks, total clicks
