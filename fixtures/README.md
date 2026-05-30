# Fixtures

Test data captured from minesweeper.online, in two flavors:

- **`snapshots/`** — single-frame HTML files (one game state each), used by the
  board-parser unit tests (`extension/tests/board-parser.test.ts`). See
  [snapshots/README.md](snapshots/README.md) for the naming convention and the
  catalogue of captured states.
- **`recordings/`** — multi-frame **session** captures: a full game-by-game series
  of HTML frames recorded live with `tools/dom-trace`, used as a behavioral
  reference (how the board evolves, end-of-game reveals, etc.). Each recording is a
  compressed frame series plus a bookmarked README.

Snapshots are the curated, named states the parser is tested against. Recordings
are raw ground-truth we consult when unsure how the live site behaves — the
distilled findings live in [docs/dom-behavior.md](../docs/dom-behavior.md).
