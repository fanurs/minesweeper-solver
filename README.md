# Minesweeper Mirror

A Chrome extension for [minesweeper.online](https://minesweeper.online) that
adds custom aesthetics, sound effects, and **records each game** as a compact
binary session file for your own performance analysis.

It reads the page DOM only — **no automation, no solving, no cheating**. The
extension never interacts with the server beyond your normal play; it just
restyles the board, plays sounds on your clicks, and saves a clean log of how
each game unfolded.

> :construction: **Early development.** The project scaffold, the `.msm`
> session-file format spec, and DOM test fixtures are in place; the board
> parser, recorder, sounds, and restyling are not implemented yet.

## What it does

- **Custom aesthetics** — inject CSS to restyle the board (colors, fonts,
  animations) over the site's own styles.
- **Sound effects** — play audio on reveal, flag, win, and loss.
- **Session recording** — capture the full game as a compact binary `.msm`
  file: board state changes, cursor trajectory, mouse events, and end-of-game
  stats (Time, 3BV, 3BV/s, IOE, clicks, …). Files are downloaded locally on
  game end — no server, no account, nothing leaves your machine.

## Tech stack

- **Extension:** TypeScript, [Vite](https://vitejs.dev) (IIFE build),
  [Vitest](https://vitest.dev) (jsdom), Manifest V3.
- **Session format:** custom little-endian binary `.msm` — see
  [docs/format-session.md](docs/format-session.md),
  [docs/format-events.md](docs/format-events.md), and
  [docs/recording-lifecycle.md](docs/recording-lifecycle.md).

## Repository layout

```
extension/        Chrome extension (TypeScript → dist/)
  src/            content script, service worker, styles, shared types
  tests/          Vitest tests against fixture HTML
fixtures/         DOM snapshots from minesweeper.online for parser tests
docs/             .msm session-file format + recording lifecycle specs
PLAN.md           long-term roadmap
```

## Building (dev)

```sh
cd extension
npm install
npm run build      # outputs to extension/dist/
```

Then load it in Chrome via `chrome://extensions` → enable **Developer mode** →
**Load unpacked** → select the `extension/` directory. (A polished install
flow is planned; see [PLAN.md](PLAN.md).)

## License

See [LICENSE](LICENSE).
