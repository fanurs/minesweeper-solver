# Minesweeper Mirror

A Chrome extension for [minesweeper.online](https://minesweeper.online) that
adds custom aesthetics, sound effects, and **records each game** as a compact
binary session file for your own performance analysis.

It reads the page DOM only — **no automation, no solving, no cheating**. The
extension never interacts with the server beyond your normal play; it just
restyles the board, plays sounds on your clicks, and saves a clean log of how
each game unfolded.

> :construction: **Early development.** The `.msm` format spec, DOM fixtures, and
> the **board parser** are in place, and the content script logs what it detects
> on the page. The session recorder, sound, and restyling are not implemented yet.

## What it does

- **Custom aesthetics** — inject CSS to restyle the board (colors, fonts,
  animations) over the site's own styles.
- **Sound effects** — synthesized audio on reveal, chord, flag, and loss, with an
  efficiency **combo** that rises in pitch as you keep making clean moves, plus a
  win flourish.
- **Session recording** — capture the full game as a compact binary `.msm`
  file: board state changes, cursor trajectory, mouse events, and end-of-game
  stats (Time, 3BV, 3BV/s, IOE, clicks, …). Completed games are saved in the
  extension and exported to disk on demand — no server, no account, nothing
  leaves your machine.

## Tech stack

- **Extension:** TypeScript, [Vite](https://vitejs.dev) (build; migrating to
  [WXT](https://wxt.dev)), [Vitest](https://vitest.dev) (jsdom), Manifest V3.
- **Session format:** custom little-endian binary `.msm` — see
  [docs/format-session.md](docs/format-session.md),
  [docs/format-events.md](docs/format-events.md), and
  [docs/recording-lifecycle.md](docs/recording-lifecycle.md).

## Repository layout

```
extension/        Chrome extension (TypeScript → dist/)
  src/            content script, service worker, styles, shared types
  tests/          Vitest tests against fixture HTML
fixtures/         DOM captures from minesweeper.online
  snapshots/      single-frame states for parser unit tests
  recordings/     multi-frame live session captures (behavioral reference)
docs/             .msm format, recording lifecycle, DOM behavior, sound specs
```

## Building (dev)

```sh
cd extension
npm install
npm run build      # outputs to extension/dist/
```

Then load it in Chrome via `chrome://extensions` → enable **Developer mode** →
**Load unpacked** → select the `extension/` directory. Open a game on
minesweeper.online and the DevTools console will log what the parser detects.
(A polished install flow is planned.)

## License

See [LICENSE](LICENSE).
