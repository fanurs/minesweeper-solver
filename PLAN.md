# Minesweeper Chrome Extension — Plan

## Goal

Build a Chrome extension that enhances minesweeper.online with:
- Custom visual aesthetics (CSS restyling)
- Sound effects on clicks, flags, wins, deaths
- Real-time board state recording (state diffs + timestamps + cursor events)
- A local Python companion that receives data and writes clean session logs to disk

No auto-play, no cheating. Pure read + restyle. Undetectable by the server.

---

## Architecture

```
Chrome Extension (content script, JS)
  ├── DOM Reader         — polls cell classes at 10 Hz idle / 30 Hz active
  ├── CSS Injector       — overrides site styles for custom aesthetics
  ├── Event Interceptor  — listens to clicks/flags, triggers sound effects
  └── WebSocket Client   — pushes board diffs to local Python server

Local Python Server
  ├── WebSocket server   — receives events from extension
  └── Session Logger     — writes compressed session logs to disk
```

The extension is self-contained — it works without the Python server (just no disk logging). The Python server is optional and adds persistence.

---

## Repository Layout

```
minesweeper-solver/
├── extension/                   ← Chrome extension
│   ├── manifest.json            ← MV3 manifest
│   ├── content.js               ← main content script (DOM reader, event hooks)
│   ├── background.js            ← service worker (WebSocket bridge if needed)
│   ├── sounds/                  ← audio files (.ogg or .mp3)
│   │   ├── click.ogg
│   │   ├── flag.ogg
│   │   ├── win.ogg
│   │   └── explode.ogg
│   └── styles/
│       └── board.css            ← custom board aesthetics
├── minesweeper_solver/          ← Python package
│   ├── __init__.py
│   ├── __main__.py              ← entry point: starts WebSocket server
│   ├── server.py                ← WebSocket server (receives extension events)
│   └── session.py               ← session log writer
├── sessions/                    ← output: one file per game session (gitignored)
├── assets/                      ← reference videos, frames
├── pyproject.toml
├── uv.lock
├── PLAN.md
└── README.md
```

---

## Phase 1 — DOM Reader (Foundation)

**Goal:** Reliably read board state from the live DOM.

The site uses class names on `<td>` or `<div>` elements to encode cell state. Based on the existing codebase's prior art, expected classes are something like:
- `hd_closed` — unclicked
- `hd_type1` … `hd_type8` — revealed number
- `hd_type0` — revealed empty
- `hd_flag` — flagged
- `hd_mine` / `hd_mine_clicked` — mine (game over)

**TODO:** Confirm exact class names from live DOM inspection (see note below).

Implementation:
- Content script polls `document.querySelectorAll('.cell')` (or equivalent) at configurable Hz
- Extracts a 2D array of cell states
- On each poll, diffs against previous state
- If diff is non-empty, emits an event `{timestamp, diffs: [{row, col, state}]}`
- Game lifecycle detection: watch for the smiley face class change (happy → dead → cool) to detect start/end

---

## Phase 2 — Event Interception + Sound

**Goal:** Play sounds on meaningful game events.

- Override `click` and `contextmenu` (right-click = flag) listeners on the board element
- Map action → sound:
  - Left click on unclicked cell → `click.ogg`
  - Right click → `flag.ogg` / `unflag.ogg`
  - Game won (smiley → cool) → `win.ogg`
  - Game lost (mine revealed) → `explode.ogg`
- Use the Web Audio API (`AudioContext`) for low-latency playback
- Sound files bundled in `extension/sounds/`

---

## Phase 3 — CSS Restyling

**Goal:** Custom aesthetics injected over the site's own styles.

- `board.css` injected via `content_scripts` in manifest
- Targets: cell colors by state, font, border, background, animations
- Number colors: the site uses inline color or class-based color — override with CSS variables so user can tweak one place
- Optional: CSS animations on reveal (fade-in), flag placement (bounce), explosion

Design decisions deferred to implementation — user defines the aesthetic.

---

## Phase 4 — Session Recording

**Goal:** Record every game as a compact, analyzable log.

**Log format** (newline-delimited JSON, one event per line, gzipped):
```json
{"t": 0.000, "type": "game_start", "rows": 16, "cols": 30, "mines": 99}
{"t": 1.234, "type": "cell_change", "diffs": [[3, 5, "1"], [3, 6, "0"]]}
{"t": 1.235, "type": "click", "x": 812, "y": 445, "button": "left"}
{"t": 9.801, "type": "game_end", "result": "win"}
```

- Cursor coordinates from `pynput` global listener in the Python server (OS-level, no CV)
- Board diffs from the extension via WebSocket
- Python server merges both streams by timestamp and writes `sessions/YYYY-MM-DD_HH-MM-SS.ndjson.gz`

---

## Phase 5 — Analysis (Future)

Not building yet. The session log format is designed to support:
- Replay visualization
- Per-cell click heatmaps
- Time-per-decision distribution
- Comparison across sessions

---

## Chrome Extension — manifest.json sketch

```json
{
  "manifest_version": 3,
  "name": "Minesweeper Mirror",
  "version": "0.1.0",
  "permissions": ["storage"],
  "host_permissions": ["https://minesweeper.online/*"],
  "content_scripts": [{
    "matches": ["https://minesweeper.online/*"],
    "js": ["content.js"],
    "css": ["styles/board.css"],
    "run_at": "document_idle"
  }],
  "background": {
    "service_worker": "background.js"
  }
}
```

---

## Python Server — entry point sketch

```
uv run minesweeper-solver
```

Starts a `websockets` server on `ws://localhost:8765`. The extension connects to it on load. If the server isn't running, the extension degrades gracefully (still does sounds + restyling, just no disk logging).

New Python dependency to add: `websockets`.

---

## Open Questions / Blockers

- [ ] **DOM class names** — need live HTML from minesweeper.online to confirm cell state classes. User to paste a snippet from DevTools, or we use Playwright to dump it.
- [ ] **Sound assets** — need actual .ogg/.mp3 files. Options: use royalty-free sfx, generate with Python (numpy sine waves), or user provides.
- [ ] **WebSocket from content script** — MV3 content scripts can open WebSockets directly. Need to verify this works for `ws://localhost` (not `wss://`) — may need a flag or the background service worker as a relay.

---

## What Is NOT Being Built

- No auto-solver / auto-clicker
- No screen capture / CV cell recognition (DOM is the source of truth)
- No mirror window on a second monitor
- No interaction with the server beyond normal gameplay

---

## Execution Order

1. Confirm DOM class names (need HTML snippet)
2. Scaffold `extension/` directory with manifest + stub content.js
3. Implement DOM reader + console logging (validate it reads board correctly)
4. Add WebSocket client → Python server → session log
5. Add sound effects
6. Add CSS restyling
7. Polish + README
