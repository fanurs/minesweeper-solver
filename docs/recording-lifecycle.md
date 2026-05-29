# Minesweeper Mirror — Recording Lifecycle (v1.0.0)

This document defines when recording starts, what triggers state transitions,
and what happens in each interruption scenario. The content script must follow
these rules exactly.

---

## v1 Constraints

- The tab must remain open for the entire game.
- If the tab is closed before game end (win or loss), the session buffer is
  **discarded**. No partial file is written.
- Recording only starts from a **fresh game** — board all-closed, timer at
  zero, face neutral.
- If the extension loads mid-game (timer running, cells revealed), it waits
  silently for the next fresh game.
- No URL-based game identity or cross-session stitching. The game URL is
  stored in the header for reference only.

---

## State Machine

```
IDLE
  │
  │  Fresh game detected (all four conditions met — see below)
  ▼
RECORDING ──── Tab blur ──────────────────────────────────┐
  │  ▲                                                    │ TAB_BLURRED
  │  └── Tab focus (resume cursor + polling at 30 Hz)     │ (DOM polling at 10 Hz;
  │                                                       │  cursor sampling off)
  │  Win or loss detected                                 │
  ▼                                                       │
FINISHING ◄──────────────────── Tab focus + game end ─────┘
  │
  │  Footer written, download triggered
  ▼
IDLE
```

Any unrecoverable error at any state → emit `RECORDING_ERROR` → discard
buffer → return to `IDLE`.

---

## Fresh Game Detection

Polled at **10 Hz** while in `IDLE`.

A fresh game is detected when **all four** conditions are simultaneously true:

| Condition        | DOM indicator                                                          |
|------------------|------------------------------------------------------------------------|
| Face neutral     | `#top_area_face` has class `hd_top-area-face-unpressed`               |
| Timer at zero    | `#top_area_time_100`, `#top_area_time_10`, `#top_area_time_1` all carry `hd_top-area-num0` |
| All cells closed | Every cell has class `hd_closed`, none has `hd_opened`                |
| No ResultBlock   | `#ResultBlock` absent or not visible                                   |

**Digit parsing.** Each `#top_area_time_*` and `#top_area_mines_*` element
carries a class of the form `hd_top-area-num{D}` where `D` is `0`–`9`
(or `-` for the leading slot when over-flagging causes a negative mine
counter, e.g. `hd_top-area-num-`). The parser must scan `classList` with a
regex like `/^hd_top-area-num(\d|-)$/`, not assume class-list position.

The negative-counter case is irrelevant at fresh-game detection (no flags
exist yet, so the counter is positive). It matters only mid-game and is
not stored in any record — the live counter is derivable from
`total_mines - count(flagged cells)`.

When all four conditions are true, recording begins immediately:

1. Read `rows` and `cols` from max `data-y` and `data-x` attributes on cell
   elements (zero-indexed, so add 1). If either is 0, emit `RECORDING_ERROR`
   and abort (hard error).
2. Read `mines` from the three `#top_area_mines_*` digit slots. If 0, abort
   with `RECORDING_ERROR` (hard error).
3. Measure `#AreaBlock` bounding rect; record pixel width (`init_px_w`) and
   height (`init_px_h`); compute cell pixel width and height.
4. Read game URL from `window.location.href`.
5. Allocate session buffer; write 688-byte file header (including `init_px_w`
   and `init_px_h`).
6. Emit `SESSION_EVENT GAME_START` (`t = 0`).
7. Emit `CURSOR_ANCHOR` at `t = 0` with current cursor position.
8. Start DOM polling at 30 Hz and cursor sampling at 30 Hz.
9. Transition to `RECORDING`.

---

## Poll Rates by State

| State       | DOM poll rate | Cursor sample rate |
|-------------|---------------|--------------------|
| IDLE        | 10 Hz         | off                |
| RECORDING   | 30 Hz         | 30 Hz              |
| TAB_BLURRED | 10 Hz         | off                |
| FINISHING   | off           | off                |

---

## Cursor Sampling

During `RECORDING`, cursor position is sampled at ~30 Hz via a `setInterval`
reading `mousemove` last-known coordinates (not on every `mousemove` event,
which fires too frequently).

- Every sample emits a `CURSOR` record (9 bytes, no timestamp).
- Every 30th sample emits a `CURSOR_ANCHOR` instead (15 bytes, with uint32
  timestamp).
- After any interpolation boundary event (see below), the next sample is
  always a `CURSOR_ANCHOR` regardless of count.

---

## Tab Blur / Focus

**On tab blur** (`document.visibilitychange` → hidden, or `window blur`):

1. Emit `SESSION_EVENT TAB_BLUR`.
2. Stop cursor sampling interval.
3. Drop DOM poll rate to 10 Hz.

**On tab focus** (`visibilitychange` → visible, or `window focus`):

1. Emit `SESSION_EVENT TAB_FOCUS`.
2. Re-query `#AreaBlock` bounding rect; recompute cell size (page may have
   been resized while blurred). Update `prevScrollX/Y` to current values.
3. Emit `CURSOR_ANCHOR` with current position and current timestamp.
4. Resume cursor sampling at 30 Hz.
5. Resume DOM polling at 30 Hz.

The site game timer pauses when the tab is hidden. `duration_ms` in the
footer reflects the timer value and therefore excludes blur intervals.

---

## Viewport Changes

### Scroll (`window scroll` event)

1. Compute scroll deltas: `dx = (window.scrollX - prevScrollX) / cellPxW`,
   `dy = (window.scrollY - prevScrollY) / cellPxH`. Update `prevScrollX/Y`.
2. Emit `SCROLL_EVENT` (tag `0x21`) with dx, dy in cell units.
3. Re-query `#AreaBlock` bounding rect; recompute cell size.
4. Emit `CURSOR_ANCHOR` with updated position.

### Resize (`window resize` event)

1. Re-query `#AreaBlock` bounding rect; recompute cell size.
2. Emit `RESIZE_EVENT` (tag `0x41`) with new board pixel width and height.
3. Emit `CURSOR_ANCHOR` with updated position.

### Zoom (`window devicePixelRatio` change, detected via `matchMedia`)

1. Emit `ZOOM_EVENT` (tag `0x22`) with new scale factor.
2. Re-query `#AreaBlock` bounding rect; recompute cell size.
3. Emit `CURSOR_ANCHOR` with updated position.

In all three cases, the `CURSOR_ANCHOR` immediately follows the event record
with no intervening records.

---

## Game Reset Mid-Recording

If the player clicks the smiley face to reset while a game is in progress:

1. On the next DOM poll, all four fresh-game conditions become true.
2. The current in-progress buffer is **discarded** (treated as forfeit).
3. A new session begins immediately via the same path as idle → recording.

No special case is needed; this is the same code path as normal fresh-game
detection from `IDLE`.

---

## Game End Detection

Polled at the current poll rate (30 Hz in `RECORDING`, 10 Hz in
`TAB_BLURRED`).

| Face class                 | Result | Action                                  |
|----------------------------|--------|-----------------------------------------|
| `hd_top-area-face-win`     | Win    | Parse ResultBlock stats, write footer.  |
| `hd_top-area-face-lose`    | Loss   | Parse ResultBlock stats, write footer.  |

Both wins and losses get the same parsing path: the site renders a
(possibly partial) ResultBlock in both cases, so we attempt to read every
stat in both. On loss, the ZNE/ZNT/IOS fields are often shown as `–` and
their bits will end up unset — that is normal, not an error.

On detection:

1. Emit `SESSION_EVENT GAME_WIN` or `GAME_LOSS`.
2. **Stop DOM polling synchronously** in the same task that emitted the
   event. This prevents the post-game replay UI (which mounts a few frames
   later on wins) from being interpreted as ongoing board changes.
3. Stop cursor sampling. Transition to `FINISHING`.
4. Parse `#ResultBlock` for stat fields. For each field found and parsed,
   set the corresponding bit in `null_bitmap`. Fields not found, fields
   shown as `–` (en-dash), and fields that fail to parse leave their bit
   as `0` and their bytes as `0x00`. Individual parse failures are
   non-fatal — keep going.
5. Derive `duration_ms` from the `time_s` value (× 1000) if present; its
   bit is unset otherwise.
6. Write footer (65 bytes total including `0xFF` sentinel) to buffer.
7. Trigger browser download: filename
   `{epoch_start_ms}_{difficulty}_{result}.msm` into
   `minesweeper-mirror/` in the user's Downloads folder.
8. Free buffer; transition to `IDLE`.

---

## Event Ordering Under Concurrent Events

Browser events (tab blur, game-end detection, viewport changes) can arrive
while a mandatory `CURSOR_ANCHOR` sequence is in progress. Precedence rules:

1. **CURSOR_ANCHOR always completes first.** If any event arrives after a
   `RESIZE_EVENT`, `SCROLL_EVENT`, `ZOOM_EVENT`, or `TAB_FOCUS` but before
   its `CURSOR_ANCHOR` is written, write the `CURSOR_ANCHOR` first (using
   last known cursor position), then emit the competing event.
2. **GAME_WIN / GAME_LOSS** follow the same rule: complete any pending
   `CURSOR_ANCHOR`, then emit the game-end `SESSION_EVENT` and footer.
3. **The initialization sequence** (steps 1–9 in Fresh Game Detection) is
   treated as atomic: browser event listeners are not processed until step 9
   completes. This guarantees `GAME_START` is always the first record and
   `CURSOR_ANCHOR` at `t=0` is always the second record.

## Error Handling

See [format-session.md](format-session.md#error-handling) for the full
hard-error / soft-error split. Quick summary:

**Hard errors abort recording and discard the buffer:**

1. Emit `SESSION_EVENT RECORDING_ERROR` to the in-memory buffer (in-process
   debug hook only — never written to disk since the buffer is discarded
   immediately afterwards).
2. Stop all polling intervals and cursor sampling.
3. Discard the in-memory buffer. No file is written.
4. `console.warn('[minesweeper-mirror] recording error — session discarded')`.
5. Transition to `IDLE`.

Hard error triggers:

- `rows`, `cols`, or `mines` is 0 at game start (DOM structure mismatch).
- An in-game cell carries a class combination that maps to state `0xFF`
  (would-be unknown state — DOM contract has changed).
- Per-event timestamp overflow (≥ 49.7 days; precede with
  `SESSION_EVENT TIMESTAMP_OVERFLOW`).

**Soft errors log and continue:**

- ResultBlock missing or partial at game end → footer is written with
  whichever bits we could fill.
- Individual stat regex fails → that bit stays 0, neighbouring stats are
  still attempted.
- Mine-counter slot contains the `hd_top-area-num-` minus class while
  recording — log and skip; the value isn't stored anyway.

`RECORDING_ERROR` and `TIMESTAMP_OVERFLOW` records will never appear in a
valid completed `.msm` file (they always cause the buffer to be discarded).

---

## File Lifecycle Summary

| Phase                 | File state                                                |
|-----------------------|-----------------------------------------------------------|
| Fresh game detected   | Buffer allocated in memory; 688-byte header written.      |
| Recording             | Events appended to in-memory buffer.                      |
| Game ends cleanly     | Footer appended; download triggered; buffer freed.        |
| Tab closed mid-game   | Buffer discarded. Nothing written to disk.                |
| Browser crash         | Buffer lost (same as tab close mid-game).                 |
| Unrecoverable error   | Buffer discarded. Nothing written to disk.                |

A typical expert game produces 100–300 KB in memory. Well within browser
memory constraints.

---

## Explicit v1 Exclusions

The following are intentionally not recorded in v1:

- Games already in progress when the extension first loads.
- Mine positions after a loss (revealed in DOM at game end; not captured).
- Keyboard events (the site has keyboard shortcuts; ignored in v1).
- Cross-session resume after tab close mid-game.
- URL-based game identity or server-side game state lookup.
