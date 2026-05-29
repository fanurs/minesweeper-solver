# Minesweeper Mirror — Recording Lifecycle (v1.0.0)

This document defines when recording starts, what triggers state transitions,
and what happens in each interruption scenario. The content script must follow
these rules exactly.

---

## v1 Constraints

- The tab must remain open for the entire game being recorded.
- If the tab is closed before that game ends (win or loss), the **in-progress**
  session is **discarded** — no partial file. Completed games are already saved
  to IndexedDB (see [Persistence & Export](#persistence--export)) and survive.
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
  │  └── Tab focus (resume cursor + polling at 30 Hz)     │ (DOM polling ~1 Hz,
  │                                                       │  cursor sampling off)
  │  Win or loss detected                                 │
  ▼                                                       │
FINISHING ◄──────────────────── Game end (any focus) ─────┘
  │
  │  Footer written; saved to IndexedDB
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
5. Allocate session buffer; write 690-byte file header (including `init_px_w`
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
| TAB_BLURRED | ~1 Hz\*       | off                |
| FINISHING   | off           | off                |

\*Chrome throttles `setInterval` in hidden tabs to ~1 Hz (worse after a few
minutes) and pauses `requestAnimationFrame` entirely, so the blurred poll is
**best-effort ~1 Hz**, not a guaranteed 10 Hz. Drive all sampling with
`setInterval` (never `requestAnimationFrame`), and force a synchronous re-poll on
return to visible (see [Tab Blur / Focus](#tab-blur--focus)) to catch any
transition missed while throttled.

---

## Cursor Sampling

During `RECORDING`, cursor position is sampled at ~30 Hz via a `setInterval`
reading `mousemove` last-known coordinates (not on every `mousemove` event,
which fires too frequently).

- Every sample emits a `CURSOR` record (9 bytes, no timestamp).
- Every 30th sample emits a `CURSOR_ANCHOR` instead (13 bytes, with uint32
  timestamp).
- After any interpolation boundary event (see below), the next sample is
  always a `CURSOR_ANCHOR` regardless of count.
- `mousemove` does **not** fire while the pointer is outside the viewport, so the
  last-known position would otherwise freeze into a false flat trajectory. Listen
  for `mouseleave` / `pointerleave` on the document and emit a `CURSOR` sample
  with an explicit out-of-board sentinel position (a coordinate outside
  `[0, cols) × [0, rows)`) so "cursor left the window" is recorded rather than
  fabricated; resume normal sampling on `mouseenter`.

---

## Tab Blur / Focus

**On tab blur** (`document.visibilitychange` → hidden, or `window blur`):

1. Emit `SESSION_EVENT TAB_BLUR`.
2. Stop cursor sampling interval.
3. Drop DOM polling to a best-effort low rate. Chrome throttles hidden tabs to
   ~1 Hz (see [Poll Rates](#poll-rates-by-state)), so the nominal 10 Hz is not
   actually achievable while blurred — treat it as ~1 Hz.

**On tab focus** (`visibilitychange` → visible, or `window focus`):

1. Emit `SESSION_EVENT TAB_FOCUS`.
2. Re-query `#AreaBlock` bounding rect; recompute cell size (page may have
   been resized while blurred). Update `prevScrollX/Y` to current values.
3. **Synchronously re-read the full board and reconcile state**: emit
   `BOARD_CHANGE` records for any cells that changed while the tab was throttled,
   and check the face class — a win/loss may have completed while hidden, in
   which case run Game End Detection now.
4. Emit `CURSOR_ANCHOR` with current position and current timestamp.
5. Resume cursor sampling at 30 Hz.
6. Resume DOM polling at 30 Hz.

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

## Forfeit / Reset Mid-Recording

If the player abandons the current game before it ends (win/loss) — clicking the
smiley to reset, or starting a new board:

1. On the next DOM poll, all four fresh-game conditions become true.
2. The current in-progress buffer is **discarded**. A forfeit is neither a win
   nor a loss, so it is not recorded — v1 has no `GAME_ABANDONED` event.
3. A new session begins immediately via the same path as idle → recording.

No special case is needed; this is the same code path as normal fresh-game
detection from `IDLE`. Navigating away or closing the tab mid-game likewise
discards the in-progress buffer (completed sessions in IndexedDB are unaffected).

---

## Game End Detection

Polled at the current poll rate (30 Hz in `RECORDING`, best-effort ~1 Hz in
`TAB_BLURRED` — see [Poll Rates by State](#poll-rates-by-state)). This is why
both end paths below mandate a forced synchronous final poll on detection.

| Face class                 | Result | Action                                  |
|----------------------------|--------|-----------------------------------------|
| `hd_top-area-face-win`     | Win    | Parse ResultBlock stats, write footer.  |
| `hd_top-area-face-lose`    | Loss   | Parse ResultBlock stats, write footer.  |

Both wins and losses get the same parsing path: the site renders a
(possibly partial) ResultBlock in both cases, so we attempt to read every
stat in both. On loss, the ZNE/ZNT/IOS fields are often shown as `–` and
their bits will end up unset — that is normal, not an error.

On detection (the **face class is the authoritative game-end signal** — not the
mine counter or the replay UI). First complete any pending mandatory
`CURSOR_ANCHOR` (see Event Ordering), then stop cursor sampling. The win and loss
paths differ in one important way — whether the terminal board frame is captured:

**Loss path** (`hd_top-area-face-lose`):

1. **Do one forced, synchronous final poll** of the board (do not wait for the
   next scheduled tick — it may be throttled). The loss frame reveals every
   unflagged mine as `hd_type10` and marks wrong flags as `hd_type11`, all in one
   frame. Emit a `BOARD_CHANGE` for each changed cell — `0x0B` for revealed
   mines, `0x0D` for wrong flags — all sharing the final timestamp `t`. Mines the
   player had **correctly flagged** stay `0x0A` and need no record. `hd_type10` /
   `hd_type11` are known classes and must **not** trip the `0xFF` hard-abort.
2. Emit `SESSION_EVENT GAME_LOSS`, then **stop DOM polling synchronously** so no
   later frame (e.g. a post-game replay UI) is mis-read as a board change.

**Win path** (`hd_top-area-face-win`):

1. Emit `SESSION_EVENT GAME_WIN`, then **stop DOM polling synchronously in the
   same task** — before the next poll. On a win the site immediately auto-flags
   every remaining mine (`hd_closed` → `hd_closed hd_flag`, counter → 000) even
   though the player never placed those flags; stopping first keeps them out of
   the record. The win-time mine map is derivable anyway (every still-unopened
   cell is a mine). This also avoids capturing the post-game replay UI.

**Both paths then finish identically:**

3. Transition to `FINISHING`.
4. Parse `#ResultBlock` for stat fields. For each field found and parsed, set its
   bit in `null_bitmap`. Fields not found, shown as `–` (en-dash), or that fail
   to parse leave their bit `0` and bytes `0x00`. Individual parse failures are
   non-fatal — keep going. (On loss, `3BV` is rendered as "solved / total" —
   store the total; `estimated_time` is loss-only and may still be absent.)
5. Derive `duration_ms` from the `time_s` value (× 1000) if present; its bit
   is unset otherwise.
6. Write footer (65 bytes total including `0xFF` sentinel) to buffer.
7. **Persist** the completed session to the IndexedDB store under the name
   `{epoch_start_ms}_{difficulty}_{result}.msm` (see
   [Persistence & Export](#persistence--export)). Do **not** auto-download.
8. Free the in-memory buffer; transition to `IDLE`.

---

## Persistence & Export

Sessions are **not** auto-downloaded — one download per game would mean a
"Save As" prompt per game on browsers configured to ask, which is exactly the
per-play spam we avoid.

**Persistence (automatic, silent).** On clean game end the completed `.msm`
buffer is written as a `Blob` to an **IndexedDB** object store (`sessions`),
keyed by `epoch_start_ms` with metadata `{ difficulty, result, bytes,
exported: false }`. IndexedDB (not `chrome.storage`) is used because it stores
binary Blobs natively and holds many 100–300 KB sessions without quota friction.
Persisted sessions survive tab close, window close, browser restart, and
service-worker termination.

**Export (explicit, on demand).** The user exports via an explicit action (the
extension toolbar/popup **Export** button). Export drains the store: for each
not-yet-exported session, the content script sends the bytes to the service
worker, which calls `chrome.downloads.download` into a `minesweeper-mirror/`
subfolder. Batching means **at most one** "Save As" prompt for the whole drain
(and none if the user's "Ask where to save each file" Chrome setting is off).
Exported rows are marked `exported: true` (not deleted, so re-export is
possible); a retention cap evicts old exported rows.

> **Why not export on tab close?** `beforeunload` / `pagehide` cannot reliably
> run an asynchronous download (the page is being torn down), and the legacy
> `unload` event is being removed from Chrome. So close-time export is not
> achievable. Durability instead comes from the per-game IndexedDB persistence
> above — it has already happened before any close.

**Close warning (best-effort nudge).** If un-exported sessions exist when the
user tries to close the tab/window, a `beforeunload` handler shows the browser's
generic "Leave site?" confirmation as a reminder. It can only *warn* (its text is
not customizable and it cannot save) — the data is already safe in IndexedDB
regardless. Any in-progress (unfinished) game is discarded on close.

**Service-worker role.** Issuing the download is the **only** thing the service
worker does. All recording state, timers, and the buffer live in the content
script (the MV3 service worker is ephemeral — terminated after ~30 s idle); the
worker is a stateless `chrome.downloads` RPC. Requires the `downloads` permission.

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
  (would-be unknown state — DOM contract has changed). The end-of-game classes
  `hd_type10` / `hd_type11` are **known** (→ `0x0B` / `0x0D`), not `0xFF`, and
  must not trigger this.
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

| Phase                 | File state                                                       |
|-----------------------|------------------------------------------------------------------|
| Fresh game detected   | Buffer allocated in memory; 690-byte header written.             |
| Recording             | Events appended to in-memory buffer.                             |
| Game ends cleanly     | Footer appended; session saved to IndexedDB; in-memory buffer freed. |
| Export action         | Stored sessions downloaded to `minesweeper-mirror/`; rows marked exported. |
| Tab closed mid-game   | In-progress buffer discarded; completed sessions in IndexedDB survive. |
| Browser crash         | In-progress buffer lost; completed sessions in IndexedDB survive. |
| Unrecoverable error   | In-progress buffer discarded; completed sessions unaffected.     |

A typical expert game produces 100–300 KB in memory. Well within browser
memory constraints.

---

## Explicit v1 Exclusions

The following are intentionally not recorded in v1:

- Games already in progress when the extension first loads.
- Win-time auto-flagging of remaining mines (the site flags them on win; not
  recorded — the mine map is derivable as every still-unopened cell). The loss
  mine reveal, by contrast, **is** captured (states `0x0B`/`0x0D`).
- Keyboard events (the site has keyboard shortcuts; ignored in v1).
- Cross-session resume after tab close mid-game.
- URL-based game identity or server-side game state lookup.
