# Minesweeper Mirror — Event Type Reference

All record layouts assume little-endian byte order.
Per-event timestamps (`t`) are uint32 milliseconds since `GAME_START` (`t=0`).
Tag byte is included in all byte-offset tables (offset 0 = tag).

Coordinates `x` and `y` are float32 in **board-relative cell units**:
origin `(0.0, 0.0)` is the top-left corner of `#AreaBlock`;
`(1.0, 0.0)` is one cell-width right; `(0.0, 1.0)` is one cell-height down.
Values outside `[0, cols)` × `[0, rows)` are valid and indicate the cursor
is outside the board. See [format-session.md](format-session.md) for the
full coordinate system definition.

---

## 0x10 — CURSOR (9 bytes)

Cursor position sample, emitted at ~30 Hz. No timestamp — interpolate from
surrounding `CURSOR_ANCHOR` records. Never emitted across a `TAB_BLUR` /
`TAB_FOCUS` boundary or any viewport-changing event (`SCROLL_EVENT`,
`ZOOM_EVENT`, `RESIZE_EVENT`).

| Offset | Size | Type    | Field | Description                          |
|--------|------|---------|-------|--------------------------------------|
| 0      | 1    | uint8   | `tag` | `0x10`                               |
| 1      | 4    | float32 | `x`   | Cursor x, board-relative cell units. |
| 5      | 4    | float32 | `y`   | Cursor y, board-relative cell units. |

---

## 0x11 — CURSOR_ANCHOR (13 bytes)

Cursor position sample with full timestamp. Emitted every 30th cursor sample
(~once per second at 30 Hz). Also emitted as the very first cursor record in
a session (immediately after `GAME_START`) and immediately after each
`TAB_FOCUS`, `SCROLL_EVENT`, `ZOOM_EVENT`, and `RESIZE_EVENT`.

| Offset | Size | Type    | Field | Description                          |
|--------|------|---------|-------|--------------------------------------|
| 0      | 1    | uint8   | `tag` | `0x11`                               |
| 1      | 4    | uint32  | `t`   | Milliseconds since game start.       |
| 5      | 4    | float32 | `x`   | Cursor x, board-relative cell units. |
| 9      | 4    | float32 | `y`   | Cursor y, board-relative cell units. |

---

## 0x20 — MOUSE_EVENT (15 bytes)

Raw mouse button event. Emitted for every button press or release anywhere
in the viewport (not just over the board). Cursor position at the moment of
the event is included so click location is unambiguous without interpolation.

| Offset | Size | Type    | Field    | Description                          |
|--------|------|---------|----------|--------------------------------------|
| 0      | 1    | uint8   | `tag`    | `0x20`                               |
| 1      | 4    | uint32  | `t`      | Milliseconds since game start.       |
| 5      | 1    | uint8   | `action` | See action codes below.              |
| 6      | 4    | float32 | `x`      | Cursor x, board-relative cell units. |
| 10     | 4    | float32 | `y`      | Cursor y, board-relative cell units. |

### Action Codes

| Code | Name          | Description                    |
|------|---------------|--------------------------------|
| 0x01 | `LEFT_DOWN`   | Left mouse button pressed.     |
| 0x02 | `LEFT_UP`     | Left mouse button released.    |
| 0x03 | `RIGHT_DOWN`  | Right mouse button pressed.    |
| 0x04 | `RIGHT_UP`    | Right mouse button released.   |
| 0x05 | `MIDDLE_DOWN` | Middle mouse button pressed.   |
| 0x06 | `MIDDLE_UP`   | Middle mouse button released.  |

**Derived actions — computed at analysis time, not stored**

| Raw event pattern                           | Derived interpretation      |
|---------------------------------------------|-----------------------------|
| `LEFT_DOWN` + `LEFT_UP`, same cell          | Simple left click           |
| `RIGHT_DOWN` + `RIGHT_UP`, same cell        | Flag / unflag               |
| `LEFT_DOWN` + `RIGHT_DOWN` (or vice versa)  | Chord attempt begin         |
| Both released while cursor on number cell   | Chord (1.5-click)           |
| `MIDDLE_DOWN` + `MIDDLE_UP`                 | Chord (alternative method)  |

Chording detection is left to analysis so the raw event sequence is preserved
without interpretation loss. Repeated flag/unflag on the same cell produces
alternating `BOARD_CHANGE` records (`0x0A` → `0x09` → `0x0A` …) and is
captured exactly as-is — this is valid player behaviour and not an error.

---

## 0x21 — SCROLL_EVENT (13 bytes)

Page scroll event. `dx` and `dy` are the change in `window.scrollX` and
`window.scrollY` respectively (in CSS pixels), divided by the current cell
pixel size to convert to **cell units**. Negative `dy` = scroll up; positive
`dy` = scroll down. A `CURSOR_ANCHOR` is emitted immediately after this
record (board rect is recalculated post-scroll before the anchor is written).

| Offset | Size | Type    | Field | Description                          |
|--------|------|---------|-------|--------------------------------------|
| 0      | 1    | uint8   | `tag` | `0x21`                               |
| 1      | 4    | uint32  | `t`   | Milliseconds since game start.       |
| 5      | 4    | float32 | `dx`  | Horizontal scroll delta, cell units. |
| 9      | 4    | float32 | `dy`  | Vertical scroll delta, cell units.   |

Cursor position at time of scroll is recoverable from the interpolated cursor
stream at the same timestamp `t`.

---

## 0x22 — ZOOM_EVENT (9 bytes)

Page zoom level changed (e.g. Ctrl+scroll or browser zoom controls). Triggers
a board rect recalculation. A `CURSOR_ANCHOR` is emitted immediately after
this record.

| Offset | Size | Type    | Field   | Description                                       |
|--------|------|---------|---------|---------------------------------------------------|
| 0      | 1    | uint8   | `tag`   | `0x22`                                            |
| 1      | 4    | uint32  | `t`     | Milliseconds since game start.                    |
| 5      | 4    | float32 | `scale` | New CSS zoom scale factor (1.0 = 100%, no zoom).  |

---

## 0x30 — BOARD_CHANGE (10 bytes)

Emitted whenever a cell transitions to a new state, as detected by DOM
polling. Multiple cells may change in the same poll (e.g. a chord or cascade
reveal); each cell gets its own `BOARD_CHANGE` record sharing the same
timestamp `t`.

| Offset | Size | Type   | Field   | Description                             |
|--------|------|--------|---------|-----------------------------------------|
| 0      | 1    | uint8  | `tag`   | `0x30`                                  |
| 1      | 4    | uint32 | `t`     | Milliseconds since game start.          |
| 5      | 1    | uint8  | `row`   | Zero-indexed row (0 = top row).         |
| 6      | 1    | uint8  | `col`   | Zero-indexed column (0 = left column).  |
| 7      | 1    | uint8  | `state` | New cell state. See state codes below.  |
| 8      | 2    | uint8[2] | `_pad` | Reserved, always `0x00 0x00`.          |

### Cell State Codes

| Code | DOM class(es)                          | Description                          |
|------|----------------------------------------|--------------------------------------|
| 0x00 | `hd_opened hd_type0`                   | Revealed, 0 adjacent mines (empty)   |
| 0x01 | `hd_opened hd_type1`                   | Revealed, 1 adjacent mine            |
| 0x02 | `hd_opened hd_type2`                   | Revealed, 2 adjacent mines           |
| 0x03 | `hd_opened hd_type3`                   | Revealed, 3 adjacent mines           |
| 0x04 | `hd_opened hd_type4`                   | Revealed, 4 adjacent mines           |
| 0x05 | `hd_opened hd_type5`                   | Revealed, 5 adjacent mines           |
| 0x06 | `hd_opened hd_type6`                   | Revealed, 6 adjacent mines           |
| 0x07 | `hd_opened hd_type7`                   | Revealed, 7 adjacent mines           |
| 0x08 | `hd_opened hd_type8`                   | Revealed, 8 adjacent mines           |
| 0x09 | `hd_closed` (no flag)                  | Closed, unflagged                    |
| 0x0A | `hd_closed hd_flag`                    | Flagged                              |
| 0x0B | `hd_opened hd_type10`                  | Mine revealed (game over, all mines) |
| 0x0C | —                                      | Reserved (was: "mine that was directly clicked"). The site does not visually distinguish the triggering mine, so this state is never emitted. The mine the player clicked is still derivable at analysis time by correlating the last `LEFT_UP` `MOUSE_EVENT` position with `BOARD_CHANGE` records at the same timestamp. |
| 0x0D | `hd_opened hd_type11`                  | Wrong flag (cell was flagged but contained no mine; revealed on loss) |
| 0xFF |                                        | Unknown / parse fallback             |

**Notes**

- State `0x09` (closed/unflagged) appears when a flag is removed. It does
  not indicate a cell being "un-revealed" (the site does not allow this).
- States `0x0B`–`0x0D` appear only on game loss.
- The initial board (all cells `0x09`) is implied by the header; no
  `BOARD_CHANGE` records are emitted for the starting state.
- `hd_pressed` is a **transient input-feedback class** added to a closed cell
  while the left mouse button is held over it. At most one cell is pressed at
  a time. Flagged cells cannot be pressed. Press-state is **not** encoded in
  `BOARD_CHANGE`: it is fully reconstructible at analysis time from
  `MOUSE_EVENT` + `CURSOR` records and cell geometry. The parser must
  recognise `hd_pressed` so it can ignore it when computing the cell's
  persistent state (treat `hd_closed hd_pressed` as `0x09`).
- Cell elements carry additional skin classes (e.g. `size26`). The parser
  must use `classList.contains()` / regex matching on the full class list,
  not equality against a fixed string.
- `0xFF` must never be written by the recorder. If an unknown DOM class
  combination is encountered, emit `SESSION_EVENT RECORDING_ERROR` and abort.
- **Reader behavior for state `0xFF`:** treat the containing `BOARD_CHANGE`
  record as corrupt, emit a warning, and refuse to parse the file.

---

## 0x40 — SESSION_EVENT (9 bytes)

Sparse lifecycle events marking recording state transitions and cursor
interpolation boundaries.

| Offset | Size | Type   | Field  | Description                       |
|--------|------|--------|--------|-----------------------------------|
| 0      | 1    | uint8  | `tag`  | `0x40`                            |
| 1      | 4    | uint32 | `t`    | Milliseconds since game start.    |
| 5      | 1    | uint8  | `type` | See type codes below.             |
| 6      | 3    | uint8[3] | `_pad` | Reserved, always `0x00 0x00 0x00`.|

### Session Event Type Codes

| Code | Name                 | Description                                                         |
|------|----------------------|---------------------------------------------------------------------|
| 0x01 | `GAME_START`         | Recording begins. Always the first record. `t = 0`.                |
| 0x02 | `GAME_WIN`           | Win detected (face class `hd_top-area-face-win`). Footer written immediately after. |
| 0x03 | `GAME_LOSS`          | Loss detected (face class `hd_top-area-face-lose`). Footer written immediately after. |
| 0x04 | `TAB_BLUR`           | Tab lost focus. Cursor sampling stops. DOM polling drops to 10 Hz. |
| 0x05 | `TAB_FOCUS`          | Tab regained focus. `CURSOR_ANCHOR` emitted immediately after.      |
| 0x06 | `TIMESTAMP_OVERFLOW` | uint32 `t` would overflow. Recording stops, buffer discarded.       |
| 0x07 | `RECORDING_ERROR`    | Unrecoverable error. Recording stops, buffer discarded.             |

**Notes**

- `GAME_START` is always `t = 0` and always the first record in the file.
- `CURSOR_ANCHOR` at `t = 0` is always the second record in the file.
- `TAB_BLUR` / `TAB_FOCUS` delimit intervals where cursor tracking is
  suspended. Do not interpolate cursor positions across these pairs.
- `TIMESTAMP_OVERFLOW` and `RECORDING_ERROR` mark abnormal termination
  (hard errors — see [format-session.md](format-session.md#error-handling)).
  A valid completed file never contains these — it always ends with
  `GAME_WIN` or `GAME_LOSS` followed immediately by the footer (`0xFF`).
  These records are emitted to the in-memory buffer immediately before the
  buffer is discarded; they exist to support in-process debug hooks, not
  for disk persistence.
- Soft errors (e.g. an individual ResultBlock stat fails to parse) do **not**
  emit `RECORDING_ERROR`. The recorder logs to the console and continues;
  the missing stat's bit in the footer's `null_bitmap` stays unset.
- **TAB_BLUR race with mandatory CURSOR_ANCHOR:** if a `TAB_BLUR` event
  fires after a `RESIZE_EVENT`, `SCROLL_EVENT`, `ZOOM_EVENT`, or `TAB_FOCUS`
  but before its mandatory `CURSOR_ANCHOR` can be written, the recorder must
  complete the `CURSOR_ANCHOR` first (using the last known cursor position),
  then emit `TAB_BLUR`. Rule 5 (CURSOR_ANCHOR immediately follows its
  trigger) takes precedence over TAB_BLUR ordering.
- **GAME_WIN/GAME_LOSS race:** if game end is detected while a viewport-event
  + `CURSOR_ANCHOR` sequence is in progress, the `CURSOR_ANCHOR` is completed
  first, then `GAME_WIN`/`GAME_LOSS` is emitted. No other record may appear
  between a viewport-event and its `CURSOR_ANCHOR`.
- There is no `VIEWPORT_EVENT` session event type. Viewport changes are
  represented by their dedicated record types (`SCROLL_EVENT` `0x21`,
  `ZOOM_EVENT` `0x22`, `RESIZE_EVENT` `0x41`), each of which implies a
  board rect recalculation and is always followed by a `CURSOR_ANCHOR`.

---

## 0x41 — RESIZE_EVENT (11 bytes)

Emitted when the board element's pixel dimensions change due to a window
resize. This is event-driven (fires on `window resize`) — not sampled at a
fixed rate. Triggers a board rect recalculation. A `CURSOR_ANCHOR` is
emitted immediately after.

Storing pixel dimensions preserves the physical scale of cell units: analysts
can recover how many pixels the cursor travelled by multiplying cell-unit
distances by the cell pixel size at that point in the session.

| Offset | Size | Type   | Field          | Description                                    |
|--------|------|--------|----------------|------------------------------------------------|
| 0      | 1    | uint8  | `tag`          | `0x41`                                         |
| 1      | 4    | uint32 | `t`            | Milliseconds since game start.                 |
| 5      | 2    | uint16 | `board_px_w`   | Board element pixel width after resize.        |
| 7      | 2    | uint16 | `board_px_h`   | Board element pixel height after resize.       |
| 9      | 2    | uint8[2] | `_pad`       | Reserved, always `0x00 0x00`.                  |

**Notes**

- uint16 for pixel dimensions supports up to 65535 px per axis, which covers
  any realistic display (4K is 3840 px wide).
- The initial board pixel dimensions at game start are stored in the file
  header fields `init_px_w` and `init_px_h`. A `RESIZE_EVENT` is only
  emitted when dimensions *change* after game start. Readers can reconstruct
  the board pixel size at any point in the session by applying `RESIZE_EVENT`
  records sequentially from the header's initial values.
- `ZOOM_EVENT` also changes effective board pixel size from the browser's
  perspective, but the board element's `getBoundingClientRect()` already
  reflects zoom. A separate `RESIZE_EVENT` is not emitted for zoom — the
  `ZOOM_EVENT` record itself (with `scale`) is sufficient.

---

## Record Ordering Guarantees

1. `SESSION_EVENT GAME_START` is always the first record (offset 688 in file).
2. `CURSOR_ANCHOR` at `t = 0` is always the second record.
3. All subsequent records are in non-decreasing `t` order.
4. Within the same timestamp, `BOARD_CHANGE` records appear after any
   `MOUSE_EVENT` records at the same `t`.
5. A `CURSOR_ANCHOR` immediately follows every `TAB_FOCUS`, `SCROLL_EVENT`,
   `ZOOM_EVENT`, and `RESIZE_EVENT` — no other record may appear between
   the triggering record and its `CURSOR_ANCHOR`.
6. `SESSION_EVENT GAME_WIN` or `GAME_LOSS` is always the last record before
   the footer sentinel (`0xFF`).

---

## Record Size Summary

| Tag  | Name            | Bytes |
|------|-----------------|-------|
| 0x10 | CURSOR          | 9     |
| 0x11 | CURSOR_ANCHOR   | 13    |
| 0x20 | MOUSE_EVENT     | 15    |
| 0x21 | SCROLL_EVENT    | 13    |
| 0x22 | ZOOM_EVENT      | 9     |
| 0x30 | BOARD_CHANGE    | 10    |
| 0x40 | SESSION_EVENT   | 9     |
| 0x41 | RESIZE_EVENT    | 11    |
| 0xFF | Footer sentinel | 65    |

---

## Reserved Tags

| Range      | Status                              |
|------------|-------------------------------------|
| 0x00–0x0F  | Reserved (never used as record tag) |
| 0x10–0x11  | Cursor records (v1.0.0)             |
| 0x20–0x22  | Mouse / scroll / zoom (v1.0.0)      |
| 0x30       | Board change (v1.0.0)               |
| 0x40–0x41  | Session event / resize (v1.0.0)     |
| 0x42–0x4F  | Reserved for future minor versions  |
| 0x50–0xFE  | Reserved for future major versions  |
| 0xFF       | Footer sentinel (not a record tag)  |
