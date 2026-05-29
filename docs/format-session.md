# Minesweeper Mirror — Session File Format

Extension: `.msm` (Minesweeper Mirror)
Byte order: **little-endian** throughout

---

## Overview

A session file is a sequence of binary records written in chronological order.
Every record is prefixed with a 1-byte **tag** that identifies its type and
therefore its fixed byte length. Tags are not self-synchronizing — the format
is a sequential stream, not a searchable byte sequence. An unknown tag is
treated as a parse error; see Error Handling below.

```
[FILE_HEADER]          ← fixed 690 bytes
[RECORD] [RECORD] ...  ← sequential, tagged records
[FILE_FOOTER]          ← fixed 65 bytes, written only on clean game end
```

---

## File Header (fixed, 690 bytes)

| Offset | Size | Type   | Field         | Description                                              |
|--------|------|--------|---------------|----------------------------------------------------------|
| 0      | 32   | utf8   | `version`     | Format version string, null-padded. e.g. `"1.0.0"`.     |
| 32     | 2    | uint16 | `rows`        | Board row count (1–65535).                               |
| 34     | 2    | uint16 | `cols`        | Board column count (1–65535).                            |
| 36     | 2    | uint16 | `mines`       | Total mine count (0–65535).                              |
| 38     | 8    | uint64 | `epoch_start` | Unix epoch of game start, milliseconds UTC.              |
| 46     | 128  | utf8   | `url`         | Full game URL, null-padded. e.g. `"https://minesweeper.online/game/6103894316"`. |
| 174    | 2    | uint16 | `init_px_w`   | Board element pixel width at game start.                 |
| 176    | 2    | uint16 | `init_px_h`   | Board element pixel height at game start.                |
| 178    | 512  | utf8   | `comment`     | Free-form UTF-8 comment, null-padded. Default all zeros. |

**Total: 690 bytes.**

**Notes**

- All string fields are null-padded to their fixed size. If the value is
  shorter, remaining bytes are `0x00`. If longer, it is truncated at the
  field boundary. Readers should trim trailing null bytes when displaying.
- `version` currently `"1.0.0"`. Future breaking changes increment the major
  component. Readers encountering an unrecognised major version should refuse
  to parse and report the version string to the user.
- `epoch_start` is uint64 (8 bytes). 2^64 ms is effectively unbounded.
- `url` is 128 bytes including the scheme (`https://`). Sufficient for all
  current minesweeper.online URL formats.
- `init_px_w` / `init_px_h` are the `#AreaBlock` bounding rect pixel
  dimensions measured at game start. uint16 supports up to 65535 px, covering
  any realistic display. These are the reference pixel dimensions for the
  initial coordinate space; `RESIZE_EVENT` records encode subsequent changes.
- `comment` is 512 bytes. Intended for human notes, tool metadata, or
  session tags. Not parsed by the recorder; written as-is.
- `rows` and `cols` are uint16 (1–65535). Standard difficulties are tiny
  (expert is 30×16), but minesweeper.online custom boards can exceed 255 in
  either dimension, which a uint8 field would silently wrap — hence uint16.
- `rows`, `cols`, and `mines` must each be ≥ 1. A recorder encountering
  zero for any of these must emit `RECORDING_ERROR` and abort. A reader
  encountering zero must treat the file as corrupt and refuse to parse.

---

## Coordinate System

All cursor and mouse positions are stored in **board-relative cell units**
as two float32 values `(x, y)`:

- Origin `(0.0, 0.0)` is the top-left corner of the board element.
- `(1.0, 0.0)` is one cell-width to the right of the origin.
- `(0.0, 1.0)` is one cell-height below the origin.
- Values outside `[0, cols)` × `[0, rows)` indicate the cursor is outside
  the board (e.g. `(-1.2, 3.0)` is one cell-width to the left of the board).

**Cell size calculation:** measured once at game start by taking the full
board element's (`#AreaBlock`) bounding rect and dividing pixel width by
`cols` and pixel height by `rows` independently. This averages sub-pixel
rounding across the entire grid. Width and height are treated independently
even though cells are visually square — DOM measurements may differ
fractionally per axis.

**Recalculation triggers:** the board rect is re-queried and cell size
recalculated after every `RESIZE_EVENT`, `SCROLL_EVENT`, `ZOOM_EVENT`, and
`TAB_FOCUS`. A `CURSOR_ANCHOR` is emitted immediately after each
recalculation.

**HTML takes precedence for click attribution:** `BOARD_CHANGE` records
reflect actual DOM state changes, not inferred click targets. If coordinate
math suggests a click landed on cell A but the DOM reveals cell B changed,
`BOARD_CHANGE` records cell B. Analysts should use `BOARD_CHANGE` as the
ground truth for click effects.

---

## Record Structure

Each record begins with a 1-byte tag. The tag determines the record's total
byte length (tag byte included). Records are read sequentially; there is no
index or random-access mechanism.

| Tag  | Name            | Total bytes | Description                              |
|------|-----------------|-------------|------------------------------------------|
| 0x10 | `CURSOR`        | 9           | Cursor position sample (~30 Hz)          |
| 0x11 | `CURSOR_ANCHOR` | 13          | Cursor position + full timestamp         |
| 0x20 | `MOUSE_EVENT`   | 15          | Raw mouse button event                   |
| 0x21 | `SCROLL_EVENT`  | 13          | Mouse wheel scroll                       |
| 0x22 | `ZOOM_EVENT`    | 9           | Page zoom change                         |
| 0x30 | `BOARD_CHANGE`  | 10          | One cell changed state                   |
| 0x40 | `SESSION_EVENT` | 9           | Lifecycle event (start/end/blur/etc.)    |
| 0x41 | `RESIZE_EVENT`  | 11          | Board element pixel dimensions changed   |

Record layouts are defined in [format-events.md](format-events.md).

---

## Cursor Trajectory Encoding

Cursor samples are emitted at approximately 30 Hz during active recording.
To avoid storing a 4-byte timestamp per sample, a **delta + anchor** scheme
is used:

- Most samples are `CURSOR` records (tag + x + y = 9 bytes, no timestamp).
- Every 30th sample (~once per second) is a `CURSOR_ANCHOR` (tag + t + x + y
  = 13 bytes, with full uint32 timestamp).
- Timestamps for non-anchor samples are linearly interpolated between the
  surrounding anchors at read time.

**Interpolation boundaries:** never interpolate across a `TAB_BLUR` /
`TAB_FOCUS` pair or across any viewport-changing event (`SCROLL_EVENT`,
`ZOOM_EVENT`, `RESIZE_EVENT`). A `CURSOR_ANCHOR` is always emitted
immediately after each such boundary.

---

## Timestamps

Per-event timestamps are **uint32, milliseconds since game start** (`t=0` at
`GAME_START`). uint32 accommodates up to ~49.7 days, which exceeds any
realistic single-session game duration.

If a timestamp would overflow uint32 (game running longer than 49 days):
emit `SESSION_EVENT TIMESTAMP_OVERFLOW`, stop recording, discard the
in-memory buffer, and return to `IDLE`. No file is written.

`epoch_start` in the header uses uint64 (wall-clock Unix time, not
game-relative) and does not overflow in practice.

---

## File Footer (fixed, 65 bytes)

Written whenever the game ends and we observe a face transition to win or
loss, regardless of how much of the ResultBlock we can parse. If the tab is
closed before game end, no footer is written and the buffer is discarded.

All stat fields are **nullable**: a 4-byte `null_bitmap` immediately after
`result` encodes which fields are present. Bit `n` (0 = LSB) corresponds to
the stat at position `n` in the order below. Bit = `1` means the field is
present and valid; bit = `0` means the field is absent and its bytes are
all `0x00`. Readers must check the bitmap before interpreting any stat field.

Stats may be absent for any reason — very short games, custom boards,
ResultBlock structural changes, regex misses on a single field, etc.
**Individual stat parse failures are non-fatal**: the recorder leaves that
field's bit unset (0) and its bytes zero, then continues to the next field.
The footer is always written if we reached game end, even if every stat bit
is zero. This applies to both wins and losses — fixtures show that loss
games also contain a (partial) ResultBlock, so we attempt to parse stats in
both cases.

`null_bitmap` is uint32 to leave room for future stat additions without a
breaking format change.

| Offset | Size | Type    | Field            | Bit | Description                                     |
|--------|------|---------|------------------|-----|-------------------------------------------------|
| 0      | 1    | uint8   | `tag`            | —   | Always `0xFF` (footer sentinel).                |
| 1      | 1    | uint8   | `result`         | —   | `0x01` = win, `0x02` = loss.                    |
| 2      | 4    | uint32  | `null_bitmap`    | —   | Presence flags for stat fields (see above).     |
| 6      | 4    | uint32  | `duration_ms`    |  0  | Game timer, ms. Source: `time_s × 1000` when available, else absent. |
| 10     | 4    | float32 | `time_s`         |  1  | Time in seconds (e.g. `10.459`).                |
| 14     | 2    | uint16  | `bbbv`           |  2  | 3BV: minimum clicks to clear board optimally. On loss the DOM shows "X / Y" (completed / total) — store Y. |
| 16     | 4    | float32 | `bbbv_per_s`     |  3  | 3BV/s: 3BV ÷ time. Primary speed metric.        |
| 20     | 2    | uint16  | `clicks_l`       |  4  | Left clicks (reveals + chords).                 |
| 22     | 2    | uint16  | `clicks_r`       |  5  | Right clicks (flags). Defaults to 0 if the right-clicks `<span>` is absent (game ended before any flag). |
| 24     | 4    | float32 | `cps`            |  6  | Clicks per second: total clicks ÷ time.         |
| 28     | 1    | uint8   | `efficiency`     |  7  | Efficiency %: (3BV ÷ left clicks) × 100. Range 0–255 (can exceed 100). |
| 29     | 4    | float32 | `ioe`            |  8  | IOE: 3BV ÷ total clicks. Range 0–1.             |
| 33     | 2    | uint16  | `ops`            |  9  | Operations: distinct opening cascades.          |
| 35     | 4    | float32 | `thrp`           | 10  | Throughput: normalised speed metric.            |
| 39     | 4    | float32 | `corr`           | 11  | Correctness: ratio of decisive reveals.         |
| 43     | 2    | uint16  | `zini`           | 12  | ZiNi: count of non-trivial board cells.         |
| 45     | 4    | float32 | `zne`            | 13  | ZiNi Efficiency: IOE adjusted for ZiNi.         |
| 49     | 4    | float32 | `znt`            | 14  | ZiNi Normalised Throughput: 3BV/s ÷ ZiNi.       |
| 53     | 4    | float32 | `rqp`            | 15  | RQP: time² ÷ 3BV. Legacy metric, lower = better.|
| 57     | 4    | float32 | `ios`            | 16  | IOS: speed relative to think time.              |
| 61     | 4    | float32 | `estimated_time` | 17  | Estimated time. Loss-only stat; not shown on all losses (absent on e.g. beginner losses) — nullable (bit 17). |

**Total: 65 bytes** (including sentinel and bitmap).

**Notes**

- `duration_ms` is bit 0 of the bitmap (nullable like every other stat). If
  the recorder cannot derive it (no parseable `time_s` in the ResultBlock),
  the bit is unset and the bytes are zero. A reader that needs duration
  should fall back to `time_s` (also nullable, same source).
- `result` is always present. The site renders a (partial) ResultBlock on
  both wins and losses, so we attempt parsing in both cases. On loss the
  ZNE/ZNT/IOS fields are often rendered as `–` (en-dash) — those bits are
  unset. Fields like `estimated_time` are typically loss-only.
- Stats are stored exactly as reported by the site (no rounding, no
  recomputation). Field descriptions are best-effort; exact formulas are
  defined by minesweeper.online.
- `0xFF` cannot appear as a record tag (record tags are `0x10`–`0x41` in
  v1.0.0), so it unambiguously marks end-of-file when a reader encounters
  it in sequential parsing.
- **Reader: EOF without `0xFF`.** If a reader reaches EOF before
  encountering `0xFF`, the file is incomplete. Readers must treat such
  files as corrupt and refuse to parse them.

---

## File Naming

```
{epoch_start_ms}_{difficulty}_{result}.msm
```

Examples:
```
1748390400000_expert_win.msm
1748391234567_beginner_loss.msm
```

`difficulty`: `beginner` | `intermediate` | `expert` | `custom`
`result`: `win` | `loss`

On game end the completed buffer is persisted to the extension's IndexedDB
session store (see
[recording-lifecycle.md](recording-lifecycle.md#persistence--export)); it is
**not** auto-downloaded. The user exports stored sessions on demand (an explicit
Export action), at which point each is written as a download under a
`minesweeper-mirror/` subfolder of the default Downloads directory using this
name.

In-progress sessions live only in memory and are discarded if the tab closes
mid-game; completed sessions already in IndexedDB survive tab/window close.

---

## Error Handling

The recorder distinguishes **hard errors** (recording cannot continue) from
**soft errors** (recover and continue). The intent is to never crash the
recorder over a fixable parse hiccup, while still aborting cleanly when the
captured state would be meaningless.

### Hard errors (abort + discard buffer)

Trigger the following sequence:

1. Emit `SESSION_EVENT RECORDING_ERROR` with the current timestamp (to the
   in-memory buffer; for in-process debug hooks only — never persisted).
2. Stop all DOM polling and cursor sampling.
3. Discard the in-memory buffer.
4. Return to `IDLE` state.
5. Log a warning to the browser console. No user-visible alert.

Hard error conditions:

- `rows`, `cols`, or `mines` is `0` at recording start.
- A `BOARD_CHANGE` would have to write state `0xFF` (unknown DOM class
  combination on a cell that was previously valid). Indicates the site's
  DOM contract has changed. Note: the end-of-game classes `hd_type10`
  (→ `0x0B`) and `hd_type11` (→ `0x0D`) are **known**, not unknown — they
  signal game over and must never trigger this abort.
- Per-event timestamp would overflow uint32 (game running ≥ 49.7 days).
  Emit `SESSION_EVENT TIMESTAMP_OVERFLOW` then follow the hard-error path.

No partial file is ever written to disk after a hard error.

### Soft errors (log + continue)

Soft errors do **not** discard the buffer. The recorder logs a warning to
the console and proceeds. Examples:

- The `ResultBlock` is absent or partially absent at game end → footer is
  written with `null_bitmap = 0x00000000` and zero stat bytes.
- A single stat field (e.g. `bbbv_per_s`) fails to parse from text →
  that bit stays unset, that field's bytes stay zero, neighbouring stats
  are still attempted.
- `duration_ms` cannot be derived (no parseable `time_s`) → its bit
  stays unset; footer is still written.
- The mine-counter slot carries an unexpected class (e.g. `hd_top-area-num-`
  during over-flagging) → recorder logs and treats the slot as `0`.

The footer is **always** written if the game face transitions to win or
loss while we are recording. A footer with all stats absent is preferred to
discarding the session.

---

## Complete File Layout (Example — Beginner Win)

```
Bytes   0– 31:  "1.0.0\0\0..."            version (32 bytes, null-padded)
Bytes  32– 33:  0x09 0x00                  rows = 9 (uint16 LE)
Bytes  34– 35:  0x09 0x00                  cols = 9 (uint16 LE)
Bytes  36– 37:  0x0A 0x00                  mines = 10 (uint16 LE)
Bytes  38– 45:  ...                        epoch_start (uint64 LE)
Bytes  46–173:  "https://minesweeper..."   url (128 bytes, null-padded)
Bytes 174–175:  ...                        init_px_w (uint16 LE, e.g. 0xF0 0x00 = 240)
Bytes 176–177:  ...                        init_px_h (uint16 LE)
Bytes 178–689:  "\0\0..."                  comment (512 bytes, default empty)

Byte  690:      0x40                       SESSION_EVENT
Bytes 691–694:  0x00 0x00 0x00 0x00        t = 0
Byte  695:      0x01                       type = GAME_START
Bytes 696–698:  0x00 0x00 0x00             _pad

Byte  699:      0x11                       CURSOR_ANCHOR (t=0, initial position)
Bytes 700–703:  0x00 0x00 0x00 0x00        t = 0
Bytes 704–707:  ...                        x (float32)
Bytes 708–711:  ...                        y (float32)

... [interleaved CURSOR, MOUSE_EVENT, BOARD_CHANGE, SESSION_EVENT,
     SCROLL_EVENT, ZOOM_EVENT, RESIZE_EVENT records] ...

Byte    N:      0xFF                       footer sentinel
Byte  N+1:      0x01                       result = win
Bytes N+2– N+5: 0xFF 0xFF 0x01 0x00        null_bitmap = bits 0–16 set (win: estimated_time absent)
Bytes N+6– N+9: ...                        duration_ms (uint32, bit 0)
Bytes N+10–N+64: ...                       stat fields (time_s … estimated_time)
```

---

## Versioning

The `version` field is a null-padded UTF-8 string following semver
(`MAJOR.MINOR.PATCH`). Compatibility rules:

- **Same major:** backwards-compatible. New minor versions may add new record
  tags in the `0x42`–`0x4F` reserved range. Old readers encountering an
  unknown tag should treat it as a parse error in v1.0.0 — a future minor
  version may introduce a length-prefixed skip mechanism.
- **Different major:** breaking change. Readers must refuse to parse and
  report the version string.

Reserved tag ranges:

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
