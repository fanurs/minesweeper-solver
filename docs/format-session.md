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
[FILE_HEADER]          ← fixed 688 bytes
[RECORD] [RECORD] ...  ← sequential, tagged records
[FILE_FOOTER]          ← fixed 59 bytes, written only on clean game end
```

---

## File Header (fixed, 684 bytes)

| Offset | Size | Type   | Field         | Description                                              |
|--------|------|--------|---------------|----------------------------------------------------------|
| 0      | 32   | utf8   | `version`     | Format version string, null-padded. e.g. `"1.0.0"`.     |
| 32     | 1    | uint8  | `rows`        | Board row count (1–255).                                 |
| 33     | 1    | uint8  | `cols`        | Board column count (1–255).                              |
| 34     | 2    | uint16 | `mines`       | Total mine count (0–65535).                              |
| 36     | 8    | uint64 | `epoch_start` | Unix epoch of game start, milliseconds UTC.              |
| 44     | 128  | utf8   | `url`         | Full game URL, null-padded. e.g. `"https://minesweeper.online/game/6103894316"`. |
| 172    | 2    | uint16 | `init_px_w`   | Board element pixel width at game start.                 |
| 174    | 2    | uint16 | `init_px_h`   | Board element pixel height at game start.                |
| 176    | 512  | utf8   | `comment`     | Free-form UTF-8 comment, null-padded. Default all zeros. |

**Total: 688 bytes.**

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
  = 15 bytes, with full uint32 timestamp).
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

## File Footer (fixed, 57 bytes)

Written only when the game ends cleanly (win or loss). If the tab is closed
before game end, no footer is written and the buffer is discarded.

All stat fields are **nullable**: a 2-byte `null_bitmap` immediately after
`result` encodes which fields are present. Bit `n` (0 = LSB) corresponds to
the stat at position `n` in the order below. Bit = `1` means the field is
present and valid; bit = `0` means the field is absent and its bytes are
all `0x00`. Readers must check the bitmap before interpreting any stat field.

Stats may be absent on very short games, custom boards, or if the
ResultBlock fails to parse. On loss, all stats are always absent
(`null_bitmap = 0x0000`): the site does not show ResultBlock stats for
losses and the recorder does not attempt to parse them.

| Offset | Size | Type    | Field         | Bit | Description                                     |
|--------|------|---------|---------------|-----|-------------------------------------------------|
| 0      | 1    | uint8   | `tag`         | —   | Always `0xFF` (footer sentinel).                |
| 1      | 1    | uint8   | `result`      | —   | `0x01` = win, `0x02` = loss.                    |
| 2      | 2    | uint16  | `null_bitmap` | —   | Presence flags for stats fields (see above).    |
| 4      | 4    | uint32  | `duration_ms` | —   | Game timer, ms. Always present if footer exists.|
| 8      | 4    | float32 | `time_s`      |  0  | Time in seconds (e.g. `10.459`).               |
| 12     | 2    | uint16  | `bbbv`        |  1  | 3BV: minimum clicks to clear board optimally.  |
| 14     | 4    | float32 | `bbbv_per_s`  |  2  | 3BV/s: 3BV ÷ time. Primary speed metric.       |
| 18     | 2    | uint16  | `clicks_l`    |  3  | Left clicks (reveals + chords).                |
| 20     | 2    | uint16  | `clicks_r`    |  4  | Right clicks (flags).                          |
| 22     | 4    | float32 | `cps`         |  5  | Clicks per second: total clicks ÷ time.        |
| 26     | 1    | uint8   | `efficiency`  |  6  | Efficiency %: (3BV ÷ left clicks) × 100.       |
| 27     | 4    | float32 | `ioe`         |  7  | IOE: 3BV ÷ total clicks. Range 0–1.            |
| 31     | 2    | uint16  | `ops`         |  8  | Operations: distinct opening cascades.         |
| 33     | 4    | float32 | `thrp`        |  9  | Throughput: normalised speed metric.           |
| 37     | 4    | float32 | `corr`        | 10  | Correctness: ratio of decisive reveals.        |
| 41     | 2    | uint16  | `zini`        | 11  | ZiNi: count of non-trivial board cells.        |
| 43     | 4    | float32 | `zne`         | 12  | ZiNi Efficiency: IOE adjusted for ZiNi.        |
| 47     | 4    | float32 | `znt`         | 13  | ZiNi Normalised Throughput: 3BV/s ÷ ZiNi.     |
| 51     | 4    | float32 | `rqp`         | 14  | RQP: time² ÷ 3BV. Legacy metric, lower=better.|
| 55     | 4    | float32 | `ios`         | 15  | IOS: speed relative to think time.            |

**Total: 59 bytes** (including sentinel and bitmap).

**Notes**

- `duration_ms` is always written (not nullable). It is read from the game
  timer DOM element (`#top_area_time_*`) at game end. If that element is
  missing or unparseable, the recorder must emit `RECORDING_ERROR` and
  discard the session — no partial footer is written.
- `result` is always present. On loss, `null_bitmap = 0x0000` and all stat
  bytes are `0x00`.
- `0xFF` cannot appear as a record tag (record tags are `0x10`–`0x41` in
  v1.0.0), so it unambiguously marks end-of-file when a reader encounters
  it in sequential parsing.
- **Reader: EOF without `0xFF`.** If a reader reaches EOF before
  encountering `0xFF`, the file is incomplete. Readers must treat such
  files as corrupt and refuse to parse them.
- Stats are stored as reported by the site. We do not recompute them.
  Field descriptions are best-effort; exact formulas are defined by
  minesweeper.online.

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

Files are held in memory during recording. On game end, the complete buffer
is offered as a browser download into a `minesweeper-mirror/` subfolder in
the user's default Downloads directory.

In-progress sessions have no on-disk representation. If the tab is closed
mid-game, the buffer is silently discarded.

---

## Error Handling

Any unexpected condition during recording — unknown DOM structure, failed
assumption, type error — triggers the following sequence:

1. Emit `SESSION_EVENT RECORDING_ERROR` with the current timestamp.
2. Stop all DOM polling and cursor sampling.
3. Discard the in-memory buffer.
4. Return to `IDLE` state.
5. Log a warning to the browser console. No user-visible alert.

No partial file is ever written to disk.

---

## Complete File Layout (Example — Beginner Win)

```
Bytes   0– 31:  "1.0.0\0\0..."            version (32 bytes, null-padded)
Byte   32:      0x09                       rows = 9
Byte   33:      0x09                       cols = 9
Bytes  34– 35:  0x0A 0x00                  mines = 10 (uint16 LE)
Bytes  36– 43:  ...                        epoch_start (uint64 LE)
Bytes  44–171:  "https://minesweeper..."   url (128 bytes, null-padded)
Bytes 172–173:  ...                        init_px_w (uint16 LE, e.g. 0xF0 0x00 = 240)
Bytes 174–175:  ...                        init_px_h (uint16 LE)
Bytes 176–687:  "\0\0..."                  comment (512 bytes, default empty)

Byte  688:      0x40                       SESSION_EVENT
Bytes 689–692:  0x00 0x00 0x00 0x00        t = 0
Byte  693:      0x01                       type = GAME_START
Bytes 694–696:  0x00 0x00 0x00             _pad

Byte  697:      0x11                       CURSOR_ANCHOR (t=0, initial position)
Bytes 698–701:  0x00 0x00 0x00 0x00        t = 0
Bytes 702–705:  ...                        x (float32)
Bytes 706–709:  ...                        y (float32)

... [interleaved CURSOR, MOUSE_EVENT, BOARD_CHANGE, SESSION_EVENT,
     SCROLL_EVENT, ZOOM_EVENT, RESIZE_EVENT records] ...

Byte    N:      0xFF                       footer sentinel
Byte  N+1:      0x01                       result = win
Bytes N+2– N+3: 0xFF 0xFF                  null_bitmap = all present
Bytes N+4– N+7: ...                        duration_ms (uint32)
Bytes N+8–N+58: ...                        stat fields (time_s … ios)
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
