// Types that mirror the .msm format spec.
// Keep these aligned with docs/format-session.md and docs/format-events.md.

export const FORMAT_VERSION = '1.0.0';

export const HEADER_BYTES = 688;
export const FOOTER_BYTES = 65;

export const TAG = {
  CURSOR: 0x10,
  CURSOR_ANCHOR: 0x11,
  MOUSE_EVENT: 0x20,
  SCROLL_EVENT: 0x21,
  ZOOM_EVENT: 0x22,
  BOARD_CHANGE: 0x30,
  SESSION_EVENT: 0x40,
  RESIZE_EVENT: 0x41,
  FOOTER: 0xff,
} as const;

export const RECORD_BYTES = {
  [TAG.CURSOR]: 9,
  [TAG.CURSOR_ANCHOR]: 13,
  [TAG.MOUSE_EVENT]: 15,
  [TAG.SCROLL_EVENT]: 13,
  [TAG.ZOOM_EVENT]: 9,
  [TAG.BOARD_CHANGE]: 10,
  [TAG.SESSION_EVENT]: 9,
  [TAG.RESIZE_EVENT]: 11,
} as const;

export const MOUSE_ACTION = {
  LEFT_DOWN: 0x01,
  LEFT_UP: 0x02,
  RIGHT_DOWN: 0x03,
  RIGHT_UP: 0x04,
  MIDDLE_DOWN: 0x05,
  MIDDLE_UP: 0x06,
} as const;

export const SESSION_EVENT_TYPE = {
  GAME_START: 0x01,
  GAME_WIN: 0x02,
  GAME_LOSS: 0x03,
  TAB_BLUR: 0x04,
  TAB_FOCUS: 0x05,
  TIMESTAMP_OVERFLOW: 0x06,
  RECORDING_ERROR: 0x07,
} as const;

export const CELL_STATE = {
  OPEN_0: 0x00,
  OPEN_1: 0x01,
  OPEN_2: 0x02,
  OPEN_3: 0x03,
  OPEN_4: 0x04,
  OPEN_5: 0x05,
  OPEN_6: 0x06,
  OPEN_7: 0x07,
  OPEN_8: 0x08,
  CLOSED: 0x09,
  FLAGGED: 0x0a,
  MINE: 0x0b,
  // 0x0c reserved — was "mine_clicked", site does not visually distinguish
  WRONG_FLAG: 0x0d,
} as const;

export const FOOTER_RESULT = {
  WIN: 0x01,
  LOSS: 0x02,
} as const;

export interface MsmHeader {
  version: string;
  rows: number;
  cols: number;
  mines: number;
  epochStartMs: bigint;
  url: string;
  initPxW: number;
  initPxH: number;
  comment: string;
}

export type Record =
  | { kind: 'cursor'; x: number; y: number }
  | { kind: 'cursor_anchor'; t: number; x: number; y: number }
  | { kind: 'mouse_event'; t: number; action: number; x: number; y: number }
  | { kind: 'scroll_event'; t: number; dx: number; dy: number }
  | { kind: 'zoom_event'; t: number; scale: number }
  | { kind: 'board_change'; t: number; row: number; col: number; state: number }
  | { kind: 'session_event'; t: number; type: number }
  | { kind: 'resize_event'; t: number; boardPxW: number; boardPxH: number };

// Footer stats: every field nullable, encoded via bitmap.
// Bit positions match the table in format-session.md.
export interface MsmStats {
  durationMs?: number;     // bit 0
  timeS?: number;          // bit 1
  bbbv?: number;           // bit 2
  bbbvPerS?: number;       // bit 3
  clicksL?: number;        // bit 4
  clicksR?: number;        // bit 5
  cps?: number;            // bit 6
  efficiency?: number;     // bit 7
  ioe?: number;            // bit 8
  ops?: number;            // bit 9
  thrp?: number;           // bit 10
  corr?: number;           // bit 11
  zini?: number;           // bit 12
  zne?: number;            // bit 13
  znt?: number;            // bit 14
  rqp?: number;            // bit 15
  ios?: number;            // bit 16
  estimatedTime?: number;  // bit 17
}

export const STAT_BITS = {
  durationMs: 0,
  timeS: 1,
  bbbv: 2,
  bbbvPerS: 3,
  clicksL: 4,
  clicksR: 5,
  cps: 6,
  efficiency: 7,
  ioe: 8,
  ops: 9,
  thrp: 10,
  corr: 11,
  zini: 12,
  zne: 13,
  znt: 14,
  rqp: 15,
  ios: 16,
  estimatedTime: 17,
} as const;

export interface MsmFooter {
  result: number; // FOOTER_RESULT
  stats: MsmStats;
}

export interface MsmSession {
  header: MsmHeader;
  records: Record[];
  footer: MsmFooter;
}
