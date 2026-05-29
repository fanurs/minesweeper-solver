// Shared types for the DOM board parser.
// Cell states mirror the .msm format (see docs/format-events.md cell-state codes).

export type CellState =
  | "closed" //      0x09  hd_closed (no flag); also hd_pressed (transient)
  | "open0" //       0x00  hd_opened hd_type0 (blank)
  | "open1"
  | "open2"
  | "open3"
  | "open4"
  | "open5"
  | "open6"
  | "open7"
  | "open8" //       0x01–0x08  hd_opened hd_type1–8
  | "flag" //        0x0A  hd_closed hd_flag (incl. win-time auto-flags)
  | "mine" //        0x0B  hd_opened hd_type10 (unflagged mine, revealed on loss)
  | "mine_hit" //    hd_opened hd_type12 (the detonated mine you clicked; → 0x0C, spec decision pending)
  | "wrong_flag" //  0x0D  hd_opened hd_type11 (flagged a non-mine, shown on loss)
  | "unknown"; //    0xFF  unrecognized — must never occur on a valid board

export interface Cell {
  row: number; // zero-indexed, from data-y (0 = top)
  col: number; // zero-indexed, from data-x (0 = left)
  state: CellState;
}

export interface BoardConfig {
  rows: number;
  cols: number;
  mines: number;
}

export type FaceState = "unpressed" | "win" | "lose" | "unknown";

export type GameResult = "win" | "loss" | "in_progress";

// Post-game stats parsed from #ResultBlock. Every field is optional — the site
// omits some on losses / very short games (rendered as an en-dash), matching the
// nullable footer in docs/format-session.md.
export interface GameStats {
  timeS?: number;
  bbbv?: number; //        3BV (total; on loss the DOM shows "solved / total")
  bbbvPerS?: number;
  clicksL?: number;
  clicksR?: number;
  cps?: number;
  efficiency?: number; //  percent
  ioe?: number;
  ops?: number;
  thrp?: number;
  corr?: number;
  zini?: number;
  zne?: number;
  znt?: number;
  rqp?: number;
  ios?: number;
  estimatedTime?: number; // loss-only, not always present
}

export interface BoardSnapshot {
  cells: Cell[];
  config: { rows: number; cols: number; mines: number | null };
  face: FaceState;
  result: GameResult;
  timerSeconds: number | null;
  stats?: GameStats; // present only when a non-empty #ResultBlock is parsed
}
