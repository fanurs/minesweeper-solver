// All shared types for the extension.
// Filled in once DOM class names are confirmed from fixture HTML files.

export type CellState =
  | "closed"       // unclicked
  | "open0"        // revealed, empty
  | "open1"        // revealed, number 1
  | "open2"
  | "open3"
  | "open4"
  | "open5"
  | "open6"
  | "open7"
  | "open8"
  | "flag"         // flagged
  | "mine"         // mine revealed (game over)
  | "mine_clicked" // the mine that was hit
  | "unknown";     // parse fallback

export interface Cell {
  row: number;
  col: number;
  state: CellState;
}

export interface BoardConfig {
  rows: number;
  cols: number;
  mines: number;
}

export interface GameStats {
  time: number;        // seconds
  bbbv: number;        // 3BV
  bbbvPerSec: number;  // 3BV/s
  ioe: number;         // IOE (clicks used / 3BV)
  clicks: {
    left: number;
    right: number;
    total: number;
  };
}

export type GameResult = "win" | "loss" | "in_progress";

export interface BoardSnapshot {
  timestamp: number;   // ms since game start
  cells: Cell[];
  result: GameResult;
}
