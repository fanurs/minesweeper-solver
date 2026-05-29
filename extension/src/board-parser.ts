// DOM board parser for minesweeper.online.
//
// Pure read-only functions over a DOM root (the live `document` in the
// extension, or a parsed fixture Document in tests). No side effects, no
// recording — see docs/format-events.md and docs/recording-lifecycle.md for the
// contract these mirror.
//
// Robustness rules (from the fixtures): cell class order is not stable and cells
// carry extra skin/decorative classes (e.g. `size26`, `cell-ticket-flower`,
// transient `hd_pressed`), so we match KNOWN tokens via classList / a `hd_type`
// regex and ignore everything else — an unrecognized extra class must never make
// a cell read as "unknown".

import type {
  BoardSnapshot,
  Cell,
  CellState,
  FaceState,
  GameResult,
  GameStats,
} from "./types";

const AREA_SELECTOR = "#AreaBlock";
const HD_TYPE_RE = /\bhd_type(\d+)\b/;
// Full class token: `hd_top-area-num0`..`9` or `hd_top-area-num-` (minus glyph).
// Anchored ^…$ per token because a trailing `\b` does not match after `-`.
const TOP_NUM_RE = /^hd_top-area-num(-|\d)$/;

/** Map one cell element's classes to a logical state. */
export function cellStateOf(el: Element): CellState {
  // A flag is `hd_closed hd_flag` (incl. win-time auto-flags) — check first.
  if (el.classList.contains("hd_flag")) return "flag";

  const m = HD_TYPE_RE.exec(el.className);
  if (m) {
    const n = Number(m[1]);
    if (!Number.isNaN(n)) {
      if (n >= 0 && n <= 8) return ("open" + n) as CellState;
      if (n === 10) return "mine";
      if (n === 11) return "wrong_flag";
      if (n === 12) return "mine_hit"; // detonated mine the player clicked
    }
    return "unknown";
  }

  // Closed (incl. transient `hd_closed hd_pressed`, treated as closed).
  if (el.classList.contains("hd_closed")) return "closed";
  return "unknown";
}

/**
 * All board cells, read from elements carrying data-x/data-y under #AreaBlock.
 * The `.clear` row separators have no data attributes and are skipped.
 */
export function parseCells(root: ParentNode): Cell[] {
  const area = root.querySelector(AREA_SELECTOR);
  if (!area) return [];
  const out: Cell[] = [];
  for (const el of Array.from(
    area.querySelectorAll<HTMLElement>("[data-x][data-y]"),
  )) {
    const col = Number(el.dataset.x);
    const row = Number(el.dataset.y);
    if (Number.isNaN(col) || Number.isNaN(row)) continue;
    out.push({ row, col, state: cellStateOf(el) });
  }
  return out;
}

/** Board dimensions = max index + 1. Returns 0×0 for an empty board. */
export function boardDimensions(cells: Cell[]): { rows: number; cols: number } {
  let rows = 0;
  let cols = 0;
  for (const c of cells) {
    if (c.row + 1 > rows) rows = c.row + 1;
    if (c.col + 1 > cols) cols = c.col + 1;
  }
  return { rows, cols };
}

/**
 * Read a 3-digit seven-segment counter (mines or timer) given the id prefix
 * (`top_area_mines_` or `top_area_time_`). Handles the leading minus glyph
 * (`hd_top-area-num-`) on a negative mine counter. Returns null if unreadable.
 */
function counterDigit(el: Element): string | undefined {
  for (const tok of el.className.split(/\s+/)) {
    const m = TOP_NUM_RE.exec(tok);
    if (m) return m[1];
  }
  return undefined;
}

function readCounter(root: ParentNode, prefix: string): number | null {
  let digits = "";
  let negative = false;
  for (const slot of ["100", "10", "1"]) {
    const el = root.querySelector("#" + prefix + slot);
    if (!el) return null;
    const ch = counterDigit(el);
    if (ch === undefined) return null;
    if (ch === "-") negative = true;
    else digits += ch;
  }
  if (digits === "") return null;
  const mag = Number(digits);
  if (Number.isNaN(mag)) return null;
  return negative ? -mag : mag;
}

export function parseMines(root: ParentNode): number | null {
  return readCounter(root, "top_area_mines_");
}

export function parseTimerSeconds(root: ParentNode): number | null {
  return readCounter(root, "top_area_time_");
}

export function parseFace(root: ParentNode): FaceState {
  const el = root.querySelector("#top_area_face");
  if (!el) return "unknown";
  if (el.classList.contains("hd_top-area-face-win")) return "win";
  if (el.classList.contains("hd_top-area-face-lose")) return "lose";
  if (el.classList.contains("hd_top-area-face-unpressed")) return "unpressed";
  return "unknown";
}

export function faceToResult(face: FaceState): GameResult {
  if (face === "win") return "win";
  if (face === "lose") return "loss";
  return "in_progress";
}

/**
 * Fresh game = all four conditions simultaneously
 * (see docs/recording-lifecycle.md "Fresh Game Detection").
 */
export function isFreshGame(root: ParentNode): boolean {
  if (parseFace(root) !== "unpressed") return false;
  if (parseTimerSeconds(root) !== 0) return false;
  const rb = root.querySelector("#ResultBlock");
  if (rb && (rb.textContent ?? "").trim() !== "") return false;
  const cells = parseCells(root);
  if (cells.length === 0) return false;
  return cells.every((c) => c.state === "closed");
}

/**
 * Post-game stats from #ResultBlock.
 *
 * TODO(next): full extraction — Time, 3BV ("solved / total" on loss), Clicks
 * (L [+R]), CPS, Efficiency %, IOE, Ops, ThrP, Corr, ZiNi, ZNE, ZNT, RQP, IOS,
 * Estimated time — handling the en-dash placeholder for absent stats and
 * ignoring the reward block after `<hr class="result-hr">`. Returns undefined
 * until implemented.
 */
export function parseStats(_root: ParentNode): GameStats | undefined {
  return undefined;
}

/** One combined read of the board + top bar (+ stats once implemented). */
export function parseSnapshot(root: ParentNode): BoardSnapshot {
  const cells = parseCells(root);
  const { rows, cols } = boardDimensions(cells);
  const face = parseFace(root);
  const snapshot: BoardSnapshot = {
    cells,
    config: { rows, cols, mines: parseMines(root) },
    face,
    result: faceToResult(face),
    timerSeconds: parseTimerSeconds(root),
  };
  const stats = parseStats(root);
  if (stats) snapshot.stats = stats;
  return snapshot;
}
