// Minesweeper Mirror — content script (v0 "observer").
//
// Injected on minesweeper.online. For now it ONLY detects the board and
// console.logs what the parser sees, so we can iterate. No recording, no
// sounds, no restyling yet — those come next (see PLAN.md / docs/).

import { parseSnapshot } from "./board-parser";

const TAG = "[minesweeper-mirror]";
const POLL_MS = 500; // gentle observer cadence; the real recorder will poll faster

let lastSig = "";

/** Read the board once and log it — but only when something changed. */
function observe(): void {
  const board = document.querySelector("#AreaBlock");
  if (!board) {
    if (lastSig !== "no-board") {
      lastSig = "no-board";
      console.log(`${TAG} no board on this page`);
    }
    return;
  }

  const snap = parseSnapshot(document);
  const counts: Record<string, number> = {};
  for (const c of snap.cells) counts[c.state] = (counts[c.state] ?? 0) + 1;

  const sig = JSON.stringify([
    snap.config.rows,
    snap.config.cols,
    snap.config.mines,
    snap.timerSeconds,
    snap.face,
    snap.result,
    counts,
  ]);
  if (sig === lastSig) return;
  lastSig = sig;

  console.log(`${TAG} board`, {
    size: `${snap.config.rows}x${snap.config.cols}`,
    mines: snap.config.mines,
    timer: snap.timerSeconds,
    face: snap.face,
    result: snap.result,
    cells: counts,
  });
}

console.log(`${TAG} content script loaded — observing board (v0, log-only)`);
observe();
setInterval(observe, POLL_MS);
