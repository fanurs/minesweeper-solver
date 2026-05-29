import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { describe, it, expect } from "vitest";

import {
  boardDimensions,
  cellStateOf,
  isFreshGame,
  parseCells,
  parseFace,
  parseMines,
  parseSnapshot,
  parseTimerSeconds,
} from "../src/board-parser";

// Vitest runs with cwd = the extension/ package dir; fixtures are a sibling dir.
const FIXTURES_DIR = resolve(process.cwd(), "..", "fixtures");

/** Load a fixture HTML file and parse it into a Document. */
function loadDoc(name: string): Document {
  const html = readFileSync(resolve(FIXTURES_DIR, name), "utf8");
  return new DOMParser().parseFromString(html, "text/html");
}

const ALL_FIXTURES = [
  "beginner_fresh_01.html",
  "beginner_midgame_01.html",
  "beginner_win_01.html",
  "beginner_loss_01.html",
  "intermediate_fresh_01.html",
  "intermediate_midgame_01.html",
  "intermediate_midgame_pressed_01.html",
  "intermediate_overflagged_01.html",
  "intermediate_win_01.html",
  "intermediate_loss_01.html",
  "expert_fresh_01.html",
  "expert_midgame_01.html",
  "expert_overflagged_01.html",
  "expert_win_01.html",
  "expert_loss_01.html",
];

describe("dimensions", () => {
  it("beginner is 9x9", () => {
    expect(boardDimensions(parseCells(loadDoc("beginner_fresh_01.html")))).toEqual(
      { rows: 9, cols: 9 },
    );
  });
  it("intermediate is 16x16", () => {
    expect(
      boardDimensions(parseCells(loadDoc("intermediate_fresh_01.html"))),
    ).toEqual({ rows: 16, cols: 16 });
  });
  it("expert is 16 rows x 30 cols", () => {
    expect(boardDimensions(parseCells(loadDoc("expert_fresh_01.html")))).toEqual({
      rows: 16,
      cols: 30,
    });
  });
});

describe("cell counts and no unknown states", () => {
  const counts: Array<[string, number]> = [
    ["beginner_fresh_01.html", 81],
    ["intermediate_fresh_01.html", 256],
    ["expert_fresh_01.html", 480],
  ];
  for (const [name, n] of counts) {
    it(`${name} has ${n} cells`, () => {
      expect(parseCells(loadDoc(name)).length).toBe(n);
    });
  }
  it("no fixture produces an 'unknown' cell state", () => {
    for (const name of ALL_FIXTURES) {
      const unknown = parseCells(loadDoc(name)).filter(
        (c) => c.state === "unknown",
      );
      expect(unknown, name).toHaveLength(0);
    }
  });
});

describe("mine counter", () => {
  it("fresh boards show the full mine count", () => {
    expect(parseMines(loadDoc("beginner_fresh_01.html"))).toBe(10);
    expect(parseMines(loadDoc("intermediate_fresh_01.html"))).toBe(40);
    expect(parseMines(loadDoc("expert_fresh_01.html"))).toBe(99);
  });
  it("over-flagged boards show a negative counter", () => {
    expect(
      parseMines(loadDoc("intermediate_overflagged_01.html")),
    ).toBeLessThan(0);
    expect(parseMines(loadDoc("expert_overflagged_01.html"))).toBeLessThan(0);
  });
});

describe("timer and face", () => {
  it("fresh timer is 0 and face is unpressed", () => {
    const doc = loadDoc("beginner_fresh_01.html");
    expect(parseTimerSeconds(doc)).toBe(0);
    expect(parseFace(doc)).toBe("unpressed");
  });
  it("win face / loss face", () => {
    expect(parseFace(loadDoc("expert_win_01.html"))).toBe("win");
    expect(parseFace(loadDoc("expert_loss_01.html"))).toBe("lose");
  });
});

describe("cell states", () => {
  it("a fresh board is entirely closed", () => {
    const cells = parseCells(loadDoc("expert_fresh_01.html"));
    expect(cells.every((c) => c.state === "closed")).toBe(true);
  });
  it("a win auto-flags every remaining mine", () => {
    const cells = parseCells(loadDoc("beginner_win_01.html"));
    expect(cells.filter((c) => c.state === "flag").length).toBe(10);
  });
  it("a loss reveals mines and wrong flags", () => {
    const cells = parseCells(loadDoc("expert_loss_01.html"));
    expect(cells.filter((c) => c.state === "mine").length).toBeGreaterThan(0);
    expect(cells.filter((c) => c.state === "wrong_flag").length).toBe(1);
  });
  it("a loss marks the detonated mine (hd_type12) distinctly", () => {
    const inter = parseCells(loadDoc("intermediate_loss_01.html"));
    expect(inter.filter((c) => c.state === "mine_hit").length).toBe(1);
    expect(inter.filter((c) => c.state === "mine").length).toBeGreaterThan(0);
  });
  it("a pressed cell reads as closed (transient hd_pressed ignored)", () => {
    const doc = loadDoc("intermediate_midgame_pressed_01.html");
    const pressed = doc.querySelector<HTMLElement>(".hd_pressed");
    expect(pressed, "fixture should contain a pressed cell").toBeTruthy();
    expect(cellStateOf(pressed!)).toBe("closed");
  });
  it("a decorative class (cell-ticket-flower) does not break state", () => {
    const ticket = loadDoc("expert_loss_01.html").querySelector<HTMLElement>(
      ".cell-ticket-flower",
    );
    expect(ticket, "fixture should contain a ticket cell").toBeTruthy();
    expect(cellStateOf(ticket!)).toBe("open1");
  });
});

describe("fresh game detection", () => {
  it("is true for fresh boards", () => {
    expect(isFreshGame(loadDoc("beginner_fresh_01.html"))).toBe(true);
    expect(isFreshGame(loadDoc("expert_fresh_01.html"))).toBe(true);
  });
  it("is false for mid-game and finished boards", () => {
    expect(isFreshGame(loadDoc("expert_midgame_01.html"))).toBe(false);
    expect(isFreshGame(loadDoc("expert_win_01.html"))).toBe(false);
    expect(isFreshGame(loadDoc("expert_loss_01.html"))).toBe(false);
  });
});

describe("snapshot", () => {
  it("summarizes a fresh beginner board", () => {
    const s = parseSnapshot(loadDoc("beginner_fresh_01.html"));
    expect(s.config).toEqual({ rows: 9, cols: 9, mines: 10 });
    expect(s.face).toBe("unpressed");
    expect(s.result).toBe("in_progress");
    expect(s.timerSeconds).toBe(0);
  });
});
