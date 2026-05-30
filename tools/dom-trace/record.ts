/**
 * dom-trace recorder (simple HTML mode).
 *
 * Launches a throwaway *private* Edge window (fresh isolated profile, logged
 * out), opens minesweeper.online, and saves the FULL page HTML (head + body)
 * every time the board OR surrounding UI changes. The ticking timer / flag
 * counter (#top_area) are excluded from change-detection so they don't trigger
 * a frame every tick.
 *
 * You just play. Close the Edge window (or press Ctrl-C) to stop. Diffs between
 * frames are computed later from the saved snapshots.
 *
 * Output goes OUTSIDE the repo (under the OS temp dir) so it never bloats
 * git / Dropbox.
 *
 *   npm run record -- [--label expert] [--out DIR] [--interval 150]
 *                     [--edge PATH] [--profile DIR] [--url URL]
 */
import { chromium } from "playwright-core";
import { mkdirSync, writeFileSync, appendFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const EDGE = arg("edge", "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe");
const PROFILE = arg("profile", join(tmpdir(), "msm-rec-profile"));
const URL = arg("url", "https://minesweeper.online/");
const OUT_ROOT = arg("out", join(tmpdir(), "msm-traces"));
const LABEL = arg("label", "expert").replace(/[^a-z0-9_-]/gi, "_");
const INTERVAL = Math.max(50, Number(arg("interval", "150")));

const STAMP = new Date().toISOString().replace(/[:.]/g, "-");
const DIR = join(OUT_ROOT, `${LABEL}-${STAMP}`);

// Launch a fresh, isolated (private) Edge and open the game.
const ctx = await chromium.launchPersistentContext(PROFILE, {
  executablePath: EDGE,
  headless: false,
  viewport: null,
  args: ["--start-maximized"],
});
const page = ctx.pages()[0] ?? (await ctx.newPage());
await page.goto(URL, { waitUntil: "domcontentloaded" }).catch(() => {});

mkdirSync(DIR, { recursive: true });
const INDEX = join(DIR, "index.ndjson");
console.log(`[dom-trace] recording @ ${INTERVAL}ms -> ${DIR}`);
console.log(`[dom-trace] play your games; close the Edge window (or Ctrl-C) to stop.`);

/** Read the page (runs in the page). */
function grab(): { key: string; full: string } {
  // Save the WHOLE page (head + everything).
  const full = document.documentElement.outerHTML;
  // Change-detect on the whole page EXCEPT the top bar, so the per-second timer
  // and flag counter don't trigger a new frame, but board moves AND surrounding
  // UI (difficulty menu, no-guessing toggle, settings) all do.
  const top = document.getElementById("top_area");
  const key = top ? full.split(top.outerHTML).join("<TOP/>") : full;
  return { key, full };
}

let last = "";
let frames = 0;
let stopped = false;

async function tick(): Promise<void> {
  if (stopped) return;
  let snap: { key: string; full: string };
  try {
    snap = await page.evaluate(grab);
  } catch {
    return; // page navigating or closed
  }
  if (!snap.key || snap.key === last) return;
  last = snap.key;
  const hash = createHash("sha1").update(snap.full).digest("hex").slice(0, 8);
  const file = join(DIR, `frame-${String(frames).padStart(5, "0")}-${hash}.html`);
  writeFileSync(file, snap.full, "utf8");
  appendFileSync(INDEX, JSON.stringify({ n: frames, t: Date.now(), hash, file }) + "\n");
  frames++;
  if (frames % 25 === 0) console.log(`[dom-trace] ${frames} frames captured...`);
}

const timer = setInterval(() => void tick(), INTERVAL);

let closing = false;
async function shutdown(reason: string): Promise<void> {
  if (closing) return;
  closing = true;
  stopped = true;
  clearInterval(timer);
  await tick().catch(() => {}); // capture the final frame
  console.log(`\n[dom-trace] stopped (${reason}) - ${frames} frames in ${DIR}`);
  try {
    await ctx.close();
  } catch {
    /* already gone */
  }
  process.exit(0);
}

ctx.on("close", () => void shutdown("window closed"));
process.on("SIGINT", () => void shutdown("Ctrl-C"));
process.on("SIGTERM", () => void shutdown("terminated"));
