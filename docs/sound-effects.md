# Minesweeper Mirror — Sound Effects (v1)

Optional, client-side **sound effects** played live by the content script while
a game is recorded. Sounds are a presentation layer on top of the recorder
model defined in [format-events.md](format-events.md) and
[recording-lifecycle.md](recording-lifecycle.md): every sound is derived from
the same `MOUSE_EVENT` (`0x20`), `BOARD_CHANGE` (`0x30`), and `SESSION_EVENT`
(`0x40`) signals the recorder already produces. Sound playback **never** alters,
gates, or is written to the `.msm` stream — it is a pure consumer of recorder
state. Audio failures (suspended context, decode error) are logged and
swallowed; they never emit `RECORDING_ERROR`.

All terminology (`LEFT_DOWN`, `BOARD_CHANGE`, state codes `0x00`–`0x0A`,
`GAME_LOSS`, `hd_pressed`) is used exactly as in those documents.

---

## Scope

Sound effects fire only while the recorder is in the `RECORDING` state. No sound
is produced in `IDLE`, `TAB_BLURRED`, or `FINISHING`, with the single exception
that the loss sting is allowed to play during the `RECORDING → FINISHING`
transition that `GAME_LOSS` triggers.

---

## v1 Event → Sound Mapping

Five events produce sound in v1. Each is detected from the recorder model, not
from ad-hoc DOM listeners, so the audio layer stays consistent with what the
`.msm` file records.

| Event             | Trigger condition                                                                                                                                  | Sound      |
|-------------------|----------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| **Left press**    | `MOUSE_EVENT` `LEFT_DOWN` (`0x01`) over a **closed, unflagged** cell (state `0x09`) — i.e. the cell that would receive the transient `hd_pressed` feedback class. | `press`    |
| **Cell reveal**   | A poll yielding reveals to opened states (`0x00`–`0x08`) where the press was over a **closed** cell — and it is not a chord (below).                | `reveal`   |
| **Chording**      | A left press/release over an **already-opened number cell** (`0x01`–`0x08`) that yields **two or more** `BOARD_CHANGE` reveals sharing one timestamp `t`. | `chord`    |
| **Hit a mine**    | `SESSION_EVENT` `GAME_LOSS` (`0x03`).                                                                                                               | `mine`     |
| **Flag / unflag** | `BOARD_CHANGE` to flagged (`0x0A`) → `flag`; `BOARD_CHANGE` from `0x0A` back to closed (`0x09`) → `unflag`.                                          | `flag` / `unflag` |

### Detection notes

The detector runs against the same per-poll batch of `BOARD_CHANGE` records the
recorder emits, plus the `MOUSE_EVENT` stream.

- **Left press (`press`).** Bound to `LEFT_DOWN`, not the reveal, so the click
  feels responsive before the DOM poll resolves. It is the audible analog of the
  `hd_pressed` input-feedback class (which is itself never a `BOARD_CHANGE`).
  Flagged (`0x0A`) and already-opened cells cannot be pressed → no `press` sound.

- **Reveal vs. chord disambiguation.** Both reveal opened states (`0x00`–`0x08`);
  the difference is the cell under the cursor at press time and cardinality.
  Classify **once per timestamp `t`** (not once per `BOARD_CHANGE`), so a cascade
  is a single decision:
  - **Chord** = the press was over a cell already in an opened *number* state
    (`0x01`–`0x08`) **and** the batch at `t` contains **≥ 2** reveals. This is the
    "revealed, *correct* number cell" rule: **correctness is inferred from the
    effect** (the chord actually revealed cells), because flag-vs-mine
    correctness is not knowable synchronously at press time. A chord on a number
    cell whose flags are wrong/insufficient reveals nothing → plays nothing.
  - **Reveal** = the press was over a *closed* cell (`0x09`). A cascade from one
    closed `0x00` cell may open many cells in one poll but is still **one**
    `reveal` (cardinality alone does not make it a chord).

- **Mine (`mine`).** Driven solely by `GAME_LOSS`, not by the `0x0B`/`0x0D`
  board states (which appear *as a consequence* of loss and would multi-trigger).
  The site does not visually distinguish the triggering mine, so v1 plays one
  loss sting.

- **Flag / unflag.** Detected purely from `BOARD_CHANGE` transitions involving
  `0x0A`, independent of which input caused them. Rapid alternating flag/unflag
  alternates the sounds — valid behaviour, subject to debouncing.

- **Loss-time board noise must be suppressed.** On loss the recorder performs one
  final poll that captures the mine reveal (`0x0B`/`0x0D`) and then stops (see
  [recording-lifecycle.md](recording-lifecycle.md)). The audio layer must
  **suppress** `reveal`/`chord` sounds for that terminal batch and play only the
  `mine` sting — once the lose-face / `GAME_LOSS` is observed, discard remaining
  reveals so the mine cascade does not fire a flurry of `reveal` blips.

---

## Audio Approach

### Option A — Bundled CC0 assets

Ship pre-recorded one-shots in the package (`assets/sound/*.{ogg,webm}`), decoded
once into `AudioBuffer`s. Concrete **CC0 / public-domain** sources (no
attribution, safe to redistribute):

- [Kenney — Interface Sounds](https://kenney.nl/assets/interface-sounds) and
  [Kenney — UI Audio](https://kenney.nl/assets/ui-audio) (CC0) — clean click/tick
  one-shots for `press`, `flag`, `unflag`.
- [OpenGameArt — CC0 Sound Effects](https://opengameart.org/content/cc0-sound-effects)
  — impacts for `chord` / `mine`.
- [Freesound](https://www.freesound.org/) (filter to CC0 only) for explosion/
  reveal textures.

Pros: rich, recognizable timbres. Cons: binary weight, curation/normalization,
per-asset licence bookkeeping.

### Option B — Runtime synthesis (Web Audio API)

Generate every sound from `OscillatorNode`s and short noise `AudioBuffer`s shaped
by gain envelopes — no asset files:

- **`press`** — 6–10 ms triangle blip ~1000 Hz, fast decay.
- **`reveal`** — sine/triangle ping ~660 Hz, ~40 ms.
- **`chord`** — two-note ping (~660 Hz → ~880 Hz) so it's distinct from a reveal.
- **`flag` / `unflag`** — higher tick (~1200 Hz) vs lower tick (~800 Hz) so the
  toggle direction is audible.
- **`mine`** — white-noise burst through a down-swept low-pass, ~250–400 ms decay.

Always ramp gain with `linearRampToValueAtTime`/`setTargetAtTime` rather than
starting/stopping at non-zero gain, to avoid click artifacts.

Pros: **zero asset friction**, tiny bundle, lowest latency. Cons: utilitarian
timbres; envelopes need tuning.

### Recommendation — synthesis-first, asset-optional

Default v1 to **Option B**. Architect the player around a small **sound registry**
keyed by logical name (`press`, `reveal`, `chord`, `flag`, `unflag`, `mine`) so a
future minor version can swap a synth voice for a decoded CC0 `AudioBuffer`
per-event without touching the detector.

---

## Web Audio Implementation

### AudioContext lifecycle & autoplay gesture

- Create **one** `AudioContext` for the content script. A `master` `GainNode`
  feeds `ctx.destination`; per-event `GainNode`s feed `master`.
- Browsers create the context `suspended` until a user gesture. Resume it from
  within a gesture handler: the first in-page `LEFT_DOWN`/`RIGHT_DOWN` (already
  captured for `MOUSE_EVENT`) doubles as the unlock gesture — on first mouse-down,
  if `ctx.state !== 'running'`, call `ctx.resume()`. The very first `press` may be
  silent while the context resumes; acceptable for v1.
- Construct with `{ latencyHint: 'interactive' }`. The context outlives
  individual games and is never `close()`d while the content script is alive.

### Preloading / low latency

- **Synthesis path:** nothing to decode; pre-allocate the shared `mine` noise
  buffer once at context creation. Oscillator/gain nodes are cheap, created
  per-shot.
- **Asset path (future):** `fetch` each file via `chrome.runtime.getURL(...)`,
  `decodeAudioData()` once at startup, retain the `AudioBuffer`s.

### Overlapping / rapid sounds

- `AudioBufferSourceNode`/`OscillatorNode` are one-shot — create a **fresh
  source per shot, reuse the buffer**; nodes are GC'd when finished.
- Cascade reveals are **one** logical sound (classified once per timestamp), the
  primary defense against spam.
- **Polyphony cap:** `MAX_VOICES` (default **8**); drop the newest shot beyond it.

### Debouncing

- Per-event min-interval: reject a shot if the same logical sound played less than
  `debounceMs` ago (defaults: `press`/`reveal` 20 ms, `flag`/`unflag` 40 ms,
  `chord` 60 ms, `mine` 0 ms — never debounce loss).
- `press` and `reveal` are distinct names, so a single click's press-then-reveal
  both play.

### Volume / mixing

- Each logical sound routes through its own `GainNode` into `master`. Per-event
  **enable** skips playback entirely (not zero-gain). `master` gain is the global
  volume; mute short-ramps to 0.

---

## Settings / Config Sketch

Persisted via `chrome.storage.sync` so settings follow the user across devices.

```jsonc
{
  "sound": {
    "enabled": true,            // global on/off
    "masterVolume": 0.7,        // 0.0–1.0 → master GainNode.gain
    "events": {
      "press":  { "enabled": true, "volume": 0.4 },
      "reveal": { "enabled": true, "volume": 0.8 },
      "chord":  { "enabled": true, "volume": 0.9 },
      "flag":   { "enabled": true, "volume": 0.7 },
      "unflag": { "enabled": true, "volume": 0.7 },
      "mine":   { "enabled": true, "volume": 1.0 }
    },
    "engine": "synth"           // "synth" (default) | "assets" (future)
  }
}
```

Effective gain = `events[name].volume` × `masterVolume`. `engine: "assets"` is
reserved for a future minor version and falls back to `"synth"` if decode fails.

---

## Explicit Non-Goals (v1)

- No background music / ambient loops — one-shots only.
- No per-cell pitch (number value → pitch, position → pan).
- No win fanfare — `GAME_WIN` is silent (game-end is loss-only for audio).
- No cursor / scroll / zoom / resize / blur / focus sounds.
- No distinct "triggering mine" cue (state `0x0C` is never emitted).
- No spatialization, reverb, or DSP beyond per-shot gain envelopes.
- No playback while replaying saved `.msm` files (a `.msm` player is out of scope).
- Sound never gates or mutates recording.

---

## Sources

- [MDN — Web Audio API best practices](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices)
- [MDN — AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode)
- [Chrome — Web Audio, Autoplay Policy and Games](https://developer.chrome.com/blog/web-audio-autoplay)
- [alemangui — the ugly click and the human ear](http://alemangui.github.io/ramp-to-value)
- [Kenney — Interface Sounds (CC0)](https://kenney.nl/assets/interface-sounds) · [OpenGameArt — CC0](https://opengameart.org/content/cc0-sound-effects) · [Freesound](https://www.freesound.org/)
