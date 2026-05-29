# Minesweeper Mirror — Sound Effects (v1)

Optional, client-side **sound effects** played live by the content script while a
game is recorded. Sounds are a presentation layer on top of the recorder model
([format-events.md](format-events.md), [recording-lifecycle.md](recording-lifecycle.md)):
every sound is derived from the same `MOUSE_EVENT` (`0x20`), `BOARD_CHANGE`
(`0x30`), and `SESSION_EVENT` (`0x40`) signals the recorder already produces.
Sound playback **never** alters, gates, or is written to the `.msm` stream — it
is a pure consumer of recorder state, and audio failures are logged and swallowed
(never `RECORDING_ERROR`).

This is built for **competitive players**, which drives every decision below.

---

## Design principles

1. **Sound the *outcome*, not the input.** One sound per *meaningful result*
   (reveal / chord / flag / unflag), fired when the board change is detected —
   not on raw mouse-down. Pressing isn't an achievement; revealing is. There is
   **no standalone "press" sound** (it would double the audio events during fast
   play and reward pressing over accomplishing).
2. **Every in-game sound is short** (~30–60 ms). Top players chord in rapid
   bursts, so reveal *and* chord must be tight, not just flags.
3. **Latency beats length.** Fire immediately; sounds run from the 30 Hz poll, so
   worst-case lag is ~33 ms — below the ~50 ms "feels instant" bar. (If that ever
   feels soft, fire the base tick on the synchronous `mouseup` and only the combo
   pitch from the poll.)
4. **The combo is an efficiency meter.** Minesweeper's core skill is efficiency
   (IOE = 3BV / clicks). A clean, no-waste run makes the audio *rise in pitch*;
   wasted actions drop it back. This rewards exactly what competitive play optimizes.
5. **Never punish with sound.** A wasted move silently resets the combo — there is
   **no harsh "fail" buzz** (a constantly-used tool must not nag). Losing the rising
   pitch is the feedback.
6. **Never block, never bottleneck.** Fire-and-forget one-shots into a fixed voice
   pool; if saturated, drop silently (a missed blip is imperceptible; audio lag is not).

### Click-rate budget

Design target: **sustained ≤ 10 CPS, instantaneous ≤ 15 CPS**. 5 CPS is already
elite; >10 CPS happens only as rare bursts (e.g. fast beginner boards). Above the
target the only requirement is that audio **must not interfere with gameplay or
crash** — it degrades by dropping voices, never by stalling. (Pilot data from real
play: peak **6 CPS** over any 1 s, tightest inter-click gap **30 ms** — so the
voice pool below is mostly headroom, not a hot path.)

---

## Move taxonomy → sound

Detected from the per-poll `BOARD_CHANGE` batch plus the `MOUSE_EVENT` stream.
"Productive" = the action opened ≥ 1 new cell.

| Move | Board effect | Productive? | Sound | Combo |
|---|---|---|---|---|
| **Reveal — number** | opens 1 numbered cell | ✓ | `reveal` (short, combo-pitched) | +1 |
| **Reveal — cascade** (clicked a 0) | opens many cells at once | ✓ | `cascade` (richer reveal variant) | +1 |
| **Chord — success** | opens adjacent cells | ✓ | `chord` (short, distinct timbre, combo-pitched) | +1 |
| **Chord — no-op** (flags ≠ number) | nothing | ✗ wasted | *silent* | **reset** |
| **Flag** | cell → flagged | neutral | `flag` (fixed pitch) | neutral |
| **Unflag** | flag → closed | neutral | `unflag` (fixed pitch) | neutral¹ |
| **No-op click** (on opened/flagged cell, no chord) | nothing | ✗ wasted | *silent* | **reset** |
| **Hit mine** (bad reveal or chord) | loss | terminal | `mine` (short low thud) | — |
| **Win** (last safe cell) | win | terminal | `win` (combo-scaled arpeggio) | — |

¹ Flagging is **combo-neutral** — elite "no-flag" players never flag, and flag
placement is legitimate setup we can't grade live. *Exception:* a flag immediately
removed (flag↔unflag on the same cell within `flagThrashWindowMs`) is treated as a
**wasted** action and **resets** the combo.

### Detection notes

- **Reveal vs. chord vs. cascade.** Classify **once per timestamp `t`** (a cascade
  is one decision, not N):
  - **Chord** = the click was over an already-opened *number* cell (`0x01`–`0x08`)
    and the batch opened ≥ 1 cell. Correctness is inferred from the *effect* (it
    revealed something) — flag/mine correctness isn't knowable synchronously, and
    chording is method-agnostic (the player may use L+R, the 1.5-click, or
    middle-click; pilot data shows middle-click chording is common). A chord that
    opens nothing is the wasted "no-op".
  - **Reveal** = the click was over a *closed* cell (`0x09`). If the batch opened
    ≥ `cascadeMinCells`, use the richer `cascade` voice; otherwise `reveal`.
- **Flag / unflag.** From `BOARD_CHANGE` transitions to/from flagged (`0x0A`),
  independent of input method.
- **Mine.** Driven solely by `GAME_LOSS` (`0x03`), not the `0x0B`/`0x0D` reveal
  states. On loss the recorder does one final poll that captures the mine reveal
  then stops; the audio layer must **suppress** `reveal`/`chord`/`cascade` for that
  terminal batch and play only the `mine` thud. The site doesn't mark the
  triggering mine, so it's one sound. A loss is not celebrated — `mine` is a short,
  deflating low thud (mutable; players don't replay losses).
- **Win.** On `GAME_WIN`, play `win`: an ascending arpeggio up the combo `scale`,
  its height/length reflecting the combo reached — the efficiency payoff.

---

## Combo system (the efficiency meter)

A single `comboStep` spans all productive moves (reveal + chord + cascade); flags
are neutral.

- **Productive move** → `comboStep = min(comboStep + 1, maxSteps)`.
- **Wasted move** (no-op click, failed chord, flag↔unflag thrash) → `comboStep = 0`.
- **Idle** ≥ `idleResetMs` with no productive move → `comboStep = 0`. (Thinking is
  *not* waste in Minesweeper, so this is generous — pilot data shows the longest
  real pause was 0.8 s, so 3 s rarely triggers; it's a staleness guard.)
- **Pitch:** the `reveal`/`chord`/`cascade` voice is transposed up by
  `scale[comboStep]` semitones (a repeating pentatonic table → it always sounds
  musical ascending), capped at `maxSteps`, then holds. At `comboStep = 0` it plays
  at base pitch. No sound on reset — the dropped pitch *is* the signal.

Net effect: a flawless efficient clear sings steadily upward; a fumble quietly
drops you to the bottom of the ladder. The audio literally tracks your IOE.

---

## Voice management

- One `AudioContext`; a `master` `GainNode` → `ctx.destination`; per-event
  `GainNode`s → `master`. Each shot is a fresh `OscillatorNode`/`AudioBufferSourceNode`
  (one-shot; reuse the buffer, GC the node).
- **Pool of `maxVoices` (default 12).** New sound takes a free voice; if none are
  free, **steal the oldest** with a `declickFadeMs` (~4 ms) release ramp so it fades
  instead of popping. **Never hard-stop on a new move** (that pops and feels
  chopped). Terminal sounds (`mine`, `win`) never contend (the game is over).
- Past the cap, **drop the new shot silently** — never queue or block.
- No per-event debounce-drop: we *want* every productive move to sound (it feeds
  the combo). Overlap is handled by short sounds + the pool, not by suppression.

---

## Tunable parameters (single source of truth)

All knobs live in **one** object so they can become user-configurable later
without hunting through code. Defaults below; persisted via `chrome.storage.sync`.

```jsonc
{
  "sound": {
    "enabled": true,
    "masterVolume": 0.7,           // 0.0–1.0 → master GainNode.gain

    "maxVoices": 12,               // concurrent one-shots; excess dropped silently
    "declickFadeMs": 4,            // release ramp when stealing a voice

    "combo": {
      "idleResetMs": 3000,         // no productive move this long → reset
      "maxSteps": 8,               // pitch ladder caps here, then holds
      "scale": [0, 2, 4, 7, 9],    // semitone offsets, repeats +12 per octave (major pentatonic)
      "flagThrashWindowMs": 800,   // flag then unflag same cell within this = wasted
      "cascadeMinCells": 4         // ≥ this many opened in one poll → "cascade" voice
    },

    "events": {
      "reveal":  { "enabled": true, "volume": 0.70, "baseHz": 523, "durMs": 45 },  // combo-pitched
      "chord":   { "enabled": true, "volume": 0.80, "baseHz": 523, "durMs": 45 },  // combo-pitched, distinct timbre
      "cascade": { "enabled": true, "volume": 0.80,                "durMs": 90 },  // richer reveal, combo-pitched
      "flag":    { "enabled": true, "volume": 0.60, "hz": 880,     "durMs": 35 },  // fixed, combo-neutral
      "unflag":  { "enabled": true, "volume": 0.60, "hz": 587,     "durMs": 35 },
      "mine":    { "enabled": true, "volume": 0.80,                "durMs": 220 }, // low deflating thud
      "win":     { "enabled": true, "volume": 0.90 }                              // combo-scaled arpeggio
    },

    "engine": "synth"              // "synth" (default) | "assets" (future)
  }
}
```

Effective gain for a shot = `events[name].volume` × `masterVolume`; `enabled:false`
at either level skips the shot entirely. Tune the *feel* by editing this block only.

---

## Web Audio implementation

- **Autoplay gesture:** the context starts `suspended`; resume it from the first
  in-page mouse-down (already captured for `MOUSE_EVENT`) — `if (ctx.state !==
  'running') ctx.resume()`. The very first sound may be silent while it resumes;
  acceptable. Construct with `{ latencyHint: 'interactive' }`.
- **Synthesis path (default):** nothing to decode; pre-allocate the shared `mine`
  noise buffer once. Combo pitch is just `baseHz * 2 ** (semitones / 12)`.
- **Asset path (future):** `fetch` via `chrome.runtime.getURL`, `decodeAudioData`
  once at startup, retain `AudioBuffer`s in the registry; `engine: "assets"` selects
  it and falls back to `"synth"` on decode failure.
- Always ramp gain with `linearRampToValueAtTime`/`setTargetAtTime` (never start/stop
  at non-zero gain) to avoid click artifacts.

---

## Audio approach: synthesis-first, asset-optional

Default to **runtime synthesis** (zero asset friction, tiny bundle, lowest latency,
and it composes with the combo pitch ladder). Keep the player behind a **sound
registry** keyed by logical name so a future minor version can swap a synth voice
for a decoded CC0 `AudioBuffer` per event without touching the detector. Candidate
CC0 sources if recorded assets are ever wanted (no attribution, redistributable):
[Kenney Interface/UI Audio](https://kenney.nl/assets/interface-sounds),
[OpenGameArt CC0](https://opengameart.org/content/cc0-sound-effects),
[Freesound (CC0 filter)](https://www.freesound.org/).

---

## Explicit non-goals (v1)

- No standalone press/click sound (outcome-driven only).
- No background music / ambient loops — one-shots only.
- No per-cell pitch by *value* or *position* (combo pitch is by *efficiency*, not
  which number/where).
- No cursor / scroll / zoom / resize / blur / focus sounds.
- No distinct "triggering mine" cue (state `0x0C` is never emitted).
- No spatialization, reverb, or DSP beyond per-shot gain envelopes.
- No playback while replaying saved `.msm` files (a player is out of scope).
- Sound never gates or mutates recording.

---

## Sources

- [MDN — Web Audio API best practices](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API/Best_practices) · [AudioBufferSourceNode](https://developer.mozilla.org/en-US/docs/Web/API/AudioBufferSourceNode)
- [Chrome — Web Audio, Autoplay Policy and Games](https://developer.chrome.com/blog/web-audio-autoplay)
- [alemangui — the ugly click and the human ear](http://alemangui.github.io/ramp-to-value)
