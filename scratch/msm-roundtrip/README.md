# MSM Format Roundtrip Prototype

Validates the `.msm` binary session format spec by encoding a synthetic
session, decoding it back, and asserting structural equality.

This is a **prototype**, not production code. It exists to:

- Sanity-check the byte arithmetic in `docs/format-session.md` and
  `docs/format-events.md` before we commit to it in the extension.
- Surface ambiguities (endianness, padding, edge cases) early.
- Serve as a reference for the final TypeScript implementation that will
  live in `extension/src/msm/`.

## Run

```powershell
cd scratch/msm-roundtrip
npm install
npm test
```

## Layout

```
scratch/msm-roundtrip/
├── package.json
├── tsconfig.json
├── src/
│   ├── types.ts      ← TypeScript types mirroring the spec
│   ├── encoder.ts    ← Session → Uint8Array
│   ├── decoder.ts    ← Uint8Array → Session
│   └── roundtrip.ts  ← Build synthetic session, encode, decode, assert
└── README.md
```

## What we learn

- Whether the spec's byte offsets are self-consistent.
- Whether `DataView` + `Uint8Array` is ergonomic enough for the real
  implementation (spoiler: yes).
- What the realistic file size is for a representative session.
