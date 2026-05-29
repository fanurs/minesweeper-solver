import { strict as assert } from 'node:assert';

import { decodeSession } from './decoder.js';
import { encodeSession } from './encoder.js';
import {
  CELL_STATE,
  FORMAT_VERSION,
  FOOTER_BYTES,
  FOOTER_RESULT,
  HEADER_BYTES,
  MOUSE_ACTION,
  MsmSession,
  SESSION_EVENT_TYPE,
} from './types.js';

function buildSyntheticSession(): MsmSession {
  // Mimic an expert (30x16, 99 mines) game that runs a few seconds:
  // - GAME_START at t=0
  // - initial CURSOR_ANCHOR at t=0
  // - some cursor samples (anchor every 30th)
  // - a mouse left-down → left-up sequence that reveals a cell
  // - a board change (cell opens to type 1)
  // - a scroll event mid-game (mandatory CURSOR_ANCHOR follows)
  // - a flag (right-down → right-up + board change to flagged)
  // - GAME_WIN

  const records: MsmSession['records'] = [];

  records.push({ kind: 'session_event', t: 0, type: SESSION_EVENT_TYPE.GAME_START });
  records.push({ kind: 'cursor_anchor', t: 0, x: 0, y: 0 });

  for (let i = 1; i <= 29; i++) {
    records.push({ kind: 'cursor', x: i * 0.1, y: i * 0.1 });
  }
  records.push({ kind: 'cursor_anchor', t: 1000, x: 3.0, y: 3.0 });

  records.push({ kind: 'mouse_event', t: 1050, action: MOUSE_ACTION.LEFT_DOWN, x: 3.2, y: 3.2 });
  records.push({ kind: 'mouse_event', t: 1100, action: MOUSE_ACTION.LEFT_UP, x: 3.2, y: 3.2 });
  records.push({ kind: 'board_change', t: 1101, row: 3, col: 3, state: CELL_STATE.OPEN_1 });

  records.push({ kind: 'scroll_event', t: 2000, dx: 0, dy: 1.5 });
  records.push({ kind: 'cursor_anchor', t: 2000, x: 3.2, y: 1.7 });

  records.push({ kind: 'mouse_event', t: 2500, action: MOUSE_ACTION.RIGHT_DOWN, x: 4.5, y: 5.5 });
  records.push({ kind: 'mouse_event', t: 2550, action: MOUSE_ACTION.RIGHT_UP, x: 4.5, y: 5.5 });
  records.push({ kind: 'board_change', t: 2551, row: 5, col: 4, state: CELL_STATE.FLAGGED });

  records.push({ kind: 'resize_event', t: 3000, boardPxW: 800, boardPxH: 432 });
  records.push({ kind: 'cursor_anchor', t: 3000, x: 4.5, y: 5.5 });

  records.push({ kind: 'zoom_event', t: 3500, scale: 1.25 });
  records.push({ kind: 'cursor_anchor', t: 3500, x: 4.5, y: 5.5 });

  records.push({ kind: 'session_event', t: 4000, type: SESSION_EVENT_TYPE.GAME_WIN });

  return {
    header: {
      version: FORMAT_VERSION,
      rows: 16,
      cols: 30,
      mines: 99,
      epochStartMs: 1748390400000n,
      url: 'https://minesweeper.online/game/6103894316',
      initPxW: 780,
      initPxH: 416,
      comment: 'roundtrip test session — synthetic',
    },
    records,
    footer: {
      result: FOOTER_RESULT.WIN,
      stats: {
        durationMs: 4000,
        timeS: 4.000,
        bbbv: 150,
        bbbvPerS: 37.5,
        clicksL: 80,
        clicksR: 70,
        cps: 37.5,
        efficiency: 99,
        ioe: 0.95,
        ops: 22,
        thrp: 1.234,
        corr: 0.876,
        zini: 130,
        zne: 0.5,
        znt: 0.3,
        rqp: 0.107,
        ios: 8.4,
        // estimatedTime intentionally omitted — bit 17 stays 0
      },
    },
  };
}

function assertSessionsEqual(a: MsmSession, b: MsmSession): void {
  assert.equal(a.header.version, b.header.version, 'header.version');
  assert.equal(a.header.rows, b.header.rows, 'header.rows');
  assert.equal(a.header.cols, b.header.cols, 'header.cols');
  assert.equal(a.header.mines, b.header.mines, 'header.mines');
  assert.equal(a.header.epochStartMs, b.header.epochStartMs, 'header.epochStartMs');
  assert.equal(a.header.url, b.header.url, 'header.url');
  assert.equal(a.header.initPxW, b.header.initPxW, 'header.initPxW');
  assert.equal(a.header.initPxH, b.header.initPxH, 'header.initPxH');
  assert.equal(a.header.comment, b.header.comment, 'header.comment');

  assert.equal(a.records.length, b.records.length, 'record count');
  for (let i = 0; i < a.records.length; i++) {
    assertRecordsEqual(a.records[i]!, b.records[i]!, i);
  }

  assert.equal(a.footer.result, b.footer.result, 'footer.result');
  const aKeys = Object.keys(a.footer.stats).sort();
  const bKeys = Object.keys(b.footer.stats).sort();
  assert.deepEqual(aKeys, bKeys, 'footer stat key set');
  for (const key of aKeys as Array<keyof MsmSession['footer']['stats']>) {
    const av = a.footer.stats[key]!;
    const bv = b.footer.stats[key]!;
    // float32 loses precision; allow tiny epsilon
    if (Number.isInteger(av)) {
      assert.equal(av, bv, `footer.stats.${key}`);
    } else {
      assert.ok(
        Math.abs(av - bv) < 1e-4,
        `footer.stats.${key}: ${av} vs ${bv} (float32 precision)`,
      );
    }
  }
}

function assertRecordsEqual(a: MsmSession['records'][number], b: MsmSession['records'][number], idx: number): void {
  assert.equal(a.kind, b.kind, `record[${idx}].kind`);
  for (const key of Object.keys(a) as Array<keyof typeof a>) {
    const av = (a as any)[key];
    const bv = (b as any)[key];
    if (typeof av === 'number' && !Number.isInteger(av)) {
      assert.ok(
        Math.abs(av - bv) < 1e-4,
        `record[${idx}].${String(key)}: ${av} vs ${bv} (float32 precision)`,
      );
    } else {
      assert.equal(av, bv, `record[${idx}].${String(key)}`);
    }
  }
}

function main(): void {
  const session = buildSyntheticSession();
  const bytes = encodeSession(session);

  console.log(`Encoded ${session.records.length} records.`);
  console.log(`Total file size: ${bytes.byteLength} bytes`);
  console.log(`  Header:  ${HEADER_BYTES} bytes`);
  console.log(`  Records: ${bytes.byteLength - HEADER_BYTES - FOOTER_BYTES} bytes`);
  console.log(`  Footer:  ${FOOTER_BYTES} bytes`);

  // Sanity: first byte after header should be 0x40 (GAME_START SESSION_EVENT)
  assert.equal(bytes[HEADER_BYTES], 0x40, 'first record tag is SESSION_EVENT');
  // Last 65 bytes start with 0xFF
  assert.equal(bytes[bytes.byteLength - FOOTER_BYTES], 0xff, 'footer sentinel present at expected offset');

  const decoded = decodeSession(bytes);

  // Re-encode the decoded session and verify byte-equality
  const reencoded = encodeSession(decoded);
  assert.equal(
    reencoded.byteLength,
    bytes.byteLength,
    'roundtripped byte length matches',
  );
  for (let i = 0; i < bytes.byteLength; i++) {
    if (bytes[i] !== reencoded[i]) {
      throw new Error(
        `byte mismatch at offset ${i}: 0x${bytes[i]!.toString(16)} vs 0x${reencoded[i]!.toString(16)}`,
      );
    }
  }

  // Also verify decoded structure matches the source session
  assertSessionsEqual(session, decoded);

  console.log('\n✓ Encoder/decoder roundtrip OK');
  console.log('✓ Byte-equal after re-encode');
  console.log('✓ Decoded structure equals source');

  // Print null_bitmap for visual inspection
  const view = new DataView(bytes.buffer, bytes.byteOffset + bytes.byteLength - FOOTER_BYTES, FOOTER_BYTES);
  const bitmap = view.getUint32(2, true);
  console.log(`\nnull_bitmap = 0x${bitmap.toString(16).padStart(8, '0')} (${bitmap.toString(2).padStart(18, '0')})`);
  console.log('  bit 17 (estimated_time) should be 0:', ((bitmap >>> 17) & 1) === 0 ? 'OK' : 'FAIL');
  console.log('  bits 0-16 should all be 1:', (bitmap & 0x1ffff) === 0x1ffff ? 'OK' : 'FAIL');
}

main();
