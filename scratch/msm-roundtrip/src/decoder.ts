import {
  FOOTER_BYTES,
  HEADER_BYTES,
  MsmFooter,
  MsmHeader,
  MsmSession,
  RECORD_BYTES,
  Record,
  TAG,
} from './types.js';

const LITTLE_ENDIAN = true;

function readFixedUtf8(view: DataView, offset: number, length: number): string {
  const bytes = new Uint8Array(view.buffer, view.byteOffset + offset, length);
  let end = 0;
  for (let i = 0; i < length; i++) {
    if (bytes[i] !== 0) end = i + 1;
  }
  return new TextDecoder('utf-8').decode(bytes.subarray(0, end));
}

export function decodeHeader(bytes: Uint8Array): MsmHeader {
  if (bytes.byteLength < HEADER_BYTES) {
    throw new Error(`header buffer too small: ${bytes.byteLength} < ${HEADER_BYTES}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset, HEADER_BYTES);

  return {
    version: readFixedUtf8(view, 0, 32),
    rows: view.getUint8(32),
    cols: view.getUint8(33),
    mines: view.getUint16(34, LITTLE_ENDIAN),
    epochStartMs: view.getBigUint64(36, LITTLE_ENDIAN),
    url: readFixedUtf8(view, 44, 128),
    initPxW: view.getUint16(172, LITTLE_ENDIAN),
    initPxH: view.getUint16(174, LITTLE_ENDIAN),
    comment: readFixedUtf8(view, 176, 512),
  };
}

interface DecodedRecord {
  record: Record;
  bytesConsumed: number;
}

function decodeRecordAt(bytes: Uint8Array, offset: number): DecodedRecord {
  const tag = bytes[offset];
  if (tag === undefined) {
    throw new Error(`unexpected EOF at offset ${offset}`);
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset);

  switch (tag) {
    case TAG.CURSOR:
      return {
        record: {
          kind: 'cursor',
          x: view.getFloat32(1, LITTLE_ENDIAN),
          y: view.getFloat32(5, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.CURSOR],
      };
    case TAG.CURSOR_ANCHOR:
      return {
        record: {
          kind: 'cursor_anchor',
          t: view.getUint32(1, LITTLE_ENDIAN),
          x: view.getFloat32(5, LITTLE_ENDIAN),
          y: view.getFloat32(9, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.CURSOR_ANCHOR],
      };
    case TAG.MOUSE_EVENT:
      return {
        record: {
          kind: 'mouse_event',
          t: view.getUint32(1, LITTLE_ENDIAN),
          action: view.getUint8(5),
          x: view.getFloat32(6, LITTLE_ENDIAN),
          y: view.getFloat32(10, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.MOUSE_EVENT],
      };
    case TAG.SCROLL_EVENT:
      return {
        record: {
          kind: 'scroll_event',
          t: view.getUint32(1, LITTLE_ENDIAN),
          dx: view.getFloat32(5, LITTLE_ENDIAN),
          dy: view.getFloat32(9, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.SCROLL_EVENT],
      };
    case TAG.ZOOM_EVENT:
      return {
        record: {
          kind: 'zoom_event',
          t: view.getUint32(1, LITTLE_ENDIAN),
          scale: view.getFloat32(5, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.ZOOM_EVENT],
      };
    case TAG.BOARD_CHANGE:
      return {
        record: {
          kind: 'board_change',
          t: view.getUint32(1, LITTLE_ENDIAN),
          row: view.getUint8(5),
          col: view.getUint8(6),
          state: view.getUint8(7),
        },
        bytesConsumed: RECORD_BYTES[TAG.BOARD_CHANGE],
      };
    case TAG.SESSION_EVENT:
      return {
        record: {
          kind: 'session_event',
          t: view.getUint32(1, LITTLE_ENDIAN),
          type: view.getUint8(5),
        },
        bytesConsumed: RECORD_BYTES[TAG.SESSION_EVENT],
      };
    case TAG.RESIZE_EVENT:
      return {
        record: {
          kind: 'resize_event',
          t: view.getUint32(1, LITTLE_ENDIAN),
          boardPxW: view.getUint16(5, LITTLE_ENDIAN),
          boardPxH: view.getUint16(7, LITTLE_ENDIAN),
        },
        bytesConsumed: RECORD_BYTES[TAG.RESIZE_EVENT],
      };
    default:
      throw new Error(
        `unknown record tag 0x${tag.toString(16).padStart(2, '0')} at offset ${offset}`,
      );
  }
}

function decodeFooter(bytes: Uint8Array, offset: number): MsmFooter {
  if (offset + FOOTER_BYTES > bytes.byteLength) {
    throw new Error('footer would extend past EOF');
  }
  const view = new DataView(bytes.buffer, bytes.byteOffset + offset, FOOTER_BYTES);

  const tag = view.getUint8(0);
  if (tag !== TAG.FOOTER) {
    throw new Error(`expected footer sentinel 0xFF at offset ${offset}, got 0x${tag.toString(16)}`);
  }
  const result = view.getUint8(1);
  const bitmap = view.getUint32(2, LITTLE_ENDIAN);

  type Reader = (v: DataView, o: number) => number;
  const u16: Reader = (v, o) => v.getUint16(o, LITTLE_ENDIAN);
  const u32: Reader = (v, o) => v.getUint32(o, LITTLE_ENDIAN);
  const u8s: Reader = (v, o) => v.getUint8(o);
  const f32: Reader = (v, o) => v.getFloat32(o, LITTLE_ENDIAN);

  const fields: Array<{
    bit: number;
    offset: number;
    read: Reader;
    key: keyof MsmFooter['stats'];
  }> = [
    { bit: 0,  offset:  6, read: u32, key: 'durationMs' },
    { bit: 1,  offset: 10, read: f32, key: 'timeS' },
    { bit: 2,  offset: 14, read: u16, key: 'bbbv' },
    { bit: 3,  offset: 16, read: f32, key: 'bbbvPerS' },
    { bit: 4,  offset: 20, read: u16, key: 'clicksL' },
    { bit: 5,  offset: 22, read: u16, key: 'clicksR' },
    { bit: 6,  offset: 24, read: f32, key: 'cps' },
    { bit: 7,  offset: 28, read: u8s, key: 'efficiency' },
    { bit: 8,  offset: 29, read: f32, key: 'ioe' },
    { bit: 9,  offset: 33, read: u16, key: 'ops' },
    { bit: 10, offset: 35, read: f32, key: 'thrp' },
    { bit: 11, offset: 39, read: f32, key: 'corr' },
    { bit: 12, offset: 43, read: u16, key: 'zini' },
    { bit: 13, offset: 45, read: f32, key: 'zne' },
    { bit: 14, offset: 49, read: f32, key: 'znt' },
    { bit: 15, offset: 53, read: f32, key: 'rqp' },
    { bit: 16, offset: 57, read: f32, key: 'ios' },
    { bit: 17, offset: 61, read: f32, key: 'estimatedTime' },
  ];

  const stats: MsmFooter['stats'] = {};
  for (const f of fields) {
    if (((bitmap >>> f.bit) & 1) === 0) continue;
    stats[f.key] = f.read(view, f.offset);
  }

  return { result, stats };
}

export function decodeSession(bytes: Uint8Array): MsmSession {
  const header = decodeHeader(bytes);
  const records: Record[] = [];

  let offset = HEADER_BYTES;
  while (offset < bytes.byteLength) {
    const tag = bytes[offset];
    if (tag === TAG.FOOTER) break;
    const { record, bytesConsumed } = decodeRecordAt(bytes, offset);
    records.push(record);
    offset += bytesConsumed;
  }

  if (offset >= bytes.byteLength) {
    throw new Error('EOF reached before footer sentinel 0xFF');
  }

  const footer = decodeFooter(bytes, offset);
  if (offset + FOOTER_BYTES !== bytes.byteLength) {
    throw new Error(
      `trailing bytes after footer: expected end at ${offset + FOOTER_BYTES}, got ${bytes.byteLength}`,
    );
  }

  return { header, records, footer };
}
