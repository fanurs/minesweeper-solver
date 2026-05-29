import {
  FOOTER_BYTES,
  HEADER_BYTES,
  MsmFooter,
  MsmHeader,
  MsmSession,
  RECORD_BYTES,
  Record,
  STAT_BITS,
  TAG,
} from './types.js';

const LITTLE_ENDIAN = true;

function writeFixedUtf8(view: DataView, offset: number, length: number, value: string): void {
  const bytes = new TextEncoder().encode(value);
  if (bytes.length > length) {
    throw new Error(
      `string of byte length ${bytes.length} exceeds fixed field of ${length} bytes`,
    );
  }
  const u8 = new Uint8Array(view.buffer, view.byteOffset + offset, length);
  u8.fill(0);
  u8.set(bytes, 0);
}

export function encodeHeader(header: MsmHeader): Uint8Array {
  const buf = new Uint8Array(HEADER_BYTES);
  const view = new DataView(buf.buffer);

  writeFixedUtf8(view, 0, 32, header.version);
  view.setUint8(32, header.rows);
  view.setUint8(33, header.cols);
  view.setUint16(34, header.mines, LITTLE_ENDIAN);
  view.setBigUint64(36, header.epochStartMs, LITTLE_ENDIAN);
  writeFixedUtf8(view, 44, 128, header.url);
  view.setUint16(172, header.initPxW, LITTLE_ENDIAN);
  view.setUint16(174, header.initPxH, LITTLE_ENDIAN);
  writeFixedUtf8(view, 176, 512, header.comment);

  return buf;
}

export function encodeRecord(record: Record): Uint8Array {
  switch (record.kind) {
    case 'cursor': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.CURSOR]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.CURSOR);
      v.setFloat32(1, record.x, LITTLE_ENDIAN);
      v.setFloat32(5, record.y, LITTLE_ENDIAN);
      return buf;
    }
    case 'cursor_anchor': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.CURSOR_ANCHOR]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.CURSOR_ANCHOR);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setFloat32(5, record.x, LITTLE_ENDIAN);
      v.setFloat32(9, record.y, LITTLE_ENDIAN);
      return buf;
    }
    case 'mouse_event': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.MOUSE_EVENT]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.MOUSE_EVENT);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setUint8(5, record.action);
      v.setFloat32(6, record.x, LITTLE_ENDIAN);
      v.setFloat32(10, record.y, LITTLE_ENDIAN);
      return buf;
    }
    case 'scroll_event': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.SCROLL_EVENT]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.SCROLL_EVENT);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setFloat32(5, record.dx, LITTLE_ENDIAN);
      v.setFloat32(9, record.dy, LITTLE_ENDIAN);
      return buf;
    }
    case 'zoom_event': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.ZOOM_EVENT]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.ZOOM_EVENT);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setFloat32(5, record.scale, LITTLE_ENDIAN);
      return buf;
    }
    case 'board_change': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.BOARD_CHANGE]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.BOARD_CHANGE);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setUint8(5, record.row);
      v.setUint8(6, record.col);
      v.setUint8(7, record.state);
      return buf;
    }
    case 'session_event': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.SESSION_EVENT]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.SESSION_EVENT);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setUint8(5, record.type);
      return buf;
    }
    case 'resize_event': {
      const buf = new Uint8Array(RECORD_BYTES[TAG.RESIZE_EVENT]);
      const v = new DataView(buf.buffer);
      v.setUint8(0, TAG.RESIZE_EVENT);
      v.setUint32(1, record.t, LITTLE_ENDIAN);
      v.setUint16(5, record.boardPxW, LITTLE_ENDIAN);
      v.setUint16(7, record.boardPxH, LITTLE_ENDIAN);
      return buf;
    }
  }
}

type StatWriter = (v: DataView, offset: number, value: number) => void;
const u16: StatWriter = (v, o, x) => v.setUint16(o, x, LITTLE_ENDIAN);
const u32: StatWriter = (v, o, x) => v.setUint32(o, x, LITTLE_ENDIAN);
const u8s: StatWriter = (v, o, x) => v.setUint8(o, x);
const f32: StatWriter = (v, o, x) => v.setFloat32(o, x, LITTLE_ENDIAN);

interface StatField {
  bit: number;
  offset: number;
  write: StatWriter;
  key: keyof MsmFooter['stats'];
}

const STAT_FIELDS: StatField[] = [
  { bit: STAT_BITS.durationMs,    offset:  6, write: u32, key: 'durationMs' },
  { bit: STAT_BITS.timeS,         offset: 10, write: f32, key: 'timeS' },
  { bit: STAT_BITS.bbbv,          offset: 14, write: u16, key: 'bbbv' },
  { bit: STAT_BITS.bbbvPerS,      offset: 16, write: f32, key: 'bbbvPerS' },
  { bit: STAT_BITS.clicksL,       offset: 20, write: u16, key: 'clicksL' },
  { bit: STAT_BITS.clicksR,       offset: 22, write: u16, key: 'clicksR' },
  { bit: STAT_BITS.cps,           offset: 24, write: f32, key: 'cps' },
  { bit: STAT_BITS.efficiency,    offset: 28, write: u8s, key: 'efficiency' },
  { bit: STAT_BITS.ioe,           offset: 29, write: f32, key: 'ioe' },
  { bit: STAT_BITS.ops,           offset: 33, write: u16, key: 'ops' },
  { bit: STAT_BITS.thrp,          offset: 35, write: f32, key: 'thrp' },
  { bit: STAT_BITS.corr,          offset: 39, write: f32, key: 'corr' },
  { bit: STAT_BITS.zini,          offset: 43, write: u16, key: 'zini' },
  { bit: STAT_BITS.zne,           offset: 45, write: f32, key: 'zne' },
  { bit: STAT_BITS.znt,           offset: 49, write: f32, key: 'znt' },
  { bit: STAT_BITS.rqp,           offset: 53, write: f32, key: 'rqp' },
  { bit: STAT_BITS.ios,           offset: 57, write: f32, key: 'ios' },
  { bit: STAT_BITS.estimatedTime, offset: 61, write: f32, key: 'estimatedTime' },
];

export function encodeFooter(footer: MsmFooter): Uint8Array {
  const buf = new Uint8Array(FOOTER_BYTES);
  const view = new DataView(buf.buffer);

  view.setUint8(0, TAG.FOOTER);
  view.setUint8(1, footer.result);

  let bitmap = 0;
  for (const field of STAT_FIELDS) {
    const value = footer.stats[field.key];
    if (value !== undefined) {
      bitmap |= 1 << field.bit;
      field.write(view, field.offset, value);
    }
  }
  view.setUint32(2, bitmap >>> 0, LITTLE_ENDIAN);

  return buf;
}

export function encodeSession(session: MsmSession): Uint8Array {
  const recordBuffers = session.records.map(encodeRecord);
  const totalSize =
    HEADER_BYTES +
    recordBuffers.reduce((sum, r) => sum + r.byteLength, 0) +
    FOOTER_BYTES;

  const out = new Uint8Array(totalSize);
  let offset = 0;

  out.set(encodeHeader(session.header), offset);
  offset += HEADER_BYTES;

  for (const buf of recordBuffers) {
    out.set(buf, offset);
    offset += buf.byteLength;
  }

  out.set(encodeFooter(session.footer), offset);
  return out;
}
