#!/usr/bin/env python3
"""Pack a dom-trace recording into a single, queryable SQLite file.

Each frame's full-page HTML is pretty-printed (one tag per line, so diffs are
readable) and gzip-compressed, stored keyed by frame number. Any frame can then
be fetched instantly by number with `query.py` — no full extraction, and the
file stays compact. Standard library only (sqlite3 + gzip); no dependencies.

    python pack_sqlite.py --src <recordingDir> --out <recording.sqlite>

Schema:
    frames(n INTEGER PRIMARY KEY, t INTEGER, hash TEXT, html BLOB)  -- html = gzip(pretty)
    bookmarks(n INTEGER, label TEXT)                               -- frame -> description
    meta(key TEXT PRIMARY KEY, value TEXT)
"""
import argparse
import gzip
import json
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="recording dir with frame-*.html + index.ndjson")
    ap.add_argument("--out", required=True, help="output .sqlite path")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.unlink(missing_ok=True)

    frames = {int(f.name.split("-")[1]): f for f in src.glob("frame-*.html")}

    times: dict[int, tuple] = {}
    idx = src / "index.ndjson"
    if idx.exists():
        for line in idx.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                r = json.loads(line)
                times[r["n"]] = (r.get("t"), r.get("hash"))

    con = sqlite3.connect(out)
    con.execute("CREATE TABLE frames(n INTEGER PRIMARY KEY, t INTEGER, hash TEXT, html BLOB)")
    con.execute("CREATE TABLE bookmarks(n INTEGER, label TEXT)")
    con.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT)")

    count = 0
    for n in sorted(frames):
        raw = frames[n].read_text(encoding="utf-8", errors="replace")
        pretty = raw.replace("><", ">\n<")  # one tag per line -> granular diffs
        blob = gzip.compress(pretty.encode("utf-8"), 9)
        t, h = times.get(n, (None, None))
        con.execute("INSERT INTO frames VALUES (?,?,?,?)", (n, t, h, blob))
        count += 1

    con.execute("INSERT INTO meta VALUES ('n_frames', ?)", (str(count),))
    con.execute("INSERT INTO meta VALUES ('source', ?)", (src.name,))
    con.commit()
    con.close()

    print(f"wrote {out} : {count} frames, {out.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
