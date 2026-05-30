#!/usr/bin/env python3
"""Load a frame->description bookmark TSV into a packed recording's `bookmarks` table.

    python load_bookmarks.py --db <recording.sqlite> --tsv <bookmarks.tsv>

TSV: each line is `<frame>\\t<description>`. Blank lines and lines starting with
`#` are ignored. Replaces any existing bookmarks. Standard library only.
"""
import argparse
import sqlite3
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--tsv", required=True)
    args = ap.parse_args()

    rows = []
    for line in Path(args.tsv).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        n, _, label = line.partition("\t")
        rows.append((int(n.strip()), label.strip()))

    con = sqlite3.connect(args.db)
    con.execute("DELETE FROM bookmarks")
    con.executemany("INSERT INTO bookmarks(n, label) VALUES (?,?)", rows)
    con.commit()
    con.close()
    print(f"loaded {len(rows)} bookmarks into {args.db}")


if __name__ == "__main__":
    main()
