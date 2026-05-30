#!/usr/bin/env python3
"""Query a packed recording (see pack_sqlite.py). Standard library only.

    python query.py <recording.sqlite> show <n> [--out FILE]   # frame n's HTML (stdout or file)
    python query.py <recording.sqlite> diff <a> <b>            # unified diff of frames a and b
    python query.py <recording.sqlite> bookmarks               # list bookmarks
    python query.py <recording.sqlite> info                    # frame count / range
"""
import argparse
import difflib
import gzip
import sqlite3
import sys


def html_of(con: sqlite3.Connection, n: int) -> str:
    row = con.execute("SELECT html FROM frames WHERE n=?", (n,)).fetchone()
    if row is None:
        sys.exit(f"no frame {n}")
    return gzip.decompress(row[0]).decode("utf-8", "replace")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("show"); p.add_argument("n", type=int); p.add_argument("--out")
    p = sub.add_parser("diff"); p.add_argument("a", type=int); p.add_argument("b", type=int)
    sub.add_parser("bookmarks")
    sub.add_parser("info")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    if args.cmd == "show":
        html = html_of(con, args.n)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(html)
            print(f"wrote {args.out}")
        else:
            sys.stdout.write(html)

    elif args.cmd == "diff":
        a = html_of(con, args.a).splitlines(keepends=True)
        b = html_of(con, args.b).splitlines(keepends=True)
        sys.stdout.writelines(difflib.unified_diff(a, b, f"frame-{args.a}", f"frame-{args.b}"))

    elif args.cmd == "bookmarks":
        rows = con.execute("SELECT n, label FROM bookmarks ORDER BY n").fetchall()
        if not rows:
            print("(no bookmarks loaded)")
        for n, label in rows:
            print(f"{n:5d}  {label}")

    elif args.cmd == "info":
        n = con.execute("SELECT COUNT(*), MIN(n), MAX(n) FROM frames").fetchone()
        print(f"frames={n[0]}  range={n[1]}..{n[2]}")


if __name__ == "__main__":
    main()
