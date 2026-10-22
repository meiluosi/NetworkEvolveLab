#!/usr/bin/env python3
"""Report the first line, last line, and character count of each PDF page.

Used to verify the ICWSM page budget (body + references <= 11 pages) and to
locate where the reference list and the checklist start.
"""
from __future__ import annotations

import argparse
import re
import sys

import pypdf


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", nargs="?", default="build/icwsm/NetworkEvolveLab_ICWSM.pdf")
    ap.add_argument("--markers", nargs="*", default=[])
    args = ap.parse_args()

    reader = pypdf.PdfReader(args.pdf)
    print(f"total pages: {len(reader.pages)}")
    markers = [m.lower() for m in args.markers]
    for i, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        chars = len(norm(raw))
        first = lines[0][:78] if lines else "<empty>"
        last = lines[-1][:78] if lines else "<empty>"
        hits = []
        for m in markers:
            if m in raw.lower():
                hits.append(m)
        flag = ("  <<< " + ",".join(hits)) if hits else ""
        print(f"p{i:>3} chars={chars:>5} lines={len(lines):>3} | {first}{flag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
