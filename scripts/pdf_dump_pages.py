#!/usr/bin/env python3
"""Dump the text of selected PDF pages (for diagnosing float placement)."""
import sys

import pypdf

path = sys.argv[1]
pages = [int(a) for a in sys.argv[2:]]
reader = pypdf.PdfReader(path)
for p in pages:
    txt = reader.pages[p - 1].extract_text() or ""
    lines = [ln for ln in txt.splitlines() if ln.strip()]
    print(f"########## PAGE {p}  ({len(lines)} lines) ##########")
    for ln in lines:
        print("  " + ln[:150])
    print()
