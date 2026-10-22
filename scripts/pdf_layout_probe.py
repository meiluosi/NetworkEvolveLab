#!/usr/bin/env python3
"""按坐标分析 PDF 每一页的排版：左栏/右栏各有多少行，右栏是否为空。

pypdf 的 extract_text() 在双栏页面上返回的顺序不等于视觉顺序，
因此判断「右栏是不是空的」必须用字符的实际 x/y 坐标。
"""
from __future__ import annotations

import sys
from collections import defaultdict

import pypdf

PAGE_W = 612.0   # letter paper in points
PAGE_H = 792.0


def analyse(path: str) -> None:
    reader = pypdf.PdfReader(path)
    print(f"{'page':>5} {'left lines':>11} {'right lines':>12} {'top float y0':>14} {'bottom y1':>10}")
    for i, page in enumerate(reader.pages, start=1):
        rows = defaultdict(set)
        ys = []

        def visitor(text, cm, tm, font_dict, font_size):
            s = text.strip()
            if not s:
                return
            # 绝对坐标 = 文本矩阵 × CTM
            x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
            y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
            ys.append(y)
            col = "L" if x < PAGE_W / 2 else "R"
            rows[(col, round(y / 3))].add(x)

        page.extract_text(visitor_text=visitor)
        left = sum(1 for (c, y) in rows if c == "L")
        right = sum(1 for (c, y) in rows if c == "R")
        ymax = max(ys) if ys else 0
        ymin = min(ys) if ys else 0
        print(f"{i:>5} {left:>11} {right:>12} {ymin:>14.0f} {ymax:>10.0f}")


if __name__ == "__main__":
    analyse(sys.argv[1] if len(sys.argv) > 1
            else "build/icwsm/NetworkEvolveLab_ICWSM.pdf")
