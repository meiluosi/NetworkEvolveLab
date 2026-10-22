#!/usr/bin/env python3
"""统计 paper/main.tex 各章节的词数（用于压缩排版时定位冗余段落）。"""
import re
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex"
text = open(path, encoding="utf-8").read()

body = text.split("\\begin{document}")[1]
# 去掉附录/参考文献之后的注释块不影响统计，只去掉注释行
body = "\n".join(ln for ln in body.splitlines() if not ln.lstrip().startswith("%"))

parts = re.split(r"\\section\*?\{([^}]*)\}", body)
rows = []
for i in range(1, len(parts), 2):
    name = parts[i]
    chunk = parts[i + 1]
    # 去掉浮动体（图表内容不计入散文词数）
    chunk_nofloat = re.sub(r"\\begin\{(figure\*?|table\*?)\}.*?\\end\{\1\}", " ", chunk, flags=re.S)
    rows.append((name, len(chunk.split()), len(chunk_nofloat.split())))

print(f"{'section':46s} {'all':>7s} {'prose':>7s}")
tot_all = tot_prose = 0
for name, a, p in rows:
    print(f"{name[:46]:46s} {a:7d} {p:7d}")
    tot_all += a
    tot_prose += p
print(f"{'TOTAL':46s} {tot_all:7d} {tot_prose:7d}")
