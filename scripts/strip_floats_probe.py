#!/usr/bin/env python3
"""诊断脚本：把正文里的浮动体全部剥掉，量出「纯文字」的页数。

用途
----
AAAI 双栏排版里，`figure*` / `table*`（跨栏浮动体）只能在页顶放置；当浮动队列里
积压了一个跨栏浮动体时，LaTeX 的输出例程会把当前页提前结束，这会让半页白白留白。
本脚本剥掉所有浮动体后编译一次，得到「纯文字需要多少页」，从而把总页数拆成
「文字页数」与「浮动体造成的额外页数」两部分，用来判断该砍正文还是该重排浮动体。

用法：
    python scripts/strip_floats_probe.py build/icwsm/NetworkEvolveLab_ICWSM.tex
"""
from __future__ import annotations

import os
import re
import sys

FLOAT_ENVS = ("figure*", "figure", "table*", "table")


def strip_floats(text: str) -> str:
    """删除所有浮动体环境（含其内容），保留 \\label 以免交叉引用断裂。"""
    out = text
    for env in FLOAT_ENVS:
        pattern = re.compile(
            r"\\begin\{" + re.escape(env) + r"\}.*?\\end\{" + re.escape(env) + r"\}",
            re.S,
        )
        while True:
            m = pattern.search(out)
            if not m:
                break
            block = m.group(0)
            labels = re.findall(r"\\label\{([^}]+)\}", block)
            keep = "".join(f"\\label{{{lab}}}" for lab in labels)
            out = out[:m.start()] + keep + out[m.end():]
    return out


def main() -> int:
    src = sys.argv[1] if len(sys.argv) > 1 else "build/icwsm/NetworkEvolveLab_ICWSM.tex"
    if not os.path.exists(src):
        print(f"找不到 {src}")
        return 1
    with open(src, encoding="utf-8") as fh:
        text = fh.read()
    # 参考文献与附录都不剥（只量正文）；这里对整份文档剥，正文页数是上界。
    stripped = strip_floats(text)
    out = os.path.join(os.path.dirname(src), "_probe_nofloats.tex")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(stripped)
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
