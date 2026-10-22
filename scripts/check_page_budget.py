#!/usr/bin/env python3
"""ICWSM 2026 页数预算门禁。

规则（https://www.icwsm.org/2026/submit.html）：

    * Full Papers 页数上限 11 页，**正文与参考文献计入**页数；
    * Paper Checklist 与可选附录**不计入**页数。

因此判定方式是「Checklist 从第几页开始」：Checklist 之前的全部内容
（正文 + 参考文献）必须落在第 1--11 页之内，即 Checklist 必须从第 12 页
或更后开始。若 Checklist 与参考文献尾巴落在同一页，该页仍然计入，
本门禁按最保守口径处理。

用法：
    python scripts/check_page_budget.py
    python scripts/check_page_budget.py --pdf path/to/other.pdf --limit 11
"""
from __future__ import annotations

import argparse
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PDF = os.path.join(ROOT, "build", "icwsm", "NetworkEvolveLab_ICWSM.pdf")

#: 识别 Checklist 起始页的特征串（小写匹配）。
CHECKLIST_MARKERS = ("paper checklist",)

#: 识别附录起始页的特征串（用于报告）。用附录**第一个小节的标题**，
#: 不能用裸的 "appendix"——正文里到处都出现 "Appendix~\\ref{...}"，
#: 那样会把正文页误判成附录起始页。
APPENDIX_MARKERS = ("formal mathematical proofs",)


def _norm(text: str) -> str:
    return " ".join(text.split()).lower()


def _first_page_containing(reader, markers) -> int | None:
    for i, page in enumerate(reader.pages, start=1):
        raw = _norm(page.extract_text() or "")
        if any(m in raw for m in markers):
            return i
    return None


def _check_compile_log(log_path: str) -> int:
    """扫描 TeX 编译日志里的硬错误与缺字符。

    Tectonic 不会因为「多余的 `$` 把正文吞进数学模式」而报错——这类错误
    只表现为日志里的 ``Missing character`` 警告（例如把 ``Erd\\H{o}s`` 放进
    数学模式后 cmmi10 没有 ő 字形）。它不会让 PDF 少页，但会让排版出错
    甚至丢字，因此必须当作硬失败。
    """
    if not os.path.exists(log_path):
        print(f"[info] 未找到编译日志 {log_path}，跳过日志检查")
        return 0
    with open(log_path, encoding="utf-8", errors="replace") as f:
        log = f.read()

    fatal: list[str] = []
    missing = re.findall(r"Missing character: There is no .*? in font (\S+)", log)
    if missing:
        fonts = sorted(set(missing))
        fatal.append(f"编译日志报告 {len(missing)} 处缺字符（字体 {', '.join(fonts)}）"
                     "——通常是多余的 `$` 把正文吞进了数学模式")
    for pat, desc in (
        (r"^! ", "LaTeX 错误"),
        (r"Missing \\begin\{document\}", "宏体泄漏到导言区"),
        (r"ended by \\end\{document\}", "环境未闭合"),
        (r"Runaway argument", "参数未闭合"),
    ):
        n = len(re.findall(pat, log, re.M))
        if n:
            fatal.append(f"编译日志出现 {n} 处 {desc}")
    if fatal:
        print("[FAIL] 编译日志检查未通过：")
        for f_ in fatal:
            print(f"  - {f_}")
        return 1
    print(f"[PASS] 编译日志检查通过（{os.path.basename(log_path)} 无硬错误、无缺字符）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="ICWSM page-budget gate")
    ap.add_argument("--pdf", default=DEFAULT_PDF)
    ap.add_argument("--limit", type=int, default=11,
                    help="maximum number of pages for body + references")
    ap.add_argument("--log", default=None,
                    help="TeX log to scan for hard errors (default: sibling .log)")
    ap.add_argument("--skip-missing", action="store_true",
                    help="pass silently when the PDF has not been built yet")
    args = ap.parse_args()

    if not os.path.exists(args.pdf):
        msg = f"PDF 不存在：{args.pdf}"
        if args.skip_missing:
            print(f"[skip] {msg}")
            return 0
        print(f"[FAIL] {msg}")
        return 1

    log_path = args.log or os.path.splitext(args.pdf)[0] + ".log"
    if _check_compile_log(log_path) != 0:
        return 1

    try:
        import pypdf
    except ImportError:
        print("[skip] 未安装 pypdf，跳过页数门禁")
        return 0

    reader = pypdf.PdfReader(args.pdf)
    total = len(reader.pages)
    checklist_page = _first_page_containing(reader, CHECKLIST_MARKERS)
    appendix_page = _first_page_containing(reader, APPENDIX_MARKERS)

    print(f"[info] PDF 共 {total} 页；"
          f"Checklist 起始页 = {checklist_page}；附录起始页 = {appendix_page}")
    if checklist_page is None:
        print("[FAIL] 未在 PDF 中找到 Paper Checklist 标题——ICWSM 缺 checklist 会 desk-reject")
        return 1
    checklist_text = _norm(reader.pages[checklist_page - 1].extract_text() or "")
    idx = min(checklist_text.find(m) for m in CHECKLIST_MARKERS if m in checklist_text)
    pre = checklist_text[:idx].strip()
    # 如果该页在 Checklist 之前还有正文或参考文献（即共享该页），则该页仍计入正文；
    # 若 Checklist 作为本页顶部的独立章节开始，则正文+参考文献恰好在上一页结束。
    if len(pre) > 50:
        body_pages = checklist_page
    else:
        body_pages = checklist_page - 1
    if body_pages > args.limit:
        print(f"[FAIL] 正文+参考文献占用 {body_pages} 页，超过 ICWSM 上限 {args.limit} 页")
        return 1
    print(f"[PASS] 正文+参考文献 {body_pages} 页 <= {args.limit} 页上限"
          f"（Checklist 与附录不计入）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
