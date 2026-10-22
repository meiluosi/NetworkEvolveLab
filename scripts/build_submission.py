#!/usr/bin/env python3
"""构建 ICWSM / AAAI Press 单文件投稿包。

为什么需要这一步
----------------
AAAI Press（ICWSM 的出版方）的格式要求明确写着：

    "Your LaTeX source file (excluding references) must consist of a **single**
     file (use of the ``input'' command is not allowed)."

而本项目的论文源是**模块化**的（`paper/main.tex` 通过 `\\input{}` 引入
`paper/appendix.tex` 与 `paper/tables/*.tex`）。这种模块化不是风格问题，而是
**诚信机制**：所有定量表格都由 `scripts/generate_paper_tables.py` 从冻结的结果
JSON 机器生成，论文正文只引用宏，因此数字不可能与实验漂移。把它拍平成单文件
正是为了同时满足两者：

  * 仓库里保留模块化源 + `--check` 门禁 → 数字仍然机器生成、不可漂移；
  * 投稿包里是拍平后的单文件 → 满足 AAAI 的格式要求。

用法
----
    python scripts/build_submission.py            # 生成 build/icwsm/
    python scripts/build_submission.py --check    # 校验拍平结果与源文件同步（CI 用）

拍平规则：递归展开 `\\input{...}`，用显式标记注释包住每一块被展开的内容，
便于人工核对；不做任何其它文本改写（因此拍平前后的排版结果逐像素一致）。
"""
from __future__ import annotations

import argparse
import filecmp
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
BUILD = os.path.join(ROOT, "build", "icwsm")
OUT_NAME = "NetworkEvolveLab_ICWSM.tex"

INPUT_RE = re.compile(r"^[ \t]*\\input\{([^}]+)\}[ \t]*$", re.MULTILINE)

MARK_BEGIN = "%% ===== BEGIN INLINED: {path} ====="
MARK_END = "%% ===== END INLINED: {path} ====="

#: 参考文献里超过该作者数的条目，投稿包中按 AAAI 惯例显示为 "et al."
#: （aaai25.bst 原生支持 "and others"）。
#: 为什么只在**投稿包**里截断：仓库里的 paper/references.bib 保留由
#: Crossref/arXiv 逐条机器核实的**完整**作者列表（那是元数据的事实来源），
#: 而投稿 PDF 里的 53 人作者列表会吃掉约一整页。完整列表可由条目中的
#: DOI 直接解析得到，读者不会丢失任何信息。
MAX_AUTHORS_IN_SUBMISSION = 6


def _fold(m: "re.Match[str]", keep: int, truncated: list[str]) -> str:
    """折叠单个 bib 条目的作者列表。"""
    atype, key, pre, authors, post = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    parts = authors.split(" and ")
    if len(parts) <= keep + 1:
        return m.group(0)
    truncated.append(key)
    return f"@{atype}{{{key},{pre}{' and '.join(parts[:keep])} and others{post}\n}}"


def _truncate_long_authors(bib_text: str) -> tuple[str, list[str]]:
    """把超长作者列表折叠成 "前 N 位 and others"（AAAI 的 et al. 惯例）。"""
    truncated: list[str] = []
    out = re.sub(r"@(\w+)\{([^,]+),(.*?author\s*=\s*\{)(.*?)(\},.*?)\n\}",
                 lambda m: _fold(m, MAX_AUTHORS_IN_SUBMISSION, truncated),
                 bib_text, flags=re.S)
    return out, truncated


def _resolve(base_dir: str, rel: str) -> str:
    """LaTeX 的 \\input 会自动补 .tex 扩展名，这里照做。"""
    names = [rel] if rel.endswith(".tex") else [rel, rel + ".tex"]
    for name in names:
        for base in (base_dir, PAPER):
            cand = os.path.join(base, name)
            if os.path.exists(cand):
                return cand
    raise FileNotFoundError(f"找不到被 \\input 的文件：{rel}（相对 {base_dir}）")


def flatten(path: str, seen: list[str]) -> str:
    """递归展开 \\input{}，返回单文件正文。"""
    path = os.path.abspath(path)
    if path in seen:
        raise ValueError(f"检测到循环 \\input：{path}")
    seen = seen + [path]
    base = os.path.dirname(path)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()

    def repl(m: "re.Match[str]") -> str:
        rel = m.group(1).strip()
        target = _resolve(base, rel)
        inner = flatten(target, seen)
        return (f"{MARK_BEGIN.format(path=rel)}\n"
                f"{inner}\n"
                f"{MARK_END.format(path=rel)}")

    return INPUT_RE.sub(repl, text)


def build(out_dir: str = BUILD) -> str:
    os.makedirs(out_dir, exist_ok=True)
    flat = flatten(os.path.join(PAPER, "main.tex"), [])
    out_tex = os.path.join(out_dir, OUT_NAME)
    with open(out_tex, "w", encoding="utf-8") as fh:
        fh.write(flat)

    # 参考文献（AAAI 允许 .bib 单独提交）
    # 超长作者列表在**投稿包**里折叠为 "et al."；仓库里的
    # paper/references.bib 保留 Crossref/arXiv 机器核实的完整作者列表。
    with open(os.path.join(PAPER, "references.bib"), encoding="utf-8") as fh:
        bib_src = fh.read()
    bib_out, truncated = _truncate_long_authors(bib_src)
    with open(os.path.join(out_dir, "references.bib"), "w", encoding="utf-8") as fh:
        fh.write(bib_out)
    if truncated:
        print(f"[info] 投稿包中 {len(truncated)} 条超长作者列表按 AAAI 惯例折叠为 et al.: "
              f"{', '.join(truncated)}")
    # 官方样式文件（逐字复制，不修改）
    for fn in ("aaai25.sty", "aaai25.bst"):
        shutil.copy2(os.path.join(PAPER, "venue", fn),
                     os.path.join(out_dir, fn))
    # 图片
    src_fig = os.path.join(PAPER, "figures")
    dst_fig = os.path.join(out_dir, "figures")
    if os.path.isdir(dst_fig):
        shutil.rmtree(dst_fig)
    shutil.copytree(src_fig, dst_fig)
    return out_tex


def check(out_dir: str = BUILD) -> int:
    """校验：拍平结果与源文件同步，且投稿包不残留 \\input。"""
    out_tex = os.path.join(out_dir, OUT_NAME)
    if not os.path.exists(out_tex):
        print("投稿包不存在，请先运行：python scripts/build_submission.py")
        return 1
    with open(out_tex, encoding="utf-8") as fh:
        current = fh.read()
    expected = flatten(os.path.join(PAPER, "main.tex"), [])
    problems = []
    if current != expected:
        problems.append("拍平文件与源文件不同步（请重新运行 build_submission.py）")
    residual = [m.group(1) for m in INPUT_RE.finditer(current)]
    if residual:
        problems.append(f"投稿包里仍残留 \\input：{residual}")
    for fn in ("references.bib", "aaai25.sty", "aaai25.bst"):
        if not os.path.exists(os.path.join(out_dir, fn)):
            problems.append(f"投稿包缺少 {fn}")
    # 投稿包里的 references.bib 必须与「源文件 + et al. 折叠」的确定性变换一致
    bib_expect_path = os.path.join(out_dir, "references.bib")
    if os.path.exists(bib_expect_path):
        with open(os.path.join(PAPER, "references.bib"), encoding="utf-8") as fh:
            bib_src = fh.read()
        bib_expect, _ = _truncate_long_authors(bib_src)
        with open(bib_expect_path, encoding="utf-8") as fh:
            bib_cur = fh.read()
        if bib_cur != bib_expect:
            problems.append("投稿包 references.bib 与源文件不同步（请重新运行 build_submission.py）")
    if not os.path.isdir(os.path.join(out_dir, "figures")):
        problems.append("投稿包缺少 figures/ 目录")
    # AAAI 禁用项（只看真正的命令，注释行不算）
    forbidden = ["usepackage{hyperref}", "usepackage[margin", "usepackage{microtype}",
                 "usepackage{xcolor}", "usepackage{geometry}", "bibliographystyle{plainnat}",
                 "pagestyle{", "vspace{-", "vskip{-", "clearpage", "newpage",
                 "pagebreak", "addtolength", "baselinestretch", "linespread"]
    code_lines = [ln for ln in current.splitlines()
                  if not ln.lstrip().startswith("%")]
    code = "\n".join(code_lines)
    for pat in forbidden:
        if pat in code:
            problems.append(f"投稿包含 AAAI 禁用项：{pat}")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK  单文件投稿包自检通过：{out_tex}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the ICWSM/AAAI single-file submission")
    ap.add_argument("--check", action="store_true",
                    help="only verify that the submission package is in sync")
    ap.add_argument("--out-dir", default=BUILD)
    args = ap.parse_args()
    if args.check:
        return check(args.out_dir)
    out = build(args.out_dir)
    print(f"[done] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
