"""论文 LaTeX 静态完整性检查（无需安装 TeX 发行版）。

检查项：
  1. 正文（main.tex / appendix.tex）用到的每个 ``\\Macro`` 都在 ``tables/macros.tex`` 中定义；
  2. 每个 ``\\input{...}`` 的目标文件存在；
  3. 每个 ``\\ref{...}`` / ``\\eqref{...}`` 都有对应的 ``\\label{...}``；
  4. 每个 ``\\label{...}`` 都至少被引用一次（未引用标签会提示）；
  5. 每个 ``\\includegraphics{...}`` 引用的图文件真实存在；
  6. 自动生成的表格片段带有「请勿手工编辑」标记（防止被手改后与结果漂移）。

用途：作为 CI 的一道关卡，避免论文引用了不存在的宏 / 图 / 标签。
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
DOCS = [os.path.join(PAPER, "main.tex"),
        os.path.join(PAPER, "appendix.tex"),
        os.path.join(PAPER, "checklist.tex")]
MACROS = os.path.join(PAPER, "tables", "macros.tex")
GENERATED = ["sota_matrix.tex", "sota_matrix_compact.tex", "hypothesis_verdicts.tex",
             "attribution.tex", "dataset_metrics.tex", "llm_arm.tex", "scaling.tex",
             "matched_threshold.tex", "h2_structural.tex", "llm_placebo.tex", "real_stance.tex"]

# LaTeX 内建/其它包提供的命令，不应由本项目定义
BUILTIN = {
    "textbf", "textit", "emph", "texttt", "textsc", "underline", "mathrm", "mathbf",
    "mathcal", "mathbb", "mathit", "operatorname", "left", "right", "frac", "sum",
    "leq", "geq", "geq", "neq", "approx", "times", "cdot", "pm", "mp", "ldots", "dots",
    "leq", "in", "notin", "subset", "subseteq", "cup", "cap", "setminus", "infty",
    "partial", "nabla", "forall", "exists", "neg", "land", "lor", "mapsto", "to",
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow", "Leftrightarrow", "langle",
    "rangle", "lfloor", "rfloor", "sqrt", "log", "exp", "min", "max", "arg", "hat",
    "bar", "vec", "tilde", "dot", "over", "frac", "binom", "begin", "end", "item",
    "section", "subsection", "subsubsection", "paragraph", "label", "ref", "eqref",
    "cite", "citep", "citet", "citeauthor", "bibliography", "bibliographystyle",
    "includegraphics", "input", "include", "documentclass", "usepackage", "newcommand",
    "renewcommand", "providecommand", "textbf", "url", "href", "verb", "footnote",
    "thanks", "author", "title", "date", "maketitle", "abstract", "keywords",
    "newtheorem", "theorem", "lemma", "proof", "remark", "proposition", "qed",
    "hline", "toprule", "midrule", "bottomrule", "multicolumn", "multirow",
    "centering", "small", "footnotesize", "scriptsize", "large", "Large", "huge",
    "textwidth", "linewidth", "columnwidth", "textwidth", "par", "quad", "qquad",
    "leftmargin", "itemsep", "topsep", "parsep", "array", "tabular", "table", "figure",
    "caption", "label", "vspace", "hspace", "clearpage", "newpage", "appendix",
    "newcommand", "fbox", "framebox", "rule", "raisebox", "textbf",
    "bfseries", "itshape", "itshape", "rmfamily", "sffamily", "ttfamily", "slshape",
    "ensuremath", "text", "textrm", "textnormal", "operatorname", "substack",
    "underbrace", "overbrace", "stackrel", "overset", "underset", "quad", "quad",
    "lesssim", "gtrsim", "approx", "coloneqq", "triangleq", "blacktriangleleft",
    "checkmark", "dagger", "S", "P", "E", "LaTeX", "TeX", "IEEEtran",
    # 希腊字母与常用数学符号（LaTeX 内建）
    "alpha", "beta", "gamma", "delta", "Delta", "epsilon", "varepsilon", "zeta",
    "eta", "theta", "Theta", "vartheta", "iota", "kappa", "lambda", "Lambda",
    "mu", "nu", "xi", "pi", "Pi", "rho", "varrho", "sigma", "Sigma", "tau",
    "upsilon", "phi", "varphi", "Phi", "chi", "psi", "Psi", "omega", "Omega",
    "Gamma", "Xi", "Upsilon", "pm", "mp", "times", "div", "ast", "star", "circ",
    "bullet", "oplus", "ominus", "otimes", "perp", "parallel", "angle", "triangle",
    "therefore", "because", "mid", "ll", "gg", "subsetneq", "supsetneq", "doteq",
    "equiv", "sim", "simeq", "cong", "propto", "asymp", "approx", "doteqdot",
    # AAAI / ICWSM 模板与 url.sty 提供的命令
    "UrlFont", "urlstyle", "pdfinfo", "frenchspacing", "pdfpagewidth",
    "pdfpageheight", "affiliations", "emails", "links", "link", "protect",
    "setcounter", "setlength", "newcounter", "step", "refstepcounter",
    # ICWSM paper checklist 的官方作答宏（本项目按 AAAI 规则去掉了颜色）
    "answerYes", "answerNo", "answerNA", "answerTODO",
    # 双盲评审匿名化宏
    "newif", "ifanonymized", "anonymizedtrue", "anonymizedfalse", "else", "fi",
}


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


# --- 数学模式文本污染扫描 -------------------------------------------------
#: 在数学模式里出现这些命令是**故意**的：它们把文本原样排进公式。
MATH_TEXT_SAFE = {
    "text", "textrm", "textnormal", "textbf", "textit", "texttt", "textsc",
    "mathrm", "mathbf", "mathit", "mathbb", "mathcal", "mathsf", "mathtt",
    "boldsymbol", "operatorname", "hbox", "makebox", "fbox", "label",
}

#: 数学模式里合法的多字母算子名（函数名）。
MATH_WORDS = {
    "sin", "cos", "tan", "cot", "sec", "csc", "log", "ln", "lg", "exp",
    "min", "max", "sup", "inf", "arg", "deg", "dim", "det", "ker", "gcd",
    "hom", "lim", "mod", "sgn", "Re", "Im", "Pr", "Tr", "span", "rank",
}

#: 重音命令（Erd\H{o}s / R\'enyi 之类）——数学字体没有这些字形。
ACCENT_CMD = re.compile(r"\\[`'\"^~=.]|\\[uvHckbtr]\b")

#: 非 ASCII 的拉丁重音字母（数学模式里会丢字）。
ACCENTED = re.compile(r"[À-ɏ]")


def strip_comments(text: str) -> str:
    """去掉 LaTeX 注释（未被反斜杠转义的 % 之后的内容）。"""
    out = []
    for line in text.splitlines():
        buf = []
        i = 0
        while i < len(line):
            c = line[i]
            if c == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def _numeric_macros(macros_text: str) -> set[str]:
    """返回宏体是纯数字的宏名（例如 ``\\HOneMaxQ`` -> ``0.500``）。"""
    out = set()
    for m in re.finditer(r"\\newcommand\{\\([^}]+)\}\{(.*)\}", macros_text):
        body = m.group(2).strip()
        if body and re.fullmatch(r"[0-9.+\-eE%\\ ]+", body):
            out.add(m.group(1))
    return out


def scan_math_text_contamination(text: str, numeric_macros=None):
    """返回数学模式里混入了正文的可疑片段列表。

    背景：多余的 ``$`` 会把整句散文吞进数学模式。LaTeX **不会**报错
    （只要 ``$`` 数量成对），只会让该段以 cmmi10 数学斜体排版、丢掉重音
    字符（在 .log 里表现为 "Missing character"），在 PDF 上几乎看不出来。
    这里在静态层面把它抓出来。
    """
    numeric_macros = numeric_macros or set()
    code = strip_comments(text)
    findings = []
    in_math = False
    i = 0
    n = len(code)
    while i < n:
        c = code[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", code[i:])
            if m:
                name = m.group(1)
                if in_math and name in MATH_TEXT_SAFE:
                    # \text{...} 之类：跳过其花括号内容
                    j = i + m.end()
                    while j < n and code[j] in " \n":
                        j += 1
                    if j < n and code[j] == "{":
                        depth = 0
                        while j < n:
                            if code[j] == "{":
                                depth += 1
                            elif code[j] == "}":
                                depth -= 1
                                if depth == 0:
                                    j += 1
                                    break
                            j += 1
                        i = j
                        continue
                if in_math and ACCENT_CMD.match(code[i:]):
                    findings.append(("重音命令", code[i:i + 8].replace("\n", " ")))
                i += m.end()
                continue
            # 反斜杠 + 单字符转义（\%, \$, \{ ...）
            i += 2
            continue
        if c == "$":
            if not in_math and numeric_macros:
                # 进入数学模式：若紧跟在数字宏后面，这个 `$` 是多余的
                j = i - 1
                while j >= 0 and code[j] in " \n":
                    j -= 1
                if j >= 0:
                    prev = None
                    if code[j] == "}":
                        depth = 0
                        k = j
                        while k >= 0:
                            if code[k] == "}":
                                depth += 1
                            elif code[k] == "{":
                                depth -= 1
                                if depth == 0:
                                    break
                            k -= 1
                        inner = code[k + 1:j] if k >= 0 else ""
                        if re.fullmatch(r"[0-9.+\-eE%\\ ]+", inner.strip() or "x"):
                            prev = inner.strip()
                    else:
                        mm = re.search(r"\\([A-Za-z]+)$", code[:j + 1])
                        if mm and mm.group(1) in numeric_macros:
                            prev = mm.group(1)
                    if prev is not None:
                        findings.append(("数字宏后多余的 $", prev))
            in_math = not in_math
            i += 1
            continue
        if in_math:
            if ACCENTED.match(c):
                findings.append(("非 ASCII 重音字母", c))
                i += 1
                continue
            if c.isalpha():
                j = i
                while j < n and code[j].isalpha():
                    j += 1
                word = code[i:j]
                # 前一个字符若是反斜杠（宏名）或后一个字符是反斜杠（命令），
                # 说明这是控制序列的一部分，不是裸单词。
                prev = code[i - 1] if i > 0 else ""
                nxt = code[j] if j < n else ""
                if prev != "\\" and nxt != "\\" and len(word) >= 3 \
                        and word not in MATH_WORDS:
                    findings.append(("数学模式里的裸单词", word))
                i = j
                continue
        i += 1
    return findings


def main():
    problems = []
    warnings = []

    macros = set()
    macro_defs = re.findall(r"\\newcommand\{\\([^}]+)\}", read(MACROS))
    macros.update(macro_defs)

    used = set()
    labels = set()
    refs = set()
    inputs = set()
    graphics = set()

    for doc in DOCS:
        if not os.path.exists(doc):
            problems.append(f"缺少论文源文件: {doc}")
            continue
        text = read(doc)
        used |= set(re.findall(r"\\([A-Za-z]+)\*?\s*\{", text))
        used |= set(re.findall(r"\\([A-Za-z]+)\*?(?![\w{])", text))
        labels |= set(re.findall(r"\\label\{([^}]+)\}", text))
        refs |= set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
        inputs |= set(re.findall(r"\\input\{([^}]+)\}", text))
        graphics |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", text))

    # 1. 未定义宏
    undefined = sorted(m for m in used
                       if m not in macros and m not in BUILTIN and not m.startswith("input"))
    # 过滤掉明显是 latex 结构命令的（多行命令名匹配会有噪声，这里只报告可疑的）
    suspicious = [m for m in undefined if m[0].isupper() and len(m) > 2]
    if suspicious:
        problems.append(f"正文使用了未定义的宏: {suspicious}")

    # 2. \input 目标存在性
    for target in sorted(inputs):
        path = os.path.join(PAPER, target)
        if not os.path.exists(path):
            if not os.path.exists(path + ".tex"):
                problems.append(f"\\input 目标不存在: {target}")

    # 3/4. label / ref 配对
    dangling = sorted(refs - labels)
    if dangling:
        problems.append(f"\\ref 指向不存在的 label: {dangling}")
    orphan = sorted(labels - refs)
    if orphan:
        warnings.append(f"从未被引用的 label（可能是冗余）: {orphan}")

    # 5. 图文件存在性
    for g in sorted(graphics):
        path = os.path.join(PAPER, g)
        if not os.path.exists(path):
            problems.append(f"\\includegraphics 引用的文件不存在: {g}")

    # 6. 自动生成片段的「勿手工编辑」标记
    for name in GENERATED:
        path = os.path.join(PAPER, "tables", name)
        if not os.path.exists(path):
            problems.append(f"缺少自动生成表格片段: tables/{name}")
            continue
        if "AUTO-GENERATED" not in read(path):
            problems.append(f"自动生成片段缺少 'AUTO-GENERATED' 标记: tables/{name}")

    # 7. 引用完整性：每个 \cite 键必须在 references.bib 中存在（防引用幻觉）；
    #    反之，bib 中的每条文献都应至少被引用一次（防死条目 / 遗漏核心方法）
    bib_path = os.path.join(PAPER, "references.bib")
    if not os.path.exists(bib_path):
        problems.append("缺少 paper/references.bib")
    else:
        bib = set(re.findall(r"@\w+\{([^,]+),", read(bib_path)))
        cited = set()
        for doc in DOCS:
            if not os.path.exists(doc):
                continue
            for m in re.findall(r"\\cite[a-zA-Z]*\{([^}]*)\}", read(doc)):
                for k in m.split(","):
                    k = k.strip()
                    if k:
                        cited.add(k)
        missing = sorted(cited - bib)
        if missing:
            problems.append(f"引用键在 references.bib 中不存在（引用幻觉风险）: {missing}")
        uncited = sorted(bib - cited)
        if uncited:
            problems.append(f"references.bib 中有条目从未被引用: {uncited}")
        print(f"[info] 参考文献 {len(bib)} 条，全部被引用：{len(cited)} 条")

    # 8. 结构要件：正式论文必须有 Related Work 与 Conclusion
    main_text = read(DOCS[0]) if os.path.exists(DOCS[0]) else ""
    for needed, hint in (("related work", r"\\section\{[^}]*Related Work"),
                         ("conclusion", r"\\section\{[^}]*(Discussion|Conclusion)")):
        if not re.search(hint, main_text):
            warnings.append(f"正文未见「{needed}」章节（正式投稿通常需要）")

    # 9. 宏名必须是纯字母
    #    LaTeX2e 2021-11-15 内核（Tectonic 发行版所带）在解析
    #    ``\\newcommand{\\Name}`` 时，若 Name 中含数字会把参数计数解析错位，
    #    导致宏体泄漏进导言区并触发 "Missing \begin{document}"。
    #    这类故障只会在真正编译时暴露，因此在此加机器门禁。
    bad_names = sorted(n for n in macro_defs if not n.isalpha())
    if bad_names:
        problems.append(f"宏名含非字母字符（LaTeX 内核会错位解析）: {bad_names}")

    # 10. 自动生成片段里不得出现未转义的 %（% 在 LaTeX 里是注释符，
    #     会把该行剩余内容连同行尾的 \\\\ 一起吃掉，导致表格结构损坏）
    for name in GENERATED + ["macros.tex"]:
        path = os.path.join(PAPER, "tables", name)
        if not os.path.exists(path):
            continue
        for i, line in enumerate(read(path).splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue
            if "%" in line.replace("\\%", ""):
                problems.append(f"tables/{name}:{i} 出现未转义的 % （会吞掉该行剩余内容）")

    # 11. ICWSM 2026 强制要求：
    #     (a) 必须附 Paper Checklist，缺则 desk-reject；
    #     (b) Checklist 里的每一问都必须作答（不得残留 \answerTODO）；
    #     (c) 顺序必须是 正文 -> 参考文献 -> Checklist -> 附录。
    checklist_path = os.path.join(PAPER, "checklist.tex")
    if not os.path.exists(checklist_path):
        problems.append("缺少 ICWSM 强制要求的 paper/checklist.tex（缺则 desk-reject）")
    else:
        cl = read(checklist_path)
        if "\\answerTODO" in cl:
            n = cl.count("\\answerTODO")
            problems.append(f"checklist.tex 仍有 {n} 处未作答的 \\answerTODO（必须全部作答）")
        for macro in ("\\answerYes", "\\answerNo", "\\answerNA"):
            if macro not in cl:
                problems.append(f"checklist.tex 未使用官方作答宏 {macro}")
        if "\\section{Paper Checklist}" not in cl:
            problems.append("checklist.tex 缺少 'Paper Checklist' 章节标题")
    main_text = read(DOCS[0]) if os.path.exists(DOCS[0]) else ""
    order = [main_text.find("\\bibliography{references}"),
             main_text.find("\\input{checklist}"),
             main_text.find("\\appendix")]
    if -1 in order or not (order[0] < order[1] < order[2]):
        problems.append(
            "ICWSM 要求的章节顺序被破坏：必须为 参考文献 -> Paper Checklist -> 附录 "
            f"(当前位置 {order})")

    # 12. LaTeX 环境必须配对。
    #     历史上出现过「按段落替换压缩正文」把 \end{remark} 一起吞掉的情况，
    #     编译会报 "ended by \end{document}"。这里做成机器门禁，
    #     保证任何后续的正文压缩都不会再悄悄破坏环境结构。
    envs = ["figure", "figure*", "table", "table*", "minipage", "enumerate",
            "itemize", "equation", "equation*", "aligned", "cases", "abstract",
            "links", "proposition", "theorem", "lemma", "definition", "remark",
            "proof", "quote", "tabular"]
    for doc in DOCS:
        if not os.path.exists(doc):
            continue
        body = read(doc)
        # 去掉注释行后再统计（注释里可能有 \begin/\end 字样）
        code = "\n".join(ln for ln in body.splitlines()
                         if not ln.lstrip().startswith("%"))
        stack: list[str] = []
        bad = None
        for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", code):
            kind, name = m.group(1), m.group(2)
            if name not in envs:
                continue
            if kind == "begin":
                stack.append(name)
            else:
                if not stack:
                    bad = f"多余的 \\end{{{name}}}（行 {code[:m.start()].count(chr(10)) + 1}）"
                    break
                open_name = stack.pop()
                if open_name != name:
                    bad = (f"环境嵌套错位：\\begin{{{open_name}}} 被 "
                           f"\\end{{{name}}} 关闭（行 {code[:m.start()].count(chr(10)) + 1}）")
                    break
        if bad is None and stack:
            bad = f"未闭合的环境：{', '.join(reversed(stack))}"
        if bad:
            problems.append(f"{os.path.basename(doc)} 环境不配对：{bad}")

    # 13. 数学模式里不得混入正文散文。
    #     多余的 `$` 会让整句被排成数学斜体并丢掉重音字符，
    #     LaTeX 不报错、PDF 上几乎看不出来，只有 .log 里的
    #     "Missing character" 能事后发现。这里在静态层面提前拦下。
    scan_targets = list(DOCS)
    built = os.path.join(ROOT, "build", "icwsm", "NetworkEvolveLab_ICWSM.tex")
    if os.path.exists(built):
        scan_targets.append(built)
    numeric_macros = _numeric_macros(read(MACROS)) if os.path.exists(MACROS) else set()
    for doc in scan_targets:
        if not os.path.exists(doc):
            continue
        for kind, frag in scan_math_text_contamination(read(doc), numeric_macros):
            problems.append(
                f"{os.path.basename(doc)}: 数学模式里混入正文（{kind}: {frag!r}）"
                "——检查是否有多余的 `$`")

    # 14. 参考文献裁剪清单（docs/dropped_references.txt）里的键不得再出现在
    #     references.bib 中——清单是「因页数预算被裁掉」的记录，若某条被重新
    #     引入而正文引用没同步，bib 与正文就会再次不一致。
    dropped_path = os.path.join(ROOT, "docs", "dropped_references.txt")
    if os.path.exists(dropped_path):
        dropped = {ln.strip() for ln in read(dropped_path).splitlines()
                   if ln.strip() and not ln.startswith("#")}
        bib_path2 = os.path.join(PAPER, "references.bib")
        if os.path.exists(bib_path2):
            cur = set(re.findall(r"@\w+\{([^,]+),", read(bib_path2)))
            reintro = sorted(dropped & cur)
            if reintro:
                problems.append(
                    f"以下条目已列入裁剪清单，却仍出现在 references.bib 中"
                    f"（若要恢复引用，需同步正文 \\citep）: {reintro}")
            print(f"[info] 参考文献裁剪清单 {len(dropped)} 条，当前 bib {len(cur)} 条")

    for w in warnings:
        print(f"[warn] {w}")
    if problems:
        print("\n[FAIL] 论文静态完整性检查未通过：")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print(f"[PASS] 论文静态完整性检查通过"
          f"（宏 {len(macros)} 个 / label {len(labels)} 个 / 引用 {len(refs)} 个 / "
          f"图 {len(graphics)} 个 / 输入 {len(inputs)} 个）")


if __name__ == "__main__":
    main()
