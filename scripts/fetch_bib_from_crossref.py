"""按 DOI 从 Crossref 拉取权威元数据，生成 references.bib 条目。

为什么用脚本而不是手写：
  参考文献的**作者列表**极易被 LLM 凭记忆补全而产生「引用幻觉」。
  这里对每条 DOI 逐条向 Crossref 解析，拿到真实的作者、标题、容器、年份、
  卷期页码，并保留 DOI；解析失败的条目会被丢弃并报告，绝不臆造。

用法：python scripts/fetch_bib_from_crossref.py > /tmp/newbib.bib
"""
import json
import sys
import time
import urllib.request
import urllib.parse

# bibkey -> (doi, 条目类型提示)
ENTRIES = [
    ("argyle2023outofone", "10.1017/pan.2023.2"),
    ("bisbee2024synthetic", "10.1017/pan.2024.5"),
    ("aher2023using", "10.48550/arXiv.2208.10264"),
    ("horton2023largelanguage", "10.3386/w31122"),
    ("sclar2024quantifying", "10.48550/arXiv.2310.11324"),
    ("wang2024largelanguage", "10.18653/v1/2024.acl-long.511"),
    ("zheng2023judging", "10.48550/arXiv.2306.05685"),
    ("cummins2026threat", "10.1177/25152459261461505"),
    ("ziems2024can", "10.1162/coli_a_00502"),
    ("gao2023s3", "10.48550/arXiv.2307.14984"),
    ("vezhnevets2023concordia", "10.48550/arXiv.2312.03664"),
    ("oas2024oasis", "10.48550/arXiv.2411.11581"),
    ("duskin2024echo", "10.1145/3614419.3643996"),
    ("ge2020understanding", "10.1145/3397271.3401431"),
    ("fan2019deepsocial", "10.1145/3298689.3347011"),
    ("huszar2022algorithmic", "10.1073/pnas.2025334119"),
    ("xfeed2026political", "10.1038/s41586-026-10098-2"),
    ("gross2008adaptive", "10.1098/rsif.2007.1229"),
    ("bornholdt2000topological", "10.1103/physrevlett.84.6114"),
    ("bornholdt2002smallworld", "10.1103/physrevlett.88.128701"),
    ("kozma2008consensus", "10.1103/physreve.77.016102"),
    ("holme2012temporal", "10.1016/j.physrep.2012.03.001"),
    ("snijders2010introduction", "10.1016/j.socnet.2009.02.004"),
    ("baldassarri2007dynamics", "10.1177/000312240707200507"),
    ("centola2007complex", "10.1086/521848"),
    ("granovetter1973strength", "10.1086/225469"),
    ("burt1992structural", "10.4159/9780674029095"),
    ("leskovec2005graphs", "10.1145/1081870.1081893"),
    ("guess2023feed", "10.1126/science.abp9364"),
    ("nyhan2023likeminded", "10.1038/s41586-023-06297-w"),
    ("levy2021socialmedia", "10.1257/aer.20191777"),
    ("brady2021social", "10.1126/sciadv.abe5641"),
    ("conover2011political", "10.1609/icwsm.v5i1.14126"),
    ("azzimonti2018social", "10.3386/w24462"),
    ("deffuant2000mixing", "10.1142/S0219525900000078"),
    ("weisbuch2002meet", "10.1002/cplx.10031"),
    ("sobkowicz2018cognitive", "10.18564/jasss.3867"),
    ("tian2023dynamics", "10.1016/j.arcontrol.2023.04.001"),
    ("lu2022asymmetry", "10.3390/sym14122499"),
    ("grimm2020odd", "10.18564/jasss.4259"),
    ("sandve2013ten", "10.1371/journal.pcbi.1003285"),
    ("wu2015general", "10.1109/wsc.2015.7408240"),
    ("epstein1996growing", "10.7551/mitpress/3374.001.0001"),
    ("fortunato2007resolution", "10.1073/pnas.0605965104"),
    ("peixoto2014hierarchical", "10.1103/physrevx.4.011047"),
    ("traag2019leiden", "10.1038/s41598-019-41695-z"),
    ("rosvall2008maps", "10.1073/pnas.0706851105"),
    ("cantoncorrea2026crowdsourced", "10.15581/003.39.1.032"),
]

# Crossref 的 type → BibTeX entry type
TYPE_MAP = {
    "journal-article": "article",
    "proceedings-article": "inproceedings",
    "book": "book",
    "posted-content": "misc",
    "report": "misc",
    "dataset": "misc",
    "monograph": "book",
    "reference-entry": "inproceedings",
    "component": "misc",
}


def crossref(doi):
    url = ("https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
           + "?mailto=meiluosi@users.noreply.github.com")
    req = urllib.request.Request(url, headers={"User-Agent": "NetworkEvolveLab/1.0 (bib fetch)"})
    with urllib.request.urlopen(req, timeout=40) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]


def clean(s):
    if not s:
        return ""
    s = str(s)
    s = s.replace("\\", "\\textbackslash ").replace("&", "\\&").replace("%", "\\%")
    s = s.replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
    s = s.replace("{", "").replace("}", "")
    return " ".join(s.split())


def authors_field(authors):
    if not authors:
        return None
    names = []
    for a in authors:
        fam = clean(a.get("family", ""))
        giv = clean(a.get("given", ""))
        if fam and giv:
            names.append(f"{fam}, {giv}")
        elif fam:
            names.append(fam)
        elif a.get("name"):
            names.append(clean(a["name"]))
    return " and ".join(names) if names else None


def pages_of(m):
    p = m.get("page", "")
    return clean(p).replace("--", "--") if p else None


def main():
    # 允许命令行临时追加 "key=doi"（用于补零星条目），不修改本文件
    extra = []
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, doi = arg.split("=", 1)
            extra.append((k.strip(), doi.strip()))
    out = []
    failed = []
    for key, doi in list(ENTRIES) + extra:
        try:
            m = crossref(doi)
        except Exception as exc:
            failed.append((key, doi, f"{type(exc).__name__}: {exc}"))
            print(f"[skip] {key}  {doi}  ({type(exc).__name__})", file=sys.stderr)
            continue
        etype = TYPE_MAP.get(m.get("type", ""), "misc")
        fields = []
        title = clean((m.get("title") or [""])[0])
        if title:
            fields.append(("title", title))
        au = authors_field(m.get("author", []))
        if au:
            fields.append(("author", au))
        if etype == "article":
            j = clean((m.get("container-title") or [""])[0])
            if j:
                fields.append(("journal", j))
        elif etype == "inproceedings":
            b = clean((m.get("container-title") or [""])[0])
            if b:
                fields.append(("booktitle", b))
        elif etype == "book":
            p = clean(m.get("publisher", ""))
            if p:
                fields.append(("publisher", p))
        year = None
        for f in ("published-print", "published-online", "issued", "created"):
            parts = (m.get(f) or {}).get("date-parts")
            if parts and parts[0] and parts[0][0]:
                year = parts[0][0]
                break
        if year:
            fields.append(("year", str(year)))
        vol = clean(m.get("volume", ""))
        if vol:
            fields.append(("volume", vol))
        iss = clean(m.get("issue", ""))
        if iss:
            fields.append(("number", iss))
        pg = pages_of(m)
        if pg:
            fields.append(("pages", pg))
        pub = clean(m.get("publisher", ""))
        if pub and etype != "book":
            fields.append(("publisher", pub))
        fields.append(("doi", doi))
        url = m.get("URL") or ("https://doi.org/" + doi)
        fields.append(("url", url))

        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields)
        out.append(f"@{etype}{{{key},\n{body}\n}}")
        print(f"[ok] {key}", file=sys.stderr)
        time.sleep(0.15)

    print("\n\n".join(out))
    print(f"\n% fetched={len(out)} failed={len(failed)}", file=sys.stderr)
    for f in failed:
        print(f"% FAILED {f}", file=sys.stderr)


if __name__ == "__main__":
    main()
