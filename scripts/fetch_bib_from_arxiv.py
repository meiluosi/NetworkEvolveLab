"""从 arXiv API 取回 6 篇 arXiv 预印本的权威元数据（作者/标题/年份）。"""
import re
import sys
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

IDS = {
    "aher2023using": "2208.10264",
    "sclar2024quantifying": "2310.11324",
    "zheng2023judging": "2306.05685",
    "gao2023s3": "2307.14984",
    "vezhnevets2023concordia": "2312.03664",
    "oas2024oasis": "2411.11581",
}
NS = {"a": "http://www.w3.org/2005/Atom"}


def clean(s):
    if not s:
        return ""
    s = str(s)
    s = s.replace("\\", "\\textbackslash ").replace("&", "\\&").replace("%", "\\%")
    s = s.replace("_", "\\_").replace("#", "\\#").replace("$", "\\$")
    s = s.replace("{", "").replace("}", "")
    return " ".join(s.split())


def main():
    ids = ",".join(IDS.values())
    url = ("https://export.arxiv.org/api/query?id_list=" + ids +
           "&max_results=50")
    req = urllib.request.Request(url, headers={"User-Agent": "NetworkEvolveLab/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        xml = resp.read()
    root = ET.fromstring(xml)
    by_id = {}
    for e in root.findall("a:entry", NS):
        eid = e.find("a:id", NS).text
        m = re.search(r"abs/([\w.\-/]+)$", eid)
        if not m:
            continue
        base = m.group(1).split("v")[0]
        by_id[base] = e
    out = []
    for key, aid in IDS.items():
        e = by_id.get(aid)
        if e is None:
            print(f"[skip] {key}: arXiv 未返回 {aid}", file=sys.stderr)
            continue
        title = clean(e.find("a:title", NS).text)
        authors = " and ".join(
            f"{clean(a.find('a:name', NS).text)}" for a in e.findall("a:author", NS))
        published = e.find("a:published", NS).text
        year = published[:4]
        out.append(
            "@misc{" + key + ",\n"
            f"  title = {{{title}}},\n"
            f"  author = {{{authors}}},\n"
            f"  year = {{{year}}},\n"
            "  eprint = {" + aid + "},\n"
            "  archivePrefix = {arXiv},\n"
            "  primaryClass = {" + (e.find("{http://arxiv.org/schemas/atom}primary_category").get("term") if e.find("{http://arxiv.org/schemas/atom}primary_category") is not None else "") + "},\n"
            "  url = {https://arxiv.org/abs/" + aid + "},\n"
            "  doi = {10.48550/arXiv." + aid + "}\n"
            "}")
        print(f"[ok] {key}", file=sys.stderr)
    print("\n\n".join(out))


if __name__ == "__main__":
    main()
