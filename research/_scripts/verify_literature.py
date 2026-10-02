"""Verify every candidate citation against OpenAlex and cache metadata + abstracts.

Run: python research/_scripts/verify_literature.py
Outputs: research/_scripts/openalex_cache.json (metadata, abstracts, verification status)
"""
import json, time, urllib.parse, urllib.request
from pathlib import Path

DOIS = [
 "10.1097/00001888-200308000-00003","10.1001/archinte.165.13.1493","10.1136/bmjqs-2012-001615",
 "10.1136/bmjqs-2016-005401","10.23970/ahrqepccer258","10.1136/amiajnl-2011-000089",
 "10.1001/jamainternmed.2023.2366","10.48550/arXiv.2102.09692","10.1145/3579605",
 "10.48550/arXiv.2305.14325","10.18653/v1/2024.emnlp-main.992","10.48550/arXiv.2402.06782",
 "10.18653/v1/2024.findings-acl.33","10.1109/MIPR67560.2025.00078","10.1016/j.media.2023.103042",
 "10.3389/fped.2021.662183","10.1016/j.dib.2019.104788","10.1016/j.jpeds.2008.01.033",
 "10.1053/jpsu.2002.32893","10.48550/arXiv.1705.07874","10.1016/S2589-7500(21)00208-9",
 "10.1016/j.inffus.2019.12.012","10.1016/j.media.2022.102470","10.48550/arXiv.1706.04599",
 "10.1186/s12916-019-1466-7","10.1145/1102351.1102430","10.1136/bmjqs-2018-008370",
 "10.1038/s41591-018-0300-7","10.1038/s41586-023-06291-2","10.48550/arXiv.1711.05225",
 "10.1136/bmj-2023-078378","10.1136/bmj.g7594","10.1038/s41591-022-01772-9",
 "10.1038/s41591-020-1034-x","10.1038/s41591-020-1037-7","10.1038/s41591-025-03953-8",
 "10.1136/bmjopen-2016-012799","10.1038/s41591-024-03425-5","10.1097/01.pcc.0000149131.72248.e6",
 "10.1001/jama.2024.0179","10.1001/jama.2024.0196","10.1161/CIR.0000000000000484",
 "10.1016/S0196-0644(86)80993-3","10.1542/peds.113.1.24",
]

def abstract(inv):
    if not inv:
        return None
    pos = {}
    for w, idxs in inv.items():
        for i in idxs:
            pos[i] = w
    return " ".join(pos[i] for i in sorted(pos))

cache = {}
for doi in DOIS:
    url = "https://api.openalex.org/works/doi:" + urllib.parse.quote(doi) + "?mailto=research@example.org"
    try:
        req = urllib.request.Request(url, headers={"User-Agent":"lit-review/1.0"})
        with urllib.request.urlopen(req, timeout=25) as r:
            w = json.load(r)
        cache[doi] = {
            "verified": True,
            "title": w.get("display_name"),
            "year": w.get("publication_year"),
            "venue": ((w.get("primary_location") or {}).get("source") or {}).get("display_name"),
            "authors": [a["author"]["display_name"] for a in w.get("authorships", [])],
            "oa": (w.get("open_access") or {}).get("oa_status"),
            "abstract": abstract(w.get("abstract_inverted_index")),
            "cited_by": w.get("cited_by_count"),
        }
        print("OK ", doi, "|", cache[doi]["title"][:70])
    except Exception as e:
        cache[doi] = {"verified": False, "error": str(e)}
        print("FAIL", doi, "|", e)
    time.sleep(0.4)

Path("research/_scripts/openalex_cache.json").write_text(
    json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
print("cached", len(cache))
