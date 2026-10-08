"""Build kb/rules/registry.json — CIB&RC legality/registration layer for rule chemicals.

Sources (GOI, free, official — downloaded to media/cibrc/, text in kb/raw/cibrc/):
  formulations_registered_31.03.2026  — registered formulations (Insecticides Act 1968)
  banned_refused_restricted           — banned / refused / restricted lists (31.07.2026)
  mup_*                               — Major Uses of Pesticides (certificate-based, 31.03.2026)

Design (risk-minimized):
  * brands / prices deliberately EXCLUDED (no authoritative free source; staleness risk)
  * every status carries source_pdf + line for the validator's drift check
  * status precedence: banned > restricted > registered > not_listed
  * the recommender hard-gates only `banned`; restricted is surfaced as information
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "kb" / "raw" / "cibrc"
OUT = ROOT / "kb" / "rules" / "registry.json"

SRC_FORM = "cibrc/formulations_registered_31.03.2026"
SRC_BAN = "cibrc/banned_refused_restricted"
MUP = {
    "insecticide": "cibrc/mup_insecticides_31.03.2026",
    "fungicide": "cibrc/mup_fungicides_31.03.2026",
    "herbicide": "cibrc/mup_herbicides_31.03.2026",
    "bio": "cibrc/mup_bio_insecticides_31.03.2026",
}
TXT = {src: RAW / (Path(src).name + ".txt") for src in [SRC_FORM, SRC_BAN, *MUP.values()]}

# Curated actives + formulation probe per rule chemical (expert-maintained).
# probe None => formulation-level match skipped (string carries doses, not formulation).
CURATED: dict[str, dict] = {
    "Cartap hydrochloride 4 G":                  {"actives": ["Cartap"], "probe": "4 G"},
    "Chlorantraniliprole 0.4% G":                {"actives": ["Chlorantraniliprole"], "probe": "0.4% G"},
    "Fipronil 0.3% G":                           {"actives": ["Fipronil"], "probe": "0.3% G"},
    "Acephate 75% WP":                           {"actives": ["Acephate"], "probe": "75% WP"},
    "Tricyclazole 75% WP":                       {"actives": ["Tricyclazole"], "probe": "75% WP"},
    "Isoprothiolane EC":                         {"actives": ["Isoprothiolane"], "probe": "EC"},
    "Hexaconazole 5% EC":                        {"actives": ["Hexaconazole"], "probe": "5% EC"},
    "Streptocycline 2 g + Copper oxychloride 20 g": {"actives": ["Streptocycline", "Copper oxychloride"], "probe": None},
    "Beauveria bassiana (NBAIR)":                {"actives": ["Beauveria bassiana"], "probe": None},
    "Colocasia esculenta extract (botanical)":   {"actives": ["Colocasia"], "probe": None},
    "buprofezin 25SC 0.05% (2 ml/L)":            {"actives": ["Buprofezin"], "probe": "25 SC"},
    "Dichlorvos 76 EC 0.05%":                    {"actives": ["Dichlorvos"], "probe": "76 EC"},
    "Oxadiargyl 80% WP":                         {"actives": ["Oxadiargyl"], "probe": "80% WP"},
    "Metasulfuron-methyl + chloromuron-ethyl":   {"actives": ["Metsulfuron", "Metasulfuron", "Chlorimuron", "Chlormuron"], "probe": None},
    "Oxiflufen 300 g/ha pre-emergence + 2,4-D 500 g/ha post-emergence":
        {"actives": ["Oxyfluorfen", "Oxiflufen", "2,4-D"], "probe": None},
    "Pretilachlor 30.7 EC":                      {"actives": ["Pretilachlor"], "probe": "30.7 EC"},
    # Brown spot / leaf scald fungicides (MUP 31.03.2026). MUP spells "Ediphenphos",
    # formulations list spells "Edifenphos" — probe both spellings as actives.
    "Ediphenphos 50% EC":                        {"actives": ["Ediphenphos", "Edifenphos"], "probe": "50% EC"},
    "Azoxystrobin 8.3% + Mancozeb 66.7% WG":     {"actives": ["Azoxystrobin", "Mancozeb"], "probe": "8.3% + Mancozeb 66.7% WG"},
    "Picoxystrobin 10% + Isoprothiolane 25% EC": {"actives": ["Picoxystrobin", "Isoprothiolane"], "probe": "10% + Isoprothiolane 25% EC"},
    "Carbendazim 5% GR":                         {"actives": ["Carbendazim"], "probe": "5% GR"},
    "Validamycin 3% L":                          {"actives": ["Validamycin"], "probe": "3% L"},
    "Mancozeb 75% WP":                           {"actives": ["Mancozeb"], "probe": "75% WP"},
}


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="replace").split("\n")


def nums_of(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]


KNOWN_CODES = {"EC", "WP", "SP", "SG", "DF", "WG", "SC", "WDG", "GR", "SL", "OD",
               "SE", "EW", "FS", "DS", "CS", "ME", "WSP", "G", "W", "P", "F"}


def codes_of(s: str) -> list[str]:
    s = s.upper()
    codes = re.findall(r"\d+(?:\.\d+)?\s*%\s*([A-Z]{1,3})\b", s)
    codes += re.findall(r"\d+(?:\.\d+)?\s+([A-Z]{1,3})\b", s)
    codes += re.findall(r"\d(?:[A-Z]{2,3})\b", s)          # 25SC style
    codes += [t for t in re.findall(r"\b([A-Z]{1,3})\b", s) if t in KNOWN_CODES]
    out = []
    for c in codes:
        out.append(re.sub(r"^\d+(?:\.\d+)?", "", c).upper() or c.upper())
    return [c for c in out if c]


def contains_all(line: str, nums: list[float], codes: list[str]) -> bool:
    lnums, lcodes = nums_of(line), set(codes_of(line))
    if nums and not all(n in lnums for n in nums):
        return False
    if codes and not all(c in lcodes for c in codes):
        return False
    return bool(nums or codes)


def scan_formulations(actives: list[str]) -> list[dict]:
    lines = read_lines(TXT[SRC_FORM])
    hits = []
    for i, ln in enumerate(lines, 1):
        if any(a.lower() in ln.lower() for a in actives):
            text = ln.strip()
            if text:
                hits.append({"text": text, "line": i})
    return hits


RESTRICT_TEXT = {
    "banned_for_manufacture_import_use": "banned in India for manufacture, import and use",
    "banned_for_use_export_only": "banned for use in India (manufacture continues for export only)",
    "withdrawn": "registration withdrawn",
}


def scan_banned(actives: list[str]) -> list[dict]:
    """Return hits with section category (I banned / II refused / III restricted)."""
    lines = read_lines(TXT[SRC_BAN])
    section, category = "I", "banned"
    sub = "banned_for_manufacture_import_use"
    hits = []
    for i, ln in enumerate(lines, 1):
        if re.search(r"^\s*II\.\s+PESTICIDES REFUSED", ln, re.I):
            section, category = "II", "refused"
        elif re.search(r"^\s*III\.\s+PESTICIDES RESTRICTED", ln, re.I):
            section, category = "III", "restricted"
        elif "banned for use but continued to manufacture for" in ln.lower():
            # header wraps across two lines ("...for" / "export") — match without "export"
            sub = "banned_for_use_export_only"
        elif "Pesticides Withdrawn" in ln:
            sub = "withdrawn"
        if section in ("I", "III"):
            for a in actives:
                if a.lower() in ln.lower():
                    entry = {"category": category, "line": i,
                             "text": ln.strip(), "src": SRC_BAN}
                    if section == "I":
                        entry["restriction"] = sub
                        entry["restriction_text"] = RESTRICT_TEXT[sub]
                    hits.append(entry)
                    break
    return hits


def scan_mup_rice(actives: list[str], cap: int = 3) -> list[dict]:
    """Find rice/paddy rows near any mention of the active across all MUP docs."""
    evidence: list[dict] = []
    for src in MUP.values():
        lines = read_lines(TXT[src])
        for i, ln in enumerate(lines):
            if not any(a.lower() in ln.lower() for a in actives):
                continue
            window = range(max(0, i - 1), min(len(lines), i + 12))
            for j in window:
                row = lines[j]
                if re.match(r"\s*(?:\([^)]*\)\s*)?Rice\b|^\s*Paddy\b", row, re.I) or \
                        re.search(r"\(Rice\)", row, re.I):
                    text = row.strip()
                    if text and not any(e["line"] == j + 1 and e["pdf"] == src for e in evidence):
                        evidence.append({"pdf": src, "line": j + 1, "text": text[:160]})
                    break
            if len(evidence) >= cap:
                return evidence[:cap]
    return evidence


def match_formulation(probe: str | None, registered: list[dict]) -> str:
    if probe is None:
        return "unverified"
    nums, codes = nums_of(probe), codes_of(probe)
    for r in registered:
        if contains_all(r["text"], nums, codes):
            return "exact"
    for r in registered:
        if nums and all(n in nums_of(r["text"]) for n in nums):
            return "numeric"          # strength matches, code differs (or absent)
    if registered:
        return "active_only"
    return "none"


def build() -> dict:
    fp = json.loads((ROOT / "kb" / "rules" / "fert_pest.json").read_text(encoding="utf-8"))
    rules = [(r.get("chemical"), "pesticide") for r in fp.get("pesticide", [])] + \
            [(r.get("herbicide"), "weeds") for r in fp.get("weeds", [])]
    cache = {src: read_lines(p) for src, p in TXT.items()}

    entries: dict[str, dict] = {}
    for chem, section in rules:
        if not chem or chem not in CURATED:
            raise SystemExit(f"FAIL: no curated registry mapping for rule chemical: {chem!r}")
        c = CURATED[chem]
        reg = scan_formulations(c["actives"])
        ban = scan_banned(c["actives"])
        banned = [b for b in ban if b["category"] in ("banned", "refused")]
        restricted = [b for b in ban if b["category"] == "restricted"]

        if banned:
            status = "banned"
        elif restricted:
            status = "restricted"
        elif reg:
            status = "registered"
        else:
            status = "not_listed"

        # per-active honesty: an entry may mix actives that differ in status
        presence = {
            a: {
                "in_formulations_list": any(a.lower() in r["text"].lower() for r in reg),
                "in_major_uses": any(a.lower() in ln.lower()
                                     for src in MUP.values() for ln in cache[src]),
            }
            for a in c["actives"]
        }

        entry = {
            "section": section,
            "actives": c["actives"],
            "status": status,
            "formulation_match": match_formulation(c["probe"], reg),
            "registered_formulations": reg[:6],
            "formulations_src": SRC_FORM,
            "active_presence": presence,
            "banned": banned[0] if banned else None,
            "restricted": restricted or None,
            "restriction_mentions_rice": bool(
                restricted and any(re.search(r"rice|paddy", r["text"], re.I)
                                   for r in restricted)),
            "rice_evidence": scan_mup_rice(c["actives"]),
        }
        if status == "not_listed":
            entry["note"] = ("active not found in CIB&RC formulations list as on 31.03.2026 "
                            "(botanical/uncertified or spelling variant) — rule kept as cited "
                            "agronomy advice, not confirmed as a registered product")
        elif banned:
            entry["note"] = f"DO NOT RECOMMEND: {banned[0].get('restriction_text', banned[0]['category'])}"
        elif restricted and not entry["restriction_mentions_rice"]:
            entry["note"] = ("restriction is crop-specific and does not mention rice — "
                             "rice use is not affected by the restriction")
        entries[chem] = entry

    doc = {
        "meta": {
            "authority": "CIB&RC, Directorate of Plant Protection, Quarantine & Storage (DPPQS), Govt. of India",
            "purpose": "Legality/registration layer for kb/rules chemicals. Brands and prices "
                       "deliberately excluded (no authoritative free source).",
            "as_of": {"formulations": "2026-03-31", "banned": "2026-07-31", "major_uses": "2026-03-31"},
            "sources": [SRC_FORM, SRC_BAN, *MUP.values()],
            "generated_by": "kb/build_registry.py",
        },
        "entries": entries,
    }
    OUT.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return doc


if __name__ == "__main__":
    doc = build()
    print(f"registry.json: {len(doc['entries'])} entries -> {OUT.relative_to(ROOT)}")
    for chem, e in doc["entries"].items():
        rice = len(e["rice_evidence"])
        print(f"  [{e['status']:10}] [{e['formulation_match']:9}] rice={rice}  {chem}")
