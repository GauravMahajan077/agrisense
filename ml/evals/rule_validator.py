"""AgriSense rule validator — expert-level tester for KB, recommender, and dataset.

Sections:
  A. KB schema & integrity        (fert_pest.json, varieties.json)
  B. Citation verification        (source exists, line resolves, drift snapshot)
  C. Recommender safety & logic   (abstention, leakage, provenance, determinism, fuzz)
  D. Domain-rule checks           (friend's agronomic rules vs CSV correlations)
  E. Dataset quality              (ranges, dupes, label/level coherence, leakage risk)
  F. Product registry & gate      (registry.json schema, banned-chemical hard gate)

Usage:  python ml/evals/rule_validator.py [--update-snapshot]
Exit:   0 = no FAIL, 1 = at least one FAIL (CI-ready)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics as st
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KB = ROOT / "kb"
RULES = KB / "rules"
RAW = KB / "raw"
SNAPSHOT = RULES / ".citations_snapshot.json"
DATASET = ROOT / "konkan_rice_disease_risk_10k.csv.xls"
sys.path.insert(0, str(ROOT / "ml" / "src"))

RESULTS: list[tuple[str, str, str]] = []  # (section, status, message)


def check(section: str, cond: bool, msg: str, warn_only: bool = False) -> None:
    status = "PASS" if cond else ("WARN" if warn_only else "FAIL")
    RESULTS.append((section, status, msg))


# ───────────────────────── A. KB schema & integrity ─────────────────────────

VALID_SOURCES_HINT = "source_pdf + line mandatory (contributor rule #1)"


def section_a() -> dict:
    rules = {}
    for name in ("fert_pest.json", "varieties.json"):
        path = RULES / name
        try:
            rules[name] = json.loads(path.read_text(encoding="utf-8"))
            check("A", True, f"{name}: valid JSON")
        except Exception as e:  # noqa: BLE001
            check("A", False, f"{name}: invalid JSON — {e}")
            return {}

    fp = rules["fert_pest.json"]
    vj = rules["varieties.json"]

    for sect in ("fertilizer", "pesticide", "weeds"):
        items = fp.get(sect, [])
        check("A", isinstance(items, list) and len(items) > 0,
              f"fert_pest.{sect}: {len(items)} rules")
        for i, r in enumerate(items):
            tag = f"fert_pest.{sect}[{i}]"
            check("A", "source_pdf" in r and "line" in r,
                  f"{tag}: provenance present ({VALID_SOURCES_HINT})", warn_only=True)
            if sect == "fertilizer":
                ok = any(k in r for k in ("n_kg_ha", "notes"))
                check("A", ok, f"{tag}: has n_kg_ha or notes")
                check("A", "zone" in r and "crop" in r, f"{tag}: zone+crop present")
            if sect == "pesticide":
                for k in ("target", "chemical", "dose"):
                    check("A", k in r, f"{tag}: field '{k}' present")
                dose = str(r.get("dose", ""))
                nums = [float(x) for x in re.findall(r"\d+\.?\d*", dose)]
                # unit sanity: granular kg/ha or g-per-10L sprays
                if "kg/ha" in dose and nums:
                    check("A", max(nums) <= 30,
                          f"{tag}: granular dose {dose} within sane bound (<=30 kg/ha)")
                check("A", not re.search(r"\bundefined\b|\bnull\b|nan", dose, re.I),
                      f"{tag}: dose has no placeholder junk")
            if sect == "weeds":
                for k in ("crop", "herbicide", "dose", "timing"):
                    check("A", k in r, f"{tag}: field '{k}' present")

    # duplicates (same chemical+target or same zone+notes)
    chem = [(r.get("target"), r.get("chemical")) for r in fp.get("pesticide", [])]
    dup = [k for k, n in Counter(chem).items() if n > 1 and k[1]]
    check("A", not dup, f"pesticide: no duplicate target+chemical pairs ({dup})" if dup
          else "pesticide: no duplicate target+chemical pairs")
    notes = [r.get("notes") for r in fp.get("fertilizer", []) if r.get("notes")]
    dupn = [k for k, n in Counter(notes).items() if n > 1]
    check("A", not dupn, f"fertilizer: duplicate notes found: {dupn}" if dupn
          else "fertilizer: no duplicate notes")

    vs = vj.get("varieties", [])
    check("A", len(vs) >= 10, f"varieties: {len(vs)} records (>=10 expected)")
    for i, r in enumerate(vs):
        check("A", "name" in r and "source_pdf" in r,
              f"varieties[{i}] '{r.get('name')}': name+source present")
        res = r.get("resistance")
        check("A", res is None or isinstance(res, list),
              f"varieties[{i}]: resistance is list or null")
    sw = vj.get("sowing_windows", [])
    check("A", len(sw) >= 4, f"sowing_windows: {len(sw)} district windows")

    # --- ipm.json ---
    ipm_path = RULES / "ipm.json"
    if not ipm_path.exists():
        check("A", False, "ipm.json missing (IPM-first layer required)")
        return rules
    try:
        ipm = json.loads(ipm_path.read_text(encoding="utf-8"))
        check("A", True, "ipm.json: valid JSON")
    except Exception as e:  # noqa: BLE001
        check("A", False, f"ipm.json: invalid JSON — {e}")
        return rules
    actions = ipm.get("actions", [])
    check("A", len(actions) >= 8, f"ipm actions: {len(actions)} (>=8 expected)")
    allowed_types = {"monitoring", "cultural", "mechanical", "variety", "biological"}
    chem_pat = re.compile(
        r"kg\s*/\s*ha|g\s*/\s*(10|litre|liter)|ml\s*/\s*(10|litre|liter)|\bppm\b"
        r"|\bEC\b|\bWP\b|\bSP\b|\bSC\b|\bWDG\b", re.I)
    known_chems = re.compile(
        r"tricyclazole|hexaconazole|isoprothiolane|cartap|fipronil|acephate"
        r"|chlorantraniliprole|quinalphos|chlorpyrifos|streptocycline|pretilachlor"
        r"|oxadiargyl|carbendazim|dimethoate|monocrotophos", re.I)
    for i, a in enumerate(actions):
        tag = f"ipm[{i}]"
        for k in ("target", "type", "action", "timing", "source_pdf", "line"):
            check("A", k in a, f"{tag}: field '{k}' present")
        check("A", a.get("type") in allowed_types,
              f"{tag}: type '{a.get('type')}' is non-chemical category")
        text = f"{a.get('action', '')} {a.get('timing', '')}"
        check("A", not known_chems.search(text),
              f"{tag}: no chemical name in action text", warn_only=True)
        check("A", not chem_pat.search(str(a.get("action", ""))),
              f"{tag}: action text carries no dose/formulation (IPM must be non-chemical)")
    return rules


# ─────────────────────── B. Citation verification ───────────────────────────

def resolve_source(source_pdf: str) -> Path | None:
    if not source_pdf:
        return None
    if "/" in source_pdf:
        # nested citation: "cibrc/<name>" -> kb/raw/cibrc/<name>.txt
        sub, _, name = source_pdf.partition("/")
        folder = RAW / sub
        if folder.is_dir():
            for p in sorted(folder.glob("*.txt")):
                if name.lower() in p.stem.lower() or p.stem.lower() in name.lower():
                    return p
        return None
    for p in sorted(RAW.glob("*.txt")):
        if source_pdf.lower() in p.stem.lower():
            return p
    return None


def section_b(rules: dict) -> None:
    if not rules:
        return
    snapshot: dict[str, str] = {}
    old: dict[str, str] = {}
    if SNAPSHOT.exists():
        try:
            old = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = {}

    citations: list[tuple[str, dict]] = []
    fp = rules["fert_pest.json"]
    for sect in ("fertilizer", "pesticide", "weeds"):
        for i, r in enumerate(fp.get(sect, [])):
            citations.append((f"fert_pest.{sect}[{i}]", r))
    for i, r in enumerate(rules["varieties.json"].get("varieties", [])):
        citations.append((f"varieties[{i}]", r))
    ipm_file = RULES / "ipm.json"
    if ipm_file.exists():
        for i, r in enumerate(json.loads(ipm_file.read_text(encoding="utf-8"))
                              .get("actions", [])):
            citations.append((f"ipm[{i}]", r))
    # registry citations (banned / restricted / rice evidence lines).
    # Normalized to source_pdf+line so they flow through the same resolution
    # + drift tracking; vocab overlap is skipped for them (registry fields are
    # absent -> vocab empty -> check self-skips) because CIB&RC layout text and
    # rule spelling variants (Oxiflufen vs Oxyfluorfen) legitimately differ.
    reg_file = RULES / "registry.json"
    if reg_file.exists():
        reg_entries = json.loads(reg_file.read_text(encoding="utf-8")).get("entries", {})
        for chem, e in reg_entries.items():
            if (b := e.get("banned")) and b.get("line"):
                citations.append((f"registry[{chem}].banned",
                                  {"source_pdf": b.get("src"), "line": b["line"]}))
            for j, x in enumerate(e.get("restricted") or []):
                citations.append((f"registry[{chem}].restricted[{j}]",
                                  {"source_pdf": x.get("src"), "line": x.get("line")}))
            for j, x in enumerate(e.get("rice_evidence") or []):
                citations.append((f"registry[{chem}].rice[{j}]",
                                  {"source_pdf": x.get("pdf"), "line": x.get("line")}))

    checked = resolved = 0
    for tag, r in citations:
        src = r.get("source_pdf")
        line = r.get("line")
        if not src or line is None:
            continue
        checked += 1
        path = resolve_source(str(src))
        if path is None:
            check("B", False, f"{tag}: source '{src}' not found in kb/raw/")
            continue
        resolved += 1
        text = read_text(path)
        lines = text.split("\n")
        if not (1 <= int(line) <= len(lines)):
            check("B", False, f"{tag}: line {line} out of range 1..{len(lines)} for {path.name}")
            continue
        cited = lines[int(line) - 1].strip()
        key = f"{src}#{line}"
        snapshot[key] = cited
        # ASCII sources: cited line must overlap rule's distinctive vocabulary
        if not str(src).endswith(".ocr"):
            vocab = set()
            for field in ("chemical", "herbicide", "target", "notes", "zone", "crop",
                          "name", "action"):
                v = r.get(field)
                if isinstance(v, str):
                    vocab |= {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", v)}
            line_words = {w.lower() for w in re.findall(r"[a-zA-Z]{4,}", cited)}
            if vocab and cited:
                overlap = vocab & line_words
                check("B", bool(overlap),
                      f"{tag}: cited line vocabulary matches rule "
                      f"(src={src}:{line}, overlap={sorted(overlap)[:4]})")
        if not cited and int(line) <= len(lines):
            check("B", False, f"{tag}: cited line {line} is empty in {path.name}")

    check("B", checked > 0, f"{checked} citations scanned, {resolved} sources resolved")

    if SNAPSHOT.exists() and old:
        drift = [k for k, v in snapshot.items() if old.get(k) and old[k] != v]
        removed = [k for k in old if k not in snapshot]
        check("B", not drift, f"citation drift detected: {drift[:5]}" if drift
              else f"no citation drift vs snapshot ({len(old)} tracked)")
        check("B", not removed, f"citations removed since snapshot: {removed[:5]}" if removed
              else "no citations removed since snapshot", warn_only=True)

    if not SNAPSHOT.exists() or "--update-snapshot" in sys.argv:
        SNAPSHOT.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        check("B", True, f"snapshot written: {len(snapshot)} citation lines "
                         f"({'updated' if old else 'initial run'})")


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# ──────────────────── C. Recommender safety & logic ─────────────────────────

def section_c() -> None:
    try:
        from recommender import Context, load_rules, recommend
    except Exception as e:  # noqa: BLE001
        check("C", False, f"recommender import failed: {e}")
        return
    rules = load_rules()

    def run(**kw):
        return recommend(Context(**kw), rules)

    def items(d, kind):
        return [x["item"] for x in d.get(kind, [])]

    # 1. coverage: every disease label now has a cited chemical rule.
    #    (Brown_Spot/Leaf_Scald abstained before the MUP 31.03.2026 rules were added.)
    pos = {
        "Sheath_Blight": "Tricyclazole",
        "Leaf_Blast": "Hexaconazole",
        "Bacterial_Leaf_Blight": "Streptocycline",
        "Brown_Spot": "Ediphenphos",
        "Leaf_Scald": "Mancozeb",
    }
    for lbl, chem in pos.items():
        got = items(run(disease_label=lbl), "pesticide")
        check("C", any(chem in g for g in got),
              f"coverage: {lbl} returns {chem} (got {got})")

    # 2. cross-disease leakage (ALL-token matching must isolate classes)
    leak_cases = [
        ("Bacterial_Leaf_Blight", ("Tricyclazole", "Isoprothiolane", "Hexaconazole"),
         "BLB must not get sheath/blast-only fungicides"),
        ("Sheath_Blight", ("Streptocycline",), "Sheath blight must not get antibiotic (BLB)"),
        ("Leaf_Blast", ("Streptocycline",), "Leaf blast must not get antibiotic (BLB)"),
        ("Leaf_Blast", ("Tricyclazole",), "Leaf blast must not get sheath-only Tricyclazole"),
        ("Leaf_Blast", ("Carbendazim 5% GR",),
         "Leaf blast must not get brown-spot-only Carbendazim GR"),
        ("Brown_Spot", ("Tricyclazole", "Hexaconazole"),
         "Brown spot must not get sheath/blast-only fungicides"),
        ("Leaf_Scald", ("Streptocycline", "Tricyclazole"),
         "Leaf scald must not get BLB antibiotic or blast-only Tricyclazole"),
    ]
    for lbl, banned, msg in leak_cases:
        got = items(run(disease_label=lbl), "pesticide")
        bad = [g for g in got if any(b in g for b in banned)]
        check("C", not bad, f"leakage: {msg} (found {bad})" if bad else f"leakage: {msg}")

    # 4. no disease -> no pesticide flood
    d = run(district="Ratnagiri", soil_n=90)
    check("C", not d["pesticide"], "no disease_label -> pesticide empty")
    check("C", not d["abstain"] and len(d["fertilizer"]) >= 3,
          "fertilizer query returns >=3 items, abstain=False")

    # 5. provenance 100% on returned items
    missing = []
    for kind in ("fertilizer", "pesticide", "weed"):
        for it in d.get(kind, []) if kind != "weed" else d.get("weed", []):
            s = it.get("source") or {}
            if not s.get("pdf") or s.get("line") is None:
                missing.append(it.get("item"))
    check("C", not missing, f"provenance: all returned items cite pdf+line "
                            f"(violations: {missing})")

    # 6. score bounds
    out_of_bounds = []
    for lbl in pos:
        for it in run(disease_label=lbl)["pesticide"]:
            if not (0.0 <= it["score"] <= 1.0):
                out_of_bounds.append((lbl, it["item"], it["score"]))
    check("C", not out_of_bounds, f"scores within [0,1] (violations: {out_of_bounds})")

    # 7. determinism: identical input -> identical output (3 runs)
    a = json.dumps(run(district="Ratnagiri", soil_n=90), sort_keys=True)
    det = all(json.dumps(run(district="Ratnagiri", soil_n=90), sort_keys=True) == a
              for _ in range(2))
    check("C", det, "determinism: 3 identical runs produce identical payload")

    # 8. monotonicity: low soil N ranks high-N dose >= its score at high soil N
    #    (tested on the scorer directly — recommend() truncates to top-3)
    from recommender import score_fertilizer
    n100 = next((r for r in rules.get("fertilizer", [])
                 if r.get("n_kg_ha") == 100 and r.get("p2o5_kg_ha") == 40), None)
    if n100 is not None:
        lo = score_fertilizer(n100, Context(soil_n=80)).score
        hi = score_fertilizer(n100, Context(soil_n=300)).score
        check("C", lo >= hi,
              f"monotonicity: N100:40:40 score lowN={lo} >= highN={hi}")
    else:
        check("C", False, "monotonicity: N100:40:40 rule missing from KB", warn_only=True)

    # 9. rainfall anomaly boosts spray priority
    base = run(disease_label="Leaf_Blast")["pesticide"][0]["score"]
    rainy = run(disease_label="Leaf_Blast", rainfall_anomaly_pct=40)["pesticide"][0]["score"]
    check("C", rainy >= base, f"weather sensitivity: anomaly score {rainy} >= baseline {base}")

    # 10. fuzz: hostile contexts must not crash or explode
    fuzz = [
        {}, {"district": ""}, {"district": "Nowhere"}, {"soil_n": -50},
        {"soil_n": 99999}, {"soil_p": 0, "soil_k": 0}, {"season": "kharif", "stage": ""},
        {"disease_label": "Healthy"}, {"disease_label": "blast"},
        {"disease_label": "A" * 200}, {"rainfall_anomaly_pct": -100},
        {"district": "Ratnagiri", "disease_label": "Sheath_Blight",
         "soil_n": 90, "rainfall_anomaly_pct": 40},
    ]
    crashed = []
    for kw in fuzz:
        try:
            out = recommend(Context(**kw), rules)
            for kind in ("fertilizer", "pesticide", "weed"):
                for it in out[kind]:
                    assert 0 <= it["score"] <= 1
        except Exception as e:  # noqa: BLE001
            crashed.append((kw, str(e)))
    check("C", not crashed, f"fuzz: {len(fuzz)} hostile contexts, 0 crashes ({crashed})")

    # 11. stopwords bridge protection (regression of the safety bug we fixed)
    d = run(disease_label="Bacterial_Leaf_Blight")
    stem = [g for g in items(d, "pesticide") if "folder" in g.lower() or "borer" in g.lower()]
    check("C", not stem,
          f"stopwords: BLB never matches stem-borer insecticides (got {stem})")

    # 12. IPM-first contract: ipm key comes first, is non-empty for rice queries,
    #     and every action carries provenance + stays chemical-free
    first_key = list(run(district="Ratnagiri", soil_n=90).keys())[0]
    check("C", first_key == "ipm", f"IPM-first: 'ipm' is the first payload key (got '{first_key}')")
    dose_re = re.compile(r"kg\s*/\s*ha|g\s*/\s*10|ml\s*/\s*10|\bEC\b|\bWP\b|\bSP\b")
    chem_re = re.compile(
        r"tricyclazole|hexaconazole|cartap|fipronil|acephate|chlorpyrifos"
        r"|streptocycline|pretilachlor", re.I)
    ipm_leaks = []
    for lbl in ("Sheath_Blight", "Leaf_Blast", "Bacterial_Leaf_Blight",
                "Brown_Spot", "Leaf_Scald"):
        out = run(disease_label=lbl, district="Ratnagiri")
        if not out["ipm"]:
            check("C", False, f"IPM coverage: {lbl} returned zero non-chemical actions")
            continue
        for a in out["ipm"]:
            text = f"{a['item']} {json.dumps(a['detail'])}"
            if dose_re.search(a["item"]) or chem_re.search(text):
                ipm_leaks.append((lbl, a["item"]))
            if not (a.get("source") or {}).get("pdf") or a["source"].get("line") is None:
                ipm_leaks.append((lbl, "missing provenance", a["item"]))
    check("C", not ipm_leaks,
          f"IPM actions: chemical-free + provenance across all 5 diseases ({ipm_leaks})")
    out = run(disease_label="Sheath_Blight", district="Ratnagiri")
    check("C", "ipm_note" in out and bool(out["ipm_note"]),
          "IPM: ipm_note explains chemical-only-when-justified")
    # IPM must be returned BEFORE any pesticide list in payload order
    keys = list(out.keys())
    check("C", keys.index("ipm") < keys.index("pesticide"),
          f"IPM-first ordering: ipm {keys.index('ipm')} < pesticide {keys.index('pesticide')}")


# ───────────────── D. Domain rules (friend) vs CSV correlations ─────────────

def frac(rows, pred) -> float:
    return sum(1 for r in rows if pred(r)) / len(rows) if rows else 0.0


def section_d(data: list[dict] | None) -> None:
    if not data:
        check("D", False, "dataset unavailable — domain-rule checks skipped", warn_only=True)
        return
    f = lambda r, k: float(r[k])  # noqa: E731
    by = lambda lbl: [r for r in data if r["primary_disease_risk"] == lbl]  # noqa: E731
    healthy = by("None (Healthy)")
    check("D", bool(healthy), f"healthy class present: n={len(healthy)}")

    # Friend's rule 1 — Leaf Blast: low night T (<22.5), RH>82, wet, excess N
    blast, other = by("Leaf Blast"), [r for r in data if r["primary_disease_risk"] != "Leaf Blast"]
    r1 = [
        ("night T<22.5 enriched in blast vs rest",
         frac(blast, lambda r: f(r, "temperature_min") < 22.5) >
         frac(other, lambda r: f(r, "temperature_min") < 22.5)),
        ("RH>82 enriched in blast vs rest",
         frac(blast, lambda r: f(r, "relative_humidity") > 82) >
         frac(other, lambda r: f(r, "relative_humidity") > 82)),
        ("excessive N enriched in blast",
         frac(blast, lambda r: r["nitrogen_applied_level"] == "Excessive") >
         frac(other, lambda r: r["nitrogen_applied_level"] == "Excessive")),
    ]
    for msg, ok in r1:
        check("D", ok, f"LeafBlast rule: {msg}")

    # Friend's rule 2 — BLB: rain>100, tmax 28-35, humidity, surplus N
    blb = by("Bacterial Leaf Blight")
    r2 = [
        ("rain>100 enriched vs rest",
         frac(blb, lambda r: f(r, "rainfall_7d_forecast") > 100) >
         frac(other, lambda r: f(r, "rainfall_7d_forecast") > 100)),
        ("tmax 28-35 enriched vs rest",
         frac(blb, lambda r: 28 <= f(r, "temperature_max") <= 35) >
         frac(other, lambda r: 28 <= f(r, "temperature_max") <= 35)),
        ("surplus N enriched vs rest",
         frac(blb, lambda r: r["nitrogen_applied_level"] == "Excessive") >
         frac(other, lambda r: r["nitrogen_applied_level"] == "Excessive")),
    ]
    for msg, ok in r2:
        check("D", ok, f"BLB rule: {msg}")

    # Friend's rule 3 — Sheath Blight: water >7.5-8.0, tillering/PI stage
    sb = by("Sheath Blight")
    r3 = [
        ("deep water (>7.5cm) enriched vs rest",
         frac(sb, lambda r: f(r, "field_water_level_cm") > 7.5) >
         frac(other, lambda r: f(r, "field_water_level_cm") > 7.5)),
        ("tillering/panicle-initiation stage enriched",
         frac(sb, lambda r: r["growth_stage"] in ("Tillering", "Panicle Initiation")) >
         frac(other, lambda r: r["growth_stage"] in ("Tillering", "Panicle Initiation"))),
    ]
    for msg, ok in r3:
        check("D", ok, f"SheathBlight rule: {msg}")

    # Friend's rule 4 — Brown Spot: low N, pH<5.6
    bs = by("Brown Spot")
    r4 = [
        ("pH<5.6 enriched vs rest",
         frac(bs, lambda r: f(r, "soil_ph") < 5.6) > frac(other, lambda r: f(r, "soil_ph") < 5.6)),
        ("Low N enriched vs rest",
         frac(bs, lambda r: r["nitrogen_applied_level"] == "Low") >
         frac(other, lambda r: r["nitrogen_applied_level"] == "Low")),
    ]
    for msg, ok in r4:
        check("D", ok, f"BrownSpot rule: {msg}")

    # Friend's rule 5 — Healthy: balanced N, moderate RH, low rain stress
    r5 = [
        ("Optimum N enriched in healthy",
         frac(healthy, lambda r: r["nitrogen_applied_level"] == "Optimum") >
         frac(blast + blb + sb + bs, lambda r: r["nitrogen_applied_level"] == "Optimum")),
        ("healthy mean RH lower than sick mean RH",
         st.mean([f(r, "relative_humidity") for r in healthy]) <
         st.mean([f(r, "relative_humidity") for r in blast + blb + sb + bs])),
    ]
    for msg, ok in r5:
        check("D", ok, f"Healthy rule: {msg}")


# ───────────────────────── E. Dataset quality ───────────────────────────────

EXPECTED_RANGES = {
    "temperature_min": (15, 32), "temperature_max": (22, 45),
    "relative_humidity": (30, 100), "rainfall_7d_forecast": (0, 500),
    "consecutive_rainy_days": (0, 15), "field_water_level_cm": (0, 25),
    "soil_ph": (4.0, 9.0),
}
LEVEL_BANDS = [("Low", 0, 30), ("Moderate", 30, 60), ("High", 60, 80), ("Critical", 80, 100)]


def section_e(data: list[dict] | None) -> list[dict] | None:
    if data is None:
        check("E", False, "dataset missing — did you keep konkan_rice_disease_risk_10k.csv.xls in root?",
              warn_only=True)
        return None
    check("E", len(data) == 10000, f"row count = {len(data)} (expected 10000)")

    # completeness
    empties = Counter(k for r in data for k, v in r.items() if str(v).strip() == "")
    check("E", not empties, f"no empty cells (found {dict(empties)})")

    # ranges
    for col, (lo, hi) in EXPECTED_RANGES.items():
        vals = [float(r[col]) for r in data]
        bad = sum(1 for v in vals if not (lo <= v <= hi))
        check("E", bad == 0, f"range {col}: [{lo},{hi}] violations={bad}")

    # tmin < tmax sanity
    bad = sum(1 for r in data if float(r["temperature_min"]) >= float(r["temperature_max"]))
    check("E", bad == 0, f"temperature_min < temperature_max for all rows (violations={bad})")

    # risk_level vs score coherence
    bad = []
    for r in data:
        s, lvl = float(r["overall_risk_score"]), r["risk_level"]
        exp = next((name for name, lo, hi in LEVEL_BANDS if lo < s <= hi
                    or (name == "Low" and s <= 30)), None)
        if lvl != exp:
            bad.append((s, lvl, exp))
    check("E", not bad, f"risk_level consistent with score bands (violations={bad[:3]})")
    check("E", set(r["risk_level"] for r in data) <= {n for n, _, _ in LEVEL_BANDS},
          f"levels subset of {sorted({n for n, _, _ in LEVEL_BANDS})}")

    # label set matches model classes + friend's vocabulary
    labels = set(r["primary_disease_risk"] for r in data)
    expected = {"Leaf Blast", "Bacterial Leaf Blight", "Sheath Blight", "Brown Spot", "None (Healthy)"}
    check("E", labels == expected, f"label set matches 5 classes: {sorted(labels)}")

    # healthy <-> Low band coupling check (label leakage smell)
    h_all_low = all(r["risk_level"] == "Low" for r in data if r["primary_disease_risk"] == "None (Healthy)")
    low_all_h = all(r["primary_disease_risk"] == "None (Healthy)" for r in data if r["risk_level"] == "Low")
    if h_all_low and low_all_h:
        check("E", False, "LEAKAGE: risk_level 'Low' is 1:1 with 'None (Healthy)' — "
                          "model can cheat by mapping level->label; regenerate labels or drop level from features",
              warn_only=True)
    else:
        check("E", True, "no 1:1 coupling between risk_level and healthy label")

    # duplicates
    keys = [tuple(r.values()) for r in data]
    check("E", len(keys) == len(set(keys)), f"no exact duplicate rows ({len(keys)-len(set(keys))} found)")

    # near-constant / dtype checks
    for col in data[0]:
        vals = set(r[col] for r in data)
        check("E", len(vals) > 1, f"column '{col}' has variance ({len(vals)} unique)")

    # class balance
    cnt = Counter(r["primary_disease_risk"] for r in data)
    ratio = max(cnt.values()) / min(cnt.values())
    check("E", ratio <= 4.0, f"class balance ok (max/min ratio={ratio:.1f} <=4)", warn_only=True)

    # score must be derivable from features? proxy: any single feature perfectly separates label
    separable = []
    for col in ("field_water_level_cm", "soil_ph", "relative_humidity"):
        for lbl in expected:
            a = [float(r[col]) for r in data if r["primary_disease_risk"] == lbl]
            b = [float(r[col]) for r in data if r["primary_disease_risk"] != lbl]
            if a and b and (max(a) < min(b) or min(a) > max(b)):
                separable.append(f"{col}->{lbl}")
    check("E", not separable,
          f"no trivially separable feature->label axis (found {separable})", warn_only=True)

    # overall_risk_score distribution sanity
    scores = [float(r["overall_risk_score"]) for r in data]
    check("E", 5 <= st.mean(scores) <= 60,
          f"score mean {st.mean(scores):.1f} plausible")
    return data


# ───────────────────── F. Product registry & banned-gate ─────────────────────

VALID_STATUS = {"banned", "restricted", "registered", "not_listed"}
VALID_MATCH = {"exact", "numeric", "active_only", "unverified", "none"}
DISEASE_LABELS = ["Leaf_Blast", "Sheath_Blight", "Bacterial_Leaf_Blight",
                  "Brown_Spot", "Leaf_Scald", "Healthy"]


def section_f(rules: dict) -> None:
    """Legality layer: registry.json schema + the banned-chemical hard gate."""
    reg_file = RULES / "registry.json"
    if not reg_file.exists():
        check("F", False, "kb/rules/registry.json missing (run kb/build_registry.py)")
        return
    try:
        reg = json.loads(reg_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        check("F", False, f"registry.json invalid JSON — {e}")
        return

    meta = reg.get("meta") or {}
    as_of = meta.get("as_of") or {}
    for k in ("formulations", "banned", "major_uses"):
        check("F", bool(as_of.get(k)), f"meta.as_of.{k} = {as_of.get(k)!r}")

    entries = reg.get("entries") or {}
    check("F", len(entries) > 0, f"{len(entries)} registry entries")

    # coverage: every rule chemical/herbicide has an entry (and no strays)
    fp = rules.get("fert_pest.json")
    if fp:
        rule_keys = [r.get("chemical") for r in fp.get("pesticide", [])] + \
                    [r.get("herbicide") for r in fp.get("weeds", [])]
        missing = [k for k in rule_keys if k not in entries]
        strays = [k for k in entries if k not in rule_keys]
        check("F", not missing, f"all rule chemicals covered (missing: {missing})")
        check("F", not strays, f"no stray registry entries (strays: {strays})")

    # schema per entry
    for chem, e in entries.items():
        st_ = e.get("status")
        check("F", st_ in VALID_STATUS, f"registry[{chem}]: status={st_!r}")
        check("F", e.get("formulation_match") in VALID_MATCH,
              f"registry[{chem}]: formulation_match={e.get('formulation_match')!r}")
        check("F", isinstance(e.get("actives"), list) and e["actives"],
              f"registry[{chem}]: actives listed")
        if st_ == "banned":
            b = e.get("banned") or {}
            check("F", bool(b.get("line") and b.get("src")),
                  f"registry[{chem}]: banned entry cites source+line")
            check("F", "DO NOT RECOMMEND" in (e.get("note") or ""),
                  f"registry[{chem}]: note carries DO NOT RECOMMEND")
        if st_ == "restricted":
            check("F", bool(e.get("restricted")),
                  f"registry[{chem}]: restricted entry carries restriction detail")
        if st_ == "registered":
            check("F", bool(e.get("rice_evidence")),
                  f"registry[{chem}]: registered + rice evidence cited")

    # the specific safety-critical assertion: Dichlorvos is banned, cited
    dv_key = next((k for k in entries if "Dichlorvos" in k), None)
    check("F", dv_key is not None, "Dichlorvos present in registry")
    if dv_key:
        e = entries[dv_key]
        check("F", e.get("status") == "banned",
              f"Dichlorvos status is 'banned' (got {e.get('status')!r})")
        b = e.get("banned") or {}
        src = resolve_source(str(b.get("src") or ""))
        ok = False
        if src and b.get("line"):
            lines = read_text(src).split("\n")
            if 1 <= int(b["line"]) <= len(lines):
                cited = lines[int(b["line"]) - 1]
                ok = "Dichlorvos" in cited
                check("F", ok, f"Dichlorvos ban line {b['line']} resolves to "
                               f"'{cited.strip()[:60]}'")
        if not src:
            check("F", False, f"Dichlorvos ban source '{b.get('src')}' not found")

    # behavioral: forced match MUST be gated out + flagged with citation
    try:
        import recommender as R
    except Exception as e:  # noqa: BLE001
        check("F", False, f"cannot import recommender: {e}")
        return

    kb_rules = R.load_rules()
    dv_rule = next((r for r in (fp or {}).get("pesticide", [])
                    if "Dichlorvos" in r.get("chemical", "")), None) if fp else None
    if dv_rule:
        # its target is a non-rice borer query — force the distinctive-token match
        fake_label = "_".join((dv_rule.get("target") or "force").split())
        forced = R.score_pesticide(dv_rule, R.Context(disease_label=fake_label))
        check("F", forced is not None,
              "Dichlorvos rule can match when its target is queried (gate is testable)")
        if forced:
            flags: list[dict] = []
            kept = R.gate_banned([forced], kb_rules["registry"]["entries"], flags)
            check("F", not kept, "banned gate: Dichlorvos removed from recommendations")
            check("F", len(flags) == 1 and flags[0].get("source", {}).get("line"),
                  "banned gate: safety flag raised with source+line")

    # behavioral: no disease query may ever surface Dichlorvos
    for lbl in DISEASE_LABELS:
        out = R.recommend(R.Context(disease_label=lbl), rules=kb_rules)
        surf = [p["item"] for p in out.get("pesticide", []) if "Dichlorvos" in p["item"]]
        surf += [p["item"] for p in out.get("weed", []) if "Dichlorvos" in p["item"]]
        check("F", not surf, f"query '{lbl}': Dichlorvos absent from output")
        check("F", isinstance(out.get("safety_flags"), list),
              f"query '{lbl}': payload has safety_flags list")
        check("F", str(out.get("model_version", "")).startswith("kb-"),
              f"query '{lbl}': model_version={out.get('model_version')!r}")
        if out.get("safety_flags"):
            check("F", out.get("needs_expert") is True,
                  f"query '{lbl}': safety flags force needs_expert")

    # registry block attached to pesticide payloads when available
    out = R.recommend(R.Context(disease_label="Sheath_Blight"), rules=kb_rules)
    for p in out.get("pesticide", []):
        rg = p.get("registry")
        check("F", isinstance(rg, dict) and rg.get("status") in VALID_STATUS,
              f"payload '{p['item']}': registry block status valid")
        check("F", str(rg.get("authority", "")).startswith("CIB&RC") if rg else False,
              f"payload '{p['item']}': registry authority cited")


# ──────────────────────────────── runner ────────────────────────────────────

def load_dataset() -> list[dict] | None:
    if not DATASET.exists():
        return None
    with open(DATASET, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-snapshot", action="store_true")
    args = ap.parse_args()
    if args.update_snapshot:
        sys.argv.append("--update-snapshot")

    print("=" * 78)
    print("AGRISENSE RULE VALIDATOR — expert tester")
    print("=" * 78)

    rules = section_a()
    section_b(rules)
    section_c()
    data = load_dataset()
    section_e(data)
    section_d(data)
    section_f(rules)

    widths = {"A": "KB schema", "B": "Citations", "C": "Recommender",
              "D": "Domain rules", "E": "Dataset", "F": "Registry/gate"}
    print()
    for sec in ("A", "B", "C", "D", "E", "F"):
        rows = [r for r in RESULTS if r[0] == sec]
        if not rows:
            continue
        npass = sum(1 for r in rows if r[1] == "PASS")
        nwarn = sum(1 for r in rows if r[1] == "WARN")
        nfail = sum(1 for r in rows if r[1] == "FAIL")
        print(f"\n── {sec}. {widths[sec]}  — {npass} pass, {nwarn} warn, {nfail} fail")
        for _, status, msg in rows:
            if status == "PASS":
                print(f"  ✓ {msg}")
            elif status == "WARN":
                print(f"  ⚠ {msg}")
            else:
                print(f"  ✗ FAIL: {msg}")

    total = len(RESULTS)
    fails = [r for r in RESULTS if r[1] == "FAIL"]
    warns = [r for r in RESULTS if r[1] == "WARN"]
    print("\n" + "=" * 78)
    print(f"TOTAL: {total - len(fails) - len(warns)} pass, {len(warns)} warn, "
          f"{len(fails)} fail  ->  {'OK' if not fails else 'FAILED'}")
    print("=" * 78)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
