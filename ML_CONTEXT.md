# AgriSense — Project Context (single source of truth)

> This file records **everything** about the AgriSense ML work: what the user asked us to
> remember, hard constraints, what has been built, every finding with its source, every rule
> in the knowledge base, the current state, and what is still open. Read this before any work.
> Last updated: 2026-10-08.

---

## 1. What AgriSense is

AgriSense is a **research-grade rice Decision Support System (DSS)** for the Konkan region of
Maharashtra, India. It helps farmers decide:

- **C1 — Disease diagnosis:** classify a rice leaf photo into one of 6 classes
  (`Bacterial_Leaf_Blight, Brown_Spot, Healthy, Leaf_Blast, Leaf_Scald, Sheath_Blight`).
- **C2 — Fertilizer/pesticide recommendation:** a source-grounded KB recommender (no training,
  pure JSON rules + weighted ranking) that returns cited, legal, IPM-first advice.
- **C3 — Risk engine:** calibrated disease-risk score + conformal interval using disease
  confidence, IMD rainfall anomaly, phenology, and DWE yield stats.
- **Price:** MandiLens (AGMARKNET) published artifacts → `sell|wait` signal.
- **Advisory:** deterministic EN/MR templates, always `expert_pending=true`.

Scope is **ML only** (frontend/backend is a teammate). Mode: single-day sprint, free tools only.

---

## 2. User instructions to ALWAYS remember (hard constraints)

1. **Local machine is OFF-LIMITS for heavy work.** No training, no TF/PyTorch model loading,
   no big OCR/vision loops. Thermal/heatsink issue → serious hardware damage risk.
   Local = text extraction + tiny inference only (1–2 images max).
2. **All training / heavy compute runs on Kaggle notebooks (free GPU T4).**
3. **Free tools only:** TF/Keras, ultralytics (AGPL), scikit-learn, OpenCV, pdfplumber/pdftotext.
   No paid APIs.
4. **Every action goes through Gaurav for approve/reject** (approval gate before any write/edit).
5. **Safety rules for the recommender (non-negotiable):**
   - No chemical rule without `source_pdf` + `line` (verify with `kb/find_line.py`).
   - No KB match → `abstain` + `needs_expert`, never a guess.
   - IPM-first: non-chemical actions returned before any pesticide; `ipm.json` stays chemical-free.
   - Banned chemicals are hard-gated (CIB&RC registry) → removed + `safety_flags` + `needs_expert`.
   - Strict ALL-distinctive-token disease matching (stopwords must never bridge classes).
6. **Validator exit 0 = shippable.** Run `python ml/evals/rule_validator.py` before every handoff;
   `--update-snapshot` after intentional KB edits.
7. **User-facing chat should be beginner-friendly.**
8. **Never hallucinate facts** — every claim must be verifiable from a real file/source.
   If unsure, verify first (read the file), then act.

---

## 3. Work log (what has been done)

### 3.1 KB extraction (Pass 1) — DONE
- All 39 PDFs in `media/` → `kb/raw/*.txt` via `pdftotext -layout` (39/39 OK, 5.7 MB).
- Marathi scanned PDFs OCR'd locally (Tesseract `mar+eng`, 200 DPI):
  `1. Paddy.ocr.txt` (8 pp), `advisory-marathi.ocr.txt` (14 pp) via `kb/ocr_marathi.py`.

### 3.2 Structured rules (Pass 2) — DONE
- `kb/rules/varieties.json` — 12 rice varieties + 5 district sowing windows.
- `kb/rules/fert_pest.json` — fertilizer (11), pesticide (19), weeds (4).
- `kb/rules/ipm.json` — 10 non-chemical IPM actions (IPM-first layer).
- `kb/tables/rainfall_konkan_2026.csv` — 2026 seasonal rainfall departure by district (IMD).

### 3.3 CIB&RC registry (option 4) — DONE
- Downloaded 6 official GOI PDFs → `media/cibrc/` + text in `kb/raw/cibrc/`:
  - `formulations_registered_31.03.2026.txt` — registered formulations (Insecticides Act 1968)
  - `banned_refused_restricted.txt` — banned/refused/restricted lists (as on 31.07.2026)
  - `mup_insecticides_31.03.2026.txt`, `mup_fungicides_31.03.2026.txt`,
    `mup_herbicides_31.03.2026.txt`, `mup_bio_insecticides_31.03.2026.txt` — Major Uses of Pesticides
- `kb/build_registry.py` → `kb/rules/registry.json` (22 entries).
- Recommender upgraded to **kb-v3**: `_rule_key`, `gate_banned` → `safety_flags` + `needs_expert`,
  `registry_summary` per-item registry block, `load_rules` merge.
- Validator: `resolve_source()` cibrc/ subfolder fix + **section F** (registry schema/coverage/
  ban-line resolution/forced gate/6-label absence scan).
- `test_recommender.py` rewritten → green.
- Docs: `ml/API_CONTRACTS.md`, `kb/INDEX.md`, `ml/README.md`.

### 3.4 Brown spot + leaf scald chemical rules — DONE (this session)
- Verified every MUP row by reading the actual file (no assumptions).
- Added **4 in-label Brown_Spot rules** + **3 off-label Leaf_Scald rules** to `fert_pest.json`.
- Added 6 CURATED entries to `build_registry.py` + crop-specific-restriction note; rebuilt registry.
- Updated validator section_c: abstention checks → positive controls + leakage guards.
- Validator: **509 pass / 1 warn / 0 fail** (the 1 warn is a pre-existing dataset leakage warning).
- Smoke test: **OK** — all 5 disease labels return cited chemicals.

---

## 4. Data summaries & findings

### 4.1 Disease model (C1)
- `agrisense_b0.keras` (EfficientNet-B0): val macro-F1 **0.9488**, test **0.9315**,
  worst-class recall 0.8804, TFLite≡Keras 100%.
- **Field stress test (60 real photos): acc 0.50**, 13 confident-wrongs (conf ≥ 0.8).
  Main errors: Leaf_Blast→Sheath_Blight (9), Leaf_Blast→Brown_Spot (5), Leaf_Blast→Healthy (4).
- **Honest conclusion:** abstention ALONE cannot reach ≥95% accepted-accuracy — errors are
  *confident* (mean 0.75). Domain adaptation (crop + head finetune on Kaggle) is required first.
- Preprocessing contract (do not break): RGB → Lanczos resize **exactly 256×256** → float32
  **0..255** → graph resizes to 224. Never /255, never nearest-neighbour, never other HxW, never BGR.
  `abstain_threshold = 0.5`.

### 4.2 Dataset (friend's CSV, 10k rows)
- `konkan_rice_disease_risk_10k.csv.xls` — 10,000 rows, 5 classes
  (`Bacterial Leaf Blight, Brown Spot, Leaf Blast, None (Healthy), Sheath Blight`).
- **LEAKAGE WARNING (validator E):** `risk_level` 'Low' is 1:1 with 'None (Healthy)' — a model can
  cheat by mapping level→label. **Drop `risk_level` from features before training the risk model.**
- Class balance ok (max/min ratio 2.9 ≤ 4). No exact duplicate rows. Score mean 44.7 plausible.
- `DATASET_PATH` is still unfilled — needed before the Kaggle fine-tune can run.

### 4.3 Price (MandiLens)
- MAE **₹521.20**/quintal · WAPE **10.53%** · directional acc **49.15%** · interval coverage 71.6%
  vs 80% target. Published as-is (honest). ⚠️ No LICENSE file — check before public use.

### 4.4 CIB&RC registry findings (verified from files)
- **Banned file structure** (`banned_refused_restricted.txt`):
  - Section I "PESTICIDES / FORMULATIONS BANNED IN INDIA" (L6)
  - Section II "PESTICIDES REFUSED REGISTRATION" (L139)
  - Section III "PESTICIDES RESTRICTED FOR USE IN THE COUNTRY" (L161)
- **Dichlorvos is `banned`** (L109, S.O. 1196(E) 20.03.2020) → hard-gated out of recommendations.
- **Benomyl banned** (L18, S.O. 3951(E) 08.08.2018); **Captafol banned** (L107/179/183);
  **Fentin banned** (L153-154) → excluded from scald actives.
- **Mancozeb is `restricted`** (L229/L239) — crop-specific: "banned for use on Guava, Jowar and
  Tapioca" (S.O. 4294(E) 03.10.2023). **NOT rice** → registry note says rice use unaffected.
- **Oxyfluorfen `restricted`** (potato/groundnut only — not rice).
- **Spelling trap:** MUP spells "Ediphenphos"; formulations list spells "Edifenphos" (L915).
  Registry probes both spellings.
- **Metominostrobin: 0 hits in formulations list → NOT India-registered → EXCLUDE** (TNAU rec unusable).
- **Copper oxychloride has no single-product rice row** in MUP (only combos).

### 4.5 Brown spot / leaf scald evidence (verified this session)
- **No "scald" anywhere in the Indian MUP 2026** → any scald chemical rule is off-label by
  definition and must carry a transparent note.
- IRRI diagnostic (`media/scald_brownspot/irri_diagnostic.txt`): Leaf Scald §3.1.4 (L394-417),
  causal organism *Microdochium oryzae*. Diagnostic only — no doses.
- IRRI leaf-scald actives (from prior webfetch): seed treatment benomyl, carbendazim, quitozene,
  thiophanate-methyl; field spray benomyl, fentin acetate, edifenphos, validamycin; foliar
  captafol, mancozeb, copper oxychloride. India-registered viable subset: mancozeb, validamycin,
  edifenphos, carbendazim, thiophanate-methyl, copper oxychloride.
- eagri (`media/scald_brownspot/eagri_rice_diseases.txt`): Brown Spot header L78; main-field rec
  L129 "Edifenphos 500 ml or Mancozeb 2 kg/ha when grade reaches 3; repeat after 15 days".
  L145 (Carbendazim/Mancozeb) is **narrow brown leaf spot (Cercospora)** — NOT our Brown_Spot.
  eagri has no scald.
- TNAU / SL-DOA files: no scald content.

---

## 5. Knowledge base rules (current state)

### 5.1 Fertilizer (11 rules) — `fert_pest.json`
Zone-specific NPK doses for kharif rice across Konkan (South/North Coastal, rice-chickpea,
rice-sweet corn, rice-cowpea organic, DSR under mulch, rice-rice, rice-honeybee, rice-sugarcane),
Konkan Annapurna Briquettes 175 kg/ha, INM/biofertilizer. Sources: Konkan DSS report, Agresco
2018-2023, Paddy bulletin.

### 5.2 Pesticide (19 rules) — `fert_pest.json`
| # | Target | Chemical | Dose | Source:line |
|---|---|---|---|---|
| 0 | stem borer + leaf folder | Cartap hydrochloride 4 G | 18.75 kg/ha | Joint Agresco-2018:257 |
| 1 | stem borer + leaf folder | Chlorantraniliprole 0.4% G | 10 kg/ha | 1. Paddy.ocr:420 |
| 2 | stem borer + leaf folder | Fipronil 0.3% G | 20.8 kg/ha | 1. Paddy.ocr:420 |
| 3 | stem borer (ETL spray) | Acephate 75% WP | 12.5 g/10 L | 1. Paddy.ocr:348 |
| 4 | sheath blight | Tricyclazole 75% WP | 10 g/10 L (0.1%) | 1. Paddy.ocr:256 |
| 5 | sheath blight | Isoprothiolane EC | 10 ml/10 L (0.1%) | 1. Paddy.ocr:256 |
| 6 | sheath blight + leaf blast | Hexaconazole 5% EC | 20 ml/10 L | advisory-marathi.ocr:165 |
| 7 | bacterial leaf blight | Streptocycline 2 g + Copper oxychloride 20 g | 2+20 g/10 L | advisory-marathi.ocr:208 |
| 8 | stem borer | Beauveria bassiana (NBAIR) | 10 ml/L or 5 g/L | Konkan DSS:252 |
| 9 | case worm | Colocasia esculenta extract | as per trial | Konkan DSS:245 |
| 10 | mango hopper (context) | buprofezin 25SC 0.05% | 2 ml/L | Joint Agresco-2018:248 |
| 11 | cashew borer | Dichlorvos 76 EC 0.05% | 10 ml/10 L | Joint Agresco-2018:252 |
| 12 | **brown spot** | **Ediphenphos 50% EC** | **500-600 ml/ha (a.i. 250-300 g)** | **cibrc/mup_fungicides:666** |
| 13 | **brown spot** | **Azoxystrobin 8.3% + Mancozeb 66.7% WG** | **1500 g/ha (a.i. 124.5+1000)** | **cibrc/mup_fungicides:2927** |
| 14 | **brown spot** | **Picoxystrobin 10% + Isoprothiolane 25% EC** | **1000-1250 ml/ha (a.i. 350-437.5)** | **cibrc/mup_fungicides:4648** |
| 15 | **brown spot** | **Carbendazim 5% GR** | **12.5 kg/ha (a.i. 0.62 kg)** | **cibrc/mup_fungicides:215** |
| 16 | **leaf scald (off-label)** | **Mancozeb 75% WP** | **1.5-2 kg/ha** | **cibrc/mup_fungicides:1056** |
| 17 | **leaf scald (off-label)** | **Validamycin 3% L** | **2000 ml/ha (a.i. 60 g)** | **cibrc/mup_fungicides:2559** |
| 18 | **leaf scald (off-label)** | **Ediphenphos 50% EC** | **500-600 ml/ha (a.i. 250-300 g)** | **cibrc/mup_fungicides:666** |

**Brown_Spot rules are in-label** (CIB&RC MUP claims "Brown leaf spot"/"Brown spot"/"Brown Leaf Spot").
**Leaf_Scald rules are off-label** — no Indian label claim exists; IRRI recommends these actives
for scald (*Microdochium oryzae*); the dose shown is the CIB&RC rice registration for the same
chemical on blast/sheath blight/brown spot. Each scald rule carries a `notes` field explaining this.

### 5.3 Weeds (4 rules) — `fert_pest.json`
Oxadiargyl 80% WP (100 g a.i./ha, 2-3 DAS), Metasulfuron-methyl + chloromuron-ethyl (4 g a.i./ha,
25 DAS), Oxiflufen + 2,4-D (300 + 500 g/ha), Pretilachlor 30.7 EC (0.50 kg a.i./ha).

### 5.4 IPM (10 actions) — `ipm.json`
Water management (5-10 cm standing water, stage-adjusted), KKV cono weeder, Tephrosia green
manure in nursery, ETL scouting (5% dead hearts / 1 moth per sq.m), resistant varieties
(BM-4, Ratnagiri-7, Karjat-184, Karjat-8). **Chemical-free by contract.**

### 5.5 Registry (22 entries) — `registry.json`
Status precedence `banned > restricted > registered > not_listed`. All 22 rule chemicals covered,
no strays. Dichlorvos banned; Mancozeb + Azoxystrobin+Mancozeb + Oxiflufen restricted
(crop-specific, not rice — note explains). Brands/prices deliberately excluded.

---

## 6. Recommender behavior (kb-v3)

- **IPM-first:** `ipm` key is the first payload key; `ipm_note` explains chemical-only-when-justified.
- **Strict matching:** only DISTINCTIVE tokens count. `Brown_Spot`=[brown, spot] (both required),
  `Leaf_Scald`=[scald], `Leaf_Blast`=[blast], `Bacterial_Leaf_Blight`=[bacterial, blight],
  `Sheath_Blight`=[sheath, blight]. Stopwords: rice/paddy/leaf/healthy/the/and/of/on/in.
- **Coverage now:** all 5 foliar classes return cited chemicals:
  - Sheath_Blight → Tricyclazole, Isoprothiolane, Hexaconazole
  - Leaf_Blast → Hexaconazole
  - Bacterial_Leaf_Blight → Streptocycline + Copper oxychloride
  - Brown_Spot → Ediphenphos, Azoxystrobin+Mancozeb, Picoxystrobin+Isoprothiolane
  - Leaf_Scald → Mancozeb, Validamycin, Ediphenphos
- **Banned gate:** Dichlorvos removed + `safety_flags` + `needs_expert`.
- **Registry block:** each pesticide/weed payload carries `registry` (status, formulation_match,
  as_of, rice_evidence, restriction info, note).

---

## 7. Validation & test status

- `python ml/evals/rule_validator.py` → **509 pass / 1 warn / 0 fail** (exit 0).
  - The 1 warn is pre-existing: dataset `risk_level` 'Low' ↔ 'None (Healthy)' leakage.
- `python ml/evals/test_recommender.py` → **OK** (all 5 disease labels return cited chemicals;
  banned gate works; registry blocks attached).
- Snapshot: `kb/rules/.citations_snapshot.json` (102 citation lines tracked).

---

## 8. Open items / next steps

| Item | Action | Where |
|---|---|---|
| Fill `DATASET_PATH` + field photos | Run `ml/kaggle/field_finetune.py` **on Kaggle** | Kaggle |
| Fit `risk_artifact.json` | Needs `risk_features.csv`; run `ml/kaggle/fit_risk.py` on Kaggle | Kaggle |
| Drop `risk_level` from risk-model features | Leakage warning (validator E) | ml/kaggle/fit_risk.py |
| DWE 2020-24 district rice tables | Parse `kb/raw/DWE_*.txt` → `tables/dwe_rice_*.csv` | local |
| IMD daily rainfall for anomaly | Use `rf_p25_*_clm.nc` climatology + IMD API | ml/src |
| MandiLens LICENSE check | Before public use | — |
| Clean temp files | e.g. `kb/_peek_brownspot.py` if present | local |

---

## 9. Key files map

| Path | Purpose |
|---|---|
| `ML_PLAN.md` | 10-hour sprint plan, hard constraints, research contributions |
| `ML_CONTEXT.md` | THIS FILE — project context, findings, rules, instructions |
| `README.md` | Beginner-friendly overview (findings, facts, rules, working, setup) |
| `kb/INDEX.md` | What was extracted from which file, coverage status |
| `kb/rules/fert_pest.json` | Fertilizer/pesticide/weed rules (all cited) |
| `kb/rules/ipm.json` | Non-chemical IPM actions |
| `kb/rules/varieties.json` | Rice varieties + sowing windows |
| `kb/rules/registry.json` | CIB&RC legality/registration layer (22 entries) |
| `kb/build_registry.py` | Rebuilds registry.json from `kb/raw/cibrc/*.txt` |
| `kb/raw/cibrc/mup_fungicides_31.03.2026.txt` | Major Uses of Pesticides — fungicides (brown spot rows L666/L215/L2927/L4648; blast L1056; sheath blight L2559) |
| `kb/raw/cibrc/banned_refused_restricted.txt` | Banned/refused/restricted lists (sections I/II/III) |
| `kb/raw/cibrc/formulations_registered_31.03.2026.txt` | Registered formulations (Edifenphos L915, etc.) |
| `media/scald_brownspot/` | eagri, IRRI diagnostic, TNAU, SL-DOA PDFs+txt for brown spot/scald |
| `ml/src/recommender.py` | KB recommender (kb-v3) |
| `ml/src/risk_engine.py`, `price.py`, `advisory.py`, `abstain.py`, `crop_leaf.py` | Other ML modules |
| `ml/evals/rule_validator.py` | 509-check validator (sections A-F) |
| `ml/evals/test_recommender.py` | Recommender smoke test |
| `ml/evals/results.md` | Eval numbers (C1/C2/C3/Price/Advisory) |
| `ml/API_CONTRACTS.md` | Teammate-facing API contracts |
| `ml/kaggle/field_finetune.py`, `fit_risk.py` | Kaggle training/risk-fit scripts |