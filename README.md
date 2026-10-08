# AgriSense — Rice Disease & Farming Advisor (Konkan, Maharashtra)

> A research-grade **Decision Support System (DSS)** that helps rice farmers in the Konkan
> region decide: *"What disease does my crop have?"*, *"What should I spray / apply?"*,
> *"How risky is my crop right now?"*, *"Should I sell now or wait?"*.
>
> This README is written for **beginners** — no ML background needed. For the full technical
> record see [`ML_CONTEXT.md`](ML_CONTEXT.md) (everything we did, every finding, every rule).

---

## 1. What does AgriSense do? (in plain words)

| Feature | What it answers | How |
|---|---|---|
| **Disease diagnosis** | "What disease does my rice leaf have?" | A photo → AI model → one of 6 classes |
| **Fertilizer / pesticide advice** | "What should I apply, and is it legal?" | A knowledge base of real government documents → cited recommendations |
| **Risk score** | "How worried should I be?" | Combines disease confidence + rainfall + growth stage + yield stats |
| **Price signal** | "Sell now or wait?" | Mandi (market) price data from AGMARKNET |
| **Advisory** | "What do I tell the farmer?" | Ready-made messages in English and Marathi |

**Important:** every recommendation is *cited* — it points to the exact government PDF and line
number it came from. If we don't have a reliable answer, the system says **"I don't know — ask an
expert"** instead of guessing. That is a deliberate design choice.

---

## 2. The 6 rice diseases we detect

1. **Bacterial Leaf Blight** — bacterial, causes yellowing/wilting from leaf tips.
2. **Brown Spot** — fungal (*Bipolaris oryzae*), brown oval spots, common in low-N, acidic soils.
3. **Healthy** — no disease.
4. **Leaf Blast** — fungal (*Pyricularia oryzae*), diamond-shaped lesions with grey centres.
5. **Leaf Scald** — fungal (*Microdochium oryzae*), zonate (banded) lesions from leaf edges/tips.
6. **Sheath Blight** — fungal, lesions on the leaf sheath near water line.

---

## 3. Key findings (what we learned from the data)

### 3.1 The disease model works well on lab photos, but real fields are harder
- On standard test photos: **93% accuracy** (macro-F1 0.93).
- On **60 real field photos**: only **50% accuracy** — the model was trained on clean lab-style
  images, but farmers take messy whole-plant photos.
- The model is often *confidently wrong* (mean confidence 0.75 on mistakes) — so simply
  "not trusting low confidence" is **not enough**. We need to **crop the leaf first** and
  **fine-tune on field photos** (planned on Kaggle's free GPU).
- **Honest conclusion:** domain adaptation is required before we can promise ≥95% accuracy.

### 3.2 The dataset has a hidden "cheat code" (leakage)
- The friend's 10,000-row dataset has a column `risk_level` where **"Low" always means
  "Healthy"**. A model could cheat by reading that column instead of learning real patterns.
- **Action:** we must **drop `risk_level`** from the features before training the risk model.

### 3.3 Price model is honest but not perfect
- MandiLens price model: average error **₹521 per quintal** (about 10.5%), direction correct
  ~49% of the time. Published as-is with the reliability numbers attached — no overclaiming.

---

## 4. The knowledge base (rules) — what we built

We read **39 government/agricultural PDFs** (Agresco bulletins, Konkan DSS research report,
Marathi advisories, CIB&RC pesticide registrations) and turned them into structured rules.
Every rule carries its **source file + line number**.

### 4.1 Fertilizer (11 rules)
Zone-specific NPK doses for kharif rice across Konkan (South/North Coastal, rice-chickpea,
rice-sweet corn, organic rice-cowpea, DSR under mulch, rice-rice, rice-honeybee, rice-sugarcane),
Konkan Annapurna Briquettes 175 kg/ha, INM/biofertilizer.

### 4.2 Pesticide (19 rules) — all 5 foliar diseases now covered
| Disease | Recommended chemical(s) | Source |
|---|---|---|
| Sheath blight | Tricyclazole, Isoprothiolane, Hexaconazole | Paddy bulletin / Marathi advisory |
| Leaf blast | Hexaconazole | Marathi advisory |
| Bacterial leaf blight | Streptocycline + Copper oxychloride | Marathi advisory |
| **Brown spot** | **Ediphenphos, Azoxystrobin+Mancozeb, Picoxystrobin+Isoprothiolane, Carbendazim 5% GR** | **CIB&RC MUP 2026 (in-label)** |
| **Leaf scald** | **Mancozeb, Validamycin, Ediphenphos** | **IRRI actives + CIB&RC rice dose (off-label, clearly noted)** |

> **Brown spot** rules are **in-label** — the Indian pesticide authority (CIB&RC) explicitly lists
> these chemicals for brown spot on rice.
>
> **Leaf scald** has **no Indian label claim at all** (we checked the entire 2026 Major Uses of
> Pesticides — zero "scald" entries). So the scald rules use chemicals that the **International
> Rice Research Institute (IRRI)** recommends for scald, with the dose taken from the chemical's
> Indian rice registration for a related disease. Each scald rule carries a transparent
> **"off-label" note** so nobody is misled.

### 4.3 Weed control (4 rules)
Oxadiargyl, Metsulfuron-methyl + chloromuron-ethyl, Oxiflufen + 2,4-D, Pretilachlor.

### 4.4 IPM first (10 non-chemical actions)
Before any chemical, the system returns **Integrated Pest Management** actions: water management,
mechanical weeding (KKV cono weeder), green manuring, scouting thresholds, and resistant
varieties (BM-4, Ratnagiri-7, Karjat-184, Karjat-8). Chemicals are only a fallback.

### 4.5 Legality layer (CIB&RC registry, 22 entries)
Every chemical is checked against the official Indian registry:
- **Banned** (e.g., Dichlorvos) → **never recommended**, flagged for expert review.
- **Restricted** (e.g., Mancozeb — restricted only for Guava/Jowar/Tapioca, **not rice**) →
  shown with the restriction details.
- **Registered** → normal recommendation.
- **Not listed** → shown with a caution note.

---

## 5. How it works (architecture in one picture)

```
Farmer photo ──► Disease model (EfficientNet-B0) ──► 6-class label + confidence
                                                          │
                                                          ▼
Farmer context (district, soil, stage, rainfall) ──► Recommender (JSON rules + registry)
                                                          │
                                                          ▼
                                              IPM actions first, then cited chemicals
                                                          │
                                                          ▼
                                              Risk score + interval + advisory (EN/MR)
```

- **No training at runtime** — the recommender is pure rules + arithmetic (fast, transparent).
- **Every answer is cited** — you can open the source PDF and check.
- **No answer when unsure** — `abstain` + `needs_expert` instead of a guess.

---

## 6. Setup & how to run

### Requirements
- Python 3.10+ (we used 3.13)
- Free tools only: `pdfplumber` / `pdftotext`, `opencv-python`, `numpy`, `scikit-learn`
  (for the model training on Kaggle: TensorFlow/Keras)

### Quick start (local, light work only)
```bash
# 1. Validate the knowledge base + recommender (exit 0 = shippable)
python ml/evals/rule_validator.py

# 2. Smoke-test the recommender (all 5 diseases return cited chemicals)
python ml/evals/test_recommender.py

# 3. Rebuild the CIB&RC registry after editing rule chemicals
python kb/build_registry.py
```

### Where heavy work runs (NEVER on this laptop)
| Task | Where | Why |
|---|---|---|
| Model fine-tuning | **Kaggle GPU** (`ml/kaggle/field_finetune.py`) | Local machine has a thermal/heatsink issue — heavy compute risks hardware damage |
| Risk model fit | **Kaggle** (`ml/kaggle/fit_risk.py`) | seconds of compute |
| Light inference (1–2 images) | local, OK | tiny |

### API endpoints (for the backend teammate)
See [`ml/API_CONTRACTS.md`](ml/API_CONTRACTS.md) — 5 stateless endpoints:
`/ml/disease`, `/ml/recommend`, `/ml/risk`, `/ml/price`, `/ml/advisory`.
Every response includes `model_version` and `needs_expert`.

---

## 7. Safety rules (non-negotiable)

1. **No chemical without a source** — every rule cites `source_pdf` + `line`.
2. **No guess when unsure** — `abstain` + `needs_expert`.
3. **IPM first** — non-chemical actions before any pesticide.
4. **Banned chemicals are hard-gated** — removed + safety flag + expert review.
5. **Beginner-friendly language** — the farmer-facing chat must be simple.
6. **Never hallucinate** — every claim must be verifiable from a real file.

---

## 8. What's still open (next steps)

| Item | Where |
|---|---|
| Fill `DATASET_PATH` + field photos → run fine-tune | Kaggle |
| Fit `risk_artifact.json` (needs `risk_features.csv`) | Kaggle |
| Drop `risk_level` from risk-model features (leakage) | `ml/kaggle/fit_risk.py` |
| Parse DWE 2020-24 district rice tables | local |
| IMD daily rainfall for the risk anomaly | `ml/src` |
| MandiLens LICENSE check before public use | — |

---

## 9. Where is everything?

| File | What it is |
|---|---|
| [`ML_CONTEXT.md`](ML_CONTEXT.md) | **Full technical record** — everything we did, every finding, every rule |
| [`ML_PLAN.md`](ML_PLAN.md) | The 10-hour sprint plan + research contributions |
| [`kb/INDEX.md`](kb/INDEX.md) | What was extracted from which PDF |
| [`kb/rules/`](kb/rules/) | The actual rules (fert_pest.json, ipm.json, varieties.json, registry.json) |
| [`ml/src/`](ml/src/) | The Python modules (recommender, risk, price, advisory, crop, abstain) |
| [`ml/evals/`](ml/evals/) | Validator + tests + eval results |
| [`ml/API_CONTRACTS.md`](ml/API_CONTRACTS.md) | Teammate-facing API contracts |

---

*AgriSense — research-grade, source-grounded, honest-by-design. Built in a single-day sprint
with free tools only.*