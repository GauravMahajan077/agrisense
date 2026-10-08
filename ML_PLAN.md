# AgriSense ML — 10-Hour Sprint Plan (research-grade DSS, ML only)

Status: in_progress | Scope: ML only (frontend/backend = teammate) | Mode: 10-hour single-day sprint

## 0. Hard constraints (ALWAYS)
- **Local machine is OFF-LIMITS for heavy work.** No training, no TF/PyTorch model loading, no big OCR/vision loops. Thermal/heatsink issue → serious hardware damage risk. Local = text extraction + tiny inference only.
- **All training / heavy compute runs on Kaggle notebooks (free GPU T4).**
- **Free tools only:** TF/Keras, ultralytics (AGPL), scikit-learn, OpenCV, pdfplumber/pdftotext. No paid APIs.
- Every action goes through Gaurav for approve/reject.

## 1. Assets
| Asset | Status |
|---|---|
| `agrisense_bundle.zip` → `agrisense_b0.keras` (EfficientNet-B0), class_names.json, .tflite | ✅ val macro-F1 **0.9488**, test **0.9315**, worst-class recall 0.8804, TFLite≡Keras 100% |
| Field stress test (60 real photos) | ⚠️ acc **0.50**; 13 confident-wrongs (conf≥0.8); main errors: Leaf_Blast→Sheath_Blight, Leaf_Blast→Healthy → **domain shift, not a bad model** |
| `newpaddyprofessional.ipynb` (Kaggle 3-cell wrapper: CFG → MODULE → RUN) | ✅ training entry point |
| Dataset (labeled paddy classes) | ⚠️ `DATASET_PATH = ___` (fill) — insufficient per class, poor quality (PaddyDoctor-style: needs crop) |
| Price model = **MandiLens** (github.com/heybadrinath/MandiLens) | ✅ AGMARKNET pipeline; published JSON artifacts; 189 series / 120 markets; MAE ₹521, WAPE 10.5%; ⚠️ no LICENSE file |
| IMD rainfall climatology `rf_p25_{jun..sep}_clm.nc` (0.25°, 30-yr), `rf_1deg_*` | ✅ risk-engine climate baseline |
| Domain PDFs + screenshots (56 files in `media/`) | ✅ knowledge-base source |

### Preprocessing contract (from class_names.json — violations break predictions)
RGB decode → Lanczos resize to **exactly 256×256** → emit float32 **0..255** → graph resizes to 224.
**Never:** /255 normalization, nearest-neighbour resize, any other HxW, BGR. abstain_threshold = 0.5.

## 2. Three research contributions
- **C1 — Selective disease diagnosis:** crop-before-classify + calibration + abstention (`needs_expert`) + Grad-CAM. Field acc target: 0.50 → ≥0.75.
- **C2 — Source-grounded fertilizer/pesticide recommender:** KB extracted from the PDFs → rule-filter + transparent weighted rank + citations + abstain. Zero training.
- **C3 — Calibrated risk engine:** disease logits/conf + IMD rainfall anomaly + phenology + DWE yield stats → isotonic-calibrated score + conformal interval.

## 3. Sprint schedule (single day)
| Block | Work | Where |
|---|---|---|
| H1 | This plan + `ml/` skeleton + API contracts doc for teammate | local (light) |
| H2 | **KB extraction pass 1:** all `media/*.pdf` → `kb/raw/*.txt` (pdftotext/pdfplumber, sequential, low priority) | local (light) |
| H3 | **KB pass 2:** rice/paddy-relevant rules → `kb/rules/*.json`, DWE/rainfall tables → `kb/tables/*.csv` | local (light) |
| H4 | Crop experiment: OpenCV leaf-crop script + report of delta on field images | local (script) / Kaggle (run) |
| H5 | **Kaggle notebook #1 (ready-to-upload):** crop pipeline (OpenCV + optional YOLOv8n) + head retrain/fine-tune with class-balanced sampling + TTA | Kaggle |
| H6 | Abstention/calibration module + Grad-CAM (TFLite path) | local (light inference only if tested on 1-2 imgs) |
| H7 | `recommender.py` + `risk_engine.py` + unit tests (pure functions) | local |
| H8 | MandiLens integration contract + price artifact consumption | local |
| H9 | Advisory NLG templates EN/MR | local |
| H10 | Eval report (`evals/results.md`) + packaging + README for teammate | local |

## 4. Disease fix strategy (C1) — ordered by ROI
1. **Preprocessing-contract unit test** on the 60 field images (resample to 256-Lanczos-0..255) — free points if current inference deviates.
2. **Crop-before-classify:** classical OpenCV leaf saliency crop (HSV green mask → largest contour → pad) first — zero training; YOLOv8n leaf detector on Kaggle only if crop delta < target. User photo note: whole-plant shots, lesion needs crop → biggest known lever.
3. **Abstention tuning:** use conf+margin (already in field_stress_test.csv) → risk-coverage curve; set threshold so confident-wrongs ≤2 and accepted-precision ≥95%; below threshold → expert queue. This is the honest-DSS research angle.
4. **Field adaptation (Kaggle, ≤20 min):** rebalance + crop-augmented PaddyDoctor data, fine-tune head (or embedding-kNN second opinion); evaluate on held-out field images.
5. **Data collection protocol:** +50 field photos (10/class) with crop framing → single biggest accuracy lever (document in README).

## 5. Knowledge base (C2) — `kb/` layout
```
kb/
  raw/          # pdftotext output per PDF (source of truth, never edited)
  rules/        # structured JSON records, every record carries {source_pdf, page}
  tables/       # DWE 2020-24, rainfall departures, soil/rainfall stats as CSV
  images/       # OCR text from advisory screenshots (if needed, Kaggle-side)
  INDEX.md      # what was extracted from which file, coverage status
```
Rule schema (fertilizer/pesticide):
`{id, crop, variety?, season, stage, district?, soil_range?, npk_urea, pesticide, dose, frequency, ph_interval, notes, source_pdf, page, confidence}`
Priority sources: Agresco 2018-23 (MPKV), `1. Paddy`, `Rice variety Dr.BSKKV Dapoli`, `Konkan_Rice_DSS_Research_Report`, `advisory-marathi`, IMD agromet screenshots.
Inference: filter (crop+stage+season+soil) → weighted rank (soil fit, climate anomaly, disease match) → top-3 + citations + `abstain` when coverage < threshold. Eval: leave-one-year-out hit@3 + 15-case expert agreement (κ).

## 6. Risk engine (C3)
Features: disease probs + confidence, rainfall anomaly (current IMD vs `rf_p25_*` climatology), growth stage from sowing date, district DWE yield percentile (2020-24).
Model: logistic/GBM + isotonic calibration + conformal interval; fit on Kaggle (seconds), ship as JSON coefficients/artifact. Output: `{score, interval[lo,hi], drivers[], priority}`.

## 7. Price (MandiLens)
Consume published artifacts (static JSON — no model server, no local training): observed prices, market comparison, 7-day outlook with intervals, reliability context. AgriSense contract: `{commodity, market, price_observed, forecast: [{date, p50, p80_lo, p80_hi}], signal: sell|wait, source_quality, as_of}`. ⚠️ Add LICENSE check before shipping; attribute AGMARKNET/GODL.

## 8. API contracts (teammate-facing) — detail in `ml/API_CONTRACTS.md`
`POST /ml/disease`, `POST /ml/recommend`, `POST /ml/risk`, `POST /ml/price`, `POST /ml/advisory` — all stateless, every response has `model_version` + `needs_expert`.

## 9. Evaluation (what we report)
| Piece | Metric | Protocol | Target |
|---|---|---|---|
| C1 | field acc, AURC, coverage@95%prec, ECE | 60 field imgs, base vs crop vs crop+TTA vs abstain | field acc ≥0.75; conf-wrongs into abstain |
| C2 | hit@3, MRR, coverage, expert κ | leave-one-year-out (2018-21→2022-23) | hit@3 ≥0.8, κ ≥0.6 |
| C3 | AUCPR, Brier, interval coverage | temporal split | 90–95% coverage |
| Price | MandiLens published: MAE ₹521 / WAPE 10.53% / dir-acc 49.15% (honest) | their locked holdout | documented as-is |
| Advisory | expert accept rate | 10 samples | ≥8/10 |

## 10. Non-goals
No MLOps infra, no LLM training, no satellite/drone pipeline, no IoT firmware, no from-scratch CNN retrain (unless all cheap levers fail), no local heavy compute — ever.

## 11. Open TODOs
- [x] KB extraction: all 39 PDFs → `kb/raw/*.txt` + structured rules (`kb/rules/`) + rainfall table
- [x] Crop + abstention modules (`ml/src/crop_leaf.py`, `ml/src/abstain.py`) — unit tested
- [x] Kaggle notebook script (`ml/kaggle/field_finetune.py`) + risk fit (`fit_risk.py`)
- [x] Recommender/risk/price/advisory modules + API contracts + eval numbers (`ml/evals/results.md`)
- [ ] Fill `DATASET_PATH` + field photos → run `field_finetune.py` **on Kaggle**
- [ ] OCR Marathi PDFs (`1. Paddy`, `advisory-marathi`) on Kaggle → unlocks foliar-disease chemicals in KB
- [ ] Fit `risk_artifact.json` on Kaggle (needs `risk_features.csv`)
- [ ] MandiLens LICENSE check before public use
- [ ] Recover old `production_hybrid.keras` (superseded by b0 bundle — optional)
