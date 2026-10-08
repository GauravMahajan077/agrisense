# AgriSense ML — teammate-facing index

**Contracts (for backend):** `API_CONTRACTS.md` — 5 endpoints, JSON schemas, `needs_expert` rules.
**Plan:** `../ML_PLAN.md` · **KB:** `../kb/INDEX.md` · **Eval numbers:** `evals/results.md`

## What runs where
| Thing | Where | Note |
|---|---|---|
| Training / fine-tune (`kaggle/field_finetune.py`) | **Kaggle GPU only** | NEVER local (thermal damage constraint) |
| Risk artifact fit (`kaggle/fit_risk.py`) | Kaggle CPU/GPU | seconds |
| Inference of `.tflite` model | Backend server / Kaggle | local machine = light tests only (1–2 images max) |
| KB extraction (pdfplumber/pdftotext) | local, OK | already done → `../kb/raw/` |
| Recommender / risk / advisory / price (pure Python) | backend | no ML runtime, just JSON rules + arithmetic |

## Modules (`src/`)
- `crop_leaf.py` — OpenCV leaf crop → fixes field-photo domain shift (tested).
- `abstain.py` — confidence/margin/healthy thresholds → `needs_expert` routing (tested).
- `recommender.py` — KB rule filter + weighted rank, **IPM-first** (non-chemical actions returned before any pesticide), **strict** disease↔chemical matching, **CIB&RC banned-chemical hard gate** (`safety_flags` + `needs_expert` instead of a banned product), per-item `registry` legality block (tested; all 5 foliar classes covered — Brown_Spot: 4 in-label MUP rules, Leaf_Scald: 3 off-label IRRI-actives rules).
- `risk_engine.py` — sigmoid score + isotonic calib + conformal interval via `models/risk_artifact.json` (fallback priors built-in).
- `price.py` — MandiLens published snapshot → `/ml/price` shape, honest `sell|wait` signal + reliability.
- `advisory.py` — deterministic EN/MR templates, always `expert_pending=true`.

## Kaggle run order (today)
1. Upload: `agrisense_bundle.zip`, dataset folder, field photos, this `ml/` folder.
2. `kaggle/field_finetune.py` → new bundle + `abstain.json` + `field_probs.csv` (EDIT the 3 CFG paths first).
3. `kaggle/fit_risk.py` (needs `risk_features.csv`; build from backend logs or manual annotation).
   Future: train XGBoost risk model on `../konkan_rice_disease_risk_10k.csv.xls` (10k rows, validated —
   but drop `risk_level` from features: it leaks the healthy label, see validator WARN).
4. Download artifacts → backend consumes.

## Inference contract (do not break)
RGB → Lanczos resize **256×256** → **float 0..255** (never /255) → model resizes 224.
Classes (ordered): `Bacterial_Leaf_Blight, Brown_Spot, Healthy, Leaf_Blast, Leaf_Scald, Sheath_Blight`.

## Safety rules baked in
- No chemical leaves the recommender without `source_pdf` + line in `kb/rules/`.
- IPM-first: `ipm` key renders above `pesticide`; IPM actions must be chemical-free (validator enforces).
- No KB match → `abstain + needs_expert`, never a guess.
- **Banned chemicals are hard-gated** (CIB&RC registry: `banned > restricted > registered > not_listed`) —
  they are removed from `pesticide`/`weed`, reported in `safety_flags` with the ban citation, and force
  `needs_expert`. Rebuild after rule edits: `python ../kb/build_registry.py`.
- All advisories default `expert_pending=true`.
- MandiLens reliability metrics always attached to price responses.
- Local machine: no heavy compute — Kaggle only (hardware constraint, always).

## Validation (run before every handoff)
`python evals/rule_validator.py` — 440+ checks across KB schema, citation drift,
recommender safety (leakage/abstention/fuzz), agronomic domain rules, dataset quality,
registry schema + banned-gate behavior (section F). Exit code 0 = safe to ship;
`--update-snapshot` after intentional KB edits.