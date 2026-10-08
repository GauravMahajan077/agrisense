# AgriSense ML → Backend API Contracts (v0.1)

All endpoints: **stateless**, JSON in/out. Every response carries `model_version` and `needs_expert`.
Backend owns DB/auth; ML never writes DB. Errors: `{"error": "...", "code": 4xx/5xx}`.

## POST /ml/disease
Request: `{"image_b64": "...", "field_context": {"crop": "paddy", "sowing_date": "YYYY-MM-DD", "district": "..."}}`
Response:
```json
{
  "label": "Leaf_Blast",
  "probs": {"Bacterial_Leaf_Blight": 0.01, "Brown_Spot": 0.04, "Healthy": 0.02,
            "Leaf_Blast": 0.79, "Leaf_Scald": 0.09, "Sheath_Blight": 0.05},
  "confidence": 0.79, "margin": 0.70,
  "needs_expert": false, "abstain_reason": null,
  "gradcam_url": "https://.../gradcam/<id>.png",
  "model_version": "b0-crop-v1", "preprocess_contract": "rgb-lanczos256-0to255"
}
```
Rules: `needs_expert=true` when confidence < threshold (calibrated) OR margin < 0.15 OR label=Healthy with conf<0.9.

## POST /ml/recommend
Request: `{"crop":"paddy","variety":"...","season":"kharif","stage":"tillering","district":"Ratnagiri","soil":{"n":..,"p":..,"k":..,"ph":..},"disease_label":"Brown_Spot"}`
Response:
```json
{
  "ipm": [{"item":"Scout fields; treat only at ETL: 5% dead hearts or 1 female moth/m2",
           "score":0.85, "detail":{"type":"monitoring", "...":"..."},
           "source":{"pdf":"1. Paddy.ocr","line":350}, "rationale":["disease matches IPM target"]}],
  "ipm_note": "Non-chemical actions first; chemical treatment only when scouting or economic threshold justifies (IPM).",
  "fertilizer": [{"item":"N100:P40:K40/ha","score":0.85,"detail":{...},"source":{"pdf":"Konkan_Rice_DSS_Research_Report","line":352},"rationale":["zone match","soil N 90 low -> 100 kg/ha dose ranked up"]}],
  "pesticide":  [{"item":"Hexaconazole 5% EC","score":0.7,"detail":{...},"source":{"pdf":"advisory-marathi.ocr","line":165},
                  "rationale":["disease matches KB target"],
                  "registry":{"authority":"CIB&RC (DPPQS, Govt. of India)","status":"registered",
                              "formulation_match":"exact","as_of":{"formulations":"2026-03-31", "...":"..."},
                              "rice_evidence":[{"pdf":"cibrc/mup_fungicides_31.03.2026","line":2544,"text":"Paddy (Rice) Blast ..."}]}}],
  "weed": [],
  "safety_flags": [{"item":"Dichlorvos 76 EC 0.05%","status":"banned",
                    "reason":"banned for use in India (manufacture continues for export only)",
                    "source":{"pdf":"cibrc/banned_refused_restricted","line":109},
                    "rule_source":{"pdf":"Joint Agresco-2018","line":252}}],
  "abstain": false, "coverage": 0.5, "needs_expert": false,
  "rationale": ["soil N low → basal N rank boosted", "rainfall anomaly +18% → blast risk up"],
  "model_version": "kb-v3"
}
```
Rules: `ipm` is ALWAYS the first key and renders above `pesticide` in the UI (IPM-first). NEVER return a chemical absent from KB; always cite source pdf+line; `abstain=true` if fertilizer/pesticide facets are uncovered (`needs_expert` additionally true when a disease was given but KB has no cited chemical OR any `safety_flags` are present). IPM actions must never contain a chemical name or dose.

Registry (CIB&RC legality layer, `kb/rules/registry.json`): every `pesticide`/`weed` payload carries a `registry` block when known — `status` ∈ `banned|restricted|registered|not_listed` with precedence `banned > restricted > registered > not_listed`, plus `formulation_match` (`exact|numeric|active_only|unverified|none`) and `rice_evidence` citations (Major Uses of Pesticides, 31.03.2026). **A `banned` chemical is hard-gated out of `pesticide`/`weed` and surfaced in `safety_flags` instead** (with the ban citation) — the UI should show the flag and `needs_expert=true`. `restricted` entries are NOT gated (restrictions are crop-specific, e.g. Oxyfluorfen is banned only on potato/groundnut); they carry `restriction_mentions_rice` + `restricted` lines so the UI can show context. Brands/prices are deliberately excluded (no authoritative free source).

## POST /ml/risk
Request: `{"district":"...", "lat":0,"lon":0,"crop_stage":"panicle","disease_probs":{...},"sowing_date":"..."}`
Response:
```json
{"score": 0.73, "interval": [0.61, 0.84], "priority": "high",
 "drivers": [{"feature":"rainfall_anomaly","value":"+22%"}, {"feature":"disease_prob","value":"Leaf_Blast 0.79"}],
 "model_version": "risk-v1"}
```

**XGBoost variant (`ml/risk_xgb/`, `model_version: "risk-xgb-v1"`)** — when
`ml/models/risk_xgb_artifact.json` + `risk_xgb_model.json` are present, the backend
calls `risk_xgb.predict()` with **raw field features** instead of the 5 engineered
`RiskInput` fields. Request adds the field context; `primary_disease_risk` is the
most likely class from `/ml/disease`:
```json
{"growth_stage":"Flowering","rice_variety_type":"Short Duration (Karjat-3)",
 "nitrogen_applied_level":"Excessive","primary_disease_risk":"Leaf Blast",
 "temperature_min":20.0,"temperature_max":28.0,"relative_humidity":92.0,
 "rainfall_7d_forecast":80.0,"consecutive_rainy_days":6,
 "field_water_level_cm":12.0,"soil_ph":5.8}
```
Response shape is unchanged (`score`, `interval`, `priority`, `drivers`,
`needs_expert`). `predict()` never raises: missing values → NaN (XGBoost native),
unseen categories → all-zero one-hot, out-of-range → clipped to training range,
degenerate input or missing artifact → linear-prior fallback
(`model_version: "risk-xgb-fallback-priors"`, `needs_expert: true`). Priority bands
match the linear engine: high ≥ 0.70, medium ≥ 0.45, low < 0.45. See
`ml/risk_xgb/README.md` for training + honest caveats (the 10k dataset is synthetic
and deterministic — R² 0.926 reflects the generator, not real-world generalization).

## POST /ml/price (MandiLens artifacts)
Request: `{"commodity":"rice","market":"...","state":"Maharashtra"}`
Response:
```json
{"observed": {"date":"2026-07-20","min":0,"modal":0,"max":0},
 "forecast": [{"date":"...","p50":0,"p80_lo":0,"p80_hi":0}],
 "signal": "wait", "reliability": {"wape": 0.1053, "directional_acc": 0.4915},
 "source": "AGMARKNET/GODL-India via MandiLens", "as_of": "2026-07-20"}
```
Rules: signal = `sell` if p50 trend next 5 days > modal*(1+ε) else `wait`; always show interval + reliability (honest uncertainty).

## POST /ml/advisory
Request: `{"lang":"mr|en","disease":{...},"recommend":{...},"risk":{...},"price":{...}}`
Response:
`{"text_en":"...","text_mr":"...","structured_refs":["/ml/disease/.."],"expert_pending":true|false,"model_version":"adv-tpl-v1"}`
Rules: deterministic templates only (no free-form generation); expert must approve before dispatch (`expert_pending=true` default).
