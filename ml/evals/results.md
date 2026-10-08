# ML Eval Results (sprint day)

## C1 — Disease diagnosis

### Field stress test (60 real photos, current `agrisense_b0` model, no crop)
| Metric | Value |
|---|---|
| Base accuracy | **0.500** (30/60) |
| Mean conf on correct | 0.889 |
| Mean conf on wrong | 0.751 |
| Confident-wrongs (conf ≥ 0.8) | **13** |
| Top confusions | Leaf_Blast→Sheath_Blight (9), Leaf_Blast→Brown_Spot (5), Leaf_Blast→Healthy (4) |

### Risk-coverage curve from existing predictions (`field_abstention.json`)
| conf threshold | coverage | acc on accepted |
|---|---|---|
| 0.50 | 1.000 | 0.483 |
| 0.60 | 0.917 | 0.509 |
| 0.70 | 0.717 | 0.581 |
| 0.80 | 0.600 | 0.639 |
| 0.90 | 0.517 | 0.645 |

**Honest conclusion:** abstention ALONE cannot reach the ≥95% accepted-accuracy
target on this model — errors are *confident* (mean 0.75), so the curve is flat.
Domain adaptation (crop + head finetune on Kaggle, `ml/kaggle/field_finetune.py`)
is **required first**; abstention then layers on top. This is the C1 research story:
*domain shift breaks confidence-based safety — crop preprocessing + adaptation
restores it.*

### Benchmarks (held-out, same-distribution)
val macro-F1 **0.9488** · test macro-F1 **0.9315** · worst-class recall 0.8804 · TFLite≡Keras 100%.

### Targets after Kaggle run
field acc ≥ **0.75**; conf-wrongs routed to expert ≤ **2**; accepted-set acc ≥ 0.95.

## C2 — Fertilizer/pesticide recommender (KB)
| Check | Status |
|---|---|
| Fertilizer rules (South/North Konkan NPK, briquettes) | ✅ 3 rules, all cited |
| Pesticide rules (stem borer, case worm, botanicals) | ✅ 3 rules, all cited |
| Strict disease→chemical matching (stopwords) | ✅ tested: all 5 foliar classes return cited chemicals; no cross-class leakage (BLB↛sheath/blast, Brown_Spot↛sheath/blast, Leaf_Scald↛BLB/blast) |
| False-match safety test | ✅ Bacterial_Leaf_Blight no longer matches "leaf folder" insecticide |
| Leave-one-year-out hit@3 | ⏳ needs more year-coverage in rules |
| Foliar-disease chemicals | ✅ Brown_Spot: 4 in-label MUP rules (Ediphenphos, Azoxystrobin+Mancozeb, Picoxystrobin+Isoprothiolane, Carbendazim 5% GR); Leaf_Scald: 3 off-label IRRI-actives rules (Mancozeb, Validamycin, Ediphenphos) with transparent off-label notes |

## C3 — Risk engine
- Artifact-less fallback (priors) verified: disease 0.79, rain anomaly +25%, panicle stage → score **0.501**, interval [0.401, 0.601], band `medium`.
- ⏳ Fit calibration + conformal q on Kaggle (`ml/kaggle/fit_risk.py`, next).

## Price — MandiLens (published, locked holdout)
MAE **₹521.20**/quintal · WAPE **10.53%** · directional acc **49.15%** · interval coverage 71.6% vs 80% target (published as-is, honest). AgriSense `wait/sell` signal wired with reliability attached (`ml/src/price.py`).

## Advisory
EN + MR deterministic templates verified; `expert_pending=true` always; refs to `/ml/*` present.