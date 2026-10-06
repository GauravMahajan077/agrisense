# CONTEXT — Agrisense (read this first in any new session)

Session state file. Updated at the end of every phase. Committed to git.

## What this project is
Rice-leaf disease classification for Kaggle (TF 2.20 / Keras 3, 2x T4). Full rules live in
[`AGENTS.md`](AGENTS.md) — that file is authoritative for *how* to work here.

- **Source of truth:** `agri/agrisense_kaggle.py` — 23 `# %%` cells (1 markdown + 22 code), 1,955 lines.
- **Generated output:** `agri/cells/*` — produced by `python split_cells.py`, never hand-edited.
- **`agri/split_cells.py`** — generator *and* round-trip verifier (every non-empty line of every cell
  must appear in the source, in order). This is what keeps source ↔ cells from drifting.
- **`agri/verify.py`** — the only test that reads the REAL source and the REAL generated cells.
  Run: `python -B verify.py`.

## Decisions already made (do not relitigate)
1. **`agrisense.py` + 3-cell notebook restructure is DEFERRED** until Phases 0–4 pass.
   1,824 lines of shared globals cannot be validated without a Kaggle run. `PIPELINE_VERSION`
   (Phase 1/2) already solves the stale-paste problem on the Kaggle side.
2. **All logic unit tests were deleted** (see cleanup below) because every one tested a
   *hand-copied re-implementation*, not the shipped source. Proof: `test_cell5_sweep.py` contained
   its own `clusters_at` with the `return cid, 0` fix, so the suite reported ALL PASS while
   `agrisense_kaggle.py` line 783 still crashed. False confidence is worse than no tests.
   Validation is now `split_cells.py` (round-trip) + `verify.py` (source checks).
3. **`.opencode - Copy/` is kept deliberately** (personal backup, 159.6 MB, gitignored — do not delete).
4. If a phase genuinely needs a unit test (e.g. the new brute-force dedupe), ASK before creating a file.

## Verified audit findings (reproduced locally, TF 2.21 / Keras 3.13.2)
| Claim | Verdict |
|---|---|
| `clusters_at` returns bare array at `nd=0` | **REAL CRASH** — `ValueError: too many values to unpack` at source lines 905 and 924. Default `"sweep": [0,2,4,6,7]` contains 0, so Cell 5 crashes every run. |
| Cell 17 `.resize((SIZE,SIZE), np.float32)[None] / 255.` | **REAL CRASH ×2** — `ValueError: Unknown resampling filter (<class 'numpy.float32'>)` and `TypeError: 'Image' object is not subscriptable`; plus `/255.` feeds 0–1 to a model expecting 0–255. |
| Cell 19 dynamic `Input((None,None,3))` | **REAL** — TFLite reports `[1 1 1 3]`, so line 1773 picks `(1,1)` and parity runs on 1×1 images. |
| `amp: True` → `mixed_float16` → TFLite export | **WORSE THAN AUDIT** — conversion raises `ConverterError` ('tf.Conv2D' op is neither a custom op nor a flex op) for fixed *and* dynamic input. Also: `set_global_policy("float32")` after training does **not** change already-built layers (policies are baked in) → the export graph must be **rebuilt** under float32, not re-wrapped. |
| Cell 15 `model.predict((x,y))` crashes | **NOT REPRODUCED** on Keras 3.13.2 — works, correct shape. Do not "fix" a non-bug; the dataset-level `predict_idx` is still adopted for speed + length assert. |
| `slice` loses static shape in Cell 10 | **NOT REPRODUCED** — shape is already `(224,224,3)` after `resize`; `set_shape` is a harmless no-op guard. |

## Other confirmed defects (not from the audit)
- `verify.py` line 62 asserts `"key PAIRS merged"` but the source prints `"key pairs merged"` → `verify.py` exits 1.
- Hygiene (all fixed in Phase 2, commit `f01e9f3`): pip installs unpinned, crawler has no byte cap
  (`.content`), User-Agent says `contact: set-your-email-here`, `multi_gpu: True` default,
  no `smoke` mode, no `PIPELINE_VERSION`.

## Phase status
| Phase | Status |
|---|---|
| **0** — `AGENTS.md` + `git init` + baseline commit | ✅ DONE (commit `f9c8a74`) |
| **cleanup** — delete false-confidence tests, temp orphans | ✅ DONE (commit `a7d823c`) |
| **1** — P0 crash fixes | ✅ DONE (commit `d9492b7`) |
| **2** — P1 simplify (remove sweep/LSH/cache/`effective`/`oversample`/finetune-crawl, `multi_gpu: False`, smoke mode, `PIPELINE_VERSION`, pin pip, UA email, crawl byte cap) | ✅ DONE (commit `f01e9f3`) |
| **3** — P2 honest eval (8-variant D4 brute force, same-class merge only, cross-class dropped from val/test, source-held-out split, macro-F1 per source + bootstrap CI, abstain rule) | ✅ DONE (commit `ab04f7d`) |
| **4** — P3 field-test loader (eval only) | ⬜ NEXT |
| restructure → `agrisense.py` + 3-cell notebook | ⬜ deferred until 0–4 pass |

## Phase 1 — what was actually done (commit `d9492b7`)
1. `clusters_at`: `if not nd: return cid` → `return cid, 0` (fixes `ValueError: too many values to unpack` at lines 905/924).
2. Cell 15 `predict_idx`: predict on the whole dataset once + `assert len(p) == n`.
3. Cell 17: `np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))[None]`, **removed `/255.`**.
4. Cell 10: `h.set_shape([SIZE, SIZE, 3])` guards at the end of `_aug` and `_eval_t`.
5. Cell 18/19 export: rebuild `clean` under `float32` policy (weight transfer guarded by asserts),
   fixed `tf.keras.Input((PRE, PRE, 3), batch_size=1)`, parity reads the interpreter's real
   allocated shape, contract JSON updated to `(1, 256, 256, 3)`.
   **New finding during validation:** the `Resizing` layer must ALSO be built under the float32
   policy — building it after restoring `mixed_float16` gives it an f16 compute policy →
   `tf.ResizeBilinear` on f16 → `ConverterError` ("neither a custom op nor a flex op"). The
   policy restore now happens only after `clean` exists.
6. Cell 19 parity: `k_out = clean.predict(...)` (was `model.predict` → crashed on 256px probe),
   and the interpreter's input shape is asserted to be exactly `(PRE, PRE)` instead of the old
   silent `(1,1)` fallback.

**Validation (all real, pasted in session):**
- `split_cells.py` round-trip + `verify.py` → ALL PASS, exit 0.
- AST-extracted shipped-source tests (no files created, cannot go stale like the deleted tests):
  `clusters_at` unpacking/monotonicity, Cell 17 preprocess shape/dtype/0-255, `_aug`/`_eval_t`
  static shapes, `predict_idx` length assert — ALL PASS.
- Full Cell 18 + Cell 19 end-to-end on the REAL source (real `build_model` under `mixed_float16`,
  real export, real TFLite conversion): every layer float32, TFLite input `[1,256,256,3]`,
  **PARITY 100% at 256x256, max |delta prob| = 0.0000**, bundle written. ALL PASS, exit 0.

**Verification limits:** local = TF 2.21 / Keras 3.13.2 / Python 3.13. Kaggle = TF 2.20.
Local green ≠ Kaggle green. A Kaggle smoke run must be pasted before claiming Phase 1 works there.

## Phase 2 — what was actually done (commit `f01e9f3`)
1. **CFG:** `multi_gpu: False` (single GPU default), `smoke: False`, dedupe reduced to
   `{enable, dihedral, near_dist, bands}` (sweep + rehash removed), imbalance reduced to
   `{mode: "class_weight"}` (effective/oversample removed), crawl `role: "stress_test"` only +
   `max_bytes: 5_000_000`.
2. **Cell 2:** `_pip` now takes `(import_name, install_spec)` tuples; pins `imagehash==4.3.2`,
   `ddgs==9.16.0` (both verified to exist on PyPI).
3. **Cell 4:** smoke caps manifest to 40 img/class (`man.groupby("class").head(40)`).
4. **Cell 5:** hash cache removed (fresh hash every run — no stale-key risk); sweep block removed;
   single `near_dist` merge via `clusters_at` (banded merge kept until Phase 3's brute force).
5. **Cell 8:** `assert mode == "class_weight"`; plain `CategoricalCrossentropy` LOSS.
6. **Cell 9/10:** oversample/`rep_per_img` removed; `gen_train` simplified in RAM + disk paths.
7. **Cell 14:** `cw` always class_weight; `epochs = 1 if CFG["smoke"] else st["epochs"]`;
   `PIPELINE_VERSION` printed.
8. **Cell 17:** UA contact → `gau.mah077@gmail.com`; `_fetch_bytes(url, max_bytes)` streaming cap
   replaces `.content`; smoke skips crawl; `assert role == "stress_test"` (finetune removed).
9. **Cell 18/19:** export + TFLite/bundle wrapped in `if not CFG["smoke"]:` (else prints
   "SMOKE: export/bundle skipped"); `PIPELINE_VERSION` printed at both entry points.
10. **`PIPELINE_VERSION = "2.0.0"`** defined in Cell 0, printed at start/train/export/bundle.
11. **README.md** updated: multi_gpu default, imbalance section, finetune role, troubleshooting
    row, smoke-mode note.

**Validation (all real, pasted in session):**
- `split_cells.py` round-trip + `verify.py` → ALL PASS, exit 0.
- AST-extracted shipped-source checks (40 assertions, temp file outside repo): CFG shape,
  removed-feature absence (`rep_per_img`, `effective_beta`, `oversample_cap`, `set-your-email`,
  `"sweep":`, `"rehash":`, `canon_keys.csv`), pip pins, smoke guards in Cells 4/14/17/18/19,
  `_fetch_bytes` defined+used, role assert — ALL PASS, exit 0.
- `clean`/`parity` referenced only inside the non-smoke branches (grep-verified).

**Verification limits:** same as Phase 1 — local TF 2.21 ≠ Kaggle TF 2.20. Smoke mode exists
specifically so the user can paste a fast Kaggle run; that paste is the Phase 2 acceptance gate.

## Phase 3 — what was actually done (commit `ab04f7d`)
1. **CFG:** `held_out_source: "dedeikh"` (train anshul6+indo3, test dedeikh, 5 shared classes),
   `abstain_threshold: 0.5`, `bootstrap_iters: 2000`; dedupe reduced to `{enable, dihedral,
   near_dist}` (bands removed — brute force needs no banding).
2. **Cell 4:** routes `held_out_source` rows into `man_held` before dedupe/split; `man_held = None`
   when unset; WARNING if the source is absent from the manifest.
3. **Cell 5:** `_canon_key` → `_d4_keys` (all 8 D4-variant pHash ints, not the min);
   `clusters_at` (banded LSH) → `d4_dist_matrix` (full (n,n) uint8 min-over-64-variant-pairs
   Hamming matrix, chunked 256, `np.bitwise_count`) + `brute_clusters` (union-find over
   **same-class** pairs only) + `cross_class_mask` (images within `near_dist` of a different-class
   image); `leak_scan` now takes `D` directly (no re-hash, no banding); cross-class diagnostics
   rewritten as a `source:class <-> source:class` pair table from `cross_mask` (keeps the
   `"check the source_alias mapping"` flag).
4. **Cell 6:** cross-class near-duplicates moved val/test → train (unreliable labels, never scored);
   leak scan call updated to `leak_scan(D, man, tight=TIGHT)`.
5. **Cell 15:** `report(cm, title, classes=None)` generalized (k = len(classes), default CLASSES).
6. **Cell 16.5 (new):** source-held-out eval — preprocess like Cell 17 (0-255 float, SIZE resize),
   `model.predict`, restrict to shared classes, `report` with classes, bootstrap 95% CI on macro-F1,
   abstain at `abstain_threshold` (coverage + acc + macro-F1 on covered), per-source macro-F1 on
   both held-out and in-source test, writes `source_held_out.csv`.
7. **Cell 17:** abstain line added to field stress test output.
8. **verify.py:** 23 cells / 22 boundaries / 22 code cells; Phase 3 markers for Cells 1/4/5/6/16.5/17.
9. **`agri/test_phase3_dedupe.py`** (user-approved): AST-extracts the SHIPPED `d4_dist_matrix`,
   `brute_clusters`, `cross_class_mask`, `leak_scan`; 16 checks — identity/symmetry/chunking,
   D4-min over variant orbits (0 and 1-bit cases), same-class-only merging, cross-class flagging,
   leak detection — ALL PASS.

**Validation (all real, pasted in session):**
- `split_cells.py` round-trip + `verify.py` → ALL PASS, exit 0 (23 cells).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0.
- Grep: no leftover `clusters_at`/`_canon_key`/`bands`/`leak_scan(man, keys` references.
- AST name check: every Cell 16.5 dependency (`confusion`, `report`, `te`, `yte`, `yte_pred`,
  `SIZE`, `Image`, `OUT`, `CFG`, `CLASSES`, `NC`, `model`, `man_held`) is defined in the source.

**Verification limits:** same as Phases 1–2 — local TF 2.21 ≠ Kaggle TF 2.20. The Phase 3
acceptance gate is a user-pasted Kaggle smoke run showing the new Cell 5 dedupe output, the
Cell 6 cross-class move, and the Cell 16.5 held-out eval.

## Headline metric (the number that matters)
Source-held-out: train on `anshul6` + `indo3`, test on `dedeikh` across the 5 shared classes
(`Sheath_Blight` has no second source → excluded). In-source test F1 is secondary and expected
to be much higher. **Built in Phase 3** (Cell 16.5, commit `ab04f7d`).
