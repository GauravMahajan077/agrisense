# CONTEXT — Agrisense (read this first in any new session)

Session state file. Updated at the end of every phase. Committed to git.

## What this project is
Rice-leaf disease classification for Kaggle (TF 2.20 / Keras 3, 2x T4). Full rules live in
[`AGENTS.md`](AGENTS.md) — that file is authoritative for *how* to work here.

- **Module (source of truth):** `agri/new/agrisense.py` — `Pipeline` class, 21 stage methods, `run(cfg)`.
- **Notebook wrapper:** `agri/new/agrisense_notebook.py` — 3 cells (CONFIG → MODULE → RUN); Cell 2 is
  generated from the module via the `# %% include:agrisense.py` directive.
- **Legacy:** `agri/old/agrisense_kaggle.py` — 23 `# %%` cells (1 markdown + 22 code), kept on purpose.
- **Generated output:** `agri/new/cells/*` — produced by `python split_cells.py`, never hand-edited.
- **`agri/new/split_cells.py`** — generator *and* round-trip verifier (resolves the include directive;
  every non-empty line of every cell must appear in the resolved source, in order).
- **`agri/new/verify.py`** — the only test that reads the REAL source and the REAL generated cells.
  Run from `agri/new/`: `python -B verify.py`.

## Decisions already made (do not relitigate)
1. **`agrisense.py` + 3-cell notebook restructure is DONE** (Phase 5). The legacy
   `old/agrisense_kaggle.py` is kept on purpose (user decision). The notebook is self-contained:
   Cell 2 = module source via the include directive, no dataset upload, no re-upload on changes.
2. **All logic unit tests were deleted** (see cleanup below) because every one tested a
   *hand-copied re-implementation*, not the shipped source. Proof: `test_cell5_sweep.py` contained
   its own `clusters_at` with the `return cid, 0` fix, so the suite reported ALL PASS while
   `old/agrisense_kaggle.py` line 783 still crashed. False confidence is worse than no tests.
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
| **4** — P3 field-test loader (eval only) | ✅ DONE (commit `08bf8a2`) |
| **5** — restructure → `agrisense.py` + 3-cell notebook | ✅ DONE (commit `eec8434`) |

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
3. **Cell 4:** smoke caps manifest to 40 img/source/class (`man.groupby(["source", "class"]).head(40)`).
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
9. **`agri/new/test_phase3_dedupe.py`** (user-approved): AST-extracts the SHIPPED `d4_dist_matrix`,
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

## Phase 4 — what was actually done (commit `08bf8a2`)
1. **`agri/new/field_test.py` (new, standalone):** eval-only loader for real field photos. numpy +
   PIL + TFLite interpreter (tf.lite.Interpreter with tflite_runtime fallback). Loads
   `model.tflite` + `class_names.json` from `agrisense_bundle.zip` (or direct paths), applies
   the exact Cell 18 preprocessing contract (RGB → LANCZOS resize to `recommended_input_size`
   256×256 → float32 0–255, **no /255**). Photo discovery: subfolder-per-class or `--labels`
   CSV, unknown classes skipped with a warning, `--limit` caps per class. Report: per-class
   P/R/F1, macro-F1, accuracy, worst-class recall, bootstrap 95% CI, abstain coverage +
   acc/macro-F1 on covered. Writes `predictions.csv` + `confusion.csv`. CLI under
   `if __name__ == "__main__"` so it is importable. Never trains, never crawls.
2. **Cell 18:** contract dict now ships `"abstain_threshold": CFG["abstain_threshold"]` (the
   loader reads it; `--abstain` overrides).
3. **verify.py:** Phase 4 markers — `field_test.py` exists/parses, never `/255`, uses LANCZOS,
   reads the contract, has abstain; Cell 18 contract ships `abstain_threshold`.
4. **`agri/new/test_phase4_field_test.py`** (user-approved): imports the SHIPPED loader directly.
   Preprocessing contract (shape/dtype/0-255/no-resize on 256×256), photo discovery (subfolder,
   CSV, limit, unknown filter), metrics (macro-F1 = 11/15, acc 0.75, worst recall 5/8), abstain
   (coverage 0.75, acc 1.0), bootstrap CI ordering, and an end-to-end run on a tiny locally
   built TFLite model (4 synthetic images, both CSVs written).

**Validation (all real, pasted in session):**
- `split_cells.py` round-trip + `verify.py` → ALL PASS, exit 0 (23 cells).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0 (regression).
- `python -B test_phase4_field_test.py` → ALL PASS, exit 0 (16 checks incl. real TFLite e2e).
- Two bugs caught by the tests and fixed before commit: (a) JPEG is lossy → the 256×256
  no-resize check now uses PNG; (b) `run_eval` passed argmax indices to `abstain_report`
  instead of the full probability matrix → now collects `probs` and passes them.

**Verification limits:** the loader itself is fully testable locally (TF 2.21 + PIL installed)
and was tested end-to-end on a real TFLite model. The only thing not verified locally is the
actual `agrisense_bundle.zip` from a Kaggle run — that needs the Phase 2/3 smoke-run paste.

## Phase 5 — what was actually done (restructure, commit `eec8434`)
1. **`agri/new/agrisense.py` (new, source of truth):** all logic in a `Pipeline` class — 21 stage
   methods (1:1 with the legacy 23-cell notebook) + `run(cfg)` entry point that prints
   `PIPELINE_VERSION` and calls the stages in order. **No `__main__` pipeline block** (in a
   notebook `__name__ == "__main__"`, so a block there would fire on paste). Module-level
   `DEFAULT_CFG` (the notebook's CONFIG cell is the editable copy; verify.py AST-compares the
   two). `PIPELINE_VERSION = "3.1.0"`.
2. **Behavior-preservation details:** `SaveBestF1.best` stays a CLASS attribute (one global
   best across stage callbacks); `BEST_PATH` and `BudgetStop`'s `t_train0`/`budget_min` became
   constructor args (no module globals); `imagehash`/`ddgs`/`requests` imported lazily at point
   of use (pip-installed by `deps()`; a fresh Kaggle session has none of them); env vars set at
   module top before the tensorflow import; model stage renamed `make_model()` so it does not
   shadow `self.model`.
3. **`agri/new/agrisense_notebook.py` (new):** 3-cell wrapper — Cell 1 CONFIG (`CFG` dict, the only
   cell you edit), Cell 2 MODULE (`# %% include:agrisense.py` directive), Cell 3 RUN
   (`run(CFG)`). Self-contained when pasted: no dataset upload, no re-upload on changes.
4. **`agri/new/split_cells.py`:** SRC is now argv-configurable (default `agrisense_notebook.py`;
   `python split_cells.py ../old/agrisense_kaggle.py` regenerates the legacy cells); new
   `resolve_includes()` replaces `# %% include:<path>` lines with the referenced file's content
   before parse + round-trip verify.
5. **`agri/new/verify.py`:** rewritten for Phase 5 — module parses, `Pipeline` has all 21 stage
   methods, `run()` calls them in order, no `__main__` block, every entry point prints
   `PIPELINE_VERSION`, smoke wired through manifest/train/export/bundle/crawl, inference paths
   never divide by 255 (AST-based, comments don't count), lazy imports, audit markers; notebook
   has 4 cells (1 md + 3 code), Cell 1 CFG AST-matches `DEFAULT_CFG`, include directive present;
   generated cells = 3 code cells, MODULE cell not stale (ends with current module); legacy
   notebook kept + parses + 23 cells; field_test.py + README checks kept.
6. **`agri/new/test_phase3_dedupe.py`:** retargeted from `agrisense_kaggle.py` to `agrisense.py` —
   the four helpers are now `Pipeline` methods (none touch `self`), so they are AST-extracted
   from the class and bound to a dummy instance.

**Validation (all real, pasted in session):**
- `python split_cells.py` → 4 cells (1 md + 3 code), round-trip OK; legacy path verified
  separately (23 cells round-trip OK).
- `python -B verify.py` → ALL PASS, exit 0 (60 checks).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0 (16 checks, regression).
- `python -B test_phase4_field_test.py` → ALL PASS, exit 0 (16 checks incl. real TFLite e2e).
- One real bug caught during validation: `_d4_keys` referenced `imagehash` but the import was
  local to `dedupe()` → `NameError` in the worker thread. Fixed with a lazy import inside
  `_d4_keys`; the dead import in `dedupe()` was removed.

## Phase 5.1 — audit fixes (PIPELINE_VERSION 3.1.0)

**Trigger:** user-pasted code review found 4 real bugs + 1 hygiene issue. User's fix prompt
(reproduce → fix in order, "No other changes") was the approval.

**Reproduced first** (real output via temp script): (1) `min_class()` re-split wiped the
cross-class move (5/5 in train → 3/5); (2) smoke cap `groupby("class").head(40)` ran before
held-out routing → dedeikh dropped entirely, every class < `min_class` → "min_class OK" with 0
classes, splits 0/0/0, crash later; (3) `SaveBestF1.best` is a class attribute → survived
between runs, smoke score could block saving and `load_model` could load the smoke model;
(4) `"jit": True` default → XLA on with single GPU, `MacroF1` uses `tf.math.confusion_matrix`
(possible XLA compile failure); hygiene: contact email hardcoded in crawler UA.

**Fixes in `agri/new/agrisense.py`:**
1. `min_class()`: computes `mc = 10 if smoke else min_class`; captures the class-filter mask and
   re-aligns `self.keys8`; re-applies the cross-class move after each `grouped_split`
   (`xclass & split != "train"` → train); raises `SystemExit("min_class dropped every class…")`
   on empty `CLASSES`.
2. `manifest()`: smoke cap is per source+class (`groupby(["source", "class"]).head(40)`), so the
   held-out source survives smoke.
3. `dedupe()`: stores `self.keys8` (aligned with `self.man`) and `self.man["xclass"]` so later
   stages can re-align/re-apply; else branch sets `keys8=None`, `xclass=False`.
4. `train()`: first line resets `SaveBestF1.best = -1.0` (no bleed between runs in one kernel).
5. `DEFAULT_CFG["jit"] = False` (XLA off by default; `MacroF1` + `confusion_matrix` XLA risk).
6. `DEFAULT_CFG["crawl"]["contact"] = "gau.mah077@gmail.com"`; crawler UA reads it from CFG.
7. `held_out()`: new held-out-vs-train D4 overlap report — hashes held-out images, min D4
   distance to train keys, prints fraction within `near_dist` and within 7 bits, warns if >5%
   near-duplicate (answers "does dedeikh overlap anshul6?").
8. `PIPELINE_VERSION = "3.1.0"`; notebook CONFIG cell updated to AST-match (`jit` False,
   `crawl.contact`).

**Validation (all real, pasted in session):**
- `python -B split_cells.py` → 4 cells regenerated (3.1.0).
- `python -B verify.py` → ALL PASS, exit 0 (71 checks incl. 11 new Phase 5.1 checks).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0.
- `python -B test_phase4_field_test.py` → ALL PASS, exit 0.
- Fix verification script → all 6 fixes OK (cross-class 5/5 after min_class; dedeikh 240 imgs
  after cap, 6 classes kept, splits 336/72/72; SystemExit on empty CLASSES; reset present;
  jit False; contact in CFG; overlap report present).
- Overlap logic e2e (real imagehash): 1px-shift near-dupe → min D4 6 (within tight 7), distinct
  → 20. `imagehash==4.3.2` installed locally for the test.

**Verification limits:** same as Phase 5 — local TF 2.21 ≠ Kaggle TF 2.20. The 3.1.0 acceptance
gate is a user-pasted Kaggle smoke run (expect `PIPELINE_VERSION 3.1.0`, `SMOKE: manifest
capped … (40/source/class)`, dedupe/split/min_class OK, 1 epoch/stage, `SMOKE: export/bundle
skipped`), then the full run → read held-out macro-F1 + CI.

## Phase 5.2 — Kaggle smoke-run fixes (PIPELINE_VERSION 3.1.1)

**Trigger:** the 3.1.0 smoke run on Kaggle crashed in `held_out()` with
`FileNotFoundError ... /_dl_rice-leafs-disease-dataset/x/...`. User supplied a Claude-authored
fix; I compared it against my own diagnosis, adopted it with two refinements, and verified.

**Root cause (use-after-delete):** `preload()` deletes the CLI download trees after loading
train/val/test into RAM, but `held_out()` re-opened the held-out images from those same trees.
Latent in 3.0.0 (old smoke cap dropped dedeikh, so held_out() never had images); the 3.1.0
per-source+class cap exposed it. Would also have crashed the full run.

**Two more real issues from the same log:**
- XLA was still on: Keras 3 `compile()` defaults to `jit_compile="auto"`, and the pipeline only
  passed the flag when `JIT` was true → `Compiled cluster using XLA!`, 95 s Stage A, `Delay
  kernel timed out`. Fix: always pass `jit_compile=bool(self.JIT)`.
- All three sources printed `[cli]` despite Add Input mounts attached: `_mounted()` only looked
  one level deep in `/kaggle/input`. Fix: search `*`, `*/*`, `*/*/*` for the slug tail.

**Fixes in `agri/new/agrisense.py`:**
1. New `_preload_held()`: decodes held-out images at SIZE + D4-hashes them in one pass, storing
   `self.Xh` (uint8), `self.HK` (n,8 uint64), `self.HOK` (bool), aligned with the filtered
   `self.man_held`. Called from `preload()` BEFORE the CLI downloads are deleted.
2. `held_out()`: overlap check uses `self.HK[keep & self.HOK]`; eval pixels use
   `self.Xh[keep].astype(np.float32)` — never touches disk.
3. `train()`: `kw = {"jit_compile": bool(self.JIT)}` (forces XLA off when `jit: False`).
4. `_mounted()`: searches nested Add Input layouts (`*`, `*/*`, `*/*/*`).
5. `cleanup()`: frees `self.Xh` before `gc.collect()`.
6. `PIPELINE_VERSION = "3.1.1"`.

**Validation (all real, pasted in session):**
- `python -B split_cells.py` → 4 cells regenerated (3.1.1).
- `python -B verify.py` → ALL PASS, exit 0 (77 checks incl. 6 new Phase 5.2 checks).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0.
- `python -B test_phase4_field_test.py` → ALL PASS, exit 0.
- Use-after-delete simulation: `_preload_held` decodes+hashes 2/2, files deleted, overlap from
  RAM detects near-dupe (min D4 = 6), eval pixels float32 0-255 from RAM, cleanup frees Xh.

**Verification limits:** same as Phase 5.1 — local TF 2.21 ≠ Kaggle TF 2.20. The 3.1.1
acceptance gate is a user-pasted Kaggle smoke re-run. Expected: sources print `[mount]` (no
`freed ... GB` line), Stage A takes seconds (no XLA/`Delay kernel` messages), a real
`held-out vs train D4 overlap: x%` line, held-out report + bootstrap CI + abstain print, then
`SMOKE: export skipped` / `SMOKE: bundle skipped`. If still `[cli]`, run `!ls /kaggle/input
/kaggle/input/*` and match the exact mount path. Then full run → watch the PARITY line in
export/bundle (not yet exercised on Kaggle).

## Phase 5.3 — full-run crash: cross-class move split a cluster (PIPELINE_VERSION 3.1.2)

**Trigger:** the 3.1.1 full run on Kaggle (smoke=False) crashed in `split()` with
`AssertionError: LEAK: 1 clusters in both train and test`. Log showed `cluster overlap
train/val: 0` / `train/test: 1` after `cross-class near-duplicates: 10 images; 5 moved from
val/test to train`.

**Root cause:** the cross-class move in `split()` moved individual flagged images to train
(`moved = self.cross_mask & (self.man["split"] != "train")`). When a flagged image was a member
of a multi-member cluster (111 such clusters in the full run), moving just that image split the
cluster across splits → the structural leak assert fired. `min_class()` had the same
individual-image re-apply (`self.man["xclass"] & ...`) with no assert, so it would have leaked
silently into the final splits. Latent since Phase 5.1 (noted during reproduction: "if the
cross-class move splits a multi-member cluster, split()'s assert fires").

**Fix in `agri/new/agrisense.py`:** both places now move the WHOLE cluster of any
cross-class/xclass image to train (`bad_clusters = set(...); moved = cluster.isin(bad_clusters)
& split != "train"`). Cluster members are near-duplicates (merged at `near_dist`), so if one
carries an unreliable label they all do — and moving whole clusters preserves the grouped-split
invariant (a cluster is entirely within one split).

**Validation (all real, pasted in session):**
- `python -B split_cells.py` → 4 cells regenerated (3.1.2).
- `python -B verify.py` → ALL PASS, exit 0 (79 checks incl. 2 new Phase 5.3 checks).
- `python -B test_phase3_dedupe.py` → ALL PASS, exit 0.
- `python -B test_phase4_field_test.py` → ALL PASS, exit 0.
- Crash reproduction: synthetic 3-member cluster with one cross-class flag → `split()` no longer
  asserts, whole cluster lands in train, `min_class()` keeps the invariant, no cluster spans two
  splits.

**Verification limits:** same as Phase 5.2 — local TF 2.21 ≠ Kaggle TF 2.20. The 3.1.2
acceptance gate is a user-pasted Kaggle full re-run (smoke=False). Expected: `split sizes`
prints, no `LEAK` assert, `cluster overlap train/val: 0` / `train/test: 0` / `val/test: 0`,
then training → held-out macro-F1 + bootstrap CI (headline metric) → export/bundle PARITY line.

**Verification limits:** same as Phases 1–4 — local TF 2.21 ≠ Kaggle TF 2.20. The Phase 5
acceptance gate is a user-pasted Kaggle smoke run of the new 3-cell notebook (Cell 1 CONFIG →
Cell 2 MODULE → Cell 3 RUN).

## Headline metric (the number that matters)
Source-held-out: train on `anshul6` + `indo3`, test on `dedeikh` across the 5 shared classes
(`Sheath_Blight` has no second source → excluded). In-source test F1 is secondary and expected
to be much higher. **Built in Phase 3** (Cell 16.5, commit `ab04f7d`).
