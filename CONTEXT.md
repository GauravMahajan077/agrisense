# CONTEXT — Agrisense (read this first in any new session)

Session state file. Updated at the end of every phase. Committed to git.

## What this project is
Rice-leaf disease classification for Kaggle (TF 2.20 / Keras 3, 2x T4). Full rules live in
[`AGENTS.md`](AGENTS.md) — that file is authoritative for *how* to work here.

- **Source of truth:** `agri/agrisense_kaggle.py` — 22 `# %%` cells (1 markdown + 21 code), 1,824 lines.
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
- Hygiene: pip installs unpinned (line 378), crawler has no byte cap (`.content`, line 1653),
  User-Agent says `contact: set-your-email-here` (line 1611), `multi_gpu: True` default,
  no `smoke` mode, no `PIPELINE_VERSION`.

## Phase status
| Phase | Status |
|---|---|
| **0** — `AGENTS.md` + `git init` + baseline commit | ✅ DONE (commit `f9c8a74`) |
| **cleanup** — delete false-confidence tests, temp orphans | ✅ DONE (this commit) |
| **1** — P0 crash fixes | ⬜ NEXT |
| **2** — P1 simplify (remove sweep/LSH/cache/`effective`/`oversample`/finetune-crawl, `multi_gpu: False`, smoke mode, `PIPELINE_VERSION`, pin pip, UA email, crawl byte cap) | ⬜ |
| **3** — P2 honest eval (8-variant D4 brute force, same-class merge only, cross-class dropped from val/test, source-held-out split, macro-F1 per source + bootstrap CI, abstain rule) | ⬜ |
| **4** — P3 field-test loader (eval only) | ⬜ |
| restructure → `agrisense.py` + 3-cell notebook | ⬜ deferred until 0–4 pass |

## Phase 1 — exact changes queued (in `agri/agrisense_kaggle.py`)
1. `clusters_at`: `if not nd: return cid` → `return cid, 0` (lines 782–783).
2. Cell 15 `predict_idx`: predict on the whole dataset once + `assert len(p) == n`.
3. Cell 17: `np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))[None]` and **remove `/255.`**.
4. Cell 10: `h.set_shape([SIZE, SIZE, 3])` guards at the end of `_aug` and `_eval_t`.
5. Cell 18/19 export: rebuild `clean` under `float32` policy (weight transfer guarded by asserts),
   fixed `tf.keras.Input((PRE, PRE, 3), batch_size=1)`, parity reads the interpreter's real
   allocated shape, contract JSON updated to `(1, 256, 256, 3)`.
Then: `python split_cells.py` → `python -B verify.py` (must exit 0) and re-run any checks.
**Verification limits:** local = TF 2.21 / Keras 3.13.2 / Python 3.13. Kaggle = TF 2.20.
Local green ≠ Kaggle green. A Kaggle smoke run must be pasted before claiming Phase 1 works there.

## Headline metric (the number that matters)
Source-held-out: train on `anshul6` + `indo3`, test on `dedeikh` across the 5 shared classes
(`Sheath_Blight` has no second source → excluded). In-source test F1 is secondary and expected
to be much higher. Built in Phase 3.
