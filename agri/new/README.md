# Agrisense — paddy disease classification

Kaggle pipeline. `agrisense.py` is the module — all logic in a `Pipeline` class with one
method per stage. `agrisense_notebook.py` is a thin 3-cell wrapper (CONFIG → MODULE → RUN)
that `split_cells.py` expands into paste-ready cells. The legacy 23-cell
`old/agrisense_kaggle.py` is kept for reference; its cell numbers map 1:1 to the module's
stage methods.

**Your laptop does nothing but copy text.** 8 GB RAM plus thermal throttling is exactly the
workload that kills laptop GPUs. Not one line of this runs locally.

---

## 0. Copy-paste workflow

`agrisense.py` is the **single source of truth** for the logic. `agrisense_notebook.py` is a
thin 3-cell wrapper around it: **Cell 1 CONFIG** (the `CFG` dict you edit), **Cell 2 MODULE**
(generated from `agrisense.py` via the `# %% include:agrisense.py` directive), **Cell 3 RUN**
(`run(CFG)`). `split_cells.py` expands the wrapper into paste-ready files.

For actual pasting, use the generated per-cell files — one file per cell means open it,
`Ctrl+A`, `Ctrl+C`, paste. No eyeballing boundaries, no half-pasted cells.

```
agri/
  old/                      <- OLD stuff, ignore it
    agrisense_kaggle.py     <- legacy 23-cell notebook, kept for reference only
  new/                      <- NEW stuff, this is what you use
    agrisense.py            <- module: all logic (Pipeline class, 21 stages, run())
    agrisense_notebook.py   <- 3-cell wrapper: CONFIG -> MODULE -> RUN
    split_cells.py          <- run this after any edit to refresh cells/
    field_test.py           <- eval-only loader for real field photos (section 7.5)
    cells/
      README.md             <- paste order table
      01_cell_1_config_the_only_cell_you_edit.py
      02_cell_2_module_generated_from_agrisense_py_by_split_c.py
      03_cell_3_run_prints_pipeline_version_and_runs_the_whol.py
      04_cell_4_download_model_bundle.py        <- helper: download the trained bundle
      05_cell_5_test_model_on_uploaded_photo.py <- helper: test on a photo you upload
```

The two helper cells (04, 05) are **not** part of the notebook — they are post-run extras.
`split_cells.py` does not regenerate them, so a re-run of `python split_cells.py` wipes them
(re-copy from git history if that happens).

Rules:

- `.md` files go into a **Markdown** cell; `.py` files into a **Code** cell.
- Paste top to bottom, in the numbered order.
- After editing `agrisense.py` or `agrisense_notebook.py`, run `python split_cells.py` (from
  the `agri/new/` directory) to resync. It resolves the include directive and verifies a
  round-trip against the combined source first, so a broken split fails loudly instead of
  silently producing a partial cell.
- Only **Cell 1 (`CFG`)** normally needs editing, and it is easiest to edit directly in Kaggle
  rather than round-tripping the file. `verify.py` AST-compares Cell 1's `CFG` against the
  module's `DEFAULT_CFG` so the two cannot drift.
- The legacy 23-cell notebook is regenerable with
  `python split_cells.py ../old/agrisense_kaggle.py` (this overwrites `cells/`).

---

## 1. Platform: Kaggle

| | Colab Free | Kaggle |
|---|---|---|
| System RAM | ~12.7 GB | **~16 GB** |
| GPU | T4, disconnects | T4 / P100, stable |
| Internet | Off by default | **Off by default — you must enable it** |
| Session | unpredictable | 12 h cap |

You already lost two Colab sessions. Settled.

Kaggle UI, before running anything:

1. Settings → Accelerator → **GPU T4 x2** (or P100 x2) → Save
2. Settings → **Internet: On** — Kaggle ships with this **off**, and the crawler in Cell 17
   needs it
3. **Attach all four datasets via Add Input.** Not optional in practice — see below.

### Disk is the binding constraint, so use Add Input for everything

`/kaggle/working` is a **~20 GB** volume, and the source sizes do not fit inside it:

| source | zip | extracted | peak if CLI-downloaded |
|---|---|---|---|
| `anshulm257` | 1.1 GB | 1.1 GB | 2.2 GB |
| `dedeikhsandwisaputra` | 0.4 GB | 0.4 GB | 0.8 GB |
| `shayanriyaz` | **8.1 GB** | **8.1 GB** | **16.2 GB** |
| `tedisetiady` | 0.3 GB | 0.3 GB | 0.6 GB |

That last row is the problem. CLI downloads need the zip *and* its expansion on disk at the same
time, so 16.2 GB of an 8.1 GB dataset leaves no room and the next write fails with `Errno 28`.
A dataset attached via **Add Input** lives in `/kaggle/input` and costs **zero**
`/kaggle/working` space.

Cell 3 is built around that, in this order:

1. **The source's `mount` path**, if it exists — free, and instant.
2. **Any `/kaggle/input` dir matching the slug tail** — free, and automatic.
3. **`kaggle datasets download`**, but only after a disk preflight that refuses to start if
   `zip + expansion + 0.5 GB` will not fit.

Three further things Cell 3 does, each a fix for a real failure:

- **Deletes the zip after extracting it**, so peak usage is 1x the dataset, not 2x.
- **Deletes the partial tree if a download fails**, so one bad source cannot make every
  later cell die with `Errno 28`.
- **Raises after the fetch loop if under 1 GB is left**, naming the CLI-downloaded sources that
  caused it, while Add Input is still a suggestion instead of a restart.

Cell 3 also frees the `_dl_*` trees after Cell 9 preloads into RAM, returning ~10 GB before the
exports.

### Authentication for the slug downloads

Your run showed the CLI **working** for public slugs with no `kaggle.json` — so the fallback is
viable. It is still the wrong choice here, purely on disk (see above).

Cell 3's fallback routes, in order:

1. **Add Input (recommended).** Attach in the UI; Cell 3 reuses the mount. No credentials, no disk.
2. **Secrets.** Add `KAGGLE_USERNAME` and `KAGGLE_KEY` in the left-sidebar Secrets tab.
3. **Explicit.** Set `mount` to the dataset's `/kaggle/input/...` path and `type: "local"` so a
   missing mount is a hard error instead of a silent CLI attempt.

If all three fail, Cell 3 prints the exact options rather than a bare exit code. Because Cell 3
catches per-source, one bad dataset does not kill the run — it prints `[FAIL]` and the rest load.

**Do not fight the CLI.** Option 1 costs four clicks and removes both the credential problem
and the disk problem. Prefer it.

### Using both T4s

`CFG["multi_gpu"] = False` (default). Single GPU is the default because `MirroredStrategy`
adds friction (XLA off, batch split) for no accuracy gain. Set it to `True` to try both T4s
for roughly 1.7x throughput; two details it handles:

- Variables are created inside `strategy.scope()` so they become `MirroredVariable`s.
- XLA is auto-disabled (`jit_compile` is skipped), because XLA plus `MirroredStrategy` has
  known friction on T4.

`CFG["jit"] = False` (default). XLA is off by default even on a single GPU: `MacroF1` uses
`tf.math.confusion_matrix`, a possible XLA compile failure, and at ~35 steps/epoch XLA
compilation overhead is not worth it. `train()` passes `jit_compile=False` explicitly because
Keras 3's `compile()` defaults to `jit_compile="auto"`, which would otherwise enable XLA
anyway. Set it to `True` to try XLA on a single GPU.

### Stack note: Python 3.13 / TF 2.20 / Keras 3

Kaggle currently serves this stack. It is stricter than TF 2.15 in ways that matter here, and
the pipeline is already adjusted:

| Trap | Symptom | Handled |
|---|---|---|
| `get_memory_info(PhysicalDevice)` | `TypeError: TFE_GetMemoryInfo(): incompatible function arguments` | Pass `g.name`; wrapped in `try/except` so a probe can never kill the run |
| `Metric.add_weight` signature | Keras 3 takes `shape` first, positional | All keyword arguments |
| `load_model` custom metric | Keras 3 needs registration | `@register_keras_serializable(package="agrisense")` |
| `imagehash` not installed | `ImportError` in Cell 6 | Cell 2 pip-installs `PIL` + `imagehash` (and `ddgs` opportunistically) |
| `/proc/meminfo` | missing on Windows | `_ram_gb()` probes `psutil` first, so Cell 0 also runs on your laptop (CPU only) |


---

## 2. Speed: why 4 hours is not a requirement

Your 4 hours were not training. They were: repeated 1 GB downloads, copying every file into
a merged directory, decoding high-res JPEGs inside the training loop, a full-set `predict()`
per epoch, and a long tail of TFLite `Flex` op retries plus a broken TFJS environment.

Real cost after this rewrite, on a T4:

| Step | Time |
|---|---|
| Resolve 4 Add Input mounts (no download) | ~20 s |
| Grouping (8 dihedral pHash variants/img) | ~3 min |
| Preload 256px uint8 into RAM, once | ~2 min |
| Stage A, 4 epochs | ~1 min |
| Stage B, 6 epochs | ~2 min |
| Stage C, 6 epochs | ~3 min |
| Eval, export, TFLite + parity | ~2 min |
| **Total** | **~15–20 min** |

With **Add Input for all four**, there is no download step at all — mounted datasets are already
on the machine. That is worth roughly 3–6 minutes and, more importantly, is the only way the
8.1 GB source fits alongside everything else.

`CFG["train_budget_min"] = 20` caps **training only**, starting at Cell 14 — mount resolution,
grouping and preloading do not eat it. A callback stops training when it is spent, across all
three stages, so you cannot end up in a 4-hour run.

The four levers doing the work:

1. **Preload once.** `shayanriyaz` is 8 GB of JPEGs at 2.4 MB each. Decoded to 256px uint8 it
   is 658 MB. JPEG decode never happens during training. `im.draft("RGB", ...)` lets libjpeg do
   a scaled decode, which is what makes those files tolerable at all. This alone is roughly an
   order of magnitude.
2. **Augmentation on CPU, after preload.** Keeps the cache at uint8 and keeps augmentation out
   of the Keras graph.
3. **Mixed float16 + both T4s.** `MirroredStrategy` is about 1.7x. XLA is skipped when
   sharding, since `jit_compile` and `MirroredStrategy` have known friction together.
4. **16 epochs instead of 20–25**, EarlyStopping patience 3.

Two things that were costing RAM for nothing: a dataset-level `shuffle(20000)` over 196 KB
images is a ~3.9 GB buffer, and `from_tensor_slices` on a multi-GB array embeds it as a graph
constant that trips the 2 GB protobuf limit. Cell 10 permutes **indices** in Python inside a
`from_generator`, so the shuffle buffer holds integers and the image array stays in host memory.

If you need more speed, `CFG["img_size"] = 192` gives ~1.35x with a small accuracy cost.

---

## 3. Taxonomy: 6 classes

Confirmed against the **real** Cell 4 manifest, not assumed. Every folder resolves through
`CFG["alias"]`, with `CFG["source_alias"]` taking precedence.

| Canonical class | anshul6 | dedeikh | indo3 | total | sources |
|---|:-:|:-:|:-:|---:|:-:|
| Bacterial_Leaf_Blight | 636 | 438 | 80 | 1154 | 3 |
| Brown_Spot | 646 | 876 | — | 1522 | 3 |
| Healthy | 653 | 438 | — | 1091 | 2 |
| Leaf_Blast | 634 | 438 | 80 | 1152 | 3 |
| Leaf_Scald | 628 | 438 | — | 1066 | 2 |
| Sheath_Blight | 632 | — | — | 632 | **1** |
| | 3829 | 2628 | 240 | **6617** | |

`dedeikh`'s Brown_Spot is 876 because Narrow Brown Spot (438) merges in. `indo3` contributes
only 160 — its Tungro folder is excluded.

### Three things the real manifest exposed

1. **`Sheath_Blight` is single-source.** `dedeikh` has no sheath blight folder at all, so those
   632 images come from `anshul6` alone — which also ships **pre-augmented**. If its labels are
   wrong, this class is wrong and nothing in the metrics would reveal it. Cell 4 now prints a
   `classes backed by a single source` line so this is visible every run, not discovered later.
2. **`anshul6` is suspiciously uniform** — 636/646/653/634/628/632, i.e. balanced to within 3%
   across all six classes. Real leaf-disease collections are never that even. That is the
   signature of augmentation applied to hit a target count, which is exactly why Cell 5 groups
   by dihedral-invariant hash: expect the post-dedupe count to fall well below 6617.
3. **`indo3` is white-paper imagery.** Its author states the photos were *"taken using white
   paper as background"*, Southeast Sulawesi, September 2020, 22.9 MB total. Only 160 of 6617
   images come from it, so the domain shift is minor — but those images are laboratory shots,
   not field photos.

### `blight` in indo3, and why it is a per-source alias

Cell 4 first reported `blight` as UNMAPPED. It is deliberately absent from the global `alias`
table, because "blight" alone is untrustworthy — it could be leaf blight, stem blight or fire
blight, and a wrong guess here silently poisons a class.

It is unambiguous **inside indo3**, whose 240 images sit in exactly three folders: `leafblast`,
`tungro` and `blight`. Two of the three name themselves, so the third can only be the bacterial
disease. Rather than weaken the global rule, the mapping is scoped:

```python
"source_alias": {"indo3": {"blight": "Bacterial_Leaf_Blight"}},
```

Cell 4 resolves per-source first, then the global table, and prints which rule matched each
image. It also refuses to start if a `source_alias` entry shadows a global alias with a
*different* class, since that would win silently.

**Tungro stays excluded.** 80 images → ~56 after the train split, under `min_class` 120. Do not
lower the threshold to rescue it: a class that exists in the output but cannot be learned is
worse than an absent one, because it produces confident wrong answers. It needs a real source.

### Two classes are missing from the original design

| Dropped | Sole source | Why |
|---|---|---|
| **Hispa** | shayanriyaz (CC0) | 8.04 GB download fails with `Errno 28` three ways while `/kaggle/working` has 19.5 GB free — a Kaggle-side quota we cannot see or size. Author describes it as *"a collection of multiple data sets I found online"*, so it also probably duplicates the others. |
| **Tungro** | indo3 | 80 images, below `min_class`. |

Both are recoverable. For Hispa: attach `shayanriyaz/riceleafs` via **Add Input**, uncomment the
source in `CFG["sources"]`, add `"Hispa"` to `CFG["classes"]`. The `alias` entry is already there
and is currently dropped by the orphan guard.

**The bug this replaces.** Your previous merge script ended with `return name.title()` — a
fabricated class name derived from a bare filename. Your shipped model has **8 outputs** while
the `CLASS_NAMES` you sent had **9 entries**, and one of those (`Leaf_Smut`) came from no real
dataset folder. There is now no fallback anywhere. Unresolvable images are counted and printed
under `UNMAPPED folders`, and you are told to add the alias. Cells 12 and 18 both assert
`len(CLASSES) == model.output_shape[-1]`.

---

## 4. Leakage: the thing that made your old metrics wrong

`anshulm257`'s folder is literally named `Rice_Leaf_AUG` — it ships pre-augmented. Its author
also describes `shayanriyaz` as *"a collection of multiple data sets I found online"*, so it
overlaps the others. Splitting randomly on a flat merged folder therefore puts augmented
siblings of the same source leaf on both sides of the split.

Three things had to be right, and the first version got all three wrong:

**1. The hash must be dihedral-aware.** pHash is *not* flip- or rotation-invariant, and
`Rice_Leaf_AUG` is exactly flips and rotations — so a plain pHash does **not** cluster the
siblings it was built to cluster. Cell 5 hashes all 8 D4 variants per image and builds a
brute-force pairwise distance matrix, taking the **min over the 8×8 variant pairs** — strictly
more information than a single canonical key, and complete by construction (no banding to get
wrong).

**2. Duplicates must be kept.** The first version called `drop_duplicates("cluster")`, which
left one image per cluster — every cluster a singleton, which made the "grouped split" and its
leak assert **vacuous**: the assert could never fail. Cell 5 now keeps every image and tags it
with a cluster id; Cell 6 splits on clusters. That is what the assert is supposed to mean.

**3. The merge must be complete for the threshold used.** The old LSH banding only guaranteed
a candidate pair when Hamming distance <= 3, so a threshold of 6 silently missed a third of
them. Cell 5 now computes the full pairwise matrix with `np.bitwise_count` (chunked so the
working set stays small) — complete for every threshold by construction.

Cell 5 also prints the diagnostics that tell you whether grouping behaved: number of clusters,
how many are multi-member, the **largest cluster size**, and how many clusters **span more than
one class** (a label-noise warning). Single-linkage on visually similar leaves can chain into
oversized clusters, so the largest-cluster figure matters — if it is in the hundreds, lower
`dedupe.near_dist` to 0 and rely on exact canonical keys only.

### What the real 6617-image run showed

```
clusters: 5114  |  multi-member: 660  |  images inside multi-member clusters: 2163 (33%)
images folded into an existing cluster: 1503 (23% of the manifest)
largest cluster: 62 images  |  median 1
clusters spanning >1 source: 3
cross-class near-duplicates: 388 images (5.9%) — excluded from val/test in Cell 6
```

(The 388/5.9% figure is the old run's cluster-based measure. The current code reports
cross-class collisions per-image from `cross_class_mask` — images within `near_dist` of a
different-class image — so the exact number on a fresh run will be close but not identical.)

**Largest cluster 62, not hundreds** — no single-linkage chaining, so `near_dist = 4` stays.
33% of images sit in multi-member clusters, which is `anshul6`'s pre-augmentation, and is
exactly what the grouped split exists to contain.

**Only 3 clusters span more than one source.** The three datasets are essentially disjoint, so
`anshul6`'s augmentation did *not* leak into `dedeikh`'s images.

**Cross-class collisions are label noise, not clusters.** Merging is same-class only — a
cross-class near-duplicate is at least one mislabelled image, and folding it into a cluster
would bake the contradiction in. Cell 5 flags them separately (`cross_class_mask`) and Cell 6
moves them out of val/test into train, so they are never scored. 388 images (5.9%) is inherent
to scraped data.

**`dedeikh` is the noisy source**: 55 of the 66 colliding pairs are `dedeikh:X <-> dedeikh:Y`,
concentrated in the three confusable spot diseases (`Healthy`/`Leaf_Blast` 14,
`Brown_Spot`/`Leaf_Blast` 13, `Brown_Spot`/`Healthy` 11). Its `Healthy` and `Brown_Spot` classes
in particular look like scraped sets with some mislabelling. This is exactly why it is the
source-held-out test set (below).

### The cross-class pair table doubles as the `blight` alias check

Cell 5 breaks cross-class collisions down by `(source, class)` pair. This is the empirical test
of `CFG["source_alias"]`: had indo3's `blight` folder really been Leaf Blast, those 80 images
would have been near-duplicates of Leaf_Blast images and `indo3:Bacterial_Leaf_Blight` would
have appeared in that table. It does not appear at all — zero collisions for that class, while
`indo3:Leaf_Blast` collides 6 times. Absence of contradiction, plus the folder-structure
argument, is as much as this dataset can prove.

### The leak assert is structural — read the scan instead

Cell 6's cluster-overlap assert **cannot fail**: the split is assigned per cluster, so overlap
is impossible by construction. Treating it as a safety net is a mistake, and the pipeline says
so in its own output.

One caveat (fixed in 3.1.2): the cross-class move used to relocate individual flagged images
to train, which split a multi-member cluster across splits and tripped the assert on the real
data. It now moves the **whole cluster** of any cross-class image, so the per-cluster invariant
holds again.

The check that *can* fail is the cross-split near-duplicate scan. Cell 5 merges same-class
near-duplicates at Hamming <= 4; images that are visually near-identical but pHash-different
(re-cropped, re-encoded, rotated off-grid) never get merged and **can** straddle train and
val/test. Cell 6 re-scans the brute-force D4 distance matrix at Hamming <= 7 — one step tighter
than the merge — and reports any pair that ended up in different splits. It costs no re-hashing
because the matrix already exists.

### The split was not stratified, and the seed did nothing

Cell 6 is called a *grouped stratified split* and balances clusters. Its first implementation
balanced only the **global** total:

```python
target = {k: tot * v for k, v in sp.items()}     # one target for ALL classes
k = max(sp, key=lambda x: target[x] - got[x])    # pick the emptiest split
```

Nothing ever looked at which class a cluster belonged to. On the real data that produced:

```
                       test  train   val    70/15/15 target
Brown_Spot              59   1404    59     1065 / 228 / 229
Sheath_Blight          202    230   200      442 /  95 /  95
```

Brown_Spot got **92% of itself into train** and was then scored on **59 images**. Sheath_Blight
— the weakest, single-source class — was trained on 230 and scored on 200. Macro-F1 over that
table describes the split, not the model. The global totals looked perfect (`4632/993/992`
against `4631/992/992`), which is exactly why it passed unnoticed.

`rng = np.random.RandomState(seed)` was created and **never used**, so `CFG["seed"]` had no
effect on the split at all. Two runs were bit-identical.

The rewrite places one class at a time, **rarest first**, so a thin class gets its clusters
placed before a common class can absorb them. Within a class the most-under-quota split wins,
scored as a *relative* deficit (`got/target`) so classes of different sizes compete fairly. A
cluster spanning several classes is claimed by whichever class reaches it first — hence rare
first. Cell 6 now prints per-class deviation from target, which is the check that catches this
class of bug:

```
per-class deviation from target ratio:
  Brown_Spot              n= 1522  max dev  0.4%
  ...
  worst per-class deviation: 1.7%
```

**Grouping beats stratification when they conflict.** If a class's images are all one cluster,
that cluster lands whole in one split and val/test get zero for it. That is correct — splitting
it would leak — but the class's metric is undefined, so Cell 6 says so explicitly rather than
printing a meaningless macro-F1.

### The leak scan found real leakage

```
cross-split near-duplicate scan (hamming <= 7, tighter than the merge threshold of 4):
  164 near-duplicate pair(s) straddle splits:
       82  train/val   e.g. Healthy vs Healthy  (cluster sizes [1, 1])
       75  test/train  e.g. Bacterial_Leaf_Blight vs Bacterial_Leaf_Blight
        7  test/val    e.g. Sheath_Blight vs Sheath_Blight
```

Every example is the **same class in two singleton clusters** — genuine near-twins (re-cropped
or re-encoded) that Hamming <= 4 missed. All 164 in singleton clusters, so the defect is the
threshold, not the banding.

The scan is deliberately one step tighter than the merge, so it reports the residual leaks the
merge missed. If the count is large, raise `dedupe.near_dist` toward 7 — the cost is
over-merging (largest cluster grows), and Cell 5 prints the largest cluster and its composition
precisely so that trade-off is visible rather than assumed. Drop back to `4` or `3` if the
largest cluster runs away.

### Source-held-out eval — the headline metric

In-source val/test (Cells 15–16) is secondary. The number that matters is **source-held-out**:
train on `anshul6` + `indo3`, test on `dedeikh` across the 5 shared classes. `Sheath_Blight` is
single-source in `anshul6`, so it stays in training but is excluded from the held-out eval (no
held-out images exist for it).

`CFG["held_out_source"] = "dedeikh"` routes that source out of training in Cell 4 — it never
touches train/val/test, the dedupe, or the class weights. Cell 16.5 then evaluates the trained
model on it:

- **macro-F1 over the 5 shared classes**, with a **bootstrap 95% CI** (`bootstrap_iters`).
- **Abstain rule**: predictions below `abstain_threshold` confidence are skipped; coverage and
  accuracy/macro-F1 on the covered subset are reported. A farmer-facing tool should abstain
  rather than guess.
- **Per-source macro-F1** on both the held-out set and the in-source test, so you can see which
  source the model generalises to worst.

`dedeikh` is the noisy source (55 of 66 colliding pairs), so this is the honest number — expect
it to be well below the in-source test. That gap is the real-world performance estimate.


---

## 5. Imbalance

`class_weight` only. `effective` and `oversample` were removed on purpose (audit): class_weight
is the one that worked, and the others added config surface without a measured win.

Weights are computed in Cell 8 from the **manifest**, which is what fixes the
`'_PrefetchDataset' object has no attribute 'class_names'` crash — counting never touches a
prefetched dataset. `class_weight` uses `N/(NC·n_c)`.

Cell 7 drops any class that ends up with fewer than `min_class` (120) images **in the training
split**, and re-runs the split until the constraint holds. This check has to live after the
split: a class can clear the raw pre-dedupe count and still be nearly emptied by grouping.
A thin class costs you a label slot and drags macro-F1 down.

Checkpoints monitor `val_macro_f1`, not `val_accuracy`. Accuracy is dominated by the majority
class, so optimising it actively trades away your rare classes. There is **one** global best,
written to one file — the first version reset the counter per stage, which meant Stage C
saved its first epoch regardless of score, and then picked between stages by file size.

Cell 15 prints **worst-class recall** — for a farmer-facing tool, one class at 10% recall is
worse than useless, because it confidently reports "healthy" for a diseased leaf.

---

## 6. Why progressive unfreezing, not per-layer learning rates

Your backbone was frozen the whole time. For diseases that differ by small spot morphology,
that is a large loss — bigger than any reweighting scheme.

The obvious way to unfreeze it is wrong, and the first version of this file got it wrong in a
way that raised no error at all. Keras names EfficientNet's internals `stem_conv`,
`block1a_dwconv`, `block1a_expand_conv` and so on — **none of them contain the string
`"efficientnet"`**. So any name-based match finds only the single nested base-model wrapper.
`core[-15:]` on a one-element list returns that whole element, the children then get
explicitly set to `trainable=False` *after* their parent, and the head's `Dense(128)` fails the
`name == "pred"` test. Net effect: **all three stages trained only the final softmax layer, on
top of a random frozen Dense(128).** Progressive unfreezing was a complete no-op and the run
would have produced a plausible, useless model.

Cell 12 walks `base.layers` directly and Cell 14 **asserts** the expected number of trainable
backbone layers per stage, plus that the head itself is trainable. If either assertion fails
you find out in seconds rather than after 20 minutes.

| Stage | Unfreeze | Epochs | LR |
|---|---|---|---|
| A | backbone frozen | 4 | 1e-3 |
| B | last 18 backbone layers | 6 | 1e-4 |
| C | last 70 backbone layers | 6 | 3e-5 |

The backbone is called with `training=False` on purpose. That flag only switches Dropout and
BatchNorm to inference mode — Conv and Dense kernels still receive gradients whenever their
layer is `trainable=True`. It also guarantees BatchNorm never updates its running statistics,
which is what you want on 8k images. (This is why there is no `bn_trainable` flag any more.)

---

## 7. Crawled photos

Web images are **not** bulk-injected into training. Searching "rice leaf brown spot" returns
pesticide ads, diagrams, and mislabelled stock photos — roughly 20–40% label noise. Injecting
that at volume makes the model worse and you won't notice until it's shipped.

Default `role: "stress_test"`. Cell 17 downloads to a separate directory, records a
`provenance.csv` with source URL, query, and licence, and then reports how the model does on
photos it has never seen. That train↔field gap is the most useful number in the whole
notebook, and it is the one thing no clean lab dataset can give you.

Notes:
- Wikimedia Commons is tried first because it does not block datacenter IPs. DuckDuckGo
  usually does get blocked from Kaggle. The cell degrades gracefully if so.
- `role` is `stress_test` only — the `finetune` role was removed on purpose (audit). Cell 17
  asserts this so a stale config cannot silently re-enable it.
- Web images are **not licensed for redistribution**. Fine for internal evaluation; do not
  ship them in a product.

---

## 7.5 Field-test loader (eval only)

The crawler (Cell 17) tests on web images. For **real farm photos**, use the standalone loader
`agri/new/field_test.py` — it runs anywhere (laptop, phone, edge box) with just numpy + Pillow +
the TFLite runtime, and needs no training pipeline.

```
python field_test.py --bundle agrisense_bundle.zip --photos /path/to/field/photos
```

It loads the deployed `model.tflite` + `class_names.json` from the bundle and applies the exact
Cell 18 preprocessing contract (RGB → LANCZOS resize to 256×256 → float32 0–255, **no /255**),
so the number it reports is what the shipped model actually does on your photos.

Photos layout: one subfolder per class (`class/*.jpg`, same as the crawler), or a flat folder
with `--labels labels.csv` (`path,class`). Unknown class folders are skipped with a warning.

Report: per-class precision/recall/F1, macro-F1, accuracy, worst-class recall, bootstrap 95% CI,
and the abstain rule (coverage + accuracy/macro-F1 on covered). Writes `predictions.csv` and
`confusion.csv` next to the photos (or `--out DIR`). `--abstain` overrides the threshold shipped
in the contract; `--limit N` caps images per class for a quick check.

The loader is eval-only by design — it never trains, never crawls, never writes back to the
model. It is the honest test: if the field photos are real farm shots, this number is the
real-world performance estimate.

For a quick single-photo check **inside the Kaggle notebook itself**, use the helper cell
`cells/05_cell_5_test_model_on_uploaded_photo.py` — it loads the same bundle and shows a
"Choose a photo" upload button plus a Predict button (no need to upload to `/kaggle/working`
first).

---

## 8. Run order

Cell numbers below match the filenames in `cells/`, so the table and the files line up.

| Cell | File | Do this |
|---|---|---|
| 1 | `01_cell_1_config_the_only_cell_you_edit.py` | Edit `CFG` only. |
| 2 | `02_cell_2_module_*.py` | The whole pipeline (21 stages). Do not hand-edit. |
| 3 | `03_cell_3_run_*.py` | `run(CFG)` — prints `PIPELINE_VERSION`, runs every stage in order. |
| 4 | `04_cell_4_download_model_bundle.py` | After the run: download `agrisense_bundle.zip` (Output tab → Download, or `kaggle kernels output`). Helper, not part of the pipeline. |
| 5 | `05_cell_5_test_model_on_uploaded_photo.py` | After the run: upload a photo and predict (Choose a photo → Predict). Helper, not part of the pipeline. |
| — | `field_test.py` (section 7.5) | After the run: test the downloaded bundle on real field photos. Eval only, runs anywhere. |

The 21 stages inside Cell 2 map 1:1 to the legacy 23-cell notebook
(`old/agrisense_kaggle.py`):
`env → deps → fetch → manifest → dedupe → split → min_class → class_weights → preload →
pipelines → macro_f1 → make_model → callbacks → train → val_report → test_report → held_out
→ crawl → export → bundle → cleanup`. Stop-and-read checkpoints are the manifest, the
dedupe/split report, the val report and the source-held-out eval — each stage prints its
output in order.

**Smoke mode:** set `CFG["smoke"] = True` for a fast sanity run — 40 images/source/class, 1
epoch/stage, no crawl, no export/bundle. Run it before any full run; it exercises every stage
end-to-end in a few minutes.

Reuse the model in a later run: add `agrisense_bundle.zip` as a Kaggle Dataset, or `Save
Version` and attach the Output.

---

## 9. Reading the results

| Signal | Meaning and action |
|---|---|
| `UNMAPPED` in Cell 4 | Add to `CFG["alias"]`. Do not proceed on a guessed mapping. |
| `largest cluster` in the hundreds (Cell 5) | Single-linkage chaining. Set `dedupe.near_dist: 0`. |
| `clusters spanning >1 class` (Cell 5) | Label noise in the source data, not a code bug. |
| `cross-class near-duplicates` (Cell 5) | Label noise. Moved to train in Cell 6, never scored. |
| `SOURCE-HELD-OUT macro-F1` ≪ in-source test (Cell 16.5) | Expected and the point. The gap is the real-world estimate. |
| `abstain` coverage low (Cell 16.5/17) | Model is unsure on field images. Lower `abstain_threshold` or collect more data. |
| `every cluster is a singleton` (Cell 6) | Grouping was inert, so the leak assert proves nothing. |
| `DROPPED` in Cell 7 | Class could not reach 120 train images. Add a source or lower `min_class`. |
| `backbone 0/M` in stage B or C | Unfreezing regressed. The Cell 14 assert should have fired. |
| `worst recall` ≈ 0 | Minority collapsed. Add a source for that class or lower `min_class`. |
| `GAP` > 0.05 | Split still leaks, or is too small. |
| `GAP` < 0 | Val was pessimistic. Fine — trust test. |
| TFLite `PARITY` < 100% | Do not ship the converted model. |
| Field accuracy ≪ test accuracy | Expected and the point. Collect real farm photos. |

---

## 10. Two honest caveats

1. **Everything except the Cell 17 stress test is lab-ish imagery.** Even after dedupe, all
   four sources are mostly controlled-background or scraped stock photos. The source-held-out
   eval (Cell 16.5) is the honest in-dataset estimate — training on `anshul6`+`indo3` and
   testing on `dedeikh` — but even that is not a real field. Only a held-out set of real farm
   photos fixes that, and no amount of class weighting substitutes for it.
2. **Licensing.** `shayanriyaz` is **CC0** — the only clean one. The other three are
   "Unknown". Fine for a student project; check before distributing anything commercially.

---

## 11. Known export pitfalls (already handled)

| Symptom | Cause | Fix in this pipeline |
|---|---|---|
| `ERROR_NEEDS_FLEX_OPS` | Augmentation ops inside the graph | Augmentation lives in `tf.data`, never in the graph |
| `'numpy' has no attribute 'object'` | numpy ≥ 1.24 | Don't convert TFJS here; separate notebook with pinned deps |
| `No module named 'distutils'` | setuptools change | Same |
| `tf.compat.v1 has no attribute 'estimator'` | tensorflowjs converter vs TF ≥ 2.16 | Same — `pip install tensorflow==2.15.1 tensorflowjs numpy==1.26.4` |
| `'_PrefetchDataset' has no attribute class_names` | counting a prefetched dataset | Counts come from the manifest |
| `tf.image.random_resized_crop` not found | it does not exist | `sample_distorted_bounding_box` (Cell 10) |
| `TFE_GetMemoryInfo(): incompatible arguments` | passed a `PhysicalDevice`, not a name | `get_memory_info(g.name)`; returns `current`/`peak` |
| `AttributeError: 'int' object has no attribute 'Callback'` | `K` shadowed by a class count | callbacks module is `KC` |
| `'NoneType' object has no attribute ...` in `_aug` | labels not one-hot | one-hot emitted inside the pipeline (Cell 10) |
| Keras rejects a 256px val batch | eval never resized to `SIZE` | `_eval_t` resizes 256→224 |
| Only the softmax layer trains, silently | name match on `"efficientnet"` never hits internals | `set_trainable` walks `base.layers`; Cell 14 asserts it |
| `ModuleNotFoundError: imagehash` | not a Kaggle preinstall | Cell 2 pip-installs it |
| `protobuf` size limit on `from_tensor_slices` | big array embedded as a graph constant | `from_generator` keeps the array in host memory |
| **`OSError: [Errno 28] No space left on device`** | `/kaggle/working` is ~20 GB; an 8.1 GB zip expands to another 8.1 GB, and the zip was not deleted afterwards, so the volume filled and *every* later write failed — including a 3 KB CSV | Attach sources via **Add Input** (zero disk cost). Cell 3 preflights the CLI path, deletes the zip after extraction, removes the partial tree on failure, and raises early with the culprit named |
| Export looks untrained | `clean` nests `model`, so weights are shared not copied | Cell 18 verifies the sharing |

---

## 12. Frontend preprocessing contract

`class_names.json` ships this machine-readably, but the short version:

1. Decode to RGB.
2. **Downscale the long edge to ~256px with a high-quality filter** — PIL `LANCZOS`, canvas
   `imageSmoothingQuality: 'high'`, or `cv2.INTER_AREA`.
3. Emit `(1, 256, 256, 3)` `float32` in 0–255, channels_last.

The graph does 256→224 bilinear, which at that ratio is nearly 1:1 and aliases negligibly.
**Do not feed a 4000px phone photo straight in**: in-graph non-antialiased bilinear would alias
away exactly the spot texture this model exists to read, and TFLite cannot do an antialiased
resize in-graph. This is also what removes the train/serve skew, because training preloaded
with PIL `BILINEAR`, which *does* antialias.

**Delete the `resizeNearestNeighbor` step** from your existing web app.

Cell 19 runs a Keras-vs-TFLite parity check over 16 validation images and prints the argmax
agreement rate. If it is below 100%, do not ship the converted model.
