# CELL 9 — preload to a RAM array (the speed core)
# 8 GB of high-res JPEG -> ~650 MB of 256px uint8. Decoded once, never again.
def preload(df, pre=PRE, workers=None):
    workers = workers or CFG["workers"]
    paths = df["path"].tolist()
    arr = np.zeros((len(paths), pre, pre, 3), np.uint8)
    ok = np.ones(len(paths), bool)
    def work(i):
        try:
            with Image.open(paths[i]) as im:
                im.draft("RGB", (pre, pre))      # scaled JPEG decode: 5-10x faster
                arr[i] = np.asarray(im.convert("RGB").resize((pre, pre), Image.BILINEAR),
                                    np.uint8)
        except Exception:
            ok[i] = False; arr[i] = 0
    t = time.time()
    with ThreadPoolExecutor(workers) as ex:
        for _ in ex.map(work, range(len(paths))): pass
    arr, df = arr[ok], df[ok].reset_index(drop=True)
    print(f"  {len(arr)} imgs -> {arr.nbytes/1e9:.2f} GB in {time.time()-t:.0f}s "
          f"({arr.nbytes/max(len(arr),1)/1e6:.0f} KB/img)"
          + (f"  ({int((~ok).sum())} unreadable dropped)" if not ok.all() else ""))
    return arr, df

# budget covers train + val + test together
per_mb = PRE * PRE * 3 / 1e6
need_gb = len(man) * per_mb / 1000
budget = CFG["ram_frac"] * RAM_AVAIL_GB
USE_RAM = CFG["preload_ram"] and need_gb <= budget
print(f"{len(man)} imgs x {per_mb*1000:.0f} KB = {need_gb:.2f} GB vs budget {budget:.2f} GB "
      f"-> {'RAM preload' if USE_RAM else 'disk stream'}")

if USE_RAM:
    Xtr, tr = preload(tr)
    Xva, va = preload(va)
    Xte, te = preload(te)
else:
    Xtr = Xva = Xte = None
ytr = np.array([CIDX[c] for c in tr["class"]], np.int32)
yva = np.array([CIDX[c] for c in va["class"]], np.int32)
yte = np.array([CIDX[c] for c in te["class"]], np.int32)

# (oversample was removed on purpose — class_weight only, see Cell 8)

# Once the pixels are in RAM, the on-disk copies are dead weight — ~10 GB of it. /kaggle/working
# is ~20 GB and the Keras/TFLite exports plus checkpoints need that space later. Only safe when
# USE_RAM, because the disk-stream path still reads from these paths in Cell 13.
if USE_RAM:
    by_name = {r["name"]: str(r["path"]) for r in ROOTS}
    freed = 0.0
    for s in CFG["sources"]:
        if "/kaggle/input" in by_name.get(s["name"], ""):
            continue                                    # a mount, not ours to delete
        dl = WORK / ("_dl_" + s["slug"].split("/")[-1]) if s.get("slug") else None
        if dl and dl.is_dir():
            freed += _dir_gb(dl)
            shutil.rmtree(dl, ignore_errors=True)
    if freed > 0.1:
        print(f"freed {freed:.1f} GB of CLI downloads after preload -> {_free_gb():.1f} GB free")
