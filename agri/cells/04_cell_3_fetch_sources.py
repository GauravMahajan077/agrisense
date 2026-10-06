# CELL 3 — fetch sources
# Resolution order for type "auto":
#   1. the explicit `mount` path            -> free, costs no /kaggle/working space
#   2. any /kaggle/input dir matching the slug tail
#   3. `kaggle datasets download`, GATED on free disk
#
# Why the gate exists: /kaggle/working is ~20 GB and shayanriyaz is an 8.1 GB zip that
# expands to another ~8 GB. Without the gate the download fills the volume, the partial file
# is left behind, and every later cell dies with Errno 28 — including a 3 KB CSV write.
# A mounted dataset has none of that cost, so Add Input is always the better path here.
def _n_images(d): return sum(1 for _ in d.rglob("*") if _.suffix.lower() in IMG_EXT)

def _first_images(d):
    """Walk down to the first directory that actually holds images.
    We deliberately return the SOURCE ROOT, not the image dir: Cell 4 resolves classes from
    each file's ancestor chain relative to this root, which works for any nesting depth."""
    d = Path(d)
    if not d.is_dir(): return d
    if _n_images(d): return d
    for c in sorted(d.iterdir()):
        if c.is_dir() and _n_images(c): return c
    return d

def _mounted(slug):
    """Find an Add Input mount whose directory name matches the slug tail."""
    want = slug.split("/")[-1].lower()
    base = Path("/kaggle/input")
    if not base.is_dir(): return None
    for d in base.iterdir():
        if d.is_dir() and want in d.name.lower() and _n_images(d):
            print(f"   Add Input mount '{d.name}' ({_n_images(d)} imgs)")
            return _first_images(d)
    return None

def _cli_download(src):
    """Download + extract one slug. Returns the extraction root.

    Three deliberate choices, each a fix for a failure we actually hit:
      * PREFLIGHT — refuse before writing anything if the zip plus its expansion cannot fit;
      * DELETE THE ZIP — peak usage is zip + extracted, so removing the zip right after
        extraction halves peak disk. The old code left the zip sitting next to the copy;
      * CLEAN UP ON FAILURE — remove the partial tree, so one bad download cannot make every
        subsequent cell fail with Errno 28."""
    slug = src["slug"]
    out = WORK / ("_dl_" + slug.split("/")[-1])
    if (out / "_ok").exists():
        return _first_images(out / "x")

    if not CFG.get("allow_cli_download", True):
        raise RuntimeError("CLI download disabled (CFG['allow_cli_download'] = False)")
    zip_gb = float(src.get("zip_gb") or 0.0)
    # Peak = zip on disk + extracted tree. JPEGs barely recompress, so extracted ~= zip.
    need = zip_gb * 2 + 0.5
    have = _free_gb()
    if zip_gb and have < need:
        raise RuntimeError(
            f"refusing to download {zip_gb:.1f} GB into {have:.1f} GB free "
            f"(needs ~{need:.1f} GB for zip + extraction).\n"
            f"  FIX (best): Add Input -> attach '{slug.split('/')[-1]}'. Mounted datasets cost\n"
            f"       no /kaggle/working space at all, and Cell 3 detects it automatically.\n"
            f"  ALTERNATIVELY: drop this source from CFG['sources'] — check Cell 7 for which\n"
            f"       classes it was the only source for (here: Hispa).")

    # Force the CLI to stage through WORK, not the system temp dir. On Kaggle /tmp is a much
    # smaller filesystem than /kaggle/working, and the kaggle client downloads through a temp
    # file — which is why an 8 GB dataset failed with Errno 28 while WORK still had 19.5 GB
    # free. Pointing TMPDIR/TMP/TEMP at WORK makes the preflight above meaningful.
    staging = WORK / "_tmp"
    staging.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env.update({"TMPDIR": str(staging), "TMP": str(staging), "TEMP": str(staging)})

    try:
        out.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["kaggle", "datasets", "download", "-d", slug,
                            "-p", str(out), "-q"], capture_output=True, text=True, env=env)
        if r.returncode:
            raise RuntimeError((r.stderr or r.stdout or "").strip()[:200])

        zips = list(out.glob("*.zip"))
        if not zips:
            raise RuntimeError("download reported success but produced no .zip")
        (out / "x").mkdir(parents=True, exist_ok=True)
        for z in zips:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(out / "x")
            z.unlink()                       # <-- halves peak disk; the only copy left is the tree
        (out / "_ok").write_text("ok")
        return _first_images(out / "x")
    except OSError as e:
        shutil.rmtree(out, ignore_errors=True)
        # An OSError from the CLI arrives as a bare "[Errno 28] No space left on device",
        # which reads like the working volume is full. It is not: /kaggle/working still has
        # free space, so the CLI is writing somewhere we cannot see or size. Say so.
        raise RuntimeError(
            f"{type(e).__name__} raised while running the kaggle CLI: {e}\n"
            f"  NOT the working volume: /kaggle/working still has {_free_gb():.1f} GB free,\n"
            f"  and TMPDIR was already redirected at it. The limit is a Kaggle-side quota or a\n"
            f"  CLI staging path outside our control.\n"
            f"  FIX: attach '{slug.split('/')[-1]}' via Add Input. A mounted dataset costs zero\n"
            f"  working disk and bypasses the CLI entirely — Cell 3 finds it automatically.\n"
            f"  ALTERNATIVELY: remove this source from CFG['sources'] to proceed without it.")
    except Exception:
        shutil.rmtree(out, ignore_errors=True)   # never leave a half-written tree behind
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

def fetch(src):
    kind = src.get("type", "auto")
    slug = src.get("slug")

    if kind in ("local", "auto") and src.get("mount"):
        p = Path(src["mount"])
        if p.is_dir() and _n_images(p):
            print(f"   mount '{p}' ({_n_images(p)} imgs)")
            return _first_images(p)
        if kind == "local":
            raise FileNotFoundError(
                f"{p} not found — add it via Kaggle 'Add Input', or set type 'auto'/'kaggle'")

    if not slug:
        raise RuntimeError(f"source {src.get('name')} has neither 'mount' nor 'slug'")
    m = _mounted(slug)
    if m is not None:
        return m
    if kind == "local":
        raise FileNotFoundError(f"no mount for '{slug}' — attach it via Kaggle 'Add Input'")
    return _cli_download(src)

ROOTS, FAILED = [], []
for s in CFG["sources"]:
    t = time.time()
    try:
        r = fetch(s)
        ROOTS.append({**s, "path": r})
        how = "mount" if "/kaggle/input" in str(r) else "cli"
        print(f"[ok]   {s['name']:11s} {_n_images(r):6d} imgs  [{how}]  {r}  ({time.time()-t:.0f}s)")
    except Exception as e:
        FAILED.append(s["name"]); print(f"[FAIL] {s['name']:11s} {e}")
assert ROOTS, "no datasets available — check CFG['sources'], Add Input, and Internet"

left = _free_gb()
print(f"\n{len(ROOTS)} source(s) ready; failed: {FAILED or 'none'}")
print(f"disk left: {left:.1f} GB")
# Catch the state that produced Errno 28 in Cell 4 BEFORE it happens, while we still know
# which download caused it and can still suggest Add Input instead of a full restart.
if left < 1.0:
    which = [s["name"] for s in ROOTS if "/kaggle/input" not in str(s["path"])]
    raise SystemExit(
        f"only {left:.1f} GB free on /kaggle/working — later cells will fail with Errno 28.\n"
        f"  CLI-downloaded sources: {which or 'none'}\n"
        f"  FIX: attach those via Kaggle 'Add Input' (free), or remove them from CFG['sources'].\n"
        f"  Then restart the session to clear the partial downloads.")
