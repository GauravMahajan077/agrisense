# CELL 0 — environment probe
# Laptop-safe on purpose: no hard exit, no Kaggle-only paths. You can dry-run this on your
# machine to sanity-check imports. It will refuse to *train* without a GPU, not refuse to start.
import os, sys, time, shutil, tempfile, subprocess
from pathlib import Path

# NOTE: PYTHONHASHSEED is read by the interpreter at startup, so setting it here is a no-op.
# It is set only because Kaggle lets you control it from the environment panel. Nothing in
# this pipeline depends on it (all seeding is explicit).


os.environ.setdefault("PYTHONHASHSEED", "1337")
_NC = max(1, os.cpu_count() or 4)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hides the absl/CUDA banner noise
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(min(8, _NC)))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
T0 = time.time()

# Bumped on every intentional pipeline change. Printed at every entry point (start, train,
# export, bundle) so a stale paste is visible in the log instead of silently shipping.
PIPELINE_VERSION = "2.0.0"

import numpy as np, tensorflow as tf

KAGGLE = Path("/kaggle").is_dir()
WORK = Path("/kaggle/working") if KAGGLE else Path(tempfile.gettempdir()) / "agrisense"
WORK.mkdir(parents=True, exist_ok=True)

def _ram_gb():
    """Cross-platform RAM probe. Never raises — a diagnostic must not be able to kill a run."""
    try:
        import psutil
        v = psutil.virtual_memory(); return v.total / 1e9, v.available / 1e9
    except Exception:
        pass
    try:
        d = {l.split(":")[0].strip(): int(l.split()[1])
             for l in open("/proc/meminfo") if ":" in l}
        return d["MemTotal"] / 1e6, d["MemAvailable"] / 1e6
    except Exception:
        return float("nan"), float("nan")

def _free_gb(p=None):
    """Free bytes on the filesystem holding WORK, in GB. Never raises.

    Disk headroom is the binding constraint on Kaggle: /kaggle/working is ~20 GB and one
    source (shayanriyaz) is an 8 GB zip that expands to another 8 GB. Every download in
    Cell 3 is therefore gated on this number."""
    try:
        return shutil.disk_usage(p or WORK).free / 1e9
    except Exception:
        return float("nan")

RAM_GB, RAM_AVAIL_GB = _ram_gb()
free_gb = _free_gb()

def _tmp_gb():
    """Free space on the system temp dir.

    Reported separately because it is the resource that actually failed: the `kaggle` CLI
    stages its download through temp, and on Kaggle /tmp is a much smaller filesystem than
    /kaggle/working. Cell 3 now redirects TMPDIR at WORK so the two agree — but the sizes are
    worth seeing, because "19.5 GB free on /kaggle/working" coexisting with Errno 28 on an
    8 GB download is otherwise a mystery.
    """
    import tempfile as _tf
    for cand in (_tf.gettempdir(), "/tmp"):
        try:
            g = shutil.disk_usage(cand).free / 1e9
            return f"{g:.1f} GB free on {_tf.gettempdir()}"
        except Exception:
            continue
    return "temp space unknown"

def _dir_gb(d):
    """Recursive on-disk size of a tree, in GB. Used to report what we just freed."""
    try:
        return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file()) / 1e9
    except Exception:
        return 0.0

print(f"PIPELINE_VERSION {PIPELINE_VERSION} | python {sys.version.split()[0]} | cpus {_NC} | "
      f"platform {'kaggle' if KAGGLE else 'local'}")
print(f"tf {tf.__version__} | numpy {np.__version__} | keras {tf.keras.__version__}")
print(f"RAM {RAM_GB:.1f} GB total / {RAM_AVAIL_GB:.1f} GB available | disk {free_gb:.1f} GB free in WORK")
print(f"temp: {_tmp_gb()} (Cell 3 redirects TMPDIR here, so WORK is the only limit)")
print(f"work dir: {WORK}")

gpus = tf.config.list_physical_devices("GPU")
# Best-effort memory growth. One effect that matters here: it stops TF pre-allocating all
# 13.7 GB per T4, which is what turns a late allocation into "OOM at the last epoch" instead
# of a clean failure. Must run before any GPU op; if it fails we carry on.
#
# NOTE: it does NOT enable get_memory_info(). That was my earlier claim and it was wrong —
# TF still raises ValueError("Memory statistics not tracked") because the BFC allocator needs
# separate instrumentation. We read VRAM from nvidia-smi instead, which always works here.
for g in gpus:
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print(f"  set_memory_growth({g.name}) refused: {type(e).__name__}")

def _vram_lines():
    """Per-GPU total/used VRAM in MB, via nvidia-smi. Empty list if unavailable."""
    try:
        r = subprocess.run(["nvidia-smi",
                            "--query-gpu=index,name,memory.total,memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        if r.returncode == 0:
            return [ln.strip() for ln in r.stdout.strip().splitlines() if ln.strip()]
    except Exception:
        pass
    return []

_vram = _vram_lines()
if _vram:
    for ln in _vram:
        parts = [p.strip() for p in ln.split(",")]
        # Never let a display probe kill the run: a GPU name containing a comma, or a driver
        # that changes the column set, must degrade to a printed line, not an exception.
        try:
            idx, name, total, used = parts
            print(f"  GPU {idx}: {name}  VRAM {int(total)/1024:.1f} GB total / "
                  f"{int(used)/1024:.2f} GB used")
        except Exception:
            print(f"  GPU (unparsed nvidia-smi line): {ln}")
else:
    print("  (nvidia-smi unavailable — VRAM figures omitted, training unaffected)")
print(f"GPUs detected: {len(gpus)}")
if not gpus:
    print("!! No GPU. Cells 0-9 still run; Cell 14 will be very slow or fail.")
    print("!! On Kaggle: Settings -> Accelerator -> GPU T4 x2 -> Save.")
print(f"disk headroom: {_free_gb():.1f} GB")
