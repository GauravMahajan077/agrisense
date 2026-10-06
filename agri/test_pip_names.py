"""Test _pip's import-name -> PyPI-name mapping and its post-install re-probe.

Regression: _pip collected the IMPORT names into `missing` and pip-installed those verbatim.
"PIL" is the import name of "Pillow"; there is no PyPI package called "PIL", so on a clean
environment the install fails and the pipeline dies on `import PIL`. The comment claimed this
was handled. It was not.

Run: python -B test_pip_names.py
"""
import importlib.util
import subprocess as _sp
import sys
from unittest import mock

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def make_pip(mapping, reprobe=True):
    """Copy of the pipeline's _pip. reprobe=False reproduces the OLD bug faithfully."""
    def _pip(*pkgs, required=True):
        _PYPI = mapping
        missing = [p for p in pkgs if importlib.util.find_spec(p) is None]
        if not missing:
            print(f"present: {' '.join(pkgs)}")
            return None
        names = [_PYPI.get(p, p) for p in missing]
        _sp.run([sys.executable, "-m", "pip", "install", "-q", *names], check=True)
        if reprobe:
            importlib.invalidate_caches()
            for p in missing:
                if importlib.util.find_spec(p) is None:
                    raise ImportError(f"{p} still not importable after installing {' '.join(names)}")
        return names
    return _pip


print("=== 1. the OLD behaviour: pip is asked to install 'PIL' ===")
calls = []
_pip_old = make_pip({}, reprobe=False)
with mock.patch("importlib.util.find_spec", side_effect=lambda n: None), \
     mock.patch.object(_sp, "run", side_effect=lambda a, **k: calls.append(a[3:])):
    _pip_old("PIL", "imagehash")
flat = calls[0]
check("old code asks pip for 'PIL'", "PIL" in flat, str(flat))
check("old code never asks for 'Pillow'", "Pillow" not in flat, str(flat))
print(f"  -> pip would run: pip install {' '.join(map(str, flat))}")
print(f"  -> PyPI has no 'PIL' package, so this raises PackageNotFoundError on a clean env")

print("\n=== 2. the NEW behaviour: maps to the PyPI name ===")
calls = []
_pip_new = make_pip({"PIL": "Pillow"})
# First probe says "missing" (pre-install); after pip runs, the re-probe must see it present.
installed = {"done": False}


def spec_then_installed(name):
    if not installed["done"]:
        return None
    return object()


def run_then_installed(cmd, **k):
    installed["done"] = True
    calls.append(cmd[3:])


with mock.patch("importlib.util.find_spec", side_effect=spec_then_installed), \
     mock.patch.object(_sp, "run", side_effect=run_then_installed):
    names = _pip_new("PIL", "imagehash")
flat = calls[0]
check("new code asks pip for 'Pillow'", "Pillow" in flat, str(flat))
check("new code never asks pip for 'PIL'", "PIL" not in flat, str(flat))
check("non-aliased packages pass through unchanged", "imagehash" in flat, str(flat))
check("returns the resolved install names", names == ["Pillow", "imagehash"], str(names))

print("\n=== 3. post-install re-probe catches a silent no-op install ===")
# pip exits 0 but the module still does not import. Must raise, not print 'installed and OK'.
_pip_fail = make_pip({"PIL": "Pillow"})
seq = {"n": 0}


def fake_spec(name):
    # first probe (pre-install) says missing; post-install probe also says missing
    return None


with mock.patch("importlib.util.find_spec", side_effect=fake_spec), \
     mock.patch.object(_sp, "run", lambda a, **k: None):
    try:
        _pip_fail("PIL")
        check("raises when install did not take effect", False, "returned normally")
    except ImportError as e:
        check("raises when install did not take effect", True)
        check("message names both the import name and the PyPI name",
              "PIL" in str(e) and "Pillow" in str(e), str(e))

print("\n=== 4. when the module IS present, nothing is installed ===")
calls = []
_pip_present = make_pip({"PIL": "Pillow"})
with mock.patch("importlib.util.find_spec", side_effect=lambda n: object()), \
     mock.patch.object(_sp, "run", side_effect=lambda a, **k: calls.append(a)):
    _pip_present("PIL", "imagehash")
check("no pip invocation when everything is present", not calls, str(calls))

print("\n=== 5. the mapping is the one the pipeline actually uses ===")
import re
import pathlib
src = pathlib.Path("agrisense_kaggle.py").read_text(encoding="utf-8")
m = re.search(r'_PYPI = \{([^}]*)\}', src)
check("pipeline defines _PYPI", m is not None)
check("pipeline maps PIL -> Pillow", m and re.search(r'"PIL"\s*:\s*"Pillow"', m.group(1)),
      m.group(1) if m else "")
check("pipeline calls _pip('PIL', 'imagehash')", '_pip("PIL", "imagehash")' in src)
check("post-install re-probe exists", "still not importable after installing" in src)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
