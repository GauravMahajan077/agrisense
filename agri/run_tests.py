"""Run the whole test suite. Run from the agri/ directory:  python -B run_tests.py

Kept as a script rather than a shell loop on purpose: PowerShell `$(...)` subshells
interleaving with piped python hang intermittently, and a suite that only sometimes reports is
worse than no suite.
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
TEMP = pathlib.Path(r"C:\Users\gaura\AppData\Local\Temp\opencode")

LOCAL = [
    "test_split_strat.py",
    "test_pip_names.py",
]
TEMP_SUITE = [
    "test_cell6_leak.py",
    "test_cell5_diag.py",
    "test_source_alias.py",
    "test_alias_guard.py",
    "test_cell3.py",
]

fails = []
for name in LOCAL + TEMP_SUITE:
    p = HERE / name if name in LOCAL else TEMP / name
    if not p.exists():
        print(f"  SKIP  {name} (missing)")
        continue
    r = subprocess.run([sys.executable, "-B", str(p)], cwd=HERE, capture_output=True, text=True)
    last = (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr).strip() else "?"
    ok = r.returncode == 0 and "ALL PASS" in last
    print(f"  {'PASS' if ok else 'FAIL'}  {name:26s} {last[:70]}")
    if not ok:
        fails.append(name)
        print((r.stdout or r.stderr)[-1200:])

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
