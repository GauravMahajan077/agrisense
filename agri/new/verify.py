"""Single source of truth for verifying the Phase 5 restructure.

Run from the agri/new/ directory:  python -B verify.py

Checks:
  * agrisense.py            — the module: parses, Pipeline has all 21 stage methods,
                              run() calls them in order, no __main__ pipeline block,
                              inference paths never divide by 255, lazy imports.
  * agrisense_notebook.py   — the 3-cell wrapper: 4 cells (1 md + 3 code), Cell 1 CFG
                              AST-matches DEFAULT_CFG, Cell 2 has the include directive,
                              Cell 3 calls run(CFG).
  * cells/                  — generated from the notebook: 3 code cells, all parse,
                              MODULE cell is not stale, CONFIG cell matches. Helper cells
                              (04 download, 05 test) may also live here — they are not
                              notebook cells and are excluded from the count.
  * ../old/agrisense_kaggle.py — legacy 23-cell notebook, kept on purpose, still parses.
  * field_test.py           — the field-test loader (Phase 4).
  * README.md               — key claims.
"""
import ast
import pathlib
import re
import sys

fails = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))
    if not cond:
        fails.append(name)


def find_assign(tree, name):
    """Value node of the first top-level `name = <value>` assignment, or None."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == name:
                    return node.value
    return None


def method_source(src, cls, method):
    """Source text of a method body, or '' if not found."""
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return ast.get_source_segment(src, item) or ""
    return ""


def run_calls(tree):
    """Ordered list of self.<attr>() calls in Pipeline.run()."""
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Pipeline":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "run":
                    out = []
                    for stmt in item.body:
                        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                                and isinstance(stmt.value.func, ast.Attribute)
                                and isinstance(stmt.value.func.value, ast.Name)
                                and stmt.value.func.value.id == "self"):
                            out.append(stmt.value.func.attr)
                    return out
    return []


def has_div255(src):
    """True if the code actually divides by 255 (comments don't count)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            r = node.right
            if isinstance(r, ast.Constant) and r.value in (255, 255.0):
                return True
    return False


STAGES = ["env", "deps", "fetch", "manifest", "dedupe", "split", "min_class",
          "class_weights", "preload", "pipelines", "macro_f1", "make_model",
          "callbacks", "train", "val_report", "test_report", "held_out",
          "crawl", "export", "bundle", "cleanup"]

print("=== module: agrisense.py ===")
mod = pathlib.Path("agrisense.py")
check("agrisense.py exists", mod.exists())
if mod.exists():
    msrc = mod.read_text(encoding="utf-8")
    try:
        mtree = ast.parse(msrc)
        check("module parses", True)
    except SyntaxError as e:
        mtree = None
        check("module parses", False, str(e))

    check("PIPELINE_VERSION is 3.1.2", 'PIPELINE_VERSION = "3.1.2"' in msrc)
    check("DEFAULT_CFG present", find_assign(mtree, "DEFAULT_CFG") is not None if mtree else False)

    if mtree:
        cls = next((n for n in mtree.body if isinstance(n, ast.ClassDef) and n.name == "Pipeline"), None)
        check("Pipeline class exists", cls is not None)
        if cls:
            methods = {n.name for n in cls.body if isinstance(n, ast.FunctionDef)}
            missing = [s for s in STAGES if s not in methods]
            check("Pipeline has all 21 stage methods", not missing, f"missing {missing}")
            check("run() calls stages in order", run_calls(mtree) == STAGES,
                  f"got {run_calls(mtree)}")

        # No `if __name__ == "__main__":` block: in a notebook __name__ == "__main__",
        # so a pipeline block there would fire on paste. The module must not have one.
        has_main = any(isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
                       and any(isinstance(c, ast.Constant) and c.value == "__main__"
                               for c in ast.walk(n.test))
                       for n in mtree.body)
        check("no __main__ pipeline block", not has_main)

        # Every entry point prints PIPELINE_VERSION so a stale paste is visible.
        for m in ("run", "train", "export", "bundle"):
            check(f"{m}() prints PIPELINE_VERSION", "PIPELINE_VERSION" in method_source(msrc, "Pipeline", m))

    # smoke mode is wired through the stages that must skip in smoke.
    check("smoke mode present", '"smoke"' in msrc)
    check("smoke caps manifest", 'self.cfg["smoke"]' in method_source(msrc, "Pipeline", "manifest"))
    check("smoke shortens epochs", 'self.cfg["smoke"]' in method_source(msrc, "Pipeline", "train"))
    check("smoke skips export", 'self.cfg["smoke"]' in method_source(msrc, "Pipeline", "export"))
    check("smoke skips bundle", 'self.cfg["smoke"]' in method_source(msrc, "Pipeline", "bundle"))
    check("smoke skips crawl", 'self.cfg["smoke"]' in method_source(msrc, "Pipeline", "crawl"))

    # Inference paths feed 0-255 floats (EfficientNet preprocesses internally). The ONLY
    # /255 in the module must be the training augmentation (_aug), never held_out/crawl/export.
    aug = method_source(msrc, "Pipeline", "_aug")
    check("training aug divides by 255 (0..1 colour ops)", has_div255(aug))
    for m in ("held_out", "crawl", "export", "bundle"):
        src = method_source(msrc, "Pipeline", m)
        check(f"{m}() never divides by 255", not has_div255(src))

    # Lazy imports: imagehash/ddgs/requests are pip-installed by deps(), so they must be
    # imported at point of use, not at module top (a fresh Kaggle session has none of them).
    check("imagehash imported lazily in _d4_keys", "import imagehash" in method_source(msrc, "Pipeline", "_d4_keys"))
    check("ddgs imported lazily", "from ddgs import DDGS" in method_source(msrc, "Pipeline", "ddg_urls"))
    check("requests imported lazily in crawl", "import requests" in method_source(msrc, "Pipeline", "crawl"))

    # Behavior-preservation details from the audit.
    check("SaveBestF1.best is a class attribute", "best = -1.0" in method_source(msrc, "SaveBestF1", "__init__") or "best = -1.0" in msrc)
    check("BudgetStop takes constructor args", "def __init__(self, t_train0, budget_min)" in msrc)
    check("model stage is make_model (no self.model shadow)", "def make_model(self)" in msrc)
    check("MacroF1 is serializable", "register_keras_serializable" in msrc)
    check("dedupe has brute-force D4 matrix", "d4_dist_matrix" in msrc)
    check("dedupe has brute_clusters", "brute_clusters" in msrc)
    check("dedupe has cross_class_mask", "cross_class_mask" in msrc)
    check("dedupe has cross-source counter", "clusters spanning >1 source" in msrc)
    check("dedupe has pairs-merged label", "same-class key pairs" in msrc)
    check("dedupe has largest-cluster composition", "largest cluster composition" in msrc)
    check("dedupe has source_alias pair report", "check the source_alias mapping" in msrc)
    check("split moves cross-class to train", "moved from val/test to train" in msrc)
    check("split uses D-based leak_scan", "self.leak_scan(self.D, self.man" in msrc)
    check("manifest routes held-out source", "man_held" in msrc and "held_out_source" in msrc)
    check("held_out has headline eval", "SOURCE-HELD-OUT EVAL" in msrc)
    check("held_out has bootstrap CI", "bootstrap 95% CI" in msrc)
    check("held_out has abstain", "abstain@" in msrc)
    check("held_out writes source_held_out.csv", "source_held_out.csv" in msrc)
    check("crawl has abstain line", "abstain@" in method_source(msrc, "Pipeline", "crawl"))
    check("export contract ships abstain_threshold", "abstain_threshold" in method_source(msrc, "Pipeline", "export"))
    check("source_alias guard present", "shadow" in msrc)
    check("DEFAULT_CFG has held_out_source", '"held_out_source"' in msrc)
    check("DEFAULT_CFG has abstain_threshold", '"abstain_threshold"' in msrc)
    check("DEFAULT_CFG has bootstrap_iters", '"bootstrap_iters"' in msrc)

    # Phase 5.1 review fixes (reproduced before fixing, all four bugs + hygiene).
    check("jit defaults to False (XLA off)", '"jit": False' in msrc)
    check("crawl contact lives in CFG", '"contact"' in msrc and "gau.mah077@gmail.com" in msrc)
    check("crawl UA reads contact from CFG",
          "self.cfg['crawl']['contact']" in method_source(msrc, "Pipeline", "crawl"))
    check("dedupe stores xclass mask on manifest",
          'self.man["xclass"]' in method_source(msrc, "Pipeline", "dedupe"))
    check("dedupe stores keys8 for overlap check",
          "self.keys8" in method_source(msrc, "Pipeline", "dedupe"))
    check("min_class re-applies cross-class move",
          'self.man["xclass"]' in method_source(msrc, "Pipeline", "min_class"))
    check("min_class guards empty CLASSES",
          "min_class dropped every class" in method_source(msrc, "Pipeline", "min_class"))
    check("min_class uses mc in smoke",
          'mc = 10 if self.cfg["smoke"]' in method_source(msrc, "Pipeline", "min_class"))
    check("manifest smoke cap is per source+class",
          'groupby(["source", "class"]).head(40)' in method_source(msrc, "Pipeline", "manifest"))
    check("train resets SaveBestF1.best",
          "SaveBestF1.best = -1.0" in method_source(msrc, "Pipeline", "train"))
    check("held_out has D4 overlap report",
          "held-out vs train D4 overlap" in method_source(msrc, "Pipeline", "held_out"))

    # Phase 5.2 — Kaggle smoke-run fixes (use-after-delete crash + XLA + mounts).
    check("preload decodes+hashes held-out before freeing",
          "_preload_held" in method_source(msrc, "Pipeline", "preload"))
    check("held_out uses preloaded held-out pixels",
          "self.Xh[keep]" in method_source(msrc, "Pipeline", "held_out"))
    check("held_out uses stored held-out keys",
          "self.HK[keep & self.HOK]" in method_source(msrc, "Pipeline", "held_out"))
    check("train forces jit_compile off",
          '{"jit_compile": bool(self.JIT)}' in method_source(msrc, "Pipeline", "train"))
    check("_mounted searches nested layouts",
          '"*/*/*"' in method_source(msrc, "Pipeline", "_mounted"))
    check("cleanup frees held-out pixels",
          "self.Xh = None" in method_source(msrc, "Pipeline", "cleanup"))

    # Phase 5.3 — full-run crash: cross-class move split a multi-member cluster across splits.
    check("split moves whole cross-class clusters to train",
          'bad_clusters = set(self.man.loc[self.cross_mask, "cluster"])'
          in method_source(msrc, "Pipeline", "split"))
    check("min_class re-applies whole-cluster xclass move",
          'bad_clusters = set(self.man.loc[self.man["xclass"], "cluster"])'
          in method_source(msrc, "Pipeline", "min_class"))

print("\n=== notebook: agrisense_notebook.py ===")
nb = pathlib.Path("agrisense_notebook.py")
check("agrisense_notebook.py exists", nb.exists())
if nb.exists():
    nsrc = nb.read_text(encoding="utf-8")
    try:
        ast.parse(nsrc)
        check("notebook source parses", True)
    except SyntaxError as e:
        check("notebook source parses", False, str(e))

    marks = [m.start() for m in re.finditer(r"(?m)^# %%(?: \[(markdown)\])?\s*$", nsrc)]
    check("4 cells (1 markdown + 3 code)", len(marks) == 4, f"got {len(marks)}")
    check("has markdown cell", "# %% [markdown]" in nsrc)
    check("Cell 2 has the include directive", "# %% include:agrisense.py" in nsrc)
    check("Cell 3 calls run(CFG)", "run(CFG)" in nsrc)

    # Cell 1 CFG must AST-match the module's DEFAULT_CFG so the editable copy cannot drift.
    if mod.exists():
        ntree = ast.parse(nsrc)
        cfg_node = find_assign(ntree, "CFG")
        mtree2 = ast.parse(msrc)
        dcfg_node = find_assign(mtree2, "DEFAULT_CFG")
        check("Cell 1 CFG exists", cfg_node is not None)
        check("module DEFAULT_CFG exists", dcfg_node is not None)
        if cfg_node is not None and dcfg_node is not None:
            check("Cell 1 CFG AST-matches DEFAULT_CFG", ast.dump(cfg_node) == ast.dump(dcfg_node))

print("\n=== generated cells ===")
cells = sorted(pathlib.Path("cells").glob("*.py"))
# Helper cells (04 download, 05 test) live in cells/ too but are NOT notebook cells.
nb_cells = [f for f in cells if any(t in f.name for t in ("cell_1_", "cell_2_", "cell_3_"))]
check("3 code cells generated", len(nb_cells) == 3, f"got {len(nb_cells)}")
for f in cells:
    try:
        ast.parse(f.read_text(encoding="utf-8"))
    except SyntaxError as e:
        check(f"{f.name} parses", False, str(e))
check("all cells parse", not any("parses" in x for x in fails))

# The MODULE cell must contain the CURRENT module, not a stale copy. The cell body is the
# header comment + the module source, so the module source must be a suffix of the cell.
mod_cell = next((f for f in cells if "cell_2_module" in f.name), None)
check("MODULE cell exists", mod_cell is not None)
if mod_cell and mod.exists():
    body = mod_cell.read_text(encoding="utf-8").rstrip("\n")
    check("MODULE cell is not stale (ends with current module)",
          body.endswith(msrc.rstrip("\n")))

cfg_cell = next((f for f in cells if "cell_1_config" in f.name), None)
check("CONFIG cell exists", cfg_cell is not None)
if cfg_cell:
    t = cfg_cell.read_text(encoding="utf-8")
    check("CONFIG cell has CFG dict", "CFG = {" in t)
    check("CONFIG cell has the merge guard", "merge_narrow_brown" in t)

run_cell = next((f for f in cells if "cell_3_run" in f.name), None)
check("RUN cell exists", run_cell is not None)
if run_cell:
    check("RUN cell calls run(CFG)", "run(CFG)" in run_cell.read_text(encoding="utf-8"))

print("\n=== legacy: agrisense_kaggle.py ===")
leg = pathlib.Path("../old/agrisense_kaggle.py")
check("legacy notebook kept", leg.exists())
if leg.exists():
    lsrc = leg.read_text(encoding="utf-8")
    try:
        ast.parse(lsrc)
        check("legacy notebook parses", True)
    except SyntaxError as e:
        check("legacy notebook parses", False, str(e))
    lmarks = [m.start() for m in re.finditer(r"(?m)^# %%", lsrc)]
    check("legacy has 23 cells", len(lmarks) == 23, f"got {len(lmarks)}")

print("\n=== field-test loader ===")
ft = pathlib.Path("field_test.py")
check("field_test.py exists", ft.exists())
if ft.exists():
    try:
        ast.parse(ft.read_text(encoding="utf-8"))
        check("field_test.py parses", True)
    except SyntaxError as e:
        check("field_test.py parses", False, str(e))
    tft = ft.read_text(encoding="utf-8")
    check("loader never divides by 255", "/ 255" not in tft and "/255" not in tft)
    check("loader uses LANCZOS resize", "LANCZOS" in tft)
    check("loader reads the contract", "recommended_input_size" in tft)
    check("loader has abstain", "abstain_threshold" in tft)

print("\n=== README ===")
rm = pathlib.Path("README.md").read_text(encoding="utf-8")
check("README documents 6 classes", "Taxonomy: 6 classes" in rm)
check("README documents the source_alias rationale", "source_alias" in rm)
check("README shows Sheath_Blight as single-source", "| **1** |" in rm)

print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)