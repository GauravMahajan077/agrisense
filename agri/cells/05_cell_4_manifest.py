# CELL 4 — manifest
# Class comes from the NEAREST ancestor folder that hits an alias. Works for
# Rice_Leaf_AUG/<Class>/, train/<Class>/, validation/<Class>/, and arbitrary nesting.
#
# Resolution order per folder, first hit wins:
#   1. SRC_ALIAS[source]  — per-source overrides, for names only safe in context
#   2. ALIAS              — the global table
# Unresolvable images are REPORTED under UNMAPPED, never given an invented label.
rows, unmapped = [], defaultdict(int)
via_counts = defaultdict(int)

for r in ROOTS:
    root = r["path"]
    sal = SRC_ALIAS.get(r["name"], {})

    for f in root.rglob("*"):
        if not (f.is_file() and f.suffix.lower() in IMG_EXT): continue
        parts = f.relative_to(root).parts[:-1]
        cls = via = None
        for d in reversed(parts):
            k = norm(d)
            if k in sal:
                cls, via = sal[k], f"source_alias[{r['name']}]"
                break
            if k in ALIAS:
                cls, via = ALIAS[k], "alias"
                break
        if cls is None:
            unmapped[f"{r['name']}:{'/'.join(parts) or '<root>'}"] += 1
        else:
            rows.append({"path": str(f), "class": cls, "source": r["name"], "via": via})
            via_counts[via] += 1

man = pd.DataFrame(rows)
if man.empty:
    raise SystemExit("manifest empty — CFG['alias'] does not match your folder names")
man.to_csv(OUT / "manifest_raw.csv", index=False)

print(f"manifest: {len(man)} images from {man['source'].nunique()} source(s)\n")
if unmapped:
    print("!! UNMAPPED folders — no label was invented. Add the folder to CFG['alias'], or")
    print("   CFG['source_alias'] if the name is only safe for one source:")
    for k, v in sorted(unmapped.items(), key=lambda x: -x[1])[:25]:
        print(f"     {v:6d}  {k}")
    print(f"     ({sum(unmapped.values())} images excluded from training)")
    print()
print("resolved by: " + ", ".join(f"{k}={v}" for k, v in sorted(via_counts.items())))
print()
print(man.groupby(["source", "class"]).size().unstack(fill_value=0).to_string())

# Per-class source coverage. A class backed by ONE source is the weakest evidence in the set:
# if that source's labels are wrong the class is wrong, and no metric here would say so.
_cov = man.groupby("class")["source"].nunique()
_single = sorted(_cov[_cov == 1].index)
print(f"\nclasses backed by a single source: {_single or 'none'}")
for c in _single:
    print(f"   {c:<24} <- {sorted(man[man['class'] == c]['source'].unique())}")
