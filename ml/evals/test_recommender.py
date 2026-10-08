"""Smoke test for recommender: 5 disease labels + fert/weed + registry gate.

Exit 0 = all safety assertions hold; exit 1 = assertion failed (CI-ready).
"""
import sys

sys.path.insert(0, "ml/src")
from recommender import Context, gate_banned, load_rules, recommend, score_pesticide  # noqa: E402

rules = load_rules()
print("JSON OK:", {k: (len(v) if isinstance(v, list) else "obj") for k, v in rules.items()})
registry = rules["registry"]["entries"]
failures: list[str] = []


def brief(d, label):
    pest = d["pesticide"]
    name = label or "(no disease)"
    if not pest:
        print(f"  {name:22} -> pesticide: NONE, needs_expert={d['needs_expert']}, abstain={d['abstain']}")
    else:
        top = pest[0]
        reg = top.get("registry") or {}
        print(f"  {name:22} -> top: {top['item']} @ {top['score']} | "
              f"{top['source']['pdf']}:{top['source']['line']} | {len(pest)} match(es) | "
              f"registry={reg.get('status')}/{reg.get('formulation_match')}")


print("--- DISEASE QUERIES ---")
for lbl in ["Bacterial_Leaf_Blight", "Brown_Spot", "Leaf_Blast",
            "Leaf_Scald", "Sheath_Blight"]:
    out = recommend(Context(disease_label=lbl), rules)
    brief(out, lbl)
    # SAFETY: banned chemical must never surface, payload must carry the flag list
    surf = [p["item"] for p in out["pesticide"] + out["weed"] if "Dichlorvos" in p["item"]]
    if surf:
        failures.append(f"{lbl}: Dichlorvos surfaced: {surf}")
    if not isinstance(out.get("safety_flags"), list):
        failures.append(f"{lbl}: safety_flags missing")
    if out.get("safety_flags") and not out["needs_expert"]:
        failures.append(f"{lbl}: safety flags must force needs_expert")

print("--- FERTILIZER + WEED (district=Ratnagiri, soil_n=90) ---")
d = recommend(Context(district="Ratnagiri", soil_n=90), rules)
for f in d["fertilizer"]:
    print("  fert:", f["item"], f["score"], f["source"])
for w in d["weed"]:
    reg = w.get("registry") or {}
    print("  weed:", w["item"], w["score"], w["source"],
          f"| registry={reg.get('status')} rice_restriction={reg.get('restriction_mentions_rice')}")
    if w.get("registry") is None:
        failures.append(f"weed '{w['item']}': registry block missing")
print("  abstain:", d["abstain"], "coverage:", d["coverage"], "needs_expert:", d["needs_expert"])

print("--- BANNED GATE (forced match) ---")
fp_rules = rules["pesticide"]
dv = next(r for r in fp_rules if "Dichlorvos" in r["chemical"])
forced = score_pesticide(dv, Context(disease_label="_".join(dv["target"].split())))
flags: list = []
kept = gate_banned([forced], registry, flags) if forced else []
print("  forced match:", forced.item if forced else None, "-> kept:", len(kept),
      "| flags:", [f["item"] for f in flags])
if not forced or kept or not flags or not flags[0]["source"].get("line"):
    failures.append("banned gate did not remove+flag forced Dichlorvos match")

print("RESULT:", "OK" if not failures else f"FAILED: {failures}")
sys.exit(1 if failures else 0)
