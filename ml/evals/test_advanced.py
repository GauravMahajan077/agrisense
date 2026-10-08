"""Advanced test suite for the ENTIRE AgriSense ML stack.

Goes beyond rule_validator.py (KB/recommender) and test_recommender.py (smoke):
covers risk_engine, price, advisory, abstain, crop_leaf, cross-module
integration, plus deeper recommender edge cases, fuzzing, and determinism.

Run:  python -m pytest ml/evals/test_advanced.py -v
Exit: 0 = all pass, 1 = any failure (CI-ready).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "ml" / "src"
sys.path.insert(0, str(SRC))

from recommender import Context, gate_banned, load_rules, recommend, score_pesticide  # noqa: E402
from risk_engine import RiskInput, load_artifact, raw_score, risk  # noqa: E402
from price import MarketSnapshot, contract, signal, trend_pct  # noqa: E402
from advisory import build  # noqa: E402
from abstain import (  # noqa: E402
    best_threshold, decide, risk_coverage, temperature_scale, top2,
)
import crop_leaf  # noqa: E402

RISK_XGB = ROOT / "ml" / "risk_xgb"
sys.path.insert(0, str(RISK_XGB))
from risk_xgb import predict as xgb_predict  # noqa: E402

RULES = load_rules()
REGISTRY = RULES["registry"]["entries"]
DISEASES = ["Bacterial_Leaf_Blight", "Brown_Spot", "Leaf_Blast",
            "Leaf_Scald", "Sheath_Blight"]
BANNED = {"Dichlorvos"}


# ─────────────────────────── A. Recommender ───────────────────────────

class TestRecommenderCoverage:
    @pytest.mark.parametrize("label", DISEASES)
    def test_disease_returns_cited_chemical(self, label):
        out = recommend(Context(disease_label=label), RULES)
        assert out["pesticide"], f"{label}: no pesticide returned"
        for p in out["pesticide"]:
            assert p["source"]["pdf"] and p["source"]["line"] is not None, \
                f"{label}: {p['item']} missing provenance"

    @pytest.mark.parametrize("label", DISEASES)
    def test_every_pesticide_has_registry_block(self, label):
        out = recommend(Context(disease_label=label), RULES)
        for p in out["pesticide"]:
            rg = p.get("registry")
            assert rg, f"{label}: {p['item']} missing registry block"
            assert rg["status"] in {"banned", "restricted", "registered", "not_listed"}
            assert rg["formulation_match"] in {"exact", "numeric", "active_only",
                                               "unverified", "none"}
            assert str(rg["authority"]).startswith("CIB&RC")

    def test_no_banned_chemical_surfaces_anywhere(self):
        for label in DISEASES + ["Healthy", None]:
            out = recommend(Context(disease_label=label), RULES)
            surf = [p["item"] for p in out["pesticide"] + out["weed"]
                    if any(b in p["item"] for b in BANNED)]
            assert not surf, f"{label}: banned surfaced: {surf}"

    def test_safety_flags_force_needs_expert(self):
        for label in DISEASES:
            out = recommend(Context(disease_label=label), RULES)
            if out["safety_flags"]:
                assert out["needs_expert"] is True

    def test_abstain_implies_needs_expert(self):
        for label in DISEASES + [None]:
            out = recommend(Context(disease_label=label), RULES)
            if out["abstain"]:
                assert out["needs_expert"] is True

    def test_top_k_respected(self):
        out = recommend(Context(disease_label="Sheath_Blight"), RULES, top_k=3)
        assert len(out["pesticide"]) <= 3
        assert len(out["fertilizer"]) <= 3


class TestRecommenderLeakage:
    """ALL-distinctive-token matching must isolate disease classes."""

    LEAK = [
        ("Bacterial_Leaf_Blight", ["Tricyclazole", "Isoprothiolane", "Hexaconazole"]),
        ("Sheath_Blight", ["Streptocycline"]),
        ("Leaf_Blast", ["Streptocycline", "Tricyclazole", "Carbendazim 5% GR"]),
        ("Brown_Spot", ["Tricyclazole", "Hexaconazole", "Streptocycline"]),
        ("Leaf_Scald", ["Streptocycline", "Tricyclazole", "Hexaconazole"]),
    ]

    @pytest.mark.parametrize("label,banned", LEAK)
    def test_no_cross_disease_leakage(self, label, banned):
        got = [p["item"] for p in recommend(Context(disease_label=label), RULES)["pesticide"]]
        bad = [g for g in got if any(b in g for b in banned)]
        assert not bad, f"{label}: leaked {bad}"

    def test_healthy_label_no_pesticide(self):
        out = recommend(Context(disease_label="Healthy"), RULES)
        assert not out["pesticide"]

    def test_no_disease_no_pesticide(self):
        out = recommend(Context(district="Ratnagiri", soil_n=90), RULES)
        assert not out["pesticide"]

    def test_unknown_disease_abstains(self):
        out = recommend(Context(disease_label="Mystery_Blight"), RULES)
        assert not out["pesticide"]
        assert out["needs_expert"] is True


class TestRecommenderTokenMatching:
    def test_case_insensitive(self):
        a = recommend(Context(disease_label="brown_spot"), RULES)
        b = recommend(Context(disease_label="Brown_Spot"), RULES)
        assert [p["item"] for p in a["pesticide"]] == [p["item"] for p in b["pesticide"]]

    def test_underscore_vs_space_equivalent(self):
        a = recommend(Context(disease_label="Leaf_Scald"), RULES)
        b = recommend(Context(disease_label="Leaf Scald"), RULES)
        assert [p["item"] for p in a["pesticide"]] == [p["item"] for p in b["pesticide"]]

    def test_stopword_only_label_no_match(self):
        # "leaf" alone is a stopword -> must not match any pesticide
        out = recommend(Context(disease_label="Leaf"), RULES)
        assert not out["pesticide"]

    def test_shared_token_does_not_bridge(self):
        # 'blight' is shared by BLB and sheath blight; ALL tokens must match
        blb = [p["item"] for p in recommend(Context(disease_label="Bacterial_Leaf_Blight"), RULES)["pesticide"]]
        sb = [p["item"] for p in recommend(Context(disease_label="Sheath_Blight"), RULES)["pesticide"]]
        assert not set(blb) & set(sb), f"shared 'blight' bridged classes: {set(blb) & set(sb)}"

    def test_off_label_notes_on_scald(self):
        out = recommend(Context(disease_label="Leaf_Scald"), RULES)
        for p in out["pesticide"]:
            notes = str(p["detail"].get("notes", ""))
            assert "off-label" in notes.lower(), f"{p['item']}: scald rule missing off-label note"

    def test_in_label_brown_spot_has_no_off_label_note(self):
        out = recommend(Context(disease_label="Brown_Spot"), RULES)
        for p in out["pesticide"]:
            notes = str(p["detail"].get("notes", ""))
            assert "off-label" not in notes.lower(), f"{p['item']}: brown-spot rule wrongly marked off-label"


class TestRecommenderIPM:
    def test_ipm_is_first_payload_key(self):
        out = recommend(Context(district="Ratnagiri", soil_n=90), RULES)
        assert list(out.keys())[0] == "ipm"

    def test_ipm_actions_returned_for_rice(self):
        out = recommend(Context(district="Ratnagiri", soil_n=90), RULES)
        assert out["ipm"], "rice query returned zero IPM actions"

    def test_ipm_not_returned_for_non_rice(self):
        out = recommend(Context(crop="wheat", district="Ratnagiri", soil_n=90), RULES)
        assert not out["ipm"]

    def test_ipm_json_is_chemical_free(self):
        import re
        chem_pat = re.compile(
            r"kg\s*/\s*ha|g\s*/\s*(10|litre|liter)|ml\s*/\s*(10|litre|liter)|\bppm\b"
            r"|\bEC\b|\bWP\b|\bSP\b|\bSC\b|\bWDG\b", re.I)
        known = re.compile(
            r"tricyclazole|hexaconazole|isoprothiolane|cartap|fipronil|acephate"
            r"|chlorantraniliprole|quinalphos|chlorpyrifos|streptocycline|pretilachlor"
            r"|oxadiargyl|carbendazim|dimethoate|monocrotophos|mancozeb|validamycin"
            r"|edifenphos|azoxystrobin|picoxystrobin", re.I)
        for a in RULES["ipm"]:
            text = f"{a.get('action', '')} {a.get('timing', '')}"
            assert not chem_pat.search(text), f"IPM action has dose: {text}"
            assert not known.search(text), f"IPM action names a chemical: {text}"

    def test_ipm_actions_have_provenance(self):
        for a in RULES["ipm"]:
            assert a.get("source_pdf") and a.get("line") is not None


class TestRecommenderFertilizerWeed:
    def test_fertilizer_zone_match(self):
        out = recommend(Context(district="Ratnagiri", soil_n=90), RULES)
        assert len(out["fertilizer"]) >= 3

    def test_fertilizer_soil_n_boost(self):
        from recommender import score_fertilizer
        n100 = next(r for r in RULES["fertilizer"]
                    if r.get("n_kg_ha") == 100 and r.get("p2o5_kg_ha") == 40)
        lo = score_fertilizer(n100, Context(soil_n=80)).score
        hi = score_fertilizer(n100, Context(soil_n=300)).score
        assert lo >= hi, f"low-N score {lo} should rank >= high-N {hi}"

    def test_weed_crop_scoped(self):
        out = recommend(Context(crop="wheat", district="Ratnagiri"), RULES)
        assert not out["weed"]

    def test_weed_stage_filter(self):
        out = recommend(Context(district="Ratnagiri", stage="pre_sowing"), RULES)
        for w in out["weed"]:
            timing = str(w["detail"].get("timing", ""))
            assert "pre" in timing or "DAS" in timing, \
                f"pre_sowing stage returned post-emergence weed: {w['item']} ({timing})"

    def test_weed_registry_block(self):
        out = recommend(Context(district="Ratnagiri"), RULES)
        for w in out["weed"]:
            rg = w.get("registry")
            assert rg, f"weed {w['item']} missing registry block"
            assert rg["status"] in {"banned", "restricted", "registered", "not_listed"}


class TestRecommenderFuzz:
    ADVERSARIAL = ["", " ", "A" * 500, "Healthy", "blast", "blight", "leaf",
                   "rice leaf blast", "Bacterial Leaf Blight", "brown spot",
                   "Sheath Blight", "Leaf Scald", "None", "null", "NaN",
                   "Bacterial_Leaf_Blight_Extra", "x" * 300]

    def test_fuzz_labels_no_crash_no_banned(self):
        for lbl in self.ADVERSARIAL:
            out = recommend(Context(disease_label=lbl), RULES)
            for p in out["pesticide"] + out["weed"]:
                assert 0.0 <= p["score"] <= 1.0
                assert not any(b in p["item"] for b in BANNED)
                assert p["source"]["pdf"] and p["source"]["line"] is not None

    def test_fuzz_random_contexts_no_crash(self):
        rng = random.Random(42)
        districts = ["Ratnagiri", "Sindhudurg", "Raigad", "Nowhere", "", "KONKAN"]
        stages = [None, "tillering", "panicle_initiation", "pre_sowing", "transplanting", ""]
        for _ in range(200):
            ctx = Context(
                district=rng.choice(districts),
                season=rng.choice(["kharif", "rabi", ""]),
                stage=rng.choice(stages),
                soil_n=rng.choice([None, -50, 0, 90, 300, 99999]),
                soil_p=rng.choice([None, 0, 40, 999]),
                soil_k=rng.choice([None, 0, 40, 999]),
                rainfall_anomaly_pct=rng.choice([None, -100, 0, 20, 40, 999]),
                disease_label=rng.choice(self.ADVERSARIAL),
            )
            out = recommend(ctx, RULES)
            for kind in ("fertilizer", "pesticide", "weed"):
                for it in out[kind]:
                    assert 0.0 <= it["score"] <= 1.0

    def test_determinism_random_contexts(self):
        rng = random.Random(7)
        for _ in range(50):
            ctx = Context(district=rng.choice(["Ratnagiri", "Sindhudurg", None]),
                          soil_n=rng.choice([None, 90, 300]),
                          disease_label=rng.choice(DISEASES + [None]))
            a = json.dumps(recommend(ctx, RULES), sort_keys=True)
            b = json.dumps(recommend(ctx, RULES), sort_keys=True)
            assert a == b


# ─────────────────────────── B. Risk engine ───────────────────────────

class TestRiskEngine:
    def test_score_bounds(self):
        for dp in (0.0, 0.3, 0.5, 0.8, 1.0):
            for conf in (0.0, 0.5, 1.0):
                for ra in (-1.0, 0.0, 1.0):
                    for sr in (0.0, 0.5, 1.0):
                        for ys in (0.0, 0.5, 1.0):
                            out = risk(RiskInput(dp, conf, ra, sr, ys))
                            assert 0.0 <= out["score"] <= 1.0

    def test_interval_contains_score_and_bounds(self):
        out = risk(RiskInput(0.8, 0.9, 0.5, 0.7, 0.6))
        lo, hi = out["interval"]
        assert 0.0 <= lo <= out["score"] <= hi <= 1.0

    def test_priority_bands(self):
        # all features max -> sigmoid ~0.818 -> high
        assert risk(RiskInput(1.0, 1.0, 1.0, 1.0, 1.0))["priority"] == "high"
        # moderate features -> sigmoid ~0.589 -> medium
        assert risk(RiskInput(0.8, 0.8, 0.5, 0.7, 0.6))["priority"] == "medium"
        # low features -> sigmoid ~0.073 -> low
        assert risk(RiskInput(0.1, 0.1, -1.0, 0.1, 0.1))["priority"] == "low"

    def test_rainfall_anomaly_clipped(self):
        out = risk(RiskInput(0.5, 0.5, 5.0, 0.5, 0.5))
        assert out["score"] == risk(RiskInput(0.5, 0.5, 1.0, 0.5, 0.5))["score"]
        out2 = risk(RiskInput(0.5, 0.5, -5.0, 0.5, 0.5))
        assert out2["score"] == risk(RiskInput(0.5, 0.5, -1.0, 0.5, 0.5))["score"]

    def test_calibration_identity_when_none(self):
        art = {"weights": {"disease_prob": 1.6, "conf": -0.4, "rainfall_anomaly": 0.9,
                           "stage_risk": 0.7, "yield_stress": 0.6, "intercept": -1.9},
               "calibration": {"type": "none", "bins": []},
               "conformal": {"q": 0.10, "n_calib": 0},
               "model_version": "test"}
        out = risk(RiskInput(0.5, 0.5, 0.0, 0.5, 0.5), art)
        assert out["score"] == round(raw_score(
            {"disease_prob": 0.5, "conf": 0.5, "rainfall_anomaly": 0.0,
             "stage_risk": 0.5, "yield_stress": 0.5}, art["weights"]), 3)

    def test_calibration_isotonic_monotonic(self):
        art = {"weights": {"disease_prob": 1.6, "conf": -0.4, "rainfall_anomaly": 0.9,
                           "stage_risk": 0.7, "yield_stress": 0.6, "intercept": -1.9},
               "calibration": {"type": "isotonic",
                               "bins": [[0.0, 0.0], [0.3, 0.2], [0.6, 0.5], [1.0, 1.0]]},
               "conformal": {"q": 0.10, "n_calib": 0},
               "model_version": "test"}
        prev = -1.0
        for dp in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            s = risk(RiskInput(dp, 0.5, 0.0, 0.5, 0.5), art)["score"]
            assert 0.0 <= s <= 1.0
            assert s >= prev - 1e-9, f"isotonic calibration not monotonic at dp={dp}"
            prev = s

    def test_drivers_sorted_top3(self):
        out = risk(RiskInput(0.8, 0.9, 0.5, 0.7, 0.6))
        drivers = out["drivers"]
        assert len(drivers) == 3
        contribs = [abs(d["contribution"]) for d in drivers]
        assert contribs == sorted(contribs, reverse=True)
        for d in drivers:
            assert {"feature", "value", "contribution"} <= set(d)

    def test_fallback_priors_when_artifact_missing(self):
        art = load_artifact(ROOT / "ml" / "models" / "does_not_exist.json")
        assert art["model_version"] == "risk-v0-priors"
        out = risk(RiskInput(0.5, 0.5, 0.0, 0.5, 0.5), art)
        assert out["model_version"] == "risk-v0-priors"

    def test_extreme_inputs_bounded(self):
        out = risk(RiskInput(1.0, 1.0, 1.0, 1.0, 1.0))
        assert 0.0 <= out["score"] <= 1.0
        out2 = risk(RiskInput(0.0, 0.0, -1.0, 0.0, 0.0))
        assert 0.0 <= out2["score"] <= 1.0

    def test_deterministic(self):
        a = risk(RiskInput(0.7, 0.6, 0.3, 0.4, 0.5))
        b = risk(RiskInput(0.7, 0.6, 0.3, 0.4, 0.5))
        assert a == b

    def test_nan_propagates_documented_gap(self):
        # KNOWN GAP: NaN inputs are not guarded -> NaN score. Documented so a
        # future fix is caught by this test flipping.
        out = risk(RiskInput(float("nan"), 0.5, 0.0, 0.5, 0.5))
        assert out["score"] != out["score"], "NaN score should be NaN (current behavior)"


# ─────────────────────────── C. Price ───────────────────────────

def _snap(forecast, observed=None):
    return MarketSnapshot(
        commodity="rice", market="Madhavpur", state="Maharashtra",
        observed=observed or {"date": "2026-07-20", "min": 2100, "modal": 2180,
                              "max": 2260, "arrivals": 410},
        forecast=forecast, as_of="2026-07-20")


class TestPrice:
    def test_signal_wait_when_rising(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 102},
             {"date": "d3", "p50": 104}, {"date": "d4", "p50": 106},
             {"date": "d5", "p50": 108}, {"date": "d6", "p50": 110}]
        assert signal(f, cost_buffer_pct=3.0) == "wait"  # +10% > 3%

    def test_signal_sell_when_flat(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 100},
             {"date": "d3", "p50": 100}, {"date": "d4", "p50": 100},
             {"date": "d5", "p50": 100}, {"date": "d6", "p50": 100}]
        assert signal(f) == "sell"

    def test_signal_sell_when_falling(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 99},
             {"date": "d3", "p50": 98}, {"date": "d4", "p50": 97},
             {"date": "d5", "p50": 96}, {"date": "d6", "p50": 95}]
        assert signal(f) == "sell"

    def test_signal_respects_buffer(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 101},
             {"date": "d3", "p50": 102}, {"date": "d4", "p50": 103},
             {"date": "d5", "p50": 104}, {"date": "d6", "p50": 105}]
        assert signal(f, cost_buffer_pct=10.0) == "sell"   # +5% < 10%
        assert signal(f, cost_buffer_pct=2.0) == "wait"    # +5% > 2%

    def test_trend_short_forecast_zero(self):
        assert trend_pct([{"date": "d1", "p50": 100}]) == 0.0
        assert trend_pct([]) == 0.0

    def test_trend_missing_middle_p50_uses_horizon(self):
        # trend reads forecast[0] and forecast[horizon]; a missing middle
        # entry is irrelevant to the computation.
        f = [{"date": "d1", "p50": 100}, {"date": "d2"}, {"date": "d3", "p50": 110}]
        assert trend_pct(f) == 10.0

    def test_trend_missing_horizon_p50_does_not_crash(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 105}, {"date": "d3"}]
        assert trend_pct(f) == 0.0

    def test_trend_horizon_capped(self):
        f = [{"date": f"d{i}", "p50": 100 + i} for i in range(10)]
        assert trend_pct(f, days=5) == 5.0  # d5 vs d0 = +5%
        assert trend_pct(f, days=99) == 9.0  # capped at len-1

    def test_contract_shape(self):
        out = contract(_snap([{"date": "d1", "p50": 100}, {"date": "d2", "p50": 105}]))
        for key in ("commodity", "market", "state", "observed", "forecast",
                    "trend_pct_5d", "signal", "reliability", "source", "as_of",
                    "model_version", "needs_expert"):
            assert key in out
        assert out["needs_expert"] is False
        assert out["reliability"]["mae_inr_per_quintal"] == 521.20
        assert out["reliability"]["wape"] == 0.1053

    def test_contract_observed_fields(self):
        out = contract(_snap([{"date": "d1", "p50": 100}]))
        obs = out["observed"]
        for key in ("date", "min", "modal", "max", "arrivals"):
            assert key in obs

    def test_empty_forecast_sell(self):
        out = contract(_snap([]))
        assert out["trend_pct_5d"] == 0.0
        assert out["signal"] == "sell"

    def test_deterministic(self):
        f = [{"date": "d1", "p50": 100}, {"date": "d2", "p50": 105}]
        assert contract(_snap(f)) == contract(_snap(f))


# ─────────────────────────── D. Advisory ───────────────────────────

class TestAdvisory:
    def test_expert_pending_always_true(self):
        a = build(None, None, None, None)
        assert a.expert_pending is True
        a2 = build({"label": "Leaf_Blast", "confidence": 0.9, "needs_expert": False},
                   {"fertilizer": [], "pesticide": [], "abstain": False},
                   {"score": 0.5, "priority": "medium", "drivers": []},
                   {"signal": "sell", "observed": {"modal": 2180}})
        assert a2.expert_pending is True

    def test_en_and_mr_present(self):
        a = build({"label": "Leaf_Blast", "confidence": 0.9, "needs_expert": False},
                  {"fertilizer": [], "pesticide": [], "abstain": False},
                  {"score": 0.5, "priority": "medium", "drivers": []},
                  {"signal": "sell", "observed": {"modal": 2180}})
        assert a.text_en and a.text_mr
        assert "AgriSense Advisory" in a.text_en
        assert "ॲग्रीसेन्स" in a.text_mr

    def test_structured_refs_match_payloads(self):
        a = build({"label": "Leaf_Blast", "confidence": 0.9, "needs_expert": False},
                  {"fertilizer": [], "pesticide": [], "abstain": False},
                  {"score": 0.5, "priority": "medium", "drivers": []},
                  {"signal": "sell", "observed": {"modal": 2180}})
        assert a.structured_refs == ["/ml/disease", "/ml/recommend", "/ml/risk", "/ml/price"]

    def test_all_none_payloads(self):
        a = build(None, None, None, None)
        assert a.structured_refs == []
        assert "AgriSense Advisory" in a.text_en
        assert "automated advisory" in a.text_en

    def test_abstain_message(self):
        a = build(None, {"fertilizer": [], "pesticide": [], "abstain": True}, None, None)
        assert "expert recommendation required" in a.text_en
        assert "तज्ज्ञांचा सल्ला आवश्यक" in a.text_mr

    def test_price_signal_words(self):
        sell = build(None, None, None, {"signal": "sell", "observed": {"modal": 2180}})
        wait = build(None, None, None, {"signal": "wait", "observed": {"modal": 2180}})
        assert "sell now" in sell.text_en
        assert "hold produce" in wait.text_en

    def test_disease_expert_note(self):
        a = build({"label": "Leaf_Blast", "confidence": 0.9, "needs_expert": True}, None, None, None)
        assert "routed to expert" in a.text_en
        b = build({"label": "Leaf_Blast", "confidence": 0.9, "needs_expert": False}, None, None, None)
        assert "routed to expert" not in b.text_en

    def test_deterministic_templates(self):
        payloads = ({"label": "Brown_Spot", "confidence": 0.8, "needs_expert": False},
                    {"fertilizer": [{"item": "N100:P40:K40/ha", "rationale": ["zone match"]}],
                     "pesticide": [{"item": "Ediphenphos 50% EC",
                                    "detail": {"dose": "500-600 ml/ha", "timing": "at grade 3"}}],
                     "abstain": False},
                    {"score": 0.6, "priority": "medium", "drivers": [{"feature": "disease_prob"}]},
                    {"signal": "wait", "observed": {"modal": 2180}})
        assert build(*payloads) == build(*payloads)


# ─────────────────────────── E. Abstain ───────────────────────────

class TestAbstain:
    def test_top2_basic(self):
        label, conf, margin = top2({"A": 0.7, "B": 0.2, "C": 0.1})
        assert label == "A" and conf == 0.7
        assert margin == pytest.approx(0.5)  # 0.7-0.2 = 0.49999999999999994 in IEEE754

    def test_top2_single_class(self):
        label, conf, margin = top2({"A": 1.0})
        assert label == "A" and conf == 1.0 and margin == 1.0

    def test_temperature_identity(self):
        probs = {"A": 0.6, "B": 0.4}
        assert temperature_scale(probs, 1.0) == probs

    def test_temperature_invalid_raises(self):
        with pytest.raises(ValueError):
            temperature_scale({"A": 0.5}, 0.0)
        with pytest.raises(ValueError):
            temperature_scale({"A": 0.5}, -1.0)

    def test_temperature_scaling_valid_distribution(self):
        scaled = temperature_scale({"A": 0.8, "B": 0.2}, 2.0)
        assert abs(sum(scaled.values()) - 1.0) < 1e-9
        assert all(0.0 <= v <= 1.0 for v in scaled.values())

    def test_decide_low_confidence(self):
        v = decide({"A": 0.4, "B": 0.3, "C": 0.3})
        assert v.needs_expert and v.abstain_reason == "low_confidence"

    def test_decide_low_margin(self):
        v = decide({"A": 0.55, "B": 0.45})
        assert v.needs_expert and v.abstain_reason == "low_margin"

    def test_decide_healthy_unsafe(self):
        v = decide({"Healthy": 0.85, "Leaf_Blast": 0.15})
        assert v.needs_expert and v.abstain_reason == "healthy_claim_unsafe"

    def test_decide_accept(self):
        v = decide({"Leaf_Blast": 0.9, "Brown_Spot": 0.1})
        assert not v.needs_expert and v.abstain_reason is None
        assert v.label == "Leaf_Blast" and v.confidence == 0.9

    def test_risk_coverage_empty_raises(self):
        with pytest.raises(ValueError):
            risk_coverage([], 0.5)

    def test_risk_coverage_math(self):
        rows = [({"A": 0.9, "B": 0.1}, "A"),   # accepted, correct
                ({"A": 0.8, "B": 0.2}, "B"),   # accepted, wrong
                ({"A": 0.4, "B": 0.35, "C": 0.25}, "C")]  # top=0.4 < 0.5 -> rejected
        m = risk_coverage(rows, 0.5)
        assert m["coverage"] == round(2 / 3, 4)
        assert m["accuracy_on_accepted"] == 0.5

    def test_best_threshold(self):
        rows = [({"A": 0.95, "B": 0.05}, "A")] * 10 + [({"A": 0.6, "B": 0.4}, "B")] * 2
        t = best_threshold(rows, target_accuracy=0.9)
        assert t is not None
        assert risk_coverage(rows, t)["accuracy_on_accepted"] >= 0.9

    def test_best_threshold_none_when_unreachable(self):
        rows = [({"A": 0.9, "B": 0.1}, "B")] * 5
        assert best_threshold(rows, target_accuracy=0.95) is None


# ─────────────────────────── F. Crop leaf ───────────────────────────

def _img(h, w, color=(0, 0, 0)):
    import numpy as np
    return np.full((h, w, 3), color, dtype=np.uint8)


class TestCropLeaf:
    def test_leaf_mask_detects_green(self):
        import numpy as np
        img = _img(64, 64)
        img[20:44, 20:44] = (40, 160, 60)  # green square (BGR)
        mask = crop_leaf.leaf_mask(img)
        assert mask.sum() > 0

    def test_crop_to_mask_low_coverage_rejected(self):
        import numpy as np
        img = _img(64, 64)
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[31, 31] = 255  # single pixel -> coverage < 2%
        res = crop_leaf.crop_to_mask(img, mask)
        assert res.ok is False

    def test_crop_leaf_no_green_returns_original(self):
        img = _img(64, 64, color=(0, 0, 0))  # black -> no green
        res = crop_leaf.crop_leaf(img)
        assert res.ok is False
        assert res.image.shape == img.shape

    def test_crop_leaf_green_returns_crop(self):
        img = _img(64, 64)
        img[20:44, 20:44] = (40, 160, 60)
        res = crop_leaf.crop_leaf(img)
        assert res.ok is True
        assert res.box is not None
        x, y, w, h = res.box
        assert w > 0 and h > 0
        assert res.image.shape[0] == h and res.image.shape[1] == w

    def test_lanczos_resize_shape(self):
        img = _img(100, 80)
        out = crop_leaf.lanczos_resize(img, 256)
        assert out.shape == (256, 256, 3)

    def test_prepare_for_model_shape_dtype(self):
        img = _img(100, 80)
        img[30:70, 30:70] = (40, 160, 60)
        out = crop_leaf.prepare_for_model(img)
        assert out.shape == (256, 256, 3)
        assert out.dtype.name == "float32"

    def test_crop_leaf_never_raises(self):
        import numpy as np
        cases = [_img(64, 64), _img(64, 64, color=(255, 255, 255)),
                 np.zeros((64, 64, 3), dtype=np.uint8),
                 np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)]
        for img in cases:
            res = crop_leaf.crop_leaf(img)
            assert res.image is not None

    def test_crop_leaf_empty_image_does_not_crash(self):
        res = crop_leaf.crop_leaf(_img(0, 0))
        assert res.ok is False


# ─────────────────────────── G. Integration ───────────────────────────

class TestIntegration:
    def test_full_pipeline_disease_to_advisory(self):
        # disease -> recommend -> risk -> advisory
        disease = {"label": "Brown_Spot", "confidence": 0.85, "needs_expert": False}
        rec = recommend(Context(disease_label=disease["label"], district="Ratnagiri",
                                soil_n=90, rainfall_anomaly_pct=25), RULES)
        assert rec["pesticide"], "pipeline: no pesticide for Brown_Spot"
        assert rec["ipm"], "pipeline: no IPM actions"
        risk_out = risk(RiskInput(disease["confidence"], 0.8, 0.5, 0.7, 0.6))
        adv = build(disease, rec, risk_out,
                    {"signal": "wait", "observed": {"modal": 2180}},
                    crop="paddy", district="Ratnagiri", date="2026-10-08")
        assert adv.expert_pending is True
        assert "/ml/disease" in adv.structured_refs
        assert "/ml/recommend" in adv.structured_refs
        assert "/ml/risk" in adv.structured_refs
        assert "/ml/price" in adv.structured_refs
        assert "Brown_Spot" in adv.text_en
        assert "Ediphenphos" in adv.text_en or "Azoxystrobin" in adv.text_en

    def test_risk_from_disease_confidence(self):
        # high disease confidence + high anomaly -> high risk band
        hi = risk(RiskInput(0.95, 0.9, 1.0, 1.0, 1.0))
        lo = risk(RiskInput(0.1, 0.9, -1.0, 0.1, 0.1))
        assert hi["score"] > lo["score"]
        assert hi["priority"] in {"high", "medium"}

    def test_advisory_uses_recommend_abstain(self):
        # Unknown disease: no pesticide, needs_expert=True (fertilizer rules are
        # zone-agnostic so they still match -> abstain=False is correct here).
        rec = recommend(Context(disease_label="Mystery_Blight"), RULES)
        assert not rec["pesticide"]
        assert rec["needs_expert"] is True
        adv = build({"label": "Mystery_Blight", "confidence": 0.9, "needs_expert": True},
                    rec, None, None)
        # abstain=False -> no abstain message; expert routing comes via the
        # disease line instead, and fertilizer advice is still shown.
        assert "expert recommendation required" not in adv.text_en
        assert "routed to expert" in adv.text_en
        assert "Fertilizer advice" in adv.text_en

    def test_validator_exit_zero(self):
        env = dict(__import__("os").environ)
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.run(
            [sys.executable, "ml/evals/rule_validator.py"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 0, f"validator failed:\n{proc.stdout}\n{proc.stderr}"


# ─────────────────────────── H. XGBoost risk model ───────────────────────────

FIELD = {
    "growth_stage": "Flowering",
    "rice_variety_type": "Short Duration (Karjat-3)",
    "nitrogen_applied_level": "Excessive",
    "primary_disease_risk": "Leaf Blast",
    "temperature_min": 20.0, "temperature_max": 28.0,
    "relative_humidity": 92.0, "rainfall_7d_forecast": 80.0,
    "consecutive_rainy_days": 6, "field_water_level_cm": 12.0, "soil_ph": 5.8,
}


class TestRiskXgb:
    """Contract + robustness for risk_xgb.predict(). Model-specific assertions are
    guarded on the artifact being present (ml/models/ is gitignored, so a fresh
    clone runs the fallback path)."""

    def test_contract_shape(self):
        out = xgb_predict(FIELD)
        assert set(out) >= {"score", "interval", "priority", "drivers",
                            "model_version", "needs_expert"}
        assert 0.0 <= out["score"] <= 1.0
        assert out["interval"][0] <= out["score"] <= out["interval"][1]
        assert out["priority"] in {"low", "medium", "high"}

    def test_healthy_low_blast_high(self):
        healthy = xgb_predict({**FIELD, "primary_disease_risk": "None (Healthy)",
                               "relative_humidity": 60.0, "rainfall_7d_forecast": 5.0,
                               "consecutive_rainy_days": 0})
        blast = xgb_predict(FIELD)
        if healthy["model_version"] == "risk-xgb-v1":  # model path active
            assert healthy["score"] < blast["score"]
            assert healthy["priority"] == "low"
            assert blast["priority"] in {"high", "medium"}

    def test_never_raises_on_garbage(self):
        for bad in [{}, {"growth_stage": "Booting"}, {"soil_ph": "abc"},
                    {"temperature_min": None}, {"primary_disease_risk": 123}]:
            out = xgb_predict(bad)
            assert 0.0 <= out["score"] <= 1.0
            assert out["priority"] in {"low", "medium", "high"}

    def test_fallback_when_artifact_missing(self, tmp_path):
        out = xgb_predict(FIELD, model_dir=tmp_path)
        assert out["model_version"] == "risk-xgb-fallback-priors"
        assert out["needs_expert"] is True

    def test_unseen_category_and_out_of_range(self):
        out = xgb_predict({**FIELD, "growth_stage": "Booting",
                           "rice_variety_type": "Unknown", "rainfall_7d_forecast": 9999.0})
        assert 0.0 <= out["score"] <= 1.0
        assert out["priority"] in {"low", "medium", "high"}

    def test_artifact_schema_when_present(self):
        art_path = ROOT / "ml" / "models" / "risk_xgb_artifact.json"
        if not art_path.exists():
            pytest.skip("artifact not present — run the pipeline first")
        art = json.loads(art_path.read_text(encoding="utf-8"))
        assert art["model_type"] == "xgboost"
        assert len(art["feature_order"]) == 24
        assert set(art["cat_maps"]) == {"growth_stage", "rice_variety_type",
                                        "nitrogen_applied_level", "primary_disease_risk"}
        assert art["calibration"]["type"] == "isotonic"
        assert 0.0 <= art["conformal"]["q"] <= 1.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))