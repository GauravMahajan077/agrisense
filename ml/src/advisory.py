"""Advisory NLG — deterministic EN/MR templates over structured ML outputs.

No free-form generation. Every advisory references which /ml/* payload produced it.
Expert approval flag is always set by default (backend dispatches only if approved).
"""
from __future__ import annotations

from dataclasses import dataclass

MR = {
    "disease": "रोग ओळख: {label} (विश्वासार्हता {conf}%){expert_note}",
    "disease_expert": " — तज्ज्ञांच्या पडताळणीसाठी पाठवले आहे",
    "fertilizer": "खत शिफारस: {item} ({rationale})",
    "pesticide": "जंतुनाशक शिफारस: {item} — {dose} ({timing})",
    "price": "बाजारभाव: {signal} — सध्याचा भाव ₹{modal}/क्विंटल",
    "sell": "पुढील ७ दिवसांत भाव वाढण्याची शक्यता कमी — आता विक्री करण्यास अनुकूल",
    "wait": "भाव वाढण्याची शक्यता — विक्री थांबवा, पुढे पाहा",
    "risk": "धोका मूल्यांकन: {score}% ({priority}) — प्रमुख कारक: {drivers}",
    "header": "ॲग्रीसेन्स सल्ला — {crop}, {district} ({date})",
    "footer": "हा एक ऑटोमॅटिक सल्ला आहे. शेतकऱ्यांच्या कृषी तज्ज्ञांच्या पडताळणीनंतरच अंमलबजावणी करावी.",
}

EN = {
    "disease": "Disease detected: {label} (confidence {conf}%){expert_note}",
    "disease_expert": " — routed to expert for validation",
    "fertilizer": "Fertilizer advice: {item} ({rationale})",
    "pesticide": "Pesticide advice: {item} — {dose} ({timing})",
    "price": "Market: {signal} — current ₹{modal}/quintal",
    "sell": "no price rise expected next 7 days — favourable time to sell now",
    "wait": "prices likely to rise — hold produce and sell later",
    "risk": "Risk: {score}% ({priority}) — top drivers: {drivers}",
    "header": "AgriSense Advisory — {crop}, {district} ({date})",
    "footer": "This is an automated advisory. Apply after validation by your agricultural expert.",
}


@dataclass(frozen=True)
class Advisory:
    text_en: str
    text_mr: str
    expert_pending: bool
    structured_refs: list[str]
    model_version: str = "adv-tpl-v1"


def _signal_word(price: dict | None, lang: str) -> str:
    if not price:
        return ""
    t = MR if lang == "mr" else EN
    return t["sell"] if price.get("signal") == "sell" else t["wait"]


def build(disease: dict | None, recommend: dict | None, risk: dict | None,
          price: dict | None, crop: str = "paddy", district: str = "—",
          date: str = "—") -> Advisory:
    """Compose EN + MR advisory from /ml payloads. Always expert_pending=True."""
    refs: list[str] = []
    lines_en, lines_mr = [], []

    lines_en.append(EN["header"].format(crop=crop, district=district, date=date))
    lines_mr.append(MR["header"].format(crop=crop, district=district, date=date))

    if disease:
        refs.append("/ml/disease")
        note_en = EN["disease_expert"] if disease.get("needs_expert") else ""
        note_mr = MR["disease_expert"] if disease.get("needs_expert") else ""
        conf = round(disease.get("confidence", 0) * 100)
        lines_en.append(EN["disease"].format(label=disease.get("label"), conf=conf, expert_note=note_en))
        lines_mr.append(MR["disease"].format(label=disease.get("label"), conf=conf, expert_note=note_mr))

    if recommend:
        refs.append("/ml/recommend")
        if recommend.get("fertilizer"):
            f = recommend["fertilizer"][0]
            why = "; ".join(f.get("rationale", [])[:1]) or "KB-sourced"
            lines_en.append(EN["fertilizer"].format(item=f["item"], rationale=why))
            lines_mr.append(MR["fertilizer"].format(item=f["item"], rationale=why))
        for p in recommend.get("pesticide", [])[:1]:
            d = p.get("detail", {})
            lines_en.append(EN["pesticide"].format(item=p["item"], dose=d.get("dose", "—"), timing=d.get("timing", "—")))
            lines_mr.append(MR["pesticide"].format(item=p["item"], dose=d.get("dose", "—"), timing=d.get("timing", "—")))
        if recommend.get("abstain"):
            lines_en.append("No KB rule matched this context — expert recommendation required.")
            lines_mr.append("या प्रसंगासाठी ज्ञान-आधारित नियम नाही — तज्ज्ञांचा सल्ला आवश्यक.")

    if risk:
        refs.append("/ml/risk")
        drivers = ", ".join(d["feature"] for d in risk.get("drivers", [])[:2])
        lines_en.append(EN["risk"].format(score=round(risk.get("score", 0) * 100),
                                          priority=risk.get("priority"), drivers=drivers))
        lines_mr.append(MR["risk"].format(score=round(risk.get("score", 0) * 100),
                                          priority=risk.get("priority"), drivers=drivers))

    if price:
        refs.append("/ml/price")
        signal_en = _signal_word(price, "en")
        signal_mr = _signal_word(price, "mr")
        lines_en.append(EN["price"].format(signal=signal_en, modal=price.get("observed", {}).get("modal", "—")))
        lines_mr.append(MR["price"].format(signal=signal_mr, modal=price.get("observed", {}).get("modal", "—")))

    lines_en.append(EN["footer"])
    lines_mr.append(MR["footer"])
    return Advisory(text_en="\n".join(lines_en), text_mr="\n".join(lines_mr),
                    expert_pending=True, structured_refs=refs)