"""Selective prediction: confidence calibration + abstention routing.

Pure functions, no I/O, no model loading. Unit-testable with plain lists.
Thresholds are tuned on the field stress set (see evals/results.md).
"""
from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CONF_THRESHOLD = 0.5  # matches abstain_threshold in class_names.json
DEFAULT_MARGIN_THRESHOLD = 0.15
HEALTHY_CONF_THRESHOLD = 0.9  # claiming "Healthy" requires high confidence


@dataclass(frozen=True)
class Verdict:
    label: str
    confidence: float
    margin: float  # top1 prob - top2 prob
    needs_expert: bool
    abstain_reason: str | None


def top2(probs: dict[str, float]) -> tuple[str, float, float]:
    """Return (top_label, top_prob, margin over second). probs: class -> prob."""
    ordered = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_label, top_p = ordered[0]
    second_p = ordered[1][1] if len(ordered) > 1 else 0.0
    return top_label, float(top_p), float(top_p) - float(second_p)


def temperature_scale(probs: dict[str, float], temperature: float) -> dict[str, float]:
    """Platt-style temperature scaling on a probability dict (T fitted offline)."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    scaled = {k: v ** (1.0 / temperature) for k, v in probs.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def decide(
    probs: dict[str, float],
    temperature: float = 1.0,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
    margin_threshold: float = DEFAULT_MARGIN_THRESHOLD,
    healthy_conf_threshold: float = HEALTHY_CONF_THRESHOLD,
) -> Verdict:
    """Route a prediction to ACCEPT or EXPERT. Single decision point for the API."""
    calibrated = temperature_scale(probs, temperature)
    label, conf, margin = top2(calibrated)

    reason = None
    if conf < conf_threshold:
        reason = "low_confidence"
    elif margin < margin_threshold:
        reason = "low_margin"
    elif label == "Healthy" and conf < healthy_conf_threshold:
        reason = "healthy_claim_unsafe"

    return Verdict(label=label, confidence=round(conf, 4), margin=round(margin, 4),
                   needs_expert=reason is not None, abstain_reason=reason)


def risk_coverage(rows: list[tuple[dict[str, float], str]], conf_threshold: float) -> dict[str, float]:
    """Coverage/accuracy of ACCEPTED predictions at a threshold (for the R-C curve).

    rows: [(probs, true_label)] — evaluation only, never used at inference.
    """
    if not rows:
        raise ValueError("rows must not be empty")
    accepted = [(p, y) for p, y in rows if top2(p)[1] >= conf_threshold]
    coverage = len(accepted) / len(rows)
    acc = (sum(1 for p, y in accepted if top2(p)[0] == y) / len(accepted)) if accepted else 0.0
    return {"coverage": round(coverage, 4), "accuracy_on_accepted": round(acc, 4)}


def best_threshold(rows: list[tuple[dict[str, float], str]],
                   target_accuracy: float = 0.95) -> float | None:
    """Smallest threshold whose accepted-set accuracy >= target (find max coverage)."""
    candidates = sorted({round(top2(p)[1], 2) for p, _ in rows})
    best: float | None = None
    for t in candidates:
        m = risk_coverage(rows, t)
        if m["accuracy_on_accepted"] >= target_accuracy:
            best = t
            break
    return best