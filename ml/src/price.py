"""Price module — consumes MandiLens published artifacts (no training, no server).

MandiLens (github.com/heybadrinath/MandiLens) publishes compact static JSON:
AGMARKNET observed prices + 7-day forecast with empirical intervals.
Backend downloads the snapshot once; we only SHAPE it into the /ml/price contract.

Signal rule (conservative, honest): "sell" only if 5-day p50 trend exceeds
modal price by > cost_buffer_pct; otherwise "wait". Reliability metrics travel
with every response (MandiLens locked-holdout: MAE 521.20, WAPE 10.53%,
directional accuracy 49.15% — published, not hidden).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

RELIABILITY = {"mae_inr_per_quintal": 521.20, "wape": 0.1053,
               "directional_accuracy": 0.4915, "interval_coverage_note": "empirical intervals, published"}
DEFAULT_COST_BUFFER_PCT = 3.0  # transport/mandi cost assumption


@dataclass(frozen=True)
class MarketSnapshot:
    commodity: str
    market: str
    state: str
    observed: dict  # {date, min, modal, max, arrivals}
    forecast: list  # [{date, p50, p80_lo, p80_hi}]
    as_of: str


def load_snapshot(path: str | Path) -> dict:
    """Load a MandiLens published snapshot file (backend-provided path)."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def trend_pct(forecast: list[dict], days: int = 5) -> float:
    """% change of p50 over next `days` vs first forecast day. 0 if too short."""
    if len(forecast) < 2:
        return 0.0
    horizon = min(days, len(forecast) - 1)
    first, last = forecast[0].get("p50"), forecast[horizon].get("p50")
    if not first or not last:
        return 0.0
    return round(100.0 * (last - first) / first, 2)


def signal(forecast: list[dict], cost_buffer_pct: float = DEFAULT_COST_BUFFER_PCT) -> str:
    """'wait' (hold, price rising) if expected rise clears the cost buffer,
    else 'sell' (no meaningful rise expected — sell now)."""
    return "wait" if trend_pct(forecast) > cost_buffer_pct else "sell"


def contract(snap: MarketSnapshot, cost_buffer_pct: float = DEFAULT_COST_BUFFER_PCT) -> dict:
    """Shape MandiLens data into the POST /ml/price response contract."""
    obs = snap.observed
    return {
        "commodity": snap.commodity, "market": snap.market, "state": snap.state,
        "observed": {"date": obs.get("date"), "min": obs.get("min"),
                     "modal": obs.get("modal"), "max": obs.get("max"),
                     "arrivals": obs.get("arrivals")},
        "forecast": snap.forecast,
        "trend_pct_5d": trend_pct(snap.forecast),
        "signal": signal(snap.forecast, cost_buffer_pct),
        "reliability": RELIABILITY,
        "source": "AGMARKNET / GODL-India via MandiLens",
        "as_of": snap.as_of,
        "model_version": "price-mandilens-v1",
        "needs_expert": False,
    }


def sample() -> dict:
    """Offline sample payload so backend can integrate without the snapshot."""
    snap = MarketSnapshot(
        commodity="rice", market="Madhavpur", state="Maharashtra",
        observed={"date": "2026-07-20", "min": 2100, "modal": 2180, "max": 2260, "arrivals": 410},
        forecast=[{"date": "2026-07-21", "p50": 2185, "p80_lo": 2110, "p80_hi": 2260},
                  {"date": "2026-07-22", "p50": 2200, "p80_lo": 2115, "p80_hi": 2285},
                  {"date": "2026-07-23", "p50": 2244, "p80_lo": 2140, "p80_hi": 2340},
                  {"date": "2026-07-24", "p50": 2262, "p80_lo": 2150, "p80_hi": 2365}],
        as_of="2026-07-20")
    return contract(snap)


if __name__ == "__main__":
    print(json.dumps(sample(), indent=2, ensure_ascii=False))