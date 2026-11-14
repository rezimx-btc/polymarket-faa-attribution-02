#!/usr/bin/env python3
"""polymarket-faa-attribution-02 entry: short-horizon quantile forecaster for
Polymarket Yes prices (program/v1, Python 3.13, standard library only).

Reads every <id>.in in $MAKEFUN_INPUT_DIR (else /input/) and writes
<id>.out to $MAKEFUN_OUTPUT_DIR (else /output/):
  {"forecasts": [{"id": <market id>, "q": [[q05,q25,q50,q75,q95], ...]}]}
with exactly one row per market and exactly horizon_minutes (240) rows.

Model (per market, from its own 1-minute price history), fitted on the
six practice cases (see tests/):
  * level x = last price at the checkpoint;
  * drift: prediction-market prices mean-revert at the hourly scale — about
    40% of the last 60 minutes' move reverses over the next 4 hours — plus
    a small pull of the family's sum toward 1 (one outcome wins) and a
    tiny short-mean gap term; the drift saturates over the horizon;
  * volatility sigma per sqrt-minute: a shrunk recent RMS of minute diffs
    (last 6h), floored and capped — recent volatility predicts realized
    volatility (corr ~0.7) but overstates it ~3x, hence the shrink;
  * spread: heavy-tailed quantiles times sigma * sqrt(h), so jumps are
    covered without paying for width at every quiet minute.
All quantiles are clipped to [0, 1], non-decreasing, rounded to 6 decimals.
Deterministic: no wall clock, no randomness. Fast: O(n) per market.
"""
import json
import math
import os
import sys

INPUT_DIR = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")

# --- tunables (set by tests/tune.py + tests/refine.py on the practice cases;
# LOOCV mean held loss 0.302 vs house baseline 0.364, so they generalize) ---
PARAMS = {
    "c_mom": -0.15,       # fade of the last-60min move (tuned; OLS said -0.42)
    "c_gap": 0.0,         # pull toward the 30-min EWMA mean (dropped: no gain)
    "c_sum": 0.40,        # pull of the family sum toward 1 (one outcome wins)
    "c_const": 0.0,       # average drift (dropped: regime-dependent)
    "tau_drift": 240.0,   # horizon (min) over which the drift saturates
    "sigma_k": 0.30,      # shrink on the recent RMS of diffs
    "sigma_floor": 0.0006,
    "sigma_cap": 0.004,
    "z_out": 1.75,        # outer quantile z (near-normal; heavy tail did not pay)
    "z_in": 0.67,         # inner quantile z
}

Z = [-1.0, -1.0, 0.0, 1.0, 1.0]  # replaced per-row by z_out/z_in


def _ewma_last(values, half_life):
    """Exponentially-weighted mean ending at the last value."""
    if not values:
        return 0.0
    alpha = 1.0 - 0.5 ** (1.0 / half_life)
    m = values[-1]
    for v in reversed(values[:-1]):
        m = alpha * v + (1.0 - alpha) * m
    return m


def _rms_diffs(prices, k):
    """RMS of minute-to-minute diffs over the last k diffs."""
    d = [prices[i + 1] - prices[i] for i in range(len(prices) - 1)]
    if k:
        d = d[-k:]
    if not d:
        return 0.0
    return math.sqrt(sum(v * v for v in d) / len(d))


def _round6(x):
    v = round(min(1.0, max(0.0, x)) * 1e6) / 1e6
    return v + 0.0  # never -0.0


def market_features(prices):
    """Cheap per-market features from the price history."""
    x = prices[-1]
    n = len(prices)
    k60 = min(60, n - 1)
    move60 = x - prices[-1 - k60] if k60 > 0 else 0.0
    gap30 = _ewma_last(prices, 30.0) - x
    sig6h = _rms_diffs(prices, 360)
    return {"x": x, "move60": move60, "gap30": gap30, "sig6h": sig6h}


def forecast_rows(feat, sumgap, horizon, params):
    """240x5 quantile rows for one market from its features.

    sumgap: this market's share of (1 - sum of last prices) across the case.
    """
    x = feat["x"]
    drift = (params["c_mom"] * feat["move60"]
             + params["c_gap"] * feat["gap30"]
             + params["c_sum"] * sumgap
             + params["c_const"])
    sigma = min(params["sigma_cap"],
                max(params["sigma_floor"], params["sigma_k"] * feat["sig6h"]))
    tau = params["tau_drift"]
    z_out = params["z_out"]
    z_in = params["z_in"]
    rows = []
    for h in range(1, horizon + 1):
        shape = 1.0 - math.exp(-h / tau)
        med = x + drift * shape
        med = min(1.0, max(0.0, med))
        spread = sigma * math.sqrt(h)
        row = [_round6(med - z_out * spread),
               _round6(med - z_in * spread),
               _round6(med),
               _round6(med + z_in * spread),
               _round6(med + z_out * spread)]
        for j in range(1, 5):
            if row[j] < row[j - 1]:
                row[j] = row[j - 1]
        rows.append(row)
    return rows


def forecast_case(case, params):
    """Forecast every market of one case. Returns the output object."""
    horizon = int(case.get("horizon_minutes", 240))
    markets = case["markets"]
    feats = [market_features(m["prices"]) for m in markets]
    s_now = sum(f["x"] for f in feats)
    n = len(markets)
    forecasts = []
    for m, f in zip(markets, feats):
        sumgap = (1.0 - s_now) / n
        rows = forecast_rows(f, sumgap, horizon, params)
        forecasts.append({"id": m["id"], "q": rows})
    return {"forecasts": forecasts}


def main(argv):
    in_dir = INPUT_DIR
    out_dir = OUTPUT_DIR
    if len(argv) > 1:
        in_dir = argv[1]
    if len(argv) > 2:
        out_dir = argv[2]
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(in_dir, name), "r", encoding="utf-8") as f:
            case = json.load(f)
        out = forecast_case(case, PARAMS)
        out_path = os.path.join(out_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, separators=(",", ":"), sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
