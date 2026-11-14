#!/usr/bin/env python3
"""Grid-search main.py's PARAMS on the 6 practice cases, with
leave-one-case-out (LOOCV) validation to guard against overfitting.

Usage: python3 tests/tune.py
Prints the best params on all cases, per-case losses, and the LOOCV loss
(params re-tuned on the other 5 cases, evaluated on the held-out one).
"""
import itertools
import json
import math
import os
import sys

ROOT = os.path.expanduser("~/workspace/make-fun-art/polymarket-faa-v1")
KIT = os.path.expanduser("~/workspace/make-fun-art/faa-kit/polymarket-faa-attribution-02-kit")
sys.path.insert(0, ROOT)
import main as M

ANS = json.load(open(os.path.join(ROOT, "tests", "answers.json")))
CASES = {}
for k in range(6):
    cid = "p0%d" % k
    CASES[cid] = json.load(open(os.path.join(KIT, "input", cid + ".in")))

# Precompute features + answers per case/market.
DATA = {}
for cid, case in CASES.items():
    feats = [M.market_features(m["prices"]) for m in case["markets"]]
    s_now = sum(f["x"] for f in feats)
    n = len(feats)
    DATA[cid] = [(f, (1.0 - s_now) / n, ANS[cid][m["id"]])
                 for f, m in zip(feats, case["markets"])]

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
SHAPE = None  # filled per tau_drift
SQRT_H = [math.sqrt(h) for h in range(1, 241)]


def eval_params(params, case_ids):
    """Mean pinball loss (points) over the given cases."""
    tau_d = params["tau_drift"]
    shape = [1.0 - math.exp(-h / tau_d) for h in range(1, 241)]
    tot = 0.0
    cnt = 0
    for cid in case_ids:
        for feat, sumgap, actual in DATA[cid]:
            drift = (params["c_mom"] * feat["move60"]
                     + params["c_gap"] * feat["gap30"]
                     + params["c_sum"] * sumgap
                     + params["c_const"])
            sigma = min(params["sigma_cap"],
                        max(params["sigma_floor"],
                            params["sigma_k"] * feat["sig6h"]))
            x = feat["x"]
            zo = params["z_out"]
            zi = params["z_in"]
            for hi in range(240):
                med = x + drift * shape[hi]
                if med < 0.0:
                    med = 0.0
                elif med > 1.0:
                    med = 1.0
                sp = sigma * SQRT_H[hi]
                y = actual[hi]
                q0 = med - zo * sp
                q1 = med - zi * sp
                q2 = med
                q3 = med + zi * sp
                q4 = med + zo * sp
                # clip to [0,1] (rows are monotone before clipping; enforce after)
                qs = (max(0.0, min(1.0, q0)), max(0.0, min(1.0, q1)),
                      max(0.0, min(1.0, q2)), max(0.0, min(1.0, q3)),
                      max(0.0, min(1.0, q4)))
                for j in range(1, 5):
                    if qs[j] < qs[j - 1]:
                        qs = qs[:j] + (qs[j - 1],) + qs[j + 1:]
                for j, t in enumerate(TAUS):
                    d = y - qs[j]
                    tot += 100.0 * (t * d if d >= 0 else (t - 1.0) * d)
                    cnt += 1
    return tot / cnt


def grid_search(param_grid, case_ids, base=None):
    best = (float("inf"), None)
    for combo in itertools.product(*param_grid.values()):
        p = dict(base if base is not None else M.PARAMS)
        p.update(dict(zip(param_grid.keys(), combo)))
        loss = eval_params(p, case_ids)
        if loss < best[0]:
            best = (loss, dict(p))
    return best


def main():
    all_ids = sorted(CASES)
    base = dict(M.PARAMS)
    print("default params loss: %.6f" % eval_params(base, all_ids))

    # Stage 1: sigma / z with drift at OLS defaults.
    g1 = {
        "sigma_k": [0.2, 0.35, 0.5, 0.7],
        "sigma_floor": [0.0003, 0.0006, 0.001],
        "sigma_cap": [0.004, 0.008],
        "z_out": [1.65, 1.9, 2.2],
        "z_in": [0.67, 0.8],
    }
    l1, p1 = grid_search(g1, all_ids)
    print("stage1 best loss %.6f params %s" % (l1, {k: p1[k] for k in g1}))

    # Stage 2: drift with sigma/z fixed.
    g2 = {
        "c_mom": [-0.2, -0.3, -0.42, -0.55],
        "c_gap": [0.0, 0.035, 0.1],
        "c_sum": [0.0, 0.2, 0.4],
        "c_const": [0.0, -0.005, -0.0115],
        "tau_drift": [60.0, 120.0, 240.0],
    }
    p1b = dict(p1)
    l2, p2 = grid_search(g2, all_ids, base=p1)
    print("stage2 (sigma/z fixed) best loss %.6f params %s" % (l2, {k: p2[k] for k in g2}))

    # Combined: stage-1 params + stage-2 params (== p2, since stage 2 ran on p1).
    p_best = dict(p2)
    print("combined loss: %.6f" % eval_params(p_best, all_ids))
    print("combined params:", json.dumps(p_best, indent=1, sort_keys=True))

    # LOOCV: re-run stage 1+2 on 5 cases, evaluate on the held-out one.
    print("\nLOOCV (stage1+stage2 re-tuned on 5 cases, tested on the 6th):")
    loo_losses = []
    for held in all_ids:
        train = [c for c in all_ids if c != held]
        _, pa = grid_search(g1, train)
        # stage 2 must start from pa's sigma/z: emulate by evaluating g2 combos
        # with pa merged in.
        best = (float("inf"), None)
        for combo in itertools.product(*g2.values()):
            p = dict(pa)
            p.update(dict(zip(g2.keys(), combo)))
            loss = eval_params(p, train)
            if loss < best[0]:
                best = (loss, dict(p))
        held_loss = eval_params(best[1], [held])
        loo_losses.append(held_loss)
        print("  held %s: train loss %.6f -> held loss %.6f" % (held, best[0], held_loss))
    print("LOOCV mean held loss: %.6f" % (sum(loo_losses) / len(loo_losses)))
    print("baseline (all cases): %.6f" % 0.363589)


if __name__ == "__main__":
    main()
