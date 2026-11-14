#!/usr/bin/env python3
"""Local refinement around the stage-1+2 best params from tune.py.
Prints the best combo and its per-case losses (watch for single-case blowups).
"""
import itertools
import json
import os
import sys

ROOT = os.path.expanduser("~/workspace/make-fun-art/polymarket-faa-v1")
sys.path.insert(0, ROOT)
import main as M
from tests.tune import eval_params, CASES

ALL = sorted(CASES)
base = {
    "c_mom": -0.2, "c_gap": 0.0, "c_sum": 0.4, "c_const": 0.0,
    "tau_drift": 240.0, "sigma_k": 0.35, "sigma_floor": 0.0006,
    "sigma_cap": 0.004, "z_out": 1.65, "z_in": 0.67,
}
grid = {
    "c_mom": [-0.15, -0.2, -0.25, -0.3],
    "tau_drift": [180.0, 240.0, 300.0],
    "sigma_k": [0.3, 0.35, 0.4],
    "z_out": [1.6, 1.65, 1.75],
    "c_sum": [0.3, 0.4, 0.5],
}
best = (float("inf"), None)
for combo in itertools.product(*grid.values()):
    p = dict(base)
    p.update(dict(zip(grid.keys(), combo)))
    full = dict(M.PARAMS)
    full.update(p)
    loss = eval_params(full, ALL)
    if loss < best[0]:
        best = (loss, dict(full))
print("refined best loss %.6f" % best[0])
print(json.dumps(best[1], indent=1, sort_keys=True))
print("per-case losses:")
for cid in ALL:
    print("  %s %.6f" % (cid, eval_params(best[1], [cid])))
print("baseline per-case (for reference): run tests/score.py")
