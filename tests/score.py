#!/usr/bin/env python3
"""Score a program's forecasts on the 6 practice cases against answers.json.

Usage: python3 tests/score.py [out_dir]
  - runs main.py (MAKEFUN_INPUT_DIR=kit input, MAKEFUN_OUTPUT_DIR=out_dir)
    if out_dir does not already hold .out files, else scores what is there.
  - also scores the house baseline and a naive last-value forecast.
Prints per-case and overall pinball loss (points), median error, coverage_90.
"""
import json
import math
import os
import subprocess
import sys

ROOT = os.path.expanduser("~/workspace/make-fun-art/polymarket-faa-v1")
KIT = os.path.expanduser("~/workspace/make-fun-art/faa-kit/polymarket-faa-attribution-02-kit")
ANS = os.path.join(ROOT, "tests", "answers.json")

Z = [-1.644854, -0.67449, 0.0, 0.67449, 1.644854]
SIGMA_BASE = 0.0025


def pinball(y, q, tau):
    d = y - q
    return 100.0 * max(tau * d, (tau - 1.0) * d)


def score_forecasts(forecasts, answers):
    """forecasts: {market_id: [[5 nums] x 240]}; answers: {market_id: [240]}.
    Returns (loss, median_error, coverage_90, n_items)."""
    taus = [0.05, 0.25, 0.5, 0.75, 0.95]
    tot = 0.0
    n = 0
    med_err = 0.0
    cov = 0
    m = 0
    for mid, rows in forecasts.items():
        actual = answers[mid]
        for h in range(240):
            y = actual[h]
            row = rows[h]
            for j, tau in enumerate(taus):
                tot += pinball(y, row[j], tau)
                n += 1
            med_err += 100.0 * abs(y - row[2])
            m += 1
            if row[0] <= y <= row[4]:
                cov += 1
    return tot / n, med_err / m, cov / m, n


def baseline_forecasts(case):
    """House baseline: q = clamp(x + z*0.0025*sqrt(h), 0, 1), 6dp."""
    out = {}
    for mk in case["markets"]:
        x = mk["prices"][-1]
        rows = []
        for h in range(1, 241):
            s = SIGMA_BASE * math.sqrt(h)
            row = []
            for z in Z:
                v = round(min(1.0, max(0.0, x + z * s)) * 1e6) / 1e6 + 0.0
                row.append(v)
            for j in range(1, 5):
                if row[j] < row[j - 1]:
                    row[j] = row[j - 1]
            rows.append(row)
        out[mk["id"]] = rows
    return out


def naive_forecasts(case):
    """Last value for every quantile (zero spread)."""
    out = {}
    for mk in case["markets"]:
        x = round(mk["prices"][-1] * 1e6) / 1e6 + 0.0
        out[mk["id"]] = [[x] * 5 for _ in range(240)]
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "out")
    answers = json.load(open(ANS))
    cases = {}
    for k in range(6):
        cid = "p0%d" % k
        cases[cid] = json.load(open(os.path.join(KIT, "input", cid + ".in")))

    # run main.py if outputs are missing
    need_run = any(not os.path.exists(os.path.join(out_dir, cid + ".out")) for cid in cases)
    if need_run:
        os.makedirs(out_dir, exist_ok=True)
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = os.path.join(KIT, "input")
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        import time
        t0 = time.time()
        r = subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                           env=env, capture_output=True, text=True)
        dt = time.time() - t0
        print("main.py rc=%d in %.2fs" % (r.returncode, dt))
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr)
            return 1

    tot_loss = tot_med = tot_cov = tot_n = 0
    base_loss = naive_loss = 0.0
    base_n = naive_n = 0
    print("%-5s %10s %10s %10s %8s" % ("case", "loss", "med_err", "cov90", "n_mkts"))
    for cid in sorted(cases):
        fc = json.load(open(os.path.join(out_dir, cid + ".out")))["forecasts"]
        mine = {f["id"]: f["q"] for f in fc}
        loss, med, cov, n = score_forecasts(mine, answers[cid])
        bl, _, _, bn = score_forecasts(baseline_forecasts(cases[cid]), answers[cid])
        nl, _, _, nn = score_forecasts(naive_forecasts(cases[cid]), answers[cid])
        tot_loss += loss * n
        tot_med += med * n // 5 * 5  # med is per market-minute; weight by items
        tot_n += n
        base_loss += bl * bn
        base_n += bn
        naive_loss += nl * nn
        naive_n += nn
        print("%-5s %10.6f %10.4f %10.4f %8d" % (cid, loss, med, cov, len(mine)))
    print("-" * 55)
    print("OURS     loss %.6f  (n=%d)" % (tot_loss / tot_n, tot_n))
    print("BASELINE loss %.6f  (house: 0.363589 expected)" % (base_loss / base_n))
    print("NAIVE    loss %.6f" % (naive_loss / naive_n))
    print("improvement over baseline: %.2f%%" % (100.0 * (1.0 - (tot_loss / tot_n) / (base_loss / base_n))))


if __name__ == "__main__":
    sys.exit(main())
