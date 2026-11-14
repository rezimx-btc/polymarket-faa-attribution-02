#!/usr/bin/env python3
"""Fetch Polymarket price history for the practice (NVIDIA) markets and build
the 1-minute grids + answers for cases p00..p05.

Validates the reconstruction: the grid over each case's own history window
must equal the .in file's prices exactly.
Writes tests/answers.json: {case: {market_id: [240 actual prices]}} plus
a validation report.
"""
import json
import math
import os
import time
import urllib.request

KIT = os.path.expanduser("~/workspace/make-fun-art/faa-kit/polymarket-faa-attribution-02-kit")
OUT = os.path.expanduser("~/workspace/make-fun-art/polymarket-faa-v1/tests/answers.json")

GAMMA = "https://gamma-api.polymarket.com/events/slug/next-ceo-of-nvidia"
CLOB = "https://clob.polymarket.com/prices-history"

# Window: 72h before p00's checkpoint (2026-11-12T06:00Z) through the last
# answer minute (2026-11-13T06:00Z).
WIN_START = 1794204000  # 2026-11-09T06:00:00Z
WIN_END = 1794549600    # 2026-11-13T06:00:00Z


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "makefun-practice-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def minute_grid(points, start_ts, end_ts):
    """points: sorted [(t, p)]. Return dict minute_ts -> price, last point at or
    before the minute, forward-filled; grid starts at the first whole minute at
    or after the first point."""
    if not points:
        return {}
    first = points[0][0]
    grid_start = ((first + 59) // 60) * 60
    grid = {}
    i = 0
    last_p = None
    for m in range(grid_start, end_ts + 1, 60):
        while i < len(points) and points[i][0] <= m:
            last_p = points[i][1]
            i += 1
        if last_p is not None:
            grid[m] = last_p
    return grid


def main():
    cases = {}
    for k in range(6):
        cid = "p0%d" % k
        cases[cid] = json.load(open(os.path.join(KIT, "input", cid + ".in")))

    # Unique markets across cases: id -> yes_token
    markets = {}
    for cid, case in cases.items():
        for m in case["markets"]:
            markets[m["id"]] = m["yes_token"]
    print("unique markets:", len(markets))

    # Fetch history for each market (chunked: CLOB window max 15 days, ours is 4)
    grids = {}
    for mid, tok in markets.items():
        url = "%s?market=%s&startTs=%d&endTs=%d&fidelity=1" % (CLOB, tok, WIN_START, WIN_END)
        data = get_json(url)
        pts = sorted((h["t"], h["p"]) for h in data["history"])
        grids[mid] = minute_grid(pts, WIN_START, WIN_END)
        print("fetched", mid[:12], "points:", len(pts), "grid minutes:", len(grids[mid]))
        time.sleep(0.2)

    # Validate: grid over each case's history must equal the .in prices.
    def parse_iso(s):
        # "2026-11-12T06:00:00.000Z" -> unix seconds
        import datetime
        return int(datetime.datetime.strptime(s, "%Y-%m-%dT%H:%M:%S.%fZ")
                   .replace(tzinfo=datetime.timezone.utc).timestamp())

    n_bad = 0
    for cid, case in cases.items():
        at = parse_iso(case["at"])
        for m in case["markets"]:
            start = parse_iso(m["start"])
            grid = grids[m["id"]]
            ok = True
            for j, p in enumerate(m["prices"]):
                t = start + j * 60
                g = grid.get(t)
                if g is None or abs(g - p) > 1e-12:
                    ok = False
                    n_bad += 1
                    if n_bad <= 5:
                        print("MISMATCH", cid, m["id"][:12], "minute", j, "case", p, "grid", g)
                    break
            if not ok:
                print("  ^ market failed validation:", cid, m["name"])
    print("validation mismatches:", n_bad)

    # Build answers: for each case, each market, the 240 grid prices after `at`.
    answers = {}
    for cid, case in cases.items():
        at = parse_iso(case["at"])
        answers[cid] = {}
        for m in case["markets"]:
            grid = grids[m["id"]]
            row = []
            for h in range(1, 241):
                t = at + h * 60
                v = grid.get(t)
                if v is None:
                    # forward-fill from the last known grid price at/before t
                    prev = [tt for tt in grid if tt <= t]
                    v = grid[max(prev)] if prev else m["prices"][-1]
                row.append(v)
            answers[cid][m["id"]] = row
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(answers, open(OUT, "w"))
    print("wrote", OUT)
    for cid in sorted(answers):
        print(cid, "markets:", len(answers[cid]), "answer minutes each: 240")


if __name__ == "__main__":
    main()
