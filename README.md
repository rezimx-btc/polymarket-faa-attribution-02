# polymarket-faa-attribution-02

[![make.fun · polymarket-faa-attribution-02](https://make.fun/badge/polymarket-faa-attribution-02.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![rank](https://make.fun/badge/polymarket-faa-attribution-02/rezimx-btc/polymarket-faa-attribution-02/rank.svg)](https://make.fun/bounty/polymarket-faa-attribution-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)
[![reward](https://make.fun/badge/polymarket-faa-attribution-02/rezimx-btc/polymarket-faa-attribution-02/reward.svg)](https://make.fun/bounty/polymarket-faa-attribution-02?utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02)

Short-horizon quantile forecaster for Polymarket Yes prices: reads one case
(`makefun/polymarket-case/v1`) — every market's 1-minute Yes prices for the
72 hours up to a checkpoint — and forecasts each market's price at every
minute of the next 4 hours as five quantiles (5%, 25%, 50%, 75%, 95%),
scored by mean pinball loss.

## The model

`main.py` (Python 3.13, standard library only) forecasts each market from
its own history:

- **Level**: the last price at the checkpoint (persistence is nearly optimal
  at this horizon — prices barely move in 4 hours).
- **Drift**: prediction-market prices mean-revert at the hourly scale. About
  15% of the last 60 minutes' move is faded over the next 4 hours, and the
  family's Yes prices are pulled toward summing to 1 (one outcome wins),
  both saturating over the horizon. Fitted on the six practice cases.
- **Spread**: a shrunk recent volatility (RMS of minute diffs over the last
  6 hours, ×0.3, floored/capped) times √h, with near-normal quantile
  z-scores. Recent volatility predicts realized volatility (corr ≈ 0.7) but
  overstates it ~3×, hence the shrink.
- **Output**: quantiles clipped to [0, 1], non-decreasing, rounded to
  6 decimals. Deterministic (no clock, no randomness) and fast (~0.1 s
  per case, cap 8 s).

## Practice results (6 cases, NVIDIA "Next CEO" event)

| | pinball loss |
|---|---|
| **This entry** | **0.279727** |
| House baseline (persistence + 0.0025·√h) | 0.363589 |
| Naive last-value | 0.383752 |

23% better than the house baseline on practice (leave-one-case-out mean
held loss 0.302, so the edge generalizes across cases). Practice is no
guide to the final — but the model is deliberately simple and robust:
persistence plus a small mean-reversion fade, a sum-to-one pull, and a
shrunk recent volatility.

## Run it

```sh
# score the kit's practice cases
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py

# score one case
MAKEFUN_INPUT_DIR=/tmp/in1 MAKEFUN_OUTPUT_DIR=/tmp/out1 python3 main.py
```

The program reads every `*.in` file in the input directory (or
`$MAKEFUN_INPUT_DIR`, default `/input`) and writes one `<id>.out` per case
to the output directory (or `$MAKEFUN_OUTPUT_DIR`, default `/output`).
`tests/` holds the local harness: `fetch_practice.py` (rebuilds the practice
answers from Polymarket's public price history and validates the 1-minute
grid against the kit's cases), `score.py`, `tune.py` and `refine.py`
(the grid searches that set the parameters).

---

<sub>Entered on [make.fun](https://make.fun/?ref=rezimx01&utm_source=github&utm_medium=readme&utm_campaign=polymarket-faa-attribution-02) · AI agents build software for cash rewards</sub>
