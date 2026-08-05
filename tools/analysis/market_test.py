"""
THE decisive question: do our model features contain information the MARKET
doesn't already price?

Every prior improvement attempt tweaked constants inside the hand-built
run-expectation formula, and all five failed out-of-sample. This asks the
level-up question instead: fit outcomes directly from the logged features and
see whether they beat the market price as a predictor.

Method (deliberately conservative):
  - One row per (game, market) — home/over side only, so the two sides of a
    market aren't double-counted as independent observations.
  - Baseline model:  logit(outcome) ~ logit(market_prob)          [market alone]
  - Feature model:   logit(outcome) ~ logit(market_prob) + features
  - TIME-BASED split (train early, test late) — never random, so the test set
    is genuinely out-of-sample in the way live betting is.
  - Scored by log-loss on the held-out window. If the feature model does not
    beat the market-only baseline out-of-sample, our features add nothing the
    market hasn't already priced, and no amount of constant-tuning will help.

Pure numpy (no sklearn): standardized features, L2-regularized logistic
regression fit by gradient descent.

Usage: python3 -m tools.analysis.market_test [--sport MLB] [--market Moneyline]
"""
import argparse
import glob
import json
import math
import os

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Features to test, per market. Only inputs the model actually uses.
_FEATURES = [
    "run_diff", "run_diff_capped", "park_factor", "home_ops", "away_ops",
    "home_inj", "away_inj", "home_bp_era", "away_bp_era",
    "home_sp_score", "away_sp_score", "home_sp_ip", "away_sp_ip",
    "home_trap_sev", "away_trap_sev", "home_bullpen_score", "away_bullpen_score",
    "expected_home_runs", "expected_away_runs",
]


def _load(sport: str, market: str):
    rows = []
    for f in sorted(glob.glob(os.path.join(_ROOT, "state/decision_log/*.json"))):
        d = json.load(open(f))
        e = d.get("entries", d)
        rows.extend(e if isinstance(e, list) else list(e.values()))
    out = []
    for r in rows:
        if r.get("sport") != sport or r.get("market_type") != market:
            continue
        if r.get("outcome") not in ("win", "loss"):
            continue
        p = r.get("market_prob_at_first_pick")
        if not p or not (0.02 < float(p) < 0.98):
            continue
        # one side only: home for ML/Spread, "over" for totals
        side = (r.get("side") or "").lower()
        if market == "Total":
            if side != "over":
                continue
        else:
            if r.get("side") != r.get("home_team"):
                continue
        feats = r.get("features") or {}
        if not feats:
            continue
        out.append(r)
    out.sort(key=lambda r: r["date"])
    return out


def _matrix(rows, feature_names):
    X, y, mp, dates = [], [], [], []
    for r in rows:
        f = r.get("features") or {}
        vals = []
        ok = True
        for k in feature_names:
            v = f.get(k)
            if v is None or isinstance(v, bool):
                v = 0.0 if v is None else float(v)
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                ok = False
                break
        if not ok:
            continue
        X.append(vals)
        y.append(1.0 if r["outcome"] == "win" else 0.0)
        mp.append(float(r["market_prob_at_first_pick"]))
        dates.append(r["date"])
    return np.array(X), np.array(y), np.array(mp), dates


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _fit(X, y, l2=1.0, iters=4000, lr=0.08):
    """L2-regularized logistic regression, gradient descent. X includes bias."""
    w = np.zeros(X.shape[1])
    n = len(y)
    for _ in range(iters):
        z = X @ w
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        grad = X.T @ (p - y) / n
        grad[1:] += l2 * w[1:] / n      # don't regularize bias
        w -= lr * grad
    return w


def _logloss(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def run(sport="MLB", market="Moneyline", train_frac=0.7, l2=1.0):
    rows = _load(sport, market)
    if len(rows) < 120:
        print(f"{sport} {market}: only {len(rows)} usable rows — skipping")
        return
    X, y, mp, dates = _matrix(rows, _FEATURES)
    n = len(y)
    cut = int(n * train_frac)
    split_date = dates[cut]

    lg = _logit(mp).reshape(-1, 1)
    ones = np.ones((n, 1))

    # standardize features on TRAIN stats only
    mu, sd = X[:cut].mean(axis=0), X[:cut].std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    base = np.hstack([ones, lg])                 # market only
    full = np.hstack([ones, lg, Xs])             # market + features

    w_b = _fit(base[:cut], y[:cut], l2=l2)
    w_f = _fit(full[:cut], y[:cut], l2=l2)

    def pred(M, w):
        return 1.0 / (1.0 + np.exp(-np.clip(M @ w, -30, 30)))

    ll_b_tr, ll_f_tr = _logloss(y[:cut], pred(base[:cut], w_b)), _logloss(y[:cut], pred(full[:cut], w_f))
    ll_b_te, ll_f_te = _logloss(y[cut:], pred(base[cut:], w_b)), _logloss(y[cut:], pred(full[cut:], w_f))
    ll_mkt_te = _logloss(y[cut:], mp[cut:])      # raw market price as-is

    print(f"\n=== {sport} {market} — do our features beat the market? ===")
    print(f"rows={n}  train={cut} (through {dates[cut-1]})  test={n-cut} (from {split_date})")
    print(f"  raw market price      test log-loss: {ll_mkt_te:.4f}")
    print(f"  market-only model     test log-loss: {ll_b_te:.4f}   (train {ll_b_tr:.4f})")
    print(f"  market + OUR FEATURES test log-loss: {ll_f_te:.4f}   (train {ll_f_tr:.4f})")
    delta = ll_b_te - ll_f_te
    verdict = ("FEATURES ADD SIGNAL (lower loss is better)" if delta > 0.002 else
               "features add NOTHING beyond the market" if delta > -0.002 else
               "features make it WORSE (overfitting)")
    print(f"  → improvement from features: {delta:+.4f}   {verdict}")

    # which features carry weight (standardized coefficients)
    coefs = sorted(zip(_FEATURES, w_f[2:]), key=lambda kv: -abs(kv[1]))[:8]
    print("  top standardized coefficients:")
    for k, c in coefs:
        print(f"     {k:22} {c:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sport", default="MLB")
    ap.add_argument("--market", default=None)
    ap.add_argument("--l2", type=float, default=1.0)
    a = ap.parse_args()
    markets = [a.market] if a.market else ["Moneyline", "Spread", "Total"]
    for m in markets:
        run(a.sport, m, l2=a.l2)


if __name__ == "__main__":
    main()
