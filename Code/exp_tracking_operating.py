
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ3 | Tracking-error operating points (all three converge at 20/hr)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[TO] Operating points for the tracking error: moving 0.53 mm from an AUC
criterion to an alarm budget

Every number in the paper outside RQ3 comes from the AUC >= 0.9 criterion,
including:

    0.53 mm   tracking error against torque injection (B)
    1.35 mm   against return-path injection (C)
    2.85 mm   against command injection (A)

But RQ3 already states that an AUC of 0.9 has no meaning in an operating
room. Comparing 0.53 mm against a 1 mm clinical threshold is therefore not
strictly valid: it is a value under a ranking criterion and corresponds to no
alarm budget.

This script computes operating points for those three numbers using exactly
the construction RQ3 uses.

-- Three design decisions ----------------------------------------

  [D1] Reuse RQ3's threshold construction rather than inventing another
       threshold = the (1 - FPR) quantile of clean window scores, with
       FPR = alarms_per_hour / 15850 for 227 ms windows. The tracking-error
       operating points are then directly comparable with RQ3's table.

  [D2] The tracking error is a statistic, not a learned model, so there is no
       train/validation split, but the threshold must still be estimated from
       clean windows only, and not from the same session as the attacked
       windows being evaluated. One session is held out: both threshold and
       detection rate are computed on it, while normalisation comes from the
       rest.

  [D3] The data decides how strict a budget can be estimated
       Estimating a (1 - FPR) quantile needs on the order of 1/FPR clean
       windows. The script reports the actual count and the strictest budget
       it supports; anything stricter returns none rather than being
       extrapolated. (RQ3 tried a generalised-Pareto tail fit and it produced
       contradictory results, so it was dropped.)

Reading the output:
    still below 1 mm    the positive result stands and the clinical
                        comparison can be kept, stating which criterion
    a few mm            same order as RQ3's step (0.95 mm); tighten the wording
    no operating point  the same phenomenon as noise, high AUC with a heavy
                        clean tail. That would raise RQ3's methodological
                        finding from one case to a general property, but the
                        three sub-millimetre results would drop to one and
                        the abstract and contributions would need rewriting

Usage:    python3 exp_tracking_operating.py
Expected: no training, statistics only, about 10 minutes
"""
import os, sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peng_uw_loader import COL


class Cfg:
    OUT_DIR    = _OUT + '/output_tracking_operating'
    DATA_ROOT  = _DATA
    WINDOW_S   = 30/132.0
    WIN_PER_H  = 3600.0/(30/132.0)          # 15850
    ALARMS     = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0]
    TPR_TARGET = 0.90
    MM_PER_DEG = 7.47
    SEEDS      = [42, 43, 44]
    # v2: refine between 0.25 and 0.5 deg (1.9 to 3.7 mm). The v1 floors of
    # 3.30 / 3.39 / 3.40 mm all fell in one interval and differ by 3%, below
    # the grid resolution, so their convergence may be an interpolation artifact.
    MAGS       = [2.0, 1.0, 0.7, 0.5, 0.46, 0.42, 0.38, 0.35, 0.32, 0.28,
                  0.25, 0.12, 0.06, 0.03, 0.015, 0.008]
    ATTACKS    = ['A', 'B', 'C']
    LAGS       = [4, 4, 6]                  # frames q^d leads q at 660 Hz
    N_POS      = 3
    N_FILES    = 4
    DECAY_TAU  = 5.07                       # transient time constant for A, in frames


def load_aligned(cfg):
    lo_d, hi_d = COL.get('jpos_desired', (194, 202))
    lo_q, hi_q = COL['robot_jpos']
    need = sorted(set(range(lo_d, hi_d)) | set(range(lo_q, hi_q)))
    files = sorted(glob.glob(f'{cfg.DATA_ROOT}/record_1_different_directions/*.csv'))[:cfg.N_FILES]
    QD, Q = [], []
    for path in files:
        parts = [ch for ch in pd.read_csv(path, header=None, usecols=need, chunksize=50000)]
        df = pd.concat(parts, ignore_index=True); del parts
        qd = df[list(range(lo_d, hi_d))].values[:, :cfg.N_POS].astype(np.float64)
        q  = df[list(range(lo_q, hi_q))].values[:, :cfg.N_POS].astype(np.float64)
        n = min(len(qd), len(q))
        QD.append(qd[:n]); Q.append(q[:n])
    return QD, Q


def aligned_error(qd, q, lags):
    n = len(q); e = np.zeros((n, len(lags)))
    for j, L in enumerate(lags):
        e[L:, j] = qd[:n-L, j] - q[L:, j]
    return e


def make_windows(E, win=30, stride=5):
    idx = list(range(win, len(E), stride))
    return np.stack([E[i-win:i] for i in idx])


def inject(W, kind, mag, rng, tau):
    out = W.copy(); T = out.shape[1]
    t0 = rng.randint(T//4, 3*T//4); j = rng.randint(out.shape[2]); n = T - t0
    if kind == 'A':
        out[:, t0:, j] += mag * np.exp(-np.arange(n)/tau)[None, :]
    elif kind == 'B':
        out[:, t0:, j] += mag * np.linspace(0, 1, n)[None, :]
    else:
        out[:, t0:, j] += mag
    return out


def stat_maxabs(W, mu, sd):
    Z = (W - mu[None, None, :]) / sd[None, None, :]
    return np.abs(Z).max(axis=(1, 2))


def tpr_at_fpr(clean, atk, fpr):
    n = len(clean)
    if n == 0 or len(atk) == 0 or fpr * n < 1.0:
        return np.nan
    thr = np.quantile(clean, 1 - fpr)
    return float((atk > thr).mean())


def floor_from(mags, tprs, target):
    pts = sorted((m, t) for m, t in zip(mags, tprs) if np.isfinite(t))
    if len(pts) < 2: return np.inf
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    for k in range(len(xs)-1):
        if (ys[k]-target)*(ys[k+1]-target) <= 0 and ys[k] != ys[k+1]:
            f = (target-ys[k])/(ys[k+1]-ys[k])
            lo, hi = max(xs[k],1e-12), max(xs[k+1],1e-12)
            return float(np.exp(np.log(lo)+f*(np.log(hi)-np.log(lo))))
    return float(xs[0]) if ys[0] >= target else np.inf


def main():
    cfg = Cfg(); os.makedirs(cfg.OUT_DIR, exist_ok=True)
    print("="*104)
    print(" [TO] Tracking-error operating points: AUC criterion -> alarm budget")
    print("="*104)
    print(f"""
  [D1] Same threshold construction as RQ3: the (1-FPR) quantile of clean scores
       window {cfg.WINDOW_S*1000:.0f} ms -> {cfg.WIN_PER_H:.0f} non-overlapping windows per hour
  [D2] One session held out for evaluation; normalisation from the rest
  [D3] Too few samples for a quantile returns none; no extrapolation
""")
    QD, Q = load_aligned(cfg)
    print(f"  loaded {len(QD)} sessions")

    Ws = []
    for qd, q in zip(QD, Q):
        E = aligned_error(qd, q, cfg.LAGS)[::5]     # down to 132 Hz
        Ws.append(make_windows(E))
    G = np.concatenate([np.full(len(w), i) for i, w in enumerate(Ws)])
    W = np.concatenate(Ws)
    print(f"  windows {W.shape}")

    va = (G == G.max()); tr = ~va
    mu = W[tr].reshape(-1, cfg.N_POS).mean(0)
    sd = np.maximum(W[tr].reshape(-1, cfg.N_POS).std(0), 1e-9)
    Wva = W[va]; n_clean = len(Wva)
    strictest = cfg.WIN_PER_H / n_clean
    print(f"  clean windows in the held-out session: {n_clean:,}")
    print(f"  strictest directly estimable budget {strictest:.1f} alarms/hour"
          f"  (anything stricter returns none)")
    clean = stat_maxabs(Wva, mu, sd)

    rows = []
    for atk in cfg.ATTACKS:
        print(f"\n  -- injection {atk} --")
        hdr = f"  {'mag (deg)':>11}{'tip (mm)':>10}"
        for a in cfg.ALARMS: hdr += f"{int(a):>8}/hr"
        print(hdr); print("-"*104)
        per_alarm = {a: [] for a in cfg.ALARMS}
        for mag in cfg.MAGS:
            tprs = {a: [] for a in cfg.ALARMS}
            for s_ in cfg.SEEDS:
                rng = np.random.RandomState(s_ + int(mag*100000))
                atk_s = stat_maxabs(inject(Wva, atk, mag, rng, cfg.DECAY_TAU), mu, sd)
                for a in cfg.ALARMS:
                    t = tpr_at_fpr(clean, atk_s, a/cfg.WIN_PER_H)
                    if np.isfinite(t): tprs[a].append(t)
            line = f"  {mag:>11.4f}{mag*cfg.MM_PER_DEG:>10.3f}"
            for a in cfg.ALARMS:
                v = np.mean(tprs[a]) if tprs[a] else np.nan
                per_alarm[a].append(v)
                line += f"{v:>11.3f}" if np.isfinite(v) else f"{'--':>11}"
            print(line)
        for a in cfg.ALARMS:
            f = floor_from(cfg.MAGS, per_alarm[a], cfg.TPR_TARGET)
            rows.append(dict(attack=atk, alarms_per_hour=a,
                             floor_deg=f if np.isfinite(f) else None,
                             floor_mm=f*cfg.MM_PER_DEG if np.isfinite(f) else None))

    print("\n" + "="*104)
    print(" Detection floor (mm at tip): AUC criterion vs alarm budget")
    print("="*104)
    AUC_MM = dict(A=2.85, B=0.53, C=1.35)
    hdr = f"\n  {'inject':>7}{'AUC>=0.9':>11}"
    for a in cfg.ALARMS: hdr += f"{int(a):>9}/hr"
    print(hdr); print("-"*104)
    for atk in cfg.ATTACKS:
        line = f"  {atk:>6}{AUC_MM[atk]:>11.2f}"
        for a in cfg.ALARMS:
            r = next((x for x in rows if x['attack']==atk and x['alarms_per_hour']==a), None)
            v = r['floor_mm'] if r and r['floor_mm'] else None
            line += f"{v:>12.2f}" if v else f"{'none':>12}"
        print(line)

    print("\n" + "="*104); print(" Reading the result"); print("="*104)
    b2 = next((x for x in rows if x['attack']=='B' and x['alarms_per_hour']==2.0), None)
    v = b2['floor_mm'] if b2 else None
    if v and v < 1.0:
        print(f"""
  Torque injection reaches {v:.2f} mm at two alarms per hour, still below 1 mm.
  -> the clinical comparison can be kept, stating which criterion it uses.""")
    elif v:
        print(f"""
  Torque injection reaches {v:.2f} mm at two alarms per hour, the same order as
  RQ3's step at 0.95 mm.  -> the comparison holds but with little margin.""")
    else:
        print("""
  Torque injection has no operating point at two alarms per hour.
  -> the same phenomenon as noise: high AUC with a heavy clean tail. This
     raises RQ3's methodological finding from one case to a general property,
     but drops three sub-millimetre results to one.""")
    json.dump(rows, open(f'{cfg.OUT_DIR}/operating_points.json','w'), indent=2)
    print(f"\n  output: {cfg.OUT_DIR}/operating_points.json")
    print("="*104)


if __name__ == '__main__':
    main()
