
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] check | Two counting errors from overlapping windows, and the corrected operating points
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[W] Two counting errors from overlapping windows, and the corrected
operating points

A pre-submission check found two errors in the alarm-budget conversion in
exp_operating_point.py, both in the same direction, both making the result
look better than it is. This script quantifies them and recomputes the
operating points.

-- Error one: decision frequency counted non-overlapping windows --

  The original [D1] states that an hour holds 3600/0.227 = 15,850
  non-overlapping windows, and calls that conservative on the grounds that
  overlapping deployment would multiply the decision count by the stride.

  That judgement is backwards. The detector steps by 5 frames (30-frame
  window, 25 overlapping), so deployment makes 3600/(5/132) = 95,040
  decisions per hour, six times 15,850. Six times more decisions needs a
  six times lower per-window false-positive rate, hence a higher threshold
  and a larger floor. Reporting 15,850 understates the problem rather than
  being conservative.

-- Error two: validation windows are not independent --------------

  The 8,731 held-out windows were generated at stride 5, so adjacent windows
  share 25 of 30 frames. They are not 8,731 independent samples. Estimating a
  (1-FPR) quantile needs independent ones, so the inference from 8,731 to a
  strictest budget of 1.8 alarms per hour does not hold.

  This script regenerates the clean validation set from non-overlapping
  windows (stride equal to the window length), giving genuinely independent
  samples, and reports the strictest directly estimable budget.

-- The two errors together ----------------------------------------

  nominal 8,731 with 15,850 decisions   ->  1.8 alarms/hour (what the paper says)
  independent samples with 15,850       ->  measured here
  independent samples with 95,040       ->  measured here

-- Design decisions -----------------------------------------------

  [D1] Report both decision frequencies so the difference is visible
       Whether deployment scores every sliding window is an engineering
       choice. The paper does not make it for the reader, but it must say
       what the choice changes.

  [D2] Regenerate the clean validation set from non-overlapping windows
       Only the validation stride changes; the training set is untouched.
       Overlapping windows in training are data augmentation and are fine.
       The problem is only in estimating a quantile from overlapping samples.

  [D3] No extrapolation
       The original fell back to a generalised-Pareto tail fit when samples
       ran short. That extrapolation gave inconsistent results, with tighter
       budgets yielding lower thresholds, so this script reports only
       directly estimable budgets and returns none otherwise.

Usage:    python3 exp_window_accounting.py
Expected: one training run plus evaluation, about 4 minutes
Output:   output_accounting/operating_points_fixed.json
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_delta_ablation import (Cfg as BaseCfg, load_records, fit_physics_model,
                                build_window_cache, build_features,
                                MaskedMultiTaskLoss)
from exp_paper_sweep import SweepInjector


class Cfg(BaseCfg):
    OUT_DIR = _OUT + '/output_accounting'
    MODE    = 'jpos_torque_delta'
    SEED    = 42
    EPOCHS  = 120
    LR      = 3e-4
    BATCH   = 128
    HIDDEN  = 128
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1
    FAMILY  = 'step'          # only step has an operating point under a budget
    JOINT   = 0
    MAGS    = [0.010, 0.020, 0.040, 0.080, 0.150, 0.300]
    BUDGETS = [100, 50, 20, 10, 5, 2, 1]
    MM_PER_DEG = 7.46         # joint 1, from 0.800 deg = 5.97 mm


class Det(nn.Module):
    def __init__(s, d, h, la):
        super().__init__(); s.la = la
        s.lstm = nn.LSTM(d, h, 2, batch_first=True, dropout=0.2)
        s.dev = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, la*3))
        nn.init.zeros_(s.dev[-1].weight); nn.init.zeros_(s.dev[-1].bias)
        s.atk = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):
        h = s.lstm(x)[0][:, -1, :]
        return s.dev(h).view(-1, s.la, 3), s.atk(h).squeeze(-1)


def independent_subset(idx, lookback, stride):
    """
    [D2] Select a mutually non-overlapping subset from stride-generated windows.
    Adjacent starts differ by stride frames; disjointness needs starts at least
    lookback apart, so keep one in every ceil(lookback/stride).
    """
    keep = int(np.ceil(lookback / stride))
    return idx[::keep], keep


def budget_floor(clean, atk_by_mag, budget, windows_per_hour, tpr_target=0.9):
    """Smallest magnitude reaching tpr_target at this budget; None if unestimable"""
    fpr = budget / windows_per_hour
    n = len(clean)
    if fpr * n < 1.0:
        return None, fpr, False            # too few samples for this quantile
    thr = np.quantile(clean, 1 - fpr)
    for mag in sorted(atk_by_mag):
        if (atk_by_mag[mag] > thr).mean() >= tpr_target:
            return mag, fpr, True
    return None, fpr, True                 # enough samples, but no magnitude reaches 90%


def main():
    cfg = Cfg(); os.makedirs(cfg.OUT_DIR, exist_ok=True); t0 = time.time()
    dev = cfg.DEVICE
    LB = cfg.LOOKBACK
    STRIDE = getattr(cfg, 'STRIDE', 5)
    T_w = LB * cfg.SUBSAMPLE / 660.0

    wph_nonoverlap = 3600.0 / T_w
    wph_sliding    = 3600.0 / (STRIDE * cfg.SUBSAMPLE / 660.0)

    print("=" * 96)
    print(" [W] Two counting errors from overlapping windows")
    print("=" * 96)
    print(f"""
  window {LB} frames, subsample {cfg.SUBSAMPLE}, stride {STRIDE}  ->  {T_w*1000:.0f} ms
  adjacent windows share {LB-STRIDE}/{LB} frames

  decision frequency
    non-overlap  {wph_nonoverlap:>9,.0f} per hour     what the paper uses
    every window {wph_sliding:>9,.0f} per hour     deployment ({wph_sliding/wph_nonoverlap:.0f}x)
""")

    recs = load_records(cfg)
    a_, b_, r2, _, _ = fit_physics_model(recs, cfg.N_POS_JOINTS)
    JP, TQ, YD, GID = build_window_cache(recs, cfg)

    g = np.unique(GID); nv = max(1, int(round(len(g) * cfg.VAL_FRAC_RECORDS)))
    vg = np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva = np.isin(GID, vg); mtr = ~mva
    va_idx = np.where(mva)[0]
    ind_idx, keep = independent_subset(va_idx, LB, STRIDE)

    print(f"  held-out windows  {len(va_idx):>7,}   (stride {STRIDE}, overlapping)")
    print(f"  independent subset{len(ind_idx):>7,}   (one in every {keep})")
    print(f"  effective sample loss {len(va_idx)/len(ind_idx):.1f}x\n")

    # -- Training --
    print("  training...")
    rng = np.random.RandomState(cfg.SEED)
    JPtr = JP.copy(); ytr = np.zeros(len(JP), np.float32)
    pool = np.where(mtr)[0]
    for i in rng.permutation(pool)[:int(len(pool)*cfg.ATTACK_FRAC)]:
        sc = float(np.exp(rng.uniform(np.log(cfg.MAGS[0]), np.log(cfg.MAGS[-1]))))
        JPtr[i], _, _, _, _ = SweepInjector.inject(JP[i], scale=sc, rng=rng,
                                                   force_type=cfg.FAMILY,
                                                   force_joint=cfg.JOINT)
        ytr[i] = 1.0
    X = build_features(JPtr, TQ, cfg.MODE, a_, b_, cfg.N_POS_JOINTS)
    f = X[mtr].reshape(-1, X.shape[-1])
    mu = f.mean(0).astype(np.float32); sd = np.maximum(f.std(0), 1e-6).astype(np.float32)
    std_delta = float(YD[mtr].std() + 1e-8)

    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    m = Det(X.shape[-1], cfg.HIDDEN, cfg.LOOKAHEAD).to(dev)
    pw = torch.tensor(float((ytr[mtr] < 0.5).sum() / max((ytr[mtr] >= 0.5).sum(), 1)),
                      dtype=torch.float32, device=dev)
    lossf = MaskedMultiTaskLoss(cfg.W_DEV, cfg.W_ATK, cfg.W_VEL, pos_weight=pw).to(dev)
    # weight_decay=0: a nonzero value suppresses the noise family, see [WD]
    opt = torch.optim.Adam(m.parameters(), lr=cfg.LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)
    Xt = torch.from_numpy(((X[mtr]-mu)/sd).astype(np.float32)).to(dev)
    Yt = torch.from_numpy(ytr[mtr]).to(dev)
    Dt = torch.from_numpy((YD[mtr]/std_delta).astype(np.float32)).to(dev)
    for ep in range(cfg.EPOCHS):
        m.train(); perm = torch.randperm(len(Xt), device=dev)
        for s0 in range(0, len(Xt), cfg.BATCH):
            i = perm[s0:s0+cfg.BATCH]; opt.zero_grad()
            dp, ap = m(Xt[i]); lossf(dp, ap, Dt[i], Yt[i]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        if (ep+1) % 40 == 0: print(f"    epoch {ep+1}  ({time.time()-t0:.0f}s)")
    m.eval()

    def score(rows, jp):
        Xs = build_features(jp, TQ[rows], cfg.MODE, a_, b_, cfg.N_POS_JOINTS)
        Xn = ((Xs - mu) / sd).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Xn), 512):
                _, ap = m(torch.from_numpy(Xn[i:i+512]).to(dev))
                out.append(torch.sigmoid(ap).cpu().numpy())
        return np.concatenate(out)

    # -- Two sets of clean scores: nominal (all held out) and independent --
    clean_all = score(va_idx, JP[va_idx])
    clean_ind = score(ind_idx, JP[ind_idx])

    # -- Attack scores per magnitude, on the independent subset --
    atk = {}
    rg = np.random.RandomState(999)
    for mag in cfg.MAGS:
        jp = JP[ind_idx].copy()
        for k in range(len(jp)):
            jp[k], _, _, _, _ = SweepInjector.inject(jp[k], scale=mag, rng=rg,
                                                     force_type=cfg.FAMILY,
                                                     force_joint=cfg.JOINT)
        atk[mag] = score(ind_idx, jp)

    print("\n" + "=" * 96)
    print(" Corrected operating points (step injection, detection rate >= 90%)")
    print("=" * 96)
    print(f"\n  {'budget':>8}  {'nominal / 15,850':>22}  {'independent / 15,850':>22}"
          f"  {'independent / 95,040':>22}")
    print("  " + "-" * 92)

    results = []
    for bud in cfg.BUDGETS:
        row = {'budget': bud}
        cells = []
        for tag, clean, wph in [('nominal', clean_all, wph_nonoverlap),
                                ('indep_nonoverlap', clean_ind, wph_nonoverlap),
                                ('indep_sliding', clean_ind, wph_sliding)]:
            mag, fpr, ok = budget_floor(clean, atk, bud, wph)
            if not ok:
                cells.append('unestimable')
                row[tag] = None
            elif mag is None:
                cells.append('no operating point')
                row[tag] = 'none'
            else:
                mm = mag * cfg.MM_PER_DEG
                cells.append(f'{mm:.2f} mm')
                row[tag] = mm
        print(f"  {bud:>5}/h  {cells[0]:>22}  {cells[1]:>20}  {cells[2]:>20}")
        results.append(row)

    # Strictest directly estimable budget
    print()
    for tag, clean, wph, name in [
        ('nominal', clean_all, wph_nonoverlap, 'nominal samples, non-overlap'),
        ('indep_nonoverlap', clean_ind, wph_nonoverlap, 'independent, non-overlap'),
        ('indep_sliding', clean_ind, wph_sliding, 'independent, every window'),
    ]:
        strictest = wph / len(clean)
        print(f"  {name:<28} n={len(clean):>6,}  strictest {strictest:>7.1f} /hour")

    print(f"""
  -- How to read this ----------------------------------------
    The first column is what the paper currently says. It uses overlapping
    validation samples together with a non-overlapping decision frequency,
    which is inconsistent: if windows overlap enough to count as independent,

    then the decision frequency should be the sliding one too.

    The third column is the self-consistent, conservative one. If its
    strictest budget is looser than two per hour, the paper's reference point
    must loosen with it and 0.95 mm must be replaced by the matching value.
  ──────────────────────────────────────────────────────────────
""")
    with open(f'{cfg.OUT_DIR}/operating_points_fixed.json', 'w') as f_:
        json.dump(dict(
            windows_per_hour=dict(nonoverlap=wph_nonoverlap, sliding=wph_sliding),
            n_clean=dict(nominal=len(clean_all), independent=len(clean_ind)),
            rows=results), f_, indent=2, default=float)
    print(f"  wrote {cfg.OUT_DIR}/operating_points_fixed.json")
    print(f"  done in {time.time()-t0:.0f}s")
    print("=" * 96)


if __name__ == '__main__':
    main()
