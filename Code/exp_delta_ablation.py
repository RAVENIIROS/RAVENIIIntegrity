"""
Delta vs residual ablation (DeviationDetector variant)

Purpose: separate the three possible sources of the gain attributed to the
"physics residual".

    1. the difference operator (dj turns a cross-step jump into per-frame
       evidence)
    2. the torque channels themselves
    3. the variance reduction from the linear model a*tau + b

Four configurations, each adding exactly one thing:

    A. jpos                             8 dim   baseline
    B. jpos + torque                   16 dim   the original configuration
    C. jpos + torque + dj              19 dim   difference only, no physics
    D. jpos + torque + dj + residual   22 dim   full physics

The point: residual = dj - (a*tau + b) is an exact linear combination of two
columns already present in C, so D adds zero information over C. If D beats
C, it wins on signal-to-noise or on learnability, not on information.

Analytic prediction: an attack perturbs dj and residual by the same absolute
amount, but their baseline variances differ, so D's detection floor should
sit below C's by a factor 1/sqrt(1-R2). With R2 = [0.538, 0.751, 0.813] that
predicts 1.47x / 2.00x / 2.31x.

Differences from exp_physics_residual.py:
    - each scale is injected once and all four configurations share the same
      attacked windows, making the comparison paired
    - the physics fit is causally aligned: dj[t] = jp[t] - jp[t-1] pairs with
      tau[t-1], not with the torque at the end of the interval
    - train and validation split by record rather than by window, so the 50%
      window overlap cannot leak
    - additionally prints std(dj) / std(residual) and the measured floor
      ratio against the analytic prediction

Usage:    python3 exp_delta_ablation.py
Expected: 16 training runs x 40 epochs
"""


import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peng_uw_loader import PengUWLoader
from raven2_peng_pipeline_v3 import (
    raven2_fk_left, build_raven2_graph, DeviationDetector
)


# ══════════════════════════════════════════════════════════════
class Cfg:
    DATA_ROOT = _DATA
    OUT_DIR   = _OUT + '/output_delta_ablation'
    RECORDINGS = ['record_1_different_directions']
    MAX_FILES  = 8
    SUBSAMPLE  = 5
    MAX_SAMPLES_PER_CSV = 20000

    LOOKBACK, LOOKAHEAD = 30, 20
    EPOCHS = 40
    BATCH  = 128
    LR     = 1e-3
    HIDDEN = 128
    NUM_JOINTS = 7
    ATTACK_FRAC = 0.30
    SEED = 42

    N_POS_JOINTS = 3          # joints that are attacked and modelled
    VAL_FRAC_RECORDS = 0.25
    STRIDE = 1           # window stride

    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1

    SCALES = [1.0, 0.5, 0.25, 0.1]
    FEATURE_SETS = ['jpos', 'jpos_torque', 'jpos_torque_delta', 'jpos_torque_residual']
    # To isolate the torque contribution further, add 'jpos_delta' (11 dim)

    FK_TO_MM = 1.0         # ADAPTER: set to 1.0 if the FK already returns mm
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


LABEL = {
    'jpos': 'jpos',
    'jpos_delta': 'jpos + dj',
    'jpos_torque': 'jpos + tau',
    'jpos_torque_delta': 'jpos + tau + dj',
    'jpos_torque_residual': 'jpos + tau + dj + res',
}


# ══════════════════════════════════════════════════════════════
# ADAPTER 1: data loading. Replace with the real call from your loader
# ══════════════════════════════════════════════════════════════
def load_records(cfg):
    records = []
    for rec in cfg.RECORDINGS:
        records.extend(PengUWLoader.load_directory(
            os.path.join(cfg.DATA_ROOT, rec),
            subsample=cfg.SUBSAMPLE,
            max_files=cfg.MAX_FILES,
            max_samples_per_file=cfg.MAX_SAMPLES_PER_CSV,
            verbose=False,
        ))
    assert len(records) > 0, 'loader returned no data'
    for x in records:
        assert 'robot_jpos' in x and 'motor_torque' in x
    return records


# ══════════════════════════════════════════════════════════════
# ADAPTER 2: forward kinematics. Accepts a per-frame or a batched signature
# ══════════════════════════════════════════════════════════════
def fk_xyz(jpos, cfg):
    """jpos (N, >=3) -> (N, 3), in mm"""
    return np.asarray(raven2_fk_left(np.asarray(jpos)[:, :3]),
                      dtype=np.float64)[:, :3] * cfg.FK_TO_MM


# ══════════════════════════════════════════════════════════════
# Physics model: dj[t] = jp[t] - jp[t-1] pairs with tau[t-1] (causal, the
# ══════════════════════════════════════════════════════════════
def fit_physics_model(records, n_j, train_frac=0.8):
    dj_all, tq_all = [], []
    for d in records:
        n = int(len(d['robot_jpos']) * train_frac)
        jp = d['robot_jpos'][:n, :n_j]
        tq = d['motor_torque'][:n, :n_j]
        dj_all.append(np.diff(jp, axis=0))    # dj[i] = jp[i+1] - jp[i]
        tq_all.append(tq[:-1])                # start of the interval, not the end
    DJ = np.concatenate(dj_all)
    TQ = np.concatenate(tq_all)

    a  = np.zeros(n_j, dtype=np.float32)
    b  = np.zeros(n_j, dtype=np.float32)
    r2 = np.zeros(n_j, dtype=np.float32)
    for j in range(n_j):
        x, y = TQ[:, j], DJ[:, j]
        if x.std() < 1e-9:
            continue
        a[j] = np.cov(x, y, bias=True)[0, 1] / x.var()
        b[j] = y.mean() - a[j] * x.mean()
        pred = a[j] * x + b[j]
        r2[j] = 1.0 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    RES = DJ - (a[None, :] * TQ + b[None, :])
    std_dj, std_res = DJ.std(axis=0), RES.std(axis=0)

    print('\nFitting the physics model  dj_j ~ a_j * tau_j + b_j  (clean, causal)')
    for j in range(n_j):
        print(f'  joint {j}:  a={a[j]:9.4f}  b={b[j]:9.5f}   R2 = {r2[j]:.4f}')
    print('\nVariance diagnostic')
    print(f'  std(dj)              = {np.array2string(std_dj,  precision=5)}')
    print(f'  std(residual)        = {np.array2string(std_res, precision=5)}')
    print(f'  measured std(dj)/std(res) = {np.array2string(std_dj/std_res, precision=3)}')
    print(f'  analytic 1/sqrt(1-R2)     = {np.array2string(1.0/np.sqrt(1.0-r2), precision=3)}')
    print('  The two rows should agree; a large gap means the variance or R2 is wrong\n')
    return a, b, r2, std_dj, std_res


# ══════════════════════════════════════════════════════════════
# Attack injection (identical to exp_physics_residual.py plus one scale)
# ══════════════════════════════════════════════════════════════
class ScalableInjector:
    TYPES = ['step', 'ramp', 'noise']

    @staticmethod
    def inject(jpos_window, scale=1.0, rng=None):
        rng = rng or np.random
        T = jpos_window.shape[0]
        out = jpos_window.copy()
        atype = rng.choice(ScalableInjector.TYPES)
        t_start = rng.randint(T // 3, 2 * T // 3)
        joint = rng.randint(3)

        if atype == 'step':
            mag = (rng.uniform(5, 15) if joint < 2 else rng.uniform(2, 8)) * scale
            out[t_start:, joint] += mag * rng.choice([-1, 1])
        elif atype == 'ramp':
            slope = (rng.uniform(0.3, 0.7) if joint < 2 else rng.uniform(0.15, 0.35)) * scale
            slope *= rng.choice([-1, 1])
            out[t_start:, joint] += slope * np.arange(T - t_start)
        elif atype == 'noise':
            sd = (2.5 if joint < 2 else 1.0) * scale
            out[t_start:, joint] += rng.normal(0, sd, size=T - t_start)
        return out.astype(np.float32), atype


# ══════════════════════════════════════════════════════════════
# Window cache: build the clean windows and regression targets before injecting
# The regression target uses the clean future; the attack corrupts only the input
# ══════════════════════════════════════════════════════════════
def build_window_cache(records, cfg):
    L, LA = cfg.LOOKBACK, cfg.LOOKAHEAD
    JP, TQ, YD, GID = [], [], [], []
    for gid, d in enumerate(records):
        jp, tq = np.asarray(d['robot_jpos']), np.asarray(d['motor_torque'])
        n = min(len(jp), len(tq))
        if n < L + LA + 1:
            continue
        ee = fk_xyz(jp[:n], cfg)                       # (n, 3) mm
        for s in range(0, n - L - LA, cfg.STRIDE):
            t = s + L - 1                              # last frame of the window
            JP.append(jp[s:s + L])
            TQ.append(tq[s:s + L])
            YD.append(ee[t + 1:t + 1 + LA] - ee[t])    # future offset from current EE
            GID.append(gid)
    return (np.asarray(JP, dtype=np.float32),
            np.asarray(TQ, dtype=np.float32),
            np.asarray(YD, dtype=np.float32),
            np.asarray(GID, dtype=np.int64))


def inject_once(JP, cfg, scale):
    """Inject once per scale; all four feature modes share the result"""
    rng = np.random.RandomState(cfg.SEED + int(scale * 10000))
    JP_a = JP.copy()
    y = np.zeros(len(JP), dtype=np.float32)
    at = np.array(['clean'] * len(JP), dtype=object)
    idx = rng.permutation(len(JP))[:int(len(JP) * cfg.ATTACK_FRAC)]
    for i in idx:
        JP_a[i], at[i] = ScalableInjector.inject(JP[i], scale=scale, rng=rng)
        y[i] = 1.0
    return JP_a, y, at


# ══════════════════════════════════════════════════════════════
# Feature construction: C and D share the same dj so the ablation stays clean
# ══════════════════════════════════════════════════════════════
def build_features(JP_a, TQ, mode, a, b, n_j):
    """(N, T, 8) -> (N, T, F)"""
    need_dj = mode in ('jpos_delta', 'jpos_torque_delta', 'jpos_torque_residual')
    parts = [JP_a]

    if need_dj:
        dj = np.zeros_like(JP_a[:, :, :n_j])
        dj[:, 1:, :] = np.diff(JP_a[:, :, :n_j], axis=1)

    if 'torque' in mode:
        parts.append(TQ)
    if need_dj:
        parts.append(dj)

    if mode == 'jpos_torque_residual':
        tq_lag = np.zeros_like(TQ[:, :, :n_j])
        tq_lag[:, 1:, :] = TQ[:, :-1, :n_j]            # same causal alignment as the fit
        res = dj - (a[None, None, :] * tq_lag + b[None, None, :])
        res[:, 0, :] = 0.0
        parts.append(res)

    return np.concatenate(parts, axis=2).astype(np.float32)


# ══════════════════════════════════════════════════════════════
class MaskedMultiTaskLoss(nn.Module):
    """As in v3 plus pos_weight, which fixed the collapse in exp_physics_residual"""
    def __init__(self, w_dev=1.0, w_atk=0.5, w_vel=0.1, pos_weight=None):
        super().__init__()
        self.w_dev, self.w_atk, self.w_vel = w_dev, w_atk, w_vel
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, dev_p, atk_p, dev_t, atk_t):
        clean = (atk_t < 0.5)
        if clean.any():
            dp, dt = dev_p[clean], dev_t[clean]
            l_dev = F.mse_loss(dp, dt)
            l_vel = F.mse_loss(dp[:, 1:] - dp[:, :-1], dt[:, 1:] - dt[:, :-1])
        else:
            l_dev = dev_p.sum() * 0.0
            l_vel = dev_p.sum() * 0.0
        l_atk = self.bce(atk_p, atk_t)
        return self.w_dev * l_dev + self.w_vel * l_vel + self.w_atk * l_atk


def standardize(tr, va):
    mu = tr.reshape(-1, tr.shape[-1]).mean(0)
    sd = tr.reshape(-1, tr.shape[-1]).std(0) + 1e-8
    return (tr - mu) / sd, (va - mu) / sd


def run_one(X, YD, y, at, gid, edge_index, cfg):
    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    dev = cfg.DEVICE

    groups = np.unique(gid)
    n_val = max(1, int(round(len(groups) * cfg.VAL_FRAC_RECORDS)))
    rs = np.random.RandomState(cfg.SEED)
    val_g = rs.permutation(groups)[:n_val]
    m_va = np.isin(gid, val_g); m_tr = ~m_va

    Xtr, Xva = standardize(X[m_tr], X[m_va])
    std_delta = YD[m_tr].std() + 1e-8
    Ytr, Yva = YD[m_tr] / std_delta, YD[m_va] / std_delta
    ytr, yva = y[m_tr], y[m_va]
    at_va = at[m_va]

    Xtr_t = torch.from_numpy(Xtr).to(dev); Ytr_t = torch.from_numpy(Ytr).to(dev)
    ytr_t = torch.from_numpy(ytr).to(dev); Xva_t = torch.from_numpy(Xva).to(dev)
    ei = edge_index.to(dev)

    model = DeviationDetector(input_dim=X.shape[-1], hidden=cfg.HIDDEN,
                              num_joints=cfg.NUM_JOINTS,
                              lookahead=cfg.LOOKAHEAD).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.LR)
    pw = torch.tensor([(len(ytr) - ytr.sum()) / max(ytr.sum(), 1.0)], device=dev)
    lossf = MaskedMultiTaskLoss(cfg.W_DEV, cfg.W_ATK, cfg.W_VEL, pos_weight=pw).to(dev)

    n = len(Xtr_t)
    for _ in range(cfg.EPOCHS):
        model.train()
        perm = torch.randperm(n, device=dev)
        for s in range(0, n, cfg.BATCH):
            i = perm[s:s + cfg.BATCH]
            opt.zero_grad()
            dp, ap = model(Xtr_t[i], ei)
            lossf(dp, ap, Ytr_t[i], ytr_t[i]).backward()
            opt.step()

    model.eval()
    logits = []
    with torch.no_grad():
        for s in range(0, len(Xva_t), 256):
            logits.append(model(Xva_t[s:s + 256], ei)[1].cpu())
    logits = torch.cat(logits).numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob > 0.5).astype(int)

    auc = roc_auc_score(yva, prob) if len(np.unique(yva)) > 1 else float('nan')
    p, r, f1, _ = precision_recall_fscore_support(yva, pred, average='binary',
                                                  zero_division=0)
    per_at = {t: (float(pred[at_va == t].mean()) if (at_va == t).sum() else float('nan'))
              for t in ScalableInjector.TYPES}

    def tpr_at_fpr(target):
        neg = np.sort(prob[yva == 0])[::-1]
        if len(neg) == 0 or (yva == 1).sum() == 0:
            return float('nan')
        thr = neg[min(int(len(neg) * target), len(neg) - 1)]
        return float((prob[yva == 1] > thr).mean())

    return dict(auc=float(auc), f1=float(f1), precision=float(p), recall=float(r),
                per_attack_recall=per_at,
                tpr_at_fpr1=tpr_at_fpr(0.01), tpr_at_fpr01=tpr_at_fpr(0.001),
                prob=prob, y=yva, at=at_va)

# ══════════════════════════════════════════════════════════════
def detection_limit(scales, aucs, target=0.9):
    s = np.array(scales, float); v = np.array(aucs, float)
    o = np.argsort(s); s, v = s[o], v[o]
    for i in range(len(s) - 1):
        if (v[i] - target) * (v[i + 1] - target) <= 0 and v[i] != v[i + 1]:
            w = (target - v[i]) / (v[i + 1] - v[i])
            return float(np.exp(np.log(s[i]) + w * (np.log(s[i + 1]) - np.log(s[i]))))
    return float('nan')


def main():
    cfg = Cfg()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f'device = {cfg.DEVICE}')

    records = load_records(cfg)
    print(f'loaded {len(records)} records')

    a, b, r2, std_dj, std_res = fit_physics_model(records, cfg.N_POS_JOINTS)
    edge_index, _ = build_raven2_graph()

    JP, TQ, YD, GID = build_window_cache(records, cfg)
    print(f'windows {JP.shape}  Yd {YD.shape}  groups {len(np.unique(GID))}')

    print('\nPer-frame SNR estimate (joint 0, relative to std(residual))')
    print(f'{"scale":>7}{"step":>10}{"ramp":>10}{"noise":>10}')
    for sc in cfg.SCALES:
        print(f'{sc:>7.2f}{10.0*sc/std_res[0]:>10.2f}'
              f'{0.5*sc/std_res[0]:>10.2f}{2.5*sc*np.sqrt(2)/std_res[0]:>10.2f}')
    print()

    results = []
    for sc in cfg.SCALES:
        JP_a, y, at = inject_once(JP, cfg, sc)     # all four configs share these attacks
        for mode in cfg.FEATURE_SETS:
            X = build_features(JP_a, TQ, mode, a, b, cfg.N_POS_JOINTS)
            print(f'  scale={sc:<5} {mode:<22} dim={X.shape[-1]:<3}', end=' ', flush=True)
            r = run_one(X, YD, y, at, GID, edge_index, cfg)
            r.update(scale=sc, mode=mode, dim=int(X.shape[-1]))
            results.append(r)
            print(f'AUC={r["auc"]:.4f} F1={r["f1"]:.4f} '
                  f'noise_rec={r["per_attack_recall"]["noise"]:.2f} '
                  f'({time.time()-t0:.0f}s)')

    W = 24
    def table(metric):
        print(f'\nSummary: {metric}')
        print('=' * (7 + W * len(cfg.FEATURE_SETS)))
        print(f'{"Scale":>7}' + ''.join(m.rjust(W) for m in cfg.FEATURE_SETS))
        print('-' * (7 + W * len(cfg.FEATURE_SETS)))
        for sc in cfg.SCALES:
            row = f'{sc:>7.2f}'
            for m in cfg.FEATURE_SETS:
                v = next(r[metric] for r in results
                         if r['scale'] == sc and r['mode'] == m)
                row += f'{v:.4f}'.rjust(W)
            print(row)

    table('auc'); table('f1'); table('tpr_at_fpr1')

    for t in ScalableInjector.TYPES:
        print(f'\nSummary: recall ({t})')
        print(f'{"Scale":>7}' + ''.join(m.rjust(W) for m in cfg.FEATURE_SETS))
        for sc in cfg.SCALES:
            row = f'{sc:>7.2f}'
            for m in cfg.FEATURE_SETS:
                v = next(r['per_attack_recall'][t] for r in results
                         if r['scale'] == sc and r['mode'] == m)
                row += f'{v:.4f}'.rjust(W)
            print(row)

    print('\n' + '=' * 70)
    print('Detection floor (scale where AUC reaches 0.9, lower is better)')
    lims = {}
    for m in cfg.FEATURE_SETS:
        s = [r['scale'] for r in results if r['mode'] == m]
        v = [r['auc'] for r in results if r['mode'] == m]
        lims[m] = detection_limit(s, v)
        print(f'  {m:<24} {lims[m]:.4f}')

    c, d = 'jpos_torque_delta', 'jpos_torque_residual'
    if np.isfinite(lims.get(c, np.nan)) and np.isfinite(lims.get(d, np.nan)):
        obs = lims[c] / lims[d]
        pred = float(1.0 / np.sqrt(1.0 - r2[:2].mean()))
        print(f'\n  measured gain of residual over delta = {obs:.2f}x')
        print(f'  gain predicted analytically from R2 = {pred:.2f}x')
        print('  close      -> the mechanism is variance reduction, i.e. learnability')
        print('  around 1.0 -> the difference operator already saturates the gain')
        print('  far above  -> some other mechanism is at work, investigate')
    else:
        print('\n  A configuration never crosses AUC=0.9 on this grid; extend it downward')
    print('=' * 70)

    with open(f'{cfg.OUT_DIR}/results.json', 'w') as f:
        json.dump(dict(results=results, a=a.tolist(), b=b.tolist(), r2=r2.tolist(),
                       std_dj=std_dj.tolist(), std_res=std_res.tolist(),
                       limits=lims), f, indent=2)

    colors = {'jpos': '#dc2626', 'jpos_delta': '#7c3aed', 'jpos_torque': '#2563eb',
              'jpos_torque_delta': '#d97706', 'jpos_torque_residual': '#059669'}
    marks = {'jpos': 's', 'jpos_delta': 'v', 'jpos_torque': '^',
             'jpos_torque_delta': 'D', 'jpos_torque_residual': 'o'}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for m in cfg.FEATURE_SETS:
        xs = [r['scale'] for r in results if r['mode'] == m]
        for ax, key in zip(axes, ['auc', 'f1', 'tpr_at_fpr1']):
            ys = [r[key] for r in results if r['mode'] == m]
            ax.plot(xs, ys, marker=marks.get(m, 'o'), color=colors.get(m),
                    label=LABEL.get(m, m), lw=1.8, ms=6)
    for ax, name in zip(axes, ['AUC', 'F1', 'TPR @ FPR=1%']):
        ax.set_xscale('log'); ax.set_xlabel('Attack magnitude scale', fontsize=11)
        ax.set_ylabel(name, fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    axes[0].axhline(0.9, ls=':', c='gray', lw=1)
    plt.tight_layout()
    plt.savefig(f'{cfg.OUT_DIR}/delta_ablation.png', dpi=150)

    print(f'\nDone in {time.time()-t0:.0f}s')
    print(f'  {cfg.OUT_DIR}/results.json')
    print(f'  {cfg.OUT_DIR}/delta_ablation.png')


if __name__ == '__main__':
    main()
