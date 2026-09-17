"""
Paper sweep: detectability is set by how concentrated the attack is in time

What the previous round (exp_delta_ablation.py) established:

  1. The difference operator explains the entire gain. The explicit physics
     residual (dj - a*tau - b) is an exact linear combination of dj and tau,
     and over 46,205 windows the network learns it on its own, so the gain is
     zero. The variance identity std(res) = sqrt(1-R2)*std(dj) holds exactly
     (1.505 / 2.040 / 2.314), which makes this a clean negative ablation
     rather than an implementation problem.

  2. The only real bottleneck is ramp. Step and noise hold recall at 1.0000
     across all four scales, so the whole decline at low magnitude comes from
     ramp.

  3. Per-frame SNR (relative to std(residual)) tracks recall one to one:
     above 1 it saturates, below 1 it starts to fall.

  4. The cuDNN LSTM backward pass is non-deterministic. Re-running the same
     seed moves AUC by about 0.004 in the collapsed region, so the 0.04 to
     0.06 gap between C and D on ramp is not established without multiple
     seeds.

This script produces the figure in the paper:

  - sweeps each attack type separately, since sweeping them together leaves
    step and noise saturated throughout and the curve then reflects ramp only
  - targets joints 0 and 1 only. Joint 2 is prismatic, so its units differ,
    and its std(residual) is an order of magnitude smaller, giving a
    six-times higher per-frame SNR that would smear the transition region
  - runs ramp with three seeds for error bars; step and noise are saturated
    and one seed suffices
  - puts measured absolute joint deviation in degrees on the x axis, which is
    what makes the three attacks comparable
  - fixes the split at cfg.SEED so only model initialisation varies with the
    seed, which is what keeps the error bars clean

Expected floors from inverting SNR = 1 (about 15 frames corrupted per window
on average, with ramp accumulating coherently as sqrt(15)): sample points are
placed around those, with two points of margin on each side.

Usage:    python3 exp_paper_sweep.py
Expected: ramp 2 joints x 5 points x 2 configs x 3 seeds = 60 runs,
          step and noise 5 runs each, 70 in total
"""


import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raven2_peng_pipeline_v3 import build_raven2_graph, DeviationDetector
from exp_delta_ablation import (
    Cfg as BaseCfg, load_records, fit_physics_model,
    build_window_cache, build_features, MaskedMultiTaskLoss, standardize,
)


# ══════════════════════════════════════════════════════════════
class Cfg(BaseCfg):
    OUT_DIR = _OUT + '/output_paper_sweep'

    # The cuDNN LSTM backward pass is non-deterministic by default; the same
    # seed moves AUC by about 0.004. Not needed with multi-seed error bars,
    DETERMINISTIC = False

    # -- sweep definition ------------------------------------- (costs 20-30%)
    # The scales must match earlier runs exactly, or the resume logic fails to
    # match and the whole column is recomputed. Not one digit may change.
    SWEEPS = [
        # step: seed 42 exists; 43 and 44 give the ramp/step ratio a denominator error bar
        dict(attack='step',  joints=[0],
             scales=[0.05, 0.03, 0.02, 0.012, 0.008],
             modes=['jpos_torque_delta'],
             seeds=[42, 43, 44]),

        # ramp: 42/43/44 exist; 45 and 46 settle the t value for delta vs residual
        dict(attack='ramp',  joints=[0, 1],
             scales=[0.10, 0.06, 0.04, 0.025, 0.015],
             modes=['jpos_torque_delta', 'jpos_torque_residual'],
             seeds=[42, 43, 44, 45, 46]),

        # noise: the previous sweep reached 0.008 (sd 0.02 deg/frame, 0.175 of the
        # noise floor) without crossing AUC=0.9. The lower bound of a deeper sweep
        # must be set by the encoder quantisation step first, otherwise genuine
        # dict(attack='noise', joints=[0],
        #      scales=[0.005, 0.002, 0.001, 0.0005],
        #      modes=['jpos_torque_delta'],
        #      seeds=[42]),
    ]

# ══════════════════════════════════════════════════════════════
# Injector: adds force_type and force_joint, and reports measured deviation (deg)
# ══════════════════════════════════════════════════════════════
class SweepInjector:
    TYPES = ['step', 'ramp', 'noise']

    @staticmethod
    def inject(jpos_window, scale=1.0, rng=None,
               force_type=None, force_joint=None):
        rng = rng or np.random
        T = jpos_window.shape[0]
        out = jpos_window.copy()
        atype = force_type or rng.choice(SweepInjector.TYPES)
        t_start = rng.randint(T // 3, 2 * T // 3)
        joint = force_joint if force_joint is not None else rng.randint(3)

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

        d = out[:, joint] - jpos_window[:, joint]
        dev_end = float(abs(d[-1]))                      # accumulated deviation at window end
        dev_rms = float(np.sqrt((d[t_start:] ** 2).mean()))   # RMS over the corrupted span
        return out.astype(np.float32), atype, joint, dev_end, dev_rms


def inject_once(JP, cfg, scale, attack, joints, seed_offset=0):
    """Inject once per (attack, scale); all modes and seeds share those attacks"""
    rng = np.random.RandomState(cfg.SEED + int(scale * 100000) + seed_offset)
    JP_a = JP.copy()
    y = np.zeros(len(JP), dtype=np.float32)
    aj = np.full(len(JP), -1, dtype=np.int64)
    de = np.zeros(len(JP), dtype=np.float32)
    dr = np.zeros(len(JP), dtype=np.float32)
    idx = rng.permutation(len(JP))[:int(len(JP) * cfg.ATTACK_FRAC)]
    for i in idx:
        j = joints[rng.randint(len(joints))]
        JP_a[i], _, aj[i], de[i], dr[i] = SweepInjector.inject(
            JP[i], scale=scale, rng=rng, force_type=attack, force_joint=j)
        y[i] = 1.0
    m = y > 0.5
    return JP_a, y, aj, de, dr, dict(dev_end=float(de[m].mean()),
                                     dev_rms=float(dr[m].mean()))


# ══════════════════════════════════════════════════════════════
def run_one(X, YD, y, gid, edge_index, cfg, seed, return_scores=False):
    """The split is fixed at cfg.SEED; only init and batch order follow the seed"""
    torch.manual_seed(seed); np.random.seed(seed)
    if cfg.DETERMINISTIC:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    dev = cfg.DEVICE

    groups = np.unique(gid)
    n_val = max(1, int(round(len(groups) * cfg.VAL_FRAC_RECORDS)))
    val_g = np.random.RandomState(cfg.SEED).permutation(groups)[:n_val]
    m_va = np.isin(gid, val_g); m_tr = ~m_va

    Xtr, Xva = standardize(X[m_tr], X[m_va])
    std_delta = YD[m_tr].std() + 1e-8
    Ytr = YD[m_tr] / std_delta
    ytr, yva = y[m_tr], y[m_va]

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
    with torch.no_grad():
        logits = torch.cat([model(Xva_t[s:s + 256], ei)[1].cpu()
                            for s in range(0, len(Xva_t), 256)]).numpy()
    prob = 1.0 / (1.0 + np.exp(-logits))
    pred = (prob > 0.5).astype(int)

    auc = roc_auc_score(yva, prob) if len(np.unique(yva)) > 1 else float('nan')
    p, r, f1, _ = precision_recall_fscore_support(yva, pred, average='binary',
                                                  zero_division=0)

    def tpr_at_fpr(t):
        neg = np.sort(prob[yva == 0])[::-1]
        if len(neg) == 0 or (yva == 1).sum() == 0:
            return float('nan')
        return float((prob[yva == 1] > neg[min(int(len(neg) * t), len(neg) - 1)]).mean())


    del model, opt, lossf, Xtr_t, Xva_t, Ytr_t, ytr_t, ei
    if cfg.DEVICE == 'cuda':
        torch.cuda.empty_cache()
        
    out = dict(auc=float(auc), f1=float(f1), precision=float(p), recall=float(r),
               tpr_at_fpr1=tpr_at_fpr(0.01), tpr_at_fpr01=tpr_at_fpr(0.001),
               prob_std=float(prob.std()), pred_pos_rate=float(pred.mean()))
    if return_scores:
        out['prob'] = prob
        out['val_mask'] = m_va
    return out


# ══════════════════════════════════════════════════════════════
def detection_limit(xs, vs, target=0.9):
    """Linear interpolation of AUC against log(x). Monotonicity is not guaranteed,
    so take the smallest crossing."""
    x = np.asarray(xs, float); v = np.asarray(vs, float)
    o = np.argsort(x); x, v = x[o], v[o]
    hits = []
    for i in range(len(x) - 1):
        if (v[i] - target) * (v[i + 1] - target) <= 0 and v[i] != v[i + 1]:
            w = (target - v[i]) / (v[i + 1] - v[i])
            hits.append(np.exp(np.log(x[i]) + w * (np.log(x[i + 1]) - np.log(x[i]))))
    return float(min(hits)) if hits else float('nan')


def main():
    XKEY = {'noise': 'dev_rms'}
    cfg = Cfg()
    os.makedirs(cfg.OUT_DIR, exist_ok=True)
    t0 = time.time()
    print(f'device = {cfg.DEVICE}   deterministic = {cfg.DETERMINISTIC}')

    records = load_records(cfg)
    a, b, r2, std_dj, std_res = fit_physics_model(records, cfg.N_POS_JOINTS)
    edge_index, _ = build_raven2_graph()
    JP, TQ, YD, GID = build_window_cache(records, cfg)
    print(f'windows {JP.shape}  groups {len(np.unique(GID))}')

    prev_path = f'{cfg.OUT_DIR}/results.json'
    results = []
    if os.path.exists(prev_path):
        results = json.load(open(prev_path))['results']
        print(f'loaded {len(results)} cached results')
    seen = {(r['attack'], r['mode'], r['scale'], r['seed']) for r in results}

    todo = []
    for sw in cfg.SWEEPS:
        for sc in sw['scales']:
            for mode in sw['modes']:
                pend = [s for s in sw['seeds']
                        if (sw['attack'], mode, sc, s) not in seen]
                if pend:
                    todo.append((sw['attack'], sw['joints'], sc, mode, pend))
    n_runs = sum(len(t[-1]) for t in todo)
    print(f'{n_runs} training runs needed\n')

    done, ck, cache = 0, None, None
    for atk, joints, sc, mode, pend in todo:
        if ck != (atk, sc):                      # inject once per (attack, scale)
            cache, ck = inject_once(JP, cfg, sc, atk, joints), (atk, sc)
        JP_a, y, aj, de, dr, dev = cache
        X = build_features(JP_a, TQ, mode, a, b, cfg.N_POS_JOINTS)
        for seed in pend:
            r = run_one(X, YD, y, GID, edge_index, cfg, seed)
            r.update(attack=atk, joints=joints, scale=sc, mode=mode,
                     seed=seed, dim=int(X.shape[-1]), **dev)
            results.append(r)
            done += 1
            print(f'[{done}/{n_runs}] {atk:<5} sc={sc:<6} {mode:<22} '
                  f'seed={seed}  AUC={r["auc"]:.4f} F1={r["f1"]:.4f} '
                  f'dev_end={dev["dev_end"]:.4f}deg  ({time.time()-t0:.0f}s)')
            with open(prev_path, 'w') as f:      # checkpoint after every cell
                json.dump(dict(results=results), f)

    # -- aggregate seeds by (attack, mode, scale) --------------
    agg = {}
    for r in results:
        k = (r['attack'], r['mode'], r['scale'])
        agg.setdefault(k, []).append(r)

    print('\n' + '=' * 100)
    print(f'{"attack":<7}{"mode":<24}{"scale":>8}{"dev_end":>10}{"dev_rms":>10}'
          f'{"AUC":>18}{"TPR@1%":>18}')
    print('-' * 100)
    for k in sorted(agg):
        rs = agg[k]
        au = np.array([x['auc'] for x in rs])
        tp = np.array([x['tpr_at_fpr1'] for x in rs])
        s = f'{k[0]:<7}{k[1]:<24}{k[2]:>8.3f}{rs[0]["dev_end"]:>10.4f}{rs[0]["dev_rms"]:>10.4f}'
        s += (f'{au.mean():>12.4f}±{au.std():.4f}' if len(rs) > 1
              else f'{au.mean():>18.4f}')
        s += (f'{tp.mean():>12.4f}±{tp.std():.4f}' if len(rs) > 1
              else f'{tp.mean():>18.4f}')
        print(s)

    # -- detection floor, x axis = measured absolute deviation --
    print('\n' + '=' * 70)
    print('Detection floor (AUC=0.9), x axis is dev_end at window end, in degrees')
    limits = {}
    for atk in ['step', 'ramp', 'noise']:
        for mode in ['jpos_torque_delta', 'jpos_torque_residual']:
            seeds = sorted({r['seed'] for r in results
                            if r['attack'] == atk and r['mode'] == mode})
            if not seeds:
                continue
            per_seed = []
            for sd in seeds:
                sub = [r for r in results if r['attack'] == atk
                       and r['mode'] == mode and r['seed'] == sd]
                xk = XKEY.get(atk, 'dev_end')
                per_seed.append(detection_limit([r[xk] for r in sub],
                                                [r['auc'] for r in sub]))
            v = np.array(per_seed, float)
            limits[(atk, mode)] = per_seed
            if np.all(np.isnan(v)):
                print(f'  {atk:<6} {mode:<24} never crosses 0.9, extend the sweep')
            elif len(v) > 1:
                print(f'  {atk:<6} {mode:<24} {np.nanmean(v):.4f} ± {np.nanstd(v):.4f} deg')
            else:
                print(f'  {atk:<6} {mode:<24} {np.nanmean(v):.4f} deg')

    ls = limits.get(('step', 'jpos_torque_delta'))
    lr = limits.get(('ramp', 'jpos_torque_delta'))
    if ls and lr and np.isfinite(np.nanmean(ls)) and np.isfinite(np.nanmean(lr)):
        print(f'\n  ramp / step floor ratio = {np.nanmean(lr)/np.nanmean(ls):.2f}x')
        print('  This quantifies the central claim: at equal end displacement, a gradual')

    c = limits.get(('ramp', 'jpos_torque_delta'))
    d = limits.get(('ramp', 'jpos_torque_residual'))
    if c and d:
        C = np.array([x for x in c if np.isfinite(x)], float)
        D = np.array([x for x in d if np.isfinite(x)], float)
        if len(C) > 1 and len(D) > 1:
            se = np.sqrt(C.std(ddof=1)**2 / len(C) + D.std(ddof=1)**2 / len(D))
            t = (D.mean() - C.mean()) / se if se > 0 else np.nan
            print(f'\n  on ramp: delta={C.mean():.4f}+/-{C.std(ddof=1):.4f}  '
                  f'residual={D.mean():.4f}±{D.std(ddof=1):.4f}  (n={len(C)},{len(D)})')
            print(f'  relative difference {(D.mean()/C.mean()-1)*100:+.1f}%   t={t:.2f}')
            print('  |t| < 2.5  -> no substantive difference, the negative ablation holds')
            print('  |t| >= 2.5 -> report direction and size; more parameters, same information')
        else:
            print('\n  too few seeds for a statistical judgement')
    print('=' * 70)

    with open(f'{cfg.OUT_DIR}/results.json', 'w') as f:
        json.dump(dict(results=results, r2=r2.tolist(),
                       std_dj=std_dj.tolist(), std_res=std_res.tolist(),
                       limits={f'{k[0]}|{k[1]}': v for k, v in limits.items()}),
                  f, indent=2)

    # -- paper figure: absolute degrees on x, all three attacks ---
    col = {'step': '#dc2626', 'ramp': '#d97706', 'noise': '#2563eb'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for atk in ['step', 'ramp', 'noise']:
        for mode, ls_ in [('jpos_torque_delta', '-'), ('jpos_torque_residual', '--')]:
            ks = sorted([k for k in agg if k[0] == atk and k[1] == mode],
                        key=lambda k: k[2])
            if not ks:
                continue
            xs = [agg[k][0][XKEY.get(atk,'dev_end')] for k in ks]
            for ax, key in zip(axes, ['auc', 'tpr_at_fpr1']):
                m = [np.mean([r[key] for r in agg[k]]) for k in ks]
                e = [np.std([r[key] for r in agg[k]]) for k in ks]
                ax.errorbar(xs, m, yerr=e, marker='o', ls=ls_, color=col[atk],
                            capsize=3, lw=1.8, ms=5,
                            label=f'{atk} ({"res" if "residual" in mode else "dj"})')
    for ax, name in zip(axes, ['AUC', 'TPR @ FPR=1%']):
        ax.set_xscale('log')
        ax.set_xlabel('Cumulative joint deviation at window end (deg)', fontsize=11)
        ax.set_ylabel(name, fontsize=11); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].axhline(0.9, ls=':', c='gray', lw=1)
    axes[0].axvline(std_res[0], ls=':', c='k', lw=1)
    axes[0].text(std_res[0], 0.52, ' std(residual)\n per frame', fontsize=8)
    plt.tight_layout()
    plt.savefig(f'{cfg.OUT_DIR}/paper_sweep.png', dpi=150)

    print(f'\nDone in {time.time()-t0:.0f}s')
    print(f'  {cfg.OUT_DIR}/results.json')
    print(f'  {cfg.OUT_DIR}/paper_sweep.png')


if __name__ == '__main__':
    main()
