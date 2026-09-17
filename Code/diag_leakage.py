
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] prep | Three negative controls: null injection, channel sanitisation, label shuffling
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Leakage check plus a multivariate upper bound

After three rounds of improving the analytic upper bound, a gap to the
learned detector remained. Smoothness features closed most of it but not all.
Pushing the bound further risks an infinite regress (more variables, torque
prediction, non-linearity), so the question becomes whether the gap is real
or a leak.

Four tests, ordered by sharpness:

  [T0] Null injection (sharpest and cheapest)
       inject_once(scale=0.0): step adds uniform(5,15)*0 = 0, ramp adds 0,
       noise adds normal(0,0) = 0. The injected tensor is therefore identical
       to the clean one element by element, while labels are assigned as
       usual. AUC must be 0.5. Anything clearly higher means the model
       identified labelled windows with no perturbation present, which is
       conclusive pipeline leakage.

  [T1] Channel sanitisation
       Restore the attacked joints' columns to their clean values so the
       attack disappears entirely from the observable input, keeping the
       labels. This is independent of the feature layout and needs no
       knowledge of how build_features orders channels. It catches a
       different fault from T0: if inject_once accidentally perturbs other
       joints, T0 passes and T1 exposes it.

  [T2] Label shuffling
       Shuffle y at a real scale. AUC must be 0.5. This catches correlation
       between labels and structural properties.

  [T3] Only if T0 to T2 all pass: a multivariate VAR upper bound
       Attacks target joints 0 and 1, never joint 2. On coordinated
       trajectories the three joints correlate strongly, so joint 2 serves as
       a reference channel: predicting joint 0 from it plus history gives a
       much smaller error than single-channel smoothness extrapolation. Every
       earlier bound treated the joints as independent, so this is the most
       likely remaining legitimate advantage.
       Implementation: fit a linear one-step VAR on clean training data using
       the history of all joints plus torque, freeze the coefficients, and
       use the prediction residual as a matched filter.

Verdict:
    any of T0/T1/T2 above AUC 0.60  -> leakage; locate and fix it, and every
                                       detection floor is void
    all near 0.50 and T3 catches up -> no leakage; the gap is explained by
                                       multivariate modelling and the numbers
                                       stand
    all near 0.50 and T3 still short -> no leakage; the gap comes from the
                                       network's non-linear modelling. That is
                                       itself reportable, but the paper cannot
                                       then claim to approach an information
                                       limit

Usage:    python3 diag_leakage.py
Expected: 9 training runs for T0 to T2, pure numpy for T3, 20 to 40 minutes
"""

import os, sys, json, time
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from raven2_peng_pipeline_v3 import build_raven2_graph
from exp_delta_ablation import (Cfg as BaseCfg, load_records, fit_physics_model,
                                build_window_cache, build_features)
from exp_paper_sweep import inject_once, detection_limit, run_one


class Cfg(BaseCfg):
    OUT_DIR = _OUT + '/output_leakage'

    # exp_paper_sweep.Cfg defines this; exp_delta_ablation.Cfg (BaseCfg) does not,
    # and run_one reads cfg.DETERMINISTIC directly, so it must be supplied here.
    # False matches the sweep. cuDNN non-determinism moves AUC by about 0.004
    # at fixed seed; not needed with multi-seed error bars and costs 20-30%.
    DETERMINISTIC = False

    MODE = 'jpos_torque_delta'          # matches the main sweep configuration
    SEEDS = [42, 43, 44]

    # Test scale: near the learned detector's floor, where it reports AUC ~0.9
    TEST = dict(step=(0.002,  [0]),
                ramp=(0.025,  [0, 1]),
                noise=(0.0005, [0]))

    GNN_FLOOR = dict(ramp=0.174, step=0.019, noise=0.0041)
    PASS_TH = 0.60                      # above this counts as leakage
    OBS_JOINTS = 3
    VAR_LAGS = 3

    # T3 magnitude sweep
    T3_SWEEPS = [
        dict(attack='step', joints=[0],
             scales=[0.02, 0.012, 0.008, 0.005, 0.003, 0.002, 0.0012, 0.0008]),
        dict(attack='ramp', joints=[0, 1],
             scales=[0.40, 0.25, 0.174, 0.10, 0.06, 0.04, 0.025, 0.015]),
        dict(attack='noise', joints=[0],
             scales=[0.012, 0.008, 0.005, 0.003, 0.002, 0.001, 0.0005, 0.0003]),
    ]
    SEED_OFFSETS = [0, 1000, 2000]
    AUC_TARGET = 0.90


XKEY = dict(step='dev_end', ramp='dev_end', noise='dev_rms')


def main():
    cfg = Cfg(); os.makedirs(cfg.OUT_DIR, exist_ok=True)
    t_all = time.time()
    print("=" * 100)
    print(" Leakage check plus multivariate upper bound")
    print("=" * 100)

    records = load_records(cfg)
    a, b, r2, std_dj, std_res = fit_physics_model(records, cfg.N_POS_JOINTS)
    edge_index, _ = build_raven2_graph()
    JP, TQ, YD, GID = build_window_cache(records, cfg)
    print(f"  windows {JP.shape}  groups {len(np.unique(GID))}  mode={cfg.MODE}")

    log = []

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(" [T0] Null injection  (JP_a must equal JP element by element)")
    print("=" * 100)
    t0_ok = True
    for atk, (sc, joints) in cfg.TEST.items():
        JP_a, y, aj, de, dr, dev = inject_once(JP, cfg, 0.0, atk, joints, 0)
        identical = np.array_equal(JP_a, JP)
        print(f"\n  {atk}: JP_a == JP ? {identical}   "
              f"max|diff|={np.abs(JP_a - JP).max():.3e}   "
              f"attacked={int(y.sum())}/{len(y)}")
        if not identical:
            print(f"    WARNING: JP_a changed at scale=0. The injector has a bug.")
        X = build_features(JP_a, TQ, cfg.MODE, a, b, cfg.N_POS_JOINTS)
        aucs = []
        for s in cfg.SEEDS:
            r = run_one(X, YD, y, GID, edge_index, cfg, s)
            aucs.append(r['auc'])
            print(f"    seed={s}  AUC={r['auc']:.4f}  F1={r['f1']:.4f}")
        m = float(np.mean(aucs))
        verdict = 'PASS' if m < cfg.PASS_TH else 'FAIL (leakage)'
        print(f"    -> mean AUC = {m:.4f}   {verdict}")
        log.append(dict(test='T0_null', attack=atk, auc=m, aucs=aucs,
                        identical=bool(identical)))
        if m >= cfg.PASS_TH: t0_ok = False
        del X, JP_a

    if not t0_ok:
        print("\n  T0 FAILED: classification without any perturbation means leakage.")
        print("     Check: (1) does index selection in inject_once depend on the data")
        print("            (2) does build_features leak anything label-correlated")
        print("            (3) do standardisation or the split in run_one touch y")
        json.dump(log, open(f'{cfg.OUT_DIR}/leakage.json', 'w'), indent=2)
        return

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(" [T1] Channel sanitisation  (attacked columns restored to clean values)")
    print("=" * 100)
    t1_ok = True
    for atk, (sc, joints) in cfg.TEST.items():
        JP_a, y, aj, de, dr, dev = inject_once(JP, cfg, sc, atk, joints, 0)
        JP_san = JP_a.copy()
        JP_san[:, :, joints] = JP[:, :, joints]
        resid = np.abs(JP_san - JP).max()
        print(f"\n  {atk} @ scale={sc}  dev_end={dev['dev_end']:.5f}")
        print(f"    after sanitisation max|JP_san - JP| = {resid:.3e}  "
              f"(must be 0; nonzero means the injection spilled to other joints)")
        X = build_features(JP_san, TQ, cfg.MODE, a, b, cfg.N_POS_JOINTS)
        aucs = []
        for s in cfg.SEEDS:
            r = run_one(X, YD, y, GID, edge_index, cfg, s)
            aucs.append(r['auc'])
            print(f"    seed={s}  AUC={r['auc']:.4f}")
        m = float(np.mean(aucs))
        verdict = 'PASS' if m < cfg.PASS_TH else 'FAIL (leakage)'
        print(f"    -> mean AUC = {m:.4f}   {verdict}")
        log.append(dict(test='T1_sanitized', attack=atk, auc=m, aucs=aucs,
                        spill=float(resid)))
        if m >= cfg.PASS_TH: t1_ok = False
        del X, JP_a, JP_san

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(" [T2] Label shuffling")
    print("=" * 100)
    t2_ok = True
    for atk, (sc, joints) in cfg.TEST.items():
        JP_a, y, aj, de, dr, dev = inject_once(JP, cfg, sc, atk, joints, 0)
        X = build_features(JP_a, TQ, cfg.MODE, a, b, cfg.N_POS_JOINTS)
        aucs = []
        for s in cfg.SEEDS:
            y_sh = np.random.RandomState(1000 + s).permutation(y)
            r = run_one(X, YD, y_sh, GID, edge_index, cfg, s)
            aucs.append(r['auc'])
            print(f"  {atk} seed={s}  AUC={r['auc']:.4f}")
        m = float(np.mean(aucs))
        verdict = 'PASS' if m < cfg.PASS_TH else 'FAIL'
        print(f"  {atk} -> mean AUC = {m:.4f}   {verdict}")
        log.append(dict(test='T2_shuffle', attack=atk, auc=m, aucs=aucs))
        if m >= cfg.PASS_TH: t2_ok = False
        del X, JP_a

    json.dump(log, open(f'{cfg.OUT_DIR}/leakage.json', 'w'), indent=2)

    print("\n" + "=" * 100)
    print(" Leakage check summary")
    print("=" * 100)
    print(f"  T0 null injection:       {'PASS' if t0_ok else 'FAIL'}")
    print(f"  T1 channel sanitisation: {'PASS' if t1_ok else 'FAIL'}")
    print(f"  T2 label shuffling:      {'PASS' if t2_ok else 'FAIL'}")
    if not (t0_ok and t1_ok and t2_ok):
        print("\n  Leakage present. Every detection floor is void. Fix the pipeline first.")
        return
    print("\n  All three pass: no pipeline leakage. The gap is a real modelling advantage.")

    # ════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print(" [T3] Multivariate VAR upper bound")
    print("=" * 100)
    print("  Attacks target joints 0 and 1, never joint 2, and on coordinated")
    print("  trajectories the three correlate, so joint 2 is a reference channel.")

    groups = np.unique(GID)
    n_val = max(1, int(round(len(groups) * cfg.VAL_FRAC_RECORDS)))
    val_g = np.random.RandomState(cfg.SEED).permutation(groups)[:n_val]
    m_va = np.isin(GID, val_g); m_tr = ~m_va

    p, J = cfg.VAR_LAGS, cfg.OBS_JOINTS
    T = JP.shape[1]

    def design(X3, TQ3):
        """Build the VAR design matrix: p lags of all joints plus current torque"""
        N = len(X3)
        rows = []
        for lag in range(1, p + 1):
            rows.append(X3[:, p - lag: T - lag, :])          # (N, T-p, J)
        rows.append(TQ3[:, p:, :])
        Z = np.concatenate(rows, axis=-1)                    # (N, T-p, p*J+J)
        return Z

    X3 = JP[:, :, :J].astype(np.float64)
    TQ3 = TQ[:, :, :J].astype(np.float64)
    Ztr = design(X3[m_tr], TQ3[m_tr]).reshape(-1, p*J + J)
    Ytr = X3[m_tr][:, p:, :].reshape(-1, J)
    # Add the constant term
    Ztr = np.concatenate([Ztr, np.ones((len(Ztr), 1))], axis=1)
    W, *_ = np.linalg.lstsq(Ztr, Ytr, rcond=None)            # (p*J+J+1, J)
    pred_tr = Ztr @ W
    sig_var = np.maximum((Ytr - pred_tr).std(axis=0), 1e-12).astype(np.float32)
    print(f"\n  VAR({p}) + torque one-step residual std = {sig_var}")
    print(f"  compare against the poly3 smoothness residual from base_smooth_glr")
    del Ztr, Ytr, pred_tr

    def var_resid(JPx):
        Z = design(JPx[:, :, :J].astype(np.float64), TQ3).reshape(-1, p*J + J)
        Z = np.concatenate([Z, np.ones((len(Z), 1))], axis=1)
        pr = (Z @ W).reshape(len(JPx), T - p, J)
        return (JPx[:, p:, :J] - pr).astype(np.float32)

    _T = {}
    def tpl(Tn, kind, lo, hi):
        k = (Tn, kind, lo, hi)
        if k in _T: return _T[k]
        rows = []
        for t0 in range(lo, min(hi, Tn - 1)):
            s = np.zeros(Tn)
            if kind == 'dipole':
                s[t0] = 1.0
                if t0 + 1 < Tn: s[t0+1] = -1.0
            else:
                s[t0] = 1.0
            rows.append(s / max(np.linalg.norm(s), 1e-12))
        M = np.asarray(rows, np.float32); _T[k] = M
        return M

    def stat(F, kind):
        N, Tn, Jn = F.shape
        lo = max(0, int(Tn*0.2)); hi = max(lo+1, int(Tn*0.8))
        if kind == 'energy':
            s = max(2, Tn // 2)
            acc = np.zeros((N, Jn), np.float32)
            for j in range(Jn):
                v0 = (F[:, :s, j]**2).mean(axis=1) + 1e-16
                v1 = (F[:, s:, j]**2).mean(axis=1) + 1e-16
                acc[:, j] = np.abs(np.log(v1/v0))
            return acc.max(axis=1)
        M = tpl(Tn, kind, lo, hi)
        acc = np.zeros((N, Jn), np.float32)
        for j in range(Jn):
            acc[:, j] = np.abs(F[:, :, j] @ M.T).max(axis=1) / sig_var[j]
        return np.sqrt((acc**2).sum(axis=1))

    rows_t3 = []
    for sw in cfg.T3_SWEEPS:
        atk, joints, xk = sw['attack'], sw['joints'], XKEY[sw['attack']]
        kind = 'dipole' if atk == 'step' else ('energy' if atk == 'noise' else 'impulse')
        print(f"\n  -- {atk}  statistic={kind} --")
        for sc in sw['scales']:
            injs = [inject_once(JP, cfg, sc, atk, joints, so)
                    for so in cfg.SEED_OFFSETS]
            xval = float(np.mean([i[5][xk] for i in injs]))
            aucs = []
            for JP_a, y, aj, de, dr, dev in injs:
                if len(np.unique(y[m_va])) < 2: continue
                F = var_resid(JP_a)
                s = stat(F, kind)
                aucs.append(roc_auc_score(y[m_va], s[m_va]))
                del F, s
            if not aucs: continue
            rows_t3.append(dict(attack=atk, scale=float(sc), xkey=xk, x=xval,
                                auc=float(np.mean(aucs)),
                                auc_std=float(np.std(aucs))))
            print(f"    sc={sc:<8.5f} {xk}={xval:<9.5f} AUC={np.mean(aucs):.4f} "
                  f"± {np.std(aucs):.4f}")
            del injs

    print("\n" + "=" * 100)
    print(" T3 conclusion")
    print("=" * 100)
    for sw in cfg.T3_SWEEPS:
        atk = sw['attack']; gf = cfg.GNN_FLOOR[atk]
        sub = [r for r in rows_t3 if r['attack'] == atk]
        f_ = detection_limit([r['x'] for r in sub], [r['auc'] for r in sub],
                             cfg.AUC_TARGET) if len(sub) >= 2 else np.nan
        print(f"\n  {atk}:  GNN={gf:.5f}")
        if np.isfinite(f_):
            print(f"    VAR bound {f_:>11.5f}   ratio = {f_/gf:.2f}x")
            if f_ <= gf * 1.3:
                print(f"    -> caught up. The gap is explained by linear multivariate modelling.")
                print(f"       The paper may say detection approaches that information limit.")
            elif f_ <= gf * 2.5:
                print(f"    -> much reduced. The remaining {f_/gf:.1f}x is non-linear modelling.")
            else:
                print(f"    -> still {f_/gf:.1f}x short. No leakage, but the linear bound is")
                print(f"       insufficient. The paper cannot claim to approach the limit.")
        else:
            print(f"    VAR bound none")

    json.dump(dict(leakage=log, t3=rows_t3),
              open(f'{cfg.OUT_DIR}/all.json', 'w'), indent=2)

    # Figures
    n = len(cfg.T3_SWEEPS)
    fig, axes = plt.subplots(1, n, figsize=(6.0*n, 4.6))
    if n == 1: axes = [axes]
    for ax, sw in zip(axes, cfg.T3_SWEEPS):
        atk = sw['attack']; gf = cfg.GNN_FLOOR[atk]
        sub = sorted([r for r in rows_t3 if r['attack'] == atk], key=lambda r: r['x'])
        if sub:
            ax.errorbar([r['x'] for r in sub], [r['auc'] for r in sub],
                        yerr=[r['auc_std'] for r in sub], marker='o',
                        color='#DC2626', lw=2, markersize=6, capsize=2,
                        label=f'VAR({p})+torque')
        ax.axvline(gf, color='black', ls='-.', lw=2, label=f'GNN {gf:.4f}')
        ax.axhline(cfg.AUC_TARGET, color='gray', ls=':', alpha=.7)
        ax.axhline(0.5, color='black', ls=':', alpha=.35)
        ax.set_xscale('log'); ax.set_ylim(0.4, 1.02)
        ax.set_xlabel(f'{XKEY[atk]} (deg)'); ax.set_ylabel('ROC-AUC')
        ax.set_title(atk, fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.suptitle('Multivariate VAR bound vs GNN (after leakage tests passed)',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{cfg.OUT_DIR}/var_bound.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\n  output: {cfg.OUT_DIR}/all.json")
    print(f"        {cfg.OUT_DIR}/var_bound.png")
    print(f"  total {time.time()-t_all:.0f}s")
    print("=" * 100)


if __name__ == '__main__':
    main()
