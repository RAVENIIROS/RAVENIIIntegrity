
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ1 | Session-scale observable and four checks (113.7 units, 31x)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Slow-path observability: the session-scale telemetry signature of cable drift

An earlier probe concluded that drift is invisible in telemetry. That was an
overgeneralisation: it tested one feature (the physics residual under frozen
a, b) at one timescale (a 30-frame window).

Invisibility at window scale is arithmetic, not a finding:
    132 Hz x 30 frames        = 0.227 s window
    measured drift            = 0.800 deg over 6 h = 3.70e-5 deg/s
    drift per window          = 8.4e-6 deg
    ramp detection floor      = 0.174 deg
    ratio                     = about 2e4
A ramp injected at the natural drift rate sits four orders of magnitude below
the floor of any window-scale detector, so the fast path cannot cover
drift-rate attacks and an independent slow path is required.

This is also the security reason drift belongs in the threat model. It is not
a benign noise source; it defines the envelope an adversary can hide in. A
monitor that treats all slow change as benign grants roughly 0.8 degrees of
free deviation, about 6 mm at the tip.

This script tests four candidate slow-path observables, all built from
channels available at deployment (robot_jpos, motor_torque, motor_pos,
motor_vel, joint_vel) and none from the external encoder:

  [S1] Drift in the fitted coefficients
       Refit dj_j = a_j*tau_j + b_j hour by hour and see whether a_j itself
       drifts. Cable stretch changes the motor-to-joint transmission, so
       coefficient drift is the natural telemetry signature of cable
       degradation. The earlier probe froze a and b and therefore could not
       see it. Reports the a_j trajectory, a Spearman trend, and a bootstrap
       uncertainty so that drift can be judged against fitting noise.

  [S2] Paired-motor redundancy
       Column probing shows motor_pos 134/135 and 136/137 are nearly
       identical. When two motors drive one degree of freedom through a
       cable, their difference is a redundant measurement: stretch anywhere
       along the cable moves it. This is a drift observable that needs no
       external encoder.
       Note: these pairs sit on the tool joints, not the positioning joints
       0 to 2, so the principle holds but may not reach the joints that
       matter. The script detects which columns pair automatically.

  [S3] motor_pos to robot_jpos consistency
       If jpos is computed from motor_pos through a fixed transmission model,
       the relation is constant, the residual is identically zero, and it
       carries no information. That would explain the earlier negative
       result. If the relation drifts, the residual is a direct drift
       observable.
       This check is the decisive one: it separates "drift is unobservable"
       from "we were not looking at the right quantity".

  [S4] Slow-path detection floor
       On whichever observable [S1] to [S3] finds most sensitive, accumulate
       with CUSUM and measure how long a window is needed to detect drift,
       and the smallest detectable drift rate. That number, together with the
       fast path's 0.174 deg, bounds the gap between the two paths, which is
       where an adversary would hide.

Usage: python3 probe_slow_path.py
"""

import os, sys, glob, gc, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, _DATA)
from peng_uw_loader import COL

ROOT = _DATA
OUT  = _OUT + '/output_slow_path'
os.makedirs(OUT, exist_ok=True)

RECORDS = ['record_3_time_decay_idle',
           'record_3_time_decay_unloaded',
           'record_3_time_decay_500gload']
SUBSAMPLE = 20
CHUNKSIZE = 50000
N_POS = 3
GT_SCALE = 180.0 / np.pi
N_BOOT = 200          # bootstrap resamples for coefficient uncertainty

# Fast-path reference, for comparison in [S4]
FAST_FLOOR_RAMP_DEG = 0.174
WINDOW_FRAMES = 30
FS_HZ = 132.0


NEEDED = sorted(set(
    list(range(COL['gt_jpos'][0], COL['gt_jpos'][1])) +
    list(range(COL['robot_jpos'][0], COL['robot_jpos'][1])) +
    list(range(COL['motor_torque'][0], COL['motor_torque'][1])) +
    list(range(COL['motor_pos'][0], COL['motor_pos'][1])) +
    list(range(COL['motor_vel'][0], COL['motor_vel'][1]))
))


def read_chunked(path):
    parts = []
    for ch in pd.read_csv(path, header=None, usecols=NEEDED, chunksize=CHUNKSIZE):
        parts.append(ch.iloc[::SUBSAMPLE])
    df = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()
    def sl(key):
        lo, hi = COL[key]
        return df[list(range(lo, hi))].values.astype(np.float64)
    return dict(gt=sl('gt_jpos') * GT_SCALE,
                jpos=sl('robot_jpos'),
                torque=sl('motor_torque'),
                mpos=sl('motor_pos'),
                mvel=sl('motor_vel'))


def hour_of(p):
    b = os.path.basename(p)
    for h in range(8):
        if f'_h{h}.' in b or f'_h{h}_' in b:
            return h
    return None


def fit_ab(jpos, tq, n=N_POS):
    """OLS dj_j = a_j*tau_j + b_j"""
    dj = np.diff(jpos[:, :n], axis=0); t = tq[1:, :n]
    a = np.zeros(n); b = np.zeros(n); r2 = np.zeros(n)
    for j in range(n):
        x, y = t[:, j], dj[:, j]
        if x.std() < 1e-12: continue
        a[j] = np.cov(x, y, bias=True)[0, 1] / x.var()
        b[j] = y.mean() - a[j] * x.mean()
        pr = a[j]*x + b[j]
        r2[j] = 1 - ((y-pr)**2).sum() / max(((y-y.mean())**2).sum(), 1e-12)
    return a, b, r2


def fit_ab_boot(jpos, tq, n=N_POS, nboot=N_BOOT, seed=0):
    """Bootstrap standard error of a_j, to judge hourly drift against fitting noise"""
    rng = np.random.RandomState(seed)
    dj = np.diff(jpos[:, :n], axis=0); t = tq[1:, :n]
    N = len(dj)
    out = np.zeros((nboot, n))
    for k in range(nboot):
        idx = rng.randint(0, N, N)
        for j in range(n):
            x, y = t[idx, j], dj[idx, j]
            if x.std() < 1e-12: continue
            out[k, j] = np.cov(x, y, bias=True)[0, 1] / x.var()
    return out.std(axis=0)


# ══════════════════════════════════════════════════════════════
print("=" * 100)
print(" Slow-path observability: session-scale telemetry signature of cable drift")
print("=" * 100)
print(f"\n  Window-scale impossibility, quantified first as the motivation:")
win_s = WINDOW_FRAMES / FS_HZ
drift_rate = 0.800 / (6 * 3600)
per_win = drift_rate * win_s
print(f"    window duration       = {win_s*1000:.1f} ms")
print(f"    measured drift rate   = {drift_rate:.3e} deg/s  (0.800 deg / 6 h)")
print(f"    drift per window      = {per_win:.3e} deg")
print(f"    fast-path ramp floor  = {FAST_FLOOR_RAMP_DEG} deg")
print(f"    ratio                 = {FAST_FLOOR_RAMP_DEG/per_win:.2e}")
print(f"    -> a ramp at the natural drift rate sits "
      f"{FAST_FLOOR_RAMP_DEG/per_win:.0e}x below the fast-path floor.")

# == [S2] First detect automatically which motor columns pair ==
print("\n" + "=" * 100)
print(" [S2-prep] Automatic detection of paired motor_pos columns")
print("=" * 100)
# Filter None before sorting, otherwise sorted compares None with int
files0 = [(hour_of(f), f) for f in sorted(glob.glob(f'{ROOT}/{RECORDS[2]}/*.csv'))]
files0 = sorted([(h, f) for h, f in files0 if h is not None])
if not files0:
    print("  no _h* files found"); sys.exit(1)
d0 = read_chunked(files0[0][1])
MP = d0['mpos']
nm = MP.shape[1]
pairs = []
print(f"\n  {'i':>3}{'j':>4}{'corr':>10}{'mean|diff|':>13}{'std(diff)':>13}"
      f"{'std_i':>12}   verdict")
print("-" * 100)
for i in range(nm):
    for j in range(i+1, nm):
        if MP[:, i].std() < 1e-9 or MP[:, j].std() < 1e-9: continue
        r = np.corrcoef(MP[:, i], MP[:, j])[0, 1]
        dif = MP[:, i] - MP[:, j]
        # Pairing test: high correlation and a difference SD far below each SD
        redundant = r > 0.999 and dif.std() < 0.05 * MP[:, i].std()
        if r > 0.99:
            print(f"  {i:>3}{j:>4}{r:>10.5f}{np.abs(dif).mean():>13.4f}"
                  f"{dif.std():>13.4f}{MP[:, i].std():>12.2f}   "
                  f"{'redundant pair' if redundant else 'correlated, not redundant'}")
        if redundant:
            pairs.append((i, j))
print(f"\n  {len(pairs)} redundant pair(s): {pairs}")
if pairs:
    print(f"  Note: confirm which joints these belong to. If all are tool joints,")
    print(f"      the observable does not reach positioning joints 0-2 and [S2] is limited.")
del d0, MP; gc.collect()


# == Hour-by-hour scan ==
print("\n" + "=" * 100)
print(" [S1][S2][S3] Hour-by-hour scan")
print("=" * 100)
rows = []
for rec in RECORDS:
    dpath = f'{ROOT}/{rec}'
    if not os.path.exists(dpath):
        print(f"\n  -- {rec}: missing"); continue
    fl = [(hour_of(f), f) for f in sorted(glob.glob(f'{dpath}/*.csv'))]
    fl = sorted([(h, f) for h, f in fl if h is not None])
    if not fl: continue
    print(f"\n  -- {rec} --")
    print(f"     {'h':>3} | {'a0':>10}{'a1':>10}{'a2':>10}"
          f" | {'R2_0':>7}{'R2_1':>7}"
          f" | {'S3 res0':>10}{'S3 res1':>10}"
          f" | {'gt dev0':>10}")
    for h, path in fl:
        d = read_chunked(path)
        a, b, r2 = fit_ab(d['jpos'], d['torque'])
        row = dict(record=rec, hour=h, n=len(d['jpos']))
        for j in range(N_POS):
            row[f'a{j}'] = float(a[j]); row[f'b{j}'] = float(b[j])
            row[f'r2{j}'] = float(r2[j])

        # [S2] difference within the redundant pair
        for k, (i, j) in enumerate(pairs):
            dif = d['mpos'][:, i] - d['mpos'][:, j]
            row[f'pair{k}_mean'] = float(dif.mean())
            row[f'pair{k}_std']  = float(dif.std())

        # [S3] motor_pos to jpos consistency: linear regression residual
        # If jpos comes from mpos through a fixed model, this stays near zero
        for j in range(N_POS):
            X = np.column_stack([d['mpos'], np.ones(len(d['mpos']))])
            y = d['jpos'][:, j]
            w, *_ = np.linalg.lstsq(X, y, rcond=None)
            res = y - X @ w
            row[f's3_res{j}'] = float(res.std())
            row[f's3_resmean{j}'] = float(res.mean())

        # Reference: true deviation from the external encoder (dataset only)
        dev = d['jpos'][:, :N_POS] - d['gt'][:, :N_POS]
        for j in range(N_POS):
            row[f'gtdev{j}'] = float(dev[:, j].mean())

        rows.append(row)
        print(f"     {h:>3} | {a[0]:>10.5f}{a[1]:>10.5f}{a[2]:>10.5f}"
              f" | {r2[0]:>7.4f}{r2[1]:>7.4f}"
              f" | {row['s3_res0']:>10.5f}{row['s3_res1']:>10.5f}"
              f" | {row['gtdev0']:>+10.4f}")
        del d; gc.collect()

df = pd.DataFrame(rows)
df.to_csv(f'{OUT}/slow_path_by_hour.csv', index=False)

# == [S1] Is the coefficient drift significant? ==
print("\n" + "=" * 100)
print(" [S1] Coefficient drift against fitting noise")
print("=" * 100)
print("  Bootstrap standard error of a_j, to judge drift against fitting uncertainty")
d_first = read_chunked(files0[0][1])
se_a = fit_ab_boot(d_first['jpos'], d_first['torque'])
print(f"\n  bootstrap SE of a_j (h0, {N_BOOT} resamples): {se_a}")
del d_first; gc.collect()

s1_detectable = {}
for rec in df.record.unique():
    s = df[df.record == rec].sort_values('hour')
    if len(s) < 3: continue
    print(f"\n  {rec}")
    for j in range(N_POS):
        v = s[f'a{j}'].values
        rho, p = spearmanr(s.hour.values, v)
        span = v[-1] - v[0]
        n_se = abs(span) / max(se_a[j], 1e-15)
        sig = (p < 0.05 and n_se > 3)
        print(f"    a{j}: {v[0]:>11.6f} -> {v[-1]:>11.6f}   "
              f"span={span:>+11.6f} = {n_se:>6.1f} SE   "
              f"rho={rho:>+.2f} p={p:.3f}   {'<-- significant drift' if sig else ''}")
        if rec.endswith('500gload'):
            s1_detectable[j] = sig

# == [S2][S3] Trends ==
for tag, keys, title in [
    ('S2', [f'pair{k}_mean' for k in range(len(pairs))],
     'paired-motor redundancy difference'),
    ('S3', [f's3_res{j}' for j in range(N_POS)] +
           [f's3_resmean{j}' for j in range(N_POS)],
     'motor_pos to jpos consistency residual'),
]:
    if not keys: continue
    print("\n" + "=" * 100)
    print(f" [{tag}] {title}")
    print("=" * 100)
    for rec in df.record.unique():
        s = df[df.record == rec].sort_values('hour')
        if len(s) < 3: continue
        print(f"\n  {rec}")
        for k in keys:
            if k not in s: continue
            v = s[k].values
            if np.allclose(v, v[0], atol=1e-12):
                print(f"    {k:<16} constant ({v[0]:.6g})   no information"); continue
            rho, p = spearmanr(s.hour.values, v)
            rel = (v[-1]-v[0]) / max(abs(v[0]), 1e-15)
            print(f"    {k:<16} {v[0]:>12.6g} -> {v[-1]:>12.6g}   "
                  f"relative {rel:>+8.1%}  rho={rho:>+.2f} p={p:.3f}"
                  f"{'  <-- significant' if p < 0.05 and abs(rel) > 0.05 else ''}")

# == Conclusion ==
print("\n" + "=" * 100)
print(" Slow-path observability: conclusion")
print("=" * 100)
load = df[df.record.str.endswith('500gload')].sort_values('hour')
idle = df[df.record.str.endswith('idle')].sort_values('hour')

cands = []
if len(load) >= 3:
    for j in range(N_POS):
        v = load[f'a{j}'].values
        rho, p = spearmanr(load.hour.values, v)
        n_se = abs(v[-1]-v[0]) / max(se_a[j], 1e-15)
        if p < 0.05 and n_se > 3:
            cands.append((f'S1: coefficient drift in a{j}', n_se, p))
    for k in range(len(pairs)):
        col = f'pair{k}_mean'
        if col in load:
            v = load[col].values
            if not np.allclose(v, v[0], atol=1e-12):
                rho, p = spearmanr(load.hour.values, v)
                if p < 0.05: cands.append((f'S2: redundant pair {pairs[k]}', np.nan, p))
    for j in range(N_POS):
        col = f's3_resmean{j}'
        if col in load:
            v = load[col].values
            if not np.allclose(v, v[0], atol=1e-12):
                rho, p = spearmanr(load.hour.values, v)
                if p < 0.05: cands.append((f'S3: jpos-mpos residual j{j}', np.nan, p))

if cands:
    print(f"\n  {len(cands)} candidate slow-path observable(s):")
    for name, n_se, p in cands:
        se_str = f'{n_se:.1f} SE' if np.isfinite(n_se) else '-'
        print(f"    {name:<30} {se_str:>10}  p={p:.4f}")
    print(f"\n  -> drift IS observable in session-scale telemetry.")
    print(f"     The threat model can then cover two timescales:")
    print(f"       fast path (0.23 s window): injection, floors 0.019 to 0.174 deg")
    print(f"       slow path (hours):         cable degradation and slow injection hiding in it")
    print(f"     The earlier claim of invisibility should narrow to:")
    print(f"       the window-scale residual under frozen coefficients is insensitive to drift")
else:
    print(f"\n  No significant slow-path observable found.")
    print(f"  Most likely: robot_jpos is computed from motor_pos through a fixed model,")
    print(f"  so their consistency is constructed and carries no drift information.")
    print(f"  If so, drift really is unobservable without an external reference,")
    print(f"  and the earlier conclusion holds, but for this constructive reason.")
    print(f"  Drift still belongs in the threat model, because it defines the envelope")
    print(f"  an adversary can hide in (about 0.8 deg, 6 mm), which the fast path cannot cover.")
    print(f"  A slow path then needs an external reference: homing, a known pose, or an encoder.")

# Control against the idle condition
if len(idle) >= 3 and len(load) >= 3:
    print(f"\n  Loaded against idle, to rule out thermal drift:")
    for j in range(N_POS):
        dl = load[f'a{j}'].values[-1] - load[f'a{j}'].values[0]
        di = idle[f'a{j}'].values[-1] - idle[f'a{j}'].values[0]
        print(f"    a{j}: loaded span={dl:>+11.6f}  idle span={di:>+11.6f}   "
              f"ratio={dl/di if abs(di)>1e-15 else float('nan'):>8.2f}")
    print(f"    A ratio well above 1 means the drift is load-driven, not thermal")

json.dump(dict(pairs=pairs, se_a=se_a.tolist(),
               candidates=[[c[0], None if not np.isfinite(c[1]) else c[1], c[2]]
                           for c in cands],
               window_impossibility=dict(
                   window_s=win_s, drift_rate_deg_s=drift_rate,
                   per_window_deg=per_win,
                   fast_floor_deg=FAST_FLOOR_RAMP_DEG,
                   ratio=FAST_FLOOR_RAMP_DEG/per_win)),
          open(f'{OUT}/summary.json', 'w'), indent=2)

# == Figures ==
fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
colors = dict(zip(RECORDS, ['#94A3B8', '#F59E0B', '#DC2626']))
panels = [(f'a{j}', f'fitted $a_{j}$ (S1)') for j in range(N_POS)]
panels += [('s3_res0', 'jpos-mpos residual std (S3)'),
           ('gtdev0', 'joint 0 error vs external encoder\n(NOT observable at deploy)')]
if pairs:
    panels.insert(4, (f'pair0_mean', f'motor pair {pairs[0]} diff (S2)'))
for ax, (key, lab) in zip(axes.flatten(), panels[:6]):
    for rec in df.record.unique():
        s = df[df.record == rec].sort_values('hour')
        if key not in s: continue
        ax.plot(s.hour, s[key], 'o-', color=colors.get(rec, 'k'), lw=2,
                markersize=6, label=rec.replace('record_3_time_decay_', ''))
    ax.set_xlabel('hour'); ax.set_ylabel(lab, fontsize=9)
    ax.set_title(lab, fontsize=10, fontweight='bold')
    ax.legend(fontsize=6); ax.grid(alpha=.3)
fig.suptitle('Slow-path observability of cable drift. Bottom-right is the '
             'ground truth the deployed monitor cannot see.',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/slow_path.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  output: {OUT}/slow_path_by_hour.csv")
print(f"        {OUT}/summary.json")
print(f"        {OUT}/slow_path.png")
print("=" * 100)
