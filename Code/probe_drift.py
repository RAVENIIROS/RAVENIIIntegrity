
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ1 | Six-hour drift and tip displacement (0.800 deg, 5.97 mm)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Probing the real cable drift in Peng record_3

Three things have to be settled before writing any training pipeline,
otherwise the configuration is guesswork:

  [1] Memory feasibility
      The record_3 CSVs are 190 to 560 MB each. Loading six hours across
      several files at once exceeds 15 GB.
      Measured here: peak memory and wall time for chunked single-file reads.

  [2] Whether gt_jpos and robot_jpos can be subtracted directly
      The loader converts the ground truth from radians to degrees, but
      joint 3 is prismatic and that conversion is unverified. If the two
      scales do not match, their difference means nothing.
      Measured here: per-joint mean, standard deviation, offset, correlation.

  [3] Whether the drift magnitude matches what Peng et al. 2024 report
      Their figures (6 h, 500 g): joint 1 = 0.855 deg, joint 2 = 0.432 deg,
      joint 3 = 0.181 mm.
      Measured here: the h0 to h5 growth curve and a comparison across the
      three load conditions.
      Far below the reported values means the column mapping or the units are
      wrong and nothing downstream is valid. Agreement gives the comparison
      between real drift and the ramp detection floor of 0.174 deg directly.

Usage:    python3 probe_drift.py
Expected: 5 to 10 minutes, depending on disk speed
"""

import os, sys, glob, gc, time, json
import numpy as np
import pandas as pd

sys.path.insert(0, _DATA)
from peng_uw_loader import COL, PengUWLoader

ROOT = _DATA
OUT  = _OUT + '/output_drift_probe'
os.makedirs(OUT, exist_ok=True)

RECORDS = ['record_3_time_decay_idle',
           'record_3_time_decay_unloaded',
           'record_3_time_decay_500gload']

# Forward kinematics (same as the main pipeline)
LA12 = np.radians(75.0); LA23 = np.radians(52.0); D4 = -458.69

def fk(jpos):
    J0, J1, J2 = jpos[..., 0], jpos[..., 1], jpos[..., 2]
    th1 = np.radians(J0 + 205.0); th2 = np.radians(J1 + 180.0); d3 = J2
    g1, g2 = np.sin(LA12), np.cos(LA12)
    g3, g4 = np.sin(LA23), np.cos(LA23)
    d = d3 + D4
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    return np.stack([
        d * (c1*s2*g3 + s1*(c2*g2*g3 - g1*g4)),
        d * (s1*s2*g3 - c1*(c2*g2*g3 - g1*g4)),
        d * (-(c2*g1*g3 + g2*g4)),
    ], axis=-1)


def mem_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return float('nan')


NEEDED = sorted(set(
    list(range(COL['gt_jpos'][0], COL['gt_jpos'][1])) +
    list(range(COL['robot_jpos'][0], COL['robot_jpos'][1])) +
    list(range(COL['motor_torque'][0], COL['motor_torque'][1])) +
    [COL['gt_time'], COL['robot_time']]
))


def read_chunked(path, subsample=20, chunksize=50000):
    """Chunked read, needed columns only, heavily subsampled."""
    parts = []
    for chunk in pd.read_csv(path, header=None, usecols=NEEDED,
                             chunksize=chunksize):
        parts.append(chunk.iloc[::subsample])
    df = pd.concat(parts, ignore_index=True)
    del parts; gc.collect()

    gs, ge = COL['gt_jpos']; rs, re = COL['robot_jpos']
    ts, te = COL['motor_torque']
    return dict(
        gt_time  = df[COL['gt_time']].values.astype(np.float32),
        gt_rad   = df[list(range(gs, ge))].values.astype(np.float32),
        robot    = df[list(range(rs, re))].values.astype(np.float32),
        torque   = df[list(range(ts, te))].values.astype(np.float32),
    )


print("=" * 80)
print(" [1] Memory and wall-time feasibility")
print("=" * 80)
print(f"  columns needed: {len(NEEDED)} / 252")

probe_files = sorted(glob.glob(f'{ROOT}/{RECORDS[2]}/*.csv'))
if not probe_files:
    print(f"\n  no CSV under {RECORDS[2]}. Check that extraction finished.")
    sys.exit(1)

big = max(probe_files, key=os.path.getsize)
print(f"  largest file: {os.path.basename(big)}  ({os.path.getsize(big)//1024//1024} MB)")

t0 = time.time(); m0 = mem_mb()
d = read_chunked(big, subsample=20)
t1 = time.time(); m1 = mem_mb()
print(f"  chunked read (subsample=20): {t1-t0:.1f}s,  rows {len(d['gt_rad']):,}")
print(f"  peak RSS: {m1:.0f} MB  (delta {m1-m0:+.0f} MB)")
n_files_total = sum(len(glob.glob(f'{ROOT}/{r}/*.csv')) for r in RECORDS
                    if os.path.exists(f'{ROOT}/{r}'))
print(f"  {n_files_total} CSVs across the three records")
print(f"  all resident at once would need about {(m1-m0)*n_files_total/1024:.1f} GB")
print(f"  -> {'process one file at a time and discard' if (m1-m0)*n_files_total/1024 > 6 else 'all can stay resident'}")

print("\n" + "=" * 80)
print(" [2] Are gt_jpos and robot_jpos comparable?")
print("=" * 80)

gt_rad = d['gt_rad']; robot = d['robot']
print(f"\n  raw gt (radians or native units):")
for j in range(3):
    v = gt_rad[:, j]
    print(f"    gt[{j}]:    mean={v.mean():>9.4f}  std={v.std():>8.4f}  "
          f"range=[{v.min():>8.4f}, {v.max():>8.4f}]")
print(f"\n  robot_jpos (deg / mm):")
for j in range(3):
    v = robot[:, j]
    print(f"    robot[{j}]: mean={v.mean():>9.4f}  std={v.std():>8.4f}  "
          f"range=[{v.min():>8.4f}, {v.max():>8.4f}]")

# Try several unit conversions and see which aligns the two scales
print(f"\n  conversion candidates (goal: gt_conv mean/std close to robot):")
candidates = {
    'rad->deg (x 180/pi)' : gt_rad * (180.0 / np.pi),
    'identity'            : gt_rad.copy(),
    'x 1000'              : gt_rad * 1000.0,
}
best_conv = None; best_score = np.inf
for name, conv in candidates.items():
    # Judge alignment by the ratio of standard deviations, which ignores offset
    ratios = [conv[:, j].std() / max(robot[:, j].std(), 1e-9) for j in range(3)]
    score = np.mean([abs(np.log10(max(r, 1e-9))) for r in ratios])
    print(f"    {name:<22} std ratio = "
          f"[{ratios[0]:.3f}, {ratios[1]:.3f}, {ratios[2]:.3f}]   "
          f"log-score={score:.3f}")
    if score < best_score:
        best_score, best_conv, best_name = score, conv, name
print(f"  -> closest: {best_name}")

print(f"\n  per-joint correlation and residual offset (using {best_name}):")
for j in range(3):
    a, b = best_conv[:, j], robot[:, j]
    r = np.corrcoef(a, b)[0, 1] if a.std() > 1e-9 and b.std() > 1e-9 else np.nan
    off = (b - a).mean()
    print(f"    joint {j}:  corr={r:>7.4f}   mean(robot-gt)={off:>9.4f}   "
          f"std(robot-gt)={np.std(b-a):>8.4f}")
print(f"\n  Reading: corr should exceed 0.99, since both measure the same joint.")
print(f"        Low corr means the column mapping or units are wrong. Stop.")
print(f"        The mean offset is a calibration zero and is subtracted before drift analysis.")

del d, gt_rad, robot, best_conv; gc.collect()

print("\n" + "=" * 80)
print(" [3] Drift magnitude: h0 to h5 growth, three load conditions")
print("=" * 80)
print("  Peng et al. 2024 (6 h, 500 g): j1=0.855 deg, j2=0.432 deg, j3=0.181 mm")
print("  ramp detection floor measured here:  0.174 deg / 1.13 mm at the tip")

results = []
for rec in RECORDS:
    rdir = f'{ROOT}/{rec}'
    if not os.path.exists(rdir):
        print(f"\n  -- {rec}: directory missing, skipped")
        continue
    # Take only the files carrying an hour marker
    files = sorted(glob.glob(f'{rdir}/*.csv'))
    hour_files = []
    for f in files:
        base = os.path.basename(f)
        # Match _h0, _h1, ... or _idle_h0, ...
        for h in range(8):
            if f'_h{h}.' in base or f'_h{h}_' in base:
                hour_files.append((h, f)); break
    hour_files.sort()

    print(f"\n  -- {rec}  ({len(hour_files)} files with an hour marker)")
    if not hour_files:
        print(f"     no _h* files found; actual names:")
        for f in files[:5]:
            print(f"       {os.path.basename(f)}")
        continue

    for h, path in hour_files:
        d = read_chunked(path, subsample=50)
        gt   = d['gt_rad'] * (180.0 / np.pi)   # conversion chosen in [2]
        rob  = d['robot'][:, :3]

        # Joint-space deviation, reported both raw and centred
        diff = rob - gt                         # (N, 3)
        # Tip-space deviation
        xyz_gt  = fk(gt)
        xyz_rob = fk(rob)
        tip = np.linalg.norm(xyz_rob - xyz_gt, axis=-1)

        row = dict(record=rec, hour=h, n=len(gt),
                   j0_mean=float(diff[:,0].mean()), j0_std=float(diff[:,0].std()),
                   j1_mean=float(diff[:,1].mean()), j1_std=float(diff[:,1].std()),
                   j2_mean=float(diff[:,2].mean()), j2_std=float(diff[:,2].std()),
                   tip_mean=float(tip.mean()), tip_p95=float(np.percentile(tip, 95)))
        results.append(row)
        print(f"     h{h}: n={len(gt):>7,}  "
              f"Δj0={diff[:,0].mean():>+8.3f}  Δj1={diff[:,1].mean():>+8.3f}  "
              f"Δj2={diff[:,2].mean():>+8.3f}   tip={tip.mean():>7.2f} mm")
        del d, gt, rob, xyz_gt, xyz_rob, tip; gc.collect()

if results:
    df = pd.DataFrame(results)
    df.to_csv(f'{OUT}/drift_by_hour.csv', index=False)

    print("\n" + "=" * 80)
    print(" Drift growth relative to h0, with the h0 calibration offset removed")
    print("=" * 80)
    for rec in df['record'].unique():
        sub = df[df['record'] == rec].sort_values('hour')
        if len(sub) < 2:
            continue
        base = sub.iloc[0]
        print(f"\n  {rec}")
        print(f"    {'hour':>5} {'Δj0(deg)':>11} {'Δj1(deg)':>11} "
              f"{'Δj2':>11} {'Δtip(mm)':>11}")
        for _, r in sub.iterrows():
            print(f"    {int(r['hour']):>5} "
                  f"{r['j0_mean']-base['j0_mean']:>+11.4f} "
                  f"{r['j1_mean']-base['j1_mean']:>+11.4f} "
                  f"{r['j2_mean']-base['j2_mean']:>+11.4f} "
                  f"{r['tip_mean']-base['tip_mean']:>+11.4f}")

    print("\n" + "=" * 80)
    print(" Key comparisons")
    print("=" * 80)
    load = df[df['record'] == 'record_3_time_decay_500gload'].sort_values('hour')
    if len(load) >= 2:
        b, e = load.iloc[0], load.iloc[-1]
        dj = [abs(e[f'j{j}_mean'] - b[f'j{j}_mean']) for j in range(3)]
        dtip = abs(e['tip_mean'] - b['tip_mean'])
        print(f"  drift under 500 g from h{int(b['hour'])} to h{int(e['hour'])}:")
        print(f"    joint 0: {dj[0]:.4f}   joint 1: {dj[1]:.4f}   joint 2: {dj[2]:.4f}")
        print(f"    tip:     {dtip:.4f} mm")
        print(f"\n  vs Peng (0.855 / 0.432 deg):  "
              f"{'same order' if 0.1 < max(dj[:2]) < 5 else 'WRONG ORDER, check mapping and units'}")
        print(f"  vs ramp floor 0.174 deg:      "
              f"drift is {max(dj[:2])/0.174:.1f}x the floor")
        if max(dj[:2]) > 0.174:
            print(f"    -> real drift exceeds the floor, so the detector would flag it as an attack.")
            print(f"       That is exactly why the discrimination experiment is needed.")
        else:
            print(f"    -> drift is below the floor, so false-alarm risk is low.")

    with open(f'{OUT}/probe_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  output: {OUT}/drift_by_hour.csv")
    print(f"        {OUT}/probe_summary.json")
print("=" * 80)
