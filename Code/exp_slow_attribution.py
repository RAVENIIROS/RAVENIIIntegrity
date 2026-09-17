
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ1 | Compensation-fidelity sweep, where attribution fails (rho = 0.90)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Slow-path attribution: can slope consistency separate cable degradation from
a malicious slow injection?

The manuscript claims that an adversary who edits jpos but not the paired
motor redundancy departs from the calibrated slope that real degradation
obeys, and that this makes attribution possible. That was a mechanism derived
from the threat model, never tested. The covert envelope depends on it:

    if attribution fails, 0.127 deg / 0.94 mm is only a bound on what goes
    undetected, not on what goes unattributed, and the argument needs
    rewriting.

The existing calibration (probe_canary_transfer.py, 500 g, pair (4,6)):
    the slope transfers across load conditions to within 6%, the offset does
    not; LOO calibration residual 0.042 deg, 3-sigma slow-path floor
    0.127 deg.

-- Three design decisions ----------------------------------------

  [D1] There must be an adversary who can beat the mechanism
       Testing only naive injection against real degradation is circular.
       Slope consistency assumes the adversary edits jpos and not motor_pos,
       so an informed one edits motor_pos along the calibrated slope. That
       attack has to be included, and as the headline result, because it
       gives the mechanism's real boundary.

       A0  none            no injection, real cable degradation only (control)
       A1  naive           jpos only, slow ramp (mechanism should catch it)
       A2  slope_aware     motor_pos moved by a as well, slope preserved
                           (mechanism should fail)
       A3  partial(rho)    motor_pos compensated by a fraction rho, swept
                           -> how precisely must the adversary know a?

  [D2] Evaluation has to be at session scale
       The slow-path observable is an hourly mean, so the sample unit is the
       hour, not the window. Six hourly points give very low statistical
       power, so uncertainty comes from a bootstrap and the n = 6 limit is
       reported as such.

  [D3] The calibration has to be held out
       Fit on some of the 500 g hours and evaluate on the rest. Otherwise the
       residual is in-sample, as in probe_canary_transfer, and underestimated.

Outputs:
  [1] the distribution of the slope-consistency statistic per attack type,
      and the resulting attribution AUC
  [2] the attribution floor for A1: how small an injection can still be
      attributed to something other than degradation
  [3] the compensation-accuracy curve for A3: how precisely the adversary
      must know a in order to evade
      -> this decides whether 0.94 mm holds, and under what assumption
  [4] the corrected covert envelope: if A2 or A3 evade attribution, the
      envelope is set by non-detection rather than non-attribution and has to
      be restated

Depends on: by_hour.csv written by probe_canary.py
Usage:      python3 exp_slow_attribution.py
Expected:   pure numpy, 1 to 3 minutes
"""

import os, sys, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = _OUT + '/output_slow_attribution'
IN  = _OUT + '/output_canary/by_hour.csv'
os.makedirs(OUT, exist_ok=True)

if not os.path.exists(IN):
    print(f"{IN} not found; run probe_canary.py first"); sys.exit(1)

df = pd.read_csv(IN)
LOAD = 'record_3_time_decay_500gload'
UNL  = 'record_3_time_decay_unloaded'
PAIR = 'p46'                      # best pair from probe_canary_transfer
MM_PER_DEG = 7.47                 # 5.973 mm / 0.800 deg
FAST_FLOOR_DEG = 0.174
FAST_RATE_LIMIT = 0.766           # deg/s, = 0.174/0.227
N_BOOT = 2000
RNG = np.random.RandomState(0)

L = df[df.record == LOAD].sort_values('hour').reset_index(drop=True)
U = df[df.record == UNL ].sort_values('hour').reset_index(drop=True)
print("=" * 100)
print(" Slow-path attribution: the slope-consistency test")
print("=" * 100)
print(f"  500 g: {len(L)} hourly points   unloaded: {len(U)} hourly points")
print(f"  observable: pair {PAIR}    target: joint-0 drift (gt0)")
if len(L) < 5:
    print("  fewer than 5 hourly points; cannot hold out a calibration"); sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  Attack injection, applied to the session-level (d, q0) sequences
# ══════════════════════════════════════════════════════════════
def inject(d, q0, kind, mag_deg, a_true, rho=1.0):
    """
    d      : (H,) hourly mean of the paired-motor difference (editable if informed)
    q0     : (H,) hourly mean of the reported joint-0 position error (the target)
    kind   : 'none' | 'naive' | 'slope_aware' | 'partial'
    mag_deg: joint-0 offset accumulated by the last hourly point, in degrees
    a_true : the calibrated slope, used by an informed adversary to compensate
    rho    : fraction of motor_pos compensation (0 = naive, 1 = slope_aware)
    Returns (d_obs, q0_obs)
    """
    H = len(q0)
    ramp = np.linspace(0.0, mag_deg, H)      # linear slow injection, session scale
    q0_obs = q0 + ramp
    if kind == 'none':
        return d.copy(), q0.copy()
    if kind == 'naive':
        return d.copy(), q0_obs
    if kind in ('slope_aware', 'partial'):
        r = 1.0 if kind == 'slope_aware' else rho
        # To keep (d_obs, q0_obs) on q0 = a*d + b, needs delta_d = delta_q0 / a
        d_obs = d + r * (ramp / a_true)
        return d_obs, q0_obs
    raise ValueError(kind)


def fit_calib(d, q0):
    A = np.polyfit(d, q0, 1)
    return float(A[0]), float(A[1])


def consistency_stat(d, q0, a, b):
    """
    Slope-consistency statistic: RMS residual about the frozen calibration line.
    Real degradation obeys that line; an injection that moves only q0 departs.
    """
    res = q0 - (a * d + b)
    return float(np.sqrt((res ** 2).mean())), res


# ══════════════════════════════════════════════════════════════
#  [D3] Held-out calibration
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print(" [D3] Held-out calibration (fit on the first half of the hours)")
print("=" * 100)
H = len(L)
n_fit = max(3, H // 2)
d_all, q_all = L[PAIR].values.astype(float), L['gt0'].values.astype(float)
d_fit, q_fit = d_all[:n_fit], q_all[:n_fit]
d_ev,  q_ev  = d_all[n_fit:], q_all[n_fit:]
a_hat, b_hat = fit_calib(d_fit, q_fit)
a_full, b_full = fit_calib(d_all, q_all)
print(f"\n  fit on h0-h{n_fit-1}, evaluate on h{n_fit}-h{H-1}")
print(f"  held-out:  a = {a_hat:.6f}   b = {b_hat:.4f}")
print(f"  full fit:  a = {a_full:.6f}   b = {b_full:.4f}   "
      f"(reference: probe_canary_transfer reports -0.006718 / -3.8031)")
rms_ev, res_ev = consistency_stat(d_ev, q_ev, a_hat, b_hat)
rms_fit, _ = consistency_stat(d_fit, q_fit, a_hat, b_hat)
print(f"\n  consistency statistic (RMS residual):")
print(f"    fit span,  no injection   {rms_fit:.5f} deg")
print(f"    eval span, no injection   {rms_ev:.5f} deg   <- attribution noise floor")
print(f"  Reading: the eval RMS centres the null distribution, pure degradation.")
print(f"        To be attributed, an injection must push the statistic outside it.")

# Bootstrap the null distribution by resampling the eval-span residuals
null = []
for _ in range(N_BOOT):
    idx = RNG.randint(0, len(res_ev), len(res_ev))
    null.append(float(np.sqrt((res_ev[idx] ** 2).mean())))
null = np.array(null)
thr95 = float(np.percentile(null, 95))
print(f"\n  null distribution (bootstrap, n={len(res_ev)} points, {N_BOOT} draws):")
print(f"    median {np.median(null):.5f}   95th pct {thr95:.5f} deg")
print(f"  NOTE: the eval span holds only {len(res_ev)} hourly points, so power is")
print(f"    very low. Every attribution floor below is an order-of-magnitude estimate.")


# ══════════════════════════════════════════════════════════════
#  [1] The consistency statistic per attack type
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print(" [1] Consistency statistic by attack type")
print("=" * 100)
MAGS = [0.05, 0.10, 0.127, 0.20, 0.40, 0.80]
print(f"\n  Magnitude is the joint-0 offset accumulated by the last hour, in degrees.")
print(f"  Slow-path detection floor 0.127 deg; natural 6 h degradation 0.800 deg.\n")
print(f"  {'kind':<14}{'mag(deg)':>10}{'consist RMS':>14}{'/null 95%':>14}{'attrib?':>9}")
print("-" * 100)
rows = []
for kind in ['none', 'naive', 'slope_aware']:
    mags = [0.0] if kind == 'none' else MAGS
    for m in mags:
        d_o, q_o = inject(d_ev, q_ev, kind, m, a_hat)
        rms, _ = consistency_stat(d_o, q_o, a_hat, b_hat)
        flag = 'yes' if rms > thr95 else 'no'
        rows.append(dict(kind=kind, mag=m, rms=rms, ratio=rms / thr95,
                         attributed=(rms > thr95)))
        print(f"  {kind:<14}{m:>10.3f}{rms:>14.5f}{rms/thr95:>14.2f}{flag:>8}")

# Attribution AUC, naive against none, per magnitude
print(f"\n  attribution AUC (naive injection vs pure degradation, bootstrapped):")
print(f"  {'mag(deg)':>10}{'AUC':>10}{'tip(mm)':>10}")
print("-" * 100)
auc_naive = {}
for m in MAGS:
    d_o, q_o = inject(d_ev, q_ev, 'naive', m, a_hat)
    _, res_atk = consistency_stat(d_o, q_o, a_hat, b_hat)
    alt = []
    for _ in range(N_BOOT):
        idx = RNG.randint(0, len(res_atk), len(res_atk))
        alt.append(float(np.sqrt((res_atk[idx] ** 2).mean())))
    alt = np.array(alt)
    y = np.r_[np.zeros(len(null)), np.ones(len(alt))]
    sc = np.r_[null, alt]
    au = float(roc_auc_score(y, sc))
    auc_naive[m] = au
    print(f"  {m:>10.3f}{au:>10.4f}{m*MM_PER_DEG:>10.3f}")

# Attribution floor: smallest magnitude reaching AUC 0.9, log-interpolated
ms = np.array(MAGS); au = np.array([auc_naive[m] for m in MAGS])
floor_attr = np.nan
for k in range(len(ms) - 1):
    if (au[k] - 0.9) * (au[k+1] - 0.9) <= 0 and au[k] != au[k+1]:
        t = (0.9 - au[k]) / (au[k+1] - au[k])
        floor_attr = float(np.exp(np.log(ms[k]) + t * (np.log(ms[k+1]) - np.log(ms[k]))))
        break
if not np.isfinite(floor_attr):
    floor_attr = float(ms[0]) if au[0] >= 0.9 else np.inf

print(f"\n  [2] attribution floor for naive injection (AUC>=0.9): "
      f"{floor_attr:.4f} deg = {floor_attr*MM_PER_DEG:.3f} mm"
      if np.isfinite(floor_attr) else
      f"\n  [2] attribution floor for naive injection: not reached on this grid")


# ══════════════════════════════════════════════════════════════
#  [3] A3: how accurately must the adversary know a in order to evade?
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print(" [3] Required compensation accuracy (decides whether 0.94 mm holds)")
print("=" * 100)
print("\n  The adversary compensates motor_pos by a fraction rho: 0 naive, 1 fully informed.")
print("  If a small rho already evades, the mechanism is fragile and 0.94 mm cannot")
print("  be claimed as a bound on what goes unattributed.\n")
RHOS = [0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
TEST_MAG = 0.80            # injection at the natural degradation scale
print(f"  magnitude fixed at {TEST_MAG} deg ({TEST_MAG*MM_PER_DEG:.2f} mm)\n")
print(f"  {'rho':>8}{'consist RMS':>14}{'/null 95%':>14}{'attrib?':>9}{'AUC':>10}")
print("-" * 100)
rho_rows = []
for r in RHOS:
    d_o, q_o = inject(d_ev, q_ev, 'partial', TEST_MAG, a_hat, rho=r)
    rms, res_a = consistency_stat(d_o, q_o, a_hat, b_hat)
    alt = []
    for _ in range(N_BOOT):
        idx = RNG.randint(0, len(res_a), len(res_a))
        alt.append(float(np.sqrt((res_a[idx] ** 2).mean())))
    alt = np.array(alt)
    y = np.r_[np.zeros(len(null)), np.ones(len(alt))]
    au = float(roc_auc_score(y, np.r_[null, alt]))
    flag = 'yes' if rms > thr95 else 'no'
    rho_rows.append(dict(rho=r, rms=rms, ratio=rms/thr95, auc=au,
                         attributed=(rms > thr95)))
    print(f"  {r:>8.2f}{rms:>14.5f}{rms/thr95:>14.2f}{flag:>8}{au:>10.4f}")

# Find the smallest rho that evades
evade = [x['rho'] for x in rho_rows if x['auc'] < 0.9]
rho_needed = min(evade) if evade else np.nan
print(f"\n  smallest rho that evades attribution (AUC<0.9) = "
      f"{rho_needed:.2f}" if np.isfinite(rho_needed) else
      f"\n  no rho on this grid evades attribution")
if np.isfinite(rho_needed):
    err_tol = 1.0 - rho_needed
    print(f"  Equivalently: the adversary's estimate of a must be within "
          f"{err_tol*100:.0f}%")
    print(f"  For scale: a varies by about 6% between the two load conditions")
    if err_tol >= 0.066:
        print(f"  -> knowing a to within {err_tol*100:.0f}% suffices to evade,")
        print(f"     which is looser than the quantity's own spread across conditions,")
        print(f"     and is obtainable from a public dataset or by calibrating.")
        print(f"     Conclusion: slope-consistency attribution is fragile.")
    else:
        print(f"  -> evading needs better accuracy than a's own natural spread,")
        print(f"     so the mechanism has real strength.")


# ══════════════════════════════════════════════════════════════
#  [4] The covert envelope, restated
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 100)
print(" [4] The covert envelope, restated")
print("=" * 100)
print(f"\n  fast-path rate threshold   r < {FAST_RATE_LIMIT:.3f} deg/s")
print(f"  slow-path detection floor  0.127 deg = {0.127*MM_PER_DEG:.2f} mm")
print(f"  the manuscript currently claims 0.94 mm bounds what goes unattributed")
if np.isfinite(rho_needed) and (1.0 - rho_needed) >= 0.066:
    print(f"\n  This experiment shows {(1-rho_needed)*100:.0f}% compensation accuracy evades it.")
    print(f"     So 0.94 mm bounds only what goes undetected.")
    print(f"     An informed adversary can inject any magnitude while keeping the")
    print(f"     slope consistent, bounded by detection rather than attribution,")
    print(f"     and a resulting alarm reads as a call for recalibration.")
    print(f"\n  The manuscript must be rewritten as:")
    print(f"     (a) 0.94 mm bounds what goes UNDETECTED")
    print(f"     (b) slope-consistency attribution fails against an informed adversary")
    print(f"     (c) attribution needs a second observable the adversary cannot forge,")
    print(f"         such as transmission-spanning encoders (Yang et al.)")
else:
    print(f"\n  The mechanism is supported: naive injection is attributable above "
          f"{floor_attr*MM_PER_DEG:.2f} mm, "
          f"and an informed adversary needs very high compensation accuracy.")
    print(f"  The claim that 0.94 mm bounds what goes unattributed can stand.")

json.dump(dict(calib=dict(a_holdout=a_hat, b_holdout=b_hat,
                          a_full=a_full, b_full=b_full,
                          n_fit=n_fit, n_eval=len(d_ev)),
               null=dict(median=float(np.median(null)), p95=thr95,
                         n_points=len(res_ev)),
               attack_rows=rows,
               auc_naive={str(k): v for k, v in auc_naive.items()},
               floor_attr_deg=(None if not np.isfinite(floor_attr) else floor_attr),
               rho_rows=rho_rows,
               rho_needed=(None if not np.isfinite(rho_needed) else rho_needed)),
          open(f'{OUT}/results.json', 'w'), indent=2)

# == Figures ==
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
a = ax[0]
a.axhline(thr95, color='red', ls='--', lw=2, label='null 95th pct')
a.axhspan(np.percentile(null,5), np.percentile(null,95), color='gray',
          alpha=.25, label='null (pure degradation)')
for kind, c, mk in [('naive','#DC2626','o'), ('slope_aware','#2563EB','s')]:
    sub = [r for r in rows if r['kind'] == kind]
    a.plot([r['mag'] for r in sub], [r['rms'] for r in sub], mk+'-',
           color=c, lw=2, markersize=7, label=kind)
a.axvline(0.127, color='green', ls=':', lw=1.6, label='slow-path floor')
a.axvline(0.800, color='purple', ls=':', lw=1.6, label='natural 6h drift')
a.set_xscale('log'); a.set_yscale('log')
a.set_xlabel('injected joint-0 deviation (deg)')
a.set_ylabel('slope-consistency RMS (deg)')
a.set_title('does the injection depart\nfrom the calibrated slope?',
            fontsize=10, fontweight='bold')
a.legend(fontsize=6); a.grid(alpha=.3)

a = ax[1]
a.plot(MAGS, [auc_naive[m] for m in MAGS], 'o-', color='#DC2626', lw=2,
       markersize=7, label='naive injection')
a.axhline(0.9, color='gray', ls=':', lw=1.5)
a.axhline(0.5, color='black', ls=':', alpha=.4)
if np.isfinite(floor_attr):
    a.axvline(floor_attr, color='green', ls='--', lw=1.8,
              label=f'attribution floor {floor_attr:.3f}$^\\circ$')
a.set_xscale('log'); a.set_ylim(0.4, 1.02)
a.set_xlabel('injected joint-0 deviation (deg)')
a.set_ylabel('attribution AUC')
a.set_title('attribution of naive slow injection', fontsize=10, fontweight='bold')
a.legend(fontsize=7); a.grid(alpha=.3)

a = ax[2]
a.plot([r['rho'] for r in rho_rows], [r['auc'] for r in rho_rows], 'o-',
       color='#7C3AED', lw=2, markersize=7)
a.axhline(0.9, color='gray', ls=':', lw=1.5, label='AUC=0.9')
a.axhline(0.5, color='black', ls=':', alpha=.4)
a.axvline(1-0.066, color='orange', ls='--', lw=1.8,
          label="a's cross-condition spread (6.6%)")
if np.isfinite(rho_needed):
    a.axvline(rho_needed, color='red', ls='-', lw=1.8,
              label=f'evasion at $\\rho$={rho_needed:.2f}')
a.set_xlabel(r'adversary compensation fidelity $\rho$')
a.set_ylabel('attribution AUC')
a.set_ylim(0.4, 1.02)
a.set_title(f'informed adversary\n(injection fixed at {TEST_MAG}$^\\circ$)',
            fontsize=10, fontweight='bold')
a.legend(fontsize=6); a.grid(alpha=.3)

fig.suptitle('Slow-path attribution: is slope consistency robust to an '
             'informed adversary?', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUT}/attribution.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  output: {OUT}/results.json")
print(f"        {OUT}/attribution.png")
print("=" * 100)
