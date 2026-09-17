# Paper Numbers and Their Scripts

Every number in the paper can be traced here. Ordered by section.

---

## Methodology

| Figure | What it is | Script |
|---|---|---|
| per-frame displacement SD 0.17 deg | joint 1, clean training segment, after subsampling | `peng_uw_loader.py` + `delta_ablation.py` |
| physics residual SD 0.11 deg | SD of `dq - (a*tau + b)`, R2 = 0.56 | same |
| rate threshold 0.766 deg/s | `l / T_w` with l = 0.174 deg, T_w = 0.227 s | `exp_slow_attribution.py` (`FAST_RATE_LIMIT`) |
| natural degradation 3.7e-5 deg/s | 0.800 deg over six hours | `probe_drift.py` |
| 8e-6 deg per window | the row above times 0.227 s | same |
| lag 4 / 4 / 6 frames | cross-correlation peak, per joint | `exp_tracking_error.py` |
| residual reduced 54.5 / 57.1 / 69.7 % | SD before and after lag alignment | same |
| ARX held-out R2 = 1.000000 | n_a = 6, n_b = 4 | same |
| DC gains 1.012 / 0.997 / 1.001 | steady-state gain of the identified model | same |
| settling 12 / 6 / 17 frames | step response to 90 % of final value | same |
| clean validation windows 8,731 | held-out sessions | `exp_operating_point.py` |
| 15,850 windows per hour | 227 ms, non-overlapping | arithmetic |

**Leakage controls** are in `diag_leakage.py`: null injection returns AUC
0.503 / 0.506 / 0.506, and channel sanitisation and label shuffling also
return chance.

---

## RQ1: the session scale

| Figure | What it is | Script |
|---|---|---|
| 0.800 deg | six hours at 500 g, reported minus encoder, uncompensated | `probe_drift.py` |
| 5.97 mm | forward kinematics on both joint vectors, then differenced, session mean | same |
| 113.7 encoder units | six-hour excursion of the motor-pair difference | `probe_slow_path.py` |
| within-hour SD 3.6 | the control for the row above | same |
| effect size 31x | ratio of the two | same |
| monotone, rho = +/-1.00 | three of four motors in the group | same |
| load-driven 3.9 to 6.4x | 500 g against the idle control | same |
| slope transfer 6 % | motor pair (4,6), fitted at 500 g, applied to unloaded | `probe_canary_transfer.py` |
| other three pairs 23 to 62 % | (4,5), (4,7), (5,6) | same |
| offset discrepancy 0.121 deg | same transfer | same |
| leave-one-out residual 0.042 deg | pair (4,6), six hourly points | same |
| magnitude estimator 0.127 deg = 0.95 mm | three times the row above | same |
| consistency test 0.0865 deg = 0.65 mm | three sigma of the departure statistic | `exp_slow_attribution.py` |
| margin 6.3x | 0.800 / 0.127 | arithmetic |
| statistic flat at 0.145 deg | slope-aware injection, 0.05 to 0.80 deg | `exp_slow_attribution.py` |
| attribution fails at rho = 0.90 | compensation-fidelity sweep | same |

**The motor pair was found by search, not derived.** All 28 pairs of the
eight motors were scanned for correlation above 0.999 with a difference SD
below 5 % of the individual SD; only a few qualify. Pair (4,6) was selected
on leave-one-out residual, so the 0.042 deg figure is the smallest of four
candidates rather than an unbiased estimate. The paper states this.

---

## RQ2: coverage

| Figure | What it is | Script |
|---|---|---|
| tracking error A = 2.85 mm | desired-pose injection | `exp_tracking_error.py` |
| tracking error B = 0.53 mm | torque injection | same |
| tracking error C = 1.35 mm | observation-path injection, not independent evidence | same |
| physics residual C = 0.81 mm | observation-path injection | `exp_residual_input.py` |
| physics model R2 = 0.56 / 0.76 / 0.81 | per-joint linear fit | `delta_ablation.py` |
| inertia term contributes 7e-6 | change in R2 when the term is dropped | `probe_dynamics_params.py` |

Injection A's trajectory comes from the ARX step response; injection B's
from a measured torque-to-displacement gain. Neither is recorded directly in
the data.

---

## RQ3: operating points

| Criterion | step | ramp | noise | Script |
|---|---|---|---|---|
| AUC >= 0.9 | 0.142 mm | 1.300 mm | 0.031 mm | `exp_operating_point.py` |
| any estimable alarm budget | none | none | none | same, after patching |

The second row is the corrected result. See the pre-submission check in
`README.md` Section 3 and the two scripts `exp_window_accounting.py` and
`patch_operating_point.py`. The original file is preserved as
`exp_operating_point.py.orig` after patching.

**Tracking-error operating points** are in `exp_tracking_operating.py`: at 20
alarms per hour the three injection points converge to 2.6 to 2.7 mm, where
AUC had separated them by a factor of 5.4.

---

## RQ4: the command stream

| Pattern | command stream | observation stream | Script |
|---|---|---|---|
| step | 0.0389 deg = 0.29 mm | 0.0190 deg | `exp_command_path.py` |
| ramp | none | 0.1740 deg | same |
| noise | none | 0.0041 deg | same |

A ramp's AUC saturates at 0.88 from 1.26 deg onward and gains 0.002 over the
next 3.5x in magnitude. Noise sits at 0.50 across a 40x range with no trend.
Neither failure is a grid artifact.

---

## Limitations and Discussion

| Figure | What it is | Script |
|---|---|---|
| 0.662 ms | p99, single thread, batch 1, RAVEN-II control host | `bench_latency.py` |
| 0.3 % of the control period | non-overlapping windows | arithmetic |
| per-joint encoder 0.170 -> 0.101 deg | single-joint injection, 60 % fewer parameters | `exp_perjoint_graph.py` |
| 35 mm mean displacement over the horizon | the data are calibration sweeps, not surgical gestures | `delta_ablation.py` |
| lead time 3 to 10 frames | an estimate, not measured | `exp_predictive.py` can measure it |

---

## Clinical scale (cited, not measured here)

| Figure | Source |
|---|---|
| margins below 1 mm, hazard ratio 2.01 (95 % CI 1.05 to 3.83) | Fowler et al., Head & Neck, 2022 |
| mean margin 2.70 +/- 2.44 mm | same |
| 24 % of specimens within 1 mm | same |
| SSO-ASTRO consensus | Moran et al., IJROBP 88(3), 2014 |

These are cited from the literature, not measured here.

---

## A correction made before submission: the alarm budget

An earlier version reported that step injection reaches 0.95 mm at two alarms
per procedure hour. The check found two counting errors:

| | Before | After |
|---|---|---|
| decision frequency | 15,850 per hour (non-overlapping) | deployment scores every frame, 30x more |
| clean samples | 8,731 (adjacent windows share 29 of 30 frames) | about 290 (thinned to non-overlapping) |
| strictest estimable budget | 1.8 per hour | about 38 per hour |

After the correction none of the three patterns has an operating point at any
estimable budget. `exp_window_accounting.py` quantifies this and
`patch_operating_point.py` applies the fix.

What the paper keeps is the AUC-based floors (step 0.142 mm, ramp 1.300 mm,
noise 0.031 mm) and the negative result under a fixed budget.

---

## One comparison deliberately not made

An earlier version claimed agreement with Peng et al. to within 7 %. Their
six-hour joint-1 drift under 500 g is reported as 0.296 / 0.741 / 0.772 /
0.824 degrees, one value per calibration method, and those are residuals
after compensation. Our 0.800 degrees is the **uncompensated** divergence
between the reported position and the encoder.

The two measure different quantities, so the percentage was removed from the
paper and replaced by a statement that they are of the same order. Do not try
to reproduce a closer match.
