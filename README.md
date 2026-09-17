# Detection Limits of Malicious Injection in a Cable-Driven Surgical Robot

Code and reproduction steps for the accompanying paper.

The paper measures how small an injection a telemetry-only monitor can catch
on RAVEN-II, reporting floors in millimetres of tip deviation rather than
detection rates at some chosen threshold. This repository contains every
script behind those numbers.

## Headline results

Detection floors under a threshold-free criterion, in millimetres of tip
deviation:

| Injection pattern | Observation stream | Command stream |
|---|---|---|
| step  | 0.142 | 0.290 |
| ramp  | 1.300 | not detected |
| noise | 0.031 | not detected |

Which feature sees which injection point, in millimetres:

| | (A) desired pose | (B) motor torque | (C) reported position |
|---|---|---|---|
| torque-to-motion residual | no | likely no | 0.81 |
| servo tracking error | 2.85 | **0.53** | 1.35 |

Neither feature covers all three points; together they do. Under a fixed
alarm budget, none of the three patterns retains an operating point, for a
reason that belongs to the data rather than the method (Section 5).

`MANIFEST.md` maps every number in the paper to the script that produced it.

---

## Quick start

```bash
# 1. Get the data
#    Peng et al., npj Robotics 2(1):9, 2024
#    https://doi.org/10.5061/dryad.tqjq2bw84
export RAVEN_DATA=/path/to/peng_uw_dataset
export RAVEN_OUT=/path/to/output

# 2. Verify the column mapping and the leakage controls
python3 src/peng_uw_loader.py
python3 src/diag_leakage.py

# 3. Run whichever experiment you want (see Section 3)
python3 src/exp_operating_point.py
```

Without the environment variables the scripts fall back to `./data` and
`./output`.

---

## 1. Data

```
$RAVEN_DATA/
  record_1_different_directions/     injection experiments
  record_3_time_decay_500gload/      session-scale analysis, six hours at 500 g
  record_3_time_decay_unloaded/      six hours, no load
  record_3_time_decay_idle/          six hours, idle
```

Each `record_3_*` directory stores one file per hour, with `_h0` through
`_h5` in the filename.

**Column indices.** The index list in the dataset documentation is off by one
over its first section: ground-truth joint positions occupy columns 1 to 3,
not 2 to 4. `peng_uw_loader.py` uses the corrected mapping and applies three
consistency checks, on field ranges, on correlation against the reported
positions, and on correlation of the forward-kinematic z against the
encoder-derived z. Run it first; if those checks fail, nothing downstream is
meaningful.

The recordings are benign. They were collected for joint calibration, and no
public surgical-robot dataset contains real attacks, so every injection here
is synthetic. `diag_leakage.py` holds the three negative controls that guard
against the synthesis leaking labels.

---

## 2. Environment

```
Python     3.10 or later
PyTorch    2.x
numpy, scipy, scikit-learn, matplotlib, pandas
```

Training needs a GPU. Latency was measured on RAVEN-II's own control host,
which has none, and that is deliberate: the question is whether the monitor
fits the machine it would actually run on.

From PyTorch 2.6 the default of `torch.load`'s `weights_only` changed to
`True`. `train_and_export.py` writes checkpoints that load under the new
default.

Two backbone modules carry the shared machinery and are imported by most of
the rest:

```
exp_delta_ablation.py    Cfg, load_records, fit_physics_model,
                         build_window_cache, build_features,
                         MaskedMultiTaskLoss
exp_paper_sweep.py       SweepInjector, inject_once,
                         detection_limit, run_one
```

Both are included. One further import is not: `raven2_peng_pipeline_v3.py`
needs `torch_geometric`, left over from a graph-topology ablation. The final
paper does not use graph structure, since the ablation found no difference
between the DH topology and a random one (p = 0.87), so that script can be
skipped.

---

## 3. Order of reproduction

Order matters within a group, not between groups.

### Preliminaries

```bash
python3 src/peng_uw_loader.py        # verify the column mapping
python3 src/diag_leakage.py          # three negative controls, all near 0.50
```

The controls are null injection (the injected tensor is bitwise identical to
the clean one while labels are still assigned), channel sanitisation, and
label shuffling. Each removes the real signal but keeps the labels, so a
detector that still scores above chance is using something it should not. If
any of the three exceeds 0.5 meaningfully, stop.

### Does the window-scale path need a second timescale?

```bash
python3 src/probe_drift.py             # six-hour drift, tip displacement
python3 src/probe_slow_path.py         # session-scale observable, four checks
python3 src/probe_canary_transfer.py   # cross-load transfer of the calibration
python3 src/exp_slow_attribution.py    # compensation-fidelity sweep
```

### Does any single feature cover the injection taxonomy?

```bash
python3 src/exp_tracking_error.py      # lag estimation, ARX identification, floors
python3 src/exp_residual_input.py      # physics residual as input
```

### Floors under a fixed alarm budget

```bash
python3 src/exp_operating_point.py
python3 src/exp_tracking_operating.py
```

Read Section 5 before interpreting the output of these two.

### The command stream

```bash
python3 src/exp_command_path.py
```

### Supporting experiments

```bash
python3 src/exp_perjoint_graph.py      # per-joint encoder ablation
python3 src/exp_physics_loss.py        # physics-loss ablation
python3 src/exp_cross_attack_v2.py     # cross-family generalisation matrix
python3 src/exp_predictive.py          # lead time from the prediction residual
```

### Deployment artifacts

```bash
python3 src/train_and_export.py        # train and export one model
python3 src/selftest.py                # end-to-end check before hardware
python3 src/bench_latency.py           # inference latency, on the target host
```

`DEPLOY.md` covers what to change before connecting to a robot.

---

## 4. Two pitfalls that change conclusions

### weight_decay must be zero

Every custom training loop must use `weight_decay=0`. At `1e-4` the noise
family's AUC collapses from 0.9999 to 0.5065 while step and ramp are
unaffected.

Noise injection changes only the variance, not the mean, and over a
hundred-odd epochs weight decay suppresses that signal. Step and ramp are
mean shifts and survive.

`exp_physics_loss.py` carries the full bisection log under the marker `[WD]`,
and `train_and_export.py` carries an inline comment on that line so nobody
adds it back while optimising.

### Normalisation parameters must not be recomputed online

Deployment must apply the `mu` and `sd` stored at training time. Recomputing
them online lets a sustained injection be absorbed by the normalisation: the
mean shift is subtracted away, the variance change is divided out, and almost
no signal remains.

`train_and_export.py` stores both alongside the weights and records this in
the checkpoint's `note` field.

---

## 5. A correction made before submission

An earlier version of the analysis reported that step injection reaches
0.95 mm at two alarms per procedure hour. A check found two counting errors,
both in the same direction, both making the result look better than it is.

**Decision frequency counted non-overlapping windows.** The window is 30
frames at stride one, so a deployed monitor scoring every frame makes 30
times as many decisions as the non-overlapping count of 15,850 per hour. The
original comment in the script called the non-overlapping count
conservative; that judgement is backwards. If deployment is worse, reporting
the smaller number is optimistic.

**Validation windows are not independent.** Adjacent windows share 29 of 30
frames, so the 8,731 held-out windows thin to roughly 290 once made mutually
non-overlapping. Estimating a tail quantile needs independent samples, and
8,731 overlapping ones do not supply 8,731 of them.

| | Before | After |
|---|---|---|
| decision frequency | 15,850 per hour | 30x more if every frame is scored |
| clean samples | 8,731 | about 290 |
| strictest estimable budget | 1.8 per hour | about 38 per hour |

Together these mean the 0.95 mm claim cannot be estimated from this data.
After the correction, none of the three injection patterns retains an
operating point at any budget the held-out data supports.

That negative result is what the paper reports, because it belongs to the
data rather than to the method: longer clean recordings would restore the
estimate, a better detector would not.

Quantified in `exp_window_accounting.py`; the fix is applied by
`patch_operating_point.py`, which preserves the original as
`exp_operating_point.py.orig`.

---

## 6. Known reproduction differences

**GPU non-determinism.** The cuDNN LSTM backward pass is not bitwise
deterministic, so a re-run with the same seed moves AUC by about 0.004. The
floors are log-interpolated, so that jitter changes no conclusion, but the
third decimal place may differ.

**The self-test criterion is correlation, not maximum error.** The same model
on CPU and GPU, or at different batch sizes, differs by a median of about
1.3e-04, while individual samples near sigmoid saturation reach 2.4e-02.
`selftest.py` therefore judges on correlation (> 0.999) and mean absolute
error (< 0.01). A criterion on the maximum fails every time and is therefore
ignored, which is worse than having no check.

**Session-scale statistics rest on very few points.** Each load condition
gives six hourly points, of which three are used for fitting. The scripts
bootstrap rather than assume a sampling distribution.

---

## 7. What was not done

- All injections are synthetic. Injection C perturbs the recorded reported
  positions directly and is faithful, since the arm does not move. Injection
  A depends on the identified servo model and injection B on a measured
  torque-to-displacement gain, so both carry a modelling assumption the data
  cannot check.
- The session-scale audit is validated on joint 1 only and needs an external
  encoder reference, so it is a calibration-time check rather than an online
  monitor.
- Nothing has been run on a robot in motion.

---

## Licence

See `LICENSE`.
