# Connecting the Exported Detector to Hardware

`train_and_export.py` produces three files:

```
detector.pt          weights, normalisation parameters, metadata
detector_script.pt   TorchScript, self-contained, needs none of this package
selftest.npz         64 self-test samples with their expected scores
```

Current status: validated on six held-out sessions of the public dataset
(step 0.9933, ramp 0.7951, noise 0.7450, mixed 0.8421). Never run on a robot
in motion.

---

## 0. Confirm the checkpoint is complete

```python
import torch
ck = torch.load('detector.pt')          # works with weights_only=True
print(ck['input_dim'], ck['hidden'], ck['lookback'], ck['subsample'])
print(ck['mode'], ck['val_auc'])
print(ck['mu'][:3], ck['sd'][:3])
```

`mode` should be `jpos_torque_delta`, `lookback` 30, `subsample` 5.

If loading raises `UnpicklingError: Weights only load failed`, the checkpoint
came from an older version of the script, which stored numpy arrays. Either
re-run `train_and_export.py`, or load with
`torch.load(..., weights_only=False)` as a stopgap. Re-exporting is cleaner.

---

## 1. Channel mapping: the step most likely to go wrong

The dataset is read by column index; ROS reads `ravenstate` by field name.
The correspondence has to be checked by hand.

The nineteen channels the model expects, in order:

```
[0:8]    jpos      eight joint positions
[8:16]   torque    eight motor torques
[16:19]  delta     first difference of the first three joints,
                   delta[t] = jpos[t] - jpos[t-1]
```

**Unit trap.** In the dataset, joints 1 and 2 are in degrees and the rest in
radians. If ROS reports everything in radians, the first two joints need a
factor of 180/pi.

### How to verify the mapping

Record a clean segment, build features with the same code, then:

```python
dj = np.diff(jpos[:, :3], axis=0)
print(dj.std(axis=0))
```

The result should land near `[0.17, 0.22, 0.02]` degrees. An order of
magnitude off means the units are wrong; completely different values mean the
column order is wrong.

Check the torque range the same way: it should sit near +/-0.4.

---

## 2. Normalisation: never recompute online

```python
x = (x_raw - ck['mu'].numpy()) / ck['sd'].numpy()     # use what was stored
```

**Recomputing online defeats the detector.** A sustained injection is
absorbed by the normalisation: the mean shift is subtracted away, the
variance change is divided out, and almost no signal remains.

This is also recorded in the checkpoint's `note` field.

### If your machine differs substantially from the one in the dataset

New cable, fresh calibration, a different operator: the telemetry statistics
change and the stored `mu`/`sd` may not apply. In that case **retrain on your
own clean data** rather than swapping the normalisation parameters alone.
Point `load_records` at your data and re-run `train_and_export.py`.

---

## 3. Sample rate and window

```
ck['subsample'] = 5      660 Hz down to 132 Hz
ck['lookback']  = 30     30 frames = 227 ms
```

If your telemetry is not at 660 Hz, adjust `subsample` so the result stays
near 132 Hz. **The window must span 227 ms**, since every floor was measured
at that duration.

---

## 4. Threshold: not in the checkpoint

The threshold depends on how many false alarms you can accept, so it has to
be set on your own clean data:

```python
clean_scores = [score(w) for w in clean_windows]
fpr = alarms_per_hour / 15850          # 227 ms windows, 15850 per hour
threshold = np.quantile(clean_scores, 1 - fpr)
```

**The windows must be independent.** At stride one, adjacent windows share 29
of 30 frames and do not count as separate samples for a tail quantile. Thin
to non-overlapping windows first, which costs a factor of 30 in sample size.

This is the same error the paper's pre-submission check found; see
`MANIFEST.md`. Estimating a quantile at `1/FPR` needs on the order of
`1/FPR` **independent** windows, so a budget of two alarms per hour needs
roughly 8,000 of them, which is about 30 minutes of continuous recording
after thinning, not one minute.

Do not extrapolate past what the data supports. A generalised-Pareto tail fit
was tried and produced inconsistent results: tighter budgets came out with
lower thresholds.

---

## 5. Run the self-test before connecting anything

```bash
python3 selftest.py
```

It loads the checkpoint, scores the 64 stored samples, and compares against
the scores recorded at export time.

**The criterion is correlation, not maximum error.** The same model gives
slightly different scores across environments: the median difference is about
1.3e-04, but individual samples near sigmoid saturation reach 2.4e-02. That
is normal. A genuine wiring fault drops correlation from 0.9999 to below 0.5.

A criterion on the maximum error fails every time and is therefore ignored,
which is worse than having no check.

Failure tells you where to look:

```
correlation below 0.9        channel order or units, a wiring fault
correlation fine, offset big  wrong normalisation parameters
```

---

## 6. Deployment shape

```
ROS node subscribes to ravenstate
  -> maintain a 30-frame sliding window (after subsampling)
  -> score each window
  -> log or alarm above the threshold
```

**Passive monitoring first.** Do not put the monitor in the control loop. A
fault in the model then costs attention rather than a halted procedure, and
the first round measures the thing that matters most, which is the real false
alarm rate.

Inference costs 0.662 ms at the 99th percentile on a single thread of
RAVEN-II's control host, batch size one. That is a third of the 1 ms control
period; non-overlapping windows consume 0.3 % of available time.

---

## 7. What the first round should measure

In order of value:

**False alarm rate.** Record several hours of normal operation and count
alarms. Compare against the held-out expectation. A mismatch is itself a
finding: it would show distribution shift across time or across machines.

**Injection test.** Man-in-the-middle at the ROS layer, modifying the copy of
`jpos` that goes to the monitor rather than the one the control loop reads,
so the robot is unaffected. Then sweep magnitude and compare against the
paper's floors.

**Latency.** Measure inside the real ROS callback, not as a standalone
forward pass. The 0.662 ms figure excludes ROS serialisation and callback
overhead.

---

## Known gaps

- Every floor in the paper uses a **concatenated encoder**. A per-joint
  encoder reaches 41 % lower on single-joint injection with 60 % fewer
  parameters (see the companion ablation), but was validated on ramp only.
  Before deploying the better architecture, run it on all three patterns.
- Training uses mixed injection, all three patterns drawn at random, so
  single-pattern floors are 1.13 to 1.48 times looser than the paper's. That
  is the cost of not knowing which pattern an adversary will use.
