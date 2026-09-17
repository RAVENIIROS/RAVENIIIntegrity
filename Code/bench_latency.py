# [ARTIFACT] deploy | Inference latency, must be run on the control host
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[L] Inference latency benchmark: can this detector run on RAVEN-II's control host?

The RTX 4090 figure in early drafts was measured on a laptop GPU. RAVEN-II's
control host has no GPU, so that number says nothing about feasibility.

This script measures the same model on CPU and compares three possible remedies.

-- Three measurement decisions ------------------------------------

  [D1] batch=1, single thread
       Deployment scores one window at a time rather than in batches, and a
       real-time kernel typically gives the monitor process one core. Multi-
       threaded and batched numbers are optimistic. We report multi-threaded
       as a reference but judge on single thread.

  [D2] Report the tail, not the mean
       Hard real time is about the worst case: one overrun is one missed
       decision window. We report p50/p95/p99/max and judge on p99.

  [D3] Distinguish "per window" from "per frame"
       The window is 30 frames at 132 Hz. If deployment scores on every
       incoming frame (sliding), the load is one inference per 7.6 ms; if it
       scores every 30 frames (non-overlapping), one per 227 ms. The two
       differ by 30x in what they demand. Both are reported.

-- The four configurations compared -------------------------------

  baseline     the model in use (LSTM 2x128) under PyTorch eager
  half         hidden 128 -> 64, roughly a quarter of the arithmetic
  torchscript  TorchScript trace, Python overhead removed
  conv         causal convolution in place of the LSTM, same 30-frame
               receptive field but parallel over time

  The conv row is the interesting one: the LSTM's 30 serial steps are the
  most expensive structure on a CPU, while a causal convolution keeps the
  temporal receptive field and computes in one pass. If it is much faster
  without losing accuracy, that is the answer to wanting both sequence
  modelling and real time. This script measures latency only, not accuracy.

Usage:    python3 bench_latency.py
Expected: 2 minutes, no GPU and no dataset required
"""
import time, sys
import numpy as np
import torch, torch.nn as nn

CONTROL_PERIOD_MS = 1.0      # RAVEN-II control period
WINDOW_FRAMES     = 30
SUBSAMPLE_HZ      = 132.0
N_WARM, N_RUN     = 100, 1000


class LSTMDet(nn.Module):
    """The model in use"""
    def __init__(s, d=19, h=128, la=20):
        super().__init__(); s.la = la
        s.lstm = nn.LSTM(d, h, 2, batch_first=True, dropout=0.2)
        s.dev = nn.Sequential(nn.Linear(h,256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256,64), nn.ReLU(), nn.Linear(64,la*3))
        s.atk = nn.Sequential(nn.Linear(h,128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128,32), nn.ReLU(), nn.Linear(32,1))
    def forward(s, x):
        h = s.lstm(x)[0][:, -1, :]
        return s.dev(h).view(-1, s.la, 3), s.atk(h).squeeze(-1)


class ConvDet(nn.Module):
    """
    Causal convolution in place of the LSTM. Kernel 3 with dilations 1/2/4/8
    gives a receptive field of 31 >= 30, covering the same window.
    Key difference from the LSTM: the time axis is parallel, no 30-step chain.
    """
    def __init__(s, d=19, h=64, la=20):
        super().__init__(); s.la = la
        def blk(i, o, dil):
            return nn.Sequential(
                nn.Conv1d(i, o, 3, padding=2*dil, dilation=dil), nn.ReLU())
        s.net = nn.Sequential(blk(d,h,1), blk(h,h,2), blk(h,h,4), blk(h,h,8))
        s.dev = nn.Sequential(nn.Linear(h,256), nn.ReLU(),
                              nn.Linear(256,64), nn.ReLU(), nn.Linear(64,la*3))
        s.atk = nn.Sequential(nn.Linear(h,128), nn.ReLU(),
                              nn.Linear(128,32), nn.ReLU(), nn.Linear(32,1))
    def forward(s, x):
        y = s.net(x.transpose(1,2))[:, :, -1]      # take the last time step
        return s.dev(y).view(-1, s.la, 3), s.atk(y).squeeze(-1)


def bench(model, x, n_warm=N_WARM, n_run=N_RUN):
    """[D2] Return the per-call latency distribution, in ms"""
    model.eval()
    lat = np.empty(n_run)
    with torch.no_grad():
        for _ in range(n_warm):
            model(x)
        for i in range(n_run):
            t0 = time.perf_counter()
            model(x)
            lat[i] = (time.perf_counter() - t0) * 1000.0
    return lat


def report(name, lat, n_params):
    p50, p95, p99 = np.percentile(lat, [50, 95, 99])
    print(f"  {name:<16}{n_params:>10,}{p50:>9.3f}{p95:>9.3f}"
          f"{p99:>9.3f}{lat.max():>9.3f}")
    return p99


def main():
    print("=" * 92)
    print(" [L] CPU inference latency: does the detector fit the control host?")
    print("=" * 92)
    print(f"""
  control period  {CONTROL_PERIOD_MS} ms
  window          {WINDOW_FRAMES} frames @ {SUBSAMPLE_HZ:.0f} Hz = {WINDOW_FRAMES/SUBSAMPLE_HZ*1000:.0f} ms
  [D1] batch=1, single thread (one window at a time, one core)
  [D2] p99 rather than the mean (hard real time is about the worst case)
""")
    x = torch.randn(1, WINDOW_FRAMES, 19)
    npar = lambda m: sum(p.numel() for p in m.parameters())

    results = {}
    for threads in [1, 4]:
        torch.set_num_threads(threads)
        print(f"\n  -- {threads} thread(s) --")
        print(f"  {'config':<16}{'params':>10}{'p50':>9}{'p95':>9}"
              f"{'p99':>9}{'max':>9}   (ms)")
        print("-" * 92)

        m = LSTMDet(h=128)
        results[(threads,'baseline')] = report('baseline', bench(m, x), npar(m))

        m = LSTMDet(h=64)
        results[(threads,'half')] = report('hidden 64', bench(m, x), npar(m))

        try:
            m = LSTMDet(h=128).eval()
            with torch.no_grad():
                ts = torch.jit.trace(m, x)
                ts = torch.jit.optimize_for_inference(ts)
            results[(threads,'torchscript')] = report('torchscript', bench(ts, x), npar(m))
        except Exception as e:
            print(f"  {'torchscript':<16} failed: {type(e).__name__}")

        m = ConvDet(h=64)
        results[(threads,'conv')] = report('causal conv', bench(m, x), npar(m))

    # == Interpretation ==
    print("\n" + "=" * 92)
    print(" Interpretation")
    print("=" * 92)
    frame_ms = 1000.0 / SUBSAMPLE_HZ
    win_ms   = WINDOW_FRAMES / SUBSAMPLE_HZ * 1000.0
    print(f"""
  [D3] Two deployment conventions:
    sliding     one decision per incoming frame -> one inference / {frame_ms:.1f} ms
    non-overlap one decision per {WINDOW_FRAMES} frames -> one inference / {win_ms:.0f} ms
  The {CONTROL_PERIOD_MS} ms control period is a stricter constraint, binding
  only if the monitor must run in lockstep with the control loop.
""")
    print(f"  {'config':<16}{'p99 (ms)':>11}{'< 1ms?':>9}"
          f"{'< 7.6ms?':>10}{'< 227ms?':>10}")
    print("-" * 92)
    for k in ['baseline','half','torchscript','conv']:
        v = results.get((1,k))
        if v is None: continue
        print(f"  {k:<16}{v:>11.3f}{'yes' if v<CONTROL_PERIOD_MS else 'NO':>9}"
              f"{'yes' if v<frame_ms else 'NO':>10}"
              f"{'yes' if v<win_ms else 'NO':>10}")

    base = results.get((1,'baseline'))
    if base:
        print(f"""
  Model in use, single thread, p99 = {base:.3f} ms

  if < 1 ms      can run in lockstep with the control loop
  if 1 to 7.6 ms no lockstep, but sliding-window monitoring still works
  if > 7.6 ms    non-overlapping windows only (one per 227 ms), or optimise
""")
        for k, label in [('half','hidden halved'), ('torchscript','TorchScript'),
                         ('conv','causal conv')]:
            v = results.get((1,k))
            if v: print(f"    {label:<14} {v:.3f} ms   ({base/v:.1f}x speedup)")
        print(f"""
  The causal-convolution row is worth reading on its own: the LSTM's
  {WINDOW_FRAMES} serial steps are the most expensive structure on a CPU,
  while the convolution keeps the receptive field in one pass. Latency only here.
""")
    print("=" * 92)


if __name__ == '__main__':
    main()
