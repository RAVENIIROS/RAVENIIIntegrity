# [ARTIFACT] deploy | End-to-end check before connecting to hardware
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[S] Deployment self-test: confirm the exported model and the whole feature chain line up
================================================================
Run this before connecting to a robot. It loads detector.pt, runs inference on the
samples in selftest.npz, and compares against the scores recorded at export time.

The criterion is correlation, not maximum error. The same model gives slightly
different scores across environments (CPU vs GPU, batch size, numpy version):
the median difference measures 1.3e-04, but individual samples near sigmoid

saturation reach 2e-02. That is not an error. A genuine wiring mistake, wrong
channel order, wrong units (joints 1-2 are degrees, the rest radians), or the
wrong normalisation, drops correlation from 0.9999 to below 0.5. That is the

quantity with discriminative power.

Usage: python3 selftest.py [checkpoint_dir]
"""
import sys, os
import numpy as np
import torch
import torch.nn as nn


class Det(nn.Module):
    """Must match the definition in train_and_export.py verbatim"""
    def __init__(s, d, h, la):
        super().__init__(); s.la = la
        s.lstm = nn.LSTM(d, h, 2, batch_first=True, dropout=0.2)
        s.dev = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, la*3))
        s.atk = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):
        h = s.lstm(x)[0][:, -1, :]
        return s.dev(h).view(-1, s.la, 3), s.atk(h).squeeze(-1)


def main():
    d_dir = sys.argv[1] if len(sys.argv) > 1 else 'output_export'
    ck_path = os.path.join(d_dir, 'detector.pt')
    st_path = os.path.join(d_dir, 'selftest.npz')
    for p in (ck_path, st_path):
        if not os.path.exists(p):
            print(f"  not found: {p}"); sys.exit(1)

    ck = torch.load(ck_path)          # works with weights_only=True
    d  = np.load(st_path)

    print("=" * 72)
    print(" [S] Deployment self-test")
    print("=" * 72)
    print(f"""
  checkpoint      {ck_path}
  input dim       {ck['input_dim']}   (mode = {ck['mode']})
  window          {ck['lookback']} frames, subsample {ck['subsample']}x
  validation AUC  {ck['val_auc']:.4f}
  self-test set   {len(d['expected'])} samples, {int(d['label'].sum())} positive
""")

    m = Det(ck['input_dim'], ck['hidden'], ck['lookahead'])
    m.load_state_dict(ck['state_dict'])
    m.eval()

    # Normalisation must use the stored mu/sd; never recompute online
    x = (d['x_raw'] - ck['mu'].numpy()) / ck['sd'].numpy()
    with torch.no_grad():
        s = torch.sigmoid(m(torch.from_numpy(x.astype('float32')))[1]).numpy()

    e = d['expected']
    corr = float(np.corrcoef(s, e)[0, 1])
    mae  = float(np.abs(s - e).mean())
    med  = float(np.median(np.abs(s - e)))
    mx   = float(np.abs(s - e).max())

    # Thresholds live in the npz; older checkpoints lack them, so fall back
    corr_min = float(d['corr_min']) if 'corr_min' in d.files else 0.999
    mae_max  = float(d['mae_max'])  if 'mae_max'  in d.files else 0.01

    print(f"  {'correlation':<22}{corr:>10.6f}   need > {corr_min:.3f}")
    print(f"  {'mean abs error':<22}{mae:>10.6f}   need < {mae_max:.3f}")
    print(f"  {'median error':<22}{med:>10.2e}   (reference)")
    print(f"  {'max error':<22}{mx:>10.2e}   (reference, not the criterion)")

    ok = corr > corr_min and mae < mae_max
    print()
    if ok:
        print("  PASS. Feature assembly, units and normalisation all line up.")
    else:
        print("  FAIL. Check in this order:")
        print("    1. channel order: [0:8] jpos, [8:16] torque, [16:19] delta")
        print("    2. units:         joints 1-2 in degrees, the rest in radians")
        print("    3. normalisation: use ck['mu'] and ck['sd'], never recompute")
        print("    4. subsample:     the window must span 227 ms")
        if corr < 0.9:
            print("\n  Correlation is low: check 1 and 2 first, a wiring fault, not numerics.")
        elif mae >= mae_max:
            print("\n  Correlation is fine but the offset is large: check 3, wrong mu/sd.")
    print("=" * 72)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
