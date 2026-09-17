
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] deploy | Train and export one deployable model
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[X] Train one model and export it, for deployment or hardware validation

Every earlier training run in this project was disposable: compute the AUC,
then delete the model, because the paper needs the detection floor rather
than the detector. Hardware validation needs a saved model, which is a step
never taken before.

-- Four design decisions -----------------------------------------

  [D1] Train on mixed injection, not one pattern
       The cross-family experiment found that generalisation fails in three
       of six directions and that ramp never transfers. Deployment does not
       know which pattern an adversary will use, so training follows the
       "all" protocol: each attacked window draws a pattern at random with a
       log-uniform magnitude.
       Cost: floors are 1.13 to 1.48 times looser than single-pattern
       training. That is known and accepted.

  [D2] The normalisation parameters must be stored with the weights
       Deployment must use the training set's mu and sd and must never
       recompute them online. Recomputing lets a sustained injection be
       absorbed by the normalisation: the mean shift is subtracted away and
       the variance change divided out, leaving the detector blind to it.
       This is the easiest step to miss, so the script writes it into the
       checkpoint and checks it on load.

  [D3] Export TorchScript as well
       The deployment environment may not have this project's Python modules.
       TorchScript is self-contained and needs no definition of the Det class.
       Measured on RAVEN-II's control host (bench_latency), TorchScript is
       slightly slower than eager, so both are stored and the choice is made
       from measurement at deployment time.

  [D4] Store self-test samples
       Save a handful of validation windows together with their scores. At
       deployment, run them first; matching scores are what show the whole
       chain (channel order, units, normalisation) is wired correctly.
       Wrong channel order is the most common failure in this kind of
       deployment, and it raises no error: it just quietly degrades
       performance.

       The criterion is correlation, not maximum error. The same model gives
       scores differing by up to 2e-2 across normalisation paths (numpy's
       float32/float64 promotion order), especially near sigmoid saturation,
       and that is not a fault. A genuine wiring mistake drops correlation
       from 0.9999 to below 0.5, which is the quantity with discriminative
       power.

Usage:    python3 train_and_export.py
Expected: one training run, about 2 minutes
Outputs:  output_export/detector.pt          weights, normalisation, metadata
          output_export/detector_script.pt   TorchScript
          output_export/selftest.npz         self-test samples and scores
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_delta_ablation import (Cfg as BaseCfg, load_records, fit_physics_model,
                                build_window_cache, build_features,
                                MaskedMultiTaskLoss)
from exp_paper_sweep import SweepInjector


class Cfg(BaseCfg):
    OUT_DIR   = _OUT + '/output_export'
    MODE      = 'jpos_torque_delta'
    SEED      = 42
    EPOCHS    = 120
    LR        = 3e-4
    BATCH     = 128
    HIDDEN    = 128
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1
    # [D1] Mixed injection: random pattern per window, log-uniform magnitude
    FAMILIES    = ['step', 'ramp', 'noise']
    JOINTS      = dict(step=[0], ramp=[0, 1], noise=[0])
    TRAIN_RANGE = dict(step=(0.008, 0.15), ramp=(0.010, 0.60),
                       noise=(0.0003, 0.03))
    N_SELFTEST  = 64


class Det(nn.Module):
    def __init__(s, d, h, la):
        super().__init__(); s.la = la
        s.lstm = nn.LSTM(d, h, 2, batch_first=True, dropout=0.2)
        s.dev = nn.Sequential(nn.Linear(h, 256), nn.ReLU(), nn.Dropout(0.1),
                              nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, la*3))
        nn.init.zeros_(s.dev[-1].weight); nn.init.zeros_(s.dev[-1].bias)
        s.atk = nn.Sequential(nn.Linear(h, 128), nn.ReLU(), nn.Dropout(0.2),
                              nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, x):
        h = s.lstm(x)[0][:, -1, :]
        return s.dev(h).view(-1, s.la, 3), s.atk(h).squeeze(-1)


def inject_mixed(JP, cfg, mask, seed_off=0):
    """[D1] Draw a pattern at random per attacked window, log-uniform magnitude"""
    rng = np.random.RandomState(cfg.SEED + seed_off)
    out = JP.copy(); y = np.zeros(len(JP), np.float32)
    fam_of = np.full(len(JP), '', dtype=object)
    pool = np.where(mask)[0]
    for i in rng.permutation(pool)[:int(len(pool) * cfg.ATTACK_FRAC)]:
        fam = cfg.FAMILIES[rng.randint(len(cfg.FAMILIES))]
        lo, hi = cfg.TRAIN_RANGE[fam]
        sc = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        j = cfg.JOINTS[fam][rng.randint(len(cfg.JOINTS[fam]))]
        out[i], _, _, _, _ = SweepInjector.inject(JP[i], scale=sc, rng=rng,
                                                  force_type=fam, force_joint=j)
        y[i] = 1.0; fam_of[i] = fam
    return out, y, fam_of


def main():
    cfg = Cfg(); os.makedirs(cfg.OUT_DIR, exist_ok=True); t0 = time.time()
    dev = cfg.DEVICE
    print("=" * 96)
    print(" [X] Train and export a deployable detector")
    print("=" * 96)
    print(f"""
  [D1] mixed-injection training (random pattern, log-uniform magnitude)
  [D2] normalisation parameters stored with the weights
  [D3] TorchScript exported alongside
  [D4] self-test samples stored
""")
    recs = load_records(cfg)
    a_, b_, r2, _, _ = fit_physics_model(recs, cfg.N_POS_JOINTS)
    JP, TQ, YD, GID = build_window_cache(recs, cfg)

    g = np.unique(GID); nv = max(1, int(round(len(g) * cfg.VAL_FRAC_RECORDS)))
    vg = np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva = np.isin(GID, vg); mtr = ~mva
    print(f"  windows {JP.shape}   train {mtr.sum():,}  val {mva.sum():,}")

    JPtr, ytr_all, _ = inject_mixed(JP, cfg, mtr, 0)
    JPva, yva_all, fam_va = inject_mixed(JP, cfg, mva, 7777)
    JPm = JP.copy(); JPm[mtr] = JPtr[mtr]; JPm[mva] = JPva[mva]
    y = np.zeros(len(JP), np.float32); y[mtr] = ytr_all[mtr]; y[mva] = yva_all[mva]
    X = build_features(JPm, TQ, cfg.MODE, a_, b_, cfg.N_POS_JOINTS)
    print(f"  features {X.shape}   positive {int(y.sum()):,} / {len(y):,}")

    # [D2] Normalisation from the training set only, and stored
    f = X[mtr].reshape(-1, X.shape[-1])
    mu = f.mean(0).astype(np.float32)
    sd = np.maximum(f.std(0), 1e-6).astype(np.float32)
    Xtr = ((X[mtr] - mu) / sd).astype(np.float32)
    Xva = ((X[mva] - mu) / sd).astype(np.float32)
    ytr, yva = y[mtr], y[mva]
    std_delta = float(YD[mtr].std() + 1e-8)
    Ydtr = (YD[mtr] / std_delta).astype(np.float32)

    torch.manual_seed(cfg.SEED); np.random.seed(cfg.SEED)
    m = Det(X.shape[-1], cfg.HIDDEN, cfg.LOOKAHEAD).to(dev)
    pw = torch.tensor(float((ytr < 0.5).sum() / max((ytr >= 0.5).sum(), 1)),
                      dtype=torch.float32, device=dev)
    lossf = MaskedMultiTaskLoss(cfg.W_DEV, cfg.W_ATK, cfg.W_VEL,
                                pos_weight=pw).to(dev)
    # weight_decay=0: see the [WD] note in exp_physics_loss
    opt = torch.optim.Adam(m.parameters(), lr=cfg.LR, weight_decay=0.0)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.EPOCHS)

    Xt = torch.from_numpy(Xtr).to(dev); Yt = torch.from_numpy(ytr).to(dev)
    Dt = torch.from_numpy(Ydtr).to(dev); Xv = torch.from_numpy(Xva).to(dev)
    print(f"\n  training {cfg.EPOCHS} epochs...")
    for ep in range(cfg.EPOCHS):
        m.train(); perm = torch.randperm(len(Xt), device=dev)
        for s0 in range(0, len(Xt), cfg.BATCH):
            i = perm[s0:s0 + cfg.BATCH]; opt.zero_grad()
            dp, ap = m(Xt[i]); lossf(dp, ap, Dt[i], Yt[i]).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        sch.step()
        if (ep + 1) % 40 == 0: print(f"    epoch {ep+1}  ({time.time()-t0:.0f}s)")

    m.eval()
    with torch.no_grad():
        _, ap = m(Xv)
    scores = torch.sigmoid(ap).cpu().numpy()
    auc = roc_auc_score(yva, scores)
    print(f"\n  validation AUC (mixed injection) = {auc:.4f}")
    for fam in cfg.FAMILIES:
        sel = np.array([f_ == fam for f_ in fam_va[mva]]) | (yva < 0.5)
        if sel.sum() > 10 and len(np.unique(yva[sel])) > 1:
            print(f"    {fam:<6} {roc_auc_score(yva[sel], scores[sel]):.4f}")

    # == Export ==
    # Everything stored as tensors or native Python types, so that the
    # weights_only=True default of PyTorch 2.6+ loads it without being disabled.
    ck = {
        'state_dict':  {k: v.cpu() for k, v in m.state_dict().items()},
        'mu':          torch.from_numpy(mu),    # [D2] must travel with the weights
        'sd':          torch.from_numpy(sd),
        'std_delta':   std_delta,
        'input_dim':   int(X.shape[-1]),
        'hidden':      cfg.HIDDEN,
        'lookahead':   cfg.LOOKAHEAD,
        'lookback':    cfg.LOOKBACK,
        'subsample':   getattr(cfg, 'SUBSAMPLE', 5),
        'mode':        cfg.MODE,
        'physics_a':   [float(x) for x in a_],
        'physics_b':   [float(x) for x in b_],
        'val_auc':     float(auc),
        'note':        ('mu/sd must be applied as stored; recomputing them '
                        'online lets a sustained injection be absorbed by '
                        'the normalisation'),
    }
    torch.save(ck, f'{cfg.OUT_DIR}/detector.pt')
    print(f"\n  wrote {cfg.OUT_DIR}/detector.pt")

    # [D3] TorchScript
    try:
        m_cpu = Det(X.shape[-1], cfg.HIDDEN, cfg.LOOKAHEAD)
        m_cpu.load_state_dict(ck['state_dict']); m_cpu.eval()
        with torch.no_grad():
            ts = torch.jit.trace(m_cpu, torch.from_numpy(Xva[:1]))
        ts.save(f'{cfg.OUT_DIR}/detector_script.pt')
        print(f"  wrote {cfg.OUT_DIR}/detector_script.pt")
    except Exception as e:
        print(f"  TorchScript export failed: {type(e).__name__}: {e}")

    # [D4] Self-test samples
    k = min(cfg.N_SELFTEST, len(Xva))
    idx = np.random.RandomState(0).choice(len(Xva), k, replace=False)
    np.savez(f'{cfg.OUT_DIR}/selftest.npz',
             x_raw=X[mva][idx],          # unnormalised; deployment normalises
             expected=scores[idx],
             label=yva[idx],
             # Criterion: correlation > 0.999 and mean absolute error < 0.01.
             # Not maximum error; see [D4], float paths differ by up to 2e-2.
             corr_min=np.float32(0.999),
             mae_max=np.float32(0.01))
    print(f"  wrote {cfg.OUT_DIR}/selftest.npz  ({k} samples)")

    print(f"""
  -- How to use this at deployment ---------------------------
    ck = torch.load('detector.pt')
    m  = Det(ck['input_dim'], ck['hidden'], ck['lookahead'])
    m.load_state_dict(ck['state_dict']); m.eval()

    # Per window: take ck['lookback'] frames, subsample by ck['subsample'],
    # assemble features per ck['mode'], then
    x = (x_raw - ck['mu'].numpy()) / ck['sd'].numpy()   # stored, never recomputed
    score = torch.sigmoid(m(x)[1])

    Run the samples in selftest.npz before connecting hardware:
      d = np.load('selftest.npz')
      s = score(d['x_raw'])
      ok = (np.corrcoef(s, d['expected'])[0,1] > d['corr_min']
            and np.abs(s - d['expected']).mean() < d['mae_max'])
    A failure usually means wrong channel order or units, which raises no error.
  ─────────────────────────────────────────────────────────────
""")
    print(f"  done in {time.time()-t0:.0f}s")
    print("=" * 96)


if __name__ == '__main__':
    main()
