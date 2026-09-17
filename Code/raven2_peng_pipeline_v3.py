
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] prep | Shared module: forward kinematics, model, and loss
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
RAVEN-II Peng UW Pipeline — v3

Fixes from v2 to v3:
  [BUG FIX] The input features were never normalised.
            jpos is around 50, torque around 0.1, a factor of 500 apart,
            which saturates the LSTM gates (pre-activation near +/-7) and
            makes torque invisible. That explains AUC 0.5, a dead attack
            head, and poor regression. Fix: per-channel StandardScaler on X.

  [FIX 2]   The scaler is fit on the training set only. v2 computed

            std_delta over everything, which leaks.

  [FIX 3]   The regression loss is computed on clean samples only (masked),
            so the regression head learns undisturbed trajectories and the
            classification head learns physical inconsistency, and the two
            tasks stop interfering through the shared trunk. Evaluation

            reports them separately.

  [FIX 4]   Input statistics are printed before and after normalisation, so
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from peng_uw_loader import PengUWLoader


class Config:
    DATA_ROOT = _DATA
    OUT_DIR   = _OUT + '/output_peng_pipeline_v3'
    RECORDINGS = ['record_1_different_directions']
    MAX_FILES_PER_REC   = 8
    SUBSAMPLE           = 5
    MAX_SAMPLES_PER_CSV = 20000
    LOOKBACK  = 30
    LOOKAHEAD = 20
    EPOCHS = 50
    BATCH  = 128
    LR     = 1e-3
    HIDDEN = 128
    ATTACK_FRAC  = 0.30
    THRESHOLD_MM = 5.0     # Peng is calibration data with ~35 mm motion; 1 mm is meaningless


# === Forward kinematics ===
LA12 = np.radians(75.0); LA23 = np.radians(52.0); D4 = -458.69

def raven2_fk_left(jpos):
    J0, J1, J2 = jpos[..., 0], jpos[..., 1], jpos[..., 2]
    th1 = np.radians(J0 + 205.0); th2 = np.radians(J1 + 180.0); d3 = J2
    g1, g2 = np.sin(LA12), np.cos(LA12)
    g3, g4 = np.sin(LA23), np.cos(LA23)
    d = d3 + D4
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    px = d*(c1*s2*g3 + s1*(c2*g2*g3 - g1*g4))
    py = d*(s1*s2*g3 - c1*(c2*g2*g3 - g1*g4))
    pz = d*(-(c2*g1*g3 + g2*g4))
    return np.stack([px, py, pz], axis=-1)


def build_raven2_graph():
    edges, etypes = [], []
    for i, j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)]:
        edges += [(i,j),(j,i)]; etypes += ['serial']*2
    edges += [(0,2),(2,0)]; etypes += ['cable']*2
    return torch.tensor(edges, dtype=torch.long).t().contiguous(), etypes


class AttackInjector:
    """Perturb jpos (the sensor reading) and leave torque (physical) alone.
    That asymmetry is where the detection signal comes from."""
    TYPES = ['step', 'ramp', 'noise']

    @staticmethod
    def inject(jpos_window):
        T = jpos_window.shape[0]
        out = jpos_window.copy()
        atype = np.random.choice(AttackInjector.TYPES)
        t_start = np.random.randint(T // 3, 2 * T // 3)
        joint = np.random.randint(3)

        if atype == 'step':
            mag = np.random.uniform(5, 15) if joint < 2 else np.random.uniform(2, 8)
            out[t_start:, joint] += mag * np.random.choice([-1, 1])
        elif atype == 'ramp':
            slope = np.random.uniform(0.3, 0.7) if joint < 2 else np.random.uniform(0.15, 0.35)
            slope *= np.random.choice([-1, 1])
            out[t_start:, joint] += slope * np.arange(T - t_start)
        elif atype == 'noise':
            sd = 2.5 if joint < 2 else 1.0
            out[t_start:, joint] += np.random.normal(0, sd, size=T - t_start)
        return out.astype(np.float32), atype


def build_sequences(records, lookback=30, lookahead=20, attack_fraction=0.30, seed=42):
    """Return raw sequences; normalisation happens after the split to avoid leakage"""
    rng = np.random.RandomState(seed)
    X, Yd, Ya = [], [], []

    for data in records:
        jpos = data['robot_jpos']                  # (N, 8)
        torq = data['motor_torque']                # (N, 8)
        xyz  = raven2_fk_left(jpos[:, :3])         # (N, 3) mm, from CLEAN jpos

        N = len(jpos)
        if N < lookback + lookahead + 1:
            continue

        for i in range(lookback, N - lookahead):
            is_attack = rng.rand() < attack_fraction
            jpos_win = jpos[i - lookback : i].copy()
            torq_win = torq[i - lookback : i].copy()
            if is_attack:
                jpos_win, _ = AttackInjector.inject(jpos_win)

            X.append(np.concatenate([jpos_win, torq_win], axis=1))   # (LB, 16)
            Yd.append(xyz[i : i + lookahead] - xyz[i - 1])           # (LA, 3) mm
            Ya.append(float(is_attack))

    return (np.stack(X).astype(np.float32),
            np.stack(Yd).astype(np.float32),
            np.array(Ya, dtype=np.float32))


class DeviationDetector(nn.Module):
    def __init__(self, input_dim=16, hidden=128, num_joints=7, lookahead=20):
        super().__init__()
        self.num_joints, self.lookahead, self.hidden = num_joints, lookahead, hidden
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.joint_proj = nn.Linear(hidden, num_joints * hidden)
        self.gcn1 = GCNConv(hidden, hidden); self.gcn2 = GCNConv(hidden, hidden)
        self.gn1 = nn.LayerNorm(hidden);     self.gn2 = nn.LayerNorm(hidden)

        self.head_dev = nn.Sequential(
            nn.Linear(num_joints * hidden, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, lookahead * 3),
        )
        nn.init.zeros_(self.head_dev[-1].weight)   # start from predicting zero change
        nn.init.zeros_(self.head_dev[-1].bias)

        self.head_atk = nn.Sequential(
            nn.Linear(num_joints * hidden, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x, edge_index):
        B = x.size(0)
        lstm_out, _ = self.lstm(x)
        h = lstm_out[:, -1, :]
        nf = self.joint_proj(h).view(B, self.num_joints, -1).reshape(B * self.num_joints, -1)
        offsets = torch.arange(B, device=x.device) * self.num_joints
        ei_b = torch.cat([edge_index + off for off in offsets], dim=1)
        nf = F.relu(self.gn1(self.gcn1(nf, ei_b)))
        nf = F.relu(self.gn2(self.gcn2(nf, ei_b)))
        nf = nf.view(B, -1)
        return self.head_dev(nf).view(B, self.lookahead, 3), self.head_atk(nf).squeeze(-1)


# === [FIX 3] Masked loss: regression on clean samples only ===
class MaskedMultiTaskLoss(nn.Module):
    def __init__(self, w_dev=1.0, w_atk=0.5, w_vel=0.1):
        super().__init__()
        self.w_dev, self.w_atk, self.w_vel = w_dev, w_atk, w_vel
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, dev_p, atk_p, dev_t, atk_t):
        clean = (atk_t < 0.5)
        if clean.any():
            dp, dt = dev_p[clean], dev_t[clean]
            l_dev = F.mse_loss(dp, dt)
            l_vel = F.mse_loss(dp[:,1:] - dp[:,:-1], dt[:,1:] - dt[:,:-1])
        else:
            l_dev = dev_p.sum() * 0.0
            l_vel = dev_p.sum() * 0.0
        l_atk = self.bce(atk_p, atk_t)          # classification uses every sample
        total = self.w_dev * l_dev + self.w_vel * l_vel + self.w_atk * l_atk
        return total, {'dev': float(l_dev), 'vel': float(l_vel), 'atk': float(l_atk)}


def train_model(model, edge_index, X_tr, Yd_tr, Ya_tr, X_va, Yd_va, Ya_va,
                std_delta, out_dir, epochs=50, batch=128, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device=='cuda' else ''))

    model, edge_index = model.to(device), edge_index.to(device)
    std_t = torch.from_numpy(std_delta).to(device)
    criterion = MaskedMultiTaskLoss(w_dev=1.0, w_atk=0.5, w_vel=0.1)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(X_tr),
                                       torch.from_numpy(Yd_tr),
                                       torch.from_numpy(Ya_tr)),
        batch_size=batch, shuffle=True, drop_last=True,
        num_workers=0, pin_memory=(device == 'cuda'))

    Xv  = torch.from_numpy(X_va).to(device)
    Ydv = torch.from_numpy(Yd_va).to(device)
    Yav = torch.from_numpy(Ya_va).to(device)
    clean_mask = (Ya_va < 0.5)

    # Naive baseline, on clean validation samples only, matching the regression
    naive_rmse = float(np.sqrt(((Yd_va[clean_mask] * std_delta) ** 2).mean()) * np.sqrt(3))
    print(f"  Naive baseline (zero change, clean, Euclidean) = {naive_rmse:.3f} mm")

    best = float('inf'); history = []
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'Ep':>4}  {'L_dev':>8} {'L_atk':>8} {'RMSE_mm':>9} {'Skill':>15} {'AUC':>14}")
    print("-" * 66)

    for ep in range(1, epochs + 1):
        model.train()
        for bx, byd, bya in loader:
            bx  = bx.to(device, non_blocking=True)
            byd = byd.to(device, non_blocking=True)
            bya = bya.to(device, non_blocking=True)
            opt.zero_grad()
            dev_p, atk_p = model(bx, edge_index)
            loss, _ = criterion(dev_p, atk_p, byd, bya)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            dvp, avp = model(Xv, edge_index)
            _, comp = criterion(dvp, avp, Ydv, Yav)
            # Regression accuracy: clean samples only, Euclidean mm
            dvp_mm = (dvp * std_t).cpu().numpy()[clean_mask]
            ydv_mm = (Ydv * std_t).cpu().numpy()[clean_mask]
            euc = np.linalg.norm(dvp_mm - ydv_mm, axis=-1)
            dev_rmse = float(np.sqrt((euc ** 2).mean()))
            skill = 1 - dev_rmse / naive_rmse
            atk_prob = torch.sigmoid(avp).cpu().numpy()
            try:    atk_auc = roc_auc_score(Ya_va, atk_prob)
            except ValueError: atk_auc = 0.5

        history.append((ep, comp['dev'], comp['atk'], dev_rmse, skill, atk_auc))
        score = comp['dev'] + 0.5 * comp['atk']
        if score < best:
            best = score
            torch.save(model.state_dict(), f'{out_dir}/best_model.pt')

        if ep % 5 == 0 or ep == 1:
            sc = '\033[92m' if skill > 0.3 else '\033[93m' if skill > 0 else '\033[91m'
            ac = '\033[92m' if atk_auc > 0.85 else '\033[93m' if atk_auc > 0.65 else '\033[91m'
            print(f"{ep:>4}  {comp['dev']:>8.4f} {comp['atk']:>8.4f} {dev_rmse:>9.3f}"
                  f"  {sc}{skill:+.3f}\033[0m         {ac}{atk_auc:.3f}\033[0m")

    print("-" * 66)

    h = np.array(history)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(h[:,0], h[:,1], 'b-', label='L_dev (clean only)')
    axes[0].plot(h[:,0], h[:,2], 'r-', label='L_atk (BCE)')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
    axes[0].set_title('Masked multi-task loss'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(h[:,0], h[:,3], 'b-', lw=2)
    axes[1].axhline(naive_rmse, color='red', ls='--', label=f'Naive ({naive_rmse:.1f}mm)')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('RMSE (mm)')
    axes[1].set_title('Deviation prediction (clean)'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(h[:,0], h[:,5], 'g-', lw=2)
    axes[2].axhline(0.5, color='gray', ls=':')
    axes[2].axhline(0.85, color='green', ls=':', alpha=0.6, label='target')
    axes[2].set_ylim(0.4, 1.02); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('AUC')
    axes[2].set_title('Attack detection'); axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f'{out_dir}/training_curves.png', dpi=150); plt.close()
    return naive_rmse


def evaluate_model(model, edge_index, X_va, Yd_va, Ya_va, std_delta,
                   out_dir, threshold_mm=5.0, naive_rmse=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval(); std_t = torch.from_numpy(std_delta).to(device)
    with torch.no_grad():
        dev_p, atk_p = model(torch.from_numpy(X_va).to(device), edge_index.to(device))
        dev_pred_mm = (dev_p * std_t).cpu().numpy()
        atk_prob    = torch.sigmoid(atk_p).cpu().numpy()

    Yd_mm = Yd_va * std_delta
    clean = Ya_va < 0.5

    err  = np.linalg.norm(dev_pred_mm - Yd_mm, axis=-1)
    err_c = err[clean]
    model_rmse = float(np.sqrt((err_c ** 2).mean()))
    if naive_rmse is None:
        naive_rmse = float(np.sqrt((np.linalg.norm(Yd_mm[clean], axis=-1) ** 2).mean()))
    skill = 1 - model_rmse / naive_rmse
    max_dev = err_c[:, :10].max(axis=1)

    print(f"\n  -- Deviation regression (clean samples, real mm) --")
    print(f"    Model RMSE:          {model_rmse:.3f} mm")
    print(f"    Naive baseline:      {naive_rmse:.3f} mm")
    c = '\033[92m' if skill > 0.3 else '\033[93m' if skill > 0 else '\033[91m'
    print(f"    Skill score:         {c}{skill:+.4f}\033[0m")
    print(f"    Mean max deviation:  {max_dev.mean():.3f} mm")
    print(f"    95th percentile:     {np.percentile(max_dev, 95):.3f} mm")
    print(f"    Alert rate (>{threshold_mm}mm):  {100*(max_dev > threshold_mm).mean():.2f}%")

    try:    auc = roc_auc_score(Ya_va, atk_prob)
    except ValueError: auc = float('nan')
    pred = (atk_prob > 0.5).astype(np.float32)
    p, r, f1, _ = precision_recall_fscore_support(Ya_va, pred, average='binary', zero_division=0)
    ac = '\033[92m' if auc > 0.85 else '\033[93m' if auc > 0.65 else '\033[91m'
    print(f"\n  -- Attack detection (all samples) --")
    print(f"    ROC-AUC:     {ac}{auc:.4f}\033[0m")
    print(f"    Precision:   {p:.4f}")
    print(f"    Recall:      {r:.4f}")
    print(f"    F1:          {f1:.4f}")

    np.savez(f'{out_dir}/eval.npz',
             dev_pred_mm=dev_pred_mm, dev_true_mm=Yd_mm, atk_prob=atk_prob,
             atk_true=Ya_va, max_dev=max_dev, model_rmse=model_rmse,
             naive_rmse=naive_rmse, skill=skill, auc=auc,
             precision=p, recall=r, f1=f1)

    # -- Figures --
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f'Peng UW v3 — clean RMSE {model_rmse:.2f}mm / naive {naive_rmse:.2f}mm '
                 f'/ skill {skill:+.3f} / AUC {auc:.3f}', fontsize=12, fontweight='bold')

    ax = axes[0,0]
    tm = np.linalg.norm(Yd_mm[clean], axis=-1).max(axis=1)
    pm = np.linalg.norm(dev_pred_mm[clean], axis=-1).max(axis=1)
    ax.scatter(tm, pm, s=6, alpha=0.35, color='#2563EB')
    mx = max(tm.max(), pm.max()); ax.plot([0,mx],[0,mx],'k--',lw=1,alpha=0.5)
    ax.set_xlabel('True max Δ (mm)'); ax.set_ylabel('Predicted max Δ (mm)')
    ax.set_title('Regression on clean samples'); ax.grid(alpha=0.3)

    ax = axes[0,1]
    ax.hist(atk_prob[Ya_va == 0], bins=40, alpha=0.6, label='Clean',  color='#3B82F6')
    ax.hist(atk_prob[Ya_va == 1], bins=40, alpha=0.6, label='Attack', color='#DC2626')
    ax.axvline(0.5, color='black', ls='--')
    ax.set_xlabel('P(attack)'); ax.set_ylabel('Count')
    ax.set_title('Attack score separation'); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1,0]
    fpr, tpr, _ = roc_curve(Ya_va, atk_prob)
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC = {auc:.3f}')
    ax.plot([0,1],[0,1],'k--',alpha=0.5)
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC')
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1,1]
    ci = np.where(clean)[0]
    idx = ci[int(np.argmax(max_dev))]
    t = np.arange(dev_pred_mm.shape[1])
    for k, (lab, col) in enumerate(zip('XYZ', ['#2196F3','#4CAF50','#F44336'])):
        ax.plot(t, Yd_mm[idx,:,k],       '-',  color=col, lw=2,   label=f'True Δ{lab}')
        ax.plot(t, dev_pred_mm[idx,:,k], '--', color=col, lw=1.4, label=f'Pred Δ{lab}')
    ax.set_title('Worst clean sample', fontsize=10)
    ax.set_xlabel('Future frame'); ax.set_ylabel('Δ position (mm)')
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)

    plt.tight_layout(); plt.savefig(f'{out_dir}/evaluation.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    cfg = Config(); os.makedirs(cfg.OUT_DIR, exist_ok=True)
    print("=" * 68)
    print(" RAVEN-II Peng UW Pipeline — v3 (input normalization fix)")
    print("=" * 68)

    print("\n[1] Loading data")
    records = []
    for rec in cfg.RECORDINGS:
        d = os.path.join(cfg.DATA_ROOT, rec)
        if not os.path.exists(d):
            print(f"  skipping {rec}"); continue
        print(f"\n  ── {rec} ──")
        records.extend(PengUWLoader.load_directory(
            d, subsample=cfg.SUBSAMPLE, max_files=cfg.MAX_FILES_PER_REC,
            max_samples_per_file=cfg.MAX_SAMPLES_PER_CSV, verbose=True))

    print(f"\n[2] Building sequences")
    X, Yd_mm, Ya = build_sequences(records, cfg.LOOKBACK, cfg.LOOKAHEAD, cfg.ATTACK_FRAC)
    print(f"  samples: {len(X):,}   attack: {Ya.mean()*100:.1f}%")

    print(f"\n[3] Time-split (80/20)")
    n = len(X); s = int(0.8 * n)
    X_tr_raw, X_va_raw = X[:s], X[s:]
    Yd_tr_mm, Yd_va_mm = Yd_mm[:s], Yd_mm[s:]
    Ya_tr,    Ya_va    = Ya[:s],    Ya[s:]
    print(f"  train: {len(X_tr_raw):,}   val: {len(X_va_raw):,}")
    print(f"  attack fraction in validation: {Ya_va.mean()*100:.1f}%")

    # === [BUG FIX] Normalise the input features, fit on training only ===
    print(f"\n[4] Input normalisation, the step v2 was missing")
    flat = X_tr_raw.reshape(-1, X_tr_raw.shape[-1])
    mu_x = flat.mean(axis=0).astype(np.float32)
    sd_x = np.maximum(flat.std(axis=0), 1e-6).astype(np.float32)
    print(f"  before (first 3 jpos channels):   mean={mu_x[:3]}, std={sd_x[:3]}")
    print(f"  before (first 3 torque channels): mean={mu_x[8:11]}, std={sd_x[8:11]}")
    print(f"  -> a factor of {sd_x[:3].mean() / max(sd_x[8:11].mean(), 1e-9):.0f} apart, "
          f"which is why the torque signal was invisible")

    X_tr = ((X_tr_raw - mu_x) / sd_x).astype(np.float32)
    X_va = ((X_va_raw - mu_x) / sd_x).astype(np.float32)
    print(f"  after: X_tr mean={X_tr.mean():+.4f}  std={X_tr.std():.4f}   (should be 0 / 1)")

    # === Target normalisation, also fit on training only ===
    std_delta = np.maximum(Yd_tr_mm.reshape(-1,3).std(axis=0), 1e-3).astype(np.float32)
    Yd_tr = (Yd_tr_mm / std_delta[None, None, :]).astype(np.float32)
    Yd_va = (Yd_va_mm / std_delta[None, None, :]).astype(np.float32)
    print(f"  Δxyz per-axis std (mm): {std_delta}")

    print(f"\n[5] Model")
    edge_index, etypes = build_raven2_graph()
    model = DeviationDetector(input_dim=X.shape[-1], hidden=cfg.HIDDEN,
                              num_joints=7, lookahead=cfg.LOOKAHEAD)
    print(f"  graph: 7 nodes, {edge_index.shape[1]} edges "
          f"({etypes.count('serial')} serial + {etypes.count('cable')} cable)")
    print(f"  parameters: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n[6] Training (epochs={cfg.EPOCHS})")
    naive_rmse = train_model(model, edge_index, X_tr, Yd_tr, Ya_tr,
                             X_va, Yd_va, Ya_va, std_delta,
                             out_dir=cfg.OUT_DIR, epochs=cfg.EPOCHS,
                             batch=cfg.BATCH, lr=cfg.LR)

    print("\n[7] Evaluation")
    model.load_state_dict(torch.load(f'{cfg.OUT_DIR}/best_model.pt'))
    evaluate_model(model, edge_index, X_va, Yd_va, Ya_va, std_delta,
                   out_dir=cfg.OUT_DIR, threshold_mm=cfg.THRESHOLD_MM,
                   naive_rmse=naive_rmse)

    print("\n[8] Inference speed")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    dummy = torch.randn(1, cfg.LOOKBACK, X.shape[-1]).to(device)
    ei = edge_index.to(device)
    with torch.no_grad():
        for _ in range(200): model(dummy, ei)
    if device == 'cuda': torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(1000): model(dummy, ei)
    if device == 'cuda': torch.cuda.synchronize()
    print(f"  inference latency: {(time.perf_counter()-t0):.4f} ms")

    print("\n" + "=" * 68)
    print(f"  done. Output: {cfg.OUT_DIR}")
    print("=" * 68)


if __name__ == '__main__':
    main()
