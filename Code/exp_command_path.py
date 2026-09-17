
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ4 | Command-stream floors (step 0.0389 deg; ramp and noise none)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Command-path detection: preemption with a data-driven method

Both existing paths report after the fact:
    fast path   waits for telemetry to come back
    slow path   waits for hourly aggregation
On an irreversible action that matters: the servo completes in 9 to 26 ms
while a human visual response takes hundreds of milliseconds.

Alemzadeh et al. (DSN 2016) can preempt because their monitor sits before the
DAC write. That is a choice of signal path, not of method. This experiment
moves a learned detector to the same position:

    q^d sequence -> learned model -> hold the command if anomalous

The difference from them is the method (data-driven against a second-order
dynamic model), not the placement.

-- Three design decisions ----------------------------------------

  [D1] Strictly causal: the input contains no q from after the injection
       The input is q^d[t-L..t] together with q[t-L..t-1], the previous frame
       and earlier. It cannot contain q[t], which only exists once the command
       has executed; including it would not be preemptive.
       This is the fundamental difference from the other experiments: the fast
       path reads the reported position after execution, this reads the
       command before it.

  [D2] The criterion is temporal self-consistency of the command stream
       An adversary edits q^d, but q^d changes anyway. So the only available
       criterion is whether this command is consistent with the operator's
       recent motion, which is structurally what the observation path does.
       Injection model: step, ramp and noise added to q^d using the same
       SweepInjector as the observation-path experiments, so the floors stay
       comparable.

  [D3] Interception latency must be reported
       Preemption is worth nothing unless inference completes before the DAC
       write. RAVEN-II's control period is 1 ms, so inference has to fit
       inside it. The script reports per-window inference time at batch size
       one, matching deployment, rather than batched throughput.

-- Comparison with the observation-path experiment ----------------

  Same architecture, same injector, same criterion, only the input changes:
      observation path  jpos(8) + torque(8) + delta(3),        injected on jpos
      command path      jpos_d(8) + jpos_lag(8) + delta_d(3),  injected on jpos_d

  So the two floors compare directly, and the comparison is itself a result:
    command floor lower    preemption is feasible and more sensitive
    higher                 preemption costs something, a trade-off
    comparable             the placement decides only whether interception is
                           possible, not how sensitive detection is

-- Window alignment (fixed in v2) ---------------------------------

v1 cut 94,331 command windows from 28 CSVs while build_window_cache produced
only 46,205 observation windows from a different file subset. v1 truncated to
align, but the two sets came from different data, so the regression target and
the session split were both misaligned.

v2 no longer borrows the observation path's target. It computes one from the
command windows themselves: tip displacement obtained by passing the lagged
measured q through forward kinematics. Input, target and split then all come
from the same windows, and the target obeys the causal constraint in [D1],
using only measurements from t-1 and earlier.

Cost: the target's absolute scale differs from the observation-path
experiment, so regression skill is not comparable between them. Detection
floors remain comparable, since those depend only on the classification head
and the shared injector.

-- v2 results: three patterns, three shapes -----------------------

    step   0.0600 -> 0.9718   even the lowest point is well above 0.9;
                              the grid does not reach low enough
    ramp   1.2622 -> 0.8812   flat from here on, still 0.8833 at 4.3814;
                              a 3.5x increase in magnitude buys 0.002, a ceiling
    noise  0.495 to 0.505     unchanged across a 40x range, no signal at all

  The none results for ramp and noise are real, not a grid artifact:

    ramp   Detecting a ramp on the command path means asking whether a slowly
           drifting command came from a person. Operators adjust posture
           slowly and their hands drift, so the injection is statistically
           inseparable from that. More magnitude does not help, because large
           slow movements are normal operation.
    noise  The high-frequency content of q^d is already filled by hand tremor,
           so added high frequency is invisible. The observation path differs:
           the servo low-passes the reported position, suppressing high
           frequency, which is what makes injected jitter stand out.

  One mechanism explains both: the prior on the command path is what a human
  can generate. An instantaneous step violates it, since human motion
  bandwidth is a few hertz, so it is easy to detect; slow drift and
  high-frequency jitter both fall inside normal human behaviour and are not.

  v3 extends only the low end for step (the crossing sits near dev 0.01);
  ramp and noise are not re-run.

Usage:    python3 exp_command_path.py
Expected: 3 patterns x 8 magnitudes x 3 seeds = 72 runs, 2.5 to 3 hours
"""
import os, sys, json, time
import numpy as np, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score

# ══════════════════════════════════════════════════════════════
#  [WD] weight_decay must be zero
#
#  The optimizer in run_one (exp_paper_sweep) is:
#      torch.optim.Adam(model.parameters(), lr=cfg.LR)
#  with no weight_decay. Copying the loop and adding 1e-4 collapsed the
#  noise family to AUC 0.50 while step and ramp were unaffected.
#
#  Bisect, one term at a time (noise, dev_rms=0.049, same X/y/seed):
#      wd=1e-4  sched=True   clip=True    0.5065   <- the broken config
#      wd=0     sched=True   clip=True    0.9999   <- weight decay removed
#      wd=1e-4  sched=False  clip=True    0.5100   <- scheduler irrelevant
#      wd=1e-4  sched=True   clip=False   0.5065   <- clipping irrelevant
#      wd=0     sched=False  clip=False   0.9998   <- run_one's configuration
#
#  Why only noise is affected: its signature is a variance change and a weak
#  one (dev_rms 0.049 against std(dj) 0.172, just 28%). Detecting it requires
#  sensitivity to the input's second-order statistics, which needs weights of
#  a certain magnitude. Weight decay suppresses that over 120 epochs. Step
#  and ramp are mean shifts and far stronger, so earlier checks missed it.
#
#  Lesson: regularisation added while copying a training loop can silently
#  kill an entire class of signal. Match the reference implementation verbatim.
# ══════════════════════════════════════════════════════════════

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_delta_ablation import Cfg as BaseCfg, MaskedMultiTaskLoss
from exp_paper_sweep import SweepInjector, detection_limit
from peng_uw_loader import COL


class Cfg(BaseCfg):
    OUT_DIR=_OUT + '/output_command_path'
    SEEDS=[42,43,44]
    EPOCHS=120; LR=3e-4; BATCH=128; HIDDEN=128
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1

    FAMILIES=['step','ramp','noise']
    JOINTS=dict(step=[0], ramp=[0,1], noise=[0])
    TRAIN_RANGE=dict(step=(0.008,0.15), ramp=(0.010,0.60), noise=(0.0003,0.03))
    TEST_SCALES=dict(
        # v3: extend step downward. v2's lowest point (dev 0.060) still gave
        # AUC 0.972; the slope puts the crossing near dev 0.01.
        step =[0.006,0.004,0.0025,0.0015,0.0009,0.0005],
        ramp =[0.60,0.40,0.25,0.174,0.10,0.06,0.04,0.025],
        noise=[0.02,0.012,0.008,0.005,0.003,0.002,0.001,0.0005])
    AUC_TARGET=0.90
    MM_PER_DEG=7.47
    CONTROL_BUDGET_MS=1.0

    # Observation-path floors, for the final comparison (from exp_paper_sweep)
    OBS_FLOORS=dict(step=0.0190, ramp=0.1740, noise=0.0041)

COLS_D = dict(jpos_d=COL.get('jpos_desired',(194,202)),
              jpos  =COL['robot_jpos'])


LA12=np.radians(75.0); LA23=np.radians(52.0); D4=-458.69

def fk_np(j3):
    """(...,3) joint angles in degrees -> (...,3) tip xyz in mm, King et al. Eq. 40"""
    J0,J1,J2 = j3[...,0], j3[...,1], j3[...,2]
    th1=np.radians(J0+205.0); th2=np.radians(J1+180.0); d3=J2
    g1,g2=np.sin(LA12),np.cos(LA12); g3,g4=np.sin(LA23),np.cos(LA23)
    d=d3+D4; c1,s1=np.cos(th1),np.sin(th1); c2,s2=np.cos(th2),np.sin(th2)
    return np.stack([d*(c1*s2*g3+s1*(c2*g2*g3-g1*g4)),
                     d*(s1*s2*g3-c1*(c2*g2*g3-g1*g4)),
                     d*(-(c2*g1*g3+g2*g4))], axis=-1)


def build_targets(QL, lookahead, n_pos=3):
    """
    Regression target: tip displacement over the next lookahead steps.
    Built only from QL (lagged measurements), so it carries no information
    from time t, consistent with [D1]. Shape (N, lookahead, 3), in mm.
    """
    p = fk_np(QL[:,:,:n_pos].astype(np.float64))          # (N,T,3)
    T = p.shape[1]
    la = min(lookahead, T-1)
    # Displacement ahead of the last window frame; pad with the last frame
    base = p[:, -la-1, :][:, None, :]
    tgt = (p[:, -la:, :] - base).astype(np.float32)
    if la < lookahead:
        pad = np.repeat(tgt[:, -1:, :], lookahead-la, axis=1)
        tgt = np.concatenate([tgt, pad], axis=1)
    return tgt


class Det(nn.Module):
    def __init__(s,d,h,la):
        super().__init__(); s.la=la
        s.lstm=nn.LSTM(d,h,2,batch_first=True,dropout=0.2)
        s.dev=nn.Sequential(nn.Linear(h,256),nn.ReLU(),nn.Dropout(0.1),
                            nn.Linear(256,64),nn.ReLU(),nn.Linear(64,la*3))
        nn.init.zeros_(s.dev[-1].weight); nn.init.zeros_(s.dev[-1].bias)
        s.atk=nn.Sequential(nn.Linear(h,128),nn.ReLU(),nn.Dropout(0.2),
                            nn.Linear(128,32),nn.ReLU(),nn.Linear(32,1))
    def forward(s,x):
        h=s.lstm(x)[0][:,-1,:]
        return s.dev(h).view(-1,s.la,3), s.atk(h).squeeze(-1)


def build_cmd_windows(cfg):
    """
    [D1] Strictly causal command-path windows.
    Input at window t:
        q^d[t-L+1 .. t]      commands, including the frame under judgement
        q [t-L   .. t-1]     measurements, lagged one frame (not yet executed)
        delta_d              first difference of q^d

    Read straight from CSV rather than through load_records, which does not
    keep the raw matrices. Windowing matches build_window_cache exactly (same
    SUBSAMPLE, LOOKBACK, STRIDE and traversal order), or the windows will not align.
    """
    import glob, pandas as pd
    QD_w, QL_w, GID = [], [], []
    LB=cfg.LOOKBACK; ST=getattr(cfg,'STRIDE',5); SS=getattr(cfg,'SUBSAMPLE',5)
    lo_d,hi_d = COLS_D['jpos_d']; lo_q,hi_q = COLS_D['jpos']
    need = sorted(set(range(lo_d,hi_d)) | set(range(lo_q,hi_q)))
    root = getattr(cfg,'DATA_ROOT',_DATA)
    rec_dirs = getattr(cfg,'RECORDS',['record_1_different_directions'])
    files=[]
    for d in rec_dirs:
        files += sorted(glob.glob(f'{root}/{d}/*.csv'))
    if not files:
        return None, None, None
    print(f"  reading command channels from {len(files)} CSVs (cols {lo_d}:{hi_d} and {lo_q}:{hi_q})")
    for gi, path in enumerate(files):
        parts=[]
        for ch in pd.read_csv(path, header=None, usecols=need, chunksize=50000):
            parts.append(ch.iloc[::SS])
        df = pd.concat(parts, ignore_index=True); del parts
        qd = df[list(range(lo_d,hi_d))].values.astype(np.float32)
        q  = df[list(range(lo_q,hi_q))].values.astype(np.float32)
        del df
        n = min(len(qd), len(q))
        for i in range(LB+1, n, ST):
            QD_w.append(qd[i-LB:i])        # includes the current frame
            QL_w.append(q [i-LB-1:i-1])    # lagged by one frame
            GID.append(gi)
    return (np.asarray(QD_w,np.float32), np.asarray(QL_w,np.float32),
            np.asarray(GID))


def build_input(QD, QL, n_pos=3):
    """q^d(8) + q_lag(8) + delta_d(3) = 19 dims, matching the observation path"""
    dd = np.diff(QD[:,:,:n_pos], axis=1)
    dd = np.concatenate([np.zeros((len(QD),1,n_pos),np.float32), dd], axis=1)
    return np.concatenate([QD, QL, dd], axis=-1).astype(np.float32)


def inject(QD,cfg,mask,fam,scale=None,seed_off=0):
    """[D2] Inject on q^d using the same SweepInjector as the observation path"""
    rng=np.random.RandomState(cfg.SEED+seed_off+(int(scale*100000) if scale else 0))
    out=QD.copy(); y=np.zeros(len(QD),np.float32)
    de=np.zeros(len(QD),np.float32); dr=np.zeros(len(QD),np.float32)
    lo,hi=cfg.TRAIN_RANGE[fam]; pool=np.where(mask)[0]
    for i in rng.permutation(pool)[:int(len(pool)*cfg.ATTACK_FRAC)]:
        sc=scale if scale is not None else float(np.exp(rng.uniform(np.log(lo),np.log(hi))))
        j=cfg.JOINTS[fam][rng.randint(len(cfg.JOINTS[fam]))]
        out[i],_,_,de[i],dr[i]=SweepInjector.inject(QD[i],scale=sc,rng=rng,
                                                    force_type=fam,force_joint=j)
        y[i]=1.0
    m=y>0.5
    return out,y,(float(de[m].mean()) if m.any() else 0.0,
                  float(dr[m].mean()) if m.any() else 0.0)


def train_eval(X,YD,y,GID,cfg,seed):
    torch.manual_seed(seed); np.random.seed(seed); dev=cfg.DEVICE
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva
    f=X[mtr].reshape(-1,X.shape[-1])
    mu=f.mean(0).astype(np.float32); sd=np.maximum(f.std(0),1e-6).astype(np.float32)
    Xtr=((X[mtr]-mu)/sd).astype(np.float32); Xva=((X[mva]-mu)/sd).astype(np.float32)
    ytr,yva=y[mtr],y[mva]
    std_delta=YD[mtr].std()+1e-8
    Ydtr=(YD[mtr]/std_delta).astype(np.float32)
    m=Det(X.shape[-1],cfg.HIDDEN,cfg.LOOKAHEAD).to(dev)
    pw=torch.tensor(float((ytr<0.5).sum()/max((ytr>=0.5).sum(),1)),
                    dtype=torch.float32,device=dev)
    lossf=MaskedMultiTaskLoss(cfg.W_DEV,cfg.W_ATK,cfg.W_VEL,pos_weight=pw).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.LR,weight_decay=0.0)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.EPOCHS)
    Xt=torch.from_numpy(Xtr).to(dev); Yt=torch.from_numpy(ytr).to(dev)
    Dt=torch.from_numpy(Ydtr).to(dev); Xv=torch.from_numpy(Xva).to(dev)
    for ep in range(cfg.EPOCHS):
        m.train(); perm=torch.randperm(len(Xt),device=dev)
        for s0 in range(0,len(Xt),cfg.BATCH):
            i=perm[s0:s0+cfg.BATCH]; opt.zero_grad()
            dp,ap=m(Xt[i]); loss=lossf(dp,ap,Dt[i],Yt[i])
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step()
    m.eval()
    # [D3] Deployment-style latency: batch=1, one command at a time
    with torch.no_grad():
        for _ in range(20): m(Xv[:1])            # warm-up
        if dev=='cuda': torch.cuda.synchronize()
        t0=time.time()
        for k in range(200): m(Xv[k:k+1])
        if dev=='cuda': torch.cuda.synchronize()
        lat=(time.time()-t0)/200*1000
        _,ap=m(Xv)
    sc=torch.sigmoid(ap).cpu().numpy()
    auc=roc_auc_score(yva,sc) if len(np.unique(yva))>1 else 0.5
    del m,Xt,Yt,Dt,Xv
    if dev=='cuda': torch.cuda.empty_cache()
    return float(auc), float(lat)


def main():
    cfg=Cfg(); os.makedirs(cfg.OUT_DIR,exist_ok=True); t0=time.time()
    print("="*100); print(" Command-path detection: data-driven preemption"); print("="*100)
    print(f"""
  [D1] strictly causal input: q^d[t-L+1..t] + q[t-L..t-1] + delta_d
       no q[t], which exists only after execution; including it is not preemptive
  [D2] injection on q^d with the same SweepInjector, so floors are comparable
  [D3] per-command inference latency at batch=1, against a {cfg.CONTROL_BUDGET_MS} ms period
""")
    QD,QL,GID=build_cmd_windows(cfg)
    if QD is None:
        print("  records carries no raw matrix, so jpos_desired cannot be read.")
        print("     Either keep rec['raw'] in load_records, or read the CSV directly.")
        sys.exit(1)
    print(f"  command windows {QD.shape}  lagged {QL.shape}  sessions {len(np.unique(GID))}")
    # Channel sanity: q^d must carry independent content, or the experiment is void
    qd_f = QD.reshape(-1, QD.shape[-1]); q_f = QL.reshape(-1, QL.shape[-1])
    print(f"  q^d per-channel std: {qd_f.std(0)[:4].round(4)} ...")
    print(f"  q   per-channel std: {q_f.std(0)[:4].round(4)} ...")
    n_const = int((qd_f.std(0) < 1e-9).sum())
    same = float(np.abs(qd_f[:, :3] - q_f[:, :3]).max())
    print(f"  constant q^d channels {n_const}/{QD.shape[-1]};  max elementwise gap to q {same:.4f}")
    if n_const == QD.shape[-1] or same < 1e-9:
        print("  q^d has no independent content (constant, or identical to q). Aborting.")
        sys.exit(1)

    # The regression target is computed from the command windows themselves,
    # not borrowed from the observation path, whose windows come from a different subset.
    YD = build_targets(QL, cfg.LOOKAHEAD, cfg.N_POS_JOINTS)
    print(f"  regression target {YD.shape}  (QL through FK, in mm)")
    print(f"  target per-axis std: {YD.reshape(-1,3).std(0).round(3)}")

    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva

    ck=f'{cfg.OUT_DIR}/results.json'
    rows=json.load(open(ck)) if os.path.exists(ck) else []
    seen={(r['family'],r['scale'],r['seed']) for r in rows}
    todo=[(f_,s_,d_) for f_ in cfg.FAMILIES for s_ in cfg.TEST_SCALES[f_]
          for d_ in cfg.SEEDS if (f_,s_,d_) not in seen]
    print(f"  {len(todo)} runs to go\n")

    XK=dict(step=0,ramp=0,noise=1)
    for k,(fam,scale,seed) in enumerate(todo,1):
        QDtr,ytr,_=inject(QD,cfg,mtr,fam,None,0)
        QDva,yva_,dv=inject(QD,cfg,mva,fam,scale,7777)
        QDm=QD.copy(); QDm[mtr]=QDtr[mtr]; QDm[mva]=QDva[mva]
        yy=np.zeros(len(QD),np.float32); yy[mtr]=ytr[mtr]; yy[mva]=yva_[mva]
        X=build_input(QDm,QL,cfg.N_POS_JOINTS)
        auc,lat=train_eval(X,YD,yy,GID,cfg,seed)
        rows.append(dict(family=fam,scale=float(scale),seed=seed,
                         x=dv[XK[fam]],auc=auc,latency_ms=lat))
        tmp=ck+'.tmp'
        with open(tmp,'w') as fh: json.dump(rows,fh)
        os.replace(tmp,ck)
        print(f"  [{k}/{len(todo)}] {fam:<6} sc={scale:<8.5f} dev={dv[XK[fam]]:<8.4f} "
              f"s={seed} AUC={auc:.4f} lat={lat:.3f}ms  ({time.time()-t0:.0f}s)")
        del X,QDtr,QDva,QDm

    # == Summary ==
    def floor(fam):
        d={}
        for r in rows:
            if r['family']==fam: d.setdefault(r['scale'],[]).append(r)
        if len(d)<2: return np.inf
        pts=sorted((np.mean([q['x'] for q in v]),np.mean([q['auc'] for q in v]))
                   for v in d.values())
        return detection_limit([p[0] for p in pts],[p[1] for p in pts],cfg.AUC_TARGET)

    print("\n"+"="*100); print(" Command path against observation path"); print("="*100)
    print(f"\n  {'waveform':>10}{'command (deg)':>18}{'observation (deg)':>20}"
          f"{'cmd/obs':>11}{'tip (mm)':>11}")
    print("-"*100)
    for fam in cfg.FAMILIES:
        fc=floor(fam); fo=cfg.OBS_FLOORS[fam]
        r_=fc/fo if np.isfinite(fc) and fo>0 else np.nan
        print(f"  {fam:>10}{(f'{fc:.4f}' if np.isfinite(fc) else 'none'):>18}"
              f"{fo:>18.4f}{(f'{r_:.2f}x' if np.isfinite(r_) else '-'):>11}"
              f"{(fc*cfg.MM_PER_DEG if np.isfinite(fc) else np.nan):>11.3f}")

    lats=[r['latency_ms'] for r in rows]
    print(f"\n  per-command inference latency (batch=1): median {np.median(lats):.3f} ms, "
          f"p95 {np.percentile(lats,95):.3f} ms")
    print(f"  control-period budget: {cfg.CONTROL_BUDGET_MS} ms")
    ok = np.percentile(lats,95) < cfg.CONTROL_BUDGET_MS
    print(f"  -> {'within budget, preemption is feasible' if ok else 'over budget, cannot intercept'}")

    print("\n"+"="*100); print(" Reading the result"); print("="*100)
    print("""
  cmd/obs < 1   the command path is more sensitive: preemption costs nothing
  cmd/obs > 1   preemption costs sensitivity, to be weighed against interception
  cmd/obs ~ 1   the placement decides interception only, not sensitivity

  In every case the latency line is a hard constraint: beyond the control
  period, the claim can only be earlier detection, not preemption.
""")
    print("="*100)


if __name__=='__main__':
    main()
