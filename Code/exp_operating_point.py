
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ3 | Learned-detector operating points under a fixed alarm budget
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Clinically usable operating points: detection floors under a fixed
false-alarm budget

Problem: AUC >= 0.9 has no meaning in an operating room. It is a
threshold-free ranking measure and answers neither "how large an attack slips
past once a threshold is set" nor "how often does it alarm during a
procedure".

Two precedents:
  Peng et al. 2020 (ICRA)      report RMS error in mm against a 1 mm
                               requirement for semi-autonomous tasks. A
                               physical quantity against a required threshold,
                               verifiable and interpretable. But that is a
                               regression metric and this is discrimination,
                               so it cannot be carried over directly.
  Alemzadeh et al. 2016 (DSN)  take the threshold from the 99.8 to 99.9
                               percentile of 600 fault-free runs, that is,
                               fix the false-alarm budget first and then
                               report detection. That can be carried over.

The main metric here becomes:
    the smallest tip deviation reaching 90% detection at a given number of
    false alarms per hour

-- Three design decisions ----------------------------------------

  [D1] Alarms are counted over non-overlapping windows
       A 0.227 s window gives 3600/0.227 = 15,850 non-overlapping windows per
       hour. NOTE: a pre-submission check found this framing understates the
       problem rather than being conservative; see exp_window_accounting.py
       and patch_operating_point.py.

  [D2] The clinical target is one alarm per hour, which cannot be estimated
       1/hour -> per-window FPR = 1/15850 = 6.3e-5, and estimating that
       quantile needs at least 1/FPR ~ 16,000 clean validation windows, while
       only about 8,700 exist. So directly reachable operating points are
       reported as measured, and stricter ones were extrapolated with a
       generalised Pareto tail fit, marked as extrapolated with bootstrap
       intervals. NOTE: that extrapolation proved inconsistent and the patch
       disables it.

  [D3] The main output is millimetres
       Each operating point gives the smallest tip deviation reaching 90%
       detection, comparable against 1 mm. AUC is kept as a methodological
       baseline, since the paper's relative conclusions use it, but is no
       longer the main metric.

-- Training-loop bugs fixed in v2 ---------------------------------

v1 copied run_one's training loop and dropped two things, with serious
consequences:

  [B1] The regression target was not normalised
       The main experiment divides by std_delta; v1 used raw millimetres. YD
       has a standard deviation near 12 mm, so after squaring the regression
       loss is about 200 times the BCE.

  [B2] Loss weights and the W_VEL term were inconsistent
       Main experiment: W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1. v1 had only two
       terms and unverified weights.

  Together the classification gradient was about 1/570 of the regression
  gradient. With a weak signal the classification head learns nothing,
  converges to predicting one class, and AUC lands at exactly 0.5000. That is
  precisely what v1 showed: ten consecutive 0.5000 values at the low end of
  step, identical across three seeds.

  Fix: stop copying the training loop and import MaskedMultiTaskLoss and the
  normalisation from the main experiment. Copying structure is what caused
  this.

Dependency: per-window scores and labels are needed. run_one in
exp_paper_sweep returns aggregates only, so this script carries its own
training loop, structurally copied from it.

-- Diagnosing the total collapse of step in v3 --------------------

After v2 fixed pos_weight, ramp behaved normally, but all eight step
magnitudes across three seeds returned exactly 0.5000.

Two explanations ruled out:
  injection missing from features   measured: step on joint 0 moves JP by
                                    0.449 and the features move by 0.449
  weaker signal                     step's magnitude is larger than ramp's
                                    (0.449 against 0.294)

The real cause is the training magnitude distribution against the noise floor:
  step injects mag = uniform(5,15) * scale, a single impulse per frame.
  TRAIN_RANGE for step was (0.0008, 0.15), log-uniform, so the median scale is
  about 0.011, giving a median magnitude near 0.11 deg while std(dj) = 0.17.
  About 30% of the "attacked" training samples therefore inject below the
  per-frame noise floor and are simply inseparable. Faced with many
  inseparable positives, the optimal strategy is to predict the negative class
  everywhere, which gives exactly 0.5000.

  Ramp differs: at the same scale the slope accumulates over 14 frames, so the
  effective magnitude is far larger and enough separable positives exist.

  The test grid also failed to reach the top of the training distribution:
  step trained up to 0.15 (dev_end 1.5) while testing stopped at 0.30.

Fixes:
  [S1] raise the low end of step training from 0.0008 to 0.008, putting the
       median magnitude near 0.35 deg, twice std(dj)
  [S2] extend the step test grid up to 0.15, aligned with training

-- Three further fixes in v4 --------------------------------------

v3 results:
    step  100/hr 0.662 mm   10/hr 0.896 mm   2/hr 0.953 mm  (flat, usable)
    ramp  none at every FPR
    noise none at every FPR

  [F1] The grids for ramp and noise did not reach high enough
       At the top of the ramp grid (scale 0.60, dev_end 4.34) AUC is 0.98, yet
       90% detection is unreachable at a fixed FPR. Even a 4.34 deg injection
       is not enough under the operating-point metric, so the grid must extend
       upward: ramp to 1.5 (dev_end about 10.8), noise to 0.05 (dev_rms about
       0.125).

  [F2] The debounce implementation was wrong and has been removed
       It swept for k consecutive threshold crossings within sc[y>0.5], but
       boolean indexing leaves the attacked windows non-contiguous in time, so
       "consecutive" is meaningless on that array. Symptom: ramp reported TPR
       0.000 at k=1, contradicting AUC 0.98 in the same configuration, and
       step fell from 1.000 to 0.001. Doing it correctly requires keeping the
       window indices and time order, which the current storage does not.
       Debouncing is not central here, so removing it beats keeping a table
       nobody can explain.

  [F3] The step results are reused
       Three seeds by eight magnitudes are already done and sensible.

Usage:    python3 exp_operating_point.py
Expected: 3 families x 8 magnitudes x 3 seeds = 72 runs, 2 to 3 hours
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
from raven2_peng_pipeline_v3 import build_raven2_graph
from exp_delta_ablation import (Cfg as BaseCfg, load_records, fit_physics_model,
                                build_window_cache, build_features)
from exp_paper_sweep import SweepInjector
try:
    from exp_delta_ablation import MaskedMultiTaskLoss
except ImportError:
    from raven2_peng_pipeline_v3 import WeightedMaskedLoss as MaskedMultiTaskLoss


class Cfg(BaseCfg):
    OUT_DIR=_OUT + '/output_operating_point'
    MODE='jpos_torque_delta'
    SEEDS=[42,43,44]
    EPOCHS=120; LR=3e-4; BATCH=128; HIDDEN=128
    # [B2] Matching the main experiment
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1

    FAMILIES=['step','ramp','noise']
    JOINTS=dict(step=[0], ramp=[0,1], noise=[0])
    # [S1] step low end 0.0008 -> 0.008: the old median magnitude of 0.11 deg
    # sits below std(dj) = 0.17, leaving many inseparable positives.
    # Training range aligned with the top of the test grid.
    TRAIN_RANGE=dict(step=(0.008,0.15), ramp=(0.030,1.50), noise=(0.002,0.05))
    # [S2] step test grid extended to 0.15, aligned with training
    # [F1] ramp and noise extended upward: at v3's top end, 90% detection
    # was still unreachable, so the grid did not bracket the crossing.
    TEST_SCALES=dict(
        step =[0.15,0.10,0.06,0.03,0.020,0.013,0.009,0.006],
        ramp =[1.50,1.00,0.70,0.45,0.30,0.20,0.13,0.08],
        noise=[0.050,0.032,0.020,0.013,0.008,0.005,0.003,0.002])

    # Windows and timing
    WINDOW_S = 30/132.0                 # 0.227 s
    WIN_PER_HOUR = 3600.0/WINDOW_S      # 15850

    # Operating points: false alarms per hour
    ALARMS_PER_HOUR = [100.0, 50.0, 20.0, 10.0, 5.0, 2.0, 1.0]
    TPR_TARGET = 0.90
    MM_PER_DEG = 7.47
    N_BOOT = 500


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


def inject(JP,cfg,mask,fam,scale=None,seed_off=0):
    rng=np.random.RandomState(cfg.SEED+seed_off+(int(scale*100000) if scale else 0))
    out=JP.copy(); y=np.zeros(len(JP),np.float32); de=np.zeros(len(JP),np.float32)
    dr=np.zeros(len(JP),np.float32)
    lo,hi=cfg.TRAIN_RANGE[fam]; pool=np.where(mask)[0]
    for i in rng.permutation(pool)[:int(len(pool)*cfg.ATTACK_FRAC)]:
        sc=scale if scale is not None else float(np.exp(rng.uniform(np.log(lo),np.log(hi))))
        j=cfg.JOINTS[fam][rng.randint(len(cfg.JOINTS[fam]))]
        out[i],_,_,de[i],dr[i]=SweepInjector.inject(JP[i],scale=sc,rng=rng,
                                                    force_type=fam,force_joint=j)
        y[i]=1.0
    m=y>0.5
    return out,y,(float(de[m].mean()) if m.any() else 0.0,
                  float(dr[m].mean()) if m.any() else 0.0)


def train_scores(X,YD,y,GID,cfg,seed):
    """Train once and return per-window validation scores and labels.
    Per-window scores are what this script needs and run_one does not give."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev=cfg.DEVICE
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva
    f=X[mtr].reshape(-1,X.shape[-1])
    mu=f.mean(0).astype(np.float32); sd=np.maximum(f.std(0),1e-6).astype(np.float32)
    Xtr=((X[mtr]-mu)/sd).astype(np.float32); Xva=((X[mva]-mu)/sd).astype(np.float32)
    ytr,yva=y[mtr],y[mva]

    # [B1] Normalise the regression target, as the main experiment does.
    # Without it YD's std of about 12 mm makes the squared regression loss
    # roughly 200x the BCE, starving the classification head, giving AUC 0.5000.
    std_delta = YD[mtr].std() + 1e-8
    Ydtr = (YD[mtr] / std_delta).astype(np.float32)

    m=Det(X.shape[-1],cfg.HIDDEN,cfg.LOOKAHEAD).to(dev)
    # [B3] pos_weight must be a tensor. BCEWithLogitsLoss registers it as a
    # buffer, and passing a float raises a buffer assignment error.
    # v1 used a tensor; changing it to a float while fixing B1 and B2 made the
    # loss fail to construct, so training never ran and AUC came out 0.5000.
    pw=torch.tensor(float((ytr<0.5).sum()/max((ytr>=0.5).sum(),1)),
                    dtype=torch.float32, device=dev)
    # [B2] Use the main experiment's loss class rather than copying it
    lossf=MaskedMultiTaskLoss(cfg.W_DEV,cfg.W_ATK,cfg.W_VEL,pos_weight=pw).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=cfg.LR,weight_decay=0.0)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=cfg.EPOCHS)
    Xt=torch.from_numpy(Xtr).to(dev); Yt=torch.from_numpy(ytr).to(dev)
    Dt=torch.from_numpy(Ydtr).to(dev); Xv=torch.from_numpy(Xva).to(dev)
    for ep in range(cfg.EPOCHS):
        m.train(); perm=torch.randperm(len(Xt),device=dev)
        for s0 in range(0,len(Xt),cfg.BATCH):
            i=perm[s0:s0+cfg.BATCH]; opt.zero_grad()
            dp,ap=m(Xt[i])
            out=lossf(dp,ap,Dt[i],Yt[i])
            loss=out[0] if isinstance(out,(tuple,list)) else out
            loss.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        sch.step()
    m.eval()
    with torch.no_grad(): _,ap=m(Xv)
    sc=torch.sigmoid(ap).cpu().numpy()
    del m,Xt,Yt,Dt,Xv
    if dev=='cuda': torch.cuda.empty_cache()
    return sc, yva, mva


# ══════════════════════════════════════════════════════════════
#  [D1][D2] Operating-point analysis
# ══════════════════════════════════════════════════════════════
def tpr_at_fpr(scores, y, fpr, tail_fit=True):
    """
    Set the threshold from the clean-score quantile at a target FPR.
    Returns (TPR, threshold, extrapolated); falls back to a GPD tail fit.
    """
    clean = scores[y < 0.5]; atk = scores[y > 0.5]
    n = len(clean)
    if n == 0 or len(atk) == 0: return np.nan, np.nan, False
    k_needed = fpr * n
    if k_needed >= 1.0:                       # directly estimable
        thr = np.quantile(clean, 1 - fpr)
        return float((atk > thr).mean()), float(thr), False
    if not tail_fit: return np.nan, np.nan, True
    # GPD extrapolation: fit a generalised Pareto to exceedances over the 95th
    u = np.quantile(clean, 0.95)
    exc = clean[clean > u] - u
    if len(exc) < 30: return np.nan, np.nan, True
    # Method of moments, robust and needs no optimiser
    m1, v = exc.mean(), exc.var()
    if v <= 0: return np.nan, np.nan, True
    xi = 0.5*(1 - m1**2/v); beta = 0.5*m1*(m1**2/v + 1)
    p_exc = 0.05                              # P(X > u)
    q = fpr / p_exc
    if abs(xi) < 1e-8: thr = u - beta*np.log(q)
    else:
        if q <= 0: return np.nan, np.nan, True
        thr = u + beta/xi * (q**(-xi) - 1)
    thr = min(thr, 1.0)                       # sigmoid output upper bound
    return float((atk > thr).mean()), float(thr), True


def floor_at_operating_point(curve, target=0.90):
    """curve: [(x_mm, tpr), ...] -> smallest x reaching tpr>=target, log-interpolated"""
    c = sorted([p for p in curve if np.isfinite(p[1])])
    if len(c) < 2: return np.inf
    xs = np.array([p[0] for p in c]); ys = np.array([p[1] for p in c])
    for k in range(len(xs)-1):
        if (ys[k]-target)*(ys[k+1]-target) <= 0 and ys[k] != ys[k+1]:
            t = (target-ys[k])/(ys[k+1]-ys[k])
            lo, hi = max(xs[k],1e-12), max(xs[k+1],1e-12)
            return float(np.exp(np.log(lo)+t*(np.log(hi)-np.log(lo))))
    return float(xs[0]) if ys[0] >= target else np.inf




def main():
    cfg=Cfg(); os.makedirs(cfg.OUT_DIR,exist_ok=True); t0=time.time()
    print("="*100); print(" Operating points: detection floors at a fixed alarm budget"); print("="*100)
    print(f"\n  window {cfg.WINDOW_S*1000:.0f} ms -> {cfg.WIN_PER_HOUR:.0f} non-overlapping windows per hour")
    print(f"  target detection rate {cfg.TPR_TARGET:.0%}")
    print(f"\n  {'alarms/hour':>13}{'per-window FPR':>16}{'clean windows needed':>22}")
    print("-"*100)
    for a in cfg.ALARMS_PER_HOUR:
        f=a/cfg.WIN_PER_HOUR
        print(f"  {a:>12.0f}{f:>14.2e}{1/f:>15.0f}")

    recs=load_records(cfg)
    a_,b_,r2,_,_=fit_physics_model(recs,cfg.N_POS_JOINTS)
    JP,TQ,YD,GID=build_window_cache(recs,cfg)
    edge,_=build_raven2_graph()
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva
    n_clean_va=int((mva).sum()*(1-cfg.ATTACK_FRAC))
    # Training magnitude distribution against the per-frame noise floor: basis for [S1]
    print(f"\n  training magnitudes vs noise floor std(dj[0]) = {np.diff(JP[:,:,0],axis=1).std():.4f} deg")
    print(f"  {'family':>8}{'scale range':>20}{'median dev_end':>17}{'/ std(dj)':>12}")
    print("-"*100)
    _sdj = float(np.diff(JP[:,:,0],axis=1).std())
    for _f in cfg.FAMILIES:
        _lo,_hi = cfg.TRAIN_RANGE[_f]
        _med = float(np.exp((np.log(_lo)+np.log(_hi))/2))
        _dev = _med*10.0 if _f=='step' else (_med*7.0 if _f=='ramp' else _med*2.5)
        print(f"  {_f:>8}{f'[{_lo}, {_hi}]':>20}{_dev:>15.4f}{_dev/_sdj:>12.2f}")
    print(f"  A ratio below 1 means most training samples sit below the noise floor.")

    print(f"\n  validation {mva.sum():,} windows, about {n_clean_va:,} clean")
    print(f"  strictest directly estimable FPR = 1/{n_clean_va} = {1/max(n_clean_va,1):.2e}"
          f"  -> {1/max(n_clean_va,1)*cfg.WIN_PER_HOUR:.1f} alarms/hour")
    print(f"  Stricter points are GPD-extrapolated and marked with *")

    ck=f'{cfg.OUT_DIR}/scores.json'
    store=json.load(open(ck)) if os.path.exists(ck) else []
    seen={(r['family'],r['scale'],r['seed']) for r in store}
    todo=[(f_,s_,sd_) for f_ in cfg.FAMILIES for s_ in cfg.TEST_SCALES[f_]
          for sd_ in cfg.SEEDS if (f_,s_,sd_) not in seen]
    print(f"\n  {len(todo)} training runs to go\n")

    for n,(fam,scale,seed) in enumerate(todo,1):
        JPtr,ytr,_=inject(JP,cfg,mtr,fam,None,0)
        JPva,yva_,dv=inject(JP,cfg,mva,fam,scale,7777)
        JPm=JP.copy(); JPm[mtr]=JPtr[mtr]; JPm[mva]=JPva[mva]
        yy=np.zeros(len(JP),np.float32); yy[mtr]=ytr[mtr]; yy[mva]=yva_[mva]
        X=build_features(JPm,TQ,cfg.MODE,a_,b_,cfg.N_POS_JOINTS)
        sc,yv,_=train_scores(X,YD,yy,GID,cfg,seed)
        auc=roc_auc_score(yv,sc) if len(np.unique(yv))>1 else 0.5
        store.append(dict(family=fam,scale=float(scale),seed=seed,
                          dev_end=dv[0],dev_rms=dv[1],auc=float(auc),
                          scores=[float(v) for v in sc],
                          labels=[int(v) for v in yv]))
        tmp=ck+'.tmp'
        with open(tmp,'w') as fh: json.dump(store,fh)
        os.replace(tmp,ck)
        print(f"  [{n}/{len(todo)}] {fam:<6} sc={scale:<8.5f} "
              f"dev={dv[0]:<8.4f} s={seed} AUC={auc:.4f}  ({time.time()-t0:.0f}s)")
        del X,JPtr,JPva,JPm

    # == Analysis ==
    print("\n"+"="*100); print(" Detection floor per operating point (mm at tip)"); print("="*100)
    XK=dict(step='dev_end',ramp='dev_end',noise='dev_rms')
    results={}
    for fam in cfg.FAMILIES:
        print(f"\n  ── {fam} ──")
        print(f"  {'alarms/hour':>13}{'FPR':>11}{'floor (deg)':>14}{'floor (mm)':>13}"
              f"{'extrap?':>9}{'vs 1mm':>10}")
        print("-"*100)
        for aph in cfg.ALARMS_PER_HOUR:
            fpr=aph/cfg.WIN_PER_HOUR
            curve=[]; extrap=False
            byscale={}
            for r in store:
                if r['family']==fam: byscale.setdefault(r['scale'],[]).append(r)
            for scale,rs in byscale.items():
                tprs=[]
                for r in rs:
                    t,thr,ex=tpr_at_fpr(np.array(r['scores']),np.array(r['labels']),fpr)
                    if np.isfinite(t): tprs.append(t); extrap=extrap or ex
                if tprs:
                    x=np.mean([r[XK[fam]] for r in rs])
                    curve.append((x,float(np.mean(tprs))))
            fl=floor_at_operating_point(curve,cfg.TPR_TARGET)
            mm=fl*cfg.MM_PER_DEG if np.isfinite(fl) else np.nan
            results[(fam,aph)]=dict(deg=fl if np.isfinite(fl) else None,
                                    mm=mm if np.isfinite(mm) else None,
                                    extrapolated=bool(extrap))
            print(f"  {aph:>11.0f}{fpr:>11.2e}"
                  f"{(f'{fl:.5f}' if np.isfinite(fl) else 'none'):>13}"
                  f"{(f'{mm:.3f}' if np.isfinite(mm) else 'none'):>12}"
                  f"{('yes*' if extrap else 'no'):>8}"
                  f"{(f'{mm:.1f}x' if np.isfinite(mm) else '-'):>10}")

    print("\n"+"="*100); print(" Against the AUC baseline"); print("="*100)
    AUC_FLOOR=dict(step=0.0190,ramp=0.1740,noise=0.0041)
    print(f"\n  {'family':>8}{'AUC>=0.9 (mm)':>16}{'10/hr (mm)':>13}"
          f"{'1/hr (mm)':>12}{'cost':>10}")
    print("-"*100)
    for fam in cfg.FAMILIES:
        a10=results.get((fam,10.0),{}).get('mm')
        a1 =results.get((fam,1.0),{}).get('mm')
        base=AUC_FLOOR[fam]*cfg.MM_PER_DEG
        cost=a1/base if (a1 and base>0) else np.nan
        print(f"  {fam:>8}{base:>16.3f}"
              f"{(f'{a10:.3f}' if a10 else 'none'):>13}"
              f"{(f'{a1:.3f}' if a1 else 'none'):>12}"
              f"{(f'{cost:.1f}x' if np.isfinite(cost) else '-'):>10}")
    print(f"\n  AUC>=0.9 is a threshold-free ranking measure and matches no budget.")
    print(f"  A clinically usable point is nearer one alarm per hour; see the cost column.")


    json.dump({f'{k[0]}|{k[1]}':v for k,v in results.items()},
              open(f'{cfg.OUT_DIR}/operating_points.json','w'),indent=2)
    print(f"\n  output: {cfg.OUT_DIR}/operating_points.json")
    print(f"  total {time.time()-t0:.0f}s")
    print("="*100)


if __name__=='__main__':
    main()
