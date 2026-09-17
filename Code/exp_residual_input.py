
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ2 | Physics-residual floor for observation-path injection (0.81 mm)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
[R] The physics residual as an input channel: measured rather than argued

One of the companion paper's central claims:

    as an input       the residual is an exact linear combination of two
                      existing channels, therefore redundant, no gain
    as a constraint   the same relation helps, lowering the ramp floor by 19%

The second half was measured directly (exp_physics_loss with three placebos).
The first half was not.

What was actually measured is that a GLR on F1 (step 0.109) loses to a GLR on
F3 (0.077). That compares two hand-designed detectors; it does not test
whether feeding the residual to a network helps. Using it to support a claim
about a learned model is indirect evidence presented as direct.

The 'jpos_torque_residual' branch in build_features has always existed but was
never run:
    res = dj - (a * tq_lag + b)      # same causal alignment as the fit
This script runs it against the 'jpos_torque_delta' mode in use.

-- Three design decisions ----------------------------------------

  [D1] Only the input mode changes; everything else is held fixed
       Same injector, magnitude grid, session split, training loop and seed.
       The single independent variable is build_features' mode argument, so
       the two floors are directly comparable.

  [D2] Report the parameter difference honestly
       The residual mode adds 3 channels (19 to 22) and 3 columns to the LSTM
       input layer. Small but not zero, so both dimensions and parameter
       counts are printed at start-up.

  [D3] Run all three patterns
       The claim is general ("no gain as an input"), so one pattern is not
       enough. A gain on only one pattern would itself be a finer conclusion.

Reading the result:
  residual ~ delta      the redundancy claim holds, upgraded from argument
                        to measurement
  residual clearly better   the claim is wrong and the companion paper's
                        central comparison needs rewriting
  residual clearly worse    stronger than "no gain": either the extra channels
                        dilute the signal, or the residual's own noise floor
                        (R2 only 0.56 to 0.81) adds variance

-- v1 curves and the grid adjustment in v2 ------------------------

The three v1 curves disagreed in direction and two grids failed to bracket
the crossing:

  step   both modes sat above AUC 0.95 even at the lowest point, grid too high
  ramp   bracketed; the residual is 0.03 to 0.06 lower everywhere,
         floor 0.894 -> 1.265
  noise  delta flat at 0.50 with no trend; the residual climbs monotonically
         to 0.84 without reaching 0.9

  The noise pair is the important one: adding the residual moves it from
  entirely undetectable to nearly detectable. That contradicts "redundant as
  an input", and runs opposite to ramp.

  A self-consistent explanation: noise injection is an independent per-frame
  perturbation, which breaks exactly the per-frame correspondence between dq
  and tau that the residual computes explicitly, so the injection is exposed.
  Ramp injection is a sustained slope offset, already directly visible in dq,
  while the residual adds its own 52 to 66% noise floor and dilutes it. The
  residual is not redundant; it trades one noise structure for another,
  favouring per-frame attacks and penalising cumulative ones.

  v2 extends two grids:
    step  down to 0.002 (dev ~ 0.02) so the curve falls below 0.9
    noise up to 0.15   (dev ~ 0.37) so the residual curve crosses 0.9
  ramp is unchanged; v1 already bracketed it.

Usage:    python3 exp_residual_input.py
Expected: 2 modes x 3 patterns x 6 magnitudes x 3 seeds = 108 runs, 3-4 hours
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
from exp_delta_ablation import (Cfg as BaseCfg, load_records, fit_physics_model,
                                build_window_cache, build_features,
                                MaskedMultiTaskLoss)
from exp_paper_sweep import SweepInjector, detection_limit


class Cfg(BaseCfg):
    OUT_DIR=_OUT + '/output_residual_input'
    SEEDS=[42,43,44]
    EPOCHS=120; LR=3e-4; BATCH=128; HIDDEN=128
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1
    MODES=['jpos_torque_delta', 'jpos_torque_residual']   # [D1] the only variable
    FAMILIES=['step','ramp','noise']
    JOINTS=dict(step=[0], ramp=[0,1], noise=[0])
    TRAIN_RANGE=dict(step=(0.008,0.15), ramp=(0.010,0.60), noise=(0.0003,0.03))
    # v2: extend step downward and noise upward; ramp already brackets
    SCALES=dict(step =[0.030,0.013,0.009,0.006,0.004,0.002],
                ramp =[0.60,0.25,0.10,0.06,0.04,0.025],
                noise=[0.15,0.10,0.06,0.035,0.02,0.008])
    AUC_TARGET=0.90


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
    out=JP.copy(); y=np.zeros(len(JP),np.float32)
    de=np.zeros(len(JP),np.float32); dr=np.zeros(len(JP),np.float32)
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
    with torch.no_grad(): _,ap=m(Xv)
    sc=torch.sigmoid(ap).cpu().numpy()
    auc=roc_auc_score(yva,sc) if len(np.unique(yva))>1 else 0.5
    npar=sum(p.numel() for p in m.parameters())
    del m,Xt,Yt,Dt,Xv
    if dev=='cuda': torch.cuda.empty_cache()
    return float(auc), int(npar)


def main():
    cfg=Cfg(); os.makedirs(cfg.OUT_DIR,exist_ok=True); t0=time.time()
    print("="*100); print(" [R] The physics residual as an input channel"); print("="*100)
    recs=load_records(cfg)
    a_,b_,r2,_,_=fit_physics_model(recs,cfg.N_POS_JOINTS)
    JP,TQ,YD,GID=build_window_cache(recs,cfg)
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva

    print(f"\n  [D2] Dimensions and parameter counts for both modes")
    print(f"  {'mode':<26}{'dim':>6}{'params':>12}")
    print("-"*100)
    dims={}
    for md in cfg.MODES:
        Xc=build_features(JP[:8],TQ[:8],md,a_,b_,cfg.N_POS_JOINTS)
        d=Xc.shape[-1]; dims[md]=d
        mt=Det(d,cfg.HIDDEN,cfg.LOOKAHEAD)
        print(f"  {md:<26}{d:>6}{sum(p.numel() for p in mt.parameters()):>12,}")
        del mt,Xc
    print(f"\n  residual mode adds {dims[cfg.MODES[1]]-dims[cfg.MODES[0]]} channels")
    print(f"  physics model R2 = {np.round(r2,4)}  (lower R2 means a noisier residual)")

    ck=f'{cfg.OUT_DIR}/results.json'
    rows=json.load(open(ck)) if os.path.exists(ck) else []
    seen={(r['mode'],r['family'],r['scale'],r['seed']) for r in rows}
    todo=[(m_,f_,s_,d_) for m_ in cfg.MODES for f_ in cfg.FAMILIES
          for s_ in cfg.SCALES[f_] for d_ in cfg.SEEDS
          if (m_,f_,s_,d_) not in seen]
    print(f"\n  {len(todo)} runs to go\n")

    XK=dict(step=0,ramp=0,noise=1)
    for n,(md,fam,scale,seed) in enumerate(todo,1):
        JPtr,ytr,_=inject(JP,cfg,mtr,fam,None,0)
        JPva,yva_,dv=inject(JP,cfg,mva,fam,scale,7777)
        JPm=JP.copy(); JPm[mtr]=JPtr[mtr]; JPm[mva]=JPva[mva]
        yy=np.zeros(len(JP),np.float32); yy[mtr]=ytr[mtr]; yy[mva]=yva_[mva]
        X=build_features(JPm,TQ,md,a_,b_,cfg.N_POS_JOINTS)
        auc,npar=train_eval(X,YD,yy,GID,cfg,seed)
        rows.append(dict(mode=md,family=fam,scale=float(scale),seed=seed,
                         x=dv[XK[fam]],auc=auc,dim=X.shape[-1],params=npar))
        tmp=ck+'.tmp'
        with open(tmp,'w') as fh: json.dump(rows,fh)
        os.replace(tmp,ck)
        print(f"  [{n}/{len(todo)}] {md:<24}{fam:<6}sc={scale:<8.5f}"
              f"s={seed} AUC={auc:.4f}  ({time.time()-t0:.0f}s)")
        del X,JPtr,JPva,JPm

    def floor(md,fam):
        d={}
        for r in rows:
            if r['mode']==md and r['family']==fam: d.setdefault(r['scale'],[]).append(r)
        if len(d)<2: return np.inf
        pts=sorted((np.mean([q['x'] for q in v]),np.mean([q['auc'] for q in v]))
                   for v in d.values())
        return detection_limit([p[0] for p in pts],[p[1] for p in pts],cfg.AUC_TARGET)

    print("\n"+"="*100); print(" Detection floors (deg)"); print("="*100)
    print(f"\n  {'waveform':>10}{'delta (19ch)':>16}{'residual (22ch)':>18}{'ratio':>10}")
    print("-"*100)
    rs=[]
    for fam in cfg.FAMILIES:
        f0,f1=floor(cfg.MODES[0],fam), floor(cfg.MODES[1],fam)
        r_=f1/f0 if np.isfinite(f0) and np.isfinite(f1) and f0>0 else np.nan
        if np.isfinite(r_): rs.append(r_)
        print(f"  {fam:>10}{(f'{f0:.4f}' if np.isfinite(f0) else 'none'):>16}"
              f"{(f'{f1:.4f}' if np.isfinite(f1) else 'none'):>18}"
              f"{(f'{r_:.2f}x' if np.isfinite(r_) else '-'):>10}")

    print("\n"+"="*100); print(" Reading the result"); print("="*100)
    if rs:
        mr=float(np.mean(rs))
        print(f"\n  mean ratio {mr:.2f}x   (above 1 means the residual hurts)\n")
        if abs(mr-1.0)<0.10:
            print("  -> no gain from the residual as an input, matching the claim.")
            print("     The claim moves from an information argument to a measurement.")
        elif mr<0.90:
            print("  -> the residual does help as an input. The redundancy claim is")
            print("     wrong and the useless-as-input framing needs rewriting.")
        else:
            print("  -> the residual clearly hurts. That is stronger than no gain and")
            print(f"     needs explaining: the extra channels may dilute the signal, or")
            print(f"     the residual's own noise floor (R2 {np.round(r2,2)}) adds variance.")
    print("="*100)


if __name__=='__main__':
    main()
