
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ1 | Cross-load transfer test (slope 6%, offset 0.121 deg)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Held-out validation of the canary hypothesis: does the calibration transfer
across load conditions?

What probe_canary.py found:
  [3] loaded / idle = 3.95 to 6.42     -> load-driven, thermal drift ruled out
  [2] span / within-hour SD = 16 to 31 -> drift far exceeds fluctuation
  [Q1] motors 4/6/7 perfectly monotone, 5 static -> several signals with
       paired signs, which looks like a real mechanical mode
  [Q3] r(pairdiff, gt0) = -0.995       -> suspect
  [5] slow-path floor 0.00110 deg      -> wrong

Two things had to be corrected.

  [E1] The r = -0.995 is confounded by a shared time trend.
       pair-diff against hour has rho = +1.00, gt0 against hour rho = -1.00.
       Two sequences perfectly monotone in time necessarily give |r| near 1
       over six points. That is arithmetic, not evidence of a usable
       relation; any two monotone sequences give the same number.

  [E2] The slow-path floor was off by about 200x.
       The original used |slope| * 3 * SE with SE = sd/sqrt(n), n = 4526.
       Two errors: 4526 is not the number of independent samples, since the
       motor-position difference is heavily autocorrelated; and more
       fundamentally the wrong error source was chosen. Even with a perfectly
       precise pair-diff, the floor on estimating gt0 is the calibration
       residual of 0.07538 deg, because that is the scatter of gt0 about the
       fitted line. An honest floor is about 3 * 0.07538 = 0.226 deg. And
       0.07538 is itself in-sample, fitted and evaluated on the same six
       points, so it is optimistic.

  Two questions the original script conflated:
       can we tell that drift is happening    <- pair-diff noise, easy
       can we estimate the joint error        <- calibration residual, 0.226 deg
       Security monitoring needs the second.

This script does three things.

  [T1] Correlation after detrending
       Remove the linear trend in hour from pair-diff and gt0 separately,
       then correlate the residuals. Correlation that survives means the
       relation is not only a shared trend. First differences are also
       reported as a stricter detrending.

  [T2] Calibration transfer across load conditions (the decisive test)
       Fit gt0 = a*pairdiff + b at 500 g, freeze the coefficients, apply to
       idle and unloaded, and compare predicted against measured gt0.
       One set of coefficients holding across all three means the relation is
       physical and the canary is usable. A different slope per condition
       means it was only a shared trend and the canary fails.
       Reports each condition's slope, held-out RMSE, and whether the slopes
       lie within each other's confidence intervals.

  [T3] Leave-one-out calibration residual
       LOO over the six hourly points gives an out-of-sample residual to
       replace the in-sample 0.07538. The slow-path floor is three times its
       standard deviation.

Usage: python3 probe_canary_transfer.py
"""
import os, sys, glob, gc, json, itertools
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, _DATA)

OUT=_OUT + '/output_canary'
CSV=f'{OUT}/by_hour.csv'
if not os.path.exists(CSV):
    print(f"{CSV} not found; run probe_canary.py first"); sys.exit(1)
df=pd.read_csv(CSV)

PAIRS=[(4,5),(4,6),(4,7),(5,6)]
COND={'500gload':'record_3_time_decay_500gload',
      'unloaded':'record_3_time_decay_unloaded',
      'idle'    :'record_3_time_decay_idle'}
sub={k:df[df.record==v].sort_values('hour') for k,v in COND.items()}
for k,v in sub.items():
    print(f"  {k:<10} {len(v)} hourly points")

print("\n"+"="*100)
print(" [T1] Correlation after detrending (is r=-0.995 only a shared trend?)")
print("="*100)

def detrend(x, t):
    """Remove the linear trend in t"""
    A=np.polyfit(t,x,1); return x-np.polyval(A,t)

L=sub['500gload']
if len(L)<4:
    print("  too few points at 500 g"); sys.exit(1)
t=L.hour.values.astype(float); y=L['gt0'].values
print(f"\n  {'pair':>8}{'raw r':>16}{'detrended r':>18}{'first-diff r':>18}")
print("-"*100)
t1={}
for (i,j) in PAIRS:
    x=L[f'p{i}{j}'].values
    r0,p0=pearsonr(x,y)
    xd,yd=detrend(x,t),detrend(y,t)
    if xd.std()<1e-12 or yd.std()<1e-12:
        rd,pd_=np.nan,np.nan
    else:
        rd,pd_=pearsonr(xd,yd)
    dx,dy=np.diff(x),np.diff(y)
    if dx.std()<1e-12 or dy.std()<1e-12:
        rf,pf=np.nan,np.nan
    else:
        rf,pf=pearsonr(dx,dy)
    t1[(i,j)]=(r0,rd,rf)
    print(f"  {f'({i},{j})':>8}  r={r0:>+6.3f} p={p0:>6.4f}"
          f"    r={rd:>+6.3f} p={pd_:>6.4f}"
          f"    r={rf:>+6.3f} p={pf:>6.4f}")
best_dt=max((abs(v[1]) for v in t1.values() if np.isfinite(v[1])), default=np.nan)
print(f"\n  largest |r| after detrending = {best_dt:.3f}")
print(f"  Reading: strong correlation surviving means more than a shared trend.")
print(f"        Correlation vanishing means r=-0.995 was arithmetic, nothing more.")

print("\n"+"="*100)
print(" [T2] Calibration transfer across load conditions (decisive)")
print("="*100)
print("\n  Fit gt0 = a*pairdiff + b at 500 g, freeze, apply to idle and unloaded.")

print(f"\n  {'pair':>8}{'condition':>12}{'a (own fit)':>14}{'b':>12}"
      f"{'in-RMSE':>10}{'extrapolated RMSE':>20}")
print("-"*100)
t2={}
for (i,j) in PAIRS:
    xk=f'p{i}{j}'
    if xk not in L: continue
    A_ref=np.polyfit(L[xk].values, L['gt0'].values, 1)
    slopes={}
    for cname,s in sub.items():
        if len(s)<3 or xk not in s: continue
        x,yy=s[xk].values, s['gt0'].values
        if x.std()<1e-12: continue
        A=np.polyfit(x,yy,1)
        in_rmse=float(np.sqrt(((yy-np.polyval(A,x))**2).mean()))
        xf_rmse=float(np.sqrt(((yy-np.polyval(A_ref,x))**2).mean()))
        slopes[cname]=A[0]
        mark=' (ref)' if cname=='500gload' else ''
        print(f"  {f'({i},{j})' if cname=='500gload' else '':>8}{cname+mark:>12}"
              f"{A[0]:>14.6f}{A[1]:>12.4f}{in_rmse:>10.4f}{xf_rmse:>20.4f}")
    if len(slopes)>=2:
        vals=list(slopes.values())
        spread=max(vals)/min(vals) if min(vals)!=0 else np.inf
        t2[(i,j)]=dict(slopes=slopes, ratio=float(spread))
        print(f"  {'':>8}{'slope spread':>12}{spread:>14.2f}")
    print()

print(f"  Reading: a slope spread near 1 with extrapolated RMSE comparable to")
print(f"        the in-sample fit means one physical relation, canary holds.")
print(f"        Slopes differing several-fold, or far worse RMSE, means the canary fails.")
ratios=[v['ratio'] for v in t2.values() if np.isfinite(v['ratio'])]
if ratios:
    print(f"  measured slope spread: min={min(ratios):.2f} max={max(ratios):.2f}")
    verdict_t2 = ('canary holds' if max(ratios)<2 else
                  'partly holds' if max(ratios)<5 else 'canary fails')
    print(f"  -> {verdict_t2}")
else:
    verdict_t2='undetermined'

print("\n"+"="*100)
print(" [T3] Leave-one-out calibration residual -> corrected slow-path floor")
print("="*100)
print("\n  The original reported 0.00110 deg using the SE of the pair-diff mean,")
print("  which is the wrong error source. Even with a precise pair-diff, the floor")
print("  on estimating gt0 is the scatter of the calibration relation itself.")
print(f"\n  {'pair':>8}{'in-sample resid':>18}{'LOO resid':>14}{'3-sigma floor':>16}{'/0.800':>10}")
print("-"*100)
t3={}
for (i,j) in PAIRS:
    xk=f'p{i}{j}'
    if xk not in L: continue
    x,yy=L[xk].values, L['gt0'].values
    A=np.polyfit(x,yy,1)
    in_res=float((yy-np.polyval(A,x)).std())
    loo=[]
    for k in range(len(x)):
        m=np.ones(len(x),bool); m[k]=False
        if m.sum()<3: continue
        Ak=np.polyfit(x[m],yy[m],1)
        loo.append(yy[k]-np.polyval(Ak,x[k]))
    loo_res=float(np.std(loo)) if loo else np.nan
    floor=3*loo_res if np.isfinite(loo_res) else np.nan
    t3[(i,j)]=dict(in_sample=in_res, loo=loo_res, floor=floor)
    print(f"  {f'({i},{j})':>8}{in_res:>18.5f}{loo_res:>14.5f}{floor:>16.5f}"
          f"{floor/0.800:>10.3f}")
valid=[v['floor'] for v in t3.values() if np.isfinite(v['floor'])]
if valid:
    bf=min(valid)
    print(f"\n  best 3-sigma slow-path floor = {bf:.4f} deg")
    print(f"  measured 6 h drift = 0.800 deg  ->  margin {0.800/bf:.1f}x")
    print(f"  fast-path ramp floor = 0.174 deg")
    if bf < 0.174:
        print(f"  -> the slow floor sits below the fast floor: coverage overlaps, no gap.")
    elif bf < 0.800:
        print(f"  -> the slow path catches degradation at the measured scale, but the")
        print(f"     band [{0.174:.3f}, {bf:.3f}] deg is covered by neither path. Report it.")
    else:
        print(f"  -> the slow floor exceeds the measured drift: insufficient resolution.")

json.dump(dict(t1={f'{k[0]}-{k[1]}':dict(raw=v[0],detrended=v[1],diff=v[2])
                   for k,v in t1.items()},
               t2={f'{k[0]}-{k[1]}':v for k,v in t2.items()},
               t3={f'{k[0]}-{k[1]}':v for k,v in t3.items()},
               verdict_t2=verdict_t2),
          open(f'{OUT}/transfer.json','w'), indent=2)

fig,ax=plt.subplots(1,3,figsize=(16,4.6))
C={'500gload':'#DC2626','unloaded':'#F59E0B','idle':'#94A3B8'}
a=ax[0]
for cname,s in sub.items():
    if 'p46' not in s: continue
    a.scatter(s['p46'],s['gt0'],color=C[cname],s=70,label=cname)
    if len(s)>=3:
        A=np.polyfit(s['p46'].values,s['gt0'].values,1)
        xs=np.linspace(s['p46'].min(),s['p46'].max(),20)
        a.plot(xs,np.polyval(A,xs),color=C[cname],lw=1.6,ls='--')
a.set_xlabel('pair (4,6) difference'); a.set_ylabel('joint 0 GT error (deg)')
a.set_title('calibration transfer across loads\n(parallel lines = real relation)',
            fontsize=10,fontweight='bold')
a.legend(fontsize=7); a.grid(alpha=.3)
a=ax[1]
if len(L)>=4:
    xd=detrend(L['p46'].values,t); yd=detrend(L['gt0'].values,t)
    a.scatter(xd,yd,color='#2563EB',s=80)
    for k,(xx,yy_) in enumerate(zip(xd,yd)):
        a.annotate(f"h{int(L.hour.values[k])}",(xx,yy_),fontsize=7)
a.axhline(0,color='k',lw=.6); a.axvline(0,color='k',lw=.6)
a.set_xlabel('pair (4,6), detrended'); a.set_ylabel('gt0, detrended')
a.set_title('after removing shared time trend\n(scatter = trend was the whole story)',
            fontsize=10,fontweight='bold')
a.grid(alpha=.3)
a=ax[2]
names=[f'({i},{j})' for (i,j) in PAIRS if (i,j) in t3]
vals=[t3[(i,j)]['floor'] for (i,j) in PAIRS if (i,j) in t3]
insm=[3*t3[(i,j)]['in_sample'] for (i,j) in PAIRS if (i,j) in t3]
xs=np.arange(len(names)); w=.35
a.bar(xs-w/2,insm,w,label='3$\\sigma$, in-sample',color='#94A3B8',ec='black')
a.bar(xs+w/2,vals,w,label='3$\\sigma$, LOO-CV',color='#DC2626',ec='black')
a.axhline(0.800,color='green',ls='--',lw=2,label='measured 6h drift')
a.axhline(0.174,color='blue',ls=':',lw=2,label='fast-path ramp floor')
a.set_xticks(xs); a.set_xticklabels(names); a.set_yscale('log')
a.set_ylabel('joint-0 drift (deg)')
a.set_title('slow-path floor, corrected',fontsize=10,fontweight='bold')
a.legend(fontsize=6); a.grid(axis='y',alpha=.3)
fig.suptitle('Is the canary relation real, or an artifact of shared monotone trend?',
             fontsize=12,fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/transfer.png',dpi=150,bbox_inches='tight')
plt.close()
print(f"\n  output: {OUT}/transfer.json  {OUT}/transfer.png")
print("="*100)
