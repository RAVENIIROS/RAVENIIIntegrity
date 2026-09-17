
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] RQ2 | Lag estimation, ARX identification, tracking-error floors
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Servo tracking-error detector (v2: identify the real servo model, three fixes)

Three problems with v1:

  [E1] The curve for injection A had a plateau in the middle: a fourfold drop
       in magnitude (0.12 to 0.03) moved AUC only from 0.694 to 0.638.
       Cause: the first-order model made e jump to mag at t0 and then decay
       with T = 4.7 frames, which is a decaying exponential, not an impulse.
       Matching an impulse template against a decay lasting about five frames
       captures only the peak frame and discards most of the energy. A's true
       floor is below 0.364.

  [E2] B's floor was entirely determined by GAIN_B = 0.02, a guessed value.
       Physically, the gain from unit torque to per-frame displacement is the
       regression slope a_j = [1.06, 1.60, 0.17], which is 53 times larger.
       Using a_j puts B's floor near 0.026 deg. So "tracking error is useless
       against B" does not hold; it was useless under an invented gain.

  [E3] C was more sensitive than A (0.213 against 0.364). The ordering is
       counterintuitive but physically right: C edits q and not q^d, so e
       absorbs the whole injection as a clean step, while under A the arm
       follows and e keeps only a decaying transient with far less energy.
       Implication: this detector is not a dedicated witness for command
       injection. It responds to disagreement between q^d and q, and all
       three injections cause that, differing in shape and size.

-- What v2 fixes -------------------------------------------------

  [F1] Identify the real servo model from data rather than inverting a lag
       into a first-order approximation.
       Method: ARX least squares, q[t] = sum a_i q[t-i] + sum b_j q^d[t-j-d],
       with orders chosen by held-out validation. The step response follows
       from the fit and is used for injection A.
       Cross-check: the ARX frequency response against an ETFE
       (S_{q,qd} / S_{qd,qd}).
       Closed-loop identification is biased in principle, since q^d comes
       from an operator watching q, but the operator's feedback is the
       endoscopic view rather than joint values, and human reaction latency
       (hundreds of ms) far exceeds the servo time constant (7 ms), so the
       bias should be small.

  [F2] GAIN_B uses the measured a_j instead of 0.02.

  [F3] The template for A becomes a decaying exponential, with a max|e|
       baseline alongside. If matched filtering does not beat the baseline,
       the template is wrong and the simple statistic should be reported.

Usage:    python3 exp_tracking_error.py
Expected: pure numpy, 10 to 20 minutes
"""
import os, sys, glob, gc, json, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, _DATA)
from peng_uw_loader import COL

ROOT=_DATA; OUT=_OUT + '/output_tracking'
os.makedirs(OUT, exist_ok=True)

SUBSAMPLE=1; CHUNKSIZE=50000; MAXROWS=120000
N=3; LOOKBACK=30; STRIDE=5; ATTACK_FRAC=0.30
MAXLAG=20; SEEDS=[42,43,44]; AUC_TARGET=0.90; MM_PER_DEG=7.47
FS=660.0
ARX_NA=[2,3,4,6]; ARX_NB=[2,3,4]      # candidate orders

COLS=dict(jpos=(13,21), jpos_d=(194,202), torque=(114,122))
NEEDED=sorted({c for lo,hi in COLS.values() for c in range(lo,hi)})

def read_one(path):
    parts,tot=[],0
    for ch in pd.read_csv(path,header=None,usecols=NEEDED,chunksize=CHUNKSIZE):
        s=ch.iloc[::SUBSAMPLE]; parts.append(s); tot+=len(s)
        if tot>=MAXROWS: break
    df=pd.concat(parts,ignore_index=True).iloc[:MAXROWS]
    del parts; gc.collect()
    return {k: df[list(range(lo,hi))].values.astype(np.float64)
            for k,(lo,hi) in COLS.items()}


def estimate_lag(qd,q,maxlag=MAXLAG):
    lags=[]
    for j in range(q.shape[1]):
        a,b=qd[:,j]-qd[:,j].mean(), q[:,j]-q[:,j].mean()
        if a.std()<1e-12 or b.std()<1e-12: lags.append(0); continue
        best,bl=-2.0,0
        for k in range(0,maxlag+1):
            x,y=a[:len(a)-k],b[k:]
            if len(x)<1000: continue
            c=float(np.corrcoef(x,y)[0,1])
            if c>best: best,bl=c,k
        lags.append(bl)
    return np.array(lags,int)


def aligned_error(qd,q,lags):
    L=int(lags.max()); n=len(q)-L
    E=np.zeros((n,q.shape[1]))
    for j in range(q.shape[1]):
        k=int(lags[j]); E[:,j]=qd[:n,j]-q[k:k+n,j]
    return E


# ══════════════════════════════════════════════════════════════
#  [F1] ARX identification
# ══════════════════════════════════════════════════════════════
def fit_arx(q, qd, d, na, nb):
    """q[t] = sum_{i=1..na} a_i q[t-i] + sum_{j=0..nb-1} b_j qd[t-d-j]"""
    T=len(q); s=max(na, d+nb)
    rows=[]
    for i in range(1,na+1): rows.append(q[s-i:T-i])
    for j in range(nb):     rows.append(qd[s-d-j:T-d-j])
    Phi=np.column_stack(rows); Y=q[s:T]
    theta,*_=np.linalg.lstsq(Phi,Y,rcond=None)
    pred=Phi@theta
    r2=1-((Y-pred)**2).sum()/max(((Y-Y.mean())**2).sum(),1e-12)
    return theta, float(r2), s


def arx_step_response(theta,na,nb,d,horizon=40):
    """Output response to a unit step command, used for injection A"""
    a=theta[:na]; b=theta[na:na+nb]
    y=np.zeros(horizon+max(na,d+nb)+5); u=np.zeros_like(y)
    off=max(na,d+nb)+2
    u[off:]=1.0
    for t in range(off,len(y)):
        acc=0.0
        for i in range(1,na+1): acc+=a[i-1]*y[t-i]
        for j in range(nb):
            idx=t-d-j
            if idx>=0: acc+=b[j]*u[idx]
        y[t]=acc
    return y[off:off+horizon]


def etfe(q, qd, nfft=4096):
    """Empirical transfer function estimate, to cross-check the ARX fit"""
    n=min(len(q),len(qd)); n=(n//nfft)*nfft
    if n<nfft: return None,None
    Q=np.fft.rfft((q[:n]-q[:n].mean()).reshape(-1,nfft),axis=1)
    D=np.fft.rfft((qd[:n]-qd[:n].mean()).reshape(-1,nfft),axis=1)
    Sqd=(np.conj(D)*Q).mean(axis=0); Sdd=(np.conj(D)*D).mean(axis=0)
    H=Sqd/np.maximum(np.abs(Sdd),1e-20)
    f=np.fft.rfftfreq(nfft,d=1.0/FS)
    return f,H


def arx_freq_response(theta,na,nb,d,f):
    a=theta[:na]; b=theta[na:na+nb]
    w=2*np.pi*f/FS; z=np.exp(1j*w)
    num=sum(b[j]*z**(-(d+j)) for j in range(nb))
    den=1-sum(a[i-1]*z**(-i) for i in range(1,na+1))
    return num/np.maximum(np.abs(den),1e-20)*np.sign(den+1e-30)


# ══════════════════════════════════════════════════════════════
#  Injection (v2)
# ══════════════════════════════════════════════════════════════
def inject_A(qd_w,q_w,mag,step_resp,rng):
    """[F1] Servo following, using the identified step response"""
    T=qd_w.shape[0]; t0=rng.randint(T//4,3*T//4); j=rng.randint(N)
    qd_n,q_n=qd_w.copy(),q_w.copy()
    qd_n[t0:,j]+=mag
    n=T-t0; sr=step_resp[j][:n]
    if len(sr)<n: sr=np.r_[sr, np.full(n-len(sr), sr[-1] if len(sr) else 1.0)]
    q_n[t0:,j]+=mag*sr
    return qd_n,q_n,j,t0


def inject_B(qd_w,q_w,mag,gain,rng):
    """[F2] Gain taken from the measured a_j"""
    T=qd_w.shape[0]; t0=rng.randint(T//4,3*T//4); j=rng.randint(N)
    qd_n,q_n=qd_w.copy(),q_w.copy()
    q_n[t0:,j]+=gain[j]*mag*np.arange(T-t0)
    return qd_n,q_n,j,t0


def inject_C(qd_w,q_w,mag,rng):
    T=qd_w.shape[0]; t0=rng.randint(T//4,3*T//4); j=rng.randint(N)
    qd_n,q_n=qd_w.copy(),q_w.copy()
    q_n[t0:,j]+=mag
    return qd_n,q_n,j,t0


# ══════════════════════════════════════════════════════════════
_T={}
def templates(T,kind,lo,hi,tau=5.0):
    key=(T,kind,lo,hi,round(tau,3))
    if key in _T: return _T[key]
    rows=[]
    for t0 in range(lo,min(hi,T-1)):
        s=np.zeros(T)
        if kind=='step': s[t0:]=1.0
        elif kind=='ramp': s[t0:]=np.arange(T-t0)
        elif kind=='decay':                       # [F3] first-order servo transient
            s[t0:]=np.exp(-np.arange(T-t0)/max(tau,1e-6))
        elif kind=='impulse': s[t0]=1.0
        rows.append(s/max(np.linalg.norm(s),1e-12))
    M=np.asarray(rows,np.float32); _T[key]=M
    return M


def stat_matched(E,kind,tau=5.0,combine='max'):
    """
    [F4] Combine across joints with max rather than energy.

    v2 combined the three joints as sqrt(sum acc^2) and matched filtering lost
    to the max|e| baseline on all three (A by 2.52x, B 1.63x, C 1.18x). C is a
    clean step where the step template should match exactly, yet it lost by

    3.6%, which points to an implementation error rather than a wrong shape.

    Cause: the attack lands on one random joint, so the matched-filter output
    on the other two is pure noise. Energy combination adds both and dilutes
    the ratio. Taking a global max naturally picks the attacked joint only.

    The prior is a single attacked joint, so max is right across joints too.
    The energy option is kept for comparison.
    """
    Nw,T,J=E.shape
    lo,hi=max(1,int(T*0.2)),max(2,int(T*0.85))
    M=templates(T,kind,lo,hi,tau)
    acc=np.zeros((Nw,J),np.float32)
    for j in range(J):
        acc[:,j]=np.abs(E[:,:,j]@M.T).max(axis=1)
    return acc.max(axis=1) if combine=='max' else np.sqrt((acc**2).sum(axis=1))


def stat_maxabs(E):
    """[F3] Baseline: no matched filtering"""
    return np.abs(E).max(axis=(1,2))


def floor_from(xs,ys,target=AUC_TARGET):
    o=np.argsort(xs); xs,ys=np.array(xs)[o],np.array(ys)[o]
    for k in range(len(xs)-1):
        if (ys[k]-target)*(ys[k+1]-target)<=0 and ys[k]!=ys[k+1]:
            t=(target-ys[k])/(ys[k+1]-ys[k])
            lo,hi=max(xs[k],1e-12),max(xs[k+1],1e-12)
            return float(np.exp(np.log(lo)+t*(np.log(hi)-np.log(lo))))
    return float(xs[0]) if ys[0]>=target else np.inf


# ══════════════════════════════════════════════════════════════
def main():
    t_all=time.time()
    print("="*100); print(" Servo tracking-error detector (v2)"); print("="*100)
    csvs=sorted(glob.glob(f'{ROOT}/record_1_different_directions/*.csv'))[:4]
    QD,Q,TQ,GID=[],[],[],[]
    for gi,p in enumerate(csvs):
        d=read_one(p)
        QD.append(d['jpos_d'][:,:N]); Q.append(d['jpos'][:,:N])
        TQ.append(d['torque'][:,:N]); GID.append(np.full(len(d['jpos']),gi))
        del d; gc.collect()
    n_tr=max(1,len(csvs)-1)
    qd_tr,q_tr=np.concatenate(QD[:n_tr]),np.concatenate(Q[:n_tr])
    qd_va,q_va=QD[-1],Q[-1]

    lags=estimate_lag(qd_tr,q_tr)
    print(f"\n  lag = {lags} frames = {lags/FS*1000} ms")

    # == [F1] ARX identification ==
    print("\n"+"="*100); print(" [F1] ARX servo identification (per joint, orders by held-out validation)")
    print("="*100)
    arx={}
    print(f"\n  {'joint':>6}{'na':>4}{'nb':>4}{'train R2':>11}{'val R2':>10}"
          f"{'DC gain':>10}{'settle 90% (frames)':>21}")
    print("-"*100)
    for j in range(N):
        best=None
        for na in ARX_NA:
            for nb in ARX_NB:
                try:
                    th,r2tr,_=fit_arx(q_tr[:,j],qd_tr[:,j],int(lags[j]),na,nb)
                except np.linalg.LinAlgError:
                    continue
                # Held-out validation
                s=max(na,int(lags[j])+nb); T=len(q_va)
                rows=[]
                for i in range(1,na+1): rows.append(q_va[s-i:T-i,j])
                for k in range(nb): rows.append(qd_va[s-int(lags[j])-k:T-int(lags[j])-k,j])
                Phi=np.column_stack(rows); Y=q_va[s:T,j]
                pred=Phi@th
                r2va=1-((Y-pred)**2).sum()/max(((Y-Y.mean())**2).sum(),1e-12)
                if best is None or r2va>best[3]:
                    best=(na,nb,r2tr,r2va,th)
        na,nb,r2tr,r2va,th=best
        sr=arx_step_response(th,na,nb,int(lags[j]),horizon=60)
        dc=float(sr[-1])
        idx=np.where(sr>=0.9*dc)[0] if dc>0 else np.array([])
        settle=int(idx[0]) if len(idx) else -1
        arx[j]=dict(na=na,nb=nb,theta=th,step=sr,dc=dc,settle=settle,
                    r2tr=float(r2tr),r2va=float(r2va))
        print(f"  {j:>6}{na:>4}{nb:>4}{r2tr:>11.6f}{r2va:>10.6f}"
              f"{dc:>10.4f}{settle:>17}")
    print(f"\n  DC gain should be near 1, else there is a steady-state offset.")

    # ETFE cross-check
    print(f"\n  ETFE cross-check (ARX against empirical response, low frequency):")
    print(f"  {'joint':>6}{'|H| @1Hz ARX':>15}{'ETFE':>10}"
          f"{'|H| @10Hz ARX':>16}{'ETFE':>10}")
    print("-"*100)
    for j in range(N):
        f,H=etfe(q_tr[:,j],qd_tr[:,j])
        if f is None: print(f"  {j:>6}  insufficient data"); continue
        Ha=arx_freq_response(arx[j]['theta'],arx[j]['na'],arx[j]['nb'],
                             int(lags[j]),f)
        def at(fr):
            k=np.argmin(np.abs(f-fr)); return abs(Ha[k]),abs(H[k])
        a1,e1=at(1.0); a10,e10=at(10.0)
        print(f"  {j:>6}{a1:>15.4f}{e1:>10.4f}{a10:>16.4f}{e10:>10.4f}")
    print(f"  Agreement means the ARX captured the response; a gap means too few")

    # == Windowing ==
    W_qd,W_q,W_g=[],[],[]
    for gi in range(len(csvs)):
        for i in range(LOOKBACK,len(Q[gi])-int(lags.max()),STRIDE):
            W_qd.append(QD[gi][i-LOOKBACK:i]); W_q.append(Q[gi][i-LOOKBACK:i])
            W_g.append(gi)
    W_qd=np.stack(W_qd); W_q=np.stack(W_q); W_g=np.array(W_g)
    m_va=W_g==(len(csvs)-1)
    idx_va=np.where(m_va)[0]
    print(f"\n  windows {W_qd.shape}   val {m_va.sum():,}")

    E_cl=np.stack([aligned_error(W_qd[i],W_q[i],lags)
                   for i in np.where(~m_va)[0][:4000]])
    mu=E_cl.reshape(-1,N).mean(0).astype(np.float32)
    sg=np.maximum(E_cl.reshape(-1,N).std(0),1e-9).astype(np.float32)
    print(f"  clean aligned residual std = {sg}")
    del E_cl; gc.collect()

    # [F2] Torque-to-displacement gain from the measured regression slope
    gain=np.zeros(N)
    dq=np.diff(q_tr,axis=0); tq=np.concatenate(TQ[:n_tr])[1:]
    for j in range(N):
        if tq[:,j].std()>1e-12:
            gain[j]=np.cov(tq[:,j],dq[:,j],bias=True)[0,1]/tq[:,j].var()
    print(f"  [F2] torque-to-displacement gain a_j = {gain}   (v1 used 0.02)")

    step_resp={j:arx[j]['step'] for j in range(N)}
    tau_eff=float(np.mean([arx[j]['settle'] for j in range(N)
                           if arx[j]['settle']>0]) or 5.0)/2.3
    print(f"  [F3] decay template time constant = {tau_eff:.2f} frames (from 90% settling)")

    MAGS=[2.0,1.0,0.5,0.25,0.12,0.06,0.03,0.015,0.008,0.004]
    KIND=dict(A='decay',B='ramp',C='step')
    rows=[]
    print("\n"+"="*100); print(" Three injections (matched filter vs max|e| baseline)"); print("="*100)
    for atk in ['A','B','C']:
        print(f"\n  -- {atk}  template={KIND[atk]} --")
        print(f"    {'mag deg':>10}{'matched/max':>13}{'matched/enrg':>13}"
              f"{'max|e|':>12}{'tip mm':>10}")
        for mag in MAGS:
            am,ae,ab=[],[],[]
            for sd_ in SEEDS:
                rng=np.random.RandomState(sd_+int(mag*100000))
                Es,y=[],[]
                for i in idx_va:
                    at_=rng.rand()<ATTACK_FRAC
                    qw,ww=W_qd[i],W_q[i]
                    if at_:
                        if atk=='A': qw,ww,_,_=inject_A(qw,ww,mag,step_resp,rng)
                        elif atk=='B': qw,ww,_,_=inject_B(qw,ww,mag,gain,rng)
                        else: qw,ww,_,_=inject_C(qw,ww,mag,rng)
                    Es.append(aligned_error(qw,ww,lags)); y.append(float(at_))
                E=(np.stack(Es)-mu[None,None,:])/sg[None,None,:]
                y=np.array(y)
                if len(np.unique(y))>1:
                    am.append(roc_auc_score(y,stat_matched(E,KIND[atk],tau_eff,'max')))
                    ae.append(roc_auc_score(y,stat_matched(E,KIND[atk],tau_eff,'energy')))
                    ab.append(roc_auc_score(y,stat_maxabs(E)))
                del E,Es
            if not am: continue
            rows.append(dict(attack=atk,mag=float(mag),
                             auc_matched=float(np.mean(am)),
                             auc_energy=float(np.mean(ae)),
                             auc_base=float(np.mean(ab)),
                             sd=float(np.std(am))))
            print(f"    {mag:>10.4f}{np.mean(am):>13.4f}{np.mean(ae):>13.4f}"
                  f"{np.mean(ab):>12.4f}{mag*MM_PER_DEG:>10.3f}")

    print("\n"+"="*100); print(" Detection floors"); print("="*100)
    print(f"\n  {'inject':>8}{'matched/max':>14}{'tip mm':>10}"
          f"{'matched/enrg':>14}{'max|e|':>12}{'best':>14}")
    print("-"*100)
    floors={}
    for atk in ['A','B','C']:
        sub=[r for r in rows if r['attack']==atk]
        if len(sub)<2: continue
        fm=floor_from([r['mag'] for r in sub],[r['auc_matched'] for r in sub])
        fe=floor_from([r['mag'] for r in sub],[r['auc_energy'] for r in sub])
        fb=floor_from([r['mag'] for r in sub],[r['auc_base'] for r in sub])
        floors[atk]=dict(matched=fm,energy=fe,baseline=fb)
        cand={'matched/max':fm,'matched/enrg':fe,'max|e|':fb}
        best=min((v,k) for k,v in cand.items() if np.isfinite(v))[1] \
             if any(np.isfinite(v) for v in cand.values()) else '-'
        print(f"  {atk:>8}{(f'{fm:.5f}' if np.isfinite(fm) else 'none'):>14}"
              f"{(fm*MM_PER_DEG if np.isfinite(fm) else np.nan):>10.3f}"
              f"{(f'{fe:.5f}' if np.isfinite(fe) else 'none'):>14}"
              f"{(f'{fb:.5f}' if np.isfinite(fb) else 'none'):>12}{best:>14}")
    print(f"\n  [F4] matched/max should beat matched/enrg, since the attack is on one")
    print(f"       joint and energy combination adds noise from the other two.")
    print(f"       If it still loses to max|e|, the template is useless; report that.")

    print("\n"+"="*100); print(" Comparison against v1"); print("="*100)
    print(f"\n  {'inject':>8}{'v1 (deg)':>12}{'v2 (deg)':>12}"
          f"{'v3 best':>12}{'v3/v2':>10}")
    print("-"*100)
    V1=dict(A=0.36411,B=1.40276,C=0.21274)
    V2=dict(A=0.96180,B=0.11694,C=0.21274)
    for atk in ['A','B','C']:
        if atk not in floors: continue
        fm=min(v for v in floors[atk].values() if np.isfinite(v)) \
           if any(np.isfinite(v) for v in floors[atk].values()) else np.inf
        print(f"  {atk:>8}{V1[atk]:>12.5f}{V2[atk]:>12.5f}"
              f"{(f'{fm:.5f}' if np.isfinite(fm) else 'none'):>12}"
              f"{(f'{fm/V2[atk]:.2f}x' if np.isfinite(fm) else '-'):>10}")
    print(f"\n  The change in A comes from the decay template plus the identified")
    print(f"  step response; the change in B from the gain moving from 0.02 to the\n  measured a_j, {gain[0]/0.02:.0f}x larger.")

    json.dump(dict(lags=lags.tolist(), gain=gain.tolist(), tau_eff=tau_eff,
                   mu=mu.tolist(), sigma=sg.tolist(),
                   arx={str(j):dict(na=arx[j]['na'],nb=arx[j]['nb'],
                                    r2tr=arx[j]['r2tr'],r2va=arx[j]['r2va'],
                                    dc=arx[j]['dc'],settle=arx[j]['settle'],
                                    step=arx[j]['step'].tolist())
                        for j in range(N)},
                   rows=rows,
                   floors={k:{kk:(None if not np.isfinite(vv) else vv)
                              for kk,vv in v.items()} for k,v in floors.items()}),
              open(f'{OUT}/summary_v2.json','w'), indent=2)

    fig,ax=plt.subplots(1,3,figsize=(17,4.6))
    a=ax[0]
    for j in range(N):
        a.plot(np.arange(len(arx[j]['step']))/FS*1000, arx[j]['step'],
               lw=2, label=f'joint {j}')
    a.axhline(1.0,color='k',ls=':',alpha=.4)
    a.set_xlabel('time (ms)'); a.set_ylabel('normalised response')
    a.set_title('identified step response (ARX)',fontsize=11,fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=.3)
    cm=dict(A='#2563EB',B='#DC2626',C='#888780')
    lab=dict(A='A: desired pose',B='B: motor torque',C='C: reported jpos')
    a=ax[1]
    for atk in ['A','B','C']:
        sub=sorted([r for r in rows if r['attack']==atk],key=lambda r:r['mag'])
        if not sub: continue
        a.errorbar([r['mag'] for r in sub],[r['auc_matched'] for r in sub],
                   yerr=[r['sd'] for r in sub],marker='o',color=cm[atk],
                   lw=2,markersize=6,capsize=3,label=lab[atk])
    a.axhline(AUC_TARGET,color='gray',ls=':',alpha=.7)
    a.axhline(0.5,color='k',ls=':',alpha=.35)
    a.set_xscale('log'); a.set_ylim(0.4,1.02)
    a.set_xlabel('injected magnitude (deg)'); a.set_ylabel('ROC-AUC')
    a.set_title('matched filter',fontsize=11,fontweight='bold')
    a.legend(fontsize=8); a.grid(alpha=.3)
    a=ax[2]
    ks=[k for k in ['A','B','C'] if k in floors
        and np.isfinite(floors[k]['matched'])]
    x=np.arange(len(ks)); w=.35
    a.bar(x-w/2,[floors[k]['matched']*MM_PER_DEG for k in ks],w,
          label='matched',color='#2563EB',ec='black')
    a.bar(x+w/2,[floors[k]['baseline']*MM_PER_DEG if np.isfinite(floors[k]['baseline'])
                 else 0 for k in ks],w,label='max|e|',color='#888780',ec='black')
    a.axhline(1.0,color='green',ls='--',lw=1.5,label='1 mm')
    a.set_yscale('log'); a.set_xticks(x); a.set_xticklabels(ks)
    a.set_ylabel('floor (mm at tip)')
    a.set_title('matched vs baseline',fontsize=11,fontweight='bold')
    a.legend(fontsize=7); a.grid(axis='y',alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/tracking_v2.png',dpi=150,
                                    bbox_inches='tight')
    print(f"\n  output: {OUT}/summary_v2.json   {OUT}/tracking_v2.png")
    print(f"  total {time.time()-t_all:.0f}s")
    print("="*100)


if __name__=='__main__':
    main()
