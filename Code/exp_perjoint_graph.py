
import os
_DATA = os.environ.get('RAVEN_DATA', './data')
_OUT  = os.environ.get('RAVEN_OUT',  './output')

# [ARTIFACT] support | Per-joint encoder ablation (0.170 -> 0.101 deg, 60% fewer parameters)
# Reproduction steps: ../README.md   Paper-number mapping: ../MANIFEST.md
# Set RAVEN_DATA and RAVEN_OUT, or edit Cfg in this file, before running

"""
Per-joint encoding plus coordinated multi-joint attacks: is the graph useful?

The original question was whether adding the known relations between joints,
as a graph, improves performance. Earlier experiments did not answer it, for
two reasons.

  [P1] The nodes carried no joint identity
       The original architecture fed 19-dimensional input into an LSTM, took
       the final hidden state h (128 dims), and projected seven "nodes" from
       that same h with seven linear layers. Each projection acts on the whole
       of h, so node_j has no correspondence with joint j. Message passing
       then computes sum_k (W_k @ h) = (sum_k W_k) @ h, still a linear
       function of h, which the decoder's first layer can already express.
       So "the DH topology equals a random graph" was algebra, not a finding.
       Measured: no graph 0.9755 > chain 0.9721 > plus cable 0.9664 > fully
       connected 0.9588. More edges, monotonically worse.

  [P2] The attacks were single-joint
       Step and noise hit joint 1, ramp hits joints 1 and 2. Detecting those
       needs only finding which joint's trajectory is inconsistent, not
       reasoning across joints. Graphs are good at relational reasoning, and
       this task is not relational.

-- Three design decisions ----------------------------------------

  [D1] Construct an attack a graph can catch and a graph-free model cannot

       v1's attempt failed: spreading the perturbation along the largest
       eigenvector of the covariance put 96% of the magnitude on joint 2 and
       1.4% on joint 3. That degenerates into a single-joint attack on a
       different joint, leaving the graph nothing to do.

       Cause: with eigenvalues [0.00047, 0.0282, 0.0488], the leading
       direction is dominated by the highest-variance joint, since
       std(dj) = [0.172, 0.219, 0.021] and joint 3 is an order of magnitude
       smaller. The principal component reflects which joint moves most, not
       how the joints relate.

       v2 uses the smallest eigenvector of the *correlation* matrix instead,
       scaled by each joint's own standard deviation:
           C = corrcoef(dj)          correlation, scale differences removed
           dir = V[:, argmin(w)]     the direction the data almost never takes

       Each joint's absolute offset then stays within its own scale (a few
       sigma each), so any single joint looks normal, while their joint
       distribution lands in the direction of least natural variation, a
       low-probability region. Only the relation between joints reveals it.

       coordinated: the perturbation is spread across the three positioning
       joints and respects the inter-joint covariance estimated from clean
       data. Each joint alone sits inside normal variation; only the relation
       is wrong.
       single (control): the same total magnitude, all on joint 1.
       If the graph helps, it should help on coordinated and not matter on
       single.

  [D2] Four configurations, separating two confounded factors
       A  shared    shared LSTM plus 7 linear projections (original)  baseline
       B  perjoint  each node reads only its own joint's channels, no edges
                    -> isolates representation
       C  perjoint  as B plus DH and cable edges  -> isolates message passing
       D  perjoint  as B plus a random graph of the same size  -> does the
                    specific connectivity matter?
       B is the important one: without it, a gain cannot be attributed to
       representation rather than to the graph.

  [D3] The latency budget is set aside for now
       Per-joint encoding runs seven forward passes in a loop and will exceed
       RAVEN-II's 1 ms control period. Answering whether it helps has to come
       first; feasibility is a separate question. The script reports the
       measured inference time.

Reading the result:
  C > B > A, and a wider gap on coordinated -> the graph helps, and the
                                              condition under which it does
                                              has been found
  B > A but C ~ B                           -> per-joint representation helps,
                                              message passing does not
  all three comparable                      -> the graph really is ineffective,
                                              narrowing the negative result
  C ~ D                                     -> connectivity is irrelevant and
                                              only the parameter count changed

Usage:    python3 exp_perjoint_graph.py
Expected: 4 configs x 2 attacks x 6 magnitudes x 3 seeds = 144 runs, 6-8 hours
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
from exp_paper_sweep import detection_limit


class Cfg(BaseCfg):
    OUT_DIR=_OUT + '/output_perjoint'
    MODE='jpos_torque_delta'
    SEEDS=[42,43,44]
    EPOCHS=120; LR=3e-4; BATCH=128; HIDDEN=128
    W_DEV, W_ATK, W_VEL = 1.0, 0.5, 0.1

    N_NODES=7
    PJ_HIDDEN=48                 # per-joint LSTM hidden size; 7*48=336, not equal
                                 # to 128 but similar in parameter count
    ATTACKS=['single','coordinated']
    SCALES=[0.30,0.18,0.10,0.06,0.035,0.020]
    AUC_TARGET=0.90
    CONFIGS=[('A_shared_nograph',  'shared',   'none'),
             ('B_perjoint_nograph','perjoint', 'none'),
             ('C_perjoint_dhcable','perjoint', 'dhcable'),
             ('D_perjoint_random', 'perjoint', 'random')]


# ══════════════════════════════════════════════════════════════
#  Graph structure
# ══════════════════════════════════════════════════════════════
def make_edges(kind, n=7, seed=0):
    """Return a bidirectional edge_index of shape (2,E)"""
    if kind=='none': return None
    pairs=[]
    if kind=='dhcable':
        pairs=[(i,i+1) for i in range(n-1)]          # DH chain, 6 edges
        pairs.append((1,4))                           # cable edge (0-indexed)
    elif kind=='random':
        rng=np.random.RandomState(seed)
        allp=[(i,j) for i in range(n) for j in range(i+1,n)]
        idx=rng.choice(len(allp), size=n, replace=False)   # same 7 as dhcable
        pairs=[allp[k] for k in idx]
    elif kind=='complete':
        pairs=[(i,j) for i in range(n) for j in range(i+1,n)]
    src=[p[0] for p in pairs]+[p[1] for p in pairs]
    dst=[p[1] for p in pairs]+[p[0] for p in pairs]
    return torch.tensor([src,dst], dtype=torch.long)


class GraphConv(nn.Module):
    """Simplest mean-aggregation graph convolution, comparable with the original"""
    def __init__(s, d):
        super().__init__()
        s.lin_self=nn.Linear(d,d); s.lin_nb=nn.Linear(d,d)
    def forward(s, x, edge_index):
        # x: (B, N, d)
        if edge_index is None: return torch.relu(s.lin_self(x))
        B,N,d=x.shape
        src,dst=edge_index[0],edge_index[1]
        agg=torch.zeros_like(x)
        cnt=torch.zeros(B,N,1,device=x.device)
        agg.index_add_(1, dst, x[:,src,:])
        cnt.index_add_(1, dst, torch.ones(B,len(src),1,device=x.device))
        agg=agg/cnt.clamp(min=1.0)
        return torch.relu(s.lin_self(x)+s.lin_nb(agg))


# ══════════════════════════════════════════════════════════════
#  Models
# ══════════════════════════════════════════════════════════════
class SharedDet(nn.Module):
    """[D2-A] Original architecture: shared LSTM, 7 linear projections as nodes"""
    def __init__(s, d_in, h, la, n_nodes, graph):
        super().__init__(); s.la=la; s.n=n_nodes; s.graph=graph
        s.lstm=nn.LSTM(d_in,h,2,batch_first=True,dropout=0.2)
        s.proj=nn.ModuleList([nn.Linear(h,h//2) for _ in range(n_nodes)])
        s.gc=GraphConv(h//2)
        s.dev=nn.Sequential(nn.Linear(n_nodes*(h//2),256),nn.ReLU(),
                            nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),
                            nn.Linear(64,la*3))
        nn.init.zeros_(s.dev[-1].weight); nn.init.zeros_(s.dev[-1].bias)
        s.atk=nn.Sequential(nn.Linear(n_nodes*(h//2),128),nn.ReLU(),
                            nn.Dropout(0.2),nn.Linear(128,32),nn.ReLU(),
                            nn.Linear(32,1))
    def forward(s,x):
        h=s.lstm(x)[0][:,-1,:]
        nodes=torch.stack([p(h) for p in s.proj],dim=1)   # (B,N,h/2)
        nodes=s.gc(nodes, s.graph)
        f=nodes.flatten(1)
        return s.dev(f).view(-1,s.la,3), s.atk(f).squeeze(-1)


class PerJointDet(nn.Module):
    """
    [D2-B/C] Per-joint encoding: node j reads only joint j's channels.
    Channel assignment (MODE='jpos_torque_delta', 19 dims):
        jpos   0..7    one per joint 0..6 (the 8th is the grasper, into node 6)
        torque 8..15   likewise
        delta  16..18  positioning joints only; other nodes get zeros
    LSTM weights are shared across nodes, otherwise the parameter count explodes.
    """
    def __init__(s, d_per, h, la, n_nodes, graph):
        super().__init__(); s.la=la; s.n=n_nodes; s.graph=graph
        s.lstm=nn.LSTM(d_per,h,2,batch_first=True,dropout=0.2)
        s.gc=GraphConv(h)
        s.dev=nn.Sequential(nn.Linear(n_nodes*h,256),nn.ReLU(),
                            nn.Dropout(0.1),nn.Linear(256,64),nn.ReLU(),
                            nn.Linear(64,la*3))
        nn.init.zeros_(s.dev[-1].weight); nn.init.zeros_(s.dev[-1].bias)
        s.atk=nn.Sequential(nn.Linear(n_nodes*h,128),nn.ReLU(),
                            nn.Dropout(0.2),nn.Linear(128,32),nn.ReLU(),
                            nn.Linear(32,1))
    def forward(s,x_nodes):
        # x_nodes: (B, N, T, d_per)
        B,N,T,d=x_nodes.shape
        flat=x_nodes.reshape(B*N,T,d)
        h=s.lstm(flat)[0][:,-1,:].reshape(B,N,-1)
        h=s.gc(h, s.graph)
        f=h.flatten(1)
        return s.dev(f).view(-1,s.la,3), s.atk(f).squeeze(-1)


def to_per_joint(X, n_nodes=7, n_pos=3):
    """(B,T,19) -> (B,N,T,3): per node [jpos_j, torque_j, delta_j or 0]"""
    B,T,_=X.shape
    out=np.zeros((B,n_nodes,T,3),np.float32)
    for j in range(n_nodes):
        out[:,j,:,0]=X[:,:,j]           # jpos_j
        out[:,j,:,1]=X[:,:,8+j]         # torque_j
        if j<n_pos: out[:,j,:,2]=X[:,:,16+j]   # delta_j
    return out


# ══════════════════════════════════════════════════════════════
#  [D1] Attacks
# ══════════════════════════════════════════════════════════════
def estimate_cov(JP, mask, n_pos=3):
    """
    Estimate the correlation structure of first differences across joints.
    Returns (corr, std): the correlation matrix and per-joint standard deviations.
    Correlation rather than covariance, since the latter's leading direction is
    dominated by the highest-variance joint (see v1's failure in [D1]).
    """
    d=np.diff(JP[mask][:,:,:n_pos],axis=1).reshape(-1,n_pos)
    return np.corrcoef(d.T), d.std(axis=0)


def inject(JP, cfg, mask, kind, scale, corr_std, seed_off=0, n_pos=3):
    """
    single      : the whole magnitude on joint 1, ramp shaped
    coordinated : spread along the smallest eigenvector of the correlation
                  matrix, scaled by each joint's own std. Each joint alone is
                  normal; only the joint distribution is improbable.
    """
    rng=np.random.RandomState(cfg.SEED+seed_off+int(scale*1000000))
    out=JP.copy(); y=np.zeros(len(JP),np.float32); dev=np.zeros(len(JP),np.float32)
    pool=np.where(mask)[0]
    corr, sdj = corr_std
    w,V=np.linalg.eigh(corr)
    d_min=V[:,np.argmin(w)]                     # the direction the data avoids
    d_min=d_min/np.linalg.norm(d_min)
    # Scale by per-joint std, then normalise so total L2 matches single
    principal = d_min * sdj
    principal = principal/max(np.linalg.norm(principal),1e-12)
    for i in rng.permutation(pool)[:int(len(pool)*cfg.ATTACK_FRAC)]:
        T=JP.shape[1]; t0=rng.randint(T//4,3*T//4)
        ramp=np.linspace(0,1,T-t0)*scale
        if kind=='single':
            out[i,t0:,0]+=ramp
            dev[i]=scale
        else:
            # Spread along the chosen direction with the same total L2 as single
            for j in range(n_pos):
                out[i,t0:,j]+=ramp*principal[j]
            dev[i]=scale
        y[i]=1.0
    m=y>0.5
    return out,y,(float(dev[m].mean()) if m.any() else 0.0)


# ══════════════════════════════════════════════════════════════
def train_eval(X, YD, y, GID, cfg, seed, arch, graph_kind):
    torch.manual_seed(seed); np.random.seed(seed); dev=cfg.DEVICE
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva

    f=X[mtr].reshape(-1,X.shape[-1])
    mu=f.mean(0).astype(np.float32); sd=np.maximum(f.std(0),1e-6).astype(np.float32)
    Xn=((X-mu)/sd).astype(np.float32)

    edges=make_edges(graph_kind, cfg.N_NODES, seed)
    if edges is not None: edges=edges.to(dev)

    if arch=='shared':
        Xtr=Xn[mtr]; Xva=Xn[mva]
        m=SharedDet(X.shape[-1],cfg.HIDDEN,cfg.LOOKAHEAD,cfg.N_NODES,edges).to(dev)
    else:
        Xtr=to_per_joint(Xn[mtr],cfg.N_NODES,cfg.N_POS_JOINTS)
        Xva=to_per_joint(Xn[mva],cfg.N_NODES,cfg.N_POS_JOINTS)
        m=PerJointDet(3,cfg.PJ_HIDDEN,cfg.LOOKAHEAD,cfg.N_NODES,edges).to(dev)

    ytr,yva=y[mtr],y[mva]
    std_delta=YD[mtr].std()+1e-8
    Ydtr=(YD[mtr]/std_delta).astype(np.float32)
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
    with torch.no_grad():
        t_start=time.time()
        _,ap=m(Xv[:1000])
        t_infer=(time.time()-t_start)/1000*1000     # ms per window
        _,ap=m(Xv)
    sc=torch.sigmoid(ap).cpu().numpy()
    auc=roc_auc_score(yva,sc) if len(np.unique(yva))>1 else 0.5
    npar=sum(p.numel() for p in m.parameters())
    del m,Xt,Yt,Dt,Xv
    if dev=='cuda': torch.cuda.empty_cache()
    return float(auc), int(npar), float(t_infer)


def main():
    cfg=Cfg(); os.makedirs(cfg.OUT_DIR,exist_ok=True); t0=time.time()
    print("="*100); print(" Per-joint encoding and coordinated attacks: is the graph useful?"); print("="*100)
    print(f"""
  [D2] four configurations
    A_shared_nograph    shared LSTM + 7 projections, no edges  (original)
    B_perjoint_nograph  per-joint encoding, no edges           (representation)
    C_perjoint_dhcable  per-joint + DH chain + cable edges     (message passing)
    D_perjoint_random   per-joint + random graph, same size    (connectivity)

  [D1] two attacks
    single       whole magnitude on joint 1
    coordinated  same total magnitude spread across three joints
""")
    recs=load_records(cfg)
    a_,b_,r2,_,_=fit_physics_model(recs,cfg.N_POS_JOINTS)
    JP,TQ,YD,GID=build_window_cache(recs,cfg)
    g=np.unique(GID); nv=max(1,int(round(len(g)*cfg.VAL_FRAC_RECORDS)))
    vg=np.random.RandomState(cfg.SEED).permutation(g)[:nv]
    mva=np.isin(GID,vg); mtr=~mva

    corr_std=estimate_cov(JP,mtr,cfg.N_POS_JOINTS)
    corr,sdj=corr_std
    w,V=np.linalg.eigh(corr)
    d_min=V[:,np.argmin(w)]; d_min=d_min/np.linalg.norm(d_min)
    share=d_min*sdj; share=share/max(np.linalg.norm(share),1e-12)
    print(f"  per-joint std(dj): {sdj.round(5)}")
    print(f"  correlation matrix:")
    for r in corr: print(f"    {r.round(4)}")
    print(f"  correlation eigenvalues: {w.round(5)}")
    print(f"  smallest eigenvector: {d_min.round(4)}")
    print(f"  coordinated share after std scaling: {share.round(4)}")
    print(f"  -> only comparable components mean the attack really spreads;")
    print(f"     one near 1 degenerates into a single-joint attack.")
    print(f"  edge_index (dhcable): {make_edges('dhcable').shape[1]} directed edges")

    ck=f'{cfg.OUT_DIR}/results.json'
    rows=json.load(open(ck)) if os.path.exists(ck) else []
    seen={(r['config'],r['attack'],r['scale'],r['seed']) for r in rows}
    todo=[(c,at,sc,sd) for c in cfg.CONFIGS for at in cfg.ATTACKS
          for sc in cfg.SCALES for sd in cfg.SEEDS
          if (c[0],at,sc,sd) not in seen]
    print(f"\n  {len(todo)} training runs to go\n")

    for n,((cname,arch,gk),atk,scale,seed) in enumerate(todo,1):
        JPtr,ytr,_=inject(JP,cfg,mtr,atk,scale,corr_std,0)
        JPva,yva,dv=inject(JP,cfg,mva,atk,scale,corr_std,7777)
        JPm=JP.copy(); JPm[mtr]=JPtr[mtr]; JPm[mva]=JPva[mva]
        yy=np.zeros(len(JP),np.float32); yy[mtr]=ytr[mtr]; yy[mva]=yva[mva]
        X=build_features(JPm,TQ,cfg.MODE,a_,b_,cfg.N_POS_JOINTS)
        auc,npar,tms=train_eval(X,YD,yy,GID,cfg,seed,arch,gk)
        rows.append(dict(config=cname,arch=arch,graph=gk,attack=atk,
                         scale=float(scale),seed=seed,x=dv,auc=auc,
                         params=npar,infer_ms=tms))
        tmp=ck+'.tmp'
        with open(tmp,'w') as fh: json.dump(rows,fh)
        os.replace(tmp,ck)
        print(f"  [{n}/{len(todo)}] {cname:<20}{atk:<12}sc={scale:<7.4f}"
              f"s={seed} AUC={auc:.4f}  ({time.time()-t0:.0f}s)")
        del X,JPtr,JPva,JPm

    # == Summary ==
    def floor(cname,atk):
        d={}
        for r in rows:
            if r['config']==cname and r['attack']==atk:
                d.setdefault(r['scale'],[]).append(r)
        if len(d)<2: return np.inf
        pts=sorted((np.mean([q['x'] for q in v]),np.mean([q['auc'] for q in v]))
                   for v in d.values())
        return detection_limit([p[0] for p in pts],[p[1] for p in pts],cfg.AUC_TARGET)

    print("\n"+"="*100); print(" Detection floors (deg)"); print("="*100)
    print(f"\n  {'config':<22}{'params':>10}{'infer ms':>10}"
          f"{'single':>12}{'coordinated':>14}{'coord/single':>14}")
    print("-"*100)
    F={}
    for cname,arch,gk in cfg.CONFIGS:
        sel=[r for r in rows if r['config']==cname]
        if not sel: continue
        fs,fc=floor(cname,'single'),floor(cname,'coordinated')
        F[cname]=(fs,fc)
        ratio=fc/fs if np.isfinite(fs) and np.isfinite(fc) and fs>0 else np.nan
        print(f"  {cname:<22}{sel[0]['params']:>10,}{sel[0]['infer_ms']:>10.3f}"
              f"{(f'{fs:.4f}' if np.isfinite(fs) else 'none'):>12}"
              f"{(f'{fc:.4f}' if np.isfinite(fc) else 'none'):>14}"
              f"{(f'{ratio:.2f}x' if np.isfinite(ratio) else '-'):>14}")

    print("\n"+"="*100); print(" Reading the result"); print("="*100)
    A=F.get('A_shared_nograph'); B=F.get('B_perjoint_nograph')
    C=F.get('C_perjoint_dhcable'); D=F.get('D_perjoint_random')
    def rel(x,base,k):
        if not x or not base: return np.nan
        return x[k]/base[k] if np.isfinite(x[k]) and np.isfinite(base[k]) and base[k]>0 else np.nan
    print(f"\n  {'comparison':<40}{'single':>12}{'coordinated':>14}")
    print("-"*100)
    for lab,x,base in [("B vs A  (per-joint representation)",B,A),
                       ("C vs B  (message passing)",C,B),
                       ("C vs D  (does connectivity matter)",C,D)]:
        r0,r1=rel(x,base,0),rel(x,base,1)
        print(f"  {lab:<40}"
              f"{(f'{r0:.2f}x' if np.isfinite(r0) else '-'):>12}"
              f"{(f'{r1:.2f}x' if np.isfinite(r1) else '-'):>14}")
    print(f"""
  A ratio below 1 means a lower floor, which is better.

  C<B<A with a wider gap on coordinated -> the graph helps, condition found
  B<A but C~B                           -> representation helps, passing does not
  all comparable                        -> the graph is ineffective, and the
                                           negative result now covers correct nodes
  C~D                                   -> connectivity is irrelevant

  Note the infer ms column: per-joint encoding runs {cfg.N_NODES} forward
  passes in a loop and will exceed RAVEN-II's 1 ms control period. If the
  graph helps, that is a trade-off to discuss; if not, one more reason not to.
""")
    print("="*100)


if __name__=='__main__':
    main()
