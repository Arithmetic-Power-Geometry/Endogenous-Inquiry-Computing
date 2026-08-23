from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Iterable, List, Dict, Tuple
import itertools
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

METHODS = (
    "Fixed-Frame", "Random-Edit", "Flat-Search", "MDL-Flat",
    "EIC-NoMargin", "EIC-NoCost", "EIC-Greedy", "Oracle"
)

@dataclass
class TrialResult:
    suite: str
    benchmark: str
    defect: str
    control: bool
    method: str
    seed: int
    level: str
    performance: float
    loss: float
    initial_loss: float
    frame_recovered: float
    reframe_cost: float
    reframed: float
    fde: float
    candidates: int

    def row(self):
        return asdict(self)


def _perf(loss: float, scale: float) -> float:
    scale=max(float(scale),1e-12)
    return float(1.0/(1.0+max(float(loss),0.0)/scale))

def _fde(p0,p1,c):
    return float(max(0.0,p1-p0)/(1.0+max(c,0.0)))

def _select(scores: List[Tuple[float,int]], method: str, rng, base_idx=0, costs=None, margin=.015, penalty=.04):
    """Select candidate index from validation losses. Candidate 0 is current frame."""
    losses=np.array([x[0] for x in scores],dtype=float)
    costs=np.array(costs if costs is not None else [x[1] for x in scores],dtype=float)
    scale=max(float(losses[0]),1e-9)
    if method=="Fixed-Frame": return 0
    if method=="Oracle": raise RuntimeError("oracle selection handled by caller")
    if method=="Random-Edit":
        return int(rng.integers(0,len(scores)))
    if method=="Flat-Search":
        return int(np.argmin(losses))
    if method=="MDL-Flat":
        return int(np.argmin(losses/scale + penalty*costs))
    if method=="EIC-NoMargin":
        return int(np.argmin(losses/scale + penalty*costs))
    if method=="EIC-NoCost":
        best=int(np.argmin(losses))
        return best if losses[best] < losses[0]-margin*scale else 0
    if method=="EIC-Greedy":
        vals=losses/scale + penalty*costs
        best=int(np.argmin(vals))
        return best if vals[best] < vals[0]-margin else 0
    raise ValueError(method)


def hidden_variable(seed:int, control:bool, method:str, n_candidates:int=10, penalty=.04, margin=.015)->TrialResult:
    rng=np.random.default_rng(seed); n=600
    X=rng.normal(size=(n,max(n_candidates+2,5)))
    beta=0.0 if control else 2.0
    y=1.0*X[:,0]-1.2*X[:,1]+beta*X[:,2]+rng.normal(scale=.55,size=n)
    idx=rng.permutation(n); tr,va,te=idx[:320],idx[320:460],idx[460:]
    base=[0,1]
    cand=[base]+[base+[j] for j in range(2,2+n_candidates)]
    vloss=[]
    for cols in cand:
        m=LinearRegression().fit(X[tr][:,cols],y[tr]); vloss.append(mean_squared_error(y[va],m.predict(X[va][:,cols])))
    true_idx=0 if control else 1
    if method=="Oracle": sel=true_idx
    else: sel=_select([(x,len(cand[i])-2) for i,x in enumerate(vloss)],method,rng,costs=[0]+[1]*n_candidates,margin=margin,penalty=penalty)
    def tloss(i):
        cols=cand[i]; m=LinearRegression().fit(X[tr][:,cols],y[tr]); return mean_squared_error(y[te],m.predict(X[te][:,cols]))
    base_loss=tloss(0); loss=tloss(sel); scale=np.var(y[te]); p0,p1=_perf(base_loss,scale),_perf(loss,scale)
    cost=float(sel!=0); recovered=float(sel==true_idx)
    return TrialResult("core","hidden_variable","variable_addition",control,method,seed,"core",p1,loss,base_loss,recovered,cost,float(cost>0),_fde(p0,p1,cost),len(cand))


def ontology_split(seed:int, control:bool, method:str, penalty=.04, margin=.015)->TrialResult:
    rng=np.random.default_rng(seed); n=600
    coarse=rng.integers(0,2,size=n); marker=rng.normal(size=n)
    subtype=np.where(marker<0,-1.0,1.0)
    y=np.where(coarse==1,0.2,(0.2 if control else subtype))+rng.normal(scale=.3,size=n)
    idx=rng.permutation(n); tr,va,te=idx[:320],idx[320:460],idx[460:]
    thresholds=[None,-1.0,-.5,0,.5,1.0]
    def design(ix,t):
        if t is None: return np.column_stack([np.ones(len(ix)),coarse[ix]])
        a1=((coarse[ix]==0)&(marker[ix]<t)).astype(float); a2=((coarse[ix]==0)&(marker[ix]>=t)).astype(float); b=(coarse[ix]==1).astype(float)
        return np.column_stack([a1,a2,b])
    vloss=[]
    for t in thresholds:
        m=LinearRegression(fit_intercept=False).fit(design(tr,t),y[tr]); vloss.append(mean_squared_error(y[va],m.predict(design(va,t))))
    true_idx=0 if control else thresholds.index(0)
    if method=="Oracle": sel=true_idx
    else: sel=_select([(x,0 if i==0 else 1) for i,x in enumerate(vloss)],method,rng,costs=[0]+[1]*(len(thresholds)-1),margin=margin,penalty=penalty)
    def tloss(i):
        t=thresholds[i]; m=LinearRegression(fit_intercept=False).fit(design(tr,t),y[tr]); return mean_squared_error(y[te],m.predict(design(te,t)))
    base_loss=tloss(0); loss=tloss(sel); scale=np.var(y[te]); p0,p1=_perf(base_loss,scale),_perf(loss,scale); cost=float(sel!=0)
    return TrialResult("core","ontology_split","ontology_refinement",control,method,seed,"core",p1,loss,base_loss,float(sel==true_idx),cost,float(cost>0),_fde(p0,p1,cost),len(thresholds))


def causal_relation(seed:int, control:bool, method:str, penalty=.04, margin=.015)->TrialResult:
    rng=np.random.default_rng(seed); n_obs=500;n_int=300
    if control:
        x=rng.normal(size=n_obs); y=1.5*x+rng.normal(scale=.5,size=n_obs); xi=rng.normal(scale=1.4,size=n_int); yi=1.5*xi+rng.normal(scale=.5,size=n_int); true=1.5
    else:
        u=rng.normal(size=n_obs); x=u+rng.normal(scale=.45,size=n_obs); y=1.5*u+rng.normal(scale=.45,size=n_obs); u2=rng.normal(size=n_int); xi=rng.normal(scale=1.4,size=n_int); yi=1.5*u2+rng.normal(scale=.45,size=n_int); true=0.
    bo=float(LinearRegression().fit(x[:,None],y).coef_[0]); bi=float(LinearRegression().fit(xi[:,None],yi).coef_[0])
    candidates=[bo,bi]; losses=[(b-bi)**2 for b in candidates]  # interventional audit loss
    true_idx=0 if control else 1
    if method=="Oracle":
        sel=true_idx
    elif method in ("EIC-Greedy","EIC-NoMargin","EIC-NoCost"):
        resid=yi-bi*xi
        se=float(np.std(resid,ddof=1)/np.sqrt(np.sum((xi-xi.mean())**2)))
        threshold=(2.75 if method=="EIC-NoMargin" else 3.5)*max(se,0.02)
        sel=1 if abs(bo-bi)>threshold else 0
    else:
        sel=_select([(l,0 if i==0 else 1) for i,l in enumerate(losses)],method,rng,costs=[0,1],margin=margin,penalty=penalty)
    base_loss=(bo-true)**2; loss=(candidates[sel]-true)**2; p0,p1=_perf(base_loss,1),_perf(loss,1); cost=float(sel!=0)
    return TrialResult("core","causal_relation","relation_revision",control,method,seed,"core",p1,loss,base_loss,float(sel==true_idx),cost,float(cost>0),_fde(p0,p1,cost),2)


def wrong_objective(seed:int, control:bool, method:str, penalty=.04, margin=.015)->TrialResult:
    rng=np.random.default_rng(seed); true_opt=2.0 if control else -1.0; initial=2.0
    audit_a=np.linspace(-3,4,21); audit_u=-(audit_a-true_opt)**2+rng.normal(scale=.1,size=len(audit_a)); co=np.polyfit(audit_a,audit_u,2); est=float(np.clip(-co[1]/(2*co[0]),-3,4)) if co[0]<0 else initial
    candidates=[initial,est]; losses=[(a-true_opt)**2 for a in candidates]; true_idx=0 if control else 1
    if method=="Oracle": sel=true_idx
    else: sel=_select([(l,0 if i==0 else 1) for i,l in enumerate(losses)],method,rng,costs=[0,1],margin=margin,penalty=penalty)
    base_loss=losses[0]; loss=losses[sel]; p0,p1=_perf(base_loss,1),_perf(loss,1); cost=float(sel!=0)
    return TrialResult("core","wrong_objective","objective_revision",control,method,seed,"core",p1,loss,base_loss,float(sel==true_idx),cost,float(cost>0),_fde(p0,p1,cost),2)


def missing_action(seed:int, control:bool, method:str, penalty=.04, margin=.015)->TrialResult:
    rng=np.random.default_rng(seed); optimum=0.0 if control else 2.0; actions=np.array([-2.,0.,4.]); obs=-(actions-optimum)**2+rng.normal(scale=.04,size=3); initial=float(actions[np.argmax(obs)])
    co=np.polyfit(actions,obs,2); proposal=float(np.clip(-co[1]/(2*co[0]),-2,4)) if co[0]<0 else initial
    candidates=[initial,proposal]; losses=[(a-optimum)**2 for a in candidates]; true_idx=0 if control else 1
    if method=="Oracle": sel=true_idx
    else: sel=_select([(l,0 if i==0 else 1) for i,l in enumerate(losses)],method,rng,costs=[0,1],margin=margin,penalty=penalty)
    base_loss=losses[0]; loss=losses[sel]; p0,p1=_perf(base_loss,1),_perf(loss,1); cost=float(sel!=0)
    return TrialResult("core","missing_action","action_expansion",control,method,seed,"core",p1,loss,base_loss,float(sel==true_idx),cost,float(cost>0),_fde(p0,p1,cost),2)

CORE_FUNCS={"hidden_variable":hidden_variable,"ontology_split":ontology_split,"causal_relation":causal_relation,"wrong_objective":wrong_objective,"missing_action":missing_action}


def run_core(seeds=range(50), methods=METHODS, penalty=.04, margin=.015):
    rows=[]
    for name,fn in CORE_FUNCS.items():
        for control in (False,True):
            for method in methods:
                for seed in seeds:
                    kwargs={"penalty":penalty,"margin":margin}
                    if name=="hidden_variable": kwargs["n_candidates"]=10
                    rows.append(fn(int(seed),control,method,**kwargs).row())
    return pd.DataFrame(rows)


def run_scaling(seeds=range(40), candidate_counts=(2,5,10,20,50), methods=("Fixed-Frame","Random-Edit","Flat-Search","MDL-Flat","EIC-Greedy","Oracle"), penalty=.04, margin=.015):
    rows=[]
    for nc in candidate_counts:
        for method in methods:
            for seed in seeds:
                r=hidden_variable(int(seed),False,method,n_candidates=int(nc),penalty=penalty,margin=margin)
                d=r.row(); d["suite"]="scaling"; d["level"]=f"L2-k{nc}"; d["candidates"]=nc+1; rows.append(d)
    return pd.DataFrame(rows)


def run_multiedit(seeds=range(40), methods=("Fixed-Frame","EIC-Greedy","EIC-Beam","Oracle")):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); n=650; X=rng.normal(size=(n,10)); y=1.2*X[:,0]-1.0*X[:,1]+1.5*X[:,2]-1.4*X[:,3]+rng.normal(scale=.5,size=n)
        idx=rng.permutation(n); tr,va,te=idx[:350],idx[350:500],idx[500:]; base=[0,1]
        def mse(cols,ixfit=tr,ix=va):
            m=LinearRegression().fit(X[ixfit][:,cols],y[ixfit]); return mean_squared_error(y[ix],m.predict(X[ix][:,cols]))
        base_loss_test=mse(base,tr,te); scale=np.var(y[te])
        for method in methods:
            cols=base[:]
            if method=="Oracle": cols=[0,1,2,3]
            elif method=="EIC-Greedy":
                # one edit only by design
                best=(mse(cols),None)
                for j in range(2,10):
                    sc=mse(base+[j])+.04*np.var(y[va])
                    if sc<best[0]-.015*np.var(y[va]): best=(sc,j)
                if best[1] is not None: cols=base+[best[1]]
            elif method=="EIC-Beam":
                # beam over edit sequences up to two additions
                candidates=[base]
                for j in range(2,10): candidates.append(base+[j])
                for j,k in itertools.combinations(range(2,10),2): candidates.append(base+[j,k])
                vals=[]
                for c in candidates: vals.append(mse(c)/max(mse(base),1e-9)+.04*(len(c)-2))
                cols=candidates[int(np.argmin(vals))]
            loss=mse(cols,tr,te); p0,p1=_perf(base_loss_test,scale),_perf(loss,scale); cost=len(cols)-2; rec=float(set(cols)=={0,1,2,3})
            rows.append(TrialResult("multiedit","hidden_variable_pair","multi_edit_variable_addition",False,method,seed,"L3",p1,loss,base_loss_test,rec,float(cost),float(cost>0),_fde(p0,p1,cost),37).row())
    return pd.DataFrame(rows)


def run_symbolic(seeds=range(50), methods=("Fixed-Frame","Raw-Feature-Search","EIC-Symbolic","Oracle")):
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); n=600; X=rng.normal(size=(n,4)); y=1.5*(X[:,0]*X[:,1])+0.4*X[:,2]+rng.normal(scale=.4,size=n)
        idx=rng.permutation(n); tr,va,te=idx[:320],idx[320:460],idx[460:]
        base_cols=[2]
        base_m=LinearRegression().fit(X[tr][:,base_cols],y[tr]); base_loss=mean_squared_error(y[te],base_m.predict(X[te][:,base_cols])); scale=np.var(y[te])
        primitives=[]
        names=[]
        # generated candidates are not raw input variables; they are expressions from a grammar
        for i,j in itertools.combinations(range(4),2):
            primitives.append(X[:,i]*X[:,j]); names.append(f"x{i}*x{j}")
        for method in methods:
            chosen=None
            if method=="Oracle": chosen=names.index("x0*x1")
            elif method=="EIC-Symbolic":
                best=mean_squared_error(y[va],base_m.predict(X[va][:,base_cols]));
                for k,z in enumerate(primitives):
                    D=np.column_stack([X[:,2],z]); m=LinearRegression().fit(D[tr],y[tr]); sc=mean_squared_error(y[va],m.predict(D[va]))+.035*np.var(y[va])
                    if sc<best-.015*np.var(y[va]): best=sc;chosen=k
            elif method=="Raw-Feature-Search":
                best=mean_squared_error(y[va],base_m.predict(X[va][:,base_cols]));
                for j in [0,1,3]:
                    m=LinearRegression().fit(X[tr][:,[2,j]],y[tr]); sc=mean_squared_error(y[va],m.predict(X[va][:,[2,j]]))
                    if sc<best: best=sc;chosen=("raw",j)
            if isinstance(chosen,int):
                D=np.column_stack([X[:,2],primitives[chosen]]); m=LinearRegression().fit(D[tr],y[tr]); loss=mean_squared_error(y[te],m.predict(D[te])); cost=1.; rec=float(names[chosen]=="x0*x1")
            elif isinstance(chosen,tuple):
                j=chosen[1]; m=LinearRegression().fit(X[tr][:,[2,j]],y[tr]); loss=mean_squared_error(y[te],m.predict(X[te][:,[2,j]])); cost=1.; rec=0.
            else:
                loss=base_loss; cost=0.; rec=0.
            p0,p1=_perf(base_loss,scale),_perf(loss,scale)
            rows.append(TrialResult("symbolic","symbolic_generation","generated_variable",False,method,seed,"L4",p1,loss,base_loss,rec,cost,float(cost>0),_fde(p0,p1,cost),len(primitives)+1).row())
    return pd.DataFrame(rows)


def run_realdata(seeds=range(30), methods=("Fixed-Frame","Flat-Search","MDL-Flat","EIC-Greedy","Oracle"), max_candidates=28, penalty=.012, margin=.003):
    data=load_breast_cancer(); X=data.data; y=data.target
    rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed); idx=rng.permutation(len(y)); tr,va,te=idx[:330],idx[330:450],idx[450:]
        base=[0,1]  # mean radius, mean texture
        remaining=list(range(2,min(X.shape[1],2+max_candidates)))
        def fitloss(cols,train=tr,test=va):
            m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1500,solver="liblinear")); m.fit(X[train][:,cols],y[train]); pr=m.predict_proba(X[test][:,cols])[:,1]; return log_loss(y[test],pr,labels=[0,1]),m
        base_val,_=fitloss(base); candidates=[base]+[base+[j] for j in remaining]
        vals=[fitloss(c)[0] for c in candidates]
        # Oracle for this stress test is empirical best candidate on validation, not a claimed true frame.
        oracle_idx=int(np.argmin(vals))
        for method in methods:
            if method=="Oracle": sel=oracle_idx
            else: sel=_select([(v,0 if i==0 else 1) for i,v in enumerate(vals)],method,rng,costs=[0]+[1]*(len(candidates)-1),margin=margin,penalty=penalty)
            cols=candidates[sel]; _,m=fitloss(cols); pr=m.predict_proba(X[te][:,cols])[:,1]; loss=log_loss(y[te],pr,labels=[0,1]); auc=roc_auc_score(y[te],pr)
            _,m0=fitloss(base); pr0=m0.predict_proba(X[te][:,base])[:,1]; base_loss=log_loss(y[te],pr0,labels=[0,1]); p0,p1=_perf(base_loss,1),_perf(loss,1); cost=float(sel!=0)
            rows.append({**TrialResult("realdata","breast_cancer_wisconsin","real_data_variable_addition",False,method,seed,"real",p1,loss,base_loss,float(sel==oracle_idx),cost,float(cost>0),_fde(p0,p1,cost),len(candidates)).row(),"auc":float(auc),"selected_feature":data.feature_names[cols[-1]] if sel!=0 else "none"})
    return pd.DataFrame(rows)


def run_all():
    return {"core":run_core(),"scaling":run_scaling(),"multiedit":run_multiedit(),"symbolic":run_symbolic(),"realdata":run_realdata()}
