from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans

@dataclass
class TrialResult:
    benchmark: str
    defect: str
    control: bool
    method: str
    seed: int
    performance: float
    loss: float
    frame_recovered: float
    reframe_cost: float
    reframed: float
    fde: float

    def row(self):
        return asdict(self)


def _perf_from_loss(loss: float, scale: float = 1.0) -> float:
    return float(1.0 / (1.0 + max(loss, 0.0) / scale))


def _fde(p0: float, p1: float, cost: float) -> float:
    return float(max(0.0, p1 - p0) / (1.0 + cost))


def hidden_variable(seed: int, control: bool, method: str) -> TrialResult:
    rng = np.random.default_rng(seed)
    n = 480
    X = rng.normal(size=(n, 4))
    beta3 = 0.0 if control else 2.2
    y = 1.0*X[:,0] - 1.3*X[:,1] + beta3*X[:,2] + rng.normal(scale=.45, size=n)
    idx = rng.permutation(n)
    tr, va, te = idx[:260], idx[260:360], idx[360:]
    base_cols = [0,1]
    m0 = LinearRegression().fit(X[tr][:,base_cols], y[tr])
    base_loss = mean_squared_error(y[te], m0.predict(X[te][:,base_cols]))
    selected = base_cols.copy()
    cost = 0.0
    recovered = 1.0 if control else 0.0
    if method == "EIC-Greedy":
        best_cols = selected
        best_score = mean_squared_error(y[va], m0.predict(X[va][:,selected]))
        penalty = 0.045 * np.var(y[va])
        for c in [2,3]:
            cols = base_cols + [c]
            m = LinearRegression().fit(X[tr][:,cols], y[tr])
            score = mean_squared_error(y[va], m.predict(X[va][:,cols])) + penalty
            if score < best_score - 0.015*np.var(y[va]):
                best_score, best_cols = score, cols
        selected = best_cols
        cost = float(len(selected)-2)
        recovered = float((control and selected==base_cols) or ((not control) and selected==[0,1,2]))
    elif method == "Oracle":
        selected = base_cols if control else [0,1,2]
        cost = float(len(selected)-2)
        recovered = 1.0
    m = LinearRegression().fit(X[tr][:,selected], y[tr])
    loss = mean_squared_error(y[te], m.predict(X[te][:,selected]))
    scale = np.var(y[te])
    p0, p1 = _perf_from_loss(base_loss, scale), _perf_from_loss(loss, scale)
    return TrialResult("hidden_variable", "variable_addition", control, method, seed, p1, loss, recovered, cost, float(cost>0), _fde(p0,p1,cost))


def ontology_split(seed: int, control: bool, method: str) -> TrialResult:
    rng = np.random.default_rng(seed)
    n = 480
    coarse = rng.integers(0,2,size=n)  # 0=A, 1=B
    marker = rng.normal(size=n)
    # A contains two latent subtypes separated by marker sign.
    subtype = np.where(marker<0, -1.0, 1.0)
    y = np.where(coarse==1, 0.25, (0.25 if control else subtype)) + rng.normal(scale=.25,size=n)
    idx = rng.permutation(n)
    tr, va, te = idx[:260], idx[260:360], idx[360:]

    def design(ix, split=False, threshold=0.0):
        if not split:
            return np.column_stack([np.ones(len(ix)), coarse[ix]])
        a1 = ((coarse[ix]==0)&(marker[ix]<threshold)).astype(float)
        a2 = ((coarse[ix]==0)&(marker[ix]>=threshold)).astype(float)
        b = (coarse[ix]==1).astype(float)
        return np.column_stack([a1,a2,b])

    b0 = LinearRegression(fit_intercept=False).fit(design(tr),y[tr])
    base_loss = mean_squared_error(y[te], b0.predict(design(te)))
    split=False; threshold=0.0; cost=0.0; recovered=1.0 if control else 0.0
    if method=="EIC-Greedy":
        base_val=mean_squared_error(y[va],b0.predict(design(va)))
        best=(base_val,False,0.0)
        penalty=.025*np.var(y[va])
        for t in [-0.75,0.0,0.75]:
            mm=LinearRegression(fit_intercept=False).fit(design(tr,True,t),y[tr])
            sc=mean_squared_error(y[va],mm.predict(design(va,True,t)))+penalty
            if sc < best[0]-0.02*np.var(y[va]): best=(sc,True,t)
        _,split,threshold=best
        cost=float(split)
        recovered=float((control and not split) or ((not control) and split and abs(threshold)<1e-9))
    elif method=="Oracle":
        split=not control; threshold=0.0; cost=float(split); recovered=1.0
    mm=LinearRegression(fit_intercept=False).fit(design(tr,split,threshold),y[tr])
    loss=mean_squared_error(y[te],mm.predict(design(te,split,threshold)))
    scale=np.var(y[te]); p0,p1=_perf_from_loss(base_loss,scale),_perf_from_loss(loss,scale)
    return TrialResult("ontology_split","ontology_refinement",control,method,seed,p1,loss,recovered,cost,float(cost>0),_fde(p0,p1,cost))


def causal_relation(seed:int, control:bool, method:str)->TrialResult:
    rng=np.random.default_rng(seed)
    n_obs=500; n_int=300
    if control:
        x_obs=rng.normal(size=n_obs); y_obs=1.5*x_obs+rng.normal(scale=.5,size=n_obs)
        x_int=rng.normal(scale=1.5,size=n_int); y_int=1.5*x_int+rng.normal(scale=.5,size=n_int)
        true_effect=1.5
    else:
        u=rng.normal(size=n_obs); x_obs=u+rng.normal(scale=.45,size=n_obs); y_obs=1.5*u+rng.normal(scale=.45,size=n_obs)
        u2=rng.normal(size=n_int); x_int=rng.normal(scale=1.5,size=n_int); y_int=1.5*u2+rng.normal(scale=.45,size=n_int)
        true_effect=0.0
    beta_obs=float(LinearRegression().fit(x_obs.reshape(-1,1),y_obs).coef_[0])
    beta_int=float(LinearRegression().fit(x_int.reshape(-1,1),y_int).coef_[0])
    base_loss=(beta_obs-true_effect)**2
    chosen=beta_obs; cost=0.; recovered=1.0 if control else 0.0
    if method=="EIC-Greedy":
        # Intervention evidence challenges the observational causal frame.
        se=np.std(y_int-beta_int*x_int)/np.sqrt(np.sum((x_int-x_int.mean())**2))
        incompatible=abs(beta_obs-beta_int)>3.0*max(se,0.02)
        if incompatible:
            chosen=beta_int; cost=1.
        recovered=float((control and cost==0) or ((not control) and cost==1))
    elif method=="Oracle":
        chosen=true_effect; cost=float(not control); recovered=1.0
    loss=(chosen-true_effect)**2
    p0,p1=_perf_from_loss(base_loss,1.0),_perf_from_loss(loss,1.0)
    return TrialResult("causal_relation","relation_revision",control,method,seed,p1,loss,recovered,cost,float(cost>0),_fde(p0,p1,cost))


def wrong_objective(seed:int, control:bool, method:str)->TrialResult:
    rng=np.random.default_rng(seed)
    # Proxy optimum is +2. True desideratum is -1 unless control is aligned.
    def proxy(a): return -(a-2.0)**2
    true_opt=2.0 if control else -1.0
    def utility(a): return -(a-true_opt)**2
    initial=2.0
    base_loss=(initial-true_opt)**2
    chosen=initial; cost=0.; recovered=1.0 if control else 0.0
    # Audit observations reveal realized utility at sampled actions.
    audit_a=np.linspace(-3,4,15)
    audit_u=np.array([utility(a) for a in audit_a])+rng.normal(scale=.08,size=len(audit_a))
    if method=="EIC-Greedy":
        # Fit a quadratic audited objective; switch only if proxy optimum is decisively contradicted.
        co=np.polyfit(audit_a,audit_u,2)
        est=-co[1]/(2*co[0]) if co[0]<0 else initial
        est=float(np.clip(est,-3,4))
        pred_gain=np.polyval(co,est)-np.polyval(co,initial)
        if pred_gain>0.5:
            chosen=est; cost=1.
        recovered=float((control and cost==0) or ((not control) and cost==1 and abs(chosen-true_opt)<0.25))
    elif method=="Oracle":
        chosen=true_opt; cost=float(not control); recovered=1.0
    loss=(chosen-true_opt)**2
    p0,p1=_perf_from_loss(base_loss,1.0),_perf_from_loss(loss,1.0)
    return TrialResult("wrong_objective","objective_revision",control,method,seed,p1,loss,recovered,cost,float(cost>0),_fde(p0,p1,cost))


def missing_action(seed:int, control:bool, method:str)->TrialResult:
    rng=np.random.default_rng(seed)
    optimum=0.0 if control else 2.0
    def reward(a): return -(a-optimum)**2
    actions=np.array([-2.0,0.0,4.0])
    obs=np.array([reward(a) for a in actions])+rng.normal(scale=.03,size=3)
    initial=float(actions[np.argmax(obs)])
    base_loss=(initial-optimum)**2
    chosen=initial; cost=0.; recovered=1.0 if control else 0.0
    if method=="EIC-Greedy":
        # Infer a missing action by fitting the simplest concave quadratic to available action outcomes.
        co=np.polyfit(actions,obs,2)
        proposal=-co[1]/(2*co[0]) if co[0]<0 else initial
        proposal=float(np.clip(proposal,-2,4))
        predicted=np.polyval(co,proposal)
        if predicted > obs.max()+0.35:
            chosen=proposal; cost=1.
        recovered=float((control and cost==0) or ((not control) and cost==1 and abs(chosen-optimum)<0.2))
    elif method=="Oracle":
        chosen=optimum; cost=float(not control); recovered=1.0
    loss=(chosen-optimum)**2
    p0,p1=_perf_from_loss(base_loss,1.0),_perf_from_loss(loss,1.0)
    return TrialResult("missing_action","action_expansion",control,method,seed,p1,loss,recovered,cost,float(cost>0),_fde(p0,p1,cost))

BENCHMARKS={
    "hidden_variable":hidden_variable,
    "ontology_split":ontology_split,
    "causal_relation":causal_relation,
    "wrong_objective":wrong_objective,
    "missing_action":missing_action,
}
METHODS=("Fixed-Frame","EIC-Greedy","Oracle")

def run_trial(benchmark:str,seed:int,control:bool=False,method:str="EIC-Greedy"):
    return BENCHMARKS[benchmark](seed,control,method)

def run_suite(seeds=range(50))->pd.DataFrame:
    rows=[]
    for name,fn in BENCHMARKS.items():
        for control in (False,True):
            for method in METHODS:
                for seed in seeds:
                    rows.append(fn(int(seed),control,method).row())
    return pd.DataFrame(rows)
