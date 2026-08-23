from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
from eic import run_all

out=ROOT/'results'; (out/'tables').mkdir(parents=True,exist_ok=True); (out/'figures').mkdir(parents=True,exist_ok=True)
res=run_all()
for name,df in res.items(): df.to_csv(out/f'{name}_raw.csv',index=False)

core=res['core']; defective=core[~core.control]; controls=core[core.control]
agg=defective.groupby('method').agg(performance=('performance','mean'),frame_recovery=('frame_recovered','mean'),fde=('fde','mean'),reframe_rate=('reframed','mean')).reset_index()
ctrl=controls.groupby('method').agg(performance=('performance','mean'),false_reframe_rate=('reframed','mean'),correct_frame_decision=('frame_recovered','mean')).reset_index()
byfam=defective.groupby(['benchmark','method']).agg(performance=('performance','mean'),sd=('performance','std'),frame_recovery=('frame_recovered','mean')).reset_index()
agg.to_csv(out/'tables'/'core_aggregate.csv',index=False); ctrl.to_csv(out/'tables'/'core_controls.csv',index=False); byfam.to_csv(out/'tables'/'core_by_family.csv',index=False)

# paired EIC vs fixed
piv=defective.pivot_table(index=['benchmark','seed'],columns='method',values='performance')
a=piv['EIC-Greedy'].to_numpy(); b=piv['Fixed-Frame'].to_numpy(); diff=a-b
stats=pd.DataFrame([{'comparison':'EIC-Greedy - Fixed-Frame','mean_difference':diff.mean(),'sd_difference':diff.std(ddof=1),'wilcoxon_p':wilcoxon(a,b).pvalue,'ttest_t':ttest_rel(a,b).statistic,'ttest_p':ttest_rel(a,b).pvalue,'n':len(diff)}])
stats.to_csv(out/'tables'/'paired_statistics.csv',index=False)

sc=res['scaling'].groupby(['candidates','method']).agg(performance=('performance','mean'),frame_recovery=('frame_recovered','mean')).reset_index(); sc.to_csv(out/'tables'/'scaling.csv',index=False)
me=res['multiedit'].groupby('method').agg(performance=('performance','mean'),frame_recovery=('frame_recovered','mean'),mean_cost=('reframe_cost','mean')).reset_index(); me.to_csv(out/'tables'/'multiedit.csv',index=False)
sy=res['symbolic'].groupby('method').agg(performance=('performance','mean'),frame_recovery=('frame_recovered','mean'),mean_cost=('reframe_cost','mean')).reset_index(); sy.to_csv(out/'tables'/'symbolic.csv',index=False)
rd=res['realdata'].groupby('method').agg(performance=('performance','mean'),log_loss=('loss','mean'),auc=('auc','mean'),edit_rate=('reframed','mean')).reset_index(); rd.to_csv(out/'tables'/'realdata.csv',index=False)

# figures, matplotlib defaults only
order=['Fixed-Frame','Random-Edit','Flat-Search','MDL-Flat','EIC-NoMargin','EIC-NoCost','EIC-Greedy','Oracle']
plot=byfam.pivot(index='benchmark',columns='method',values='performance').reindex(columns=[m for m in order if m in byfam.method.unique()])
ax=plot.plot(kind='bar',figsize=(10,5)); ax.set_ylabel('Normalized performance'); ax.set_xlabel('Defective initial frame'); ax.set_ylim(0,1.05); plt.tight_layout(); plt.savefig(out/'figures'/'core_performance.pdf'); plt.savefig(out/'figures'/'core_performance.png',dpi=180); plt.close()

plot2=pd.concat([defective.groupby('method').frame_recovered.mean().rename('Defective frames'),controls.groupby('method').frame_recovered.mean().rename('Well-specified controls')],axis=1).reindex([m for m in order if m in core.method.unique()])
ax=plot2.plot(kind='bar',figsize=(9,5)); ax.set_ylabel('Correct frame decision rate'); ax.set_xlabel('Method'); ax.set_ylim(0,1.05); plt.tight_layout(); plt.savefig(out/'figures'/'frame_decisions.pdf'); plt.savefig(out/'figures'/'frame_decisions.png',dpi=180); plt.close()

for metric,fn,ylab in [('performance','scaling_performance','Normalized performance'),('frame_recovery','scaling_recovery','Frame recovery rate')]:
    p=sc.pivot(index='candidates',columns='method',values=metric)
    ax=p.plot(marker='o',figsize=(8,5)); ax.set_xlabel('Candidate frames'); ax.set_ylabel(ylab); ax.set_ylim(0,1.05); plt.tight_layout(); plt.savefig(out/'figures'/f'{fn}.pdf'); plt.savefig(out/'figures'/f'{fn}.png',dpi=180); plt.close()

ax=me.set_index('method')[['performance','frame_recovery']].plot(kind='bar',figsize=(7,4)); ax.set_ylim(0,1.05); ax.set_ylabel('Rate / performance'); plt.tight_layout(); plt.savefig(out/'figures'/'multiedit.pdf'); plt.close()
ax=sy.set_index('method')[['performance','frame_recovery']].plot(kind='bar',figsize=(7,4)); ax.set_ylim(0,1.05); ax.set_ylabel('Rate / performance'); plt.tight_layout(); plt.savefig(out/'figures'/'symbolic_generation.pdf'); plt.close()
ax=rd.set_index('method')[['auc']].plot(kind='bar',figsize=(7,4),legend=False); ax.set_ylim(.5,1.0); ax.set_ylabel('ROC AUC'); plt.tight_layout(); plt.savefig(out/'figures'/'realdata_auc.pdf'); plt.close()

print('Core aggregate')
print(agg.to_string(index=False))
print('\nControls')
print(ctrl.to_string(index=False))
print('\nScaling')
print(sc.to_string(index=False))
print('\nMultiedit')
print(me.to_string(index=False))
print('\nSymbolic')
print(sy.to_string(index=False))
print('\nReal data')
print(rd.to_string(index=False))
print('\nPaired stats')
print(stats.to_string(index=False))
