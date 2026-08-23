from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
required=[
 'results/tables/core_aggregate.csv','results/tables/core_controls.csv',
 'results/tables/scaling.csv','results/tables/multiedit.csv',
 'results/tables/symbolic.csv','results/tables/realdata.csv',
 'results/figures/core_performance.pdf','results/figures/frame_decisions.pdf',
 'results/figures/scaling_recovery.pdf','results/figures/multiedit.pdf',
 'results/figures/symbolic_generation.pdf','results/figures/realdata_auc.pdf']
missing=[p for p in required if not (ROOT/p).exists()]
if missing: raise SystemExit('Missing outputs: '+', '.join(missing))
core=pd.read_csv(ROOT/'results/tables/core_aggregate.csv')
assert core.performance.between(0,1).all()
controls=pd.read_csv(ROOT/'results/tables/core_controls.csv')
assert controls.false_reframe_rate.between(0,1).all()
print('Result audit passed:',len(required),'required artifacts present.')
