from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
import gradio as gr
import pandas as pd
from eic.benchmark_v2 import hidden_variable, ontology_split, causal_relation, wrong_objective, missing_action, run_scaling, run_symbolic, run_realdata, METHODS

FUNCS={'Hidden variable':hidden_variable,'Ontology split':ontology_split,'Causal relation':causal_relation,'Wrong objective':wrong_objective,'Missing action':missing_action}

def single(family,method,seed,control,candidates,penalty,margin):
    fn=FUNCS[family]; kw={'penalty':penalty,'margin':margin}
    if family=='Hidden variable': kw['n_candidates']=int(candidates)
    r=fn(int(seed),bool(control),method,**kw)
    return pd.DataFrame([r.row()])

def scaling(max_candidates,seed_count,penalty,margin):
    counts=[2,5,10,20,int(max_candidates)]
    counts=sorted(set(c for c in counts if c>=2))
    df=run_scaling(seeds=range(int(seed_count)),candidate_counts=counts,penalty=penalty,margin=margin)
    return df.groupby(['candidates','method'])[['performance','frame_recovered']].mean().reset_index()

def symbolic(seed_count):
    df=run_symbolic(seeds=range(int(seed_count)))
    return df.groupby('method')[['performance','frame_recovered','reframe_cost']].mean().reset_index()

def real(seed_count,candidates,penalty,margin):
    df=run_realdata(seeds=range(int(seed_count)),max_candidates=int(candidates),penalty=penalty,margin=margin)
    return df.groupby('method')[['performance','loss','auc','reframed']].mean().reset_index()

with gr.Blocks(title='Endogenous Inquiry Computing Lab') as demo:
    gr.Markdown('# Endogenous Inquiry Computing Lab\nInteractively inspect frame edits, penalties, margins, candidate-set size, synthetic stress tests, symbolic candidate generation, and a real-data stress test.')
    with gr.Tab('Single frame trial'):
        fam=gr.Dropdown(list(FUNCS),value='Hidden variable',label='Frame defect family')
        method=gr.Dropdown(list(METHODS),value='EIC-Greedy',label='Method')
        seed=gr.Slider(0,999,value=0,step=1,label='Seed')
        control=gr.Checkbox(False,label='Well-specified control')
        candidates=gr.Slider(2,50,value=10,step=1,label='Candidate additions (hidden-variable family)')
        penalty=gr.Slider(0,0.2,value=.04,step=.005,label='Frame-edit penalty')
        margin=gr.Slider(0,0.1,value=.015,step=.005,label='Acceptance margin')
        btn=gr.Button('Run trial'); out=gr.Dataframe(); btn.click(single,[fam,method,seed,control,candidates,penalty,margin],out)
    with gr.Tab('Difficulty scaling'):
        mc=gr.Slider(10,100,value=50,step=10,label='Largest candidate set'); ns=gr.Slider(5,100,value=40,step=5,label='Seeds'); p=gr.Slider(0,0.2,value=.04,step=.005,label='Penalty'); m=gr.Slider(0,0.1,value=.015,step=.005,label='Margin'); b=gr.Button('Run scaling sweep'); o=gr.Dataframe(); b.click(scaling,[mc,ns,p,m],o)
    with gr.Tab('Candidate generation'):
        n=gr.Slider(5,100,value=50,step=5,label='Seeds'); b2=gr.Button('Run symbolic generation'); o2=gr.Dataframe(); b2.click(symbolic,n,o2)
    with gr.Tab('Real-data stress test'):
        n3=gr.Slider(5,50,value=30,step=5,label='Random splits'); c3=gr.Slider(5,28,value=28,step=1,label='Candidate features'); p3=gr.Slider(0,0.05,value=.012,step=.001,label='Penalty'); m3=gr.Slider(0,0.02,value=.003,step=.001,label='Margin'); b3=gr.Button('Run Wisconsin breast-cancer stress test'); o3=gr.Dataframe(); b3.click(real,[n3,c3,p3,m3],o3)
    gr.Markdown('Repository: https://github.com/Arithmetic-Power-Geometry/Endogenous-Inquiry-Computing')

if __name__=='__main__': demo.launch()
