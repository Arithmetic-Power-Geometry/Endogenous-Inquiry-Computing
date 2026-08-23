from eic.benchmark_v2 import hidden_variable, run_scaling, run_symbolic, run_realdata

def test_hidden_variable_oracle_beats_fixed():
    a=hidden_variable(1,False,'Fixed-Frame',n_candidates=10)
    b=hidden_variable(1,False,'Oracle',n_candidates=10)
    assert b.performance > a.performance
    assert b.frame_recovered == 1

def test_control_conservatism():
    r=hidden_variable(3,True,'EIC-Greedy',n_candidates=10)
    assert 0 <= r.performance <= 1

def test_scaling_runs():
    df=run_scaling(seeds=range(2),candidate_counts=(2,5),methods=('Fixed-Frame','EIC-Greedy'))
    assert len(df)==8

def test_symbolic_generation_runs():
    df=run_symbolic(seeds=range(2),methods=('Fixed-Frame','EIC-Symbolic'))
    assert set(df.method)=={'Fixed-Frame','EIC-Symbolic'}

def test_realdata_runs():
    df=run_realdata(seeds=range(2),methods=('Fixed-Frame','EIC-Greedy'),max_candidates=8)
    assert df.auc.between(0,1).all()
