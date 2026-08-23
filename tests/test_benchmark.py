import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from eic.benchmark import run_suite

def test_suite_shape_and_bounds():
    df=run_suite(range(2))
    assert len(df)==5*2*3*2
    assert df.performance.between(0,1).all()
    assert set(df.method)=={'Fixed-Frame','EIC-Greedy','Oracle'}

def test_eic_improves_defective_frames():
    df=run_suite(range(10))
    d=df[~df.control]
    e=d[d.method=='EIC-Greedy'].performance.mean()
    f=d[d.method=='Fixed-Frame'].performance.mean()
    assert e > f + 0.10

def test_eic_control_false_reframe_low():
    df=run_suite(range(10))
    c=df[(df.control)&(df.method=='EIC-Greedy')]
    assert c.reframed.mean() < 0.15
