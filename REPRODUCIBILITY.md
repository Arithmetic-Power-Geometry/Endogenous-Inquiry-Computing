# Reproducibility

## Determinism
All reported stochastic experiments use deterministic integer seeds. The main experiment script regenerates raw CSV files, aggregate tables, statistical comparisons, and vector figures from source.

## Main command
```bash
python scripts/run_experiments.py
```

## Test command
```bash
pytest -q
```

## Expected headline results
- Core defective frames: Fixed-Frame mean performance about 0.360; EIC-Greedy about 0.964.
- Core controls: EIC-Greedy false-reframe rate 0.000 in the configured 250 control instances.
- Two-edit L3: one-edit EIC-Greedy does not exactly recover the required two-edit frame; EIC-Beam does.
- Symbolic L4: EIC-Symbolic generates the required interaction in all configured seeds; raw-feature search does not.
- Wisconsin stress test: two-feature initial-frame ROC AUC about 0.947; selected one-feature expansion about 0.987.

These values are regenerated rather than hard-coded into the program logic.
