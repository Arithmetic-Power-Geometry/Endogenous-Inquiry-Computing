# Endogenous Inquiry Computing (EIC)

Reproducible reference implementation for **Problem Frames as Computational States: A Theory of Endogenous Inquiry**.

Repository: https://github.com/Arithmetic-Power-Geometry/Endogenous-Inquiry-Computing

## What this package tests

EIC treats a task-defining problem frame as a computational state that may be revised when evidence justifies changing variables, ontology, causal relations, objectives, actions, or information interfaces. EICBench v2 contains:

- five paired core frame-defect families with well-specified controls;
- candidate-neighborhood scaling from 3 to 51 frames;
- a two-edit challenge where one-edit EIC-Greedy is expected to fail exact recovery;
- symbolic construction of a missing interaction variable from a primitive grammar;
- a real-data stress test using the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn;
- baseline and ablation comparisons (Fixed-Frame, Random-Edit, Flat-Search, MDL-Flat, EIC-NoMargin, EIC-NoCost, EIC-Greedy, Oracle);
- an interactive Gradio laboratory.

## One-click GitHub reproduction

Open **Actions -> reproduce-eic -> Run workflow**. The workflow installs dependencies, runs all tests, regenerates every experiment, table and figure, and uploads `results/` as a workflow artifact.

## Local reproduction

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest -q
python scripts/run_experiments.py
```

## Interactive app

```bash
python app.py
```

The app lets you vary the frame-defect family, method, random seed, control/defect state, candidate-set size, edit penalty, acceptance margin, symbolic-generation seed count, and real-data candidate count.

## Reproducibility boundary

The core and hard synthetic benchmarks are deliberately constructed so frame ground truth is known. The Wisconsin experiment is a **real-data stress test**, not evidence that EIC autonomously discovers a true scientific ontology. The manuscript also proves a static-superframe flattening result: if every future frame component is freely available from the start, frame search may collapse to ordinary model selection.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
