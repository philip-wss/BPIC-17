# BPIC-17: Process Model Discovery and Drift Analysis

Code for the first assignment of *Business Process Prediction, Simulation, and Optimization* (TUM, WS 26/27). The task is to derive a process model from the [BPI Challenge 2017](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) event log (31,509 loan applications). The model should serve as the base for a later simulation. The analysis also checks whether arrival rate, case duration and process behaviour drift over time.

The project covers:

1. **Simple event log analysis** (`notebooks/simple_event_log_analysis.ipynb`): log statistics and diagrams.
2. **Process model creation and validation** (`notebooks/quality_metrics.ipynb`):
   - Inductive and heuristics miners.
   - A sweep over the inductive miner's noise threshold.
   - 18 iterations of a hand-made BPMN model (`data/manual/BPIC_v1.bpmn` to `BPIC_v18.bpmn`).
   - PM4Py quality metrics, plus own simplicity metrics (density, node count).
3. **Advanced analysis** (`notebooks/advanced_analysis_drift.ipynb`): data and concept drift relevant to a simulation model.

## Setup

Python 3.14 is used (see `requirements.txt` for pinned versions, including `pm4py==2.7.23.11`).

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The [Graphviz](https://graphviz.org/download/) program `dot` must be installed on the system (it is used to render the Petri nets as SVG).

Download the event log from the [4TU repository](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b) and place it at `data/BPI_Challenge_2017.xes`. It is not part of the repository.

## Reproducing the results

Run all commands from the repository root.

```bash
# 1. discover inductive and heuristics models (complete cases, collapsed work items)
python -m src.miners

# 2. quality metrics for both models (--skip-alignments for a quick run)
python -m src.metrics

# 3. inductive miner with different noise thresholds
python -m src.inductive_thresholds

# 4. evaluate the manual BPMN models (alignment metrics are slow)
python -m src.manual_models --all [--with-alignments]
```

Results are written to `reports/<model>/` as PNML, SVG, `metrics.json` and `metrics.csv`. Step 2 needs the models from step 1, so run those two in order. Steps 3 and 4 are independent: step 3 discovers its own models and step 4 converts the BPMN files in `data/manual/`. `notebooks/quality_metrics.ipynb` reads the results of steps 1 and 2. The other notebooks only need the event log.
