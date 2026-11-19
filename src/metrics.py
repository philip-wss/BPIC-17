import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd
import pm4py

from src import custom_metrics
from src.config import INITIAL_DISCOVERY
from src.load import load_events
from src.miners import _KEYS, PetriNetModel, load_petri_net_pnml
from src.preprocessing import filter_complete_cases, collapse_workitems

MINERS = ("inductive", "heuristics")
RESULTS_JSON = INITIAL_DISCOVERY / "metrics.json"



""" Same log as in src/miners.py: complete cases, one event per started work
    item. Cached, so it is only built once per process.
"""
@cache
def load_log() -> pd.DataFrame:
    return collapse_workitems(filter_complete_cases(load_events()), drop_withdrawn=True)


""" Accepts either a miner name ("inductive", "heuristics"), which loads
    reports/petri_net_<name>.pnml, or an already loaded (net, im, fm) tuple.
"""
def _as_model(model: str | PetriNetModel) -> PetriNetModel:
    return load_petri_net_pnml(model) if isinstance(model, str) else model


def _as_log(log):
    return load_log() if log is None else log



""" Number of places, transitions (incl. silent ones), silent transitions and arcs. """
def size(model) -> dict:
    net = _as_model(model)[0]
    return dict(places=len(net.places), transitions=len(net.transitions),
                silent_transitions=sum(t.label is None for t in net.transitions),
                arcs=len(net.arcs))


""" Woflan soundness check. """
def soundness(model) -> dict:
    return dict(sound=bool(pm4py.check_soundness(*_as_model(model))[0]))


""" arc_degree in [0, 1], higher is simpler. """
def simplicity_arc_degree(model) -> dict:
    return dict(simplicity_arc_degree=pm4py.simplicity_petri_net(*_as_model(model), variant="arc_degree"))


""" extended_cardoso, higher is more complex. """
def simplicity_extended_cardoso(model) -> dict:
    return dict(simplicity_extended_cardoso=pm4py.simplicity_petri_net(*_as_model(model), variant="extended_cardoso"))


""" extended_cyclomatic, higher is more complex. """
def simplicity_extended_cyclomatic(model) -> dict:
    return dict(simplicity_extended_cyclomatic=pm4py.simplicity_petri_net(*_as_model(model), variant="extended_cyclomatic"))


""" |A| / (2·|P|·|T|), see custom_metrics.density. """
def simplicity_density(model) -> dict:
    return dict(simplicity_density=custom_metrics.density(_as_model(model)[0]))


""" |P| + |T|, see custom_metrics.node_count. """
def node_count(model) -> dict:
    return dict(node_count=custom_metrics.node_count(_as_model(model)[0]))


""" Token-based replay fitness: log_fitness and the share of fitting traces. """
def fitness_tbr(model, log=None) -> dict:
    fit = pm4py.fitness_token_based_replay(_as_log(log), *_as_model(model), **_KEYS)
    return dict(fitness_tbr=fit["log_fitness"], fit_traces_tbr=fit["perc_fit_traces"] / 100)


""" Alignment-based fitness. Veeery slow. """
def fitness_alignments(model, log=None, multi_processing: bool = True) -> dict:
    fit = pm4py.fitness_alignments(_as_log(log), *_as_model(model),
                                   multi_processing=multi_processing, **_KEYS)
    return dict(fitness_align=fit["log_fitness"], fit_traces_align=fit["percFitTraces"] / 100)


""" ETConformance precision on token-based replay. """
def precision_tbr(model, log=None) -> dict:
    return dict(precision_tbr=pm4py.precision_token_based_replay(_as_log(log), *_as_model(model), **_KEYS))


""" Align-ETConformance precision. Slow. """
def precision_alignments(model, log=None, multi_processing: bool = True) -> dict:
    return dict(precision_align=pm4py.precision_alignments(
        _as_log(log), *_as_model(model), multi_processing=multi_processing, **_KEYS))


""" Token-based generalization: rarely fired transitions lower the score. """
def generalization_tbr(model, log=None) -> dict:
    return dict(generalization_tbr=pm4py.generalization_tbr(_as_log(log), *_as_model(model), **_KEYS))


""" Harmonic mean of fitness and precision. """
def f_score(fitness: float, precision: float) -> float:
    return 2 * fitness * precision / (fitness + precision) if fitness + precision else 0.0

""" How run_all treats a metric:
    - needs_log:   called with the event log
    - slow:        skipped with skip_alignments=True
    - needs_sound: only computed for sound nets (NaN otherwise)
"""
@dataclass(frozen=True)
class Metric:
    func: callable
    needs_log: bool = False
    slow: bool = False
    needs_sound: bool = False


# Cheap metrics first, alignments last, so a crash in a slow metric loses as little
# as possible.
METRICS = {
    "size": Metric(size),
    "soundness": Metric(soundness),
    "simplicity_arc_degree": Metric(simplicity_arc_degree),
    "simplicity_extended_cardoso": Metric(simplicity_extended_cardoso),
    "simplicity_extended_cyclomatic": Metric(simplicity_extended_cyclomatic, needs_sound=True),
    "simplicity_density": Metric(simplicity_density),
    "node_count": Metric(node_count),
    "fitness_tbr": Metric(fitness_tbr, needs_log=True),
    "precision_tbr": Metric(precision_tbr, needs_log=True),
    "generalization_tbr": Metric(generalization_tbr, needs_log=True),
    "fitness_alignments": Metric(fitness_alignments, needs_log=True, slow=True),
    "precision_alignments": Metric(precision_alignments, needs_log=True, slow=True),
}


def _load_results(path: Path) -> dict:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _to_json(o):
    return o.item() if hasattr(o, "item") else str(o)  # numpy scalars


""" Writes the results atomically (temp file + rename), so a crash while
    writing never leaves a corrupt file behind. Also refreshes the CSV.
"""
def _save_results(results: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=_to_json)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _results_frame(results).to_csv(path.with_suffix(".csv"))


def _results_frame(results: dict) -> pd.DataFrame:
    return pd.DataFrame(results)


def _update_f_scores(res: dict):
    for kind in ["tbr", "align"]:
        fit, prec = res.get(f"fitness_{kind}"), res.get(f"precision_{kind}")
        if fit is not None and prec is not None:
            res[f"f_score_{kind}"] = f_score(fit, prec)


""" Runs every metric.

    - skip_alignments=True leaves out the alignment-based metrics.
    - skip_soundness=True does not run the soundness check, evaluated as true
    - Nets are loaded from model_dir/petri_net_<miner>.pnml.
"""
def run_all(miners=MINERS, metrics=None, log=None, path: Path = RESULTS_JSON,
            model_dir: Path = INITIAL_DISCOVERY, skip_alignments: bool = False,
            skip_soundness: bool = False) -> dict:
    metrics = list(METRICS) if metrics is None else list(metrics)
    unknown = set(metrics) - set(METRICS)
    if unknown:
        raise ValueError(f"unknown metrics: {sorted(unknown)}")
    metrics = [m for m in metrics
               if not (METRICS[m].slow and skip_alignments)
               and not (m == "soundness" and skip_soundness)]

    results = _load_results(path)
    for miner in miners:
        model = load_petri_net_pnml(miner, out_dir=model_dir)
        res = results[miner] = {}
        if skip_soundness:
            res["sound"] = True

        for name in metrics:
            spec = METRICS[name]
            if spec.needs_sound and res.get("sound") is not True:
                res[name] = float("nan")
                print(f"[{miner}] {name}: skipped (net not known to be sound)", flush=True)
                _save_results(results, path)
                continue

            print(f"[{miner}] {name} ...", flush=True)
            kwargs = {}
            if spec.needs_log:
                kwargs["log"] = _as_log(log)
            start = time.monotonic()
            try:
                values = spec.func(model, **kwargs)
            except Exception as e:
                res[f"{name}_error"] = f"{type(e).__name__}: {e}"
                traceback.print_exc()
                _save_results(results, path)
                continue
            print(f"[{miner}] {name}: {values} ({time.monotonic() - start:.1f}s)", flush=True)

            res.update(values)
            _update_f_scores(res)
            _save_results(results, path)

    return results


""" Loads saved results as a DataFrame (one column per miner). """
def load_results(path: Path = RESULTS_JSON) -> pd.DataFrame:
    return _results_frame(_load_results(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute quality metrics for the saved Petri nets.")
    parser.add_argument("--miners", nargs="+", default=list(MINERS), choices=MINERS)
    parser.add_argument("--metrics", nargs="+", default=None, choices=list(METRICS))
    parser.add_argument("--skip-alignments", action="store_true", help="skip the slow alignment metrics")
    parser.add_argument("--out", type=Path, default=RESULTS_JSON)
    args = parser.parse_args()

    for miner in args.miners:
        # the inductive miner returns sound nets by construction
        run_all([miner], args.metrics, path=args.out,
                skip_alignments=args.skip_alignments, skip_soundness=(miner == "inductive"))
    print(load_results(args.out))
