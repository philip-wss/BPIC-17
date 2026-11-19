"""Evaluates a hand-made BPMN model: converts it to a Petri net, saves it (PNML +
SVG) and computes the quality metrics on the same log as the mined models.

    python -m src.manual_models data/manual/BPIC_v1.bpmn --name manual1
    python -m src.manual_models --all [--with-alignments]

Always writes reports/<name>/petri_net_<name>.pnml/.svg and reports/<name>/metrics.json/.csv.
--all evaluates every data/manual/BPIC_v<i>.bpmn as manual<i> and prints a summary table.
Alignment metrics are slow and only computed with --with-alignments.
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import pm4py

from src.config import DATA, REPORTS
from src.metrics import load_log, load_results, run_all
from src.miners import PetriNetModel, save_petri_net_pnml, save_petri_net_svg


def bpmn_to_petri_net(path: Path) -> PetriNetModel:
    """Reads a BPMN file and converts it to a Petri net."""
    return pm4py.convert_to_petri_net(pm4py.read_bpmn(str(path)))


"""Converts, saves and evaluates one model; results go to reports/<name>/."""
def run(bpmn: Path, name: str, log=None, with_alignments: bool = False) -> dict:
    out_dir = REPORTS / name
    log = load_log() if log is None else log

    model = bpmn_to_petri_net(bpmn)
    save_petri_net_pnml(model, name, out_dir)
    svg = save_petri_net_svg(model, name, out_dir)
    print(f"{name}: {len(model[0].places)} places, "
          f"{len(model[0].transitions)} transitions -> {svg}", flush=True)
    return run_all(miners=[name], log=log, path=out_dir / "metrics.json",
                   model_dir=out_dir, skip_alignments=not with_alignments)


"""All hand-made models in data/manual as {iteration: path}, sorted by iteration."""
def manual_iterations() -> dict:
    found = {int(m.group(1)): p for p in (DATA / "manual").glob("BPIC_v*.bpmn")
             if (m := re.fullmatch(r"BPIC_v(\d+)\.bpmn", p.name))}
    return dict(sorted(found.items()))


"""Evaluates every iteration and returns the summary table. """
def run_every_model(with_alignments: bool = False) -> pd.DataFrame:
    log = load_log()
    rows = {}
    for i, path in manual_iterations().items():
        name = f"manual{i}"
        run(path, name, log=log, with_alignments=with_alignments)
        rows[name] = load_results(REPORTS / name / "metrics.json")[name]
    raw = pd.DataFrame(rows).T.astype(float)
    return pd.DataFrame({
        "simplicity": raw["simplicity_arc_degree"],
        "fitness": raw["fitness_tbr"],
        "fitness (traces passed)": raw["fit_traces_tbr"],
        "precision": raw["precision_tbr"],
        "generalisation": raw["generalization_tbr"],
    }).rename_axis("model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("bpmn", type=Path, nargs="?")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--name")
    parser.add_argument("--with-alignments", action="store_true")
    args = parser.parse_args()

    if args.all:
        print(run_every_model(args.with_alignments).round(3).to_string())
        raise SystemExit
    if args.bpmn is None or args.name is None:
        parser.error("give a BPMN file and --name, or use --all")

    run(args.bpmn, args.name, with_alignments=args.with_alignments)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(load_results(REPORTS / args.name / "metrics.json"))
