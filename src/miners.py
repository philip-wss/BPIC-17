import pm4py
from pm4py.objects.petri_net.obj import PetriNet, Marking

from src.config import INITIAL_DISCOVERY, CASE_ID, ACTIVITY, TIMESTAMP
from src.load import load_events
from src.preprocessing import filter_complete_cases, collapse_workitems

PetriNetModel = tuple[PetriNet, Marking, Marking]

_KEYS = dict(activity_key=ACTIVITY, timestamp_key=TIMESTAMP, case_id_key=CASE_ID)


""" Inductive miner (IMf): discovers a sound, block-structured Petri net.
    noise_threshold=0 is the plain inductive miner (perfect fitness); higher
    values filter infrequent behaviour and give a simpler model.
"""
def discover_inductive(log, noise_threshold: float = 0.0) -> PetriNetModel:
    return pm4py.discover_petri_net_inductive(
        log, noise_threshold=noise_threshold, **_KEYS
    )


""" Heuristics miner: builds a dependency graph from the directly-follows
    frequencies and converts it to a Petri net. Arcs need a dependency measure
    of at least dependency_threshold; and_threshold decides between parallel (AND) and exclusive (XOR)
    splits/joins.
"""
def discover_heuristics(
    log,
    dependency_threshold: float = 0.5,
    and_threshold: float = 0.65,
    loop_two_threshold: float = 0.5,
) -> PetriNetModel:
    return pm4py.discover_petri_net_heuristics(
        log,
        dependency_threshold=dependency_threshold,
        and_threshold=and_threshold,
        loop_two_threshold=loop_two_threshold,
        **_KEYS,
    )


""" Renders a Petri net (with initial and final marking) to an SVG file. """
def save_petri_net_svg(model: PetriNetModel, name: str, out_dir=INITIAL_DISCOVERY):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"petri_net_{name}.svg"
    pm4py.save_vis_petri_net(*model, str(path))
    return path


""" Writes a Petri net (with initial and final marking) to a PNML file, so it
    can be reloaded later without rediscovering it.
"""
def save_petri_net_pnml(model: PetriNetModel, name: str, out_dir=INITIAL_DISCOVERY):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"petri_net_{name}.pnml"
    pm4py.write_pnml(*model, str(path))
    return path


""" Reads a Petri net written by save_petri_net_pnml. """
def load_petri_net_pnml(name: str, out_dir=INITIAL_DISCOVERY) -> PetriNetModel:
    return pm4py.read_pnml(str(out_dir / f"petri_net_{name}.pnml"))


if __name__ == "__main__":
    # cleaned log -> complete cases only -> one event per started work item
    log = collapse_workitems(filter_complete_cases(load_events()), drop_withdrawn=True)

    miners = {
        "inductive": discover_inductive,
        "heuristics": discover_heuristics,
    }
    for name, discover in miners.items():
        model = discover(log)
        save_petri_net_pnml(model, name)
        print(f"{name}: {len(model[0].places)} places, "
              f"{len(model[0].transitions)} transitions -> {save_petri_net_svg(model, name)}")
