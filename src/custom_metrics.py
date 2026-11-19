from pm4py.objects.petri_net.obj import PetriNet


""" Density of the Petri net graph (nodes = places + transitions incl. silent
    ones, edges = arcs). Arcs can only connect a place with a transition, in
    either direction, so the maximum number of arcs is 2·|P|·|T|.
"""
def density(net: PetriNet) -> float:
    n_places, n_transitions = len(net.places), len(net.transitions)
    if n_places == 0 or n_transitions == 0:
        return float("nan")
    return len(net.arcs) / (2 * n_places * n_transitions)


""" Number of nodes of the Petri net graph: |P| + |T| (places plus transitions,
    incl. silent ones).
"""
def node_count(net: PetriNet) -> int:
    return len(net.places) + len(net.transitions)
