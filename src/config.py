from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"

RAW_LOG = DATA / "BPI_Challenge_2017.xes"

# Standard PM4Py column names
CASE_ID = "case:concept:name"
ACTIVITY = "concept:name"
TIMESTAMP = "time:timestamp"
RESOURCE = "org:resource"
LIFECYCLE = "lifecycle:transition"

# Expected true start activities of a complete case
START_ACTIVITIES = {"A_Create Application"}
# A case is complete if it contains at least one of these activities
# (anywhere in the case, not necessarily as the last event)
COMPLETION_ACTIVITIES = {"A_Pending", "A_Denied", "A_Cancelled"}

# Workflow (work item) activities carry a full lifecycle:
# schedule -> start -> (suspend/resume)* -> complete | ate_abort | withdraw
WORKITEM_PREFIX = "W_"
WORKITEM_END_TRANSITIONS = {"complete", "ate_abort", "withdraw"}

REPORTS = ROOT / "reports"
# Inductive and heuristics miner with default parameters (src/miners.py)
INITIAL_DISCOVERY = REPORTS / "initial_discovery_algorithms"
