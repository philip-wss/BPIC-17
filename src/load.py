import pandas as pd
import pm4py

from src.config import (
    RAW_LOG,
    CASE_ID, ACTIVITY, TIMESTAMP
)

""" Reads the original XES file """
def _read_raw() -> pd.DataFrame:
    if not RAW_LOG.exists():
        raise FileNotFoundError(
            f"{RAW_LOG} not found. See data/README.md."
        )
    return pm4py.read_xes(str(RAW_LOG))


LOAN_GOAL = "case:LoanGoal"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[TIMESTAMP] = pd.to_datetime(df[TIMESTAMP], utc=True)
    # 'W_Shortened completion ' has a trailing space in the raw log
    df[ACTIVITY] = df[ACTIVITY].str.strip()
    # "Not speficied" [sic] carries no information, same as "Unknown"
    df[LOAN_GOAL] = df[LOAN_GOAL].replace("Not speficied", "Unknown")
    df = df.sort_values([CASE_ID, TIMESTAMP], kind="stable").reset_index(drop=True)
    return df

""" Returns the cleaned event log, read from the raw XES file. """
def load_events() -> pd.DataFrame:
    return _clean(_read_raw())

""" Builds a case table (one row per case, indexed by case id) from the event log. """
def build_cases(events: pd.DataFrame) -> pd.DataFrame:
    grouped = events.groupby(CASE_ID, sort=False)
    case_attrs = [c for c in events.columns if c.startswith("case:") and c != CASE_ID]
    cases = grouped[case_attrs].first()
    cases["start"] = grouped[TIMESTAMP].min()
    cases["end"] = grouped[TIMESTAMP].max()
    cases["duration"] = cases["end"] - cases["start"]
    cases["length"] = grouped.size()
    cases["variant"] = grouped[ACTIVITY].agg(tuple)
    return cases

