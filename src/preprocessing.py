import pandas as pd

from src.config import (
    CASE_ID, ACTIVITY, LIFECYCLE,
    START_ACTIVITIES, COMPLETION_ACTIVITIES,
    WORKITEM_PREFIX, WORKITEM_END_TRANSITIONS,
)


""" Returns the first activity of every case (indexed by case id). """
def case_start_activities(events: pd.DataFrame) -> pd.Series:
    return events.groupby(CASE_ID, sort=False)[ACTIVITY].first().rename("start")


""" Returns, for every case (indexed by case id), how often each of the
    completion activities occurs in it (one column per activity, 0 if absent).
"""
def completion_activity_counts(
    events: pd.DataFrame,
    completion_activities=COMPLETION_ACTIVITIES,
) -> pd.DataFrame:
    case_ids = events[CASE_ID].unique()
    hits = events[events[ACTIVITY].isin(completion_activities)]
    counts = pd.crosstab(hits[CASE_ID], hits[ACTIVITY])
    return counts.reindex(
        index=case_ids, columns=sorted(completion_activities), fill_value=0
    ).rename_axis(index=CASE_ID, columns=None)


def filter_complete_cases(
    events: pd.DataFrame,
    start_activities=START_ACTIVITIES,
    completion_activities=COMPLETION_ACTIVITIES,
) -> pd.DataFrame:
    has_start = case_start_activities(events).isin(start_activities)
    n_completion = completion_activity_counts(events, completion_activities).sum(axis=1)
    single_completion = (n_completion == 1).reindex(has_start.index)
    complete = has_start.index[has_start & single_completion]
    return events[events[CASE_ID].isin(complete)].reset_index(drop=True)


""" Collapses every work item (W_ activity) to a single event: the first event
    of the work item, normally its 'schedule' event, kept at its position in the
    case.
"""
def collapse_workitems(
    events: pd.DataFrame,
    drop_withdrawn: bool = False,
    prefix=WORKITEM_PREFIX,
    end_transitions=WORKITEM_END_TRANSITIONS,
) -> pd.DataFrame:
    is_workitem = events[ACTIVITY].str.startswith(prefix)
    w = events.loc[is_workitem, [CASE_ID, ACTIVITY, LIFECYCLE]]
    keys = [w[CASE_ID], w[ACTIVITY]]
    prev = w.groupby(keys, sort=False, observed=True)[LIFECYCLE].shift()
    first = (w[LIFECYCLE] == "schedule") | prev.isna() | prev.isin(end_transitions)
    if drop_withdrawn:
        item = first.groupby(keys, sort=False, observed=True).cumsum()
        withdrawn = (w[LIFECYCLE] == "withdraw").groupby(
            keys + [item], sort=False, observed=True
        ).transform("any")
        first &= ~withdrawn
    keep = ~is_workitem
    keep.loc[first.index[first]] = True
    return events[keep].reset_index(drop=True)
