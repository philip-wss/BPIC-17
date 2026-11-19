import matplotlib.pyplot as plt
import pandas as pd

from src.config import COMPLETION_ACTIVITIES

SECONDS_PER_DAY = 86400

COLORS = {"before": "#4C78A8", "after": "#E4572E", "complete": "#2A6F97", "neutral": "#555555"}


def setup_plotting() -> None:
    plt.rcParams.update({
        "figure.figsize": (10, 4), "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True, "font.size": 10,
    })


def fmt_p(p: float) -> str:
    return "<1e-300" if float(p) == 0 else f"{float(p):.3g}"


def rel_change(before: float, after: float) -> float:
    return (after - before) / before


def add_time_columns(cases: pd.DataFrame) -> pd.DataFrame:
    cases = cases.copy()
    cases["duration_days"] = cases["duration"].dt.total_seconds() / SECONDS_PER_DAY
    start = cases["start"].dt.tz_localize(None)
    cases["start_month"] = start.dt.to_period("M").dt.to_timestamp()
    cases["start_week"] = start.dt.to_period("W-SUN").dt.start_time
    return cases


def trim_partial_weeks(weekly: pd.Series | pd.DataFrame):
    return weekly.iloc[1:-1]


def variant_table(var_before: pd.Series, var_after: pd.Series, top: int = 10) -> pd.DataFrame:
    pooled = pd.concat([var_before, var_after]).value_counts().head(top).index
    tab = pd.DataFrame({
        "share_before": var_before.value_counts(normalize=True).reindex(pooled).fillna(0),
        "share_after": var_after.value_counts(normalize=True).reindex(pooled).fillna(0),
    })
    tab["diff"] = tab["share_after"] - tab["share_before"]
    tab["variant"] = [" > ".join(v) for v in tab.index]
    tab.index = [f"V{i + 1}" for i in range(len(tab))]
    return tab


def new_transition_share(var_before: pd.Series, var_after: pd.Series) -> float:
    transitions = lambda v: set(zip(v, v[1:]))
    seen = set().union(*(transitions(v) for v in var_before.unique()))
    return float(var_after.map(lambda v: not transitions(v) <= seen).mean())


def verdict_mix(var_before: pd.Series, var_after: pd.Series) -> pd.DataFrame:
    verdict = lambda v: next(a for a in v if a in COMPLETION_ACTIVITIES)
    share = lambda s: s.map(verdict).value_counts(normalize=True)
    mix = 100 * pd.DataFrame({"before [%]": share(var_before), "after [%]": share(var_after)}).fillna(0)
    return mix.sort_values("before [%]", ascending=False)
