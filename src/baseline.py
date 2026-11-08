"""Personal baselines.

Astronauts differ a lot (a resting heart rate of 50 can be normal for one person and a red
flag for another), so every indicator is compared with that astronaut's OWN early-mission
normal, not with a population average.
"""
from __future__ import annotations

import pandas as pd

from .indicators import INDICATORS

BASELINE_DAYS = 21  # the first three weeks of the mission define "normal"


def compute_baseline(df: pd.DataFrame, n_days: int = BASELINE_DAYS) -> pd.DataFrame:
    """Return a table with the mean and standard deviation of each indicator."""
    head = df.head(n_days)
    rows = {}
    for col, meta in INDICATORS.items():
        std = head[col].std(ddof=1) if len(head) > 1 else 0.0
        std = 0.0 if pd.isna(std) else float(std)
        rows[col] = {"mean": float(head[col].mean()), "std": max(std, meta["min_std"])}
    return pd.DataFrame(rows).T


def worsening_zscores(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """z-score of every indicator, signed so that POSITIVE always means 'worse for health'.

    Example: sleep 2 SD below normal -> +2.  Resting heart rate 2 SD above normal -> +2.
    """
    out = {}
    for col, meta in INDICATORS.items():
        z = (df[col] - baseline.loc[col, "mean"]) / baseline.loc[col, "std"]
        out[col] = z * meta["worse"]
    return pd.DataFrame(out, index=df.index)
