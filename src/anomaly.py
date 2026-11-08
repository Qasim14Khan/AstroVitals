"""Anomaly detection and risk scoring.

Two complementary layers:

1. Domain risk scores (explainable): for each of the four health domains we average how many
   standard deviations the astronaut is WORSE than their own baseline, smoothed over 3 days.
   Scores map to Nominal / Watch / Alert.

2. Isolation Forest (machine learning): looks at all indicators together and flags days whose
   overall pattern is unusual for this astronaut, even if no single number crosses a threshold.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import IsolationForest

from .baseline import BASELINE_DAYS, compute_baseline, worsening_zscores
from .indicators import ALERT_THRESHOLD, DOMAINS, INDICATORS, WATCH_THRESHOLD

SMOOTH_DAYS = 3
MIN_DAYS_FOR_ML = 30


@dataclass
class Analysis:
    baseline: pd.DataFrame        # mean/std per indicator
    zscores: pd.DataFrame         # smoothed "worse-than-normal" z per indicator
    domain_scores: pd.DataFrame   # one column per domain
    domain_status: pd.DataFrame   # "Nominal" / "Watch" / "Alert" per domain per day
    unusual: pd.Series            # True where Isolation Forest flags the day
    anomaly_score: pd.Series      # higher = more unusual


def status_from_score(score: float) -> str:
    if score >= ALERT_THRESHOLD:
        return "Alert"
    if score >= WATCH_THRESHOLD:
        return "Watch"
    return "Nominal"


def analyze(
    df: pd.DataFrame,
    baseline_days: int = BASELINE_DAYS,
    smooth: int = SMOOTH_DAYS,
    contamination: float = 0.08,
    seed: int = 0,
) -> Analysis:
    """Run the full pipeline on a mission DataFrame."""
    df = df.reset_index(drop=True)
    baseline = compute_baseline(df, baseline_days)
    z = worsening_zscores(df, baseline).rolling(smooth, min_periods=1).mean()

    # Domain scores: only count "worse", cap extreme values so one sensor can't dominate.
    domain_scores = pd.DataFrame(
        {dom: z[cols].clip(lower=0, upper=6).mean(axis=1) for dom, cols in DOMAINS.items()}
    )
    domain_status = domain_scores.apply(lambda col: col.map(status_from_score))

    # Isolation Forest on the whole indicator pattern.
    if len(df) >= MIN_DAYS_FOR_ML:
        features = z.clip(-4, 6)
        model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)
        model.fit(features)
        unusual = pd.Series(model.predict(features) == -1, index=df.index)
        anomaly_score = pd.Series(-model.decision_function(features), index=df.index)
    else:
        unusual = pd.Series(False, index=df.index)
        anomaly_score = pd.Series(0.0, index=df.index)

    return Analysis(baseline, z, domain_scores, domain_status, unusual, anomaly_score)


def top_drivers(analysis: Analysis, row: int, domain: str, n: int = 2, min_z: float = 1.0):
    """Indicators in a domain contributing most to its score on a given row."""
    cols = DOMAINS[domain]
    zs = analysis.zscores.loc[row, cols].sort_values(ascending=False)
    return [(INDICATORS[c]["label"], float(v)) for c, v in zs.items() if v >= min_z][:n]


def overall_status(analysis: Analysis, row: int) -> str:
    """Worst status across the four domains on a given row."""
    statuses = analysis.domain_status.loc[row].tolist()
    for level in ("Alert", "Watch"):
        if level in statuses:
            return level
    return "Nominal"
