"""Matplotlib charts used by the dashboard."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # no display needed (works on servers and in tests)
import matplotlib.pyplot as plt
import pandas as pd

from .anomaly import Analysis
from .indicators import ALERT_THRESHOLD, DOMAINS, INDICATORS, RADIATION_LIMIT_MSV, WATCH_THRESHOLD

COLORS = {
    "line": "#1f4e9c",
    "smooth": "#0b2a5b",
    "band": "#9db8e6",
    "anomaly": "#d62728",
    "watch": "#f0a500",
    "event": "#e8e1f5",
}
DOMAIN_COLORS = {
    "Cardiovascular": "#d62728",
    "Bone & Muscle": "#8c564b",
    "Immune": "#2ca02c",
    "Behavioral": "#1f77b4",
}


def _style(ax):
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)


def _shade_events(ax, df: pd.DataFrame):
    """Shade the simulated ground-truth events (demo mode only)."""
    labels = df["injected_event"].tolist()
    start = None
    for i, lab in enumerate(labels + [""]):
        prev = labels[i - 1] if i > 0 else ""
        if lab != prev:
            if prev != "" and start is not None:
                ax.axvspan(df["day"].iloc[start], df["day"].iloc[i - 1], color=COLORS["event"], alpha=0.6, lw=0)
            start = i if lab != "" else None


def trend_chart(df: pd.DataFrame, col: str, analysis: Analysis, show_truth: bool = False):
    """One indicator over time: raw line, 7-day average, personal normal band and flagged days."""
    meta = INDICATORS[col]
    mean, std = analysis.baseline.loc[col, "mean"], analysis.baseline.loc[col, "std"]
    fig, ax = plt.subplots(figsize=(6, 2.8))
    if show_truth:
        _shade_events(ax, df)
    ax.axhspan(mean - 2 * std, mean + 2 * std, color=COLORS["band"], alpha=0.35, lw=0, label="Your normal range")
    ax.plot(df["day"], df[col], color=COLORS["line"], lw=1, alpha=0.6, label="Daily value")
    ax.plot(df["day"], df[col].rolling(7, min_periods=1).mean(), color=COLORS["smooth"], lw=2, label="7-day average")
    flagged = analysis.unusual.values
    ax.scatter(df.loc[flagged, "day"], df.loc[flagged, col], color=COLORS["anomaly"], s=18, zorder=5,
               label="Unusual day (ML)")
    ax.set_title(f"{meta['label']} ({meta['unit']})", fontsize=10, loc="left")
    ax.set_xlabel("Mission day", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2, frameon=False)
    _style(ax)
    fig.tight_layout()
    return fig


def domain_score_chart(df: pd.DataFrame, analysis: Analysis, show_truth: bool = False):
    """The four domain risk scores over time with Watch and Alert thresholds."""
    fig, ax = plt.subplots(figsize=(9, 3.3))
    if show_truth:
        _shade_events(ax, df)
    for dom in DOMAINS:
        ax.plot(df["day"], analysis.domain_scores[dom], color=DOMAIN_COLORS[dom], lw=1.8, label=dom)
    ax.axhline(WATCH_THRESHOLD, color=COLORS["watch"], ls="--", lw=1)
    ax.axhline(ALERT_THRESHOLD, color=COLORS["anomaly"], ls="--", lw=1)
    ax.text(df["day"].iloc[0], WATCH_THRESHOLD + 0.05, "Watch", fontsize=7, color=COLORS["watch"])
    ax.text(df["day"].iloc[0], ALERT_THRESHOLD + 0.05, "Alert", fontsize=7, color=COLORS["anomaly"])
    ax.set_ylabel("Risk score (SD worse than your normal)", fontsize=8)
    ax.set_xlabel("Mission day", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.0, 1.12))
    _style(ax)
    fig.tight_layout()
    return fig


def radiation_chart(df: pd.DataFrame):
    """Cumulative radiation dose against the illustrative limit."""
    fig, ax = plt.subplots(figsize=(6, 2.8))
    ax.plot(df["day"], df["cumulative_radiation_msv"], color="#6a3d9a", lw=2)
    ax.axhline(RADIATION_LIMIT_MSV, color=COLORS["anomaly"], ls="--", lw=1)
    ax.text(df["day"].iloc[0], RADIATION_LIMIT_MSV * 0.94, "Illustrative limit", fontsize=7, color=COLORS["anomaly"])
    ax.set_title("Cumulative radiation dose (mSv)", fontsize=10, loc="left")
    ax.set_xlabel("Mission day", fontsize=8)
    ax.set_ylim(0, RADIATION_LIMIT_MSV * 1.05)
    ax.tick_params(labelsize=8)
    _style(ax)
    fig.tight_layout()
    return fig
