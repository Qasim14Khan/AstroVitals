"""Rules that turn numbers into plain-language actions.

On a long mission (for example a trip to Mars) there is a communication delay with ground,
so the astronaut must be able to understand what the data means and what to do right now.
Every alert has: what we see, why it matters, and what to do. Thresholds are illustrative.

This is a research prototype on simulated data. It is NOT medical advice.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .anomaly import Analysis, top_drivers
from .indicators import DOMAINS, RADIATION_LIMIT_MSV

CONTACT = "Contact the flight surgeon (ground medical team) if it does not improve."


@dataclass
class Alert:
    severity: str   # "red" | "yellow" | "info"
    domain: str
    title: str
    detail: str
    action: str


def _recent(df: pd.DataFrame, col: str, n: int) -> float:
    return float(df[col].tail(n).mean())


def generate_alerts(df: pd.DataFrame, analysis: Analysis) -> list[Alert]:
    """Build the list of alerts for the LAST row of df."""
    alerts: list[Alert] = []
    base = analysis.baseline["mean"]
    row = len(df) - 1
    last = df.iloc[-1]

    # ---- Immune -------------------------------------------------------------------
    temp2 = _recent(df, "body_temp_c", 2)
    if last["body_temp_c"] >= 37.8 or temp2 >= 37.5:
        alerts.append(Alert(
            "red", "Immune", "Possible infection (fever)",
            f"Temperature is {last['body_temp_c']:.1f} °C (your normal is {base['body_temp_c']:.1f} °C) "
            f"with symptom score {last['symptom_score']:.1f}/10.",
            "Re-measure in a few hours, rest, drink fluids and log symptoms. Limit close contact with crewmates. "
            + CONTACT,
        ))
    elif last["body_temp_c"] >= 37.3 or _recent(df, "symptom_score", 3) >= 2.5:
        alerts.append(Alert(
            "yellow", "Immune", "Early signs of illness",
            f"Temperature {last['body_temp_c']:.1f} °C, recent symptom score {_recent(df, 'symptom_score', 3):.1f}/10.",
            "Rest, hydrate and re-check tomorrow. Keep logging symptoms.",
        ))

    # ---- Cardiovascular -----------------------------------------------------------
    sys3, dia3 = _recent(df, "systolic_bp", 3), _recent(df, "diastolic_bp", 3)
    if sys3 >= 140 or dia3 >= 90:
        alerts.append(Alert(
            "red", "Cardiovascular", "High blood pressure",
            f"3-day average {sys3:.0f}/{dia3:.0f} mmHg (your normal is {base['systolic_bp']:.0f}/{base['diastolic_bp']:.0f}).",
            "Sit quietly for 5 minutes and re-measure. Reduce caffeine and salt, keep up cardio exercise. " + CONTACT,
        ))
    elif sys3 >= base["systolic_bp"] + 10 or dia3 >= base["diastolic_bp"] + 7:
        alerts.append(Alert(
            "yellow", "Cardiovascular", "Blood pressure trending up",
            f"3-day average {sys3:.0f}/{dia3:.0f} mmHg compared with your normal "
            f"{base['systolic_bp']:.0f}/{base['diastolic_bp']:.0f}.",
            "Measure at the same time each day, review exercise and sleep, and watch the trend.",
        ))
    hr7 = _recent(df, "resting_hr", 7)
    if hr7 >= base["resting_hr"] + 8 and not any(a.title.startswith("Possible infection") for a in alerts):
        alerts.append(Alert(
            "yellow", "Cardiovascular", "Resting heart rate elevated",
            f"7-day average {hr7:.0f} bpm vs your normal {base['resting_hr']:.0f} bpm.",
            "Check hydration, sleep, stress and workload. Log any chest discomfort or palpitations and report them.",
        ))

    # ---- Bone & muscle ------------------------------------------------------------
    ex7 = _recent(df, "exercise_min", 7)
    if ex7 < 0.7 * base["exercise_min"]:
        alerts.append(Alert(
            "yellow", "Bone & Muscle", "Exercise below your usual level",
            f"7-day average {ex7:.0f} min/day vs your normal {base['exercise_min']:.0f}. "
            "Less loading speeds up bone and muscle loss in microgravity.",
            "Schedule resistive exercise sessions in the next 48 hours and report any equipment problems to the crew.",
        ))
    weight_loss = base["weight_kg"] - _recent(df, "weight_kg", 7)
    grip_loss = base["grip_strength_kg"] - _recent(df, "grip_strength_kg", 7)
    if weight_loss >= 2.0 or grip_loss >= 3.0:
        alerts.append(Alert(
            "red" if (weight_loss >= 3.0 or grip_loss >= 4.5) else "yellow",
            "Bone & Muscle", "Progressive muscle/mass loss",
            f"Weight is down {weight_loss:.1f} kg and grip strength is down {grip_loss:.1f} kg from your baseline.",
            "Review calorie, protein and vitamin D intake, increase resistive training, and tell the flight surgeon about the trend.",
        ))

    # ---- Behavioral ---------------------------------------------------------------
    sleep3 = _recent(df, "sleep_hours", 3)
    if sleep3 < 5.0:
        alerts.append(Alert("red", "Behavioral", "Severe sleep shortage",
                            f"Averaging {sleep3:.1f} h of sleep over 3 days.",
                            "Protect a sleep period tonight, dim lights and avoid screens. Fatigue raises error risk, so postpone critical tasks. "
                            + CONTACT))
    elif sleep3 < 6.0:
        alerts.append(Alert("yellow", "Behavioral", "Low sleep",
                            f"Averaging {sleep3:.1f} h of sleep over 3 days (normal {base['sleep_hours']:.1f} h).",
                            "Keep a regular sleep schedule and cut caffeine in the evening."))
    mood5, stress5 = _recent(df, "mood", 5), _recent(df, "stress", 5)
    if mood5 <= 4.0 or stress5 >= 7.0:
        alerts.append(Alert("red", "Behavioral", "High stress / low mood",
                            f"5-day averages: mood {mood5:.1f}/10, stress {stress5:.1f}/10.",
                            "Talk to a crewmate you trust, schedule a private call with family or the ground behavioral-health team, "
                            "and take protected downtime."))
    elif mood5 <= base["mood"] - 1.5 or stress5 >= base["stress"] + 2.0:
        alerts.append(Alert("yellow", "Behavioral", "Mood or stress drifting",
                            f"5-day averages: mood {mood5:.1f}/10 (normal {base['mood']:.1f}), stress {stress5:.1f}/10 (normal {base['stress']:.1f}).",
                            "Plan something restorative today: a hobby, a call home, or time at the window."))
    rt3 = _recent(df, "reaction_ms", 3)
    if rt3 >= base["reaction_ms"] * 1.15:
        alerts.append(Alert("yellow", "Behavioral", "Slower reaction time",
                            f"3-day average {rt3:.0f} ms vs normal {base['reaction_ms']:.0f} ms.",
                            "Likely fatigue or stress. Double-check procedures and avoid high-risk tasks while it persists."))

    # ---- Radiation ----------------------------------------------------------------
    cum = float(last["cumulative_radiation_msv"])
    pct = 100 * cum / RADIATION_LIMIT_MSV
    typical = float(df["radiation_msv"].median())
    if last["radiation_msv"] > 3 * typical:
        alerts.append(Alert("red", "Radiation", "Radiation spike (possible solar particle event)",
                            f"Today's dose is {last['radiation_msv']:.1f} mSv, about {last['radiation_msv'] / typical:.0f}x your typical day.",
                            "Follow the mission procedure for the radiation shelter and report the exposure to ground."))
    if pct >= 80:
        alerts.append(Alert("red", "Radiation", "Cumulative dose near limit",
                            f"{cum:.0f} mSv, {pct:.0f}% of the {RADIATION_LIMIT_MSV:.0f} mSv illustrative limit.",
                            "Discuss remaining mission plans and shielding with ground."))
    elif pct >= 50:
        alerts.append(Alert("info", "Radiation", "Cumulative dose past halfway",
                            f"{cum:.0f} mSv, {pct:.0f}% of the {RADIATION_LIMIT_MSV:.0f} mSv illustrative limit.",
                            "Spend high-risk periods in the best-shielded part of the vehicle."))

    # ---- Multi-signal pattern from the ML layer -----------------------------------
    domain_flagged = [d for d in DOMAINS if analysis.domain_status.loc[row, d] != "Nominal"]
    if bool(analysis.unusual.iloc[row]) and domain_flagged:
        alerts.append(Alert("info", "Overall", "Unusual combined pattern",
                            "The anomaly detector sees an unusual mix of signals today across: " + ", ".join(domain_flagged) + ".",
                            "Review the Trends tab to see which indicators moved together."))

    # ---- Make sure drivers are visible for domains in alert -----------------------
    for dom in DOMAINS:
        if analysis.domain_status.loc[row, dom] == "Alert" and not any(a.domain == dom and a.severity == "red" for a in alerts):
            drivers = top_drivers(analysis, row, dom)
            names = ", ".join(f"{n} ({z:+.1f} SD)" for n, z in drivers) or "several small changes"
            alerts.append(Alert("red", dom, f"{dom} risk score in alert range",
                                f"Main drivers: {names}.",
                                "Review the Trends tab for this domain. " + CONTACT))

    order = {"red": 0, "yellow": 1, "info": 2}
    return sorted(alerts, key=lambda a: order[a.severity])
