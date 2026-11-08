"""Synthetic long-duration mission data.

Real astronaut medical data is restricted, so AstroVitals is demonstrated on SIMULATED data.
The ranges and trends are modeled on publicly described spaceflight findings
(NASA Human Research Program: cardiovascular changes, bone and muscle loss, immune
dysregulation, behavioral health and radiation exposure). Each simulated astronaut gets
their own personal baseline, and we can inject known events so the detector can be tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Name -> random seed. Different seeds give different personal baselines.
ASTRONAUTS = {
    "Astronaut A (Commander)": 11,
    "Astronaut B (Engineer)": 22,
    "Astronaut C (Scientist)": 33,
}

# Mission-day windows of the injected events (1-based, inclusive).
EVENT_WINDOWS = {
    "Infection": (60, 66),
    "Solar particle event": (75, 76),
    "Exercise slump": (90, 130),
    "Stress period": (110, 126),
    "BP drift": (130, 10_000),
}


def _bump(day: np.ndarray, start: int, end: int) -> np.ndarray:
    """Smooth 0 -> 1 -> 0 curve between start and end, zero elsewhere."""
    x = (day - start) / max(end - start, 1)
    return np.where((x >= 0) & (x <= 1), np.sin(np.pi * x), 0.0)


def generate_mission(
    days: int = 180,
    seed: int = 42,
    start_date: str = "2026-01-01",
    inject_events: bool = True,
) -> pd.DataFrame:
    """Return one row per mission day with all indicators.

    inject_events=False gives a healthy control mission (noise only), which is
    useful for checking that the detector does not raise false alarms.
    """
    rng = np.random.default_rng(seed)
    day = np.arange(1, days + 1)
    t = day / days  # 0..1 progress through the mission

    # Personal "set points" - everyone is different.
    hr0 = rng.normal(58, 4)
    hrv0 = rng.normal(65, 8)
    sys0 = rng.normal(116, 5)
    dia0 = rng.normal(74, 4)
    weight0 = rng.normal(76, 6)
    grip0 = rng.normal(42, 4)
    ex0 = rng.normal(120, 10)

    d = {
        "resting_hr": hr0 + rng.normal(0, 1.8, days),
        "hrv": hrv0 + rng.normal(0, 5, days),
        "systolic_bp": sys0 + rng.normal(0, 3.5, days),
        "diastolic_bp": dia0 + rng.normal(0, 2.5, days),
        "exercise_min": ex0 + rng.normal(0, 10, days),
        "weight_kg": weight0 + rng.normal(0, 0.35, days),
        "grip_strength_kg": grip0 + rng.normal(0, 0.7, days),
        "body_temp_c": 36.6 + rng.normal(0, 0.15, days),
        "symptom_score": np.abs(rng.normal(0.3, 0.4, days)),
        "mood": 7.5 + rng.normal(0, 0.6, days),
        "stress": 3.0 + rng.normal(0, 0.8, days),
        "sleep_hours": 6.9 + rng.normal(0, 0.5, days),
        "reaction_ms": 280 + rng.normal(0, 12, days),
        "radiation_msv": np.clip(rng.normal(1.8, 0.15, days), 0.5, None),
    }

    events = np.array([""] * days, dtype=object)

    if inject_events:
        # 1. Slow musculoskeletal decline over the whole mission.
        d["weight_kg"] = d["weight_kg"] - 3.0 * t
        d["grip_strength_kg"] = d["grip_strength_kg"] - 5.0 * t

        # 2. Infection (days 60-66): fever, symptoms, faster heart, lower HRV, worse sleep.
        b = _bump(day, *EVENT_WINDOWS["Infection"])
        d["body_temp_c"] = d["body_temp_c"] + 1.1 * b
        d["symptom_score"] = d["symptom_score"] + 4.0 * b
        d["resting_hr"] = d["resting_hr"] + 9 * b
        d["hrv"] = d["hrv"] - 18 * b
        d["sleep_hours"] = d["sleep_hours"] - 1.0 * b
        events[(day >= 60) & (day <= 66)] = "Infection"

        # 3. Solar particle event (day 75-76): radiation spike.
        d["radiation_msv"] = d["radiation_msv"] + np.where(day == 75, 14.0, 0.0) + np.where(day == 76, 3.0, 0.0)
        events[(day >= 75) & (day <= 76)] = "Solar particle event"

        # 4. Exercise slump (days 90-130): motivation and equipment problems.
        b = _bump(day, *EVENT_WINDOWS["Exercise slump"])
        d["exercise_min"] = d["exercise_min"] * (1 - 0.45 * b)
        events[(day >= 90) & (day <= 130)] = "Exercise slump"

        # 5. Stress / low mood period (days 110-126): isolation and confinement.
        b = _bump(day, *EVENT_WINDOWS["Stress period"])
        d["stress"] = d["stress"] + 3.5 * b
        d["mood"] = d["mood"] - 3.0 * b
        d["sleep_hours"] = d["sleep_hours"] - 1.6 * b
        d["reaction_ms"] = d["reaction_ms"] + 40 * b
        events[(day >= 110) & (day <= 126)] = "Stress period"

        # 6. Cardiovascular drift from day 130: blood pressure and heart rate creep up.
        ramp = np.clip((day - 130) / max(days - 130, 1), 0, 1)
        d["systolic_bp"] = d["systolic_bp"] + 18 * ramp
        d["diastolic_bp"] = d["diastolic_bp"] + 11 * ramp
        d["resting_hr"] = d["resting_hr"] + 5 * ramp
        d["hrv"] = d["hrv"] - 10 * ramp
        events[day > 130] = "BP drift"

    df = pd.DataFrame(d)
    df.insert(0, "day", day)
    df.insert(1, "date", pd.date_range(start_date, periods=days, freq="D"))

    # Keep values in physical ranges and tidy up the numbers.
    for col in ["mood", "stress"]:
        df[col] = df[col].clip(1, 10)
    df["symptom_score"] = df["symptom_score"].clip(0, 10)
    df["sleep_hours"] = df["sleep_hours"].clip(2, 10)
    df["exercise_min"] = df["exercise_min"].clip(0, None)
    df = df.round({
        "resting_hr": 1, "hrv": 1, "systolic_bp": 0, "diastolic_bp": 0,
        "exercise_min": 0, "weight_kg": 2, "grip_strength_kg": 1, "body_temp_c": 2,
        "symptom_score": 1, "mood": 1, "stress": 1, "sleep_hours": 1,
        "reaction_ms": 0, "radiation_msv": 2,
    })
    df["cumulative_radiation_msv"] = df["radiation_msv"].cumsum().round(1)
    df["injected_event"] = events  # ground truth, only used for demo and testing
    return df
