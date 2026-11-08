"""Definitions of the health indicators and the four risk domains.

Each indicator has:
  label    - human-readable name
  unit     - unit shown in the UI
  worse    - +1 if a HIGHER value is worse for health, -1 if a LOWER value is worse
  min_std  - smallest personal variability we allow when computing z-scores
             (prevents tiny baselines from producing absurd alarms)
"""

INDICATORS = {
    "resting_hr": {"label": "Resting heart rate", "unit": "bpm", "worse": +1, "min_std": 1.5},
    "hrv": {"label": "Heart rate variability", "unit": "ms", "worse": -1, "min_std": 4.0},
    "systolic_bp": {"label": "Systolic blood pressure", "unit": "mmHg", "worse": +1, "min_std": 3.0},
    "diastolic_bp": {"label": "Diastolic blood pressure", "unit": "mmHg", "worse": +1, "min_std": 2.0},
    "exercise_min": {"label": "Exercise (resistive + cardio)", "unit": "min/day", "worse": -1, "min_std": 8.0},
    "weight_kg": {"label": "Body weight", "unit": "kg", "worse": -1, "min_std": 0.5},
    "grip_strength_kg": {"label": "Grip strength", "unit": "kg", "worse": -1, "min_std": 1.0},
    "body_temp_c": {"label": "Body temperature", "unit": "°C", "worse": +1, "min_std": 0.15},
    "symptom_score": {"label": "Illness symptoms (0-10)", "unit": "score", "worse": +1, "min_std": 0.5},
    "mood": {"label": "Mood (1-10)", "unit": "score", "worse": -1, "min_std": 0.5},
    "stress": {"label": "Stress (1-10)", "unit": "score", "worse": +1, "min_std": 0.7},
    "sleep_hours": {"label": "Sleep", "unit": "h", "worse": -1, "min_std": 0.4},
    "reaction_ms": {"label": "Reaction time", "unit": "ms", "worse": +1, "min_std": 10.0},
}

# The four hazards named in the challenge, mapped to the indicators that reveal them.
DOMAINS = {
    "Cardiovascular": ["resting_hr", "hrv", "systolic_bp", "diastolic_bp"],
    "Bone & Muscle": ["exercise_min", "weight_kg", "grip_strength_kg"],
    "Immune": ["body_temp_c", "symptom_score"],
    "Behavioral": ["mood", "stress", "sleep_hours", "reaction_ms"],
}

# Radiation is tracked separately as a cumulative dose (mSv).
# The limit below is an ILLUSTRATIVE value for the demo, not an official medical threshold.
RADIATION_LIMIT_MSV = 600.0

# Domain score thresholds (a score is the average "how much worse than my own normal" in SDs).
WATCH_THRESHOLD = 1.0
ALERT_THRESHOLD = 2.0

STATUS_ORDER = ["Nominal", "Watch", "Alert"]
