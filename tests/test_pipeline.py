"""Tests: does the detector find the events we inject, and stay quiet when nothing is wrong?"""
import pandas as pd
import pytest

from src.anomaly import analyze, overall_status
from src.charts import domain_score_chart, radiation_chart, trend_chart
from src.indicators import DOMAINS, INDICATORS
from src.recommend import generate_alerts
from src.simulate import ASTRONAUTS, generate_mission


def test_simulation_shape_and_reproducibility():
    a = generate_mission(days=180, seed=1)
    b = generate_mission(days=180, seed=1)
    c = generate_mission(days=180, seed=2)
    assert len(a) == 180
    for col in INDICATORS:
        assert col in a.columns
    pd.testing.assert_frame_equal(a, b)          # same seed -> same data
    assert not a["resting_hr"].equals(c["resting_hr"])  # different seed -> different person


@pytest.mark.parametrize("seed", ASTRONAUTS.values())
def test_no_false_alarms_on_healthy_mission(seed):
    df = generate_mission(seed=seed, inject_events=False)
    result = analyze(df)
    assert (result.domain_status == "Alert").sum().sum() == 0
    assert (result.domain_status != "Nominal").any(axis=1).mean() < 0.05


@pytest.mark.parametrize("seed", ASTRONAUTS.values())
def test_injected_events_are_detected(seed):
    df = generate_mission(seed=seed)
    result = analyze(df)
    status = result.domain_status
    assert (status["Immune"].iloc[59:68] == "Alert").any()          # infection, days 60-68
    assert (status["Behavioral"].iloc[109:128] == "Alert").any()    # stress period
    assert status["Cardiovascular"].iloc[-1] == "Alert"             # blood pressure drift
    assert status["Bone & Muscle"].iloc[-1] == "Alert"              # muscle loss
    # the ML layer: it should flag days inside the infection window and mostly be right
    assert result.unusual.iloc[59:68].any()
    injected = df["injected_event"] != ""
    precision = (result.unusual & injected).sum() / max(result.unusual.sum(), 1)
    assert precision >= 0.8


def test_early_mission_is_nominal():
    df = generate_mission(seed=11)
    result = analyze(df.iloc[:45])
    assert overall_status(result, 44) == "Nominal"


def test_infection_produces_fever_alert_with_action():
    df = generate_mission(seed=11).iloc[:63]
    alerts = generate_alerts(df, analyze(df))
    fever = [a for a in alerts if "infection" in a.title.lower()]
    assert fever and fever[0].severity == "red" and fever[0].action


def test_radiation_spike_alert():
    df = generate_mission(seed=11).iloc[:75]
    titles = [a.title for a in generate_alerts(df, analyze(df))]
    assert any("Radiation spike" in t for t in titles)


def test_healthy_day_has_no_red_alerts():
    df = generate_mission(seed=11, inject_events=False)
    alerts = generate_alerts(df, analyze(df))
    assert not [a for a in alerts if a.severity == "red"]


def test_manual_checkin_triggers_alerts():
    """Simulates the dashboard's daily check-in: a feverish, sleepless day."""
    df = generate_mission(seed=11, inject_events=False)
    row = df.iloc[-1].copy()
    row["day"] += 1
    row["body_temp_c"], row["symptom_score"], row["sleep_hours"] = 38.6, 7.0, 3.0
    df = pd.concat([df, row.to_frame().T], ignore_index=True)
    for col in INDICATORS:
        df[col] = df[col].astype(float)
    df["cumulative_radiation_msv"] = df["cumulative_radiation_msv"].astype(float)
    df["radiation_msv"] = df["radiation_msv"].astype(float)
    titles = [a.title for a in generate_alerts(df, analyze(df))]
    assert any("infection" in t.lower() for t in titles)


def test_charts_render():
    df = generate_mission(seed=11)
    result = analyze(df)
    for fig in (trend_chart(df, "resting_hr", result, True), domain_score_chart(df, result, True), radiation_chart(df)):
        assert fig is not None
    assert set(result.domain_scores.columns) == set(DOMAINS)
