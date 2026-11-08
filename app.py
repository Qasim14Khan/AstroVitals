"""AstroVitals - Streamlit dashboard.

Run with:  streamlit run app.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.anomaly import analyze, overall_status, top_drivers
from src.charts import domain_score_chart, radiation_chart, trend_chart
from src.indicators import DOMAINS, INDICATORS, RADIATION_LIMIT_MSV
from src.recommend import generate_alerts
from src.simulate import ASTRONAUTS, generate_mission

st.set_page_config(page_title="AstroVitals", page_icon="🚀", layout="wide")

STATUS_COLOR = {"Nominal": "green", "Watch": "orange", "Alert": "red"}
SEVERITY_LABEL = {"red": "Act now", "yellow": "Watch", "info": "For information"}

# Default values for the daily check-in form: (min, max, step) per field.
CHECKIN_FIELDS = {
    "resting_hr": (30.0, 160.0, 1.0),
    "hrv": (5.0, 150.0, 1.0),
    "systolic_bp": (80.0, 220.0, 1.0),
    "diastolic_bp": (40.0, 140.0, 1.0),
    "sleep_hours": (0.0, 14.0, 0.1),
    "exercise_min": (0.0, 300.0, 5.0),
    "mood": (1.0, 10.0, 0.5),
    "stress": (1.0, 10.0, 0.5),
    "reaction_ms": (150.0, 800.0, 5.0),
    "body_temp_c": (34.0, 42.0, 0.1),
    "symptom_score": (0.0, 10.0, 0.5),
    "weight_kg": (40.0, 150.0, 0.1),
    "grip_strength_kg": (10.0, 80.0, 0.5),
    "radiation_msv": (0.0, 50.0, 0.1),
}


# ---------------------------------------------------------------- data handling
def get_data(name: str) -> pd.DataFrame:
    """Each astronaut's data lives in session state so check-ins can be added."""
    store = st.session_state.setdefault("data", {})
    if name not in store:
        store[name] = generate_mission(days=180, seed=ASTRONAUTS[name])
    return store[name]


@st.cache_data(show_spinner=False)
def run_analysis(df: pd.DataFrame):
    return analyze(df)


def submit_checkin():
    """Callback: add today's check-in as a new mission day."""
    name = st.session_state["astronaut"]
    df = get_data(name)
    last = df.iloc[-1]
    row = {"day": int(last["day"]) + 1, "date": last["date"] + pd.Timedelta(days=1)}
    for field in CHECKIN_FIELDS:
        row[field] = float(st.session_state[f"ci_{name}_{field}"])
    row["cumulative_radiation_msv"] = round(float(last["cumulative_radiation_msv"]) + row["radiation_msv"], 1)
    row["injected_event"] = ""
    st.session_state["data"][name] = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    st.session_state["view_day"] = len(st.session_state["data"][name])
    st.session_state["checkin_done"] = f"Day {row['day']} saved and analyzed."


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🚀 AstroVitals")
st.sidebar.caption("Self-monitoring for long-duration missions")
astronaut = st.sidebar.selectbox("Astronaut", list(ASTRONAUTS), key="astronaut")
full = get_data(astronaut)

if st.session_state.get("view_day", len(full)) > len(full):
    st.session_state["view_day"] = len(full)
view_day = st.sidebar.slider(
    "Mission day to view", min_value=30, max_value=len(full), value=len(full), key="view_day",
    help="Replay the mission day by day. The analysis only uses data up to the selected day, like real life.",
)
show_truth = st.sidebar.checkbox(
    "Demo mode: shade simulated events", value=False,
    help="Highlights the events we injected into the simulation, so you can check the detector found them.",
)
st.sidebar.info("All data is simulated. This is a research prototype, not medical advice.")

df = full.iloc[:view_day].reset_index(drop=True)
analysis = run_analysis(df)
row = len(df) - 1
alerts = generate_alerts(df, analysis)
overall = overall_status(analysis, row)

# ---------------------------------------------------------------- header
st.title("AstroVitals")
st.caption(f"{astronaut} · Mission day {int(df['day'].iloc[-1])} · {df['date'].iloc[-1].strftime('%d %b %Y')}")

tab_overview, tab_trends, tab_checkin, tab_about = st.tabs(["Overview", "Trends", "Daily check-in", "About"])

# ---------------------------------------------------------------- overview
with tab_overview:
    cum = float(df["cumulative_radiation_msv"].iloc[-1])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mission day", int(df["day"].iloc[-1]))
    c2.markdown("**Overall status**")
    c2.markdown(f"## :{STATUS_COLOR[overall]}[{overall}]")
    c3.metric("Radiation dose", f"{cum:.0f} mSv", f"{100 * cum / RADIATION_LIMIT_MSV:.0f}% of illustrative limit", delta_color="off")
    c4.metric("Unusual-pattern flag (ML)", "Yes" if bool(analysis.unusual.iloc[row]) else "No")

    st.subheader("Health domains")
    cols = st.columns(len(DOMAINS))
    for col, dom in zip(cols, DOMAINS):
        status = analysis.domain_status.loc[row, dom]
        score = analysis.domain_scores.loc[row, dom]
        with col:
            st.markdown(f"**{dom}**")
            st.markdown(f"### :{STATUS_COLOR[status]}[{status}]")
            st.caption(f"Risk score {score:.1f}")
            drivers = top_drivers(analysis, row, dom)
            if drivers:
                st.caption("Driven by: " + ", ".join(name for name, _ in drivers))

    st.subheader("What to do now")
    if not alerts:
        st.success("Everything looks normal compared with your personal baseline. Keep up your routine: exercise, sleep, and daily check-in.")
    for a in alerts:
        text = f"**{a.title}** ({a.domain})  \n{a.detail}  \n**Action:** {a.action}"
        if a.severity == "red":
            st.error(text)
        elif a.severity == "yellow":
            st.warning(text)
        else:
            st.info(text)

    st.subheader("Risk scores over time")
    fig = domain_score_chart(df, analysis, show_truth)
    st.pyplot(fig)
    plt.close(fig)

# ---------------------------------------------------------------- trends
with tab_trends:
    domain = st.selectbox("Domain", list(DOMAINS) + ["Radiation"])
    if domain == "Radiation":
        fig = radiation_chart(df)
        st.pyplot(fig)
        plt.close(fig)
    else:
        indicators = DOMAINS[domain]
        for i in range(0, len(indicators), 2):
            cols = st.columns(2)
            for col, ind in zip(cols, indicators[i:i + 2]):
                with col:
                    fig = trend_chart(df, ind, analysis, show_truth)
                    st.pyplot(fig)
                    plt.close(fig)
    st.caption("Shaded blue band = your own normal range (first 21 days of the mission, ±2 SD). "
               "Red dots = days the Isolation Forest model marked as unusual.")
    with st.expander("Raw data"):
        shown = df.drop(columns=["injected_event"])
        st.dataframe(shown)
        st.download_button("Download CSV", shown.to_csv(index=False), file_name="astrovitals_data.csv", mime="text/csv")

# ---------------------------------------------------------------- check-in
with tab_checkin:
    st.subheader("Daily health check-in")
    st.write("Enter today's readings. They are added as a new mission day and the whole analysis updates. "
             "Try entering a high temperature or very little sleep and watch the alerts change.")
    if st.session_state.get("checkin_done"):
        st.success(st.session_state.pop("checkin_done"))
    last = full.iloc[-1]
    with st.form("checkin"):
        cols = st.columns(3)
        for i, (field, (lo, hi, step)) in enumerate(CHECKIN_FIELDS.items()):
            label = INDICATORS[field]["label"] if field in INDICATORS else "Radiation dose today (mSv)"
            with cols[i % 3]:
                st.number_input(
                    label, min_value=lo, max_value=hi, step=step,
                    value=float(min(max(last[field], lo), hi)), key=f"ci_{astronaut}_{field}",
                )
        st.form_submit_button("Save check-in", on_click=submit_checkin)
    if st.button("Reset this astronaut's data"):
        st.session_state["data"].pop(astronaut, None)
        st.session_state.pop("view_day", None)
        st.rerun()

# ---------------------------------------------------------------- about
with tab_about:
    st.subheader("The problem")
    st.write(
        "On long missions, astronauts face radiation, isolation and confinement, altered gravity and a closed "
        "environment. These can cause immune changes, bone and muscle loss, cardiovascular events and behavioral "
        "health problems. With communication delays to ground, astronauts need to spot changes in themselves."
    )
    st.subheader("How AstroVitals works")
    st.markdown(
        """
1. **Collect** daily indicators: heart rate, HRV, blood pressure, sleep, exercise, mood, stress, reaction time,
   temperature, symptoms, weight, grip strength and radiation dose.
2. **Personal baseline**: each astronaut is compared with their *own* first 21 days, not a population average.
3. **Risk scores**: for each of four domains (Cardiovascular, Bone & Muscle, Immune, Behavioral) we measure how many
   standard deviations worse than normal the astronaut is, smoothed over 3 days. Thresholds: Watch ≥ 1, Alert ≥ 2.
4. **Machine learning**: an Isolation Forest looks at all indicators together and flags unusual days, even when no single
   value crosses a threshold.
5. **Actions**: rules turn flags into plain-language steps, and say when to contact the flight surgeon.
        """
    )
    st.subheader("Data and limitations")
    st.markdown(
        """
- **All data is simulated.** Real astronaut data is restricted. The ranges and trends are modeled on publicly described
  NASA Human Research Program findings and could be replaced with real sensor or wearable data.
- Thresholds (including the radiation limit) are illustrative and have not been clinically validated.
- **Not medical advice.** It is a research prototype built for the NASA Space Apps Challenge.
        """
    )
    st.caption("Sources for background: NASA Human Research Program (nasa.gov/hrp) and NASA Open Science Data Repository (osdr.nasa.gov).")
