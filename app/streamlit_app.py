"""Streamlit demo (Step 9): pick a real applicant, change key fields, see the risk and why.

Run from the project root:  make app   (or: streamlit run app/streamlit_app.py)

It scores in-process with exactly the same code as the API. Set SCORING_API_URL
(for example http://localhost:8000) to send every request to the FastAPI service instead.
"""

import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st

from creditrisk.serving.service import Changes, ImpossibleChange, ScoringService, format_value

API_URL = os.environ.get("SCORING_API_URL", "").rstrip("/")

st.set_page_config(page_title="Credit risk demo", page_icon="🏦", layout="wide")


@st.cache_resource
def get_service() -> ScoringService:
    return ScoringService()


def _get(path: str) -> dict:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def list_applicants() -> list[int]:
    if API_URL:
        return _get("/applicants?limit=5000")["applicant_ids"]
    return get_service().applicant_ids


def get_profile(applicant_id: int) -> dict:
    if API_URL:
        return _get(f"/applicants/{applicant_id}")
    return get_service().profile(applicant_id).model_dump()


def get_score(applicant_id: int, changes: dict) -> dict:
    if API_URL:
        response = requests.post(f"{API_URL}/score", timeout=30,
                                 json={"applicant_id": applicant_id, "changes": changes})
        if response.status_code == 422:
            raise ImpossibleChange(str(response.json().get("detail")))
        response.raise_for_status()
        return response.json()
    return get_service().score(applicant_id, Changes(**changes)).model_dump()


ids = list_applicants()
st.session_state.setdefault("reset", 0)

# ---------------- sidebar: choose an applicant, then change fields ----------------
with st.sidebar:
    st.header("Applicant")
    st.button("🎲 Random applicant",
              on_click=lambda: st.session_state.update(applicant=random.choice(ids)))
    applicant_id = st.selectbox("Applicant ID (Kaggle test file)", ids, key="applicant")
    profile = get_profile(applicant_id)
    current = profile["current"]

    st.header("What if…")
    st.caption("Move a slider to see how the risk changes. Everything else stays the same.")
    tag = f"{applicant_id}_{st.session_state.reset}"  # new applicant or reset -> fresh sliders
    changes: dict = {}

    def slider(field: str, label: str, low: float, high: float, step: float, fmt: str) -> None:
        value = current[field]
        if value is None:
            st.caption(f"{label}: missing for this applicant")
            return
        new = st.slider(label, float(min(low, value)), float(max(high, value)), float(value),
                        float(step), format=fmt, key=f"{field}_{tag}")
        if abs(new - value) > 1e-9:
            changes[field] = new

    def optional_slider(field: str, label: str, missing_text: str, low: float, high: float,
                        step: float, start: float, fmt: str) -> None:
        if current[field] is not None:
            slider(field, label, low, high, step, fmt)
        elif not st.checkbox(missing_text, value=True, key=f"{field}_missing_{tag}"):
            changes[field] = st.slider(label, low, high, start, step, format=fmt,
                                       key=f"{field}_{tag}")

    slider("income", "Income", 25_000, 1_000_000, 5_000, "%.0f")
    slider("loan_amount", "Loan amount", 45_000, 4_050_000, 10_000, "%.0f")
    slider("annuity", "Loan annuity", 1_600, 260_000, 500, "%.0f")
    slider("age_years", "Age (years)", 21, 69, 1, "%.1f")
    optional_slider("years_employed", "Years employed", "No employment record", 0.0, 45.0, 0.5,
                    0.0, "%.1f")
    for k in (1, 2, 3):
        optional_slider(f"ext_source_{k}", f"External credit score {k}",
                        f"External score {k} unknown", 0.0, 1.0, 0.01, 0.5, "%.2f")
    st.button("Reset all changes",
              on_click=lambda: st.session_state.update(reset=st.session_state.reset + 1))

# ---------------- main page: score, reasons, profile ----------------
st.title("🏦 Credit risk demo")
try:
    original = get_score(applicant_id, {})
    result = get_score(applicant_id, changes) if changes else original
except ImpossibleChange as error:
    st.error(f"These values are not possible: {error}")
    st.stop()

st.caption(f"{result['model']}. Real applicants from the Kaggle test file, so no outcome is "
           "known. Probabilities are calibrated (checked in Step 10).")

left, middle, right = st.columns(3)
delta = (f"{(result['probability'] - original['probability']) * 100:+.1f} points vs. their own "
         "values" if changes else None)
left.metric("Probability of default", f"{result['probability']:.1%}", delta=delta,
            delta_color="inverse")
middle.metric("Compared with the average applicant", f"{result['times_average']:.1f}×",
              help=f"Average default rate in the training data: "
                   f"{result['average_default_rate']:.1%}")
right.metric("Typical applicant (model baseline)", f"{result['base_probability']:.1%}")
if changes:
    st.info("What if: " + ", ".join(f"{field.replace('_', ' ')} = {format_value(value)}"
                                    for field, value in changes.items()))

st.subheader("Why: the strongest reasons")
reasons = sorted(result["raises_risk"] + result["lowers_risk"], key=lambda r: r["impact"])
if reasons:
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(reasons) + 1.2))
    impacts = [r["impact"] for r in reasons]
    ax.barh(range(len(reasons)), impacts,
            color=["#c0392b" if v > 0 else "#27ae60" for v in impacts])
    ax.set_yticks(range(len(reasons)), [f"{r['label']} = {r['value']}" for r in reasons])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("push on the risk (red raises it, green lowers it; log-odds units)")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with st.expander("How to read this"):
    st.markdown(
        "The model starts from the **typical applicant** and every feature pushes the risk up "
        "or down. The bars are those pushes (TreeSHAP values from LightGBM): they add up "
        "exactly to this applicant's score. Only the 4 strongest each way are shown. Related "
        "features (for example the three external scores and their average) can appear "
        "separately; Step 12 groups them into proper reason codes.")

st.subheader("Applicant profile")
col_a, col_b = st.columns(2)
shown = {"Income": current["income"], "Loan amount": current["loan_amount"],
         "Loan annuity": current["annuity"], "Age (years)": current["age_years"],
         "Years employed": current["years_employed"],
         **{f"External score {k}": current[f"ext_source_{k}"] for k in (1, 2, 3)}}
col_a.dataframe(pd.DataFrame({"value": {k: format_value(v) for k, v in shown.items()}}))
col_b.dataframe(pd.DataFrame({"value": profile["details"]}))
history = [name for name, has in profile["history"].items() if has]
st.caption("History records found in: " + (", ".join(history) if history else "none"))
