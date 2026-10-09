"""Streamlit demo (Steps 9 and 16): pick an applicant, change key fields, see the risk, the
decision (approve / refer / decline) and why.

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

from creditrisk.explain.counterfactual import next_decision
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


@st.cache_data(show_spinner="Searching for the smallest change…")
def get_counterfactuals(applicant_id: int) -> dict:
    if API_URL:
        return _get(f"/applicants/{applicant_id}/counterfactuals")
    return next_decision(get_service(), applicant_id)


def is_synthetic() -> bool:
    return _get("/health").get("synthetic", False) if API_URL else get_service().synthetic


ids = list_applicants()
st.session_state.setdefault("reset", 0)

# ---------------- sidebar: choose an applicant, then change fields ----------------
with st.sidebar:
    st.header("Applicant")
    st.button("🎲 Random applicant",
              on_click=lambda: st.session_state.update(applicant=random.choice(ids)))
    source = "made-up applicants" if is_synthetic() else "Kaggle test file"
    applicant_id = st.selectbox(f"Applicant ID ({source})", ids, key="applicant")
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

# ---------------- main page: headline, then three tabs ----------------
st.title("🏦 Credit risk demo")
try:
    original = get_score(applicant_id, {})
    result = get_score(applicant_id, changes) if changes else original
except ImpossibleChange as error:
    st.error(f"These values are not possible: {error}")
    st.stop()

who = ("Made-up applicants (no real data)" if is_synthetic()
       else "Real applicants from the Kaggle test file, so no outcome is known")
st.caption(f"{result['model']}. {who}. Probabilities are calibrated (checked on the test split).")

decision = result["decision"]
BADGE = {"approve": "✅ Approve", "refer": "🟡 Refer to a person", "decline": "⛔ Decline"}
left, middle, right = st.columns(3)
delta = (f"{(result['probability'] - original['probability']) * 100:+.1f} points vs. their own "
         "values" if changes else None)
left.metric("Probability of default", f"{result['probability']:.1%}", delta=delta,
            delta_color="inverse")
was = original["decision"]["outcome"]
middle.metric("Decision", BADGE[decision["outcome"]], delta_color="off",
              delta=f"was: {was}" if was != decision["outcome"] else None)
right.metric("Compared with the average applicant", f"{result['times_average']:.1f}×",
             help=f"Average default rate in the training data: "
                  f"{result['average_default_rate']:.1%}")
if changes:
    st.info("What if: " + ", ".join(f"{field.replace('_', ' ')} = {format_value(value)}"
                                    for field, value in changes.items()))

why_tab, decision_tab, whatif_tab = st.tabs(["Score + why", "Decision", "What-if"])

with why_tab:
    st.subheader("The strongest reasons")
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
            "The model starts from the **typical applicant** "
            f"({result['base_probability']:.1%}) and every feature pushes the risk up or down. "
            "The bars are those pushes (TreeSHAP values from LightGBM): they add up exactly to "
            "this applicant's score. Only the 4 strongest each way are shown.")
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

with decision_tab:
    outcome, p = decision["outcome"], result["probability"]
    text = {
        "approve": f"The risk ({p:.1%}) is below {decision['approve_below']:.2%}: approved "
                   "automatically. At most about 10% of real defaulters end up here (9.8% on "
                   "the test split).",
        "refer": f"The risk ({p:.1%}) is between {decision['approve_below']:.2%} and "
                 f"{decision['decline_above']:.2%}: the model is not sure enough either way, so "
                 "a person decides. About 47% of applicants land here.",
        "decline": f"The risk ({p:.1%}) is above {decision['decline_above']:.2%}: declined "
                   "automatically. At most about 10% of good customers end up here (9.8% on "
                   "the test split).",
    }[outcome]
    {"approve": st.success, "refer": st.warning, "decline": st.error}[outcome](
        f"**{BADGE[outcome]}.** {text}")
    if outcome == "refer":
        st.markdown(f"**Suggestion for the reviewer (money rule):** "
                    f"{decision['money_rule']}. Lending is profitable on average when the risk is "
                    f"below {decision['break_even']:.1%} (10% margin, 50% loss on a default).")
    if outcome != "approve" and result["reason_codes"]:
        st.markdown("**Main reasons for the risk** (what a declined customer would be told):")
        st.markdown("\n".join(f"{i}. {code}" for i, code in enumerate(result["reason_codes"], 1)))
    with st.expander("How the decision is made"):
        st.markdown(
            "Each class has its own cut-off from a held-out set (class-conditional conformal "
            "prediction, Step 11), so the error promise holds for defaulters and for good "
            "customers separately. It assumes new applicants look like the old ones; the drift "
            "monitor (PSI, Step 16) checks that. The promise holds on average, not inside every "
            "group: by age, younger applicants are declined more often (Step 13).")

with whatif_tab:
    st.markdown("Move the sliders in the sidebar to test any change. Or let the model search "
                "for the **smallest** change that reaches the next better decision: borrow less "
                "(loan and annuity together) or a higher income. Age, gender and the external "
                "scores are never changed.")
    if st.button("Find the smallest change", key=f"cf_{applicant_id}"):
        advice = get_counterfactuals(applicant_id)
        if advice["goal"] is None:
            st.success("Already approved: nothing to change.")
        else:
            st.markdown(f"From **{advice['decision']}** to **{advice['goal']}** "
                        f"(risk below {advice['target']:.2%}), starting from their own values:")
            for option in advice["options"]:
                if option["factor"] is None:
                    st.markdown(f"- **{option['option']}:** not enough within the range tried.")
                else:
                    change = (f"borrow {1 - option['factor']:.0%} less" if
                              option["option"] == "smaller loan"
                              else f"income × {option['factor']:.1f}")
                    st.markdown(f"- **{option['option']}:** {change} → risk "
                                f"{option['probability']:.1%} (was "
                                f"{option['probability_before']:.1%})")
