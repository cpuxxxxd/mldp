import os
import sys
import time
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import ConfusionMatrixDisplay  # ok even if unused

st.set_page_config(page_title="Term Deposit Subscription Predictor", page_icon="🏦", layout="centered")

st.title("🏦 Term Deposit Subscription Predictor")
st.caption("Predict whether a customer is likely to subscribe to a term deposit so the bank can prioritise outbound calls.")

# ✅ Robust paths (works locally + Streamlit Cloud even if app is in a subfolder)
APP_DIR = Path(__file__).parent
ARTIFACT_PATH = APP_DIR / "best_model_pipeline.joblib"

# ---------- Friendly labels (UI) -> dataset codes (model) ----------
EDU_LABELS = {
    "basic.4y": "Primary (≈4 years)",
    "basic.6y": "Primary (≈6 years)",
    "basic.9y": "Lower secondary (≈9 years)",
    "high.school": "High school",
    "professional.course": "Professional course",
    "university.degree": "University degree",
    "illiterate": "No formal education",
    "unknown": "Unknown / not provided",
}
CONTACT_LABELS = {"cellular": "Mobile phone", "telephone": "Landline"}
DOW_LABELS = {"mon": "Mon", "tue": "Tue", "wed": "Wed", "thu": "Thu", "fri": "Fri"}
POUTCOME_LABELS = {
    "nonexistent": "No previous campaign recorded",
    "failure": "Previous campaign: did not subscribe",
    "success": "Previous campaign: subscribed",
}

def invert_map(d):
    return {v: k for k, v in d.items()}

INV_EDU = invert_map(EDU_LABELS)
INV_CONTACT = invert_map(CONTACT_LABELS)
INV_DOW = invert_map(DOW_LABELS)
INV_POUT = invert_map(POUTCOME_LABELS)

# --- FIX: ensure add_features exists for joblib unpickling ---
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    for col in ["pdays", "previous", "campaign"]:
        if col in d.columns:
            d[col] = pd.to_numeric(d[col], errors="coerce")

    if "pdays" in d.columns:
        d["contacted_before"] = (d["pdays"].fillna(999) != 999).astype(int)
        d["pdays_clean"] = d["pdays"].replace(999, np.nan)

    if "campaign" in d.columns and "previous" in d.columns:
        d["contacts_total"] = d["campaign"].fillna(0) + d["previous"].fillna(0)
        d["campaign_intensity"] = d["campaign"].fillna(0) / (d["previous"].fillna(0) + 1.0)

    if "month" in d.columns:
        month_map = {m: i for i, m in enumerate(
            ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], start=1
        )}
        mnum = d["month"].map(month_map).astype("float")
        d["month_sin"] = np.sin(2 * np.pi * (mnum / 12.0))
        d["month_cos"] = np.cos(2 * np.pi * (mnum / 12.0))

    if "day_of_week" in d.columns:
        dow_map = {"mon":1,"tue":2,"wed":3,"thu":4,"fri":5}
        dnum = d["day_of_week"].map(dow_map).astype("float")
        d["dow_sin"] = np.sin(2 * np.pi * (dnum / 5.0))
        d["dow_cos"] = np.cos(2 * np.pi * (dnum / 5.0))

    return d

# Register into modules that joblib might reference
for mod_name in ("main", "__main__"):
    if mod_name not in sys.modules:
        sys.modules[mod_name] = types.ModuleType(mod_name)
    setattr(sys.modules[mod_name], "add_features", add_features)
# --- END FIX ---

@st.cache_resource
def load_artifact(path: Path):
    # Helpful debug if missing (especially on Streamlit Cloud)
    if not path.exists():
        st.error(
            "Model artifact file not found.\n\n"
            f"Expected: {path}\n\n"
            f"App directory: {APP_DIR}\n"
            f"Files in app directory: {[p.name for p in APP_DIR.iterdir()]}\n\n"
            f"Working directory: {os.getcwd()}\n"
            f"Files in working directory: {os.listdir(os.getcwd())}"
        )
        st.stop()

    obj = joblib.load(path)

    pipe = obj["pipeline"]
    threshold = float(obj.get("threshold", 0.5))

    # feature_list should represent RAW input columns used by the pipeline
    if hasattr(pipe, "feature_names_in_"):
        feature_list = list(pipe.feature_names_in_)
    else:
        feature_list = obj.get("features", None)

    meta = {
        "artifact_path": str(path.resolve()),
        "artifact_mtime": time.ctime(path.stat().st_mtime),
        "sklearn_saved": obj.get("sklearn_version", "unknown"),
        "python_saved": obj.get("python_executable", "unknown"),
        "python_runtime": sys.executable,
    }

    defaults = obj.get("defaults", {})
    return pipe, threshold, feature_list, meta, defaults

# ✅ No try/except: load_artifact already gives a clear error + stops
pipe, default_threshold, feature_list, meta, defaults = load_artifact(ARTIFACT_PATH)

if not feature_list:
    st.error("Feature list missing. Re-export model with raw feature names.")
    st.stop()

# ---------- Session-state defaults ----------
DEFAULT_STATE = {
    "age": 35,
    "job": "admin.",
    "marital": "married",
    "education": "basic.4y",
    "default": "no",
    "housing": "no",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "poutcome": "nonexistent",
    "use_typical_timing": False,
    "unknown_campaign_history": False,
    "emp.var.rate": 1.10,
    "cons.price.idx": 93.99,
    "cons.conf.idx": -36.40,
    "euribor3m": 4.86,
    "nr.employed": 5191.0,
}

def init_state(defaults_from_joblib: dict):
    for k, v in DEFAULT_STATE.items():
        if k not in st.session_state:
            st.session_state[k] = defaults_from_joblib.get(k, v)

    # UI labels
    if "edu_label_ui" not in st.session_state:
        st.session_state["edu_label_ui"] = EDU_LABELS.get(st.session_state["education"], EDU_LABELS["unknown"])
    if "contact_label_ui" not in st.session_state:
        st.session_state["contact_label_ui"] = CONTACT_LABELS.get(st.session_state["contact"], CONTACT_LABELS["cellular"])
    if "dow_label_ui" not in st.session_state:
        st.session_state["dow_label_ui"] = DOW_LABELS.get(st.session_state["day_of_week"], "Mon")
    if "pout_label_ui" not in st.session_state:
        st.session_state["pout_label_ui"] = POUTCOME_LABELS.get(st.session_state["poutcome"], POUTCOME_LABELS["nonexistent"])

def apply_preset(preset: dict):
    for k, v in preset.items():
        st.session_state[k] = v

    st.session_state["edu_label_ui"] = EDU_LABELS.get(st.session_state["education"], EDU_LABELS["unknown"])
    st.session_state["contact_label_ui"] = CONTACT_LABELS.get(st.session_state["contact"], CONTACT_LABELS["cellular"])
    st.session_state["dow_label_ui"] = DOW_LABELS.get(st.session_state["day_of_week"], "Mon")
    st.session_state["pout_label_ui"] = POUTCOME_LABELS.get(st.session_state["poutcome"], POUTCOME_LABELS["nonexistent"])
    st.rerun()

init_state(defaults)

# Sidebar: threshold control
st.sidebar.header("Decision Settings")
threshold = st.sidebar.slider(
    "Decision threshold",
    0.05, 0.95, float(default_threshold), 0.01,
    help="Lower = more YES (more leads). Higher = fewer YES (more conservative)."
)

with st.expander("What do these fields mean?", expanded=False):
    st.write(
        "- **Education** is encoded as categories from the dataset (e.g., professional course, university degree).\n"
        "- **Contact method** is how the bank contacted the customer (mobile vs landline).\n"
        "- **Month / Day of contact** captures seasonality and call-centre timing patterns.\n"
        "- **Campaign history** describes how often the customer was contacted and what happened previously.\n"
        "- **Economic indicators** are macro values recorded at the campaign time; if unsure, keep defaults."
    )

def build_input_row(values: dict) -> pd.DataFrame:
    row = {f: values.get(f, None) for f in feature_list}
    return pd.DataFrame([row]).reindex(columns=feature_list, fill_value=np.nan)

# ---------- Presets ----------
st.markdown("### Preset Examples")
p1, p2, p3 = st.columns(3)

LOW = dict(
    campaign=4, pdays=999, previous=0, poutcome="nonexistent",
    contact="telephone", day_of_week="fri", month="nov", use_typical_timing=False,
    unknown_campaign_history=False
)
HIGH = dict(
    campaign=1, pdays=5, previous=2, poutcome="success",
    contact="cellular", day_of_week="wed", month="mar", use_typical_timing=False,
    unknown_campaign_history=False
)
NEW = dict(
    campaign=1, pdays=999, previous=0, poutcome="nonexistent",
    contact="cellular", day_of_week="mon", month="may", use_typical_timing=False,
    unknown_campaign_history=True
)

with p1:
    if st.button("Typical (Low Chance)"):
        apply_preset(LOW)
with p2:
    if st.button("Likely (Higher Chance)"):
        apply_preset(HIGH)
with p3:
    if st.button("New Customer"):
        apply_preset(NEW)

st.markdown("## Predict")

# ---------------- FORM ----------------
with st.form("predict_form", clear_on_submit=False):
    st.markdown("### Customer profile")
    c1, c2 = st.columns(2)

    with c1:
        st.number_input("Age", min_value=18, max_value=100, key="age")

        st.selectbox(
            "Job category",
            ["admin.","blue-collar","entrepreneur","housemaid","management","retired",
             "self-employed","services","student","technician","unemployed","unknown"],
            key="job"
        )

        st.selectbox("Marital status", ["married","single","divorced","unknown"], key="marital")

        edu_label = st.selectbox("Education level", list(EDU_LABELS.values()), key="edu_label_ui")
        st.session_state["education"] = INV_EDU[edu_label]

        st.selectbox("Credit default?", ["no","yes","unknown"], key="default")

    with c2:
        st.selectbox("Housing loan?", ["no","yes","unknown"], key="housing")
        st.selectbox("Personal loan?", ["no","yes","unknown"], key="loan")

        contact_label = st.selectbox("Contact method", list(CONTACT_LABELS.values()), key="contact_label_ui")
        st.session_state["contact"] = INV_CONTACT[contact_label]

        use_typical_timing = st.checkbox("I don't know campaign timing → use typical defaults", key="use_typical_timing")

        if use_typical_timing:
            st.info(
                f"Using: month={st.session_state['month']}, "
                f"day={DOW_LABELS.get(st.session_state['day_of_week'], st.session_state['day_of_week'])}"
            )
        else:
            st.selectbox("Month of contact", ["jan","feb","mar","apr","may","jun","jul","aug","sep","oct","nov","dec"], key="month")
            dow_label = st.selectbox("Day of contact", list(DOW_LABELS.values()), key="dow_label_ui")
            st.session_state["day_of_week"] = INV_DOW[dow_label]

    # -------- Campaign history (HIDE inputs when unknown) --------
    st.markdown("### Campaign history")
    st.caption("These describe how often the bank contacted the customer and past outcomes.")

    unknown_campaign = st.checkbox(
        "I don't know campaign history → use typical defaults",
        key="unknown_campaign_history"
    )

    campaign_box = st.empty()  # placeholder to hide/show inputs

    if unknown_campaign:
        st.session_state["campaign"] = 1
        st.session_state["pdays"] = 999
        st.session_state["previous"] = 0
        st.session_state["poutcome"] = "nonexistent"
        st.session_state["pout_label_ui"] = POUTCOME_LABELS["nonexistent"]
        st.info("Using defaults: campaign=1, pdays=999 (never), previous=0, poutcome=nonexistent")
    else:
        with campaign_box.container():
            h1, h2 = st.columns(2)
            with h1:
                st.number_input("Calls made in this campaign", 1, 50, key="campaign")
                st.number_input("Previous contacts (before this campaign)", 0, 50, key="previous")
            with h2:
                st.number_input("Days since last contact (999 = never)", 0, 999, key="pdays")
                pout_label = st.selectbox("Previous campaign outcome", list(POUTCOME_LABELS.values()), key="pout_label_ui")
                st.session_state["poutcome"] = INV_POUT[pout_label]

    # -------- Economic indicators --------
    with st.expander("Advanced: Economic indicators (optional)", expanded=False):
        st.caption("Macro-economic values at the time of the campaign. If unsure, keep defaults.")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.number_input("Employment variation rate (emp.var.rate)", key="emp.var.rate")
            st.number_input("Consumer price index (cons.price.idx)", key="cons.price.idx")
        with e2:
            st.number_input("Consumer confidence (cons.conf.idx)", key="cons.conf.idx")
            st.number_input("Interest rate (euribor3m)", key="euribor3m")
        with e3:
            st.number_input("Number employed (nr.employed)", key="nr.employed")

    submitted = st.form_submit_button("Predict")

# ----------------- PREDICTION -----------------
if submitted:
    values = {f: st.session_state.get(f, None) for f in feature_list}
    X_in = build_input_row(values)

    try:
        proba = float(pipe.predict_proba(X_in)[:, 1][0])
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred = int(proba >= threshold)

    st.markdown("## Result")
    st.metric("Probability of subscription (yes)", f"{proba:.3f}")
    if pred == 1:
        st.success(f"Predicted: YES (≥ {threshold:.2f}) — prioritise this customer for calling.")
    else:
        st.info(f"Predicted: NO (< {threshold:.2f}) — lower priority for calling.")

    with st.expander("Show input row used for prediction", expanded=False):
        st.dataframe(X_in)
        st.write("Non-null count:", int(X_in.notna().sum(axis=1).iloc[0]))
