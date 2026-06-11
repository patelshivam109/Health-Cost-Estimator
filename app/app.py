from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import streamlit as st

# ── Constants ──────────────────────────────────────────────────────────────────
APP_TITLE          = "Healthcare Cost Estimator"
MODEL_FILE_NAME    = "healthcare_cost_model.pkl"
FEATURES_FILE_NAME = "feature_names.pkl"
USD_TO_INR         = 95.35
RATE_DATE          = "June 11, 2026"

EXPECTED_FEATURES = [
    "age", "bmi", "children",
    "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
    "bmi_age_interaction",
]

# Hardcoded fallback model comparison table (matches provided results)
FALLBACK_MODEL_METRICS = pd.DataFrame([
    {"Model": "Linear Regression", "MAE": 4176.92, "RMSE": 5956.45, "R2": 0.8069},
    {"Model": "Random Forest",     "MAE": 2619.61, "RMSE": 4665.05, "R2": 0.8816},
    {"Model": "XGBoost",           "MAE": 2592.25, "RMSE": 4456.00, "R2": 0.8919},
    {"Model": "Tuned XGBoost",     "MAE": 2444.16, "RMSE": 4229.80, "R2": 0.9026},
])


# ── Model loading ──────────────────────────────────────────────────────────────
def find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "models").is_dir():
            return p
    return start.parent


@st.cache_resource(show_spinner=False)
def load_artifacts(root: Path) -> tuple[Any | None, list[str], str | None]:
    model_path = root / "models" / MODEL_FILE_NAME
    feat_path  = root / "models" / FEATURES_FILE_NAME
    missing = [p.name for p in (model_path, feat_path) if not p.exists()]
    if missing:
        return None, EXPECTED_FEATURES, (
            f"Missing model files: {', '.join(missing)}. "
            "Place them in the project's models/ folder."
        )
    try:
        model    = joblib.load(model_path)
        features = joblib.load(feat_path)
    except Exception as exc:
        return None, EXPECTED_FEATURES, f"Could not load model artifacts — {exc}"
    if not isinstance(features, list) or not all(isinstance(f, str) for f in features):
        return None, EXPECTED_FEATURES, "Feature names artifact must be a list of strings."
    return model, features, None


@st.cache_data(show_spinner=False)
def load_dataset(root: Path) -> pd.DataFrame:
    for name in ("insurance_cleaned.csv", "cleaned_insurance.csv", "insurance.csv"):
        path = root / "data" / name
        if path.exists():
            return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_model_metrics(root: Path) -> pd.DataFrame:
    path = root / "reports" / "model_comparison.csv"
    if path.exists():
        return pd.read_csv(path)
    # Return the hardcoded fallback so the dashboard always shows comparison data
    return FALLBACK_MODEL_METRICS.copy()


@st.cache_data(show_spinner=False)
def load_feature_importance(root: Path) -> pd.DataFrame:
    path = root / "reports" / "feature_importance.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


# ── Page config ────────────────────────────────────────────────────────────────
def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ── CSS ────────────────────────────────────────────────────────────────────────
def inject_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Force light mode on ALL elements ── */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    color-scheme: light !important;
}

/* ── App shell ── */
.stApp {
    background: #eef1f0 !important;
    color: #0f1f1c !important;
}

.main .block-container {
    padding-top: 1.25rem;
    padding-bottom: 2.5rem;
    max-width: 1180px;
}

/* ── Widget labels ── */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] *,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] label {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stNumberInputContainer"] label,
[data-testid="stNumberInputContainer"] p,
div[data-testid="stNumberInput"] > label,
div[data-testid="stNumberInput"] > div > label {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stSegmentedControl"] > label,
[data-testid="stSegmentedControl"] > div > label,
[data-testid="stSegmentedControl"] p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

[data-testid="stSelectbox"] > label,
[data-testid="stSelectbox"] p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

.main label,
.main [data-baseweb="form-control-label"],
.main .stMarkdown p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
}

div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #0f1f1c !important;
    border: 1px solid #cdddd6 !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: #0f1f1c !important;
    border: 1px solid #cdddd6 !important;
    border-radius: 7px !important;
}
div[data-baseweb="select"] span {
    color: #0f1f1c !important;
    background: transparent !important;
}

[data-testid="stSegmentedControl"] [role="radio"],
[data-testid="stSegmentedControl"] button {
    color: #2c4a42 !important;
    background: transparent !important;
}

[data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"] svg {
    color: #7aaa94 !important;
    fill: #7aaa94 !important;
}

/* ── Streamlit native card ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #cdddd6 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(15,31,28,0.06) !important;
    padding: 0.25rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.3rem 1.5rem !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #ffffff !important;
    border-radius: 10px !important;
    padding: 0.25rem !important;
    border: 1px solid #cdddd6 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #2c4a42 !important;
    background: transparent !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #ffffff !important;
    background: #156645 !important;
}
[data-testid="stTabs"] button[role="tab"] p {
    color: inherit !important;
}
[data-testid="stTabs"] button {
    color: #0f1f1c !important;
    background: transparent !important;
    font-weight: 700 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #ffffff !important;
    border-bottom-color: #156645 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1f1c !important;
}
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] * {
    color: #b8cfc9 !important;
    background: transparent !important;
    background-color: transparent !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] strong,
[data-testid="stSidebar"] b {
    color: #e2eeea !important;
}
[data-testid="stSidebar"] code {
    background: rgba(255,255,255,0.09) !important;
    color: #6fcfa8 !important;
    border-radius: 3px;
    padding: 1px 5px;
    font-size: 0.82em;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
    margin: 0.6rem 0;
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    border-radius: 7px !important;
}
[data-testid="stSidebar"] details {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] details summary,
[data-testid="stSidebar"] details p {
    color: #b8cfc9 !important;
}

/* ── Hero ── */
.hero {
    background: #163027;
    border-radius: 12px;
    padding: 1.9rem 2.2rem;
    margin-bottom: 1.1rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 240px; height: 240px;
    border-radius: 50%;
    background: rgba(72,200,140,0.07);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #48c88c;
    margin-bottom: 0.45rem;
}
.hero h1 {
    font-size: 2rem;
    font-weight: 700;
    color: #e6f0ec;
    line-height: 1.12;
    margin: 0 0 0.6rem;
}
.hero-sub {
    font-size: 0.94rem;
    color: #82b0a4;
    line-height: 1.65;
    max-width: 660px;
    margin: 0 0 0.85rem;
}
.hero-notice {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.78rem;
    font-weight: 500;
    color: #6fd6a2;
    background: rgba(72,200,140,0.10);
    border: 1px solid rgba(72,200,140,0.22);
    border-radius: 6px;
    padding: 0.35rem 0.75rem;
}

/* ── KPI strip ── */
.kpi-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0,1fr));
    gap: 0.65rem;
    margin-bottom: 1.1rem;
}
.kpi-tile {
    background: #ffffff;
    border: 1px solid #cdddd6;
    border-radius: 9px;
    padding: 0.8rem 1rem;
    box-shadow: 0 1px 3px rgba(15,31,28,0.05);
}
.kpi-lbl {
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #5a7870;
}
.kpi-val {
    font-size: 0.93rem;
    font-weight: 700;
    color: #0f1f1c;
    margin-top: 0.2rem;
    line-height: 1.3;
}

/* ── Section heading ── */
.sec-head {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f1f1c;
    margin: 0.2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #c0ddd0;
    letter-spacing: 0.01em;
}

/* ── Model comparison table highlighting ── */
.model-comparison-winner {
    background: #f0fbf5 !important;
}

/* ── Result card ── */
.result-card {
    background: #f0fbf5;
    border: 1.5px solid #7ecfa8;
    border-radius: 10px;
    padding: 1.35rem 1.4rem 1.15rem;
    margin-bottom: 1rem;
}
.result-tag {
    display: flex;
    align-items: center;
    gap: 0.45rem;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #1a6845;
    margin-bottom: 0.45rem;
}
.result-main {
    font-size: 2.35rem;
    font-weight: 700;
    color: #0b4a30;
    line-height: 1.15;
}
.result-usd {
    font-size: 0.90rem;
    font-weight: 600;
    color: #226b48;
    margin-top: 0.2rem;
}
.result-hr {
    border: none;
    border-top: 1px solid #a8ddc0;
    margin: 0.75rem 0;
}
.result-note {
    font-size: 0.80rem;
    color: #1e5438;
    line-height: 1.55;
}
.result-note strong { color: #0b4a30; }

/* ── Badges ── */
.badge {
    display: inline-block;
    font-size: 0.67rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.16rem 0.50rem;
    border-radius: 4px;
    border: 1px solid;
    vertical-align: middle;
    line-height: 1.4;
}
.badge-inr { background:#fff8e1; color:#7a4f08; border-color:#f5cc6a; }
.badge-usd { background:#eef0ff; color:#2a38b0; border-color:#b5bcf0; }
.badge-best {
    display: inline-block;
    font-size: 0.64rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.12rem 0.45rem;
    border-radius: 4px;
    background: #e6f7ee;
    color: #145c38;
    border: 1px solid #7ecfa8;
    vertical-align: middle;
    margin-left: 0.4rem;
}

/* ── Profile grid ── */
.profile-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.15rem;
}
.pcell {
    background: #f4f8f6;
    border: 1px solid #cdddd6;
    border-radius: 7px;
    padding: 0.6rem 0.8rem;
}
.pcell-lbl {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #5a7870;
}
.pcell-val {
    font-size: 0.93rem;
    font-weight: 700;
    color: #0f1f1c;
    margin-top: 0.12rem;
    line-height: 1.3;
}
.pcell-sub {
    font-size: 0.74rem;
    font-weight: 500;
    color: #5a7870;
}

/* ── Risk tags ── */
.rtag {
    display: inline-block;
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 0.14rem 0.48rem;
    border-radius: 4px;
    vertical-align: middle;
    margin-left: 0.3rem;
    line-height: 1.4;
}
.rtag-low  { background:#e6f7ee; color:#145c38; border:1px solid #7ecfa8; }
.rtag-mid  { background:#fff8e1; color:#7a4f08; border:1px solid #f5cc6a; }
.rtag-high { background:#fff0f0; color:#8b1a1a; border:1px solid #f0a0a0; }

/* ── Empty state ── */
.empty-state {
    background: #f4f8f6;
    border: 1.5px dashed #9ecdb8;
    border-radius: 9px;
    padding: 1.8rem 1.4rem;
    text-align: center;
    color: #1e5438;
    font-size: 0.92rem;
    line-height: 1.7;
    margin-bottom: 1rem;
}
.empty-state strong { color: #0b4a30; }

/* ── Profile mini-heading ── */
.mini-head {
    font-size: 0.66rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: #5a7870;
    margin: 0.85rem 0 0.45rem;
}

/* ── Primary button ── */
div.stButton > button[kind="primary"] {
    background: #156645 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.93rem !important;
    min-height: 2.7rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 2px 8px rgba(21,102,69,0.22) !important;
    transition: background 0.15s ease;
}
div.stButton > button[kind="primary"]:hover {
    background: #0e5236 !important;
}

/* ── Input styling ── */
div[data-baseweb="select"] > div { border-radius: 7px !important; }
div[data-testid="stNumberInput"] input { border-radius: 7px !important; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding-top: 1.2rem;
    margin-top: 1.8rem;
    border-top: 1px solid #cdddd6;
    font-size: 0.81rem;
    color: #5a7870;
    line-height: 1.55;
}

/* ── Dashboard and markdown blocks ── */
.sec-head,
.stMarkdown .sec-head {
    color: #0f1f1c !important;
    background: transparent !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0f1f1c !important;
    background: transparent !important;
}

/* ── Table and dataframe ── */
[data-testid="stDataFrame"] * {
    color: #0f1f1c !important;
}

/* ── Responsive ── */
@media (max-width: 860px) {
    .hero h1     { font-size: 1.65rem; }
    .kpi-strip   { grid-template-columns: 1fr 1fr; }
    .result-main { font-size: 1.85rem; }
    .profile-grid{ grid-template-columns: 1fr; }
}
@media (max-width: 520px) {
    .kpi-strip { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar(root: Path, model_ok: bool, features: list[str]) -> None:
    with st.sidebar:
        st.markdown("## 🏥 Model Info")
        st.divider()
        st.markdown("**Algorithm**  \nTuned XGBoost Regressor")
        st.markdown("**Training data**  \nUS medical insurance *(USD)*")
        st.markdown(f"**Features**  \n{len(features)} engineered inputs")
        st.markdown(f"**USD → INR**  \n₹{USD_TO_INR:,.2f} per USD")
        st.markdown(
            f"<small style='color:#6b9a8a'>Rate date: {RATE_DATE}</small>",
            unsafe_allow_html=True,
        )
        st.divider()
        if model_ok:
            st.success("✓ Model loaded successfully")
        else:
            st.warning("⚠ Model files missing")
        with st.expander("Feature schema"):
            for i, f in enumerate(features, 1):
                st.write(f"{i}. `{f}`")


# ── Helpers ────────────────────────────────────────────────────────────────────
def format_inr(amount: float) -> str:
    """Format a positive float as an Indian-style rupee string (e.g. ₹1,23,456.78)."""
    rounded = round(amount, 2)
    i_str, d_str = f"{rounded:.2f}".split(".")
    if len(i_str) <= 3:
        grouped = i_str
    else:
        last3, lead = i_str[-3:], i_str[:-3]
        parts: list[str] = []
        while len(lead) > 2:
            parts.insert(0, lead[-2:])
            lead = lead[:-2]
        if lead:
            parts.insert(0, lead)
        grouped = ",".join([*parts, last3])
    return f"₹{grouped}.{d_str}"


def bmi_category(bmi: float) -> tuple[str, str]:
    if bmi < 18.5: return "Underweight", "rtag-mid"
    if bmi < 25.0: return "Normal",       "rtag-low"
    if bmi < 30.0: return "Overweight",   "rtag-mid"
    return "Obese", "rtag-high"


def smoker_risk(smoker: str) -> tuple[str, str]:
    return ("Higher risk", "rtag-high") if smoker == "Smoker" else ("Lower risk", "rtag-low")


# ── Feature engineering ────────────────────────────────────────────────────────
def build_features(inputs: dict[str, Any], feature_names: list[str]) -> pd.DataFrame:
    r = inputs["region"]
    row = {
        "age":               inputs["age"],
        "bmi":               inputs["bmi"],
        "children":          inputs["children"],
        "sex_male":          1 if inputs["sex"] == "Male" else 0,
        "smoker_yes":        1 if inputs["smoker"] == "Smoker" else 0,
        "region_northwest":  1 if r == "northwest" else 0,
        "region_southeast":  1 if r == "southeast" else 0,
        "region_southwest":  1 if r == "southwest" else 0,
        "bmi_age_interaction": inputs["bmi"] * inputs["age"],
    }
    cols = feature_names if feature_names else EXPECTED_FEATURES
    return pd.DataFrame([{c: row.get(c, 0) for c in cols}])


# ── Profile grid (pure HTML — no widgets inside) ───────────────────────────────
def render_profile(inputs: dict[str, Any]) -> None:
    bmi_cat, bmi_cls = bmi_category(inputs["bmi"])
    smk_lbl, smk_cls = smoker_risk(inputs["smoker"])
    child_word = "child" if inputs["children"] == 1 else "children"
    st.markdown(f"""
<div class="profile-grid">
  <div class="pcell">
    <div class="pcell-lbl">Age</div>
    <div class="pcell-val">{inputs["age"]} <span class="pcell-sub">yrs</span></div>
  </div>
  <div class="pcell">
    <div class="pcell-lbl">BMI</div>
    <div class="pcell-val">{inputs["bmi"]:.1f}
      <span class="rtag {bmi_cls}">{bmi_cat}</span>
    </div>
  </div>
  <div class="pcell">
    <div class="pcell-lbl">Smoking</div>
    <div class="pcell-val">{inputs["smoker"]}
      <span class="rtag {smk_cls}">{smk_lbl}</span>
    </div>
  </div>
  <div class="pcell">
    <div class="pcell-lbl">Region</div>
    <div class="pcell-val">{inputs["region"].title()}</div>
  </div>
  <div class="pcell">
    <div class="pcell-lbl">Sex</div>
    <div class="pcell-val">{inputs["sex"]}</div>
  </div>
  <div class="pcell">
    <div class="pcell-lbl">Dependants</div>
    <div class="pcell-val">{inputs["children"]} <span class="pcell-sub">{child_word}</span></div>
  </div>
</div>""", unsafe_allow_html=True)


# ── Input panel ────────────────────────────────────────────────────────────────
def render_inputs() -> dict[str, Any]:
    st.markdown('<p class="sec-head">Patient Details</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        age    = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        bmi    = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=25.0,
            step=0.1, format="%.1f",
            help="Body Mass Index — weight (kg) ÷ height² (m²)",
        )
        smoker = st.selectbox("Smoking Status", ["Non-smoker", "Smoker"], index=0)
    with c2:
        sex      = st.selectbox("Gender", ["Female", "Male"], index=0)
        children = st.number_input("Dependant children", min_value=0, max_value=10, value=0, step=1)
        region   = st.selectbox(
            "US region", ["Northeast", "Northwest", "Southeast", "Southwest"]
        )
    return {
        "age":      int(age),
        "sex":      sex,
        "bmi":      float(bmi),
        "children": int(children),
        "smoker":   smoker,
        "region":   region.lower(),
    }


# ── Prediction panel ───────────────────────────────────────────────────────────
def render_prediction(
    model: Any | None,
    features: list[str],
    error: str | None,
    inputs: dict[str, Any],
) -> None:
    st.markdown('<p class="sec-head">Cost Estimate</p>', unsafe_allow_html=True)

    if error:
        st.error(error)
        st.info("Place the `.pkl` files in the project's `models/` folder, then rerun.")
        return
    if model is None:
        st.error("Prediction model is unavailable.")
        return

    input_df = build_features(inputs, features)
    run = st.button("Estimate Healthcare Cost →", type="primary", use_container_width=True)

    if run:
        try:
            cost_usd = float(model.predict(input_df)[0])
        except Exception as exc:
            st.error(
                f"Prediction failed — verify the model matches the feature schema.  \n"
                f"*Detail:* {exc}"
            )
            return

        cost_inr = cost_usd * USD_TO_INR
        st.markdown(f"""
<div class="result-card">
  <div class="result-tag">
    Annual insurance charge
    &nbsp;<span class="badge badge-inr">INR</span>
    &nbsp;<span class="badge badge-usd">USD</span>
  </div>
  <div class="result-main">{format_inr(cost_inr)}</div>
  <div class="result-usd">USD&thinsp;${cost_usd:,.2f} &nbsp;·&nbsp; original model output</div>
  <hr class="result-hr">
  <div class="result-note">
    Model trained on US data — native output is <strong>USD</strong>.
    INR converted at ₹{USD_TO_INR:,.2f}&thinsp;/&thinsp;USD ({RATE_DATE}).
    Use the USD figure for any billing or policy reference.
  </div>
</div>
<div class="mini-head">Input profile used</div>""", unsafe_allow_html=True)
        render_profile(inputs)

    else:
        st.markdown("""
<div class="empty-state">
  <div style="font-size:2.1rem;margin-bottom:0.5rem">🩺</div>
  Complete the patient details on the left,<br>
  then click <strong>Estimate Healthcare Cost</strong><br>
  to see the predicted annual insurance charge.
</div>
<div class="mini-head">Current profile</div>""", unsafe_allow_html=True)
        render_profile(inputs)

    with st.expander("View prepared model input"):
        st.dataframe(input_df, use_container_width=True, hide_index=True)


# ── Dashboard tab ──────────────────────────────────────────────────────────────
def render_dashboard(root: Path) -> None:
    df      = load_dataset(root)
    metrics = load_model_metrics(root)   # always returns data (fallback if file missing)

    st.markdown('<p class="sec-head">Dashboard Insights</p>', unsafe_allow_html=True)

    # ── Top KPI row ──────────────────────────────────────────────────────────
    if not df.empty:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Records",        f"{len(df):,}")
        c2.metric("Average cost",   f"${df['charges'].mean():,.2f}")
        c3.metric("Median cost",    f"${df['charges'].median():,.2f}")
        c4.metric("Smoker records", f"{int((df['smoker'] == 'yes').sum()):,}")

    importance = load_feature_importance(root)

    # ── Model comparison (always visible) ────────────────────────────────────
    with st.container(border=True):
        # Find the best model (highest R2) to highlight it
        best_idx = int(metrics["R2"].idxmax())
        best_name = metrics.loc[best_idx, "Model"]

        st.markdown(
            f'<p class="sec-head">Model Comparison'
            f'<span class="badge-best">✓ {best_name}</span></p>',
            unsafe_allow_html=True,
        )

        # Format numeric columns for display
        display_metrics = metrics.copy()
        display_metrics["MAE"]  = display_metrics["MAE"].map(lambda x: f"{x:,.2f}")
        display_metrics["RMSE"] = display_metrics["RMSE"].map(lambda x: f"{x:,.2f}")
        display_metrics["R2"]   = display_metrics["R2"].map(lambda x: f"{x:.4f}")

        st.dataframe(
            display_metrics,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn("Model", width="medium"),
                "MAE":   st.column_config.TextColumn("MAE ",  width="small"),
                "RMSE":  st.column_config.TextColumn("RMSE ", width="small"),
                "R2":    st.column_config.TextColumn("R²",         width="small"),
            },
        )

        # Inline metric strip for the best model
        km1, km2, km3 = st.columns(3)
        best_row = metrics.loc[best_idx]
        km1.metric("Best MAE",  f"{float(best_row['MAE']):,.2f}")
        km2.metric("Best RMSE", f"{float(best_row['RMSE']):,.2f}")
        km3.metric("Best R²",   f"{float(best_row['R2']):.4f}")

    # ── Two-column detail section ─────────────────────────────────────────────
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        if not df.empty:
            with st.container(border=True):
                st.markdown('<p class="sec-head">Cost Drivers — Smoking</p>', unsafe_allow_html=True)
                smoker_summary = (
                    df.groupby("smoker")["charges"]
                    .agg(["count", "mean", "median"])
                    .reset_index()
                    .rename(columns={
                        "smoker":  "Smoking status",
                        "count":   "Records",
                        "mean":    "Average cost",
                        "median":  "Median cost",
                    })
                )
                smoker_summary["Average cost"] = smoker_summary["Average cost"].map(lambda x: f"${x:,.2f}")
                smoker_summary["Median cost"]  = smoker_summary["Median cost"].map(lambda x: f"${x:,.2f}")
                st.dataframe(smoker_summary, use_container_width=True, hide_index=True)

            with st.container(border=True):
                st.markdown('<p class="sec-head">Regional Summary</p>', unsafe_allow_html=True)
                region_summary = (
                    df.groupby("region")["charges"]
                    .agg(["count", "mean", "median"])
                    .reset_index()
                    .rename(columns={
                        "region":  "Region",
                        "count":   "Records",
                        "mean":    "Average cost",
                        "median":  "Median cost",
                    })
                )
                region_summary["Region"]       = region_summary["Region"].str.title()
                region_summary["Average cost"] = region_summary["Average cost"].map(lambda x: f"${x:,.2f}")
                region_summary["Median cost"]  = region_summary["Median cost"].map(lambda x: f"${x:,.2f}")
                st.dataframe(region_summary, use_container_width=True, hide_index=True)

    with right:
        if not importance.empty:
            with st.container(border=True):
                st.markdown('<p class="sec-head">Top Feature Importance</p>', unsafe_allow_html=True)
                st.bar_chart(importance.head(6).set_index("Feature")["Importance"])
                st.dataframe(importance.head(6), use_container_width=True, hide_index=True)


# ── Plots & data tab ───────────────────────────────────────────────────────────
def render_plots_and_data(root: Path) -> None:
    df = load_dataset(root)
    if df.empty:
        st.warning("Dataset file is unavailable.")
        return

    st.markdown('<p class="sec-head">Generated Plots and Data</p>', unsafe_allow_html=True)

    chart_pairs = [
        ("Cost distribution",  "eda_cost_distribution.png"),
        ("Smoking impact",      "eda_smoking_impact.png"),
        ("Age analysis",        "eda_age_distribution.png"),
        ("BMI analysis",        "eda_bmi_distribution.png"),
        ("Region analysis",     "eda_region_analysis.png"),
        ("Correlation heatmap", "eda_correlation_heatmap.png"),
    ]

    for left_item, right_item in zip(chart_pairs[::2], chart_pairs[1::2]):
        left, right = st.columns(2, gap="large")
        for col, (title, filename) in ((left, left_item), (right, right_item)):
            with col:
                with st.container(border=True):
                    st.markdown(f'<p class="sec-head">{title}</p>', unsafe_allow_html=True)
                    path = root / "reports" / filename
                    if path.exists():
                        st.image(str(path), use_container_width=True)
                    else:
                        st.info(f"{filename} is not available.")

    with st.container(border=True):
        st.markdown('<p class="sec-head">Cleaned Dataset Preview</p>', unsafe_allow_html=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown('<p class="sec-head">Dataset Statistics</p>', unsafe_allow_html=True)
        st.dataframe(df.describe(), use_container_width=True)


# ── Model insights tab ─────────────────────────────────────────────────────────
def render_model_insights(root: Path, model: Any | None, features: list[str]) -> None:
    st.markdown('<p class="sec-head">Model Insights</p>', unsafe_allow_html=True)

    metrics = load_model_metrics(root)   # always has data
    best    = metrics.sort_values("R2", ascending=False).iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", str(best["Model"]))
    k2.metric("MAE",  f"{float(best['MAE']):,.2f}")
    k3.metric("RMSE", f"{float(best['RMSE']):,.2f}")
    k4.metric("R²",   f"{float(best['R2']):.4f}")

    with st.container(border=True):
        st.markdown('<p class="sec-head">Full Evaluation Metrics</p>', unsafe_allow_html=True)
        display = metrics.copy()
        display["MAE"]  = display["MAE"].map(lambda x: f"{x:,.2f}")
        display["RMSE"] = display["RMSE"].map(lambda x: f"{x:,.2f}")
        display["R2"]   = display["R2"].map(lambda x: f"{x:.4f}")
        st.dataframe(display, use_container_width=True, hide_index=True)

    importance = load_feature_importance(root)
    if importance.empty and model is not None and hasattr(model, "feature_importances_"):
        importance = (
            pd.DataFrame({"Feature": features, "Importance": model.feature_importances_})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )

    if not importance.empty:
        with st.container(border=True):
            st.markdown('<p class="sec-head">Feature Importance</p>', unsafe_allow_html=True)
            st.bar_chart(importance.set_index("Feature")["Importance"])
            st.dataframe(importance, use_container_width=True, hide_index=True)

    image_path = root / "reports" / "feature_importance.png"
    if image_path.exists():
        with st.container(border=True):
            st.markdown('<p class="sec-head">Feature Importance Visualization</p>', unsafe_allow_html=True)
            st.image(str(image_path), use_container_width=True)

    if importance.empty:
        st.info("Feature importance data is unavailable for the loaded model.")


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    configure_page()
    inject_styles()

    root = find_project_root(Path(__file__).resolve().parent)
    model, features, error = load_artifacts(root)

    render_sidebar(root, model is not None and error is None, features)

    # ── Hero ─────────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="hero">
  <div class="hero-eyebrow">ML-powered cost intelligence</div>
  <h1>Healthcare Cost Estimator</h1>
  <p class="hero-sub">
    Predict annual healthcare insurance charges using a tuned XGBoost
    regression model trained on demographic, lifestyle, and regional
    insurance attributes.
  </p>
  <span class="hero-notice">
    ⚠&nbsp; Training data is in <strong style="color:#6fd6a2">USD</strong>
    &nbsp;—&nbsp; INR shown for local readability at
    ₹{USD_TO_INR:,.2f}&thinsp;/&thinsp;USD&thinsp;({RATE_DATE})
  </span>
</div>""", unsafe_allow_html=True)

    # ── KPI strip ────────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="kpi-strip">
  <div class="kpi-tile">
    <div class="kpi-lbl">Algorithm</div>
    <div class="kpi-val">XGBoost Regressor</div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-lbl">Engineered features</div>
    <div class="kpi-val">{len(features)}</div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-lbl">Training currency</div>
    <div class="kpi-val">USD → INR</div>
  </div>
  <div class="kpi-tile">
    <div class="kpi-lbl">Exchange rate</div>
    <div class="kpi-val">₹{USD_TO_INR:,.2f} / USD</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Four tabs ─────────────────────────────────────────────────────────────
    dashboard_tab, prediction_tab, plots_tab, insights_tab = st.tabs([
        "📊 Dashboard Insights",
        "🩺 Healthcare Cost Prediction",
        "📈 Plots and Data",
        "🔬 Model Insights",
    ])

    with dashboard_tab:
        render_dashboard(root)

    with prediction_tab:
        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            with st.container(border=True):
                inputs = render_inputs()
        with right:
            with st.container(border=True):
                render_prediction(model, features, error, inputs)

    with plots_tab:
        render_plots_and_data(root)

    with insights_tab:
        render_model_insights(root, model, features)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  Healthcare Cost Estimation System &nbsp;·&nbsp; Internship Project<br>
  Model output is indicative only — not a substitute for professional insurance advice.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()