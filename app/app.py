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
RATE_DATE          = "June 10, 2026"

EXPECTED_FEATURES = [
    "age", "bmi", "children",
    "sex_male", "smoker_yes",
    "region_northwest", "region_southeast", "region_southwest",
    "bmi_age_interaction",
]


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


# ── Page config ────────────────────────────────────────────────────────────────
def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded",
    )


# ── CSS ────────────────────────────────────────────────────────────────────────
# KEY ARCHITECTURAL DECISION:
# We do NOT wrap Streamlit widgets inside custom HTML divs — Streamlit renders
# widgets outside the HTML flow, so any <div class="panel"> that tries to wrap
# widgets produces orphaned empty boxes.
# Instead: use Streamlit's own st.container(border=True) for the card look,
# and style [data-testid="stVerticalBlockBorderWrapper"] to override its style.
# All custom HTML (headings, result cards, badges, profile grid) contains ONLY
# static content — never Streamlit widget placeholders.

def inject_styles() -> None:
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Force light mode on ALL elements ── */
/* Streamlit sometimes renders widget internals in dark mode even on a light page.
   These rules reset every widget label, input, and control to explicit light colours. */
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

/* ════════════════════════════════════════════
   WIDGET LABELS — the root of the blue-box problem.
   Streamlit wraps every label in a <label> or <p> inside
   [data-testid="stWidgetLabel"]. In dark mode these get a
   near-black background + white text, which appears as a
   blue-highlighted block on a light page.
   Fix: force transparent background + dark text on every label.
   ════════════════════════════════════════════ */
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

/* Number input labels */
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

/* Segmented control labels */
[data-testid="stSegmentedControl"] > label,
[data-testid="stSegmentedControl"] > div > label,
[data-testid="stSegmentedControl"] p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Selectbox labels */
[data-testid="stSelectbox"] > label,
[data-testid="stSelectbox"] p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* Any other label Streamlit might produce */
.main label,
.main [data-baseweb="form-control-label"],
.main .stMarkdown p {
    background: transparent !important;
    background-color: transparent !important;
    color: #2c4a42 !important;
}

/* Number input box itself */
div[data-testid="stNumberInput"] input {
    background: #ffffff !important;
    color: #0f1f1c !important;
    border: 1px solid #cdddd6 !important;
    border-radius: 7px !important;
    font-weight: 500 !important;
}

/* Selectbox box */
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

/* Segmented control pill buttons */
[data-testid="stSegmentedControl"] [role="radio"],
[data-testid="stSegmentedControl"] button {
    color: #2c4a42 !important;
    background: transparent !important;
}

/* Help tooltip icon */
[data-testid="stWidgetLabel"] [data-testid="stTooltipHoverTarget"] svg {
    color: #7aaa94 !important;
    fill: #7aaa94 !important;
}

/* ── Streamlit native card (st.container border=True) ── */
[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important;
    border: 1px solid #cdddd6 !important;
    border-radius: 12px !important;
    box-shadow: 0 1px 4px rgba(15,31,28,0.06) !important;
    padding: 0.25rem !important;
}
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.2rem 1.4rem !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f1f1c !important;
}
/* Reset the global label fix inside sidebar — sidebar intentionally uses light text */
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

/* ── Section heading (pure HTML, not wrapping widgets) ── */
.sec-head {
    font-size: 0.92rem;
    font-weight: 700;
    color: #0f1f1c;
    margin: 0.2rem 0 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #c0ddd0;
    letter-spacing: 0.01em;
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

/* ── st.container border=True override spacing ── */
/* Remove the label gap that causes the empty-box illusion */
[data-testid="stVerticalBlockBorderWrapper"] > div {
    padding: 1.3rem 1.5rem !important;
}

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
    rounded = round(abs(amount), 2)
    sign    = "-" if amount < 0 else ""
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
    return f"₹{sign}{grouped}.{d_str}"


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


# ── Input panel — uses st.container(border=True), heading inside via st.markdown ─
def render_inputs() -> dict[str, Any]:
    # Section heading rendered as Streamlit markdown (not HTML div wrapping widgets)
    st.markdown('<p class="sec-head">Patient Details</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        age    = st.number_input("Age (years)", min_value=18, max_value=100, value=30, step=1)
        bmi    = st.number_input(
            "BMI", min_value=10.0, max_value=60.0, value=25.0,
            step=0.1, format="%.1f",
            help="Body Mass Index — weight (kg) ÷ height² (m²)",
        )
        smoker = st.segmented_control(
            "Smoking status", ["Non-smoker", "Smoker"], default="Non-smoker"
        )
    with c2:
        sex      = st.segmented_control("Biological sex", ["Female", "Male"], default="Female")
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

    # ── Two-column body ───────────────────────────────────────────────────────
    # CORRECT PATTERN: st.container(border=True) creates the card.
    # Streamlit widgets are placed INSIDE the `with` block — no HTML divs wrapping them.
    # The section heading is the first item inside the container, rendered via st.markdown.
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        with st.container(border=True):
            inputs = render_inputs()

    with right:
        with st.container(border=True):
            render_prediction(model, features, error, inputs)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown("""
<div class="footer">
  Healthcare Cost Estimation System &nbsp;·&nbsp; Internship Project<br>
  Model output is indicative only — not a substitute for professional insurance advice.
</div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()