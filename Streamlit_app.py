"""
app.py — Sprint Intelligence System
Full Streamlit Application | Premium dark AI product UI
KEY FIX: Never split HTML divs across multiple st.markdown() calls.
All card wrappers are fully self-contained to prevent raw HTML leaking.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import sys
import os

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explainability import explain_prediction
from recommendation import get_recommendations
from simulation import simulate_story_points
from optimizer import optimize_story_points
from assistant import answer_question
from retrospective import generate_retrospective

st.set_page_config(
    page_title="Sprint Intelligence System",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
  --bg:          #06080E;
  --surface:     #0C111A;
  --surface-2:   #111822;
  --surface-3:   #16202E;
  --border:      #1E2E42;
  --teal:        #00F5C8;
  --teal-glow:   rgba(0,245,200,0.12);
  --purple:      #9580FF;
  --red:         #FF4D6D;
  --amber:       #FFC043;
  --txt-h1:      #F5FAFF;
  --txt-heading: #E4EEFF;
  --txt-feature: #D0E2F4;
  --txt-body:    #96B0C8;
  --txt-muted:   #5A7090;
  --txt-dim:     #2E4258;
}

html, body, [data-testid="stAppViewContainer"] {
  background-color: var(--bg) !important;
  font-family: 'Inter', sans-serif !important;
  color: var(--txt-body) !important;
  font-size: 17px !important;
}
[data-testid="stHeader"]  { background: transparent !important; }
[data-testid="stSidebar"] { display: none !important; }
section[data-testid="stMain"] > div { background: transparent !important; }
.block-container { padding: 2.5rem 1.5rem 5rem !important; max-width: 900px !important; margin: auto; }

/* ── Inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
  background: var(--surface-2) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: #E0EEFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1.13rem !important;
  font-weight: 500 !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px var(--teal-glow) !important;
}
.stSelectbox > div > div {
  background: var(--surface-2) !important;
  border: 1.5px solid var(--border) !important;
  border-radius: 10px !important;
  color: #E0EEFF !important;
  font-size: 1.13rem !important;
}

/* ── Textarea (Ask AI) ── */
.stTextArea textarea {
  background: var(--surface-2) !important;
  border: 1.5px solid var(--border) !important;
  color: #E0EEFF !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 1.08rem !important;
  line-height: 1.65 !important;
  border-radius: 10px !important;
}
.stTextArea textarea:focus {
  border-color: var(--teal) !important;
  box-shadow: 0 0 0 3px var(--teal-glow) !important;
}
/* Placeholder — MUST be clearly visible on dark background */
.stTextArea textarea::placeholder {
  color: #6A90AE !important;
  opacity: 1 !important;
  font-style: italic !important;
  font-size: 1.05rem !important;
}

/* ── Field labels ── */
label,
.stSelectbox label,
.stNumberInput label,
.stSlider label,
.stTextArea label {
  color: #5A7A96 !important;
  font-size: 0.85rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  font-family: 'Space Grotesk', sans-serif !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] { background: var(--surface-3) !important; }
div[data-testid="stSlider"] > div > div > div { background: var(--teal) !important; }
div[data-testid="stSlider"] [role="slider"] {
  background: var(--teal) !important;
  border: 3px solid var(--bg) !important;
  box-shadow: 0 0 14px var(--teal-glow) !important;
}

/* ── Slider value label ── */
div[data-testid="stSlider"] p {
  font-size: 1.05rem !important;
  color: #A0C0DC !important;
  font-family: 'JetBrains Mono', monospace !important;
}

/* ── Button ── */
.stButton > button {
  background: linear-gradient(135deg, var(--teal) 0%, #00AAFF 100%) !important;
  color: #050810 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 800 !important;
  font-size: 1.12rem !important;
  letter-spacing: 0.04em !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.85rem 2rem !important;
  width: 100% !important;
  transition: all 0.22s ease !important;
  box-shadow: 0 4px 28px rgba(0,245,200,0.25) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 36px rgba(0,245,200,0.40) !important;
}

/* ── Divider ── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 2.8rem 0 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--surface) !important;
  border-radius: 14px !important;
  padding: 5px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: #4A6480 !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-weight: 600 !important;
  font-size: 0.95rem !important;
  border-radius: 10px !important;
  padding: 9px 20px !important;
  transition: all 0.15s ease !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #8AAEC8 !important; }
.stTabs [aria-selected="true"] {
  background: var(--surface-2) !important;
  color: var(--teal) !important;
  border: 1px solid rgba(0,245,200,0.2) !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 2rem 0 !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  padding: 1.25rem 1.3rem !important;
}
[data-testid="stMetricValue"] {
  color: #D8EEFF !important;
  font-family: 'Space Grotesk', sans-serif !important;
  font-size: 1.75rem !important;
  font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
  color: #4A6480 !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.09em !important;
  text-transform: uppercase !important;
  font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMetricDelta"] {
  font-size: 0.92rem !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

@keyframes pulse {
  0%,100% { opacity:1; box-shadow:0 0 10px var(--teal); }
  50%      { opacity:0.4; box-shadow:0 0 4px var(--teal); }
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HELPERS — all HTML is SELF-CONTAINED, never split across calls
# ═══════════════════════════════════════════════════════════════

def SH(text, color):
    """Subheading pill — returns inline HTML string, used INSIDE full card HTML only."""
    return (
        f'<div style="display:flex;align-items:center;gap:9px;margin-bottom:1.25rem;">'
        f'<div style="width:3px;height:19px;background:{color};border-radius:2px;opacity:0.9;flex-shrink:0;"></div>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:0.85rem;font-weight:700;'
        f'letter-spacing:0.17em;text-transform:uppercase;color:{color};">{text}</span>'
        f'</div>'
    )


def section_label(text, color="#00F5C8"):
    """Major section heading — prominent left bar."""
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:13px;margin:2.4rem 0 1.3rem 0;">'
        f'<div style="width:4px;height:30px;background:{color};border-radius:3px;'
        f'box-shadow:0 0 12px {color}60;flex-shrink:0;"></div>'
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:1.32rem;'
        f'font-weight:700;color:#E4EEFF;letter-spacing:-0.01em;">{text}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


def card(html, border="#1E2E42", glow="none"):
    """Fully self-contained card wrapper."""
    st.markdown(
        f'<div style="background:#0C111A;border:1px solid {border};border-radius:16px;'
        f'padding:1.7rem 1.9rem;margin-bottom:1.2rem;box-shadow:{glow};animation:fadeUp 0.3s ease;">'
        f'{html}</div>',
        unsafe_allow_html=True
    )



def risk_badge(level, probability):
    cfg = {
        "LOW":    ("#00F5C8", "rgba(0,245,200,0.07)",   "0 0 50px rgba(0,245,200,0.2),0 0 0 1px rgba(0,245,200,0.25)"),
        "MEDIUM": ("#FFC043", "rgba(255,192,67,0.07)",  "0 0 50px rgba(255,192,67,0.2),0 0 0 1px rgba(255,192,67,0.25)"),
        "HIGH":   ("#FF4D6D", "rgba(255,77,109,0.08)",  "0 0 50px rgba(255,77,109,0.24),0 0 0 1px rgba(255,77,109,0.3)"),
    }
    c, bg, glow = cfg.get(level, cfg["MEDIUM"])
    sym = "✓" if level == "LOW" else ("!" if level == "MEDIUM" else "✕")
    st.markdown(
        f'<div style="background:{bg};border:1px solid {c}38;border-radius:20px;'
        f'padding:2.6rem 2rem 2.2rem;text-align:center;box-shadow:{glow};'
        f'margin-bottom:1.3rem;animation:fadeUp 0.4s ease;">'
        f'<div style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:60px;height:60px;background:{c}18;border:2px solid {c}44;border-radius:50%;'
        f'font-family:\'Space Grotesk\',sans-serif;font-size:1.55rem;font-weight:800;color:{c};'
        f'margin-bottom:1.1rem;box-shadow:0 0 24px {c}30;">{sym}</div>'
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.82rem;font-weight:700;'
        f'letter-spacing:0.22em;text-transform:uppercase;color:{c};opacity:0.85;margin-bottom:0.5rem;">'
        f'Sprint Risk Level: {level}</div>'
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:4.6rem;font-weight:800;'
        f'color:{c};line-height:1;letter-spacing:-0.03em;text-shadow:0 0 35px {c}45;">'
        f'{probability:.1f}<span style="font-size:2.2rem;font-weight:500;opacity:0.6;">%</span></div>'
        f'<div style="margin-top:0.7rem;font-family:\'Inter\',sans-serif;font-size:0.96rem;'
        f'font-weight:500;color:#4E6882;letter-spacing:0.03em;">success probability</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def insight_row(feature, value, impact, reason, idx):
    is_risk = impact == "increases risk"
    c   = "#FF4D6D" if is_risk else "#00F5C8"
    arr = "↑" if is_risk else "↓"
    tbg = "rgba(255,77,109,0.10)" if is_risk else "rgba(0,245,200,0.10)"
    tbd = "rgba(255,77,109,0.28)" if is_risk else "rgba(0,245,200,0.28)"
    if feature == "buffer_ratio_previous":
       lbl = "Carry-over Ratio Previous"
    else:
       lbl = feature.replace("_", " ").title()
    return (
        f'<div style="display:flex;gap:1.1rem;align-items:flex-start;'
        f'padding:1.25rem 0;border-bottom:1px solid #10202E;">'
        f'<div style="min-width:34px;height:34px;flex-shrink:0;background:{tbg};'
        f'border:1px solid {c}30;border-radius:9px;display:flex;align-items:center;'
        f'justify-content:center;font-family:\'Space Grotesk\',sans-serif;'
        f'font-size:0.95rem;font-weight:700;color:{c};">{idx}</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="display:flex;align-items:center;flex-wrap:wrap;gap:9px;margin-bottom:7px;">'
        # Feature name — most prominent text in insights
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:1.1rem;'
        f'font-weight:700;color:#D8EEFF;letter-spacing:-0.005em;">{lbl}</span>'
        # Impact tag
        
        # Value — right aligned, monospace
        
        f'</div>'
        # Reason — clear readable body text
        f'<div style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;'
        f'color:#7AAABF;line-height:1.68;">{reason}</div>'
        f'</div></div>'
    )


def rec_card(priority, action, rationale):
    cfg = {
        "HIGH":   ("#FF4D6D","rgba(255,77,109,0.06)","rgba(255,77,109,0.22)"),
        "MEDIUM": ("#FFC043","rgba(255,192,67,0.06)","rgba(255,192,67,0.22)"),
        "LOW":    ("#00F5C8","rgba(0,245,200,0.06)", "rgba(0,245,200,0.22)"),
    }
    c, bg, bdr = cfg.get(priority, cfg["MEDIUM"])
    st.markdown(
        f'<div style="background:{bg};border:1px solid {bdr};border-left:3.5px solid {c};'
        f'border-radius:12px;padding:1.25rem 1.45rem;margin-bottom:1rem;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;flex-wrap:wrap;">'
        f'<span style="background:{c}22;color:{c};border:1px solid {c}38;border-radius:5px;'
        f'padding:3px 12px;font-family:\'Space Grotesk\',sans-serif;font-size:0.78rem;'
        f'font-weight:800;letter-spacing:0.14em;">{priority}</span>'
        # Action text — prominent
        f'<span style="font-family:\'Space Grotesk\',sans-serif;font-size:1.07rem;'
        f'font-weight:700;color:#D4E8FF;letter-spacing:-0.005em;">{action}</span>'
        f'</div>'
        # Rationale — readable body
        f'<div style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;'
        f'color:#7A9EBC;line-height:1.68;">{rationale}</div>'
        f'</div>',
        unsafe_allow_html=True
    )


def retro_section(title, items, color):
    icons = {"What Went Wrong": "◉", "What Went Well": "◎", "What to Improve": "◈"}
    icon = icons.get(title, "•")
    # Section title
    st.markdown(
        f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.12rem;'
        f'font-weight:700;color:{color};margin:1.5rem 0 1rem 0;'
        f'display:flex;align-items:center;gap:10px;letter-spacing:0.01em;">'
        f'<span style="font-size:1.2rem;opacity:0.9;">{icon}</span>{title}</div>',
        unsafe_allow_html=True
    )
    # Each item — fully self-contained
    for item in items:
        st.markdown(
            f'<div style="background:#111822;border:1px solid {color}18;'
            f'border-left:3px solid {color};border-radius:10px;'
            f'padding:1.1rem 1.35rem;margin-bottom:0.75rem;'
            f'font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;'
            f'color:#A4C0DC;line-height:1.72;">{item}</div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════
# MODEL LOAD
# ═══════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    for path in [
        "/mnt/user-data/uploads/classification_model.pkl",
        "classification_model.pkl",
        os.path.join(os.path.dirname(__file__), "classification_model.pkl"),
    ]:
        if os.path.exists(path):
            return joblib.load(path)
    raise FileNotFoundError("classification_model.pkl not found.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

FEATURE_COLS = list(model.feature_names_in_)


def build_input_df(team_size, planned_sp, sprint_days, total_tasks, high_priority,
                   prev_velocity, prev_planned, prev_completed, bugs, carry_over):
    vr  = planned_sp / (prev_velocity + 1)
    wpd = total_tasks / team_size
    cor = carry_over / (planned_sp + 1)
    ea  = prev_completed / (prev_planned + 1)
    brp = max((prev_planned - prev_completed) / (prev_planned + 1), 0)
    return pd.DataFrame([{
        "team_size": team_size, "planned_story_points": planned_sp,
        "sprint_duration_days": sprint_days, "total_tasks": total_tasks,
        "high_priority_tasks": high_priority, "previous_sprint_velocity": prev_velocity,
        "previous_planned_story_points": prev_planned,
        "previous_completed_story_points": prev_completed,
        "bugs_reported_previous_sprint": bugs, "carry_over_story_points": carry_over,
        "velocity_ratio": vr, "workload_per_developer": wpd, "carry_over_ratio": cor,
        "estimation_accuracy": ea, "buffer_ratio_previous": brp, "estimation_variance": 0.1,
    }])[FEATURE_COLS]


# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center;padding:3.5rem 0 2rem 0;animation:fadeUp 0.5s ease;">'
    # Live pill
    '<div style="display:inline-flex;align-items:center;gap:9px;'
    'background:rgba(0,245,200,0.06);border:1px solid rgba(0,245,200,0.2);'
    'border-radius:100px;padding:8px 22px;margin-bottom:1.8rem;">'
    '<div style="width:8px;height:8px;background:#00F5C8;border-radius:50%;'
    'box-shadow:0 0 10px #00F5C8;animation:pulse 2.2s infinite;"></div>'
    '<span style="font-family:\'Space Grotesk\',sans-serif;font-size:0.82rem;font-weight:700;'
    'color:#00F5C8;letter-spacing:0.18em;text-transform:uppercase;">'
    'AI-Powered · Pre-Sprint Intelligence</span></div>'
    # H1 — biggest thing on page
    '<h1 style="font-family:\'Space Grotesk\',sans-serif;'
    'font-size:clamp(2.8rem,6vw,3.8rem);font-weight:800;line-height:1.07;'
    'letter-spacing:-0.03em;margin:0 0 1rem 0;'
    'background:linear-gradient(135deg,#F5FAFF 45%,#00F5C8 100%);'
    '-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;">'
    'Sprint Intelligence<br>System</h1>'
    # Subtitle
    '<p style="font-family:\'Inter\',sans-serif;font-size:1.18rem;font-weight:400;'
    'color:#4E6882;margin:0;line-height:1.5;">'
    'ML-powered risk prediction &amp; planning intelligence for Scrum teams</p>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# INPUT SECTION — fully self-contained cards, NO split divs
# ═══════════════════════════════════════════════════════════════
section_label("Sprint Configuration")

# ── Card 1: Current Sprint ──────────────────────────────────
st.markdown(
    '<div style="background:#0C111A;border:1px solid #1E2E42;'
    'border-bottom:none;border-radius:18px 18px 0 0;padding:1.4rem 1.9rem 0.8rem;">'
    + SH("Current Sprint", "#00F5C8")
    + '</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<style>.card1-fields{background:#0C111A !important;}</style>',
    unsafe_allow_html=True
)
c1, c2, c3 = st.columns(3)
with c1: team_size   = st.number_input("Team Size",            min_value=1,  max_value=20,  value=6,  step=1)
with c2: sprint_days = st.selectbox(  "Sprint Duration (days)",[10, 14],                   index=1)
with c3: planned_sp  = st.number_input("Planned Story Points", min_value=5,  max_value=120, value=55, step=1)
c4, c5 = st.columns(2)
with c4: total_tasks   = st.number_input("Total Tasks",         min_value=1, max_value=60,              value=22, step=1)
with c5: high_priority = st.number_input("High Priority Tasks", min_value=0, max_value=int(total_tasks), value=min(6,int(total_tasks)), step=1)
st.markdown(
    '<div style="background:#0C111A;border:1px solid #1E2E42;'
    'border-top:none;border-radius:0 0 18px 18px;height:1.2rem;margin-bottom:0.9rem;"></div>',
    unsafe_allow_html=True
)

# ── Card 2: Previous Sprint History ────────────────────────
st.markdown(
    '<div style="background:#0C111A;border:1px solid #1E2E42;'
    'border-bottom:none;border-radius:18px 18px 0 0;padding:1.4rem 1.9rem 0.8rem;">'
    + SH("Previous Sprint History", "#9580FF")
    + '</div>',
    unsafe_allow_html=True
)
c6, c7 = st.columns(2)
with c6:
    prev_velocity = st.number_input("Previous Velocity",          min_value=5,  max_value=120, value=40, step=1)
    prev_planned  = st.number_input("Previous Planned Points",    min_value=5,  max_value=120, value=50, step=1)
with c7:
    prev_completed = st.number_input("Previous Completed Points", min_value=0,  max_value=120, value=38, step=1)
    bugs           = st.number_input("Bugs Reported (prev sprint)",min_value=0, max_value=30,  value=7,  step=1)
carry_over = st.number_input("Carry-Over Story Points", min_value=0, max_value=60, value=8, step=1)
st.markdown(
    '<div style="background:#0C111A;border:1px solid #1E2E42;'
    'border-top:none;border-radius:0 0 18px 18px;height:1.2rem;margin-bottom:1.4rem;"></div>',
    unsafe_allow_html=True
)


analyze_clicked = st.button("⚡  Analyze Sprint", type="primary")


# ═══════════════════════════════════════════════════════════════
# RESULTS
# ═══════════════════════════════════════════════════════════════
if analyze_clicked or "analysis_done" in st.session_state:

    if analyze_clicked:
        with st.spinner("Running AI analysis..."):
            input_df    = build_input_df(team_size, planned_sp, sprint_days, total_tasks,
                                         high_priority, prev_velocity, prev_planned,
                                         prev_completed, bugs, carry_over)
            prediction  = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            explanations    = explain_prediction(model, input_df, top_n=3)
            recommendations = get_recommendations(input_df)
            retro           = generate_retrospective(input_df, prediction, probability, explanations)
            st.session_state.update({
                "analysis_done":True, "input_df":input_df,
                "prediction":prediction, "probability":probability,
                "explanations":explanations, "recommendations":recommendations, "retro":retro,
            })

    input_df        = st.session_state["input_df"]
    prediction      = st.session_state["prediction"]
    probability     = st.session_state["probability"]
    explanations    = st.session_state["explanations"]
    recommendations = st.session_state["recommendations"]
    retro           = st.session_state["retro"]
    pct             = probability * 100

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # ── RESULT ───────────────────────────────────────────────
    section_label("Prediction Result")
    risk_level = "LOW" if pct >= 65 else ("MEDIUM" if pct >= 45 else "HIGH")
    risk_badge(risk_level, pct)

    # AI summary card
    top_feats = explanations[:3]

    summary_lines = []
    for e in top_feats:
        name = e["feature"].replace("_", " ").title()
        reason_short = e["reason"].split(".")[0]
        key_reasons = []

        for e in explanations:
            reason_short = e["reason"].split(".")[0]
            key_reasons.append(reason_short)

        if prediction == 1:
           reason_text = "The team setup looks stable with balanced workload and manageable risks."
        else:
           reason_text = "The current sprint setup shows signs of overload and planning inefficiencies."
    summary_text = ", ".join(summary_lines)
    if prediction == 0:
       status_span = "<strong style='color:#FF4D6D;font-weight:700;'>high risk of failure</strong>"
       if prediction == 1:
          intro = f"Based on the current sprint setup, this sprint is {status_span}."
       else:
          intro = f"Based on the current sprint setup, this sprint shows a {status_span}."
    else:
       status_span = "<strong style='color:#00F5C8;font-weight:700;'>likely to succeed</strong>"
       intro = f"This sprint is {status_span}."
    detail = f" {reason_text}"
    card(
        '<div style="display:flex;gap:14px;align-items:flex-start;">'
        '<div style="width:44px;height:44px;flex-shrink:0;background:rgba(0,245,200,0.09);'
        'border:1px solid rgba(0,245,200,0.2);border-radius:11px;'
        'display:flex;align-items:center;justify-content:center;font-size:1.3rem;margin-top:2px;">🤖</div>'
        '<div>'
        '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.8rem;font-weight:700;'
        'letter-spacing:0.16em;text-transform:uppercase;color:#2E4A60;margin-bottom:9px;">AI Analysis</div>'
        f'<div style="font-family:\'Inter\',sans-serif;font-size:1.1rem;font-weight:400;'
        f'color:#A8C8E4;line-height:1.75;">{intro}{detail}</div>'
        '</div></div>'
    )

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍  Root Causes", "💡  Recommendations", "🔮  What-If", "🤖  Ask AI", "📋  Retrospective"
    ])

    # ─── TAB 1: ROOT CAUSES ──────────────────────────────────
    with tab1:
        section_label("Root Cause Analysis", color="#9580FF")

        rows = "".join([insight_row(e["feature"],e["value"],e["impact"],e["reason"],i)
                        for i,e in enumerate(explanations,1)])
        card(
            SH("Top 3 SHAP Contributors","#9580FF") + rows,
            border="rgba(149,128,255,0.22)", glow="0 0 28px rgba(149,128,255,0.07)"
        )

        row = input_df.iloc[0]
        m1,m2,m3 = st.columns(3)
        with m1: st.metric("Velocity Ratio",      f"{row['velocity_ratio']:.2f}×",
                           delta="overcommit" if row['velocity_ratio']>1.1 else "on-track",
                           delta_color="inverse" if row['velocity_ratio']>1.1 else "normal")
        with m2: st.metric("Tasks / Developer",   f"{row['workload_per_developer']:.1f}",
                           delta="high" if row['workload_per_developer']>4 else "ok",
                           delta_color="inverse" if row['workload_per_developer']>4 else "normal")
        with m3: st.metric("Estimation Accuracy", f"{row['estimation_accuracy']:.0%}",
                           delta="low" if row['estimation_accuracy']<0.88 else "good",
                           delta_color="inverse" if row['estimation_accuracy']<0.88 else "normal")

    # ─── TAB 2: RECOMMENDATIONS ──────────────────────────────
    with tab2:
        section_label("Action Plan", color="#FFC043")
        if recommendations:
            for r in recommendations:
                rec_card(r["priority"], r["action"], r["rationale"])
        else:
            card('<div style="text-align:center;padding:1.2rem;font-family:\'Space Grotesk\',sans-serif;'
                 'font-size:1.1rem;font-weight:600;color:#00F5C8;">✓ No critical issues — sprint looks healthy.</div>',
                 border="rgba(0,245,200,0.2)")

    # ─── TAB 3: WHAT-IF ──────────────────────────────────────
    with tab3:
        section_label("What-If Simulation", color="#00F5C8")

        current_sp = int(input_df.iloc[0]["planned_story_points"])
        sim_sp = st.slider("Adjust Planned Story Points",
                           min_value=max(5,current_sp-30), max_value=min(100,current_sp+30),
                           value=current_sp, step=1, key="sim_slider")

        sr = simulate_story_points(model, input_df, sim_sp)
        old_p, new_p, delta = sr["old_probability"], sr["new_probability"], sr["improvement_percent"]

        ca,cb,cc = st.columns(3)
        with ca: st.metric("Current Points",  current_sp)
        with cb: st.metric("Simulated Points", sim_sp)
        with cc: st.metric("Δ Success Prob", f"{delta:+.1f}%",
                           delta_color="normal" if delta>=0 else "inverse")

        bco = "#00F5C8" if old_p>=60 else ("#FFC043" if old_p>=40 else "#FF4D6D")
        bcn = "#00F5C8" if new_p>=60 else ("#FFC043" if new_p>=40 else "#FF4D6D")

        card(
            f'<div style="margin-bottom:1.3rem;">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;">'
            f'<span style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:500;color:#4A6480;">Current ({current_sp} pts)</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.02rem;font-weight:600;color:{bco};">{old_p:.1f}%</span>'
            f'</div>'
            f'<div style="background:#16202E;border-radius:8px;height:10px;overflow:hidden;">'
            f'<div style="width:{min(old_p,100)}%;height:100%;background:{bco};border-radius:8px;box-shadow:0 0 12px {bco}55;"></div>'
            f'</div></div>'
            f'<div>'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:9px;">'
            f'<span style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:500;color:#4A6480;">Simulated ({sim_sp} pts)</span>'
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:1.02rem;font-weight:600;color:{bcn};">{new_p:.1f}%</span>'
            f'</div>'
            f'<div style="background:#16202E;border-radius:8px;height:10px;overflow:hidden;">'
            f'<div style="width:{min(new_p,100)}%;height:100%;background:{bcn};border-radius:8px;box-shadow:0 0 12px {bcn}55;"></div>'
            f'</div></div>'
            f'<div style="margin-top:1.4rem;padding-top:1.1rem;border-top:1px solid #10202E;'
            f'font-family:\'Inter\',sans-serif;font-size:1.0rem;color:#7A9EBC;font-style:italic;">'
            f'{sr["delta_label"]}</div>'
        )

        section_label("Optimal Story Points", color="#9580FF")
        if st.button("🎯  Find Optimal Scope", key="opt_btn"):
            with st.spinner("Optimizing..."):
                st.session_state["opt_result"] = optimize_story_points(model, input_df)

        if "opt_result" in st.session_state:
            opt = st.session_state["opt_result"]
            card(
                SH("Optimizer Result","#00F5C8") +
                f'<div style="display:flex;gap:3rem;flex-wrap:wrap;align-items:center;margin-bottom:1.3rem;">'
                f'<div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.8rem;font-weight:700;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:#3A5470;margin-bottom:5px;">Current</div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:2.5rem;font-weight:800;'
                f'color:#C0D8F0;letter-spacing:-0.025em;line-height:1;">{opt["current_story_points"]}'
                f'<span style="font-size:1.15rem;font-weight:500;color:#3A5470;margin-left:4px;">pts</span></div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.92rem;color:#4A6480;margin-top:4px;">'
                f'{opt["current_probability"]:.1f}% success</div></div>'
                f'<div style="font-size:2.2rem;color:#00F5C8;opacity:0.5;">→</div>'
                f'<div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.8rem;font-weight:700;'
                f'letter-spacing:0.12em;text-transform:uppercase;color:#00F5C8;margin-bottom:5px;">Optimal</div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:2.5rem;font-weight:800;'
                f'color:#00F5C8;letter-spacing:-0.025em;line-height:1;text-shadow:0 0 24px rgba(0,245,200,0.35);">'
                f'{opt["optimal_story_points"]}'
                f'<span style="font-size:1.15rem;font-weight:500;opacity:0.6;margin-left:4px;">pts</span></div>'
                f'<div style="font-family:\'JetBrains Mono\',monospace;font-size:0.92rem;color:#00F5C8;opacity:0.75;margin-top:4px;">'
                f'{opt["optimal_probability"]:.1f}% success</div></div></div>'
                f'<div style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;color:#7A9EBC;line-height:1.68;">'
                f'{opt["message"]}</div>',
                border="rgba(0,245,200,0.18)", glow="0 0 24px rgba(0,245,200,0.06)"
            )

    # ─── TAB 4: AI ASSISTANT ─────────────────────────────────
    with tab4:
        section_label("Sprint AI Assistant", color="#00F5C8")

        card(
            '<div style="margin-bottom:0.9rem;">'
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.12rem;font-weight:700;'
            'color:#C8E0F8;margin-bottom:0.6rem;">Ask anything about this sprint</div>'
            '<div style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;'
            'color:#6A8AAA;line-height:1.68;">Every answer uses the model predictions, SHAP '
            'explanations, and your actual sprint data — no generic responses.</div>'
            '</div>'
            '<div style="display:flex;flex-wrap:wrap;gap:8px;padding-top:1rem;border-top:1px solid #10202E;">'
            '<span style="background:rgba(0,245,200,0.08);border:1px solid rgba(0,245,200,0.2);'
            'border-radius:20px;padding:6px 16px;font-family:\'Inter\',sans-serif;'
            'font-size:0.92rem;font-weight:500;color:#00C8A8;">Why will this sprint fail?</span>'
            '<span style="background:rgba(0,245,200,0.08);border:1px solid rgba(0,245,200,0.2);'
            'border-radius:20px;padding:6px 16px;font-family:\'Inter\',sans-serif;'
            'font-size:0.92rem;font-weight:500;color:#00C8A8;">How do I improve success?</span>'
            '<span style="background:rgba(0,245,200,0.08);border:1px solid rgba(0,245,200,0.2);'
            'border-radius:20px;padding:6px 16px;font-family:\'Inter\',sans-serif;'
            'font-size:0.92rem;font-weight:500;color:#00C8A8;">Analyze workload</span>'
            '<span style="background:rgba(0,245,200,0.08);border:1px solid rgba(0,245,200,0.2);'
            'border-radius:20px;padding:6px 16px;font-family:\'Inter\',sans-serif;'
            'font-size:0.92rem;font-weight:500;color:#00C8A8;">What is my velocity risk?</span>'
            '</div>'
        )

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_q = st.text_area(
            "question_input",
            placeholder="e.g.  Why will this sprint fail?  ·  How can I improve?  ·  Analyze workload",
            height=110,
            key="chat_input",
            label_visibility="collapsed",
        )

        if st.button("💬  Ask", key="ask_btn") and user_q.strip():
            answer = answer_question(user_q, input_df, prediction, probability,
                                     explanations, recommendations)
            st.session_state["chat_history"].append({"q": user_q.strip(), "a": answer})

        for msg in reversed(st.session_state["chat_history"]):
            card(
                '<div style="margin-bottom:1rem;">'
                '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.78rem;font-weight:700;'
                'letter-spacing:0.16em;text-transform:uppercase;color:#263C52;margin-bottom:6px;">You</div>'
                f'<div style="font-family:\'Inter\',sans-serif;font-size:1.07rem;font-weight:400;'
                f'color:#B8D4EE;line-height:1.65;">{msg["q"]}</div>'
                '</div>'
                '<div style="border-top:1px solid #10202E;padding-top:1rem;">'
                '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.78rem;font-weight:700;'
                'letter-spacing:0.16em;text-transform:uppercase;color:#00C8A8;margin-bottom:7px;">🤖 AI Response</div>'
                f'<div style="font-family:\'Inter\',sans-serif;font-size:1.04rem;font-weight:400;'
                f'color:#A4C4E0;line-height:1.75;">'
                f'{msg["a"].replace(chr(10),"<br>").replace("**","<strong style=\'color:#D8EEFF;font-weight:600;\'>").replace("**","</strong>")}'
                f'</div></div>',
                border="rgba(0,245,200,0.14)"
            )

        if st.session_state["chat_history"]:
            if st.button("🗑  Clear Chat", key="clear_chat"):
                st.session_state["chat_history"] = []
                st.rerun()

    # ─── TAB 5: RETROSPECTIVE ────────────────────────────────
    with tab5:
        section_label("Sprint Retrospective", color="#9580FF")

        sc = "#00F5C8" if prediction==1 else "#FF4D6D"
        card(
            '<div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:1rem;">'
            '<div>'
            '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.8rem;font-weight:700;'
            'letter-spacing:0.16em;text-transform:uppercase;color:#2E4A60;margin-bottom:6px;">'
            'Auto-Generated Retrospective</div>'
            '<div style="font-family:\'Inter\',sans-serif;font-size:1.0rem;font-weight:400;color:#6A8AAA;">'
            'Based on pre-sprint data &amp; AI analysis — every point is specific to your team.'
            '</div></div>'
            f'<div style="background:{sc}12;border:1px solid {sc}32;border-radius:8px;'
            f'padding:8px 19px;font-family:\'Space Grotesk\',sans-serif;font-size:0.92rem;'
            f'font-weight:700;color:{sc};letter-spacing:0.07em;">'
            f'{retro["sprint_status"]} · {retro["probability"]}%</div>'
            '</div>'
        )

        retro_section("What Went Wrong",  retro["went_wrong"],  "#FF4D6D")
        retro_section("What Went Well",   retro["went_well"],   "#00F5C8")
        retro_section("What to Improve",  retro["to_improve"],  "#9580FF")


# ═══════════════════════════════════════════════════════════════
# FOOTER
# ═══════════════════════════════════════════════════════════════
st.markdown(
    '<div style="text-align:center;padding:3.5rem 0 1.5rem 0;border-top:1px solid #182840;margin-top:2rem;">'
    '<div style="font-family:\'Space Grotesk\',sans-serif;font-size:0.86rem;font-weight:500;'
    'color:#1E3048;letter-spacing:0.07em;line-height:2;">'
    'Sprint Intelligence System &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; SHAP Explainability'
    '<br><span style="color:#162436;font-size:0.8rem;">'
    'Built for Scrum Masters &amp; Agile Planning Teams</span></div></div>',
    unsafe_allow_html=True
)
