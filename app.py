import base64
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import data

# Ensure `data` module is hot-reloaded on Streamlit Cloud when data.py is updated in-place
_data_mtime = Path(data.__file__).stat().st_mtime if getattr(data, "__file__", None) else None
if (
    getattr(data, "_LOADED_MTIME", None) != _data_mtime
    or not hasattr(data, "compute_correlations_and_insights")
    or not hasattr(data, "compute_body_state_now")
    or not hasattr(data, "compute_race_predictions")
    or not hasattr(data, "evaluate_daily_training_decision")
):
    data = importlib.reload(data)
    data._LOADED_MTIME = _data_mtime

ICONS_DIR = Path(__file__).resolve().parent / "icons"

st.set_page_config(
    page_title="Garmin Personal Dashboard",
    page_icon=str(ICONS_DIR / "optimalhealth.png") if (ICONS_DIR / "optimalhealth.png").exists() else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------- Icon System
ICON_MAP = {
    "overview": "overview.png",
    "run": "running.png",
    "running": "running.png",
    "cycling": "bike.png",
    "bike": "bike.png",
    "walk": "walking.png",
    "walking": "walking.png",
    "hiking": "hiking.png",
    "swimming": "swimming.png",
    "yoga": "yoga.png",
    "strength": "bench.png",
    "other": "multisport.png",
    "multisport": "multisport.png",
    "sleep": "sleep.png",
    "recovery": "recovery.png",
    "heart": "heartbeat.png",
    "heartbeat": "heartbeat.png",
    "race": "race_predictor.png",
    "race_predictor": "race_predictor.png",
    "speed": "speed.png",
    "calories": "calories.png",
    "cadence": "cadence.png",
    "insights": "insights.png",
    "optimalhealth": "optimalhealth.png",
    "plan": "plan.png",
    "injury_risk": "injury_risk.png",
    "import": "import.png",
    "settings": "settings.png",
    "curious": "curious.png",
}


@st.cache_data
def icon_b64(name: str) -> str:
    fname = ICON_MAP.get(name.lower().strip(), name)
    if not fname.endswith(".png"):
        fname += ".png"
    p = ICONS_DIR / fname
    if not p.exists():
        p = ICONS_DIR / "multisport.png"
    if not p.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def icon_html(name: str, size: int = 20, bg: str = "#1f2937", accent: str = "#58a6ff", badge: bool = True, mr: int = 10) -> str:
    uri = icon_b64(name)
    if not uri:
        return ""
    if badge:
        box = size + 14
        return (
            f'<span class="icon-badge" style="width:{box}px;height:{box}px;background:{bg};'
            f'border:1px solid {accent}44;margin-right:{mr}px;">'
            f'<img src="{uri}" style="width:{size}px;height:{size}px;filter:invert(1) brightness(1.8);" alt="{name}"/>'
            f'</span>'
        )
    return (
        f'<img src="{uri}" style="width:{size}px;height:{size}px;filter:invert(1) brightness(1.8);'
        f'vertical-align:middle;margin-right:{mr}px;object-fit:contain;" alt="{name}"/>'
    )


def sport_icon_name(act_type: str, cat: str) -> str:
    t = (act_type or "").lower()
    if "hik" in t:
        return "hiking"
    if "swim" in t:
        return "swimming"
    if "yoga" in t or "pilates" in t or "mobility" in t:
        return "yoga"
    if "strength" in t or "weight" in t or "fitness_equipment" in t or "gym" in t:
        return "strength"
    return {"Run": "running", "Cycling": "bike", "Walk": "walking"}.get(cat, "multisport")


# --------------------------------------------------------------- Palettes & Theme
GH_SCALES = {
    "green": ["#161b22", "#0e4429", "#006d32", "#26a641", "#39d353"],
    "orange": ["#161b22", "#4a1e16", "#7d2d1e", "#c44b31", "#f78166"],
    "blue": ["#161b22", "#0c2d6b", "#1158c7", "#1f6feb", "#58a6ff"],
    "purple": ["#161b22", "#271846", "#4c2889", "#794bc4", "#a371f7"],
    "amber": ["#161b22", "#3d2a0d", "#6e4b11", "#a87419", "#e3b341"],
}
COLORS = {"Run": "#f78166", "Cycling": "#58a6ff", "Walk": "#39d353", "Strength": "#e3b341", "Other": "#a371f7"}

TAB_SPECS = [
    ("Overview", "overview"),
    ("Running", "running"),
    ("Cycling", "bike"),
    ("Walking", "walking"),
    ("Weightlifting", "strength"),
    ("Sleep", "sleep"),
    ("Recovery & Fitness", "recovery"),
    ("Race Predictions", "race_predictor"),
    ("Insights", "insights"),
    ("Activity Log", "plan"),
]

tab_icon_css = "\n".join(
    f'.stTabs [data-baseweb="tab-list"] button:nth-of-type({i}) p::before {{'
    f'content:"";display:inline-block;width:16px;height:16px;margin-right:8px;vertical-align:-3px;'
    f'background-image:url("{icon_b64(ico)}");background-size:contain;background-repeat:no-repeat;'
    f'filter:invert(1) brightness(1.8);}}'
    for i, (_, ico) in enumerate(TAB_SPECS, start=1)
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

.stApp {{
    background: radial-gradient(circle at 15% 0%, #111827 0%, #0d1117 55%);
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #e6edf3;
}}
.block-container {{
    padding-top: 1.5rem;
    padding-bottom: 3rem;
    max-width: 1360px;
}}
[data-testid="stSidebar"] {{
    background: #090d13 !important;
    border-right: 1px solid #21262d !important;
}}
header[data-testid="stHeader"] {{
    background: transparent !important;
}}

/* Icon Badge */
.icon-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: 10px;
    flex-shrink: 0;
    box-shadow: 0 3px 10px rgba(0, 0, 0, 0.35);
}}
.icon-badge img {{
    object-fit: contain;
    display: block;
}}

/* Top Hero Header */
.hero-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 14px;
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 14px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 8px 26px rgba(0,0,0,0.35);
}}
.hero-left {{
    display: flex;
    align-items: center;
    gap: 14px;
}}
.hero-title {{
    font-size: 1.45rem;
    font-weight: 800;
    color: #f0f6fc;
    letter-spacing: -0.02em;
    margin: 0;
    line-height: 1.2;
}}
.hero-sub {{
    font-size: 0.82rem;
    color: #8b949e;
    margin-top: 3px;
}}
.status-pill {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 6px 13px;
    border-radius: 999px;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    background: rgba(57, 211, 83, 0.12);
    color: #39d353;
    border: 1px solid rgba(57, 211, 83, 0.35);
}}
.status-pill.demo {{
    background: rgba(210, 153, 34, 0.14);
    color: #e3b341;
    border-color: rgba(210, 153, 34, 0.4);
}}
.status-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: currentColor;
    box-shadow: 0 0 8px currentColor;
}}

/* Section Header */
.sec-hdr {{
    display: flex;
    align-items: center;
    margin: 8px 0 14px 0;
}}
.sec-hdr h3 {{
    margin: 0;
    font-size: 1.12rem;
    font-weight: 700;
    color: #f0f6fc;
    letter-spacing: -0.01em;
}}
.sec-hdr small {{
    color: #8b949e;
    font-size: 0.8rem;
    margin-left: 10px;
}}

/* KPI Metric Card */
.kpi-card {{
    background: linear-gradient(145deg, #161b22 0%, #11161d 100%);
    border: 1px solid #262c36;
    border-radius: 12px;
    padding: 15px 17px;
    margin-bottom: 12px;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.28);
    transition: transform 0.18s ease, border-color 0.18s ease;
}}
.kpi-card:hover {{
    transform: translateY(-2px);
    border-color: #3b4454;
}}
.kpi-top {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 8px;
}}
.kpi-label {{
    font-size: 0.74rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b949e;
}}
.kpi-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.52rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.15;
}}
.kpi-sub {{
    font-size: 0.75rem;
    color: #8b949e;
    margin-top: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}
.kpi-sub .pos {{ color: #39d353; font-weight: 600; }}
.kpi-sub .neg {{ color: #f85149; font-weight: 600; }}

/* Tabs with Custom PNG Icons */
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: #12171f;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid #21262d;
    margin-bottom: 14px;
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    padding: 8px 14px;
    color: #8b949e;
    font-weight: 600;
    font-size: 0.86rem;
    border: none !important;
    background: transparent;
    transition: all 0.15s ease;
}}
.stTabs [data-baseweb="tab"]:hover {{
    color: #e6edf3;
    background: rgba(255, 255, 255, 0.04);
}}
.stTabs [aria-selected="true"] {{
    background: #1f2733 !important;
    color: #f0f6fc !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.35);
}}
.stTabs [data-baseweb="tab-highlight"] {{
    display: none !important;
}}
{tab_icon_css}

/* Heatmap Legend Footer */
.gh-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 0.76rem;
    color: #8b949e;
    margin: -6px 4px 16px 4px;
}}
.gh-legend {{
    display: inline-flex;
    align-items: center;
    gap: 5px;
}}
.gh-box {{
    width: 11px;
    height: 11px;
    border-radius: 2px;
    display: inline-block;
}}

/* Activity Feed Card */
.feed-card {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    border: 1px solid #262c36;
    border-radius: 12px;
    padding: 13px 16px;
    margin-bottom: 9px;
    background: linear-gradient(145deg, #141921 0%, #0d1117 100%);
    transition: border-color 0.15s ease, transform 0.15s ease;
}}
.feed-card:hover {{
    border-color: #3b4454;
    transform: translateX(2px);
}}
.feed-left {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.feed-title {{
    font-weight: 700;
    color: #f0f6fc;
    font-size: 0.95rem;
}}
.feed-date {{
    color: #8b949e;
    font-size: 0.78rem;
    margin-top: 2px;
}}
.feed-metrics {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 14px;
}}
.feed-pill {{
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: #c9d1d9;
    background: #161b22;
    border: 1px solid #262c36;
    border-radius: 8px;
    padding: 5px 10px;
}}

/* Training Status Banner */
.ts-banner {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px;
    background: linear-gradient(135deg, #161b22 0%, #11161e 100%);
    border: 1px solid #2b323d;
    border-radius: 14px;
    padding: 16px 20px;
    margin: 4px 0 18px 0;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
}}
.ts-left {{
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 260px;
    flex: 1;
}}
.ts-badge {{
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.ts-headline {{
    font-size: 0.84rem;
    color: #8b949e;
    margin-top: 4px;
}}
.ts-metrics {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
}}
.ts-chip {{
    background: #0d1117;
    border: 1px solid #262c36;
    border-radius: 10px;
    padding: 8px 13px;
    min-width: 125px;
}}
.ts-chip-lbl {{
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #8b949e;
}}
.ts-chip-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.98rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-top: 2px;
}}
.ts-chip-sub {{
    font-size: 0.7rem;
    color: #8b949e;
}}

/* Readiness Action Buttons ("Can I run today?" / "Can I lift today?") */
div[data-testid="stButton"] > button[kind="primary"] {{
    background: linear-gradient(135deg, #161b22 0%, #1c2430 100%) !important;
    border: 1px solid #30363d !important;
    color: #f0f6fc !important;
    font-weight: 700 !important;
    font-size: 0.92rem !important;
    border-radius: 12px !important;
    padding: 10px 18px !important;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.3) !important;
    transition: all 0.18s ease !important;
}}
div[data-testid="stButton"] > button[kind="primary"]:hover {{
    border-color: #58a6ff !important;
    background: linear-gradient(135deg, #1c2430 0%, #222d3d 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(88, 166, 255, 0.18) !important;
}}
div[role="dialog"] {{
    background: #0d1117 !important;
    border: 1px solid #30363d !important;
    border-radius: 16px !important;
}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------- Helpers
def kpi_card(col, label: str, value: str, sub: str = "", icon: str = "overview", accent: str = "#58a6ff"):
    badge = icon_html(icon, size=17, bg=f"{accent}1a", accent=accent, badge=True, mr=0)
    col.markdown(
        f'<div class="kpi-card">'
        f'  <div class="kpi-top"><span class="kpi-label">{label}</span>{badge}</div>'
        f'  <div class="kpi-val">{value}</div>'
        f'  <div class="kpi-sub">{sub if sub else "&nbsp;"}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_header(title: str, subtitle: str = "", icon: str = "overview", accent: str = "#58a6ff"):
    badge = icon_html(icon, size=18, bg=f"{accent}1a", accent=accent, badge=True, mr=10)
    sub_html = f"<small>{subtitle}</small>" if subtitle else ""
    st.markdown(f'<div class="sec-hdr">{badge}<h3>{title}</h3>{sub_html}</div>', unsafe_allow_html=True)


def layout(fig, h=330):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#12171f",
        plot_bgcolor="#12171f",
        height=h,
        margin=dict(l=14, r=14, t=42, b=14),
        legend_title_text="",
        font=dict(family="Inter, sans-serif", color="#c9d1d9"),
        title_font=dict(size=14, color="#f0f6fc"),
        hoverlabel=dict(bgcolor="#161b22", bordercolor="#30363d", font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#1f2630", zerolinecolor="#1f2630")
    fig.update_yaxes(gridcolor="#1f2630", zerolinecolor="#1f2630")
    return fig


def fmt_time(s):
    s = int(round(s))
    h, r = divmod(s, 3600)
    m, sec = divmod(r, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def pace(speed_kmh):
    return fmt_time(3600 / speed_kmh) + " /km" if speed_kmh and speed_kmh > 0 else "–"


# --------------------------------------------------------------- Data Loading
CACHE_SCHEMA_VERSION = "v4_naps_202bpm"


@st.cache_data(ttl=3600, show_spinner="Syncing Garmin Personal API…")
def load_official_api(api_key: str, access_token: str, days: int, schema_ver: str = CACHE_SCHEMA_VERSION):
    client = data.GarminPersonalAPIClient(api_key=api_key, access_token=access_token)
    return data.fetch_official_api(client, days)


@st.cache_data(ttl=3600, show_spinner="Syncing Garmin Connect…")
def load_legacy(email: str, password: str, tokens: str, days: int, schema_ver: str = CACHE_SCHEMA_VERSION):
    g = data.login(email, password, tokens)
    return data.fetch_live(g, days)


@st.cache_data(ttl=3600, show_spinner="Syncing Hevy API…")
def load_hevy_api(api_key: str, days: int):
    client = data.HevyAPIClient(api_key=api_key)
    return data.fetch_hevy(client, days)


try:
    cfg = dict(st.secrets["garmin"]) if "garmin" in st.secrets else {}
except Exception:
    cfg = {}

try:
    hevy_cfg = dict(st.secrets["hevy"]) if "hevy" in st.secrets else {}
except Exception:
    hevy_cfg = {}

secret_api_key = cfg.get("api_key", "")
secret_access_token = cfg.get("access_token", "")
has_legacy_secrets = bool(cfg.get("email") or cfg.get("tokens"))
secret_hevy_key = hevy_cfg.get("api_key", "") or cfg.get("hevy_api_key", "")

with st.sidebar:
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">'
        f'{icon_html("optimalhealth", size=22, bg="#26a64122", accent="#39d353", badge=True, mr=0)}'
        f'<div><div style="font-weight:800;font-size:1.1rem;color:#f0f6fc;">Garmin + Hevy</div>'
        f'<div style="font-size:0.74rem;color:#8b949e;">Personal API Telemetry</div></div></div>',
        unsafe_allow_html=True,
    )
    days = st.select_slider("History window (days)", [30, 60, 90, 180, 365], 365)
    ui_api_key = ""
    if not secret_api_key and not secret_access_token:
        ui_api_key = st.text_input(
            "Garmin Personal API Key",
            type="password",
            placeholder="GCDC_...",
            help="Generate at Garmin Connect Web → Account Settings → Personal API Key (or set in .streamlit/secrets.toml)",
        ).strip()
    ui_hevy_key = ""
    if not secret_hevy_key:
        ui_hevy_key = st.text_input(
            "Hevy API Key",
            type="password",
            placeholder="xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
            help="Generate at https://hevy.com/settings?developer (or set [hevy] api_key in .streamlit/secrets.toml)",
        ).strip()
    active_api_key = secret_api_key or ui_api_key
    active_hevy_key = secret_hevy_key or ui_hevy_key
    demo_mode = st.toggle(
        "Demo data",
        value=not bool(active_api_key or secret_access_token or has_legacy_secrets or active_hevy_key),
        help="Synthetic data for Garmin & Hevy, no API key needed.",
    )
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()

try:
    if demo_mode or not (active_api_key or secret_access_token or has_legacy_secrets):
        D = data.demo(days)
    elif active_api_key or secret_access_token:
        D = load_official_api(active_api_key, secret_access_token, days, CACHE_SCHEMA_VERSION)
        if "naps" not in D or ("sleep" in D and not D["sleep"].empty and "nap_min" not in D["sleep"].columns):
            st.cache_data.clear()
            D = load_official_api(active_api_key, secret_access_token, days, CACHE_SCHEMA_VERSION)
    elif has_legacy_secrets:
        D = load_legacy(cfg.get("email"), cfg.get("password"), cfg.get("tokens"), days, CACHE_SCHEMA_VERSION)
except Exception as e:
    st.error(f"Could not load Garmin data: {e}")
    st.info("Check your Personal API Key in `.streamlit/secrets.toml` (see `README.md`) or switch on Demo data in the sidebar.")
    st.stop()

hevy_error = None
try:
    if demo_mode:
        H = data.demo_hevy(days)
    elif active_hevy_key:
        H = load_hevy_api(active_hevy_key, days)
    else:
        H = {
            "workouts": pd.DataFrame(),
            "sets": pd.DataFrame(),
            "routines": pd.DataFrame(),
            "measurements": pd.DataFrame(),
            "user": {},
        }
except Exception as e:
    hevy_error = str(e)
    H = {
        "workouts": pd.DataFrame(),
        "sets": pd.DataFrame(),
        "routines": pd.DataFrame(),
        "measurements": pd.DataFrame(),
        "user": {},
    }

acts, sleep, hrv, daily, vo2, preds = (
    D[k] for k in ["activities", "sleep", "hrv", "daily", "vo2max", "predictions"]
)
naps = D.get("naps", pd.DataFrame())
all_acts = D.get("all_activities", acts)
hevy_workouts = H.get("workouts", pd.DataFrame())
hevy_sets = H.get("sets", pd.DataFrame())
hevy_routines = H.get("routines", pd.DataFrame())
hevy_measurements = H.get("measurements", pd.DataFrame())

body_state = D.get("body_state") or data.compute_body_state_now(acts, sleep, hrv, daily, hevy_workouts=hevy_workouts)
latest_vo2_val = float(vo2["vo2max"].iloc[-1]) if not vo2.empty else None
race_details = D.get("race_details")
if not race_details:
    preds, race_details = data.compute_race_predictions(
        latest_vo2_val, acts, all_acts=all_acts, daily_df=daily, body_state=body_state
    )

ts = D.get("training_status") or data._compute_training_status(acts, vo2, hrv, daily, days, sleep_df=sleep)
acts["day"] = pd.to_datetime(acts["date"]).dt.normalize()

with st.sidebar:
    st.divider()
    bs_col = body_state["state_color"]
    st.markdown(
        f'<div style="background:#12171f;border:1px solid {bs_col}44;border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
        f'  <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;">Body Readiness Now</span>'
        f'  <span style="font-size:0.72rem;font-weight:700;color:{bs_col};background:{bs_col}1f;padding:2px 8px;border-radius:999px;">{body_state["readiness_score"]}/100 · {body_state["state_badge"]}</span>'
        f'</div>'
        f'<div style="font-size:0.76rem;color:#c9d1d9;margin-top:4px;">'
        f'HRV <b>{body_state["hrv_now"] or 0:.0f} ms</b> · RHR <b>{body_state["rhr_now"] or 0:.0f} bpm</b> · BB <b>{body_state["bb_now"] or 0:.0f}</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="background:#12171f;border:1px solid {ts["color"]}44;border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
        f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px;">'
        f'  <span style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;">Training Status</span>'
        f'  <span style="font-size:0.72rem;font-weight:700;color:{ts["color"]};background:{ts["color"]}1f;padding:2px 8px;border-radius:999px;">{ts["status"]}</span>'
        f'</div>'
        f'<div style="font-size:0.78rem;color:#c9d1d9;margin-top:4px;">'
        f'7d Load: <b>{ts["acute_load"]}</b> <span style="color:#8b949e;">(Opt {ts["opt_low"]}–{ts["opt_high"]})</span> · ACWR <b>{ts["acwr"]:.2f}</b></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if not acts.empty:
        latest_act = acts.sort_values("date", ascending=False).iloc[0]
        st.markdown(
            f'<div style="background:#12171f;border:1px solid #21262d;border-radius:10px;padding:12px 14px;margin-bottom:10px;">'
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;margin-bottom:6px;">Latest Garmin Activity</div>'
            f'<div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#f0f6fc;font-size:0.9rem;">'
            f'{icon_html(sport_icon_name(latest_act.type, latest_act.category), size=16, badge=False, mr=2)}{latest_act["name"]}</div>'
            f'<div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">'
            f'{latest_act.distance_km:.2f} km · {fmt_time(latest_act.duration_min * 60)} · {pd.to_datetime(latest_act.date):%d %b %H:%M}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    if not hevy_workouts.empty:
        lw = hevy_workouts.iloc[0]
        st.markdown(
            f'<div style="background:#12171f;border:1px solid #e3b34144;border-radius:10px;padding:12px 14px;">'
            f'<div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;margin-bottom:6px;">Latest Hevy Lift</div>'
            f'<div style="display:flex;align-items:center;gap:8px;font-weight:700;color:#f0f6fc;font-size:0.9rem;">'
            f'{icon_html("strength", size=16, badge=False, mr=2)}{lw["title"]}</div>'
            f'<div style="font-size:0.78rem;color:#8b949e;margin-top:4px;">'
            f'{lw["volume_kg"]:,.0f} kg · {lw["working_sets"]} sets · {pd.to_datetime(lw["date"]):%d %b %H:%M}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------------------- Hero Banner
status_cls = "status-pill demo" if demo_mode else "status-pill"
if demo_mode:
    status_txt = "Demo Mode · Garmin + Hevy Synthetic Data"
elif active_hevy_key:
    status_txt = "Connected · Garmin + Hevy APIs"
else:
    status_txt = "Connected · Official Garmin Personal API"
hevy_sub = (
    f" · {len(hevy_workouts)} lifts ({hevy_workouts.volume_kg.sum() / 1000:,.1f}t lifted)"
    if not hevy_workouts.empty
    else ""
)
st.markdown(
    f'<div class="hero-bar">'
    f'  <div class="hero-left">'
    f'    {icon_html("optimalhealth", size=28, bg="#26a64122", accent="#39d353", badge=True, mr=4)}'
    f'    <div>'
    f'      <div class="hero-title">Personal Garmin & Hevy Dashboard</div>'
    f'      <div class="hero-sub">Last {days} days · {len(acts)} cardio workouts ({acts.distance_km.sum():,.1f} km){hevy_sub}</div>'
    f'    </div>'
    f'  </div>'
    f'  <div class="{status_cls}"><span class="status-dot"></span>{status_txt}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------- Daily Readiness Decision Popups ("Can I run today?" / "Can I lift today?")
def _render_decision_dialog_body(res: dict):
    v_col = res["color"]
    badge_map = {
        "green": ("#39d353", "rgba(57, 211, 83, 0.14)", "✓ OPTIMAL"),
        "amber": ("#e3b341", "rgba(227, 179, 65, 0.15)", "⚠ CAUTION"),
        "red": ("#f85149", "rgba(248, 81, 73, 0.15)", "✕ STRAIN"),
        "blue": ("#58a6ff", "rgba(88, 166, 255, 0.14)", "ℹ CONTEXT"),
    }
    st.markdown(
        f'<div style="background:linear-gradient(135deg, {v_col}1f 0%, #12171f 100%);'
        f'border:1px solid {v_col}66;border-left:5px solid {v_col};border-radius:12px;padding:16px 18px;margin-bottom:14px;">'
        f'  <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">'
        f'    <span style="font-size:0.74rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;">{res["day_name"]} · Live Telemetry Verdict</span>'
        f'    <span style="background:{v_col}28;color:{v_col};border:1px solid {v_col}66;padding:3px 10px;border-radius:999px;font-size:0.74rem;font-weight:800;">'
        f'      Readiness {res["readiness"]}/100'
        f'    </span>'
        f'  </div>'
        f'  <div style="font-size:1.35rem;font-weight:800;color:{v_col};margin:6px 0 4px 0;letter-spacing:-0.01em;">{res["verdict"]}</div>'
        f'  <div style="font-size:0.88rem;color:#e6edf3;line-height:1.45;">{res["summary"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Mini telemetry strip
    hrv_txt = f"{res['hrv_now']:.0f} ms" if res.get("hrv_now") is not None else "–"
    rhr_txt = f"{res['rhr_now']:.0f} bpm" if res.get("rhr_now") is not None else "–"
    bb_txt = f"{res['bb_now']:.0f}/100" if res.get("bb_now") is not None else "–"
    sl_txt = f"{res['sl_score']:.0f}/100" if res.get("sl_score") is not None else "–"
    acwr_txt = f"{res['acwr']:.2f}x" if res.get("acwr") is not None else "–"
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:14px;">'
        f'  <div style="background:#161b22;border:1px solid #262c36;border-radius:9px;padding:8px 10px;text-align:center;">'
        f'    <div style="font-size:0.66rem;color:#8b949e;text-transform:uppercase;">HRV Nightly</div>'
        f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:#f0f6fc;">{hrv_txt}</div>'
        f'  </div>'
        f'  <div style="background:#161b22;border:1px solid #262c36;border-radius:9px;padding:8px 10px;text-align:center;">'
        f'    <div style="font-size:0.66rem;color:#8b949e;text-transform:uppercase;">Resting HR</div>'
        f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:#f0f6fc;">{rhr_txt}</div>'
        f'  </div>'
        f'  <div style="background:#161b22;border:1px solid #262c36;border-radius:9px;padding:8px 10px;text-align:center;">'
        f'    <div style="font-size:0.66rem;color:#8b949e;text-transform:uppercase;">Body Battery</div>'
        f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:#f0f6fc;">{bb_txt}</div>'
        f'  </div>'
        f'  <div style="background:#161b22;border:1px solid #262c36;border-radius:9px;padding:8px 10px;text-align:center;">'
        f'    <div style="font-size:0.66rem;color:#8b949e;text-transform:uppercase;">Sleep Score</div>'
        f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:#f0f6fc;">{sl_txt}</div>'
        f'  </div>'
        f'  <div style="background:#161b22;border:1px solid #262c36;border-radius:9px;padding:8px 10px;text-align:center;">'
        f'    <div style="font-size:0.66rem;color:#8b949e;text-transform:uppercase;">Load ACWR</div>'
        f'    <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:#f0f6fc;">{acwr_txt}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div style="font-size:0.8rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#8b949e;margin-bottom:8px;">'
        'Why? Multi-Factor Physiological & Schedule Reasoning</div>',
        unsafe_allow_html=True,
    )
    for level, title, detail in res["reasons"]:
        fg, bg, pill_txt = badge_map.get(level, badge_map["blue"])
        st.markdown(
            f'<div style="background:#141921;border:1px solid #262c36;border-left:3px solid {fg};'
            f'border-radius:10px;padding:10px 13px;margin-bottom:8px;">'
            f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:3px;">'
            f'    <span style="font-weight:700;font-size:0.86rem;color:#f0f6fc;">{title}</span>'
            f'    <span style="background:{bg};color:{fg};padding:2px 8px;border-radius:999px;font-size:0.68rem;font-weight:700;">{pill_txt}</span>'
            f'  </div>'
            f'  <div style="font-size:0.81rem;color:#c9d1d9;line-height:1.42;">{detail}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:12px 14px;margin-top:10px;">'
        f'  <div style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#58a6ff;margin-bottom:4px;">'
        f'    🎯 Today\'s Recommended Prescription'
        f'  </div>'
        f'  <div style="font-size:0.86rem;font-weight:600;color:#f0f6fc;line-height:1.4;">{res["prescription"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


@st.dialog("🏃 Can I Run Today? — Live Readiness & 5K Plan Check", width="large")
def show_can_i_run_dialog():
    res = data.evaluate_daily_training_decision("run", body_state, ts, acts, hevy_workouts, hevy_sets)
    _render_decision_dialog_body(res)


@st.dialog("🏋️ Can I Lift Today? — Live Readiness & PPL Split Check", width="large")
def show_can_i_lift_dialog():
    res = data.evaluate_daily_training_decision("lift", body_state, ts, acts, hevy_workouts, hevy_sets)
    _render_decision_dialog_body(res)


btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("🏃 Can I run today?", key="btn_can_i_run", use_container_width=True, type="primary"):
        show_can_i_run_dialog()
with btn_col2:
    if st.button("🏋️ Can I lift today?", key="btn_can_i_lift", use_container_width=True, type="primary"):
        show_can_i_lift_dialog()



# --------------------------------------------------------------- GitHub-Style Heatmap
def contribution_graph(df, value_col, title, unit="min", palette="green", bins=None):
    colors = GH_SCALES.get(palette, GH_SCALES["green"])
    end = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=364)
    start -= pd.Timedelta(days=(start.weekday() + 1) % 7)  # start on Sunday
    full = pd.date_range(start, end)
    s = df.groupby("day")[value_col].sum().reindex(full, fill_value=0) if not df.empty else pd.Series(0, index=full)
    qs = [0] + list(np.quantile(s[s > 0], [0.25, 0.5, 0.75])) if (s > 0).any() else [0, 1, 2, 3]
    level = np.digitize(s.values, qs[1:] if bins is None else bins, right=True)
    level = np.where(s.values == 0, 0, np.clip(level + 1, 1, 4))
    wk = ((full - start).days // 7).values
    dow = ((full.weekday + 1) % 7).values
    z = np.full((7, wk.max() + 1), np.nan)
    txt = np.full(z.shape, "", dtype=object)
    for i, d in enumerate(full):
        z[dow[i], wk[i]] = level[i]
        txt[dow[i], wk[i]] = f"<b>{d:%a %d %b %Y}</b><br>{s.iloc[i]:.1f} {unit}"
    cs = [[i / 4, c] for i, c in enumerate(colors)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            text=txt,
            hoverinfo="text",
            colorscale=cs,
            zmin=0,
            zmax=4,
            showscale=False,
            xgap=3,
            ygap=3,
        )
    )
    fig.update_yaxes(
        autorange="reversed",
        tickvals=[1, 3, 5],
        ticktext=["Mon", "Wed", "Fri"],
        showgrid=False,
        zeroline=False,
    )
    ticks = {}
    for i, d in enumerate(full):
        if (d.day <= 7 and dow[i] == 0) or i == 0:
            ticks[wk[i]] = f"{d:%b}"
    fig.update_xaxes(tickvals=list(ticks), ticktext=list(ticks.values()), showgrid=False, zeroline=False, side="top")
    layout(fig, 205)
    fig.update_layout(title=title, margin=dict(l=12, r=12, t=58, b=10))
    st.plotly_chart(fig, width="stretch")
    active_days = int((s > 0).sum())
    swatches = "".join(f'<span class="gh-box" style="background:{c}"></span>' for c in colors)
    st.markdown(
        f'<div class="gh-footer">'
        f'  <span>Active on <b>{active_days}</b> days in the past 52 weeks · Total: <b>{s.sum():,.1f} {unit}</b></span>'
        f'  <span class="gh-legend">Less {swatches} More</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def streaks(df):
    d = sorted(set(df["day"].dropna().dt.date))
    if not d:
        return 0, 0
    best = cur = 1
    for a, b in zip(d, d[1:]):
        cur = cur + 1 if (b - a).days == 1 else 1
        best = max(best, cur)
    today = pd.Timestamp.today().date()
    last = d[-1]
    current = 0
    if (today - last).days <= 1:
        current = 1
        for a, b in zip(reversed(d[:-1]), reversed(d[1:])):
            if (b - a).days == 1:
                current += 1
            else:
                break
    return current, best


tabs = st.tabs([label for label, _ in TAB_SPECS])

# --------------------------------------------------------------- 0. Overview
with tabs[0]:
    cur, best = streaks(acts)
    c = st.columns(6)
    kpi_card(c[0], "Activities", f"{len(acts)}", f"{len(acts) / max(1, days / 7):.1f} / week", "multisport", "#58a6ff")
    kpi_card(c[1], "Total Distance", f"{acts.distance_km.sum():,.0f} km", f"{acts.distance_km.sum() / max(1, days / 7):.1f} km / wk", "running", "#f78166")
    kpi_card(c[2], "Training Time", f"{acts.duration_min.sum() / 60:,.1f} h", f"{acts.calories.sum():,.0f} kcal burned", "speed", "#39d353")
    kpi_card(c[3], "Current Streak", f"{cur} d", f"Best streak: {best} d", "insights", "#39d353")
    kpi_card(
        c[4],
        "Resting HR",
        f"{daily.resting_hr.dropna().iloc[-1]:.0f} bpm" if len(daily) and daily.resting_hr.notna().any() else "–",
        f"{daily.resting_hr.mean():.0f} bpm avg" if len(daily) and daily.resting_hr.notna().any() else "No RHR data",
        "heartbeat",
        "#f85149",
    )
    if len(sleep):
        sl_latest_ov = sleep.sort_values("date").iloc[-1]
        ov_sl_val = f"{sl_latest_ov.total_h:.1f} h"
        ov_sc_lat = f"Score {sl_latest_ov.score:.0f}" if pd.notna(sl_latest_ov.get("score")) else "Latest"
        ov_sc_avg = f"/{sleep.score.mean():.0f} avg" if sleep.score.notna().any() else ""
        ov_sl_sub = f"Avg {sleep.total_h.mean():.1f} h · {ov_sc_lat}{ov_sc_avg}"
    else:
        ov_sl_val, ov_sl_sub = "–", "No sleep data"
    kpi_card(
        c[5],
        "Sleep (Latest / Avg)",
        ov_sl_val,
        ov_sl_sub,
        "sleep",
        "#a371f7",
    )

    # Garmin Training Status & Live Somatic Readiness Banner
    ts_col = ts["color"]
    bs_col = body_state["state_color"]
    vo2_sign = f"{ts['vo2_delta_4w']:+.1f}" if ts.get("vo2_delta_4w") else "0.0"
    st.markdown(
        f'<div class="ts-banner" style="border-left: 4px solid {ts_col};">'
        f'  <div class="ts-left">'
        f'    {icon_html("optimalhealth", size=24, bg=f"{ts_col}22", accent=ts_col, badge=True, mr=2)}'
        f'    <div>'
        f'      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'        <span style="font-size:0.76rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;">Training Status</span>'
        f'        <span class="ts-badge" style="background:{ts_col}22;color:{ts_col};border:1px solid {ts_col}55;">'
        f'          <span class="status-dot"></span>{ts["status"]}'
        f'        </span>'
        f'        <span class="ts-badge" style="background:{bs_col}22;color:{bs_col};border:1px solid {bs_col}55;">'
        f'          Readiness {body_state["readiness_score"]}/100 · {body_state["state_badge"]}'
        f'        </span>'
        f'      </div>'
        f'      <div class="ts-headline">{ts["headline"]}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="ts-metrics">'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Body Readiness</div>'
        f'      <div class="ts-chip-val" style="color:{bs_col};">{body_state["readiness_score"]}/100</div>'
        f'      <div class="ts-chip-sub">BB {body_state["bb_now"] or 0:.0f} · Sleep {body_state["sl_score_now"] or 0:.0f}</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">7d Acute Load</div>'
        f'      <div class="ts-chip-val">{ts["acute_load"]} <span style="font-size:0.75rem;color:{ts_col};">({ts["load_status"]})</span></div>'
        f'      <div class="ts-chip-sub">Optimal: {ts["opt_low"]}–{ts["opt_high"]}</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Load Ratio (ACWR)</div>'
        f'      <div class="ts-chip-val">{ts["acwr"]:.2f}x</div>'
        f'      <div class="ts-chip-sub">28d Chronic: {ts["chronic_load"]}</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">HRV Status</div>'
        f'      <div class="ts-chip-val">{ts["hrv_status"]}</div>'
        f'      <div class="ts-chip-sub">RHR {body_state["rhr_now"] or 0:.0f} bpm (base {body_state["rhr_base"]:.0f})</div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    section_header("Training Consistency", f"{len(acts)} workouts across all sports", "overview", "#39d353")
    contribution_graph(acts, "duration_min", "Daily Training Volume (minutes)", unit="min", palette="green")

    if not acts.empty:
        l, r = st.columns([1.45, 1])
        wk = acts.set_index("day").groupby([pd.Grouper(freq="W"), "category"])["distance_km"].sum().reset_index()
        l.plotly_chart(
            layout(
                px.bar(
                    wk,
                    x="day",
                    y="distance_km",
                    color="category",
                    color_discrete_map=COLORS,
                    title="Weekly Distance by Sport (km)",
                )
            ),
            width="stretch",
        )
        mix = acts.groupby("category")["duration_min"].sum().reset_index()
        r.plotly_chart(
            layout(
                px.pie(
                    mix,
                    names="category",
                    values="duration_min",
                    hole=0.62,
                    color="category",
                    color_discrete_map=COLORS,
                    title="Training Time Breakdown",
                )
            ),
            width="stretch",
        )

    if len(daily) and daily.steps.notna().any():
        section_header("Daily Movement & Steps", "With 7-day rolling trend and 10,000-step goal", "walking", "#39d353")
        d_steps = daily.dropna(subset=["steps"]).copy()
        d_steps["roll7"] = d_steps["steps"].rolling(7, min_periods=1).mean()
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=d_steps["date"],
                y=d_steps["steps"],
                fill="tozeroy",
                mode="lines",
                name="Daily Steps",
                line=dict(color="rgba(57, 211, 83, 0.45)", width=1.5),
                fillcolor="rgba(57, 211, 83, 0.12)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d_steps["date"],
                y=d_steps["roll7"],
                mode="lines",
                name="7-day avg",
                line=dict(color="#39d353", width=2.5),
            )
        )
        fig.add_hline(y=10000, line_dash="dash", line_color="#8b949e", annotation_text="10k Goal")
        fig.update_layout(title=f"Daily Steps (Avg: {d_steps['steps'].mean():,.0f} / day)")
        st.plotly_chart(layout(fig, 300), width="stretch")


# --------------------------------------------------------------- 1-3. Sport Tabs
def sport_tab(tab, cat, icon_key, palette, is_run=False):
    with tab:
        df = acts[acts.category == cat].copy()
        accent = COLORS[cat]
        if df.empty:
            st.info(f"No {cat.lower()} activities recorded in the last {days} days.")
            return

        c = st.columns(6)
        kpi_card(c[0], f"{cat} Sessions", f"{len(df)}", f"{len(df) / max(1, days / 7):.1f} / week", icon_key, accent)
        kpi_card(c[1], "Total Distance", f"{df.distance_km.sum():,.1f} km", f"Avg {df.distance_km.mean():.1f} km / session", "insights", accent)
        kpi_card(c[2], "Longest Session", f"{df.distance_km.max():.1f} km", f"{fmt_time(df.duration_min.max() * 60)} max duration", "plan", accent)
        kpi_card(
            c[3],
            "Avg Pace" if is_run else "Avg Speed",
            pace(df.avg_speed_kmh.mean()) if is_run else f"{df.avg_speed_kmh.mean():.1f} km/h",
            f"Best: {pace(df.avg_speed_kmh.max())}" if is_run else f"Max: {df.avg_speed_kmh.max():.1f} km/h",
            "speed",
            accent,
        )
        kpi_card(
            c[4],
            "Avg Heart Rate",
            f"{df.avg_hr.mean():.0f} bpm" if df.avg_hr.notna().any() else "–",
            f"Peak {df.max_hr.max():.0f} bpm" if df.max_hr.notna().any() else "No HR sensor",
            "heartbeat",
            "#f85149",
        )
        kpi_card(
            c[5],
            "Elevation / Kcal",
            f"{df.elevation_m.sum():,.0f} m",
            f"{df.calories.sum():,.0f} kcal total",
            "calories",
            "#e3b341",
        )

        section_header(f"{cat} Consistency", "Daily distance contribution heatmap", icon_key, accent)
        contribution_graph(df, "distance_km", f"{cat} — Daily Distance (km)", unit="km", palette=palette)

        l, r = st.columns(2)
        wk = df.set_index("day").resample("W")["distance_km"].sum().reset_index()
        wk["roll4"] = wk["distance_km"].rolling(4, min_periods=1).mean()
        wfig = go.Figure()
        wfig.add_trace(go.Bar(x=wk["day"], y=wk["distance_km"], name="Weekly km", marker_color=accent, opacity=0.82))
        wfig.add_trace(go.Scatter(x=wk["day"], y=wk["roll4"], name="4-wk avg", line=dict(color="#f0f6fc", width=2.2)))
        wfig.update_layout(title=f"Weekly {cat} Volume (km)")
        l.plotly_chart(layout(wfig), width="stretch")

        sfig = px.scatter(
            df.assign(_sz=df["distance_km"].clip(lower=0.2)),
            x="date",
            y="avg_speed_kmh",
            size="_sz",
            color="avg_hr",
            color_continuous_scale="Turbo",
            hover_name="name",
            hover_data={"_sz": False, "distance_km": ":.2f", "duration_min": ":.1f", "avg_hr": ":.0f"},
            title="Session Speed vs. Heart Rate (bubble = distance)",
        )
        r.plotly_chart(layout(sfig), width="stretch")

        section_header(f"{cat} Session Table", f"{len(df)} workouts", "plan", accent)
        view = df.sort_values("date", ascending=False)[
            ["date", "name", "distance_km", "duration_min", "avg_speed_kmh", "avg_hr", "max_hr", "elevation_m", "calories", "training_effect"]
        ].copy()
        if is_run:
            view["pace"] = view["avg_speed_kmh"].map(pace)
        st.dataframe(view.round(2), width="stretch", hide_index=True)


sport_tab(tabs[1], "Run", "running", "orange", is_run=True)
sport_tab(tabs[2], "Cycling", "bike", "blue", is_run=False)
sport_tab(tabs[3], "Walk", "walking", "green", is_run=False)

# --------------------------------------------------------------- 4. Weightlifting (Hevy)
with tabs[4]:
    accent = "#e3b341"
    if hevy_error:
        st.error(f"Could not load Hevy data: {hevy_error}")
    if hevy_workouts.empty:
        st.info(
            "Enter your **Hevy API Key** in the sidebar (or set `[hevy] api_key = \"...\"` in `.streamlit/secrets.toml`) "
            "or enable **Demo data** to explore your weightlifting telemetry."
        )
    else:
        working_sets_df = hevy_sets[hevy_sets["set_type"] != "warmup"].copy() if not hevy_sets.empty else pd.DataFrame()
        tot_tonnage = float(hevy_workouts["volume_kg"].sum())
        tot_sets = int(hevy_workouts["working_sets"].sum())
        tot_reps = int(hevy_workouts["total_reps"].sum())
        avg_dur = float(hevy_workouts["duration_min"].mean()) if len(hevy_workouts) else 0.0
        unique_ex = int(hevy_sets["exercise_title"].nunique()) if not hevy_sets.empty else 0
        avg_rpe = float(working_sets_df["rpe"].dropna().mean()) if not working_sets_df.empty and working_sets_df["rpe"].notna().any() else None

        c = st.columns(6)
        kpi_card(
            c[0],
            "Lifting Sessions",
            f"{len(hevy_workouts)}",
            f"{len(hevy_workouts) / max(1, days / 7):.1f} / week",
            "strength",
            accent,
        )
        kpi_card(
            c[1],
            "Total Tonnage",
            f"{tot_tonnage / 1000:,.1f} t" if tot_tonnage >= 10000 else f"{tot_tonnage:,.0f} kg",
            f"Avg {tot_tonnage / max(1, len(hevy_workouts)):,.0f} kg / workout",
            "calories",
            accent,
        )
        kpi_card(
            c[2],
            "Working Sets",
            f"{tot_sets:,}",
            f"Avg {tot_sets / max(1, len(hevy_workouts)):.1f} sets / session",
            "plan",
            "#58a6ff",
        )
        kpi_card(
            c[3],
            "Total Reps",
            f"{tot_reps:,}",
            f"Avg {tot_reps / max(1, tot_sets):.1f} reps / set",
            "cadence",
            "#39d353",
        )
        kpi_card(
            c[4],
            "Session Duration",
            f"{avg_dur:.0f} min",
            f"{hevy_workouts['duration_min'].sum() / 60:,.1f} h total gym time",
            "speed",
            "#a371f7",
        )
        kpi_card(
            c[5],
            "Exercises & Effort",
            f"{unique_ex} lifts",
            f"Avg RPE {avg_rpe:.1f} / 10" if avg_rpe else "Tracked via Hevy API",
            "insights",
            "#f78166",
        )

        section_header("Weightlifting Consistency", "Daily volume lifted (kg) across the past 52 weeks", "strength", accent)
        contribution_graph(hevy_workouts, "volume_kg", "Hevy — Daily Volume Lifted (kg)", unit="kg", palette="amber")

        # Weekly Tonnage + Muscle Group Breakdown
        l, r = st.columns([1.35, 1])
        wk_lift = hevy_workouts.set_index("day").resample("W")["volume_kg"].sum().reset_index()
        wk_lift["roll4"] = wk_lift["volume_kg"].rolling(4, min_periods=1).mean()
        wfig = go.Figure()
        wfig.add_trace(go.Bar(x=wk_lift["day"], y=wk_lift["volume_kg"], name="Weekly Volume (kg)", marker_color=accent, opacity=0.84))
        wfig.add_trace(go.Scatter(x=wk_lift["day"], y=wk_lift["roll4"], name="4-wk avg", line=dict(color="#f0f6fc", width=2.3)))
        wfig.update_layout(title="Weekly Tonnage Lifted (kg)")
        l.plotly_chart(layout(wfig, 340), width="stretch")

        if not working_sets_df.empty:
            muscle_grp = (
                working_sets_df.groupby("muscle_group", as_index=False)
                .agg(working_sets=("set_index", "count"), volume_kg=("volume_kg", "sum"))
                .sort_values("working_sets", ascending=True)
            )
            mfig = px.bar(
                muscle_grp,
                x="working_sets",
                y="muscle_group",
                orientation="h",
                text="working_sets",
                color="volume_kg",
                color_continuous_scale=["#3d2a0d", "#a87419", "#e3b341", "#f78166"],
                title="Working Sets by Muscle Group",
            )
            mfig.update_traces(textposition="outside")
            mfig.update_xaxes(range=[0, max(muscle_grp["working_sets"].max() * 1.22, 5)])
            mfig.update_layout(coloraxis_showscale=False)
            r.plotly_chart(layout(mfig, 340), width="stretch")

        # Exercise Progression & 1RM Tracker
        if not working_sets_df.empty:
            section_header(
                "Exercise Progression & Estimated 1RM",
                "Track top weight, Epley 1RM, and volume overload for any exercise",
                "insights",
                accent,
            )
            ex_counts = (
                working_sets_df.groupby("exercise_title")
                .agg(sessions=("workout_id", "nunique"), sets=("set_index", "count"), max_wt=("weight_kg", "max"))
                .sort_values(["sessions", "sets"], ascending=False)
            )
            ex_options = list(ex_counts.index)
            selected_ex = st.selectbox(
                "Select Exercise",
                ex_options,
                index=0,
                format_func=lambda x: f"{x} ({ex_counts.loc[x, 'sessions']} sessions · max {ex_counts.loc[x, 'max_wt']:.1f} kg)",
            )
            ex_df = working_sets_df[working_sets_df["exercise_title"] == selected_ex].copy()

            if not ex_df.empty:
                best_wt_row = ex_df.sort_values(["weight_kg", "reps"], ascending=False).iloc[0]
                best_1rm_row = ex_df.sort_values("e1rm_kg", ascending=False).iloc[0]
                best_vol_row = ex_df.sort_values("volume_kg", ascending=False).iloc[0]

                ec = st.columns(4)
                kpi_card(
                    ec[0],
                    "Heaviest Weight (PR)",
                    f"{best_wt_row['weight_kg']:.1f} kg",
                    f"{int(best_wt_row['reps'])} reps on {pd.to_datetime(best_wt_row['date']):%d %b %Y}",
                    "strength",
                    accent,
                )
                kpi_card(
                    ec[1],
                    "Best Est. 1RM (Epley)",
                    f"{best_1rm_row['e1rm_kg']:.1f} kg",
                    f"From {best_1rm_row['weight_kg']:.1f} kg × {int(best_1rm_row['reps'])}",
                    "race_predictor",
                    "#f78166",
                )
                kpi_card(
                    ec[2],
                    "Best Single-Set Volume",
                    f"{best_vol_row['volume_kg']:,.0f} kg",
                    f"{best_vol_row['weight_kg']:.1f} kg × {int(best_vol_row['reps'])} reps",
                    "calories",
                    "#39d353",
                )
                kpi_card(
                    ec[3],
                    "Logged Working Sets",
                    f"{len(ex_df)}",
                    f"Across {ex_df['workout_id'].nunique()} workouts ({ex_df['muscle_group'].iloc[0]})",
                    "plan",
                    "#58a6ff",
                )

                ex_session = (
                    ex_df.groupby("day", as_index=False)
                    .agg(
                        top_weight_kg=("weight_kg", "max"),
                        best_e1rm_kg=("e1rm_kg", "max"),
                        session_volume_kg=("volume_kg", "sum"),
                        total_reps=("reps", "sum"),
                        sets=("set_index", "count"),
                    )
                    .sort_values("day")
                )

                p_left, p_right = st.columns(2)
                efig = go.Figure()
                efig.add_trace(
                    go.Scatter(
                        x=ex_session["day"],
                        y=ex_session["best_e1rm_kg"],
                        mode="lines+markers",
                        name="Estimated 1RM (kg)",
                        line=dict(color="#f78166", width=2.5),
                        marker=dict(size=6, color="#f78166"),
                    )
                )
                efig.add_trace(
                    go.Scatter(
                        x=ex_session["day"],
                        y=ex_session["top_weight_kg"],
                        mode="lines+markers",
                        name="Top Set Weight (kg)",
                        line=dict(color=accent, width=2.4),
                        marker=dict(size=6, color=accent),
                    )
                )
                efig.update_layout(title=f"{selected_ex} — Strength & e1RM Progression")
                p_left.plotly_chart(layout(efig, 320), width="stretch")

                vfig = go.Figure()
                vfig.add_trace(
                    go.Bar(
                        x=ex_session["day"],
                        y=ex_session["session_volume_kg"],
                        name="Session Volume (kg)",
                        marker_color="#58a6ff",
                        opacity=0.85,
                    )
                )
                vfig.update_layout(title=f"{selected_ex} — Volume per Session (kg)")
                p_right.plotly_chart(layout(vfig, 320), width="stretch")

            # Personal Records Leaderboard across exercises
            section_header("Personal Records Leaderboard", "All-time best lifts in the selected window", "race_predictor", accent)
            pr_rows = []
            for lift_name, grp in working_sets_df.groupby("exercise_title"):
                if grp["weight_kg"].max() <= 0:
                    continue
                top_w = grp.sort_values(["weight_kg", "reps"], ascending=False).iloc[0]
                top_1rm = grp.sort_values("e1rm_kg", ascending=False).iloc[0]
                pr_rows.append({
                    "Exercise": lift_name,
                    "Muscle Group": grp["muscle_group"].iloc[0],
                    "Sessions": grp["workout_id"].nunique(),
                    "Working Sets": len(grp),
                    "Heaviest Weight (kg)": round(float(top_w["weight_kg"]), 1),
                    "Top Set": f"{top_w['weight_kg']:.1f} kg × {int(top_w['reps'])}",
                    "Est. 1RM (kg)": round(float(top_1rm["e1rm_kg"]), 1),
                    "Total Volume (kg)": round(float(grp["volume_kg"].sum()), 0),
                    "PR Date": pd.to_datetime(top_1rm["date"]).strftime("%d %b %Y"),
                })
            if pr_rows:
                pr_df = pd.DataFrame(pr_rows).sort_values(["Sessions", "Est. 1RM (kg)"], ascending=False).reset_index(drop=True)
                st.dataframe(pr_df, width="stretch", hide_index=True)

        # Saved Hevy Routines (if present)
        if not hevy_routines.empty:
            section_header("Saved Hevy Routines", f"{len(hevy_routines)} active templates", "plan", "#58a6ff")
            r_cols = st.columns(min(3, len(hevy_routines)))
            for idx_r, (_, rt) in enumerate(hevy_routines.head(6).iterrows()):
                with r_cols[idx_r % len(r_cols)]:
                    st.markdown(
                        f'<div class="kpi-card">'
                        f'  <div class="kpi-top"><span class="kpi-label">{rt["exercises_count"]} EXERCISES · {rt["total_sets"]} SETS</span>'
                        f'{icon_html("strength", size=16, bg="#e3b3411a", accent=accent, badge=True, mr=0)}</div>'
                        f'  <div style="font-weight:700;font-size:1.02rem;color:#f0f6fc;margin-bottom:6px;">{rt["title"]}</div>'
                        f'  <div style="font-size:0.78rem;color:#8b949e;line-height:1.4;">{rt["exercises_summary"]}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Recent Hevy Workouts & Set-by-Set Log
        section_header("Recent Hevy Workouts", f"Showing latest {min(25, len(hevy_workouts))} sessions with set-by-set breakdown", "strength", accent)
        for idx_w, w in hevy_workouts.head(25).iterrows():
            w_date_str = pd.to_datetime(w["date"]).strftime("%a %d %b %Y · %H:%M")
            rpe_txt = f" · Avg RPE {w['avg_rpe']:.1f}" if pd.notna(w.get("avg_rpe")) else ""
            header_label = (
                f"🏋️ {w['title']} — {w_date_str}  ·  "
                f"{w['volume_kg']:,.0f} kg  ·  {w['working_sets']} sets ({w['total_reps']} reps)  ·  "
                f"{w['duration_min']:.0f} min  ·  {w['primary_muscles']}{rpe_txt}"
            )
            with st.expander(header_label, expanded=(idx_w == 0)):
                w_sets = hevy_sets[hevy_sets["workout_id"] == w["id"]].copy()
                if not w_sets.empty:
                    disp = w_sets[[
                        "exercise_title", "muscle_group", "set_index", "set_type",
                        "weight_kg", "reps", "volume_kg", "e1rm_kg", "rpe",
                    ]].rename(columns={
                        "exercise_title": "Exercise",
                        "muscle_group": "Muscle",
                        "set_index": "Set #",
                        "set_type": "Type",
                        "weight_kg": "Weight (kg)",
                        "reps": "Reps",
                        "volume_kg": "Volume (kg)",
                        "e1rm_kg": "Est. 1RM (kg)",
                        "rpe": "RPE",
                    })
                    st.dataframe(disp.round(1), width="stretch", hide_index=True)

# --------------------------------------------------------------- 5. Sleep
with tabs[5]:
    if sleep.empty:
        st.info("No sleep data available for this period.")
    else:
        sl_sorted = sleep.sort_values("date").reset_index(drop=True)
        latest_sl = sl_sorted.iloc[-1]
        lat_date_str = pd.to_datetime(latest_sl["date"]).strftime("%d %b %Y")
        lat_tot_h = float(latest_sl.get("total_h") or 0.0)
        lat_h_int, lat_m_int = divmod(int(round(lat_tot_h * 60)), 60)
        avg_tot_h = float(sleep["total_h"].mean())
        avg_h_int, avg_m_int = divmod(int(round(avg_tot_h * 60)), 60)
        diff_h = lat_tot_h - avg_tot_h

        lat_score = float(latest_sl["score"]) if pd.notna(latest_sl.get("score")) else None
        avg_score = float(sleep["score"].mean()) if sleep["score"].notna().any() else None
        best_score = float(sleep["score"].max()) if sleep["score"].notna().any() else None
        min_score = float(sleep["score"].min()) if sleep["score"].notna().any() else None

        lat_deep_m = float(latest_sl.get("deep_h") or 0.0) * 60.0
        lat_deep_pct = (float(latest_sl.get("deep_h") or 0.0) / max(0.1, lat_tot_h)) * 100.0
        avg_deep_m = float(sleep["deep_h"].mean()) * 60.0
        avg_deep_pct = (float(sleep["deep_h"].mean()) / max(0.1, avg_tot_h)) * 100.0

        lat_rem_m = float(latest_sl.get("rem_h") or 0.0) * 60.0
        lat_rem_pct = (float(latest_sl.get("rem_h") or 0.0) / max(0.1, lat_tot_h)) * 100.0
        avg_rem_m = float(sleep["rem_h"].mean()) * 60.0
        avg_rem_pct = (float(sleep["rem_h"].mean()) / max(0.1, avg_tot_h)) * 100.0

        lat_light_h = float(latest_sl.get("light_h") or 0.0)
        lat_awake_m = float(latest_sl.get("awake_h") or 0.0) * 60.0
        avg_light_h = float(sleep["light_h"].mean())
        avg_awake_m = float(sleep["awake_h"].mean()) * 60.0

        lat_stress = float(latest_sl["sleep_stress"]) if "sleep_stress" in latest_sl and pd.notna(latest_sl.get("sleep_stress")) else None
        avg_stress = float(sleep["sleep_stress"].dropna().mean()) if "sleep_stress" in sleep.columns and sleep["sleep_stress"].notna().any() else None
        lat_sl_hr = float(latest_sl["sleep_hr"]) if "sleep_hr" in latest_sl and pd.notna(latest_sl.get("sleep_hr")) else None
        avg_sl_hr = float(sleep["sleep_hr"].dropna().mean()) if "sleep_hr" in sleep.columns and sleep["sleep_hr"].notna().any() else None

        tot_naps_cnt = int(len(naps)) if isinstance(naps, pd.DataFrame) and not naps.empty else int(sleep["nap_count"].sum() if "nap_count" in sleep.columns else 0)
        tot_nap_m = float(naps["duration_min"].sum()) if isinstance(naps, pd.DataFrame) and not naps.empty else float(sleep["nap_min"].sum() if "nap_min" in sleep.columns else 0.0)
        lat_nap_m = float(latest_sl.get("nap_min") or 0.0) if "nap_min" in latest_sl and pd.notna(latest_sl.get("nap_min")) else 0.0

        section_header(
            f"Latest Night vs. {days}-Day Sleep Baseline ({lat_date_str})",
            "Every card displays both your latest recorded night and your period average side by side",
            "sleep",
            "#a371f7",
        )

        # Row 1: Latest Night Headline Metrics (with Average comparison in subtitle)
        c_lat = st.columns(6)
        kpi_card(
            c_lat[0],
            f"Last Night ({lat_date_str[:6]})",
            f"{lat_tot_h:.2f} h ({lat_h_int}h {lat_m_int:02d}m)",
            f"Avg: {avg_tot_h:.2f} h ({diff_h:+.2f} h)" + (f" · +{lat_nap_m:.0f}m nap" if lat_nap_m > 0 else ""),
            "sleep",
            "#39d353" if lat_tot_h >= 7.2 else ("#e3b341" if lat_tot_h >= 6.2 else "#f85149"),
        )
        kpi_card(
            c_lat[1],
            "Sleep Score (Latest / Avg)",
            f"{lat_score:.0f} / 100" if lat_score is not None else "–",
            f"Avg: {avg_score:.0f}/100 · Best: {best_score:.0f} · Low: {min_score:.0f}" if avg_score is not None else "Score not reported",
            "optimalhealth",
            "#39d353" if (lat_score or 0) >= 75 else ("#e3b341" if (lat_score or 0) >= 60 else "#f85149"),
        )
        kpi_card(
            c_lat[2],
            "Deep Sleep (Latest / Avg)",
            f"{lat_deep_m:.0f} min ({lat_deep_pct:.0f}%)",
            f"Avg: {avg_deep_m:.0f} min ({avg_deep_pct:.0f}% of total)",
            "recovery",
            "#1f6feb",
        )
        kpi_card(
            c_lat[3],
            "REM Sleep (Latest / Avg)",
            f"{lat_rem_m:.0f} min ({lat_rem_pct:.0f}%)",
            f"Avg: {avg_rem_m:.0f} min ({avg_rem_pct:.0f}% of total)",
            "insights",
            "#a371f7",
        )
        kpi_card(
            c_lat[4],
            "Light & Awake (Latest / Avg)",
            f"{lat_light_h:.1f} h · {lat_awake_m:.0f}m awake",
            f"Avg: {avg_light_h:.1f} h · {avg_awake_m:.0f}m awake",
            "curious",
            "#58a6ff",
        )
        kpi_card(
            c_lat[5],
            "Sleep Stress & HR (Latest / Avg)",
            f"Stress {lat_stress:.0f} · {lat_sl_hr:.0f} bpm" if (lat_stress is not None and lat_sl_hr is not None) else (f"Stress {lat_stress:.0f}" if lat_stress is not None else "–"),
            f"Avg: Stress {avg_stress:.0f} · {avg_sl_hr:.0f} bpm" if (avg_stress is not None and avg_sl_hr is not None) else "Overnight autonomic load",
            "heartbeat",
            "#f85149" if (lat_stress or 0) >= 25 else "#39d353",
        )

        # Row 2: Period Averages & Totals (with Latest comparison in subtitle)
        c = st.columns(6)
        kpi_card(
            c[0],
            f"Avg Sleep ({len(sleep)}d)",
            f"{avg_tot_h:.2f} h ({avg_h_int}h {avg_m_int:02d}m)",
            f"Latest: {lat_tot_h:.2f} h · Max: {sleep.total_h.max():.1f} h · Min: {sleep.total_h.min():.1f} h",
            "sleep",
            "#a371f7",
        )
        kpi_card(
            c[1],
            "Avg Sleep Score",
            f"{avg_score:.0f} / 100" if avg_score is not None else "–",
            f"Latest: {lat_score:.0f}/100 · Best: {best_score:.0f}/100" if (avg_score is not None and lat_score is not None) else "Score not reported",
            "optimalhealth",
            "#39d353",
        )
        kpi_card(
            c[2],
            "Avg Deep Sleep",
            f"{avg_deep_m:.0f} min ({avg_deep_pct:.0f}%)",
            f"Latest: {lat_deep_m:.0f} min ({lat_deep_pct:.0f}%)",
            "recovery",
            "#1f6feb",
        )
        kpi_card(
            c[3],
            "Avg REM Sleep",
            f"{avg_rem_m:.0f} min ({avg_rem_pct:.0f}%)",
            f"Latest: {lat_rem_m:.0f} min ({lat_rem_pct:.0f}%)",
            "insights",
            "#a371f7",
        )
        kpi_card(
            c[4],
            "Avg Light / Awake",
            f"{avg_light_h:.1f} h · {avg_awake_m:.0f}m awake",
            f"Latest: {lat_light_h:.1f} h · {lat_awake_m:.0f}m awake",
            "curious",
            "#58a6ff",
        )
        kpi_card(
            c[5],
            "Daytime Naps",
            f"{tot_naps_cnt} naps ({tot_nap_m:.0f}m)" if tot_naps_cnt > 0 else "0 naps",
            f"Latest: {naps.iloc[0]['duration_min']:.0f}m ({pd.to_datetime(naps.iloc[0]['date']):%d %b}) · Avg {tot_nap_m / tot_naps_cnt:.0f}m" if (isinstance(naps, pd.DataFrame) and not naps.empty) else "No naps in selected window",
            "sleep",
            "#e3b341" if tot_naps_cnt > 0 else "#8b949e",
        )

        section_header("Sleep Consistency", "Nightly sleep duration across the past year", "sleep", "#a371f7")
        sdf = sleep.assign(day=sleep.date, hours=sleep.total_h)
        contribution_graph(sdf, "hours", "Nightly Sleep Duration (hours)", unit="h", palette="purple", bins=[6.0, 7.0, 8.0])

        long = sleep.melt("date", ["deep_h", "rem_h", "light_h", "awake_h"], "stage", "hours")
        long["stage"] = long.stage.map({"deep_h": "Deep", "rem_h": "REM", "light_h": "Light", "awake_h": "Awake"})
        fig = px.bar(
            long,
            x="date",
            y="hours",
            color="stage",
            title="Nightly Sleep Architecture (hours)",
            color_discrete_map={"Deep": "#1f6feb", "REM": "#a371f7", "Light": "#58a6ff", "Awake": "#f85149"},
        )
        st.plotly_chart(layout(fig, 360), width="stretch")

        if sleep.score.notna().any():
            sc = sleep.dropna(subset=["score"]).copy()
            sc["roll7"] = sc["score"].rolling(7, min_periods=1).mean()
            sfig = go.Figure()
            sfig.add_trace(go.Scatter(x=sc["date"], y=sc["score"], mode="markers+lines", name="Nightly Score", line=dict(color="rgba(163,113,247,0.4)", width=1.5), marker=dict(size=5, color="#a371f7")))
            sfig.add_trace(go.Scatter(x=sc["date"], y=sc["roll7"], mode="lines", name="7-day avg", line=dict(color="#a371f7", width=2.8)))
            sfig.update_layout(title="Garmin Sleep Score Trend")
            st.plotly_chart(layout(sfig, 300), width="stretch")

        # ------------------------------------------------------- Naps & Garmin Sleep Coach Section
        section_header(
            "Naps & Daytime Recovery (Garmin Nap Detection & Sleep Coach)",
            "Recorded daytime naps, Body Battery recharge impact, and Garmin Sleep Coach need adjustments",
            "sleep",
            "#e3b341",
        )
        latest_sl = sleep.sort_values("date").iloc[-1]
        next_need_m = float(latest_sl.get("next_sleep_need_min") or latest_sl.get("sleep_need_min") or 480.0) if pd.notna(latest_sl.get("next_sleep_need_min") or latest_sl.get("sleep_need_min")) else 480.0
        base_need_m = float(latest_sl.get("baseline_need_min") or 480.0) if pd.notna(latest_sl.get("baseline_need_min")) else 480.0
        nap_adj_str = str(latest_sl.get("nap_adjustment") or "NO_CHANGE").replace("_", " ").title()
        nh_h, nh_m = divmod(int(round(next_need_m)), 60)
        bh_h, bh_m = divmod(int(round(base_need_m)), 60)

        nc = st.columns(4)
        if isinstance(naps, pd.DataFrame) and not naps.empty:
            latest_nap = naps.sort_values("start_local", ascending=False).iloc[0]
            ln_date = pd.to_datetime(latest_nap["date"]).strftime("%d %b %Y")
            ln_window = (
                f"{pd.to_datetime(latest_nap['start_local']):%H:%M}–{pd.to_datetime(latest_nap['end_local']):%H:%M}"
                if pd.notna(latest_nap.get("start_local")) and pd.notna(latest_nap.get("end_local"))
                else ln_date
            )
            ln_bb = latest_nap.get("bb_impact")
            ln_bb_str = f"{int(ln_bb):+d} Body Battery" if pd.notna(ln_bb) else "Body Battery tracked"
            bb_vals = naps["bb_impact"].dropna() if "bb_impact" in naps.columns else pd.Series(dtype=float)
            net_bb = int(bb_vals.sum()) if not bb_vals.empty else 0

            kpi_card(
                nc[0],
                "Latest Nap",
                f"{latest_nap['duration_min']:.0f} min",
                f"{ln_date} ({ln_window}) · {latest_nap.get('source', 'Watch')}",
                "sleep",
                "#e3b341",
            )
            kpi_card(
                nc[1],
                "Latest Nap Assessment",
                "Ideal Power Nap" if "Ideal" in str(latest_nap.get("feedback", "")) else "Extended Nap",
                str(latest_nap.get("feedback") or "Recorded nap"),
                "optimalhealth",
                "#39d353" if "Ideal" in str(latest_nap.get("feedback", "")) else "#e3b341",
            )
            kpi_card(
                nc[2],
                "Nap Body Battery Impact",
                f"{int(ln_bb):+d} pts" if pd.notna(ln_bb) else f"{net_bb:+d} pts",
                f"Latest nap ({ln_date}) · Window net {net_bb:+d} BB",
                "calories",
                "#39d353" if (pd.notna(ln_bb) and ln_bb > 0) else "#58a6ff",
            )
        else:
            kpi_card(nc[0], "Latest Nap", "None", "No daytime naps logged in this range", "sleep", "#8b949e")
            kpi_card(nc[1], "Nap Guidance", "15–25 min", "Ideal power nap window before 15:00 local", "optimalhealth", "#39d353")
            kpi_card(nc[2], "Nap Body Battery", "0 pts", "Naps recharge Body Battery & reduce sleep debt", "calories", "#58a6ff")

        kpi_card(
            nc[3],
            "Tonight's Sleep Coach Need",
            f"{nh_h}h {nh_m:02d}m",
            f"Baseline {bh_h}h {bh_m:02d}m · Nap adj: {nap_adj_str}",
            "recovery",
            "#f85149" if next_need_m >= 520 else ("#e3b341" if next_need_m > base_need_m else "#39d353"),
        )

        nl_col, nr_col = st.columns([1.15, 1])
        with nl_col:
            sl_plot = sleep.sort_values("date").copy()
            sl_plot["nap_h"] = (sl_plot["nap_min"] / 60.0) if "nap_min" in sl_plot.columns else 0.0
            sl_plot["need_h"] = (sl_plot["sleep_need_min"] / 60.0) if "sleep_need_min" in sl_plot.columns and sl_plot["sleep_need_min"].notna().any() else 8.0
            nfig = go.Figure()
            nfig.add_trace(
                go.Bar(
                    x=sl_plot["date"],
                    y=sl_plot["total_h"],
                    name="Night Sleep (h)",
                    marker_color="#a371f7",
                )
            )
            if sl_plot["nap_h"].sum() > 0:
                nfig.add_trace(
                    go.Bar(
                        x=sl_plot["date"],
                        y=sl_plot["nap_h"],
                        name="Daytime Nap (h)",
                        marker_color="#e3b341",
                    )
                )
            nfig.add_trace(
                go.Scatter(
                    x=sl_plot["date"],
                    y=sl_plot["need_h"],
                    mode="lines+markers",
                    name="Garmin Sleep Coach Need (h)",
                    line=dict(color="#58a6ff", width=2.2, dash="dot"),
                    marker=dict(size=5, color="#58a6ff"),
                )
            )
            nfig.update_layout(
                title="Night Sleep + Daytime Naps vs. Garmin Sleep Coach Need (hours)",
                barmode="stack",
            )
            st.plotly_chart(layout(nfig, 320), width="stretch")

        with nr_col:
            if isinstance(naps, pd.DataFrame) and not naps.empty:
                nap_tbl_rows = []
                for _, nr in naps.sort_values("start_local", ascending=False).iterrows():
                    st_t = pd.to_datetime(nr.get("start_local"))
                    en_t = pd.to_datetime(nr.get("end_local"))
                    win_s = f"{st_t:%H:%M} – {en_t:%H:%M}" if pd.notna(st_t) and pd.notna(en_t) else "–"
                    bb_v = nr.get("bb_impact")
                    bb_s = f"{int(bb_v):+d} BB" if pd.notna(bb_v) else "–"
                    ns_h = nr.get("night_sleep_h")
                    sc_v = nr.get("sleep_score")
                    ns_str = f"{ns_h:.1f} h (Score {sc_v:.0f})" if pd.notna(ns_h) and pd.notna(sc_v) else (f"{ns_h:.1f} h" if pd.notna(ns_h) else "–")
                    nap_tbl_rows.append({
                        "Date": pd.to_datetime(nr["date"]).strftime("%d %b %Y"),
                        "Time (Local)": win_s,
                        "Duration": f"{nr['duration_min']:.0f} min",
                        "BB Impact": bb_s,
                        "Night Sleep": ns_str,
                        "Garmin Nap Feedback": nr.get("feedback") or "Recorded nap",
                        "Source": nr.get("source") or "Watch",
                    })
                st.dataframe(pd.DataFrame(nap_tbl_rows), width="stretch", hide_index=True)
            else:
                st.info("No naps recorded in the selected date window. Increase the date window in the sidebar or log a nap on your Garmin watch to see nap timing, Body Battery impact, and Sleep Coach adjustments.")

# --------------------------------------------------------------- 6. Recovery & Fitness
with tabs[6]:
    c = st.columns(6)
    kpi_card(
        c[0],
        "Training Status",
        ts["status"],
        f"ACWR {ts['acwr']:.2f}x · {ts['load_status']} load",
        "optimalhealth",
        ts["color"],
    )
    kpi_card(
        c[1],
        "7d Acute Load",
        f"{ts['acute_load']}",
        f"Optimal: {ts['opt_low']}–{ts['opt_high']}",
        "speed",
        "#39d353" if ts["load_status"] == "Optimal" else "#e3b341",
    )
    vo2_delta = f"{vo2.vo2max.iloc[-1] - vo2.vo2max.iloc[0]:+.1f} vs start" if len(vo2) > 1 else "Latest estimate"
    vo2_sub = f"{vo2.vo2max.iloc[-1]:.1f} mL/kg · {vo2_delta}" if len(vo2) else vo2_delta
    kpi_card(c[2], "VO₂max", f"{vo2.vo2max.iloc[-1]:.0f}" if len(vo2) else "–", vo2_sub, "running", "#f78166")
    kpi_card(
        c[3],
        "HRV (Last Night)",
        f"{hrv.hrv.iloc[-1]:.0f} ms" if len(hrv) else "–",
        f"7d avg: {hrv.weekly_avg.dropna().iloc[-1]:.0f} ms · {ts['hrv_status']}" if len(hrv) and hrv.weekly_avg.notna().any() else ts["hrv_status"],
        "recovery",
        "#58a6ff",
    )
    kpi_card(
        c[4],
        "Resting HR",
        f"{daily.resting_hr.dropna().iloc[-1]:.0f} bpm" if len(daily) and daily.resting_hr.notna().any() else "–",
        f"Min: {daily.resting_hr.min():.0f} bpm" if len(daily) and daily.resting_hr.notna().any() else "–",
        "heartbeat",
        "#f85149",
    )
    kpi_card(
        c[5],
        "Body Battery High",
        f"{daily.body_battery_high.dropna().iloc[-1]:.0f}" if len(daily) and daily.body_battery_high.notna().any() else "–",
        f"Avg stress: {daily.stress.mean():.0f}" if len(daily) and daily.stress.notna().any() else "Daily peak charge",
        "calories",
        "#e3b341",
    )

    section_header(
        f"Training Status: {ts['status']}",
        ts["headline"],
        "optimalhealth",
        ts["color"],
    )
    tl_col, tf_col = st.columns([1.45, 1])
    hist_df = ts.get("history")
    if isinstance(hist_df, pd.DataFrame) and not hist_df.empty:
        lfig = go.Figure()
        lfig.add_traces([
            go.Scatter(x=hist_df["date"], y=hist_df["opt_high"], line=dict(width=0), showlegend=False, hoverinfo="skip"),
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["opt_low"],
                fill="tonexty",
                line=dict(width=0),
                fillcolor="rgba(57, 211, 83, 0.18)",
                name="Optimal Load Tunnel (0.8–1.3×)",
            ),
        ])
        lfig.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["chronic_load"],
                mode="lines",
                name="28d Chronic Load",
                line=dict(color="#8b949e", width=1.6, dash="dot"),
            )
        )
        lfig.add_trace(
            go.Scatter(
                x=hist_df["date"],
                y=hist_df["acute_load"],
                mode="lines+markers",
                name="7d Acute Load",
                line=dict(color=ts["color"], width=2.6),
                marker=dict(size=4, color=ts["color"]),
            )
        )
        lfig.update_layout(title="7-Day Acute Load vs. Optimal Chronic Tunnel")
        tl_col.plotly_chart(layout(lfig, 330), width="stretch")

    focus_dict = ts.get("focus") or {"Low Aerobic": 0, "High Aerobic": 0, "Anaerobic": 0}
    tot_focus = max(1, sum(focus_dict.values()))
    fdf = pd.DataFrame([
        {"Zone": k, "Load": v, "Share": f"{v} ({v / tot_focus * 100:.0f}%)"}
        for k, v in focus_dict.items()
    ])
    ffig = px.bar(
        fdf,
        x="Load",
        y="Zone",
        orientation="h",
        color="Zone",
        text="Share",
        color_discrete_map={"Low Aerobic": "#58a6ff", "High Aerobic": "#f78166", "Anaerobic": "#a371f7"},
        title="4-Week Training Load Focus",
    )
    ffig.update_traces(textposition="outside")
    ffig.update_xaxes(range=[0, max(fdf["Load"].max() * 1.28, 10)])
    ffig.update_layout(showlegend=False)
    tf_col.plotly_chart(layout(ffig, 330), width="stretch")

    l, r = st.columns(2)
    if len(vo2):
        vfig = px.line(vo2, x="date", y="vo2max", markers=True, title="Aerobic Fitness (VO₂max Trend)")
        vfig.update_traces(line=dict(color="#f78166", width=2.5), marker=dict(size=6))
        l.plotly_chart(layout(vfig), width="stretch")

    if len(hrv):
        fig = go.Figure()
        if hrv.low.notna().any():
            fig.add_traces([
                go.Scatter(x=hrv.date, y=hrv.high, line=dict(width=0), showlegend=False),
                go.Scatter(x=hrv.date, y=hrv.low, fill="tonexty", line=dict(width=0), fillcolor="rgba(46,160,67,0.2)", name="Balanced baseline"),
            ])
        fig.add_trace(go.Scatter(x=hrv.date, y=hrv.hrv, mode="markers+lines", name="Nightly HRV", line=dict(color="rgba(88,166,255,0.35)", width=1.2), marker=dict(color="#58a6ff", size=6)))
        if hrv.weekly_avg.notna().any():
            fig.add_trace(go.Scatter(x=hrv.date, y=hrv.weekly_avg, name="7-day avg", line=dict(color="#39d353", width=2.6)))
        fig.update_layout(title="Nightly Heart Rate Variability (ms)")
        r.plotly_chart(layout(fig), width="stretch")

    if len(daily):
        l2, r2 = st.columns(2)
        if daily.resting_hr.notna().any():
            rhr_df = daily.dropna(subset=["resting_hr"]).copy()
            rhr_df["roll7"] = rhr_df["resting_hr"].rolling(7, min_periods=1).mean()
            hfig = go.Figure()
            hfig.add_trace(go.Scatter(x=rhr_df["date"], y=rhr_df["resting_hr"], mode="markers+lines", name="Daily RHR", line=dict(color="rgba(248,81,73,0.35)", width=1.2), marker=dict(size=5, color="#f85149")))
            hfig.add_trace(go.Scatter(x=rhr_df["date"], y=rhr_df["roll7"], mode="lines", name="7-day avg", line=dict(color="#f85149", width=2.5)))
            hfig.update_layout(title="Resting Heart Rate (bpm)")
            l2.plotly_chart(layout(hfig), width="stretch")
        if daily.body_battery_high.notna().any() or daily.stress.notna().any():
            bfig = go.Figure()
            if daily.body_battery_high.notna().any():
                bb = daily.dropna(subset=["body_battery_high"])
                bfig.add_trace(go.Scatter(x=bb["date"], y=bb["body_battery_high"], name="Body Battery High", line=dict(color="#39d353", width=2.4)))
            if daily.stress.notna().any():
                st_df = daily.dropna(subset=["stress"])
                bfig.add_trace(go.Bar(x=st_df["date"], y=st_df["stress"], name="Daily Stress", marker_color="rgba(227,179,65,0.45)"))
            bfig.update_layout(title="Body Battery Peak & Daily Stress", barmode="overlay")
            r2.plotly_chart(layout(bfig), width="stretch")

# --------------------------------------------------------------- 7. Race Predictions
with tabs[7]:
    if not preds:
        st.info("No race predictions available (needs VO₂max or recent runs ≥ 3 km).")
    else:
        rd = race_details or {}
        eff_vdot = rd.get("effective_vdot") or 34.4
        race_vdot = rd.get("race_vdot")
        submax_vdot = rd.get("submax_vdot")
        lab_vo2 = rd.get("lab_vo2max") or (float(vo2["vo2max"].iloc[-1]) if not vo2.empty else 44.5)
        wk_km = rd.get("weekly_km") or 0.0
        long_km = rd.get("longest_run_km") or 0.0
        end_mult = rd.get("endurance_mult") or {"5K": 1.0, "10K": 1.0, "Half marathon": 1.0, "Marathon": 1.0}
        pure_preds = rd.get("pure_vdot_preds") or preds
        today_preds = rd.get("today_preds") or preds
        today_pen = rd.get("today_penalty_pct") or 0.0
        best_bench = rd.get("best_benchmark") or {}

        section_header(
            "Projected Race Finish Times (Peak / Tapered vs. Today's Readiness)",
            f"Calibrated to Effective Running VDOT {eff_vdot:.1f} · 4-wk volume {wk_km:.1f} km/wk · Today's Readiness {body_state['readiness_score']}/100",
            "race_predictor",
            "#f78166",
        )
        dist = {"5K": 5.0, "10K": 10.0, "Half marathon": 21.0975, "Marathon": 42.195}
        cols = st.columns(len(preds))
        for col, (k, secs) in zip(cols, preds.items()):
            td_s = today_preds.get(k, secs)
            td_txt = f" · Today: {fmt_time(td_s)}" if today_pen > 0 else " · Ready today"
            kpi_card(
                col,
                f"{k} (Peak / Rested)",
                fmt_time(secs),
                f"{fmt_time(secs / dist[k])}/km ({dist[k] / (secs / 3600):.1f} km/h){td_txt}",
                "race_predictor",
                "#f78166",
            )

        # Transparent Methodology & Calibration Section
        section_header(
            "How Your Race Predictions Are Calculated",
            "Exact physiological inputs, equations, and why raw watch VO₂max differs from race VDOT",
            "insights",
            "#58a6ff",
        )
        mc = st.columns(4)
        kpi_card(
            mc[0],
            "Effective Running VDOT",
            f"{eff_vdot:.2f}",
            f"72% Race PB ({race_vdot or eff_vdot:.1f}) + 28% 6w HR-Submax ({submax_vdot or eff_vdot:.1f})",
            "running",
            "#f78166",
        )
        bench_lbl = f"{best_bench.get('name', 'Best Run')[:22]} ({pd.to_datetime(best_bench['date']):%d %b})" if best_bench.get("date") is not None else "Verified Best Run"
        bench_val = f"VDOT {race_vdot:.1f}" if race_vdot else f"VDOT {eff_vdot:.1f}"
        kpi_card(
            mc[1],
            "Verified Race Anchor",
            bench_val,
            f"{bench_lbl} · {fmt_time(best_bench.get('pace_s_per_km', 338))}/km" if best_bench else "From fastest recorded race/time trial",
            "race_predictor",
            "#39d353",
        )
        kpi_card(
            mc[2],
            "Endurance Scaling",
            f"{wk_km:.1f} km/wk",
            f"Longest run {long_km:.1f} km → HM +{(end_mult.get('Half marathon', 1)-1)*100:.1f}%, Mar +{(end_mult.get('Marathon', 1)-1)*100:.1f}%",
            "plan",
            "#58a6ff",
        )
        kpi_card(
            mc[3],
            "Today's Fatigue Adjustment",
            f"+{today_pen:.1f}% slower" if today_pen > 0 else "0.0% (Fresh)",
            f"Readiness {body_state['readiness_score']}/100 · HRV {body_state['hrv_now'] or 0:.0f} ms · RHR {body_state['rhr_now'] or 0:.0f} bpm",
            "recovery",
            body_state["state_color"],
        )

        st.markdown(
            f'<div class="kpi-card" style="border-left: 4px solid #58a6ff; padding: 14px 18px; margin-bottom: 16px;">'
            f'  <div style="font-weight:700;color:#f0f6fc;font-size:0.92rem;margin-bottom:6px;">'
            f'    📐 Calculation Formula & Why Raw VO₂max ({lab_vo2:.1f} mL/kg/min) ≠ Running VDOT ({eff_vdot:.1f})'
            f'  </div>'
            f'  <div style="font-size:0.82rem;color:#c9d1d9;line-height:1.52;">'
            f'    <b>1. Why the old estimate was ~5–10 min too fast:</b> Standard watch/ACSM VO₂max (<b>{lab_vo2:.1f} mL/kg/min</b>) estimates gross laboratory oxygen uptake '
            f'(which includes <code>+3.5 mL/kg/min</code> resting metabolism and assumes elite running economy). Plugging raw {lab_vo2:.1f} directly into Jack Daniels\' race equations '
            f'falsely predicted a ~22:25 5K and ~46:26 10K.<br>'
            f'    <b>2. Exact Jack Daniels & Gilbert VDOT Solver:</b> We compute your true performance <b>VDOT ({eff_vdot:.2f})</b> directly from your verified race/split efforts '
            f'(e.g., <b>10.19 km in 57:25 @ 5:38/km = 34.8 VDOT</b> & <b>5K split in 27:22 = 34.4 VDOT</b>) blended with your last 6 weeks of HR-calibrated quality runs '
            f'(solving <code>VO₂(v) = -4.60 + 0.182258·v + 0.000104·v²</code> divided by <code>%HRR</code> between RHR {rd.get("rhr_used", 55):.0f} and watch Max HR {rd.get("mhr_used", 202):.0f} bpm, giving <b>{submax_vdot or eff_vdot:.1f} VDOT</b>).<br>'
            f'    <b>3. Time-Dependent %VO₂max & Mileage Endurance Factor:</b> For each race distance, we iteratively solve '
            f'<code>VO₂(d/t) / [0.8 + 0.1894·e^(-0.0128·t) + 0.2989·e^(-0.1933·t)] = VDOT</code>, then apply a Riegel endurance adjustment for Half Marathon (<b>+{(end_mult.get("Half marathon", 1)-1)*100:.1f}%</b>) '
            f'and Marathon (<b>+{(end_mult.get("Marathon", 1)-1)*100:.1f}%</b>) based on your current 4-week volume (<b>{wk_km:.1f} km/wk</b>) and longest run (<b>{long_km:.1f} km</b>).'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Detailed Race Table: Peak vs Pure VDOT vs Today's Fatigued State
        comp_rows = []
        for k, d_km in dist.items():
            if k not in preds:
                continue
            pk_s = preds[k]
            pr_s = pure_preds.get(k, pk_s)
            td_s = today_preds.get(k, pk_s)
            mult_pct = (end_mult.get(k, 1.0) - 1.0) * 100.0
            comp_rows.append({
                "Race Distance": k,
                "Peak / Tapered Time": fmt_time(pk_s),
                "Target Pace (/km)": fmt_time(pk_s / d_km) + " /km",
                "Avg Speed": f"{d_km / (pk_s / 3600.0):.2f} km/h",
                "Pure VDOT Ceiling": fmt_time(pr_s),
                "Endurance Volume Adj.": f"+{mult_pct:.1f}%" if mult_pct > 0 else "0.0% (Optimal)",
                f"Today's Time (Readiness {body_state['readiness_score']}/100)": fmt_time(td_s) + (f" (+{fmt_time(td_s - pk_s)})" if td_s > pk_s else ""),
            })
        st.dataframe(pd.DataFrame(comp_rows), width="stretch", hide_index=True)

        l, r = st.columns([1.2, 1])
        pdf = pd.DataFrame([
            {"Race": k, "Condition": "Peak / Tapered", "Pace (min/km)": preds[k] / dist[k] / 60.0, "Pace Label": fmt_time(preds[k] / dist[k]) + "/km"}
            for k in preds
        ] + ([
            {"Race": k, "Condition": f"Today (Readiness {body_state['readiness_score']}/100)", "Pace (min/km)": today_preds[k] / dist[k] / 60.0, "Pace Label": fmt_time(today_preds[k] / dist[k]) + "/km"}
            for k in preds
        ] if today_pen > 0 else []))
        pfig = px.bar(
            pdf,
            x="Race",
            y="Pace (min/km)",
            color="Condition",
            barmode="group",
            text="Pace Label",
            color_discrete_map={
                "Peak / Tapered": "#f78166",
                f"Today (Readiness {body_state['readiness_score']}/100)": "#e3b341",
            },
            title="Target Race Pace: Peak / Rested vs. Today's Readiness",
        )
        pfig.update_traces(textposition="outside")
        pfig.update_yaxes(range=[0, max(pdf["Pace (min/km)"]) * 1.22])
        l.plotly_chart(layout(pfig, 350), width="stretch")

        # Training pace zones derived from calibrated 5K/10K VDOT pace
        p5k = preds.get("5K", 1643) / 5.0  # sec/km
        zones = pd.DataFrame([
            {"Zone": "Easy / Recovery", "Purpose": "Aerobic base & capillary density", "Pace (/km)": f"{fmt_time(p5k * 1.24)} – {fmt_time(p5k * 1.35)}"},
            {"Zone": "Marathon Pace", "Purpose": "Sustained aerobic endurance", "Pace (/km)": f"{fmt_time(p5k * 1.12)} – {fmt_time(p5k * 1.16)}"},
            {"Zone": "Lactate Threshold (Tempo)", "Purpose": "Raise anaerobic threshold (~60m race effort)", "Pace (/km)": f"{fmt_time(p5k * 1.05)} – {fmt_time(p5k * 1.08)}"},
            {"Zone": "VO₂max Interval", "Purpose": "Maximal aerobic power (3K–5K pace)", "Pace (/km)": f"{fmt_time(p5k * 0.96)} – {fmt_time(p5k * 1.00)}"},
            {"Zone": "Repetition / Speed", "Purpose": "Running economy & neuromuscular turnover", "Pace (/km)": f"{fmt_time(p5k * 0.90)} – {fmt_time(p5k * 0.94)}"},
        ])
        with r:
            section_header("Calibrated Training Zones", "Matched to your verified VDOT & 5K block", "speed", "#58a6ff")
            st.dataframe(zones, width="stretch", hide_index=True)

        # Verified Race & Time-Trial Benchmarks + VDOT History
        top_eff = rd.get("top_efforts") or []
        vdot_tl = rd.get("vdot_timeline")
        if top_eff or (isinstance(vdot_tl, pd.DataFrame) and not vdot_tl.empty):
            b_left, b_right = st.columns([1.15, 1])
            if top_eff:
                with b_left:
                    section_header("Verified Best Efforts & Splits Used as Anchors", "Top recorded VDOT performances", "race_predictor", "#39d353")
                    eff_tbl = pd.DataFrame([
                        {
                            "Date": pd.to_datetime(e["date"]).strftime("%d %b %Y"),
                            "Activity": e["name"],
                            "Benchmark": e["type"],
                            "Time": fmt_time(e["time_s"]),
                            "Pace": fmt_time(e["pace_s_per_km"]) + " /km",
                            "HR (Avg/Max)": f"{e['avg_hr']:.0f}/{e['max_hr']:.0f} bpm" if e.get("avg_hr") and e.get("max_hr") else "–",
                            "VDOT": f"{e['vdot']:.1f}",
                        }
                        for e in top_eff
                    ])
                    st.dataframe(eff_tbl, width="stretch", hide_index=True)
            if isinstance(vdot_tl, pd.DataFrame) and not vdot_tl.empty:
                with b_right:
                    vtl = vdot_tl.sort_values("date").copy()
                    vtl["roll_vdot"] = vtl["vdot_hr"].rolling(4, min_periods=1).mean()
                    vt_fig = go.Figure()
                    vt_fig.add_trace(
                        go.Scatter(
                            x=vtl["date"],
                            y=vtl["vdot_hr"],
                            mode="markers",
                            name="Workout HR-Calibrated VDOT",
                            text=vtl["name"],
                            marker=dict(size=7, color="#58a6ff", opacity=0.75),
                        )
                    )
                    vt_fig.add_trace(
                        go.Scatter(
                            x=vtl["date"],
                            y=vtl["roll_vdot"],
                            mode="lines",
                            name="4-Run Rolling VDOT",
                            line=dict(color="#39d353", width=2.5),
                        )
                    )
                    vt_fig.add_hline(y=eff_vdot, line_dash="dash", line_color="#f78166", annotation_text=f"Effective VDOT {eff_vdot:.1f}")
                    vt_fig.update_layout(title="Submaximal HR-Calibrated VDOT Across Workouts")
                    st.plotly_chart(layout(vt_fig, 300), width="stretch")

# --------------------------------------------------------------- 8. Insights & Cross-Activity Correlations
with tabs[8]:
    ci = data.compute_correlations_and_insights(acts, sleep, hrv, daily, hevy_workouts, hevy_sets, days)
    master = ci["master"]
    insights_list = ci["insights"]
    corr_mat = ci["corr_matrix"]
    corr_labels = ci["corr_labels"]
    bs = ci.get("body_state") or body_state

    # =======================================================
    # A. LIVE SOMATIC & AUTONOMIC STATE ("HOW YOUR BODY FEELS RIGHT NOW")
    # =======================================================
    bs_col = bs["state_color"]
    section_header(
        "How Your Body Is Feeling Right Now — Live Somatic & Autonomic Diagnostic",
        "Real-time physiological state vs. your personal 28-day baselines across 6 internal systems",
        "heartbeat",
        bs_col,
    )

    st.markdown(
        f'<div class="ts-banner" style="border-left: 5px solid {bs_col}; padding: 18px 22px; margin-bottom: 16px;">'
        f'  <div class="ts-left">'
        f'    {icon_html("heartbeat", size=28, bg=f"{bs_col}22", accent=bs_col, badge=True, mr=4)}'
        f'    <div>'
        f'      <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'        <span style="font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.06em;color:#8b949e;">Current Somatic State</span>'
        f'        <span class="ts-badge" style="background:{bs_col}22;color:{bs_col};border:1px solid {bs_col}55;font-size:0.82rem;">'
        f'          <span class="status-dot"></span>{bs["state_label"]} (Readiness {bs["readiness_score"]}/100)'
        f'        </span>'
        f'      </div>'
        f'      <div style="font-size:0.92rem;color:#f0f6fc;font-weight:600;margin-top:6px;">{bs["headline"]}</div>'
        f'    </div>'
        f'  </div>'
        f'  <div class="ts-metrics">'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Somatic Readiness</div>'
        f'      <div class="ts-chip-val" style="color:{bs_col};font-size:1.25rem;">{bs["readiness_score"]} / 100</div>'
        f'      <div class="ts-chip-sub">{bs["state_badge"]}</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Last Night HRV</div>'
        f'      <div class="ts-chip-val">{bs["hrv_now"] or 0:.0f} ms</div>'
        f'      <div class="ts-chip-sub">Floor: {bs["hrv_low_band"]:.0f} ms ({bs["hrv_below_days"]}d low)</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Resting / Sleep HR</div>'
        f'      <div class="ts-chip-val">{bs["rhr_now"] or 0:.0f} / {bs["sl_hr_now"] or 0:.0f} bpm</div>'
        f'      <div class="ts-chip-sub">Base RHR: {bs["rhr_base"]:.1f} bpm</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Body Battery Peak</div>'
        f'      <div class="ts-chip-val">{bs["bb_now"] or 0:.0f} / 100</div>'
        f'      <div class="ts-chip-sub">28d Base: {bs["bb_base"]:.0f} / 100</div>'
        f'    </div>'
        f'    <div class="ts-chip">'
        f'      <div class="ts-chip-lbl">Last Night Sleep</div>'
        f'      <div class="ts-chip-val">Score {bs["sl_score_now"] or 0:.0f}</div>'
        f'      <div class="ts-chip-sub">{bs["sl_h_now"] or 0:.1f}h · Stress {bs["sl_stress_now"] or 0:.0f}</div>'
        f'    </div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # 1. Expected Somatic Sensations ("What You Are Feeling / Supposed to Feel Right Now")
    if bs.get("symptoms"):
        st.markdown(
            '<div style="font-size:0.9rem;font-weight:700;color:#f0f6fc;margin:4px 0 10px 0;">'
            '🧠 What Your Body Is Experiencing Right Now (Expected Physical & Mental Sensations)</div>',
            unsafe_allow_html=True,
        )
        sym_cols = st.columns(2)
        for idx_s, sym in enumerate(bs["symptoms"]):
            scol = sym_cols[idx_s % 2]
            s_acc = sym["color"]
            scol.markdown(
                f'<div class="kpi-card" style="border-left: 4px solid {s_acc}; padding: 15px 18px;">'
                f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:7px;gap:8px;">'
                f'    <div style="display:flex;align-items:center;gap:8px;">'
                f'      {icon_html(sym["icon"], size=17, bg=f"{s_acc}1f", accent=s_acc, badge=True, mr=2)}'
                f'      <div style="font-size:0.95rem;font-weight:700;color:#f0f6fc;">{sym["title"]}</div>'
                f'    </div>'
                f'    <span style="font-size:0.68rem;font-weight:700;text-transform:uppercase;color:{s_acc};background:{s_acc}1f;padding:3px 8px;border-radius:999px;">{sym["badge"]}</span>'
                f'  </div>'
                f'  <div style="font-size:0.82rem;color:#c9d1d9;line-height:1.48;">{sym["explanation"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 2. 6-System Biomarker Scorecard vs Personal 28-Day Baselines
    if bs.get("biomarkers"):
        st.markdown(
            '<div style="font-size:0.9rem;font-weight:700;color:#f0f6fc;margin:8px 0 10px 0;">'
            '🔬 6-System Biomarker Scorecard (Today vs. Your Personal 28-Day Baseline)</div>',
            unsafe_allow_html=True,
        )
        bm_cols = st.columns(3)
        for idx_b, bm in enumerate(bs["biomarkers"]):
            bcol = bm_cols[idx_b % 3]
            b_acc = bm["color"]
            bcol.markdown(
                f'<div class="kpi-card" style="border-top: 3px solid {b_acc}; padding: 14px 16px;">'
                f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">'
                f'    <span style="font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;color:#8b949e;">{bm["system"]}</span>'
                f'    <span style="font-size:0.66rem;font-weight:700;color:{b_acc};background:{b_acc}1f;padding:2px 7px;border-radius:999px;">{bm["status"]}</span>'
                f'  </div>'
                f'  <div style="font-family:\'JetBrains Mono\',monospace;font-size:1.28rem;font-weight:700;color:#f0f6fc;">{bm["value"]}</div>'
                f'  <div style="font-size:0.75rem;color:{b_acc};font-weight:600;margin-top:3px;">{bm["delta"]} <span style="color:#8b949e;font-weight:400;">(Base: {bm["baseline"]})</span></div>'
                f'  <div style="font-size:0.76rem;color:#c9d1d9;margin-top:7px;line-height:1.42;">{bm["note"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # 3. 14-Day Somatic Crash Timeline + Last 96-Hour Root-Cause Chain & Recovery Protocol
    hist_14 = bs.get("history_14d")
    t_left, t_right = st.columns([1.3, 1])
    if isinstance(hist_14, pd.DataFrame) and not hist_14.empty:
        with t_left:
            rfig = go.Figure()
            rfig.add_trace(
                go.Scatter(
                    x=hist_14["day"],
                    y=hist_14["readiness"],
                    mode="lines+markers",
                    name="Somatic Readiness (0–100)",
                    line=dict(color=bs_col, width=3.0),
                    marker=dict(size=7, color=bs_col),
                )
            )
            if hist_14["body_battery_high"].notna().any():
                rfig.add_trace(
                    go.Scatter(
                        x=hist_14["day"],
                        y=hist_14["body_battery_high"],
                        mode="lines+markers",
                        name="Body Battery Peak",
                        line=dict(color="#e3b341", width=1.8, dash="dot"),
                        marker=dict(size=5, color="#e3b341"),
                    )
                )
            if hist_14["sleep_score"].notna().any():
                rfig.add_trace(
                    go.Scatter(
                        x=hist_14["day"],
                        y=hist_14["sleep_score"],
                        mode="lines",
                        name="Sleep Score",
                        line=dict(color="#a371f7", width=1.8, dash="dash"),
                    )
                )
            if hist_14["hrv"].notna().any():
                rfig.add_trace(
                    go.Scatter(
                        x=hist_14["day"],
                        y=hist_14["hrv"],
                        mode="lines+markers",
                        name="Nightly HRV (ms)",
                        line=dict(color="#58a6ff", width=2.0),
                        marker=dict(size=5, color="#58a6ff"),
                    )
                )
            rfig.add_hline(y=bs["hrv_low_band"], line_dash="dot", line_color="rgba(88,166,255,0.45)", annotation_text=f"HRV Floor ({bs['hrv_low_band']:.0f} ms)")
            rfig.update_layout(title="14-Day Somatic Readiness, Body Battery, Sleep Score & HRV Trajectory")
            st.plotly_chart(layout(rfig, 370), width="stretch")

    with t_right:
        st.markdown(
            '<div style="font-size:0.88rem;font-weight:700;color:#f0f6fc;margin:2px 0 8px 0;">'
            '⏱️ Last 96-Hour Root-Cause Timeline (Why You Feel This Way)</div>',
            unsafe_allow_html=True,
        )
        for tr in bs.get("triggers", []):
            r_sc = tr["readiness"]
            t_col = "#f85149" if r_sc < 35 else ("#e3b341" if r_sc < 55 else "#39d353")
            flag_html = f'<div style="font-size:0.72rem;color:#f85149;margin-top:3px;">⚠️ Garmin Sensor Flag: {tr["device_flag"]}</div>' if tr.get("device_flag") else ""
            st.markdown(
                f'<div style="background:#12171f;border:1px solid #262c36;border-left:3px solid {t_col};border-radius:8px;padding:9px 12px;margin-bottom:7px;">'
                f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
                f'    <span style="font-weight:700;font-size:0.82rem;color:#f0f6fc;">{tr["date_label"]} · {tr["workouts"]}</span>'
                f'    <span style="font-size:0.72rem;color:#8b949e;">{tr["steps"]:,} steps · Load {tr["load"]}</span>'
                f'  </div>'
                f'  <div style="font-size:0.76rem;color:#c9d1d9;margin-top:3px;">{tr["metrics"]}</div>'
                f'  {flag_html}'
                f'</div>',
                unsafe_allow_html=True,
            )

    if bs.get("recommendations"):
        recs_html = "".join(f'<li style="margin-bottom:5px;">{r_item}</li>' for r_item in bs["recommendations"])
        st.markdown(
            f'<div class="kpi-card" style="border-left: 4px solid #39d353; padding: 14px 18px; margin-bottom: 18px;">'
            f'  <div style="font-weight:700;color:#f0f6fc;font-size:0.9rem;margin-bottom:6px;">🩺 Actionable Recovery Prescription for Today</div>'
            f'  <ul style="margin:0;padding-left:18px;font-size:0.82rem;color:#c9d1d9;line-height:1.48;">{recs_html}</ul>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    section_header(
        "Auto-Discovered Physiological & Cross-Sport Patterns",
        f"Statistical relationships mined across your last {days} days of Garmin + Hevy telemetry",
        "insights",
        "#58a6ff",
    )

    if insights_list:
        ins_cols = st.columns(2)
        for idx_i, ins in enumerate(insights_list):
            col_i = ins_cols[idx_i % 2]
            acc = ins["accent"]
            r_val = ins["r"]
            r_badge = f"r = {r_val:+.2f} · n={ins['n']}"
            col_i.markdown(
                f'<div class="kpi-card" style="border-left: 4px solid {acc}; padding: 16px 18px;">'
                f'  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;gap:8px;">'
                f'    <div style="display:flex;align-items:center;gap:8px;">'
                f'      {icon_html(ins["icon"], size=17, bg=f"{acc}1f", accent=acc, badge=True, mr=2)}'
                f'      <div>'
                f'        <div style="font-size:0.7rem;text-transform:uppercase;letter-spacing:0.06em;color:{acc};font-weight:700;">{ins["category"]}</div>'
                f'        <div style="font-size:0.96rem;font-weight:700;color:#f0f6fc;">{ins["title"]}</div>'
                f'      </div>'
                f'    </div>'
                f'    <div style="text-align:right;flex-shrink:0;">'
                f'      <div style="font-family:\'JetBrains Mono\',monospace;font-size:0.95rem;font-weight:700;color:{acc};">{ins["metric"]}</div>'
                f'      <div style="font-size:0.68rem;color:#8b949e;">{r_badge}</div>'
                f'    </div>'
                f'  </div>'
                f'  <div style="font-size:0.82rem;color:#c9d1d9;line-height:1.48;">{ins["summary"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Correlation Matrix + Top Ranked Pairs
    if not corr_mat.empty:
        section_header(
            "Multi-System Telemetry Correlation Matrix",
            "Pearson r (-1.0 to +1.0) across Cardio, Strength, Sleep, and Autonomic Recovery",
            "optimalhealth",
            "#39d353",
        )
        m_left, m_right = st.columns([1.45, 1])
        cfig = px.imshow(
            corr_mat,
            text_auto=".2f",
            color_continuous_scale=["#f85149", "#161b22", "#39d353"],
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Cross-Domain Correlation Heatmap (Pearson r)",
        )
        cfig.update_layout(coloraxis_showscale=True)
        m_left.plotly_chart(layout(cfig, 420), width="stretch")

        pair_rows = []
        cols_c = list(corr_mat.columns)
        for i in range(len(cols_c)):
            for j in range(i + 1, len(cols_c)):
                rv = float(corr_mat.iloc[i, j])
                if pd.notna(rv):
                    strength = (
                        "Strong" if abs(rv) >= 0.50
                        else "Moderate" if abs(rv) >= 0.30
                        else "Mild" if abs(rv) >= 0.15
                        else "Weak"
                    )
                    pair_rows.append({
                        "Metric A": cols_c[i],
                        "Metric B": cols_c[j],
                        "Pearson r": rv,
                        "Abs_r": abs(rv),
                        "Strength": f"{strength} ({'Positive' if rv > 0 else 'Inverse'})",
                    })
        if pair_rows:
            top_pairs = (
                pd.DataFrame(pair_rows)
                .sort_values("Abs_r", ascending=False)
                .drop(columns=["Abs_r"])
                .head(12)
                .reset_index(drop=True)
            )
            with m_right:
                st.markdown(
                    '<div style="font-size:0.86rem;font-weight:700;color:#f0f6fc;margin:6px 0 10px 0;">'
                    'Strongest Detected Signal Pairs</div>',
                    unsafe_allow_html=True,
                )
                st.dataframe(top_pairs, width="stretch", hide_index=True, height=375)

    # Deep-Dive Cross-Activity Charts
    section_header(
        "Cross-Activity Deep Dives",
        "How sleep, workout time-of-day, leg-day lifting, and stress interact",
        "recovery",
        "#a371f7",
    )
    d1, d2 = st.columns(2)

    w_timing = master.dropna(subset=["latest_workout_hour", "next_day_rhr_bpm"]).copy()
    if len(w_timing) >= 4:
        w_timing["Time Bucket"] = np.where(
            w_timing["latest_workout_hour"] < 12.0,
            "Morning (<12:00)",
            np.where(w_timing["latest_workout_hour"] < 17.5, "Afternoon (12:00–17:30)", "Evening (≥17:30)"),
        )
        tfig = px.scatter(
            w_timing,
            x="latest_workout_hour",
            y="next_day_rhr_bpm",
            color="Time Bucket",
            size=w_timing["hybrid_load"].clip(lower=15),
            color_discrete_map={
                "Morning (<12:00)": "#39d353",
                "Afternoon (12:00–17:30)": "#58a6ff",
                "Evening (≥17:30)": "#f85149",
            },
            hover_data={"day": True, "garmin_load": ":.0f", "tonight_sleep_score": ":.0f"},
            labels={"latest_workout_hour": "Workout Start Hour (24h)", "next_day_rhr_bpm": "Next-Morning RHR (bpm)"},
            title="Workout Time-of-Day vs. Next-Morning Resting HR (bubble = load)",
        )
        if len(w_timing) >= 5 and w_timing["latest_workout_hour"].std() > 0:
            m_slope, b_int = np.polyfit(w_timing["latest_workout_hour"], w_timing["next_day_rhr_bpm"], 1)
            xs = np.linspace(w_timing["latest_workout_hour"].min(), w_timing["latest_workout_hour"].max(), 30)
            tfig.add_trace(
                go.Scatter(x=xs, y=m_slope * xs + b_int, mode="lines", name="Trend", line=dict(color="#f0f6fc", width=1.8, dash="dash"))
            )
        d1.plotly_chart(layout(tfig, 340), width="stretch")

    st_rec = master.dropna(subset=["stress", "sleep_score"]).copy()
    if len(st_rec) >= 4:
        sfig = px.scatter(
            st_rec,
            x="stress",
            y="sleep_score",
            color="rhr_bpm",
            color_continuous_scale="Turbo",
            hover_data={"day": True, "sleep_h": ":.1f"},
            labels={"stress": "Daily Stress Level", "sleep_score": "Garmin Sleep Score", "rhr_bpm": "RHR (bpm)"},
            title="Daily Stress vs. Sleep Score (color = Resting HR)",
        )
        if len(st_rec) >= 5 and st_rec["stress"].std() > 0:
            m_s, b_s = np.polyfit(st_rec["stress"], st_rec["sleep_score"], 1)
            xs = np.linspace(st_rec["stress"].min(), st_rec["stress"].max(), 30)
            sfig.add_trace(
                go.Scatter(x=xs, y=m_s * xs + b_s, mode="lines", name="OLS Fit", line=dict(color="#a371f7", width=2.2, dash="dash"))
            )
        d2.plotly_chart(layout(sfig, 340), width="stretch")

    # Interactive Custom Correlation Lab
    section_header(
        "Interactive Correlation Explorer",
        "Test any two metrics (including next-day lag effects) with live linear regression",
        "curious",
        "#e3b341",
    )
    lab_metrics = {
        "garmin_load": "Garmin Cardio Load",
        "hybrid_load": "Combined Hybrid Load (Garmin + Hevy)",
        "run_eff": "Running Aerobic Efficiency (m/beat)",
        "run_km": "Daily Running Distance (km)",
        "run_hr": "Average Running HR (bpm)",
        "bike_km": "Daily Cycling Distance (km)",
        "lift_volume_kg": "Hevy Lifting Tonnage (kg)",
        "leg_volume_kg": "Hevy Leg-Day Tonnage (kg)",
        "latest_workout_hour": "Workout Start Hour (24h)",
        "steps": "Daily Steps",
        "sleep_h": "Sleep Duration (hours)",
        "sleep_score": "Garmin Sleep Score",
        "deep_pct": "Deep Sleep Share (%)",
        "rem_pct": "REM Sleep Share (%)",
        "hrv_ms": "Nightly HRV (ms)",
        "rhr_bpm": "Resting Heart Rate (bpm)",
        "body_battery": "Body Battery Peak",
        "stress": "Daily Stress Level",
    }
    avail_lab = [k for k in lab_metrics if k in master.columns and master[k].notna().sum() >= 4]
    if len(avail_lab) >= 2:
        c_x, c_y, c_lag = st.columns([1.2, 1.2, 1])
        def_x = "stress" if "stress" in avail_lab else avail_lab[0]
        def_y = "sleep_score" if "sleep_score" in avail_lab else avail_lab[min(1, len(avail_lab) - 1)]
        x_col = c_x.selectbox("X-Axis (Predictor)", avail_lab, index=avail_lab.index(def_x), format_func=lambda k: lab_metrics[k])
        y_col = c_y.selectbox("Y-Axis (Outcome)", avail_lab, index=avail_lab.index(def_y), format_func=lambda k: lab_metrics[k])
        lag_mode = c_lag.selectbox(
            "Time Alignment",
            ["Same Day (t → t)", "Next Day Outcome (t → t+1)", "48h Outcome (t → t+2)"],
            index=0,
        )
        shift_n = -1 if "t+1" in lag_mode else (-2 if "t+2" in lag_mode else 0)
        df_lab = pd.DataFrame({
            "day": master["day"],
            "x": master[x_col],
            "y": master[y_col].shift(shift_n) if shift_n != 0 else master[y_col],
        })
        workout_zero_cols = {"garmin_load", "hybrid_load", "run_km", "bike_km", "lift_volume_kg", "leg_volume_kg"}
        if x_col in workout_zero_cols:
            df_lab = df_lab[df_lab["x"] > 0]
        if y_col in workout_zero_cols:
            df_lab = df_lab[df_lab["y"] > 0]
        df_lab = df_lab.dropna(subset=["x", "y"])

        if len(df_lab) >= 4 and df_lab["x"].std() > 0 and df_lab["y"].std() > 0:
            r_custom, n_custom = data._pearson(df_lab["x"], df_lab["y"])
            slope, intercept = np.polyfit(df_lab["x"], df_lab["y"], 1)
            xs = np.linspace(df_lab["x"].min(), df_lab["x"].max(), 40)
            y_label_full = f"{lab_metrics[y_col]} ({lag_mode.split('(')[1][:-1]})"
            custom_fig = go.Figure()
            custom_fig.add_trace(
                go.Scatter(
                    x=df_lab["x"],
                    y=df_lab["y"],
                    mode="markers",
                    name="Daily Observations",
                    text=df_lab["day"].dt.strftime("%d %b %Y"),
                    hovertemplate=f"<b>%{{text}}</b><br>{lab_metrics[x_col]}: %{{x:.2f}}<br>{lab_metrics[y_col]}: %{{y:.2f}}<extra></extra>",
                    marker=dict(size=8, color="#58a6ff", opacity=0.82, line=dict(width=1, color="#161b22")),
                )
            )
            custom_fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=slope * xs + intercept,
                    mode="lines",
                    name=f"Linear Fit (r = {r_custom:+.2f})",
                    line=dict(color="#e3b341", width=2.5, dash="dash"),
                )
            )
            custom_fig.update_layout(
                title=f"{lab_metrics[x_col]} vs. {y_label_full} — Pearson r = {r_custom:+.2f} (n = {n_custom} days)",
                xaxis_title=lab_metrics[x_col],
                yaxis_title=y_label_full,
            )
            st.plotly_chart(layout(custom_fig, 360), width="stretch")
        else:
            st.info("Not enough overlapping non-zero days for this metric pair yet — try expanding the History window in the sidebar.")

# --------------------------------------------------------------- 9. Activity Log
with tabs[9]:
    f1, f2 = st.columns([2, 1])
    cats = f1.multiselect("Filter by sport", list(COLORS), default=list(COLORS))
    limit = f2.selectbox("Show recent", [30, 60, 100, 250], index=1)

    filtered = acts[acts.category.isin(cats)].sort_values("date", ascending=False).head(limit)
    section_header("Workout Feed", f"Showing {len(filtered)} activities", "plan", "#58a6ff")

    speed_ico = icon_html("speed", size=14, badge=False, mr=5)
    hr_ico = icon_html("heartbeat", size=14, badge=False, mr=5)
    cal_ico = icon_html("calories", size=14, badge=False, mr=5)

    for _, a in filtered.iterrows():
        s_ico = sport_icon_name(a.type, a.category)
        accent = COLORS.get(a.category, "#8b949e")
        badge = icon_html(s_ico, size=20, bg=f"{accent}1f", accent=accent, badge=True, mr=0)
        spd_str = pace(a.avg_speed_kmh) if a.category == "Run" else f"{a.avg_speed_kmh:.1f} km/h"
        hr_pill = (
            f'<span class="feed-pill">{hr_ico}{a.avg_hr:.0f} bpm</span>'
            if pd.notna(a.avg_hr) and a.avg_hr > 0
            else ""
        )
        te_pill = (
            f'<span class="feed-pill" style="color:{accent};border-color:{accent}44;">TE {a.training_effect:.1f}</span>'
            if pd.notna(a.training_effect) and a.training_effect > 0
            else ""
        )
        st.markdown(
            f'<div class="feed-card">'
            f'  <div class="feed-left">'
            f'    {badge}'
            f'    <div>'
            f'      <div class="feed-title">{a["name"]}</div>'
            f'      <div class="feed-date">{pd.to_datetime(a.date):%a %d %b %Y · %H:%M} · {a.type.replace("_", " ").title()}</div>'
            f'    </div>'
            f'  </div>'
            f'  <div class="feed-metrics">'
            f'    <span class="feed-pill"><b>{a.distance_km:.2f} km</b>&nbsp;·&nbsp;{fmt_time(a.duration_min * 60)}</span>'
            f'    <span class="feed-pill">{speed_ico}{spd_str}</span>'
            f'    {hr_pill}'
            f'    <span class="feed-pill">{cal_ico}{a.calories or 0:.0f} kcal</span>'
            f'    {te_pill}'
            f'  </div>'
            f'</div>',
            unsafe_allow_html=True,
        )
