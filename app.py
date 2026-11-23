import streamlit as st
import pandas as pd
import numpy as np
from html import escape

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score

try:
    import altair as alt
except ImportError:
    alt = None

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DataMind AI",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

NAV_PAGES = ["Home", "Explore Data", "ML Studio", "About"]

if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "filename" not in st.session_state:
    st.session_state.filename = None
if "model_results" not in st.session_state:
    st.session_state.model_results = None
if "page" not in st.session_state:
    st.session_state.page = "Home"

if st.session_state.pop("go_explore", False):
    st.session_state.page = "Explore Data"
if st.session_state.pop("go_ml", False):
    st.session_state.page = "ML Studio"

# --------------------------------------------------
# SMALL HELPERS & CHARTS
# --------------------------------------------------

PURPLE = "#6d3fc0"
MUTED = "#6f6786"

def render_html(markup):
    flat = " ".join(line.strip() for line in markup.splitlines() if line.strip())
    st.markdown(flat, unsafe_allow_html=True)

def _streamlit_version():
    try:
        return tuple(int(p) for p in st.__version__.split(".")[:2])
    except Exception:
        return (0, 0)

_NEW_STRETCH = {"width": "stretch"}
_OLD_STRETCH = {"use_container_width": True}
_STRETCH_ORDER = (
    (_NEW_STRETCH, _OLD_STRETCH) if _streamlit_version() >= (1, 50) else (_OLD_STRETCH, _NEW_STRETCH)
)

def call_stretch(fn, *args, **kwargs):
    for stretch in _STRETCH_ORDER:
        try:
            return fn(*args, **kwargs, **stretch)
        except TypeError:
            continue
    return fn(*args, **kwargs)

def show_df(data, **kwargs):
    call_stretch(st.dataframe, data, **kwargs)

def render_chart(chart):
    try:
        call_stretch(st.altair_chart, chart, theme=None)
    except Exception:
        call_stretch(st.altair_chart, chart)

def style_chart(chart, height):
    return (
        chart.properties(height=height)
        .configure(background="transparent")
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelFont="DM Sans", titleFont="DM Sans", labelColor=MUTED, titleColor=MUTED,
            gridColor="#eee7f8", domainColor="#d9cdef", tickColor="#d9cdef",
            labelFontSize=11, titleFontSize=11, titleFontWeight=600,
        )
    )

def bar_chart(labels, values, x_title="", y_title="", horizontal=False, height=320):
    df = pd.DataFrame({"label": [str(label) for label in labels], "value": list(values)})
    if alt is not None:
        try:
            order = list(df["label"])
            color = alt.Color("value:Q", scale=alt.Scale(range=["#d9c8f7", "#4b2a8f"]), legend=None)
            if horizontal:
                chart = alt.Chart(df).mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6).encode(
                    y=alt.Y("label:N", sort=order, title=y_title, axis=alt.Axis(labelLimit=260)),
                    x=alt.X("value:Q", title=x_title), color=color,
                    tooltip=[alt.Tooltip("label:N", title="Feature"), alt.Tooltip("value:Q", title="Value", format=",.4f")],
                )
            else:
                chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                    x=alt.X("label:N", sort=order, title=x_title, axis=alt.Axis(labelAngle=-35, labelLimit=160)),
                    y=alt.Y("value:Q", title=y_title), color=color,
                    tooltip=[alt.Tooltip("label:N", title="Value"), alt.Tooltip("value:Q", title="Count", format=",")],
                )
            render_chart(style_chart(chart, height))
            return
        except Exception:
            pass
    st.bar_chart(df.set_index("label")["value"])

def histogram_chart(series, title="", height=320):
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    values = values[np.isfinite(values)]
    if len(values) == 0:
        st.info("No finite numeric values to plot.")
        return
    counts, edges = np.histogram(values, bins=30)
    if alt is not None:
        try:
            df = pd.DataFrame({"start": edges[:-1], "end": edges[1:], "count": counts})
            chart = alt.Chart(df).mark_bar(color=PURPLE, opacity=0.9, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("start:Q", bin="binned", title=title), x2="end:Q", y=alt.Y("count:Q", title="Count"),
                tooltip=[alt.Tooltip("start:Q", title="From", format=",.3f"), alt.Tooltip("end:Q", title="To", format=",.3f"), alt.Tooltip("count:Q", title="Count", format=",")],
            )
            render_chart(style_chart(chart, height))
            return
        except Exception:
            pass
    st.bar_chart(pd.Series(counts, index=np.round(edges[:-1], 3)))

def correlation_chart(corr, height=420):
    long = corr.reset_index().melt(id_vars="index", var_name="b", value_name="corr").rename(columns={"index": "a"}).dropna()
    order = list(corr.columns)
    if alt is not None:
        try:
            chart = alt.Chart(long).mark_rect(cornerRadius=4).encode(
                x=alt.X("a:N", sort=order, title=None, axis=alt.Axis(labelAngle=-40, labelLimit=140)),
                y=alt.Y("b:N", sort=order, title=None, axis=alt.Axis(labelLimit=140)),
                color=alt.Color("corr:Q", scale=alt.Scale(domain=[-1, 0, 1], range=["#e58bb0", "#fbf7f0", "#5b2fb0"]), legend=alt.Legend(title="Correlation")),
                tooltip=[alt.Tooltip("a:N", title="Feature"), alt.Tooltip("b:N", title="Against"), alt.Tooltip("corr:Q", title="Correlation", format=".3f")],
            )
            render_chart(style_chart(chart, height))
            return
        except Exception:
            pass
    show_df(corr.round(3))

def actual_vs_predicted_chart(actual, predicted, height=380):
    df = pd.DataFrame({"Actual": np.asarray(actual, dtype=float), "Predicted": np.asarray(predicted, dtype=float)})
    if len(df) > 2000:
        df = df.sample(2000, random_state=42)
    if alt is not None:
        try:
            lo = float(min(df["Actual"].min(), df["Predicted"].min()))
            hi = float(max(df["Actual"].max(), df["Predicted"].max()))
            points = alt.Chart(df).mark_circle(size=62, color=PURPLE, opacity=0.55).encode(
                x=alt.X("Actual:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
                y=alt.Y("Predicted:Q", scale=alt.Scale(domain=[lo, hi], zero=False)),
                tooltip=[alt.Tooltip("Actual:Q", format=",.3f"), alt.Tooltip("Predicted:Q", format=",.3f")],
            )
            ideal = alt.Chart(pd.DataFrame({"Actual": [lo, hi], "Predicted": [lo, hi]})).mark_line(color="#e58bb0", strokeDash=[6, 5], strokeWidth=2).encode(x="Actual:Q", y="Predicted:Q")
            render_chart(style_chart(points + ideal, height))
            st.caption("Each dot is a held-out test record. The dashed line marks perfect predictions.")
            return
        except Exception:
            pass
    st.scatter_chart(df, x="Actual", y="Predicted")

def stat_strip(items):
    cells = "".join(
        f'<div class="stat"><div class="stat-value">{escape(str(value))}</div>'
        f'<div class="stat-label">{escape(label)}</div></div>'
        for label, value in items
    )
    render_html(f'<div class="stat-strip reveal">{cells}</div>')

def page_header(eyebrow_text, title, description):
    render_html(f"""
    <div class="reveal">
        <div class="eyebrow">{eyebrow_text}</div>
        <div class="section-heading">{title}</div>
        <p class="section-description">{description}</p>
    </div>
    """)

def sub_heading(text):
    render_html(f'<div class="sub-heading">{text}</div>')

# --------------------------------------------------
# CUSTOM DESIGN (CSS)
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500;1,600&display=swap');

:root {
    --ivory: #fbf7f0;
    --ink: #241a3a;
    --muted: #6f6786;
    --violet: #2f1a5c;
    --purple: #6d3fc0;
    --purple-2: #7c4dd6;
    --lavender: #e9dfff;
    --lav-2: #f4eefe;
    --pink: #f7c9dc;
    --line: #e7def5;
    --shadow: 0 18px 40px -22px rgba(47, 26, 92, 0.38);
}

html { font-size: 15px; }

.stApp {
    color-scheme: light;
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    background:
        radial-gradient(900px 520px at 6% -6%, rgba(214, 194, 255, 0.60), transparent 62%),
        radial-gradient(820px 520px at 100% 0%, rgba(247, 201, 220, 0.52), transparent 58%),
        radial-gradient(760px 620px at 55% 112%, rgba(233, 223, 255, 0.70), transparent 62%),
        var(--ivory);
    background-attachment: fixed;
}

header, footer, #MainMenu, [data-testid="stSidebar"],
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none !important; }

.main .block-container, [data-testid="stMainBlockContainer"] {
    max-width: 1120px;
    padding: 1rem 2.4rem 4rem;
}

.stApp p, .stApp li, .stApp label, [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] {
    color: var(--ink);
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stCaptionContainer"] p { color: var(--muted); }

h1, h2, h3, h4 { font-family: 'Playfair Display', serif; color: var(--violet); }
::selection { background: var(--lavender); color: var(--violet); }

@keyframes fadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes floaty { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-14px); } }
@keyframes drift { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes draw { from { stroke-dashoffset: 700; } to { stroke-dashoffset: 0; } }
@keyframes pulse { 0%, 100% { transform: scale(1); opacity: 1; } 50% { transform: scale(1.6); opacity: 0.55; } }

.reveal { animation: fadeUp 0.7s cubic-bezier(0.2, 0.7, 0.2, 1) both; }
.d1 { animation-delay: 0.08s; }
.d2 { animation-delay: 0.16s; }
.d3 { animation-delay: 0.24s; }

.brand { font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 600; color: var(--violet); letter-spacing: -0.5px; padding-top: 4px; }
.brand-mark { display: inline-block; background: linear-gradient(135deg, #6d3fc0, #e58bb0); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent; transition: transform 0.6s ease; }
.brand:hover .brand-mark { transform: rotate(90deg); }
.brand-ai { font-style: italic; color: var(--purple); }

.top-note { text-align: right; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding-top: 10px; }

[data-testid="stRadio"] [role="radiogroup"] {
    display: inline-flex; flex-wrap: wrap; gap: 4px; padding: 5px;
    background: rgba(255, 255, 255, 0.72); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
    border: 1px solid var(--line); border-radius: 999px; box-shadow: 0 10px 30px -18px rgba(47, 26, 92, 0.45);
}
[data-testid="stRadio"] [role="radiogroup"] > label { margin: 0; padding: 7px 17px; border-radius: 999px; cursor: pointer; transition: background 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease; }
[data-testid="stRadio"] [role="radiogroup"] > label > div:first-child { display: none; }
[data-testid="stRadio"] [role="radiogroup"] > label p { font-size: 13px; font-weight: 600; color: var(--muted); transition: color 0.25s ease; }
[data-testid="stRadio"] [role="radiogroup"] > label:hover { background: var(--lav-2); transform: translateY(-1px); }
[data-testid="stRadio"] [role="radiogroup"] > label:hover p { color: var(--purple); }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) { background: linear-gradient(135deg, #4b2a8f, #7c4dd6 70%, #a773e6); box-shadow: 0 10px 22px -10px rgba(91, 47, 176, 0.65); }
[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p { color: #ffffff !important; }

.nav-rule { height: 1px; margin: 10px 0 0; background: linear-gradient(90deg, transparent, #d9cdef 20%, #d9cdef 80%, transparent); }

.hero { display: grid; grid-template-columns: 1.3fr 0.8fr; gap: 28px; align-items: center; padding: 22px 0 14px; }
.eyebrow { color: var(--purple); font-size: 12px; font-weight: 700; letter-spacing: 2.2px; text-transform: uppercase; margin-top: 10px; }
.eyebrow-pill { display: inline-flex; align-items: center; gap: 10px; padding: 5px 13px; border-radius: 999px; background: rgba(255, 255, 255, 0.72); border: 1px solid var(--line); color: var(--purple); font-size: 11px; font-weight: 700; letter-spacing: 1.8px; text-transform: uppercase; }
.eyebrow-pill i { width: 7px; height: 7px; border-radius: 50%; background: linear-gradient(135deg, #6d3fc0, #e58bb0); animation: pulse 2.4s ease-in-out infinite; }

.hero-title { font-family: 'Playfair Display', serif; font-size: clamp(34px, 5vw, 56px); font-weight: 600; line-height: 1.06; letter-spacing: -1.5px; color: var(--violet); margin: 14px 0 12px; }
.hero-title em { font-style: italic; padding-right: 0.08em; background: linear-gradient(120deg, #6d3fc0 10%, #d86aa0 95%); -webkit-background-clip: text; background-clip: text; color: transparent; -webkit-text-fill-color: transparent; }
.hero-description { color: var(--muted); font-size: 15px; line-height: 1.7; max-width: 520px; }
.hero-tags { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 16px; }
.hero-tags span { padding: 5px 13px; border-radius: 999px; font-size: 12px; font-weight: 600; color: var(--violet); background: linear-gradient(135deg, var(--lavender), #fde8f1); border: 1px solid rgba(124, 77, 214, 0.15); transition: transform 0.25s ease, box-shadow 0.25s ease; }
.hero-tags span:hover { transform: translateY(-3px); box-shadow: 0 12px 22px -14px rgba(91, 47, 176, 0.6); }

.hero-art { position: relative; height: 230px; }
.orb { position: absolute; width: 200px; height: 200px; right: 6px; top: 0; border-radius: 50%; background: radial-gradient(circle at 30% 28%, #fbd6e6, #cdb3f6 52%, #7c4dd6); opacity: 0.9; animation: floaty 9s ease-in-out infinite; }
.glass { position: absolute; left: 0; right: 30px; bottom: 0; top: 40px; padding: 12px; border-radius: 28px; background: rgba(255, 255, 255, 0.58); border: 1px solid rgba(255, 255, 255, 0.85); backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px); box-shadow: var(--shadow); animation: floaty 11s ease-in-out infinite reverse; }
.glass svg { width: 100%; height: 100%; }
.wave { stroke-dasharray: 700; animation: draw 2.6s ease-out forwards; }
.wave.w2 { animation-delay: 0.3s; }
.wave.w3 { animation-delay: 0.6s; }

.section-heading { font-family: 'Playfair Display', serif; font-size: 30px; font-weight: 600; line-height: 1.15; letter-spacing: -0.6px; color: var(--violet); margin: 8px 0 4px; }
.section-description { color: var(--muted) !important; line-height: 1.65; font-size: 13.5px; max-width: 620px; }
.sub-heading { font-family: 'Playfair Display', serif; font-size: 21px; font-weight: 600; color: var(--violet); margin: 20px 0 8px; letter-spacing: -0.3px; }
.rule { height: 1px; margin: 34px 0 10px; background: linear-gradient(90deg, transparent, #d9cdef, transparent); }

.soft-panel { background: linear-gradient(135deg, rgba(255, 255, 255, 0.8), rgba(244, 238, 254, 0.8)); border: 1px solid var(--line); border-radius: 20px; padding: 16px 20px; margin: 12px 0; transition: transform 0.3s ease, box-shadow 0.3s ease; }
.soft-panel:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.feature-title { font-size: 14.5px; font-weight: 700; color: var(--violet); margin-bottom: 4px; }
.feature-description { font-size: 12.5px; color: var(--muted); line-height: 1.6; }

.step { display: flex; gap: 14px; padding: 14px 18px; margin-bottom: 10px; background: rgba(255, 255, 255, 0.7); border: 1px solid var(--line); border-radius: 18px; transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s ease; }
.step:hover { transform: translateX(6px); box-shadow: var(--shadow); border-color: #cdb7f3; }
.step-num { font-family: 'Playfair Display', serif; font-size: 24px; line-height: 1; color: #b79af0; transition: color 0.3s ease; }
.step:hover .step-num { color: var(--purple); }
.number { color: var(--purple); font-size: 11px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; margin-bottom: 4px; }

.stat-strip { display: flex; flex-wrap: wrap; margin: 12px 0 16px; background: rgba(255, 255, 255, 0.74); border: 1px solid var(--line); border-radius: 22px; overflow: hidden; box-shadow: var(--shadow); backdrop-filter: blur(8px); }
.stat { flex: 1 1 140px; padding: 14px 20px; border-right: 1px solid var(--line); transition: background 0.25s ease; }
.stat:last-child { border-right: none; }
.stat:hover { background: var(--lav-2); }
.stat-value { font-family: 'Playfair Display', serif; font-size: 26px; font-weight: 600; line-height: 1.1; color: var(--violet); }
.stat-label { margin-top: 6px; font-size: 11px; font-weight: 700; letter-spacing: 1.6px; text-transform: uppercase; color: var(--muted); }

.file-pill { display: inline-flex; align-items: center; gap: 10px; padding: 8px 16px; border-radius: 999px; background: rgba(255, 255, 255, 0.8); border: 1px solid var(--line); font-size: 13px; font-weight: 600; color: var(--violet); margin-top: 14px; }
.file-pill .dot { width: 8px; height: 8px; border-radius: 50%; background: #6fcf97; }

.workspace { margin-top: 16px; padding: 16px 22px; border-radius: 20px; background: linear-gradient(120deg, #2f1a5c, #5b2fb0 60%, #9a62dc); color: #fff; }
.workspace .eyebrow { color: #e9dfff; margin: 0 0 6px; }
.workspace .ws-file { font-family: 'Playfair Display', serif; font-size: 21px; color: #fff; }

.tech-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 14px 0 6px; }
.tech { padding: 18px 20px; border-radius: 18px; background: rgba(255, 255, 255, 0.72); border: 1px solid var(--line); transition: transform 0.3s ease, box-shadow 0.3s ease; }
.tech:hover { transform: translateY(-4px); box-shadow: var(--shadow); }
.tech b { display: block; font-family: 'Playfair Display', serif; font-size: 20px; color: var(--purple); margin-bottom: 4px; }
.tech span { font-size: 13px; color: var(--muted); line-height: 1.6; }

.about-text { color: var(--muted) !important; font-size: 15.5px; line-height: 1.9; max-width: 680px; }
.footer { border-top: 1px solid var(--line); padding-top: 24px; margin-top: 48px; color: #85808f; font-size: 12px; line-height: 1.8; }
.footer b { font-family: 'Playfair Display', serif; font-size: 16px; color: var(--violet); }

.stButton > button { background: linear-gradient(135deg, #4b2a8f 0%, #7c4dd6 60%, #b07ae8 100%); background-size: 160% 160%; background-position: 0% 50%; color: #fff; border: none; border-radius: 999px; padding: 0.55rem 1.4rem; font-size: 14px; font-weight: 600; box-shadow: 0 10px 24px -10px rgba(91, 47, 176, 0.6); transition: transform 0.25s ease, box-shadow 0.25s ease, background-position 0.5s ease; }
.stButton > button p { color: #fff !important; font-weight: 600; }
.stButton > button:hover { transform: translateY(-2px); background-position: 100% 50%; box-shadow: 0 16px 28px -12px rgba(91, 47, 176, 0.7); color: #fff; }
.stButton > button:active { transform: translateY(0) scale(0.98); }
.stButton > button:focus:not(:active) { color: #fff; border: none; }

[data-testid="stFileUploader"] { background: linear-gradient(135deg, #efe6fb 0%, #fde6f0 55%, #fff1e3 100%); background-size: 200% 200%; animation: drift 14s ease infinite; border: 1px solid rgba(124, 77, 214, 0.18); border-radius: 22px; padding: 12px; box-shadow: 0 14px 34px -22px rgba(91, 47, 176, 0.5); transition: transform 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease; }
[data-testid="stFileUploader"]:hover { transform: translateY(-4px); border-color: rgba(124, 77, 214, 0.45); box-shadow: 0 26px 44px -24px rgba(91, 47, 176, 0.6); }
[data-testid="stFileUploaderDropzone"] { background: rgba(255, 255, 255, 0.6); border: 1.5px dashed #b79af0; border-radius: 18px; transition: background 0.3s ease, border-color 0.3s ease; }
[data-testid="stFileUploaderDropzone"]:hover { background: rgba(255, 255, 255, 0.88); border-color: var(--purple); }
[data-testid="stFileUploaderDropzone"] * { color: var(--ink); }
[data-testid="stFileUploaderDropzone"] small { color: var(--muted) !important; }
[data-testid="stFileUploaderDropzone"] button { background: #fff; color: var(--purple); border: 1px solid #cdb7f3; border-radius: 999px; font-weight: 600; transition: all 0.25s ease; }
[data-testid="stFileUploaderDropzone"] button:hover { background: var(--purple); color: #fff; border-color: var(--purple); transform: translateY(-1px); }
[data-testid="stFileUploaderFile"] { background: rgba(255, 255, 255, 0.75); border-radius: 12px; padding: 4px 10px; }

[data-testid="stWidgetLabel"] p { font-weight: 600; font-size: 13px; color: var(--ink); }
[data-baseweb="select"] > div { background: rgba(255, 255, 255, 0.88); border: 1px solid var(--line); border-radius: 14px; color: var(--ink); transition: border-color 0.25s ease, box-shadow 0.25s ease; }
[data-baseweb="select"] > div:hover { border-color: #b79af0; box-shadow: 0 8px 20px -12px rgba(109, 63, 192, 0.55); }
[data-baseweb="select"] > div:focus-within { border-color: var(--purple); box-shadow: 0 0 0 3px rgba(124, 77, 214, 0.16); }
[data-baseweb="select"] * { color: var(--ink); }
[data-baseweb="tag"] { background: var(--lavender) !important; border-radius: 9px !important; }
[data-baseweb="tag"] span { color: var(--violet) !important; }
[data-baseweb="popover"] [data-baseweb="menu"], [data-baseweb="popover"] ul { background: #fff; border-radius: 14px; }
[data-baseweb="popover"] li { color: var(--ink); }
[data-baseweb="popover"] li:hover { background: var(--lav-2); }

[data-testid="stSlider"] [data-baseweb="slider"] { filter: hue-rotate(266deg) saturate(0.9); }

.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { padding: 10px 18px; border-radius: 12px 12px 0 0; transition: background 0.25s ease, color 0.25s ease; }
.stTabs [data-baseweb="tab"] p { color: var(--muted); font-weight: 600; font-size: 14px; }
.stTabs [data-baseweb="tab"]:hover { background: var(--lav-2); }
.stTabs [data-baseweb="tab"]:hover p { color: var(--purple); }
.stTabs [aria-selected="true"] p { color: var(--purple) !important; }
.stTabs [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, #6d3fc0, #e58bb0); height: 3px; border-radius: 3px; }
.stTabs [data-baseweb="tab-border"] { background: transparent; }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 18px; overflow: hidden; box-shadow: var(--shadow); }
[data-testid="stArrowVegaLiteChart"], [data-testid="stVegaLiteChart"] { background: rgba(255, 255, 255, 0.78); border: 1px solid var(--line); border-radius: 20px; padding: 16px 14px 8px; box-shadow: var(--shadow); transition: transform 0.3s ease; }
[data-testid="stArrowVegaLiteChart"]:hover, [data-testid="stVegaLiteChart"]:hover { transform: translateY(-2px); }
[data-testid="stAlert"] { border-radius: 16px; border: 1px solid var(--line); }
hr { border-color: var(--line); }

@media (max-width: 1024px) {
    .main .block-container, [data-testid="stMainBlockContainer"] { padding: 1rem 1.6rem 3.5rem; }
    .hero { grid-template-columns: 1.4fr 0.8fr; }
}
@media (max-width: 900px) {
    .hero { grid-template-columns: 1fr; padding: 12px 0 8px; }
    .hero-art { display: none; }
    .top-note { display: none; }
    [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 0.6rem 1rem; }
    [data-testid="stColumn"], [data-testid="column"] { min-width: min(100%, 300px); }
}
@media (max-width: 640px) {
    html { font-size: 14px; }
    .main .block-container, [data-testid="stMainBlockContainer"] { padding: 0.8rem 1rem 3rem; }
    .brand { font-size: 20px; }
    [data-testid="stRadio"] [role="radiogroup"] { display: flex; flex-wrap: nowrap; width: 100%; overflow-x: auto; scrollbar-width: none; }
    [data-testid="stRadio"] [role="radiogroup"]::-webkit-scrollbar { display: none; }
    [data-testid="stRadio"] [role="radiogroup"] > label { flex: 0 0 auto; white-space: nowrap; padding: 7px 13px; }
    .hero-title { font-size: clamp(30px, 9vw, 40px); letter-spacing: -1px; }
    .hero-description { font-size: 14px; }
    .hero-tags { margin-top: 12px; }
    .section-heading { font-size: 25px; }
    .sub-heading { font-size: 19px; }
    .stat { flex: 1 1 45%; padding: 12px 14px; }
    .stat-value { font-size: 22px; }
    .step { padding: 12px 14px; }
    .workspace .ws-file { font-size: 18px; }
    [data-testid="stFileUploader"] { padding: 8px; border-radius: 18px; }
    .tech-grid { grid-template-columns: 1fr 1fr; }
    .stTabs [data-baseweb="tab"] { padding: 8px 12px; }
    .stButton > button { width: 100%; }
    .footer { margin-top: 36px; }
}
@media (max-width: 380px) { .tech-grid { grid-template-columns: 1fr; } .hero-tags span { padding: 4px 10px; } }
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

nav1, nav2, nav3 = st.columns([2, 4.4, 1.6])
with nav1:
    render_html('<div class="brand"><span class="brand-mark">✳</span> DataMind<span class="brand-ai"> AI</span></div>')
with nav2:
    page = st.radio("Navigation", NAV_PAGES, horizontal=True, key="page", label_visibility="collapsed")
with nav3:
    render_html('<div class="top-note">Data Intelligence</div>')
render_html('<div class="nav-rule"></div>')


# --------------------------------------------------
# HOME PAGE
# --------------------------------------------------

if page == "Home":
    render_html("""
    <section class="hero">
        <div class="hero-copy reveal">
            <span class="eyebrow-pill"><i></i>Your personal data laboratory</span>
            <div class="hero-title">Make your data <em>speak.</em></div>
            <div class="hero-description">
                Raw data holds valuable information. DataMind AI helps you explore patterns, understand datasets and discover insights through intelligent analysis.
            </div>
            <div class="hero-tags"><span>Explore</span><span>Discover</span><span>Predict</span></div>
        </div>
        <div class="hero-art reveal d2">
            <div class="orb"></div>
            <div class="glass">
                <svg viewBox="0 0 360 260" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <defs>
                        <linearGradient id="g1" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#6d3fc0"/><stop offset="1" stop-color="#e58bb0"/></linearGradient>
                        <linearGradient id="g2" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#b79af0"/><stop offset="1" stop-color="#f3b3cc"/></linearGradient>
                    </defs>
                    <path class="wave" d="M10 200 C 60 100, 110 240, 170 140 S 280 40, 350 90" stroke="url(#g1)" stroke-width="3.5" stroke-linecap="round"/>
                    <path class="wave w2" d="M10 230 C 70 180, 120 250, 190 190 S 290 130, 350 150" stroke="url(#g2)" stroke-width="2.5" stroke-linecap="round"/>
                    <path class="wave w3" d="M10 160 C 50 120, 100 170, 160 100 S 270 30, 350 50" stroke="#d9cdef" stroke-width="2" stroke-linecap="round"/>
                    <circle cx="170" cy="140" r="6" fill="#6d3fc0"/><circle cx="350" cy="90" r="6" fill="#e58bb0"/><circle cx="190" cy="190" r="4" fill="#b79af0"/>
                </svg>
            </div>
        </div>
    </section>
    """)

    left, right = st.columns([1.3, 1], gap="large")
    with left:
        page_header("Get started", "Start with your data.", "Upload a CSV dataset to begin. Your data will be inspected and made available for exploration and machine learning.")
        
        uploaded_file = st.file_uploader("Choose your dataset", type=["csv"], key="home_upload", label_visibility="collapsed")
        
        if uploaded_file:
            signature = (uploaded_file.name, uploaded_file.size)
            if st.session_state.get("upload_sig") != signature:
                try:
                    new_df = pd.read_csv(uploaded_file)
                    if new_df.empty:
                        st.error("This dataset is empty.")
                    else:
                        st.session_state.dataset = new_df
                        st.session_state.filename = uploaded_file.name
                        st.session_state.model_results = None
                        st.session_state.upload_sig = signature
                except Exception as error:
                    st.error(f"Could not read the file: {error}")
                    
            if st.session_state.get("upload_sig") == signature and st.session_state.dataset is not None:
                loaded_df = st.session_state.dataset
                st.success("Dataset successfully loaded!")
                st.write(f"**{len(loaded_df):,} rows** · **{len(loaded_df.columns)} columns**")
                if st.button("Explore my dataset →"):
                    st.session_state.go_explore = True
                    st.rerun()

    with right:
        render_html("""
        <div class="reveal d1">
            <div class="step"><div class="step-num">01</div><div><div class="number">Explore</div><div class="feature-title">Understand your data</div><div class="feature-description">Inspect columns, records, missing values and statistical summaries.</div></div></div>
            <div class="step"><div class="step-num">02</div><div><div class="number">Discover</div><div class="feature-title">Find hidden patterns</div><div class="feature-description">Explore distributions, relationships and correlations through visualisations.</div></div></div>
            <div class="step"><div class="step-num">03</div><div><div class="number">Predict</div><div class="feature-title">Build intelligence</div><div class="feature-description">Train a machine learning model and evaluate its predictions.</div></div></div>
        </div>
        """)

    if st.session_state.dataset is not None:
        render_html(f'<div class="workspace reveal"><div class="eyebrow">Your current workspace</div><div class="ws-file">{escape(str(st.session_state.filename))}</div></div>')
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        if st.button("Open current dataset"):
            st.session_state.go_explore = True
            st.rerun()


# --------------------------------------------------
# DATASET EXPLORER
# --------------------------------------------------

elif page == "Explore Data":
    page_header("Data exploration", "Get to know your data.", "Explore your dataset, inspect its structure and understand the information it contains.")
    
    if st.session_state.dataset is None:
        st.info("No dataset found. Go to Home and upload a CSV file.")
    else:
        df = st.session_state.dataset
        render_html(f'<div class="file-pill reveal"><span class="dot"></span>{escape(str(st.session_state.filename))}</div>')
        
        stat_strip([
            ("Records", f"{df.shape[0]:,}"),
            ("Features", df.shape[1]),
            ("Missing cells", f"{int(df.isnull().sum().sum()):,}"),
            ("Duplicates", f"{int(df.duplicated().sum()):,}"),
        ])

        tab_records, tab_feature, tab_corr, tab_schema = st.tabs(["Records", "Feature explorer", "Correlations", "Column information"])

        with tab_records:
            sub_heading("Preview your records")
            number_of_rows = len(df) if len(df) <= 5 else st.slider("Number of rows to display", min_value=5, max_value=min(50, len(df)), value=min(10, len(df)))
            show_df(df.head(number_of_rows), hide_index=True)

        with tab_feature:
            sub_heading("Explore individual features")
            selected_column = st.selectbox("Choose a column", df.columns.tolist())
            selected_data = df[selected_column]
            
            stat_strip([
                ("Data type", str(selected_data.dtype)),
                ("Unique values", f"{selected_data.nunique():,}"),
                ("Missing values", f"{int(selected_data.isnull().sum()):,}"),
            ])

            if pd.api.types.is_numeric_dtype(selected_data):
                sub_heading("Statistical summary")
                show_df(selected_data.describe().to_frame())
                sub_heading("Distribution")
                if selected_data.nunique() <= 20:
                    counts = selected_data.dropna().value_counts().sort_index()
                    bar_chart(counts.index, counts.values, x_title=str(selected_column), y_title="Count")
                else:
                    histogram_chart(selected_data.dropna(), title=str(selected_column))
            else:
                sub_heading("Most common values")
                top_values = selected_data.astype(str).value_counts().head(15)
                bar_chart(top_values.index, top_values.values, x_title=str(selected_column), y_title="Count")

        with tab_corr:
            sub_heading("How features move together")
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.shape[1] < 2:
                st.info("Correlations need at least two numerical columns.")
            else:
                if numeric_df.shape[1] > 15:
                    st.caption("Showing the first 15 numerical columns for readability.")
                    numeric_df = numeric_df.iloc[:, :15]
                correlation_chart(numeric_df.corr())

        with tab_schema:
            sub_heading("Complete column information")
            column_info = pd.DataFrame({"Feature": df.columns, "Type": df.dtypes.astype(str).values, "Missing": df.isnull().sum().values, "Unique": df.nunique().values})
            show_df(column_info, hide_index=True)

        st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
        if st.button("Continue to ML Studio →"):
            st.session_state.go_ml = True
            st.rerun()


# --------------------------------------------------
# MACHINE LEARNING STUDIO (SMART + DOWNLOAD)
# --------------------------------------------------

elif page == "ML Studio":
    page_header("Machine learning laboratory", "Teach a machine to learn.", "Select a target column. The system automatically detects if it's a Regression (Number) or Classification (Category) problem.")

    if st.session_state.dataset is None:
        st.info("Upload a CSV dataset on the Home page first.")
    else:
        df = st.session_state.dataset
        sub_heading("Configure your model")
        
        target = st.selectbox("What should the model predict? (Target)", df.columns.tolist())
        feature_candidates = [col for col in df.columns if col != target]
        
        default_features = df.select_dtypes(include=np.number).columns.tolist()
        if target in default_features:
            default_features.remove(target)

        selected_features = st.multiselect("Choose input features", feature_candidates, default=default_features)
        test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20)

        if st.button("Train Machine Learning Model →"):
            if not selected_features:
                st.error("Select at least one input feature.")
            else:
                try:
                    with st.spinner("Training your model..."):
                        model_df = df[selected_features + [target]].dropna()
                        
                        if len(model_df) < 10:
                            raise ValueError("At least 10 valid records are required.")

                        X = model_df[selected_features]
                        y = model_df[target]

                        # SMART DETECTION
                        is_numeric_target = pd.api.types.is_numeric_dtype(y)
                        if is_numeric_target and y.nunique() > 15:
                            problem_type = "Regression"
                        else:
                            problem_type = "Classification"

                        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
                        categorical_features = [col for col in selected_features if col not in numeric_features]

                        transformers = []
                        if numeric_features:
                            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_features))
                        if categorical_features:
                            transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features))

                        preprocessor = ColumnTransformer(transformers=transformers)

                        if problem_type == "Regression":
                            model = Pipeline([("preprocessor", preprocessor), ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))])
                        else:
                            model = Pipeline([("preprocessor", preprocessor), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))])

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)
                        model.fit(X_train, y_train)
                        predictions = model.predict(X_test)

                        results = {
                            "model": model,
                            "problem_type": problem_type,
                            "actual": y_test,
                            "predictions": predictions,
                            "features": selected_features,
                            "target": target
                        }

                        if problem_type == "Regression":
                            results["r2"] = r2_score(y_test, predictions)
                            results["mae"] = mean_absolute_error(y_test, predictions)
                            results["rmse"] = np.sqrt(mean_squared_error(y_test, predictions))
                        else:
                            results["accuracy"] = accuracy_score(y_test, predictions)

                        st.session_state.model_results = results
                    st.success(f"Model training completed! (Detected: {problem_type})")
                except Exception as error:
                    st.error(f"Training error: {error}")

        results = st.session_state.model_results
        if results is not None:
            if results["target"] != target or results["features"] != selected_features:
                st.info("Configuration changed. Train the model again to update the results.")
            else:
                render_html('<div class="rule"></div>')
                sub_heading("Model evaluation")

                if results["problem_type"] == "Regression":
                    stat_strip([
                        ("R² Score", f"{results['r2']:.3f}"),
                        ("MAE", f"{results['mae']:.3f}"),
                        ("RMSE", f"{results['rmse']:.3f}"),
                    ])
                else:
                    stat_strip([
                        ("Accuracy", f"{results['accuracy']*100:.1f}%"),
                        ("Problem Type", results["problem_type"]),
                    ])

                tab_pred, tab_importance = st.tabs(["Actual vs predicted", "Feature importance"])

                with tab_pred:
                    sub_heading("Actual vs predicted values")
                    comparison = pd.DataFrame({"Actual": results["actual"].values, "Predicted": results["predictions"]})
                    if results["problem_type"] == "Regression":
                        actual_vs_predicted_chart(comparison["Actual"], comparison["Predicted"])
                    
                    show_df(comparison, hide_index=True)
                    
                    csv = comparison.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇ Download Predictions as CSV",
                        data=csv,
                        file_name='predictions.csv',
                        mime='text/csv',
                    )

                with tab_importance:
                    sub_heading("Feature importance")
                    fitted_preprocessor = results["model"].named_steps["preprocessor"]
                    fitted_model = results["model"].named_steps["regressor"] if results["problem_type"] == "Regression" else results["model"].named_steps["classifier"]

                    try:
                        transformed_names = fitted_preprocessor.get_feature_names_out()
                        importance = pd.DataFrame({"Feature": transformed_names, "Importance": fitted_model.feature_importances_}).sort_values("Importance", ascending=False).head(15)
                        clean_names = [str(name).split("__", 1)[-1] for name in importance["Feature"]]
                        bar_chart(clean_names, importance["Importance"].values, x_title="Importance", y_title="", horizontal=True, height=max(260, 32 * len(importance)))
                    except Exception:
                        st.info("Feature importance could not be displayed.")


# --------------------------------------------------
# ABOUT PAGE
# --------------------------------------------------

elif page == "About":
    page_header("About the project", "Data is more than numbers.", "DataMind AI is an intelligent data analysis and machine learning application designed to help users understand real-world datasets.")
    sub_heading("What DataMind AI does")
    render_html('<p class="about-text reveal">DataMind AI provides a workflow to upload, inspect, visualise and learn from structured datasets.</p>')
    
    sub_heading("Technology behind the system")
    render_html("""
    <div class="tech-grid reveal d1">
        <div class="tech"><b>Python</b><span>Core programming language</span></div>
        <div class="tech"><b>Pandas</b><span>Data manipulation</span></div>
        <div class="tech"><b>NumPy</b><span>Numerical operations</span></div>
        <div class="tech"><b>Scikit-learn</b><span>Machine learning</span></div>
        <div class="tech"><b>Streamlit</b><span>Interactive application interface</span></div>
    </div>
    """)
    
    sub_heading("Our approach")
    render_html('<p class="about-text reveal">The system works with user-provided datasets. Machine learning results are evaluated on held-out test data and should be interpreted carefully.</p>')


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

render_html("""
<div class="footer">
    <b>✳ DataMind AI</b><br>
    An intelligent space to explore data, discover patterns and experiment with machine learning.
</div>
""")