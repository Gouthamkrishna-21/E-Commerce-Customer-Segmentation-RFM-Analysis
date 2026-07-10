"""
InsightCart — E-Commerce Customer Analytics Platform
------------------------------------------------------
Upload a raw e-commerce transaction file. InsightCart validates it, cleans it
the way a real analyst would, computes RFM metrics, scores and segments
customers, and renders a full interactive dashboard with business
recommendations.
"""

import io
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ----------------------------------------------------------------------------
# PAGE CONFIG & STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightCart | E-Commerce Customer Analytics",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system: a single consistent blue / navy / white theme — deliberately
# restrained (one primary accent) rather than a rainbow of colors, per
# standard product-UI practice. Every CSS rule below is written on a single
# line with no leading whitespace: Streamlit runs this string through a
# Markdown parser before rendering, and any line indented 4+ spaces gets
# treated as a literal code block and rendered as visible text instead of
# being applied as styling.
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700;9..144,800&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root { --bg: #F4F6FA; --surface: #FFFFFF; --navy: #0F172A; --navy-soft: #1E3A8A; --ink: #0F172A; --ink-soft: #64748B; --rule: #E2E8F0; --primary: #2563EB; --primary-dark: #1D4ED8; --primary-soft: #DBEAFE; }
html, body, [data-testid="stAppViewContainer"], .main { background-color: var(--bg) !important; color: var(--ink); }
[data-testid="stSidebar"] { background-color: var(--navy) !important; border-right: none; }
[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] p { color: #94A3B8 !important; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1180px; }
h1, h2, h3 { font-family: 'Fraunces', serif !important; font-weight: 700 !important; color: var(--ink) !important; letter-spacing: -0.01em; }
p, div, span, label, li { font-family: 'Inter', sans-serif; }
.hero-box { background: linear-gradient(120deg, var(--navy) 0%, var(--navy-soft) 60%, var(--primary) 150%); border-radius: 18px; padding: 40px 44px; margin-bottom: 26px; box-shadow: 0 10px 30px rgba(15,23,42,0.18); }
.hero-box * { color: #FFFFFF !important; }
.hero-eyebrow { font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.16em; font-size: 0.74rem; color: #BFDBFE !important; font-weight: 600; margin-bottom: 10px; }
.hero-box h1 { font-size: 2.4rem !important; margin: 0 0 10px 0 !important; font-family: 'Fraunces', serif !important; }
.hero-box .sub { font-size: 1.05rem; max-width: 700px; color: #E2E8F0 !important; line-height: 1.6; }
.logo-row { display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }
.logo-icon { font-size: 1.9rem; }
.logo-text { font-family: 'Fraunces', serif !important; font-weight: 800 !important; font-size: 1.5rem; color: #FFFFFF !important; }
.sidebar-tagline { font-size: 0.8rem; color: #94A3B8 !important; margin-bottom: 14px; }
.tech-pill { display: inline-block; background: rgba(255,255,255,0.08); color: #CBD5E1 !important; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; padding: 3px 10px; border-radius: 20px; margin: 2px 4px 2px 0; }
.sidebar-version { font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; color: #64748B !important; margin-top: 14px; }
.kpi-card { background: var(--surface); border: 1px solid var(--rule); border-radius: 14px; padding: 18px 16px; text-align: center; box-shadow: 0 2px 8px rgba(15,23,42,0.05); height: 100%; }
.kpi-icon { font-size: 1.6rem; margin-bottom: 6px; }
.kpi-label { font-family: 'IBM Plex Mono', monospace; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.66rem; font-weight: 600; color: var(--ink-soft); margin-bottom: 4px; }
.kpi-value { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.5rem; color: var(--ink); }
.workflow-row { display: flex; align-items: center; justify-content: space-between; gap: 6px; flex-wrap: wrap; margin: 22px 0 8px 0; }
.workflow-step { background: var(--surface); border: 1px solid var(--rule); border-radius: 12px; padding: 16px 12px; text-align: center; flex: 1; min-width: 110px; box-shadow: 0 2px 6px rgba(15,23,42,0.04); }
.workflow-step .wf-icon { font-size: 1.5rem; margin-bottom: 6px; }
.workflow-step .wf-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; font-weight: 600; color: var(--ink); }
.workflow-arrow { font-size: 1.3rem; color: var(--primary); font-weight: 700; padding: 0 2px; }
.step-card { background: var(--surface); border: 1px solid var(--rule); border-radius: 14px; padding: 22px 20px; height: 100%; box-shadow: 0 2px 6px rgba(15,23,42,0.04); }
.step-card h4 { margin: 0 0 6px 0 !important; font-family: 'Fraunces', serif !important; color: var(--ink) !important; font-size: 1.05rem; }
.step-card p { color: var(--ink-soft); font-size: 0.9rem; margin: 0; line-height: 1.5; }
.step-num { font-family: 'IBM Plex Mono', monospace; font-weight: 700; font-size: 0.72rem; color: var(--primary); letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 4px; }
[data-testid="stMetric"] { background: var(--surface); border: 1px solid var(--rule); border-top: 4px solid var(--primary); border-radius: 10px; padding: 14px 16px 12px 16px; box-shadow: 0 2px 6px rgba(15,23,42,0.05); }
[data-testid="stMetricLabel"] { font-family: 'IBM Plex Mono', monospace !important; text-transform: uppercase; letter-spacing: 0.06em; font-size: 0.68rem !important; font-weight: 600 !important; color: var(--ink-soft) !important; }
[data-testid="stMetricValue"] { font-family: 'Fraunces', serif !important; color: var(--ink) !important; }
.segment-stamp { display:inline-flex; align-items:center; gap:6px; padding: 4px 14px 4px 12px; background: currentColor; border-radius: 20px; font-family: 'IBM Plex Mono', monospace; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.05em; text-transform: uppercase; }
.segment-stamp span { color: #FFFFFF; }
.rec-card { background: var(--surface); border: 1px solid var(--rule); border-top: 4px solid var(--primary); border-radius: 14px; padding: 20px 22px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(15,23,42,0.05); }
.rec-head { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.rec-icon { font-size: 1.4rem; }
.rec-title { font-family: 'Fraunces', serif; font-weight: 700; font-size: 1.15rem; color: var(--ink); }
.rec-desc { color: var(--ink-soft); font-size: 0.92rem; margin: 0 0 10px 0; line-height: 1.5; }
.rec-action { color: var(--ink); font-size: 0.92rem; margin: 0; line-height: 1.5; background: var(--primary-soft); padding: 8px 12px; border-radius: 8px; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 2px solid var(--rule); }
.stTabs [data-baseweb="tab"] { background-color: transparent; border-radius: 8px 8px 0 0; padding: 10px 18px; font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 600; }
.stTabs [aria-selected="true"] { background-color: var(--primary-soft) !important; color: var(--primary-dark) !important; border-bottom: 3px solid var(--primary) !important; }
[data-testid="stDataFrame"] { border: 1px solid var(--rule); border-radius: 8px; }
hr { border-color: var(--rule) !important; }
[data-testid="stButton"] button[kind="primary"] { background: var(--primary) !important; border: none !important; font-weight: 700 !important; border-radius: 10px !important; box-shadow: 0 4px 12px rgba(37,99,235,0.28); }
[data-testid="stButton"] button[kind="primary"]:hover { background: var(--primary-dark) !important; }
[data-testid="stButton"] button[kind="secondary"] { border-radius: 10px !important; font-weight: 600 !important; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] { background: var(--primary-soft) !important; border: 2px dashed var(--primary) !important; border-radius: 16px !important; }
.about-feature { font-size: 0.98rem; margin: 6px 0; color: var(--ink); }
.app-footer { text-align: center; color: var(--ink-soft); font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; padding: 26px 0 6px 0; border-top: 1px solid var(--rule); margin-top: 34px; line-height: 1.9; }
.app-footer b { color: var(--ink); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Ordered "grade" ramp: slate (least valuable) up through warning amber to
# blue/emerald (most valuable) — an ordinal scale, not an arbitrary category
# palette, consistent with the rest of the app's blue/navy chrome.
SEGMENT_COLORS = {
    "Lost Customers": "#94A3B8",
    "At Risk": "#F59E0B",
    "Potential Loyalists": "#38BDF8",
    "Loyal Customers": "#2563EB",
    "Champions": "#059669",
}
SEGMENT_ICONS = {
    "Champions": "🏆",
    "Loyal Customers": "💙",
    "Potential Loyalists": "🌱",
    "At Risk": "⚠️",
    "Lost Customers": "👋",
}
SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost Customers"]
PLOT_TEMPLATE = "simple_white"
FONT_FAMILY = "Inter, sans-serif"

REQUIRED_COLS = ["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
OPTIONAL_COLS = ["Description", "Country", "StockCode"]
NON_PRODUCT_CODES = {"POST", "CARRIAGE", "MANUAL", "BANK CHARGES", "DOT", "AMAZONFEE", "M", "PADS", "CRUK"}

# Different e-commerce exports (Shopify, WooCommerce, Amazon Seller reports,
# etc.) name the same fields differently. Rather than silently breaking on
# any file that isn't a byte-for-byte match to the original Kaggle schema,
# InsightCart auto-suggests a mapping from common aliases and lets the user
# confirm/adjust it before running the pipeline.
COLUMN_ALIASES = {
    "InvoiceNo": ["invoiceno", "invoice no", "invoice_no", "invoice number", "order id", "orderid", "order_no", "order number"],
    "CustomerID": ["customerid", "customer id", "customer_id", "client id", "clientid", "user id", "userid"],
    "InvoiceDate": ["invoicedate", "invoice date", "invoice_date", "order date", "orderdate", "order_date", "date", "transaction date", "purchase date"],
    "Quantity": ["quantity", "qty", "units", "unit count", "order quantity"],
    "UnitPrice": ["unitprice", "unit price", "unit_price", "price", "item price", "unit cost"],
    "Description": ["description", "product name", "productname", "product description", "item description", "product"],
    "Country": ["country", "shipping country", "customer country", "billing country"],
    "StockCode": ["stockcode", "stock code", "sku", "product code", "productcode", "item code"],
}


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def auto_map_columns(columns) -> dict:
    """Best-effort guess of which uploaded column corresponds to each
    canonical field. Returns {canonical_name: guessed_column_or_None}."""
    normalized_lookup = {_normalize(c): c for c in columns}
    guesses = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        guess = None
        # exact case-insensitive match on the canonical name itself first
        if _normalize(canonical) in normalized_lookup:
            guess = normalized_lookup[_normalize(canonical)]
        else:
            for alias in aliases:
                if _normalize(alias) in normalized_lookup:
                    guess = normalized_lookup[_normalize(alias)]
                    break
        guesses[canonical] = guess
    return guesses

SEGMENT_DESCRIPTIONS = {
    "Champions": "These customers generate the highest revenue — they buy most recently, most often, and spend the most.",
    "Loyal Customers": "Regular, dependable buyers who purchase often but haven't reached the top spending tier yet.",
    "Potential Loyalists": "Recent, promising customers who are still building up their purchase frequency.",
    "At Risk": "Previously active customers whose purchases have slowed or stopped in recent months.",
    "Lost Customers": "Customers who haven't purchased in a long time and show little to no recent engagement.",
}
RECOMMENDATIONS = {
    "Champions": "Offer VIP memberships, early access, and exclusive discounts to keep them engaged and rewarded.",
    "Loyal Customers": "Upsell and cross-sell premium products or bundles to raise their average order value.",
    "Potential Loyalists": "Send targeted emails and personalized recommendations to build habitual purchasing.",
    "At Risk": "Launch limited-time win-back offers and reach out proactively — act with urgency before they churn.",
    "Lost Customers": "Run churn surveys and re-engagement campaigns to understand why they left and win them back.",
}

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
for key, default in {
    "screen": "landing",   # "landing" -> "upload" -> "dashboard" / "about"
    "analysis_done": False,
    "is_sample": False,
    "rfm_df": None,
    "clean_df": None,
    "clean_stats": None,
    "reference_date": None,
    "source_name": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def reset_state():
    for key in ["analysis_done", "is_sample", "rfm_df", "clean_df", "clean_stats", "reference_date", "source_name"]:
        st.session_state[key] = False if key in ("analysis_done", "is_sample") else None
    st.session_state["screen"] = "landing"


# ----------------------------------------------------------------------------
# SMALL UI HELPERS
# ----------------------------------------------------------------------------
def kpi_card(col, icon, label, value):
    col.markdown(f"<div class='kpi-card'><div class='kpi-icon'>{icon}</div><div class='kpi-label'>{label}</div><div class='kpi-value'>{value}</div></div>", unsafe_allow_html=True)


def render_footer():
    st.markdown("<div class='app-footer'><b>InsightCart</b><br>E-Commerce Customer Analytics — Mentorship Project<br>© 2025</div>", unsafe_allow_html=True)


def render_workflow():
    steps = [("📤", "Upload CSV"), ("🧹", "Cleaning"), ("📊", "RFM Analysis"), ("📈", "Business Insights"), ("⬇️", "Download Reports")]
    parts = []
    for i, (icon, label) in enumerate(steps):
        parts.append(f"<div class='workflow-step'><div class='wf-icon'>{icon}</div><div class='wf-label'>{label}</div></div>")
        if i < len(steps) - 1:
            parts.append("<div class='workflow-arrow'>&#8594;</div>")
    st.markdown("<div class='workflow-row'>" + "".join(parts) + "</div>", unsafe_allow_html=True)


def render_tech_pills():
    pills = "".join(f"<span class='tech-pill'>{t}</span>" for t in ["Python", "Pandas", "Streamlit", "Plotly", "RFM Analysis"])
    st.markdown(pills, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _parse_csv_bytes(raw_bytes: bytes) -> pd.DataFrame:
    # Single-pass read: utf-8 with bad bytes replaced instead of silently
    # substituted characters is far faster than retrying whole encodings in
    # sequence on a large file. Only fall back to other encodings if this
    # genuinely fails (rare — bad delimiter, corrupt file, etc.).
    try:
        df = pd.read_csv(io.BytesIO(raw_bytes), encoding="utf-8", encoding_errors="replace", low_memory=False)
    except Exception:
        df = None
    if df is None:
        for enc in ("ISO-8859-1", "latin1"):
            try:
                df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, low_memory=False)
                break
            except Exception:
                continue
    if df is None:
        raise ValueError("Could not parse this file as CSV. Please check the format/encoding.")

    # Downcast numeric columns where present — smaller dtypes mean faster
    # groupby/agg later on large files, at no cost to accuracy for this data.
    if "Quantity" in df.columns:
        df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce", downcast="integer")
    if "UnitPrice" in df.columns:
        df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce", downcast="float")
    return df


def read_any_csv(uploaded_file) -> pd.DataFrame:
    # Cached on the raw bytes (cheap to hash) rather than on a DataFrame
    # (expensive to hash) — this file gets re-parsed on every mapping-UI
    # interaction, so this cache is what keeps a large upload from being
    # re-read from scratch on every selectbox click.
    return _parse_csv_bytes(uploaded_file.getvalue())


def validate_columns(df: pd.DataFrame):
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLS if c not in present]
    found_optional = [c for c in OPTIONAL_COLS if c in present]
    return missing, found_optional


def clean_pipeline(df: pd.DataFrame):
    stats = {"initial_rows": len(df), "columns": df.shape[1], "missing_values": int(df.isnull().sum().sum())}
    df = df.copy()

    # 1. Exact duplicate rows
    stats["duplicate_rows"] = int(df.duplicated().sum())
    df = df.drop_duplicates()

    # 2. Null CustomerIDs (can't attribute revenue -> excluded from RFM)
    stats["null_customer_id"] = int(df["CustomerID"].isnull().sum())
    df = df.dropna(subset=["CustomerID"])

    if "Description" in df.columns:
        df["Description"] = df["Description"].fillna("Unknown")

    # 3. Parse dates
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    stats["invalid_dates"] = int(df["InvoiceDate"].isnull().sum())
    df = df.dropna(subset=["InvoiceDate"])
    if len(df):
        stats["date_min"] = df["InvoiceDate"].min()
        stats["date_max"] = df["InvoiceDate"].max()
    else:
        stats["date_min"] = None
        stats["date_max"] = None

    # 4. Cancelled orders (InvoiceNo starting with 'C')
    invoice_str = df["InvoiceNo"].astype(str)
    cancelled_mask = invoice_str.str.startswith("C")
    stats["cancelled_orders"] = int(cancelled_mask.sum())
    df = df[~cancelled_mask]

    # 5. Non-positive quantity / negative price
    stats["bad_quantity"] = int((df["Quantity"] <= 0).sum())
    stats["bad_price"] = int((df["UnitPrice"] < 0).sum())
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] >= 0)]

    # 6. Non-product stock codes (postage, fees, manual adjustments, etc.)
    if "StockCode" in df.columns:
        code_mask = df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)
        stats["non_product_rows"] = int(code_mask.sum())
        df = df[~code_mask]
    else:
        stats["non_product_rows"] = 0

    # 7. Revenue column
    df["TotalSales"] = (df["Quantity"] * df["UnitPrice"]).round(2)

    stats["final_rows"] = len(df)
    stats["unique_customers"] = int(df["CustomerID"].nunique())
    stats["unique_invoices"] = int(df["InvoiceNo"].nunique())
    return df, stats


def _safe_qcut_rank(series: pd.Series, q: int, labels: list, ascending: bool = True) -> pd.Series:
    """Rank-based qcut that never fails on duplicate values / small samples."""
    n = series.notna().sum()
    if n == 0:
        return pd.Series([labels[0]] * len(series), index=series.index)
    eff_q = min(q, n)
    ranked = series.rank(method="first", ascending=ascending)
    try:
        eff_labels = labels[:eff_q]
        return pd.qcut(ranked, q=eff_q, labels=eff_labels)
    except Exception:
        mid = labels[len(labels) // 2]
        return pd.Series([mid] * len(series), index=series.index)


def compute_rfm(df: pd.DataFrame):
    reference_date = df["InvoiceDate"].max() + timedelta(days=1)
    rfm = (
        df.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("TotalSales", "sum"),
        )
        .reset_index()
    )
    rfm["Monetary"] = rfm["Monetary"].round(2)
    return rfm, reference_date


def score_and_segment(rfm: pd.DataFrame):
    rfm = rfm.copy()

    # Lower recency = better -> ascending rank, labels 5..1
    rfm["R_Score"] = _safe_qcut_rank(rfm["Recency"], 5, [5, 4, 3, 2, 1], ascending=True).astype(int)
    # Higher frequency/monetary = better -> ascending rank, labels 1..5
    rfm["F_Score"] = _safe_qcut_rank(rfm["Frequency"], 5, [1, 2, 3, 4, 5], ascending=True).astype(int)
    rfm["M_Score"] = _safe_qcut_rank(rfm["Monetary"], 5, [1, 2, 3, 4, 5], ascending=True).astype(int)

    rfm["Overall_Score"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]

    rfm["Segment"] = pd.cut(
        rfm["Overall_Score"],
        bins=[2, 5, 8, 11, 13, 15],
        labels=["Lost Customers", "At Risk", "Potential Loyalists", "Loyal Customers", "Champions"],
    ).astype(str)
    rfm.loc[~rfm["Segment"].isin(SEGMENT_ORDER), "Segment"] = "Potential Loyalists"

    rfm = rfm.sort_values("Monetary", ascending=False)
    n = len(rfm)
    q = min(10, n) if n > 0 else 1
    rank_desc = rfm["Monetary"].rank(ascending=False, method="first")
    try:
        rfm["Decile"] = pd.qcut(rank_desc, q=q, labels=[f"Decile {i}" for i in range(1, q + 1)])
    except Exception:
        rfm["Decile"] = "Decile 1"

    return rfm.reset_index(drop=True)


def load_precomputed_rfm(rfm_df: pd.DataFrame):
    """For the bundled sample report which already has scores/segments/deciles."""
    rfm_df = rfm_df.copy()
    rfm_df["Segment"] = rfm_df["Segment"].astype(str)
    return rfm_df


def find_sample_file() -> str:
    """Look for the bundled sample RFM csv in a few likely spots so this
    works whether app.py sits at repo root or in a subfolder like streamlit_app/."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "sample_rfm_report.csv"),
        os.path.join(here, "RFM_Analysis.csv"),
        os.path.join(here, "..", "RFM_Analysis.csv"),
        os.path.join(here, "..", "sample_rfm_report.csv"),
        "sample_rfm_report.csv",
        "RFM_Analysis.csv",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "Couldn't find sample_rfm_report.csv or RFM_Analysis.csv near app.py. "
        "Place one of these files alongside app.py or in the repo root."
    )


# ----------------------------------------------------------------------------
# SIDEBAR — logo, nav, tech stack, version
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='logo-row'><span class='logo-icon'>🛒</span><span class='logo-text'>InsightCart</span></div><div class='sidebar-tagline'>E-Commerce Customer<br>Analytics Platform</div>", unsafe_allow_html=True)
    st.markdown("---")

    nav_items = [
        ("landing", "🏠 Home"),
        ("upload", "📂 Upload Dataset"),
        ("dashboard", "📊 Dashboard"),
        ("about", "ℹ️ About"),
    ]
    for key, label in nav_items:
        is_current = st.session_state.screen == key
        if st.button(("▶ " if is_current else "") + label, use_container_width=True, key=f"nav_{key}"):
            if key == "dashboard" and not st.session_state.analysis_done:
                st.session_state.screen = "upload"
                st.session_state["_needs_data_notice"] = True
            else:
                st.session_state.screen = key
            st.rerun()

    if st.session_state.analysis_done:
        st.markdown("---")
        st.caption(f"Loaded: {st.session_state.source_name}")
        if st.button("🔄 Start Over / New Dataset", use_container_width=True):
            reset_state()
            st.rerun()

    st.markdown("---")
    render_tech_pills()
    st.markdown("<div class='sidebar-version'>Version 1.0</div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# SCREEN — HOME
# ----------------------------------------------------------------------------
if st.session_state.screen == "landing":
    st.markdown(
        "<div class='hero-box'>"
        "<div class='hero-eyebrow'>🛒 InsightCart</div>"
        "<h1>Find Out Which Customers Actually Drive Your Revenue</h1>"
        "<div class='sub'>E-commerce businesses generate huge volumes of transaction data but "
        "without proper segmentation, marketing budgets get spent on the wrong customers. "
        "InsightCart applies RFM (Recency, Frequency, Monetary) analysis to a raw sales file and tells "
        "you exactly who your Champions are, who's at risk of churning, and what to do about each group.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    lcol1, lcol2, lcol3 = st.columns([1, 1, 2])
    with lcol1:
        if st.button("📂 Upload Dataset", type="primary", use_container_width=True):
            st.session_state.screen = "upload"
            st.rerun()
    with lcol2:
        if st.button("🧪 Try a Sample", use_container_width=True):
            st.session_state.screen = "upload"
            st.rerun()

    st.write("")
    st.markdown("#### Why this exists")
    w1, w2, w3 = st.columns(3)
    with w1:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-coral'>🏆</div>"
            "<h4>Find your Champions</h4>"
            "<p>In a real 4,340-customer UK retail dataset, just 647 customers "
            "(under 15% of the base) generated 59% of total revenue — a classic Pareto pattern most "
            "businesses never see without this kind of analysis.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with w2:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-gold'>⚠️</div>"
            "<h4>Catch churn before it happens</h4>"
            "<p>Customers don't announce they're leaving — their orders just slow down. RFM scoring "
            "flags At-Risk and Lost segments early enough to run a win-back campaign, not a post-mortem.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with w3:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-teal'>🎯</div>"
            "<h4>Get an action, not just a chart</h4>"
            "<p>Every segment comes with a concrete marketing recommendation — VIP perks for Champions, "
            "urgent win-back offers for At Risk — so the analysis turns into a plan.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### What is RFM Analysis?")
    st.markdown(
        "<p style='color:var(--ink-soft); max-width:820px; font-size:0.98rem; line-height:1.6;'>"
        "RFM stands for <b>R</b>ecency, <b>F</b>requency, and <b>M</b>onetary — three simple questions "
        "answered for every customer using nothing but their order history. No surveys, no guesswork.</p>",
        unsafe_allow_html=True,
    )

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-navy'>🕒</div>"
            "<h4>Recency (R)</h4>"
            "<p>Days since a customer's <b>last purchase</b>. Fewer days = more engaged right now. "
            "Someone who bought last week is a warmer lead than someone who last bought eight months ago.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-teal'>🔁</div>"
            "<h4>Frequency (F)</h4>"
            "<p>The number of <b>separate orders</b> a customer has placed. More orders = more habitual, "
            "loyal purchasing behavior, not just one lucky sale.</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with r3:
        st.markdown(
            "<div class='step-card'>"
            "<div class='icon-badge badge-gold'>💰</div>"
            "<h4>Monetary (M)</h4>"
            "<p>The <b>total amount</b> a customer has spent (quantity × price, summed across every order). "
            "Higher spend = more revenue at stake if they leave.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='rec-action' style='margin-top:16px;'>"
        "<b>How segments get assigned:</b> each of R, F, and M is ranked 1–5 across your whole customer "
        "base (5 = best), then summed into an Overall Score from 3–15. That score sorts every customer "
        "into one of five segments — <b>Champions</b> (top scorers), <b>Loyal Customers</b>, "
        "<b>Potential Loyalists</b>, <b>At Risk</b>, and <b>Lost Customers</b> (lowest scorers)."
        "</div>",
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown("#### Workflow")
    render_workflow()

    render_footer()
    st.stop()

# ----------------------------------------------------------------------------
# SCREEN — UPLOAD & RUN
# ----------------------------------------------------------------------------
if st.session_state.screen == "upload":
    st.markdown(
        "<div class='hero-box'>"
        "<div class='hero-eyebrow'>Step 1 · Upload</div>"
        "<h1>📂 Upload Your Dataset</h1>"
        "<div class='sub'>Drop your transaction CSV below, or try the bundled sample "
        "dataset to explore the dashboard first.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("← Back to Home"):
        st.session_state.screen = "landing"
        st.rerun()

    if st.session_state.pop("_needs_data_notice", False):
        st.warning("📊 The Dashboard needs a dataset first — upload a file or try the sample below, then it'll open automatically.")

    if st.session_state.analysis_done:
        st.info(f"Currently loaded: **{st.session_state.source_name}**. Uploading a new file below will replace it.")

    st.write("")
    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"], label_visibility="collapsed")

    mapping_ok = False
    mapping = {}
    if uploaded_file is not None:
        try:
            raw_preview = read_any_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read the file: {e}")
            raw_preview = None

        if raw_preview is not None:
            guesses = auto_map_columns(raw_preview.columns)
            unmatched_required = [c for c in REQUIRED_COLS if guesses.get(c) is None]
            column_options = ["-- Not in file --"] + list(raw_preview.columns)

            with st.expander("🔧 Column Mapping (auto-detected — adjust if your file uses different headers)", expanded=bool(unmatched_required)):
                if unmatched_required:
                    st.warning(f"Couldn't auto-match: **{', '.join(unmatched_required)}**. Please select them manually below.")
                else:
                    st.caption("All required fields were auto-matched. Double-check them, or adjust for a differently-named export.")

                mcol1, mcol2 = st.columns(2)
                for i, canonical in enumerate(REQUIRED_COLS + OPTIONAL_COLS):
                    target_col = mcol1 if i % 2 == 0 else mcol2
                    default = guesses.get(canonical)
                    default_idx = column_options.index(default) if default in column_options else 0
                    tag = "required" if canonical in REQUIRED_COLS else "optional"
                    mapping[canonical] = target_col.selectbox(f"{canonical} ({tag})", column_options, index=default_idx, key=f"map_{canonical}")

            selected_cols = [v for v in mapping.values() if v != "-- Not in file --"]
            has_duplicates = len(selected_cols) != len(set(selected_cols))
            if has_duplicates:
                st.error("❌ Each column can only be mapped to one field — you've mapped the same source column to two fields.")

            mapping_ok = all(mapping[c] != "-- Not in file --" for c in REQUIRED_COLS) and not has_duplicates

    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=not mapping_ok)

    st.markdown("<div style='text-align:center; color:var(--ink-soft); margin: 10px 0;'>— or —</div>", unsafe_allow_html=True)
    sample_clicked = st.button("🧪 Try Sample Report (4,340-customer UK retail dataset)", use_container_width=True)

    with st.expander("📋 What columns does my file need?"):
        st.caption("Different names? No problem — the mapper above auto-detects common aliases (e.g. \"Order ID\" → InvoiceNo, \"Order Date\" → InvoiceDate) and lets you fix anything it misses.")
        cola, colb = st.columns(2)
        with cola:
            st.write("**Required:**")
            for c in REQUIRED_COLS:
                st.markdown(f"- `{c}`")
        with colb:
            st.write("**Optional (unlocks extra checks & breakdowns):**")
            for c in OPTIONAL_COLS:
                st.markdown(f"- `{c}`")

    if sample_clicked:
        with st.spinner("Loading sample report..."):
            sample_df = pd.read_csv(find_sample_file())
            st.session_state.rfm_df = load_precomputed_rfm(sample_df)
            st.session_state.clean_df = None
            st.session_state.clean_stats = None
            st.session_state.is_sample = True
            st.session_state.analysis_done = True
            st.session_state.source_name = "Sample UK Retail Dataset (precomputed)"
            st.session_state.screen = "dashboard"
        st.rerun()

    if run_clicked and uploaded_file is not None:
        try:
            raw_df = read_any_csv(uploaded_file)
        except Exception as e:
            st.error(f"Couldn't read the file: {e}")
            st.stop()

        # Apply the confirmed column mapping. Drop any pre-existing column
        # that already happens to share a canonical name but wasn't the one
        # selected for it, so the rename below can't create duplicate columns.
        rename_map = {v: k for k, v in mapping.items() if v != "-- Not in file --"}
        for canonical in COLUMN_ALIASES:
            if canonical in raw_df.columns and mapping.get(canonical) != canonical:
                raw_df = raw_df.drop(columns=[canonical])
        raw_df = raw_df.rename(columns=rename_map)

        missing, found_optional = validate_columns(raw_df)
        if missing:
            st.error(
                f"❌ This file is missing required column(s): **{', '.join(missing)}**. "
                f"RFM analysis needs {', '.join(REQUIRED_COLS)}. Please check your file and try again."
            )
            st.dataframe(raw_df.head(10), use_container_width=True)
            st.stop()

        progress = st.progress(0, text="Uploading dataset...")
        try:
            progress.progress(20, text="Uploading dataset...")
            progress.progress(45, text="Cleaning dataset...")
            clean_df, stats = clean_pipeline(raw_df)
            progress.progress(70, text="Generating RFM scores...")
            rfm, ref_date = compute_rfm(clean_df)
            progress.progress(90, text="Scoring & segmenting customers...")
            rfm = score_and_segment(rfm)
            progress.progress(100, text="Done!")
        except Exception as e:
            st.error(f"Something went wrong while processing this dataset: {e}")
            st.stop()

        if len(rfm) == 0:
            st.error("After cleaning, no valid customer transactions remained. Please check your data.")
            st.stop()

        st.session_state.clean_df = clean_df
        st.session_state.clean_stats = stats
        st.session_state.rfm_df = rfm
        st.session_state.reference_date = ref_date
        st.session_state.is_sample = False
        st.session_state.analysis_done = True
        st.session_state.source_name = uploaded_file.name
        st.session_state.screen = "dashboard"
        st.rerun()

    render_footer()
    st.stop()

# ----------------------------------------------------------------------------
# SCREEN — ABOUT
# ----------------------------------------------------------------------------
if st.session_state.screen == "about":
    st.markdown(
        "<div class='hero-box'>"
        "<div class='hero-eyebrow'>About This Project</div>"
        "<h1>ℹ️ About InsightCart</h1>"
        "<div class='sub'>InsightCart is an E-Commerce Customer Analytics Platform that turns "
        "raw transaction data into segmented, actionable customer insight.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    if st.button("← Back to Home"):
        st.session_state.screen = "landing"
        st.rerun()

    st.markdown("#### Features")
    features = ["Automated Data Cleaning", "RFM Analysis (Recency, Frequency, Monetary)", "Customer Segmentation", "Business Recommendations per Segment", "Interactive Dashboards & Charts", "Downloadable CSV Reports"]
    for f in features:
        st.markdown(f"<div class='about-feature'>✔ {f}</div>", unsafe_allow_html=True)

    st.markdown("#### Developed Using")
    render_tech_pills()

    render_footer()
    st.stop()

# ----------------------------------------------------------------------------
# SCREEN — DASHBOARD (safety checks first)
# ----------------------------------------------------------------------------
if not st.session_state.analysis_done:
    st.session_state["_needs_data_notice"] = True
    st.session_state.screen = "upload"
    st.rerun()
if st.session_state.screen != "dashboard":
    st.session_state.screen = "landing"
    st.rerun()

rfm = st.session_state.rfm_df.copy()
rfm["Segment"] = pd.Categorical(rfm["Segment"], categories=SEGMENT_ORDER, ordered=True)

source_label = "Sample UK Retail Dataset (precomputed)" if st.session_state.is_sample else st.session_state.source_name
total_customers_hero = len(rfm)
total_revenue_hero = rfm["Monetary"].sum()

st.markdown(
    "<div class='hero-box'>"
    "<div class='hero-eyebrow'>📊 Customer Dashboard</div>"
    f"<h1>{total_customers_hero:,} customers, scored</h1>"
    f"<div class='sub'>Drawn from <b>{source_label}</b> — £{total_revenue_hero:,.0f} in "
    "total spend, sorted from Champion to Lost below.</div>"
    "</div>",
    unsafe_allow_html=True,
)

tabs = st.tabs(["🧹 Data Cleaning", "🏠 Overview", "🌍 Trends & Geography", "📦 Product Performance", "🎯 Segmentation", "🔟 Decile Analysis", "🔍 Customer Explorer", "💡 Recommendations"])

# ---- TAB: Data Cleaning ------------------------------------------------
with tabs[0]:
    st.subheader("Data Cleaning Report")
    if st.session_state.is_sample:
        st.info("This is a precomputed sample report, so there's no raw-file cleaning step to show. "
                "Upload your own transaction file to see the live cleaning pipeline in action.")
    else:
        stats = st.session_state.clean_stats

        st.markdown("#### Dataset Information")
        d1, d2, d3, d4, d5 = st.columns(5)
        date_range_label = "—"
        if stats.get("date_min") is not None and stats.get("date_max") is not None:
            date_range_label = f"{stats['date_min'].year}–{stats['date_max'].year}"
        kpi_card(d1, "📥", "Rows", f"{stats['initial_rows']:,}")
        kpi_card(d2, "📋", "Columns", f"{stats['columns']:,}")
        kpi_card(d3, "🕳️", "Missing Values", f"{stats['missing_values']:,}")
        kpi_card(d4, "📑", "Duplicates", f"{stats['duplicate_rows']:,}")
        kpi_card(d5, "📅", "Date Range", date_range_label)

        st.markdown("#### After Cleaning")
        c1, c2, c3, c4 = st.columns(4)
        kpi_card(c1, "✅", "Rows After Cleaning", f"{stats['final_rows']:,}")
        kpi_card(c2, "👥", "Unique Customers", f"{stats['unique_customers']:,}")
        kpi_card(c3, "🧾", "Unique Invoices", f"{stats['unique_invoices']:,}")
        pct_kept_val = round(100 * stats["final_rows"] / stats["initial_rows"], 1) if stats["initial_rows"] else 0
        kpi_card(c4, "📈", "Rows Kept", f"{pct_kept_val}%")

        st.markdown("#### Rows removed, by reason")
        removal_rows = [
            ("Exact duplicate rows", stats["duplicate_rows"]),
            ("Missing CustomerID", stats["null_customer_id"]),
            ("Invalid / unparseable dates", stats["invalid_dates"]),
            ("Cancelled orders (Invoice starts with 'C')", stats["cancelled_orders"]),
            ("Non-positive quantity or negative price", stats["bad_quantity"] + stats["bad_price"]),
            ("Non-product line items (postage, fees, etc.)", stats["non_product_rows"]),
        ]
        removal_df = pd.DataFrame(removal_rows, columns=["Reason", "Rows Removed"])
        fig = px.bar(removal_df, x="Rows Removed", y="Reason", orientation="h", text="Rows Removed")
        fig.update_traces(marker_color="#2563EB", textposition="outside")
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="",
            template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.success(f"✅ Kept {pct_kept_val}% of rows after cleaning — {stats['final_rows']:,} clean transaction lines ready for RFM analysis.")

        with st.expander("Preview cleaned dataset"):
            st.dataframe(st.session_state.clean_df.head(50), use_container_width=True)

# ---- TAB: Overview ------------------------------------------------------
with tabs[1]:
    st.subheader("Executive Overview")

    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_monetary = rfm["Monetary"].mean()
    avg_frequency = rfm["Frequency"].mean()
    avg_recency = rfm["Recency"].mean()

    o1, o2, o3, o4, o5 = st.columns(5)
    kpi_card(o1, "👥", "Total Customers", f"{total_customers:,}")
    kpi_card(o2, "💰", "Total Revenue", f"£{total_revenue:,.0f}")
    kpi_card(o3, "🛍️", "Avg Spend / Customer", f"£{avg_monetary:,.0f}")
    kpi_card(o4, "🔁", "Avg Orders / Customer", f"{avg_frequency:,.1f}")
    kpi_card(o5, "⏱️", "Avg Recency (days)", f"{avg_recency:,.0f}")

    st.write("")
    col1, col2 = st.columns([1, 1])
    with col1:
        seg_counts = rfm["Segment"].value_counts().reindex(SEGMENT_ORDER).fillna(0).reset_index()
        seg_counts.columns = ["Segment", "Customers"]
        fig = px.pie(
            seg_counts, names="Segment", values="Customers", hole=0.45,
            color="Segment", color_discrete_map=SEGMENT_COLORS,
            title="Customer Mix by Segment",
        )
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        seg_rev = rfm.groupby("Segment", observed=False)["Monetary"].sum().reindex(SEGMENT_ORDER).fillna(0).reset_index()
        seg_rev["Share %"] = (seg_rev["Monetary"] / seg_rev["Monetary"].sum() * 100).round(1)
        fig2 = px.bar(
            seg_rev, x="Segment", y="Monetary", color="Segment",
            color_discrete_map=SEGMENT_COLORS, text=seg_rev["Share %"].astype(str) + "%",
            title="Revenue Contribution by Segment",
        )
        fig2.update_layout(
            showlegend=False, yaxis_title="Revenue (£)",
            template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA",
        )
        st.plotly_chart(fig2, use_container_width=True)

    top_seg = seg_rev.sort_values("Monetary", ascending=False).iloc[0]
    top_champ_count = int(seg_counts.loc[seg_counts["Segment"] == "Champions", "Customers"].values[0]) if "Champions" in seg_counts["Segment"].values else 0
    st.info(
        f"💡 **{top_seg['Segment']}** drive the largest share of revenue "
        f"({top_seg['Share %']}%). Champions alone number **{top_champ_count:,}** customers."
    )

# ---- TAB: Trends & Geography ---------------------------------------------
with tabs[2]:
    st.subheader("Trends & Geography")
    if st.session_state.is_sample:
        st.info("Monthly trends and country breakdowns need the raw transaction file (with dates "
                "and country) — the precomputed sample only includes final RFM scores. Upload your "
                "own CSV to see this here.")
    else:
        clean_df = st.session_state.clean_df
        month_df = clean_df.copy()
        month_df["Month"] = month_df["InvoiceDate"].dt.to_period("M").astype(str)
        monthly = (
            month_df.groupby("Month")
            .agg(Revenue=("TotalSales", "sum"), Orders=("InvoiceNo", "nunique"))
            .reset_index()
            .sort_values("Month")
        )

        min_date = clean_df["InvoiceDate"].min()
        max_date = clean_df["InvoiceDate"].max()
        last_month_end = (max_date.to_period("M").to_timestamp("M"))
        is_partial_last_month = max_date.normalize() < last_month_end.normalize()
        range_note = f"Showing every month in this file: **{min_date.strftime('%b %Y')} – {max_date.strftime('%b %Y')}**."
        if is_partial_last_month:
            range_note += f" The last month is partial — data stops on **{max_date.strftime('%d %b %Y')}**, not the full month, which is why it dips at the end."
        st.caption(range_note)

        tcol1, tcol2 = st.columns(2)
        with tcol1:
            fig_rev = px.line(monthly, x="Month", y="Revenue", markers=True, title="Sales by Month")
            fig_rev.update_traces(line_color="#2563EB", fill="tozeroy", fillcolor="rgba(37,99,235,0.12)")
            fig_rev.update_layout(
                template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
                plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA", yaxis_title="Revenue (£)",
            )
            st.plotly_chart(fig_rev, use_container_width=True)
        with tcol2:
            fig_ord = px.line(monthly, x="Month", y="Orders", markers=True, title="Orders by Month")
            fig_ord.update_traces(line_color="#059669", fill="tozeroy", fillcolor="rgba(5,150,105,0.12)")
            fig_ord.update_layout(
                template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
                plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA", yaxis_title="Orders",
            )
            st.plotly_chart(fig_ord, use_container_width=True)

        st.markdown("#### Top Countries by Revenue")
        if "Country" in clean_df.columns:
            country_rev = clean_df.groupby("Country")["TotalSales"].sum().sort_values(ascending=False).head(10).reset_index()
            country_rev.columns = ["Country", "Revenue"]
            fig_country = px.bar(country_rev.sort_values("Revenue"), x="Revenue", y="Country", orientation="h", text="Revenue")
            fig_country.update_traces(marker_color="#2563EB", texttemplate="£%{text:,.0f}", textposition="outside")
            fig_country.update_layout(
                height=380, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
                plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA", xaxis_title="Revenue (£)", yaxis_title="",
            )
            st.plotly_chart(fig_country, use_container_width=True)
            top_country = country_rev.iloc[0]
            st.info(f"🌍 **{top_country['Country']}** is the top market by revenue at £{top_country['Revenue']:,.0f}.")
        else:
            st.info("No `Country` column in this file — skipping the geography breakdown.")

# ---- TAB: Product Performance ----------------------------------------------
with tabs[3]:
    st.subheader("Product Performance")
    if st.session_state.is_sample:
        st.info("Product breakdowns need the raw transaction file (with product columns) — the "
                "precomputed sample only includes final RFM scores. Upload your own CSV to see this here.")
    else:
        clean_df = st.session_state.clean_df
        product_col = "Description" if "Description" in clean_df.columns else ("StockCode" if "StockCode" in clean_df.columns else None)
        if product_col:
            prod_summary = clean_df.groupby(product_col).agg(Revenue=("TotalSales", "sum"), Units=("Quantity", "sum")).reset_index()
            prod_summary["Label"] = prod_summary[product_col].astype(str).str.slice(0, 38)

            pcol1, pcol2 = st.columns(2)
            with pcol1:
                top_rev = prod_summary.sort_values("Revenue", ascending=False).head(10).sort_values("Revenue")
                fig_prod_rev = px.bar(top_rev, x="Revenue", y="Label", orientation="h", text="Revenue", title="Top 10 Products by Revenue")
                fig_prod_rev.update_traces(marker_color="#2563EB", texttemplate="£%{text:,.0f}", textposition="outside")
                fig_prod_rev.update_layout(
                    height=420, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
                    plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA", xaxis_title="Revenue (£)", yaxis_title="",
                )
                st.plotly_chart(fig_prod_rev, use_container_width=True)
            with pcol2:
                top_qty = prod_summary.sort_values("Units", ascending=False).head(10).sort_values("Units")
                fig_prod_qty = px.bar(top_qty, x="Units", y="Label", orientation="h", text="Units", title="Top 10 Products by Units Sold")
                fig_prod_qty.update_traces(marker_color="#059669", texttemplate="%{text:,.0f}", textposition="outside")
                fig_prod_qty.update_layout(
                    height=420, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
                    plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA", xaxis_title="Units Sold", yaxis_title="",
                )
                st.plotly_chart(fig_prod_qty, use_container_width=True)

            with st.expander("Full product table"):
                st.dataframe(
                    prod_summary[[product_col, "Revenue", "Units"]].sort_values("Revenue", ascending=False)
                    .style.format({"Revenue": "£{:,.2f}"}),
                    use_container_width=True, hide_index=True,
                )
        else:
            st.info("No `Description` or `StockCode` column in this file — skipping the product breakdown.")

# ---- TAB: Segmentation ---------------------------------------------------
with tabs[4]:
    st.subheader("RFM Segmentation Deep-Dive")

    seg_summary = (
        rfm.groupby("Segment", observed=False)
        .agg(
            Customers=("CustomerID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Frequency=("Frequency", "mean"),
            Avg_Monetary=("Monetary", "mean"),
            Total_Revenue=("Monetary", "sum"),
        )
        .reindex(SEGMENT_ORDER)
        .round(1)
        .reset_index()
    )
    st.dataframe(
        seg_summary.style.format({
            "Avg_Recency": "{:.0f}", "Avg_Frequency": "{:.1f}",
            "Avg_Monetary": "£{:,.0f}", "Total_Revenue": "£{:,.0f}",
        }),
        use_container_width=True, hide_index=True,
    )

    rfm_plot = rfm.copy()
    # Frequency is a small set of discrete integers, so plain scatter stacks
    # hundreds of full-opacity bubbles into solid vertical lines. Deterministic
    # jitter (seeded, so it's stable across reruns) spreads points horizontally
    # without distorting the underlying value; lower opacity + a thin outline
    # + a capped bubble size keep dense clusters readable instead of solid blobs.
    rng = np.random.default_rng(42)
    rfm_plot["Frequency_jitter"] = rfm_plot["Frequency"] + rng.uniform(-0.18, 0.18, size=len(rfm_plot))

    fig3 = px.scatter(
        rfm_plot, x="Frequency_jitter", y="Monetary", color="Segment",
        color_discrete_map=SEGMENT_COLORS, size="Overall_Score", size_max=14,
        opacity=0.55, category_orders={"Segment": SEGMENT_ORDER},
        hover_data={"Frequency_jitter": False, "CustomerID": True, "Recency": True, "Frequency": True},
        title="Frequency vs Monetary Value by Segment",
        log_y=True,
    )
    fig3.update_traces(marker=dict(line=dict(width=0.5, color="white")))
    fig3.update_layout(
        height=480, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
        plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA",
        xaxis_title="Frequency (orders)",
    )
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(
        rfm, x="Segment", y="Recency", color="Segment",
        color_discrete_map=SEGMENT_COLORS,
        category_orders={"Segment": SEGMENT_ORDER},
        title="Recency Distribution by Segment (lower = more recent)",
    )
    fig4.update_layout(
        showlegend=False, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
        plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---- TAB: Decile Analysis -------------------------------------------------
with tabs[5]:
    st.subheader("Decile Analysis")
    st.caption("Customers ranked by spend and split into 10 equal-sized groups — a classic Pareto check.")

    decile_summary = (
        rfm.groupby("Decile", observed=False)
        .agg(Customers=("CustomerID", "count"), Total_Revenue=("Monetary", "sum"))
        .reset_index()
    )
    decile_summary["Revenue Share %"] = (decile_summary["Total_Revenue"] / decile_summary["Total_Revenue"].sum() * 100).round(2)
    decile_summary["Decile_num"] = decile_summary["Decile"].astype(str).str.extract(r"(\d+)").astype(float)
    decile_summary = decile_summary.sort_values("Decile_num").drop(columns="Decile_num")

    fig5 = px.bar(
        decile_summary, x="Decile", y="Revenue Share %", text="Revenue Share %",
        title="Revenue Share by Decile",
    )
    fig5.update_traces(marker_color="#2563EB", texttemplate="%{text}%", textposition="outside")
    fig5.update_layout(template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F4F6FA", paper_bgcolor="#F4F6FA")
    st.plotly_chart(fig5, use_container_width=True)

    top_decile_share = decile_summary.iloc[0]["Revenue Share %"]
    st.success(f"📈 Decile 1 (top 10% of customers) generates **{top_decile_share}%** of total revenue.")

    st.dataframe(
        decile_summary.style.format({"Total_Revenue": "£{:,.0f}", "Revenue Share %": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

# ---- TAB: Customer Explorer -----------------------------------------------
with tabs[6]:
    st.subheader("Customer Explorer")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        seg_filter = st.multiselect("Segment", SEGMENT_ORDER, default=SEGMENT_ORDER)
    with fcol2:
        decile_options = sorted(rfm["Decile"].astype(str).unique(), key=lambda x: int(x.split()[-1]) if x.split()[-1].isdigit() else 0)
        decile_filter = st.multiselect("Decile", decile_options, default=decile_options)
    with fcol3:
        search_id = st.text_input("Search CustomerID")

    filtered = rfm[rfm["Segment"].astype(str).isin(seg_filter) & rfm["Decile"].astype(str).isin(decile_filter)]
    if search_id:
        filtered = filtered[filtered["CustomerID"].astype(str).str.contains(search_id.strip(), case=False)]

    st.write(f"Showing **{len(filtered):,}** of {len(rfm):,} customers")
    st.dataframe(
        filtered.sort_values("Monetary", ascending=False).style.format({"Monetary": "£{:,.2f}"}),
        use_container_width=True, height=420, hide_index=True,
    )

    csv_buf = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered results (CSV)", data=csv_buf, file_name="rfm_filtered_customers.csv", mime="text/csv")

# ---- TAB: Recommendations --------------------------------------------------
with tabs[7]:
    st.subheader("Segment-wise Business Recommendations")

    for seg in SEGMENT_ORDER:
        count = int((rfm["Segment"] == seg).sum())
        revenue_share = round(rfm.loc[rfm["Segment"] == seg, "Monetary"].sum() / rfm["Monetary"].sum() * 100, 1) if rfm["Monetary"].sum() else 0
        color = SEGMENT_COLORS[seg]
        icon = SEGMENT_ICONS.get(seg, "")
        st.markdown(
            f"<div class='rec-card' style='border-top-color:{color};'>"
            f"<div class='rec-head'><span class='rec-icon'>{icon}</span><span class='rec-title'>{seg}</span>"
            f"<span style='margin-left:auto; color:var(--ink-soft); font-size:0.85rem;'>{count:,} customers · {revenue_share}% of revenue</span></div>"
            f"<p class='rec-desc'>{SEGMENT_DESCRIPTIONS[seg]}</p>"
            f"<p class='rec-action'><b>Recommendation:</b> {RECOMMENDATIONS[seg]}</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.caption("Recommendations are strategic starting points based on segment behavior — pair them with your own campaign tooling.")

render_footer()
