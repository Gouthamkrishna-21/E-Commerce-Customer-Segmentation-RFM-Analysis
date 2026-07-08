"""
RFM Customer Segmentation Studio
---------------------------------
Upload a raw e-commerce transaction file, the app validates it, cleans it
(the same way a real analyst would), computes RFM metrics, scores &
segments customers, and renders a full interactive dashboard.
"""

import io
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
    page_title="RFM Customer Segmentation Studio",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design system: "The Ledger" — customers graded and stamped by value,
# like entries in a financial register. Paper tones, ink navy, a brass/gold
# accent for the most valuable grade, cooling to slate for the least.
CUSTOM_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap" rel="stylesheet">
<style>
:root {
    --paper: #F6F5F0;
    --paper-raised: #FFFFFE;
    --ink: #1B2333;
    --ink-soft: #5B6472;
    --rule: #DEDACB;
    --gold: #B8892B;
    --gold-soft: #EFE3C6;
}
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--paper) !important;
    color: var(--ink);
}
[data-testid="stSidebar"] {
    background-color: #EFEDE3 !important;
    border-right: 1px solid var(--rule);
}
.block-container {padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px;}
h1, h2, h3 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    color: var(--ink) !important;
    letter-spacing: -0.01em;
}
p, div, span, label, li {font-family: 'Inter', sans-serif;}

/* Eyebrow / ledger header */
.ledger-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.72rem;
    color: var(--gold);
    font-weight: 600;
    margin-bottom: 6px;
}
.ledger-hero {
    border-top: 2px solid var(--ink);
    border-bottom: 1px solid var(--rule);
    padding: 18px 0 22px 0;
    margin-bottom: 22px;
}
.ledger-hero h1 {font-size: 2.1rem !important; margin: 0 0 6px 0 !important;}
.ledger-hero .sub {color: var(--ink-soft); font-size: 0.95rem; max-width: 640px;}
.ledger-source {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    color: var(--ink-soft);
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    display: inline-block;
    padding: 3px 10px;
    border-radius: 3px;
    margin-top: 8px;
}

/* Metric cards -> ledger line items */
[data-testid="stMetric"] {
    background: var(--paper-raised);
    border: 1px solid var(--rule);
    border-top: 3px solid var(--gold);
    border-radius: 4px;
    padding: 12px 16px 10px 16px;
    box-shadow: 0 1px 2px rgba(27,35,51,0.05);
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase; letter-spacing: 0.06em;
    font-size: 0.68rem !important; font-weight: 600 !important; color: var(--ink-soft) !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important; color: var(--ink) !important;
}

/* Segment stamps */
.segment-stamp {
    display:inline-flex; align-items:center; gap:6px;
    padding: 3px 12px 3px 10px;
    border: 1.5px solid currentColor;
    border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.74rem; font-weight: 600;
    letter-spacing: 0.05em; text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {gap: 2px; border-bottom: 1px solid var(--rule);}
.stTabs [data-baseweb="tab"] {
    background-color: transparent; border-radius: 3px 3px 0 0; padding: 8px 16px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background-color: var(--gold-soft) !important;
    border-bottom: 2px solid var(--gold) !important;
}
[data-testid="stDataFrame"] {border: 1px solid var(--rule);}
hr {border-color: var(--rule) !important;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Sequential "grade" ramp: slate (least valuable) -> deep emerald (most valuable).
# Deliberately not a red/yellow/green traffic light — RFM segments are an
# ordered scale, not unrelated categories, so the color should read as a
# gradient of worth, like ink density on a graded ledger.
SEGMENT_COLORS = {
    "Lost Customers": "#8A93A3",
    "At Risk": "#B08947",
    "Potential Loyalists": "#B8892B",
    "Loyal Customers": "#6E8B4F",
    "Champions": "#1F5C4A",
}
SEGMENT_ORDER = ["Champions", "Loyal Customers", "Potential Loyalists", "At Risk", "Lost Customers"]
PLOT_TEMPLATE = "simple_white"
FONT_FAMILY = "Inter, sans-serif"


REQUIRED_COLS = ["InvoiceNo", "CustomerID", "InvoiceDate", "Quantity", "UnitPrice"]
OPTIONAL_COLS = ["Description", "Country", "StockCode"]
NON_PRODUCT_CODES = {"POST", "CARRIAGE", "MANUAL", "BANK CHARGES", "DOT", "AMAZONFEE", "M", "PADS", "CRUK"}

RECOMMENDATIONS = {
    "Champions": "Reward with loyalty perks, early access, and exclusive offers.",
    "Loyal Customers": "Upsell and cross-sell premium products to raise order value.",
    "Potential Loyalists": "Send targeted emails and personalized recommendations to convert them.",
    "At Risk": "Launch limited-time win-back promotions — act urgently.",
    "Lost Customers": "Run churn surveys and re-engagement offers to understand why they left.",
}

# ----------------------------------------------------------------------------
# SESSION STATE
# ----------------------------------------------------------------------------
for key, default in {
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


# ----------------------------------------------------------------------------
# CORE PIPELINE FUNCTIONS
# ----------------------------------------------------------------------------
def read_any_csv(uploaded_file) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    for enc in ("utf-8", "ISO-8859-1", "latin1"):
        try:
            return pd.read_csv(io.BytesIO(raw_bytes), encoding=enc)
        except Exception:
            continue
    raise ValueError("Could not parse this file as CSV. Please check the format/encoding.")


def validate_columns(df: pd.DataFrame):
    present = set(df.columns)
    missing = [c for c in REQUIRED_COLS if c not in present]
    found_optional = [c for c in OPTIONAL_COLS if c in present]
    return missing, found_optional


@st.cache_data(show_spinner=False)
def clean_pipeline(df: pd.DataFrame):
    stats = {"initial_rows": len(df)}
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


@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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


def run_pipeline(raw_df: pd.DataFrame):
    clean_df, stats = clean_pipeline(raw_df)
    rfm, ref_date = compute_rfm(clean_df)
    rfm = score_and_segment(rfm)
    return clean_df, stats, rfm, ref_date


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
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div class='ledger-eyebrow'>Customer Ledger</div>"
        "<h2 style='margin-top:0;'>RFM Studio</h2>",
        unsafe_allow_html=True,
    )
    st.caption("Upload transactions → clean → grade → decide.")

    uploaded_file = st.file_uploader("Upload transaction CSV", type=["csv"])

    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True, disabled=uploaded_file is None)

    st.markdown("---")
    sample_clicked = st.button("🧪 Try Sample Report", use_container_width=True)

    if st.session_state.analysis_done:
        st.markdown("---")
        if st.button("🔄 Reset / New Dataset", use_container_width=True):
            reset_state()
            st.rerun()

    st.markdown("---")
    with st.expander("Required columns"):
        st.write("Your file needs these columns (any order):")
        for c in REQUIRED_COLS:
            st.markdown(f"- `{c}`")
        st.write("Nice to have (enables extra checks):")
        for c in OPTIONAL_COLS:
            st.markdown(f"- `{c}`")

# ----------------------------------------------------------------------------
# HANDLE ACTIONS
# ----------------------------------------------------------------------------
if sample_clicked:
    sample_df = pd.read_csv(find_sample_file())
    st.session_state.rfm_df = load_precomputed_rfm(sample_df)
    st.session_state.clean_df = None
    st.session_state.clean_stats = None
    st.session_state.is_sample = True
    st.session_state.analysis_done = True
    st.session_state.source_name = "Sample UK Retail Dataset (precomputed)"

if run_clicked and uploaded_file is not None:
    try:
        raw_df = read_any_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read the file: {e}")
        st.stop()

    missing, found_optional = validate_columns(raw_df)
    if missing:
        st.error(
            f"❌ This file is missing required column(s): **{', '.join(missing)}**. "
            f"RFM analysis needs {', '.join(REQUIRED_COLS)}. Please check your file and try again."
        )
        st.dataframe(raw_df.head(10), use_container_width=True)
        st.stop()

    with st.spinner("Cleaning data and computing RFM scores..."):
        try:
            clean_df, stats, rfm, ref_date = run_pipeline(raw_df)
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


# ----------------------------------------------------------------------------
# LANDING PAGE (no analysis yet)
# ----------------------------------------------------------------------------
if not st.session_state.analysis_done:
    st.markdown(
        """
        <div class='ledger-hero'>
            <div class='ledger-eyebrow'>Est. from raw transactions — Register No. 001</div>
            <h1>The Customer Ledger</h1>
            <div class='sub'>Every transaction file tells you who your best customers are —
            it just doesn't say so in plain language. Upload one, and this ledger reads it,
            cleans it, and grades every customer from <b>Champion</b> down to <b>Lost</b>,
            with the revenue math to back each grade up.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What your file needs")
        st.markdown(
            "- `InvoiceNo` — order/invoice identifier\n"
            "- `CustomerID` — unique customer identifier\n"
            "- `InvoiceDate` — date of purchase\n"
            "- `Quantity` — units purchased\n"
            "- `UnitPrice` — price per unit\n\n"
            "Optional: `Description`, `StockCode`, `Country` — unlocks extra cleaning "
            "(non-product codes) and richer breakdowns."
        )
    with col2:
        st.markdown("### What happens automatically")
        st.markdown(
            "1. **Validate** required columns exist\n"
            "2. **Clean**: drop duplicates, missing customer IDs, cancelled orders, "
            "bad quantities/prices, non-product line items\n"
            "3. **Compute RFM** per customer\n"
            "4. **Score** R/F/M on a 1–5 scale and assign a segment\n"
            "5. **Visualize**: KPIs, segment mix, deciles, customer explorer, "
            "and marketing recommendations"
        )

    st.info("👈 Upload your CSV in the sidebar and click **Run Analysis** — or click **Try Sample Report** to explore the dashboard with a real 4,340-customer UK retail dataset first.")
    st.stop()


# ----------------------------------------------------------------------------
# DASHBOARD (analysis done)
# ----------------------------------------------------------------------------
rfm = st.session_state.rfm_df.copy()
rfm["Segment"] = pd.Categorical(rfm["Segment"], categories=SEGMENT_ORDER, ordered=True)

source_label = "Sample UK Retail Ledger (precomputed)" if st.session_state.is_sample else st.session_state.source_name
total_customers_hero = len(rfm)
total_revenue_hero = rfm["Monetary"].sum()

st.markdown(
    f"""
    <div class='ledger-hero'>
        <div class='ledger-eyebrow'>Ledger Reading</div>
        <h1>{total_customers_hero:,} customers, graded</h1>
        <div class='sub'>Register drawn from <b>{source_label}</b> — £{total_revenue_hero:,.0f} in
        total spend, sorted from Champion to Lost below.</div>
        <div class='ledger-source'>SOURCE: {source_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs(["🧹 Data Cleaning", "🏠 Overview", "🎯 Segmentation", "🔟 Decile Analysis", "🔍 Customer Explorer", "💡 Recommendations"])

# ---- TAB: Data Cleaning ------------------------------------------------
with tabs[0]:
    st.subheader("Data Cleaning Report")
    if st.session_state.is_sample:
        st.info("This is a precomputed sample report, so there's no raw-file cleaning step to show. "
                "Upload your own transaction file to see the live cleaning pipeline in action.")
    else:
        stats = st.session_state.clean_stats
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows before cleaning", f"{stats['initial_rows']:,}")
        c2.metric("Rows after cleaning", f"{stats['final_rows']:,}")
        c3.metric("Unique customers", f"{stats['unique_customers']:,}")
        c4.metric("Unique invoices", f"{stats['unique_invoices']:,}")

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
        fig.update_traces(marker_color="#8A93A3", textposition="outside")
        fig.update_layout(
            height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="", xaxis_title="",
            template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0",
        )
        st.plotly_chart(fig, use_container_width=True)

        pct_kept = round(100 * stats["final_rows"] / stats["initial_rows"], 1) if stats["initial_rows"] else 0
        st.success(f"✅ Kept {pct_kept}% of rows after cleaning — {stats['final_rows']:,} clean transaction lines ready for RFM analysis.")

        with st.expander("Preview cleaned transactions"):
            st.dataframe(st.session_state.clean_df.head(50), use_container_width=True)

# ---- TAB: Overview ------------------------------------------------------
with tabs[1]:
    st.subheader("Executive Overview")

    total_customers = len(rfm)
    total_revenue = rfm["Monetary"].sum()
    avg_monetary = rfm["Monetary"].mean()
    avg_frequency = rfm["Frequency"].mean()
    avg_recency = rfm["Recency"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Total Revenue", f"£{total_revenue:,.0f}")
    c3.metric("Avg Spend / Customer", f"£{avg_monetary:,.0f}")
    c4.metric("Avg Orders / Customer", f"{avg_frequency:,.1f}")
    c5.metric("Avg Recency (days)", f"{avg_recency:,.0f}")

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
        fig.update_layout(template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0")
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
            template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0",
        )
        st.plotly_chart(fig2, use_container_width=True)

    top_seg = seg_rev.sort_values("Monetary", ascending=False).iloc[0]
    top_champ_count = int(seg_counts.loc[seg_counts["Segment"] == "Champions", "Customers"].values[0]) if "Champions" in seg_counts["Segment"].values else 0
    st.info(
        f"💡 **{top_seg['Segment']}** drive the largest share of revenue "
        f"({top_seg['Share %']}%). Champions alone number **{top_champ_count:,}** customers."
    )

# ---- TAB: Segmentation ---------------------------------------------------
with tabs[2]:
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

    fig3 = px.scatter(
        rfm, x="Frequency", y="Monetary", color="Segment",
        color_discrete_map=SEGMENT_COLORS, size="Overall_Score",
        hover_data=["CustomerID", "Recency"],
        title="Frequency vs Monetary Value by Segment",
        log_y=True,
    )
    fig3.update_layout(
        height=480, template=PLOT_TEMPLATE, font_family=FONT_FAMILY,
        plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0",
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
        plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0",
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---- TAB: Decile Analysis -------------------------------------------------
with tabs[3]:
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
    fig5.update_traces(marker_color="#B8892B", texttemplate="%{text}%", textposition="outside")
    fig5.update_layout(template=PLOT_TEMPLATE, font_family=FONT_FAMILY, plot_bgcolor="#F6F5F0", paper_bgcolor="#F6F5F0")
    st.plotly_chart(fig5, use_container_width=True)

    top_decile_share = decile_summary.iloc[0]["Revenue Share %"]
    st.success(f"📈 Decile 1 (top 10% of customers) generates **{top_decile_share}%** of total revenue.")

    st.dataframe(
        decile_summary.style.format({"Total_Revenue": "£{:,.0f}", "Revenue Share %": "{:.2f}%"}),
        use_container_width=True, hide_index=True,
    )

# ---- TAB: Customer Explorer -----------------------------------------------
with tabs[4]:
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
with tabs[5]:
    st.subheader("Segment-wise Marketing Recommendations")

    for seg in SEGMENT_ORDER:
        count = int((rfm["Segment"] == seg).sum())
        revenue_share = round(rfm.loc[rfm["Segment"] == seg, "Monetary"].sum() / rfm["Monetary"].sum() * 100, 1) if rfm["Monetary"].sum() else 0
        color = SEGMENT_COLORS[seg]
        st.markdown(
            f"<span class='segment-stamp' style='color:{color};'>{seg}</span> "
            f"&nbsp; **{count:,} customers** &nbsp;•&nbsp; **{revenue_share}%** of revenue",
            unsafe_allow_html=True,
        )
        st.write(RECOMMENDATIONS[seg])
        st.markdown("---")

    st.caption("Recommendations are strategic starting points based on segment behavior — pair them with your own campaign tooling.")
