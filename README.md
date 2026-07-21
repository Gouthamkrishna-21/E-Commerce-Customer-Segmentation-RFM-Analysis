
# E-Commerce Customer Segmentation & RFM Analysis
> Turning raw transaction data into actionable customer insight using Python, SQL, and Power BI.

## 🚀 Live App
**[Try InsightCart](https://customersegmentation-rfm-analysis.streamlit.app/)** — upload any transaction CSV and get instant RFM segmentation, no setup required.

## Overview
E-commerce businesses generate huge volumes of transaction data but often can't tell which customers actually drive revenue. This project applies **RFM (Recency, Frequency, Monetary)** analysis to segment customers, flag at-risk buyers, and recommend targeted marketing actions — available as a live app and a Power BI dashboard.

## Dataset
[Kaggle: Raw + Cleaned Data](https://www.kaggle.com/datasets/gouthamsikharam/ecommerce-raw-data-and-cleaned-data) — UK e-commerce, Dec 2010–Dec 2011, 500,000+ raw transactions.

| Metric | Value | Metric | Value |
|---|---|---|---|
| Customers | 4,340 | Revenue | £10.64M |
| Orders | 20,726 | Avg Order Value | £513.47 |

## Data Cleaning
Removed null CustomerIDs (24.93%), duplicate rows (5,525), cancelled orders (8,872), and non-product entries (postage, fees). The Streamlit app runs this same pipeline automatically on any upload.

## RFM Methodology
Each customer scored 1–5 on Recency (days since last order), Frequency (unique orders), and Monetary (total spend). Combined score (3–15) sorts customers into 5 segments.

## Segments & Recommendations
| Segment | Customers | Revenue % | Action |
|---|---|---|---|
| Champions | 647 | 59.01% | Loyalty perks, early access |
| Loyal Customers | 608 | 17.69% | Upsell & cross-sell |
| Potential Loyalists | 1,024 | 13.36% | Targeted emails to convert |
| At Risk | 1,179 | 7.67% | Urgent win-back offers |
| Lost Customers | 881 | 2.27% | Churn survey, re-engagement |

**Key insight:** 647 Champions (under 15% of customers) drive 59% of revenue — a classic Pareto pattern. Decile analysis confirms this: the top 10% of spenders account for a disproportionate share of total revenue.

## Power BI Dashboard
4-page dashboard: Executive Overview, RFM Segmentation, Product Performance, Customer Value Analysis. Includes country/date filters and segment slicers.

## Tech Stack
Python (Pandas, NumPy) · Streamlit · Plotly · MySQL · Power BI · Jupyter Notebook

## Author
**Goutham Krishna Sikharam** · B.Tech — AI & Data Science 
📧 gouthamsikharam2024@gmail.com 
