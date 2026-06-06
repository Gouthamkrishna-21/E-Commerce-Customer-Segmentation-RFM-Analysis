# E-Commerce Customer Segmentation & RFM Analysis

> Applying RFM analysis to transactional data to drive targeted marketing strategies using Python, SQL, and Power BI.

---

## Project Overview

E-commerce businesses generate massive volumes of transactional data but often struggle to extract actionable insights from it. Without proper customer segmentation, marketing budgets get wasted on the wrong audiences — leading to poor retention and revenue loss.

This project applies the **RFM (Recency, Frequency, Monetary)** framework to segment customers, identify high-value and at-risk groups, and provide data-driven marketing recommendations backed by an interactive Power BI dashboard.

---

## Objectives

- Apply RFM analysis to segment **4,340 customers** from a UK e-commerce dataset
- Identify **Champion** customers driving the majority of revenue
- Flag **At-Risk** customers for urgent retention campaigns
- Build a **4-page interactive Power BI dashboard** for business decision-making
- Generate **segment-wise marketing recommendations**

---

## Dataset

| Attribute | Details |
|---|---|

| Timeframe | December 2010 – December 2011 |
| Raw Records | 500,000+ transactional rows |
| Key Columns | InvoiceNo, CustomerID, InvoiceDate, Quantity, UnitPrice, Country |

**Key Stats after cleaning:**

| Metric | Value |
|---|---|
| Total Customers | 4,340 |
| Total Orders | 20,726 |
| Total Products | 4,063 |
| Total Revenue | £10.64M |
| Average Order Value | £513.47 |

---

## Data Cleaning Steps

Raw data required rigorous cleaning before RFM computation:

1. **Removed Null CustomerIDs** — 24.93% of rows had no customer identifier and were dropped
2. **Dropped Duplicate Records** — 5,525 exact duplicate rows removed
3. **Excluded Cancelled Orders** — 8,872 rows with negative quantities (InvoiceNo starting with 'C') removed
4. **Removed Non-Product StockCodes** — Entries like `POST`, `CARRIAGE`, `MANUAL`, `BANK CHARGES` excluded

Result: A clean, deduplicated dataset ready for accurate RFM computation.

---

## RFM Methodology

Each customer was scored on three dimensions:

| Metric | Definition | Calculation |
|---|---|---|
| **Recency (R)** | Days since last purchase | Reference date − Last invoice date |
| **Frequency (F)** | Number of purchases | Count of unique invoices |
| **Monetary (M)** | Total spending | Sum of (Quantity × UnitPrice) |

Each metric was scored **1–5** using quantile-based binning. The combined RFM score (range: **3–15**) was used to classify customers into five segments.

---

## Customer Segments

| Segment | Customers | Revenue Contribution |
|---|---|---|
| Champions | 647 | 59.01% |
| Loyal Customers | 608 | 17.69% |
| Potential Loyalists | 1,024 | 13.36% |
| At Risk | 1,179 | 7.67% |
| Lost Customers | 881 | 2.27% |

**Key Insight:** Just 647 Champion customers (14.9% of the base) generate 59.01% of total revenue — a classic Pareto distribution.

---

## Decile Analysis

Customers were ranked by total monetary spend and divided into 10 equal groups. Findings:

- **Decile 1** — Highest spenders; contribute a disproportionately large share of revenue
- **Deciles 2–4** — Medium-value customers with strong upselling potential
- **Deciles 5–10** — Lower-value customers requiring engagement and retention strategies

This validates the **Pareto Principle**: a small percentage of customers drive the majority of business revenue.

---

## Power BI Dashboard

Four interactive dashboard pages were built:

| Page | Contents |
|---|---|
| Executive Overview | KPI cards, monthly sales trends, order volume, top countries |
| RFM Segmentation | Segment counts, average spend per segment, revenue distribution |
| Product Performance | Top 10 products by revenue and units sold |
| Customer Value Analysis | Decile matrix, customer distribution treemap, revenue by decile |

**Features:** Country and Month/Year filters · Segment and Overall Score slicers · Dynamic charts updating on selection

---

## Marketing Recommendations

| Segment | Strategy |
|---|---|
| Champions (647) | Reward with loyalty programs, early access, and exclusive offers |
| Loyal Customers (608) | Upsell and cross-sell premium products to increase average order value |
| Potential Loyalists (1,024) | Targeted emails and personalized recommendations to convert to loyal buyers |
| At Risk (1,179) | Limited-time promotions and win-back campaigns — act urgently |
| Lost Customers (881) | Surveys to understand churn reasons; re-engagement offers |

---

## Technologies Used

- **Python** — Pandas, NumPy for data cleaning and RFM computation
- **MySQL** — Data analysis and aggregations
- **Power BI** — Interactive dashboards and visualizations
- **Jupyter Notebook** — Exploratory data analysis and documentation
```
---

Author
Goutham Krishna Sikharam
B.Tech — Artificial Intelligence & Data Science (2026)
📧 gouthamsikharam2024@gmail.com

