# RFM Customer Segmentation Studio

A Streamlit web app that turns a raw e-commerce transaction file into a full
RFM (Recency, Frequency, Monetary) customer segmentation dashboard — the same
pipeline used in the accompanying analysis notebooks, now interactive.

## What it does

1. **Upload** a raw transaction CSV (e.g. the classic "Online Retail" dataset).
2. **Validate** — checks the file has the columns RFM analysis needs.
3. **Clean** — automatically removes:
   - Exact duplicate rows
   - Rows with missing `CustomerID`
   - Unparseable dates
   - Cancelled orders (`InvoiceNo` starting with `C`)
   - Non-positive quantities / negative prices
   - Non-product line items (postage, manual fees, etc., if `StockCode` is present)
4. **Computes RFM** per customer and scores each metric 1–5 using rank-based
   quantiles (robust even on small or lumpy datasets).
5. **Segments** customers into Champions, Loyal Customers, Potential
   Loyalists, At Risk, and Lost Customers, plus a 10-group spend decile.
6. **Dashboard** — cleaning report, executive KPIs, segmentation deep-dive,
   decile analysis, a filterable customer explorer, and marketing
   recommendations per segment.

A **"Try Sample Report"** button loads a bundled, precomputed 4,340-customer
UK retail RFM report so you (or a reviewer) can explore the dashboard
immediately without needing raw data.

## Required columns in your upload

| Column | Required? |
|---|---|
| `InvoiceNo` | Yes |
| `CustomerID` | Yes |
| `InvoiceDate` | Yes |
| `Quantity` | Yes |
| `UnitPrice` | Yes |
| `Description` | Optional |
| `StockCode` | Optional (enables non-product filtering) |
| `Country` | Optional |

## Run it locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

Keep `sample_rfm_report.csv` in the same folder as `app.py` — it powers the
"Try Sample Report" button.

## Deploying

This app has no external dependencies beyond the packages in
`requirements.txt`, so it deploys as-is to **Streamlit Community Cloud**,
Render, or any host that runs `streamlit run app.py`. Just push this folder
(including `sample_rfm_report.csv`) to a GitHub repo and point Streamlit
Cloud at `app.py`.
