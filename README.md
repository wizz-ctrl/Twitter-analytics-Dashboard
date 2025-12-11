# Twitter Analytics Dashboard

A Streamlit dashboard for visualizing Twitter analytics data with interactive charts including:

- **Heatmap**: Tweet Count by Hour of Day
- **Ribbon Charts**: Tweet Count and Engagement Score by Day and Party
- **Donut Chart**: Tweet Count by Language
- **Grouped Bar Charts**: Tweet Count and Engagement Score by Party and Language

## Features

- Dark green theme
- Interactive filters (Party, Language, Date Range)
- Power BI-style ribbon charts with smooth transitions
- Responsive design

## Run Locally

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Data

The dashboard uses `dashboard_scatter_data.parquet` containing Twitter data with columns for party affiliation, language, engagement scores, and timestamps.
