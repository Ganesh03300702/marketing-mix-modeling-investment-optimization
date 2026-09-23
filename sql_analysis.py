"""
sql_analysis.py
------------------
Loads the weekly marketing dataset into SQLite and runs SQL-based aggregation
and trend queries -- the kind of analysis that typically happens BEFORE the
Python modeling step in a real MMM workflow (pulling/aggregating spend data
from a database).

Run this AFTER generate_mmm_data.py and BEFORE mmm_pipeline.py:
    python3 data/generate_mmm_data.py
    python3 sql/sql_analysis.py
    python3 notebooks/mmm_pipeline.py
"""

import sqlite3
import pandas as pd

CSV_PATH = "project3_mmm_investment_optimization/data/weekly_marketing_data.csv"
DB_PATH = "project3_mmm_investment_optimization/data/mmm_data.db"
OUT = "project3_mmm_investment_optimization/outputs"

# ---- Load into SQLite ----
df = pd.read_csv(CSV_PATH)
conn = sqlite3.connect(DB_PATH)
df.to_sql("weekly_marketing", conn, if_exists="replace", index=False)
print(f"Loaded {len(df)} weekly rows into SQLite table `weekly_marketing`.\n")

QUERIES = {

# Total spend and revenue by channel, plus each channel's share of total spend
"channel_spend_summary": """
    SELECT
        'TV'      AS channel, ROUND(SUM(tv_spend), 2)      AS total_spend FROM weekly_marketing
    UNION ALL
    SELECT 'Digital', ROUND(SUM(digital_spend), 2) FROM weekly_marketing
    UNION ALL
    SELECT 'Social',  ROUND(SUM(social_spend), 2)  FROM weekly_marketing
    UNION ALL
    SELECT 'Print',   ROUND(SUM(print_spend), 2)   FROM weekly_marketing;
""",

# Monthly revenue trend using SQL date functions + GROUP BY
"monthly_revenue_trend": """
    SELECT
        strftime('%Y-%m', week) AS month,
        ROUND(SUM(revenue), 2)  AS monthly_revenue,
        ROUND(SUM(tv_spend + digital_spend + social_spend + print_spend), 2) AS monthly_total_spend
    FROM weekly_marketing
    GROUP BY month
    ORDER BY month;
""",

# 4-week moving average of revenue using SQL window function
"four_week_moving_avg_revenue": """
    SELECT
        week,
        revenue,
        ROUND(AVG(revenue) OVER (
            ORDER BY week ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ), 2) AS moving_avg_4wk
    FROM weekly_marketing
    ORDER BY week;
""",

# Week-over-week revenue growth % using SQL LAG() window function
"week_over_week_growth": """
    SELECT
        week,
        revenue,
        ROUND(
            (revenue - LAG(revenue) OVER (ORDER BY week)) * 100.0
            / LAG(revenue) OVER (ORDER BY week), 2
        ) AS wow_growth_pct
    FROM weekly_marketing
    ORDER BY week;
""",

# Revenue performance during promotion weeks vs non-promotion weeks
"promotion_impact": """
    SELECT
        CASE WHEN promotion_flag = 1 THEN 'Promotion Week' ELSE 'Regular Week' END AS week_type,
        COUNT(*)                AS num_weeks,
        ROUND(AVG(revenue), 2)  AS avg_revenue
    FROM weekly_marketing
    GROUP BY week_type;
""",
}

for name, query in QUERIES.items():
    result = pd.read_sql_query(query, conn)
    result.to_csv(f"{OUT}/sql_{name}.csv", index=False)
    print(f"--- {name} ---")
    print(result.head(8).to_string(index=False), "\n")

conn.close()
print(f"All SQL query outputs saved to {OUT}/")
