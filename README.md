# Marketing Mix Modeling & Investment Optimization (Python)

**Goal:** Estimate how much of weekly revenue each marketing channel drives,
using adstock (carryover) and saturation (diminishing returns) transformations
followed by regression — then use the fitted model to evaluate a budget
reallocation scenario. Built to demonstrate direct, hands-on exposure to
Marketing Mix Modeling, econometric-style regression, time-series data, and
scenario/investment analysis.

## Honesty note (read before putting anything on your resume)
This is a **from-scratch, simplified MMM** built with `pandas` / `numpy` /
`scikit-learn` on **104 weeks of synthetic data** — not a production system, and
not validated on real business data. Every number below (R², ROI, the budget
scenario result) is what THIS model produced on THIS run — nothing is assumed
or written in advance. If you re-run it, re-report whatever numbers you
actually get.

**Resume-safe framing:**
> "Developed a Python-based Marketing Mix Modeling framework using synthetic
> weekly marketing data to analyze channel contribution, adstock, diminishing
> returns, ROI, and budget allocation scenarios."

**Do NOT write:** "Developed a production-grade MMM used for investment
decisions" or state a specific revenue-lift number as if it were a proven
business result — you'd be asked to defend a claim the project doesn't support.

## Dataset
104 weeks (2 years) of synthetic data: spend across TV, Digital, Social, Print,
a promotion flag, a seasonality index, and resulting revenue. Ground-truth
adstock decay and saturation effects are injected during generation (see
`data/generate_mmm_data.py`) specifically so the pipeline's later steps can be
checked against a known pattern — standard practice before trusting a model on
real data.

## Pipeline (`notebooks/mmm_pipeline.py`)
**Spend → Adstock → Saturation → Regression → Contribution → ROI → Budget Scenario**

0. **SQL analysis (`sql/sql_analysis.py`)** — loads the weekly data into SQLite
   and runs the kind of aggregation queries that typically precede the Python
   modeling step in a real MMM workflow: channel-wise total spend, monthly
   revenue trend, a 4-week moving average and week-over-week growth (both via
   SQL window functions — `AVG() OVER`, `LAG() OVER`), and a promotion-week vs
   regular-week revenue comparison
1. **Cleaning** — imputes missing spend values via rolling median
2. **EDA** — summary stats + raw correlation matrix saved to `outputs/`
3. **Adstock** — geometric carryover transform; decay rate per channel chosen
   via grid search against correlation with revenue (simple, explainable —
   not a black-box optimizer)
4. **Saturation** — power-curve diminishing-returns transform (α = 0.6, a
   documented, fixed assumption)
5. **Collinearity check (VIF)** — computed and reported explicitly. Channels
   whose spend moves together over time are genuinely hard for any regression
   to cleanly separate — this is a real MMM limitation, not hidden here
6. **Regression** — Ridge regression (not plain OLS) specifically *because*
   of the collinearity found in step 5; L2 penalty + a `positive=True`
   constraint (spend can't have negative effect on revenue) stabilize the fit
7. **Contribution & ROI** — per-channel revenue contribution, estimated ROI
   per $, and marginal ROI at current spend level (top-10%-of-spend-weeks
   comparison, approximating where the diminishing-returns curve currently sits)
8. **Budget scenario** — shifts 10% of the lowest-marginal-ROI channel's
   budget to the highest-marginal-ROI channel, total budget held constant, and
   re-runs the fitted model to estimate the resulting revenue change

## Key results (this run — will vary if re-run)
- **Model fit:** R² = 0.89 on a held-out 21-week test set, MAPE = 2.78%
  (R² = 0.947 refit on full data)
- **Channel contribution:** TV explains ~11.7% of total revenue (ROI ≈ $0.51
  per $1 spent), Digital ~1.8% (ROI ≈ $0.11 per $1)
- **Social and Print received a 0 coefficient** from the model — a genuine
  finding, not a bug: with only 104 weeks of data and comparatively small,
  low-variance spend on these two channels, the model couldn't statistically
  separate their effect from zero once TV/seasonality were accounted for.
  Real MMM projects hit this often; it's usually addressed with more data,
  deliberate spend variation (geo or holdout tests), or Bayesian priors that
  borrow strength across channels. **This is worth explaining exactly this
  way in an interview** — it shows you understand the model's limits.
- **Budget scenario:** shifting 10% of Social's budget to TV produced a
  model-estimated **+0.27% revenue change** — a small, model-based estimate,
  not a guaranteed real-world result.

## How to run
```bash
pip install pandas numpy matplotlib scikit-learn
python3 data/generate_mmm_data.py
python3 sql/sql_analysis.py
python3 notebooks/mmm_pipeline.py
```

## Tech stack
Python, Pandas, NumPy, scikit-learn (Ridge regression), Matplotlib, SQL (SQLite —
window functions: `LAG`, `AVG OVER`) — no black-box MMM libraries, so every
transformation is visible and explainable.

## Limitations (know these before an interview)
- Synthetic data, not validated against real business outcomes
- Saturation curve shape (α) is a fixed assumption, not fit from data
- Adstock decay chosen via simple grid search, not formally optimized
- VIF shows meaningful collinearity for TV and seasonality — a real
  constraint on how confidently individual effects can be separated
- Two channels ended up statistically indistinguishable from zero effect —
  a genuine data-limitation finding, discussed above rather than hidden
