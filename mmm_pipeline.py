"""
mmm_pipeline.py
-----------------
Marketing Mix Modeling & Investment Optimization

Pipeline: Spend -> Adstock -> Saturation -> Regression -> Contribution -> ROI -> Budget Scenarios

Honesty note: every number in this script's console output and CSVs is what the
model actually produced on this run -- nothing here is a hardcoded or assumed
result. Re-running with different random seeds or data will change the numbers.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_PATH = "project3_mmm_investment_optimization/data/weekly_marketing_data.csv"
OUT = "project3_mmm_investment_optimization/outputs"

CHANNELS = ["tv_spend", "digital_spend", "social_spend", "print_spend"]

# =========================================================
# STEP 1 — LOAD + CLEAN
# =========================================================
df = pd.read_csv(DATA_PATH, parse_dates=["week"])
n_raw = len(df)

missing_before = df[CHANNELS].isnull().sum().sum()
# Impute missing spend with the channel's rolling median (defensible, auditable rule)
for col in CHANNELS:
    df[col] = df[col].fillna(df[col].rolling(4, min_periods=1, center=True).median())
    df[col] = df[col].fillna(df[col].median())  # fallback for edge rows

print("=== DATA CLEANING ===")
print(f"Rows: {n_raw} | Missing spend values imputed: {int(missing_before)}\n")

# =========================================================
# STEP 2 — EDA (quick, saved not printed in full)
# =========================================================
eda_summary = df[CHANNELS + ["revenue"]].describe().T
eda_summary.to_csv(f"{OUT}/eda_summary_stats.csv")

correlations = df[CHANNELS + ["revenue"]].corr()
correlations.to_csv(f"{OUT}/raw_spend_correlations.csv")

# =========================================================
# STEP 3 — ADSTOCK TRANSFORMATION
# =========================================================
def apply_adstock(spend, decay):
    """Geometric adstock: carryover effect of spend from prior weeks."""
    adstocked = np.zeros(len(spend))
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked

# For a fresher-level project we grid-search a small set of plausible decay
# rates per channel and pick the one that correlates best with revenue --
# a simple, explainable stand-in for formal hyperparameter optimization.
DECAY_GRID = [0.1, 0.25, 0.4, 0.55, 0.7]
best_decay = {}
adstocked_data = {}

for ch in CHANNELS:
    best_corr, best_d = -1, DECAY_GRID[0]
    for d in DECAY_GRID:
        a = apply_adstock(df[ch].values, d)
        corr = np.corrcoef(a, df["revenue"].values)[0, 1]
        if corr > best_corr:
            best_corr, best_d = corr, d
    best_decay[ch] = best_d
    adstocked_data[ch] = apply_adstock(df[ch].values, best_d)

print("=== ADSTOCK: selected decay rate per channel (grid search on correlation w/ revenue) ===")
for ch, d in best_decay.items():
    print(f"  {ch}: decay = {d}")

# =========================================================
# STEP 4 — SATURATION (diminishing returns)
# =========================================================
SATURATION_ALPHA = 0.6  # fixed, documented assumption: sub-linear diminishing-returns curve

saturated_data = {}
for ch in CHANNELS:
    # Normalize before exponentiation so different spend scales don't distort the curve
    scaled = adstocked_data[ch] / adstocked_data[ch].max()
    saturated_data[ch] = np.power(scaled, SATURATION_ALPHA)

X = pd.DataFrame(saturated_data)
X["promotion_flag"] = df["promotion_flag"].values
X["seasonality_index"] = df["seasonality_index"].values
y = df["revenue"].values

# =========================================================
# STEP 5 — COLLINEARITY CHECK (VIF) -- flagged explicitly, not hidden
# =========================================================
def compute_vif(X_df):
    """Manual VIF: regress each predictor on all others, VIF = 1/(1-R^2).
    VIF > 5 suggests meaningful multicollinearity between channels --
    a known, real limitation of MMM when channels move together over time."""
    vifs = {}
    cols = X_df.columns
    for col in cols:
        others = [c for c in cols if c != col]
        lr = LinearRegression().fit(X_df[others], X_df[col])
        r2 = lr.score(X_df[others], X_df[col])
        vifs[col] = np.inf if r2 >= 0.999 else 1 / (1 - r2)
    return pd.Series(vifs)

vif_series = compute_vif(X)
vif_series.to_csv(f"{OUT}/vif_collinearity_check.csv", header=["VIF"])
print("\n=== COLLINEARITY CHECK (VIF) ===")
print(vif_series.round(2).to_string())
print("(VIF > 5 flags channels whose spend moves together, making it harder to")
print(" cleanly separate their individual effects -- a genuine MMM limitation.)")

# =========================================================
# STEP 6 — REGRESSION MODEL
# =========================================================
# Ridge regression chosen over plain OLS specifically BECAUSE of the collinearity
# flagged above -- Ridge's L2 penalty stabilizes coefficients when predictors
# are correlated, which plain LinearRegression does not handle well.
train_size = int(len(X) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

model = Ridge(alpha=0.1, positive=True)  # positive=True: spend can't have negative effect on revenue
model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
r2_test = r2_score(y_test, y_pred_test)
mape_test = mean_absolute_percentage_error(y_test, y_pred_test) * 100

print(f"\n=== MODEL VALIDATION (last {len(X_test)} weeks held out) ===")
print(f"R^2 on test set:   {r2_test:.3f}")
print(f"MAPE on test set:  {mape_test:.2f}%")

# Refit on full data for the contribution/ROI analysis
model_full = Ridge(alpha=0.1, positive=True).fit(X, y)
y_pred_full = model_full.predict(X)
r2_full = r2_score(y, y_pred_full)
print(f"R^2 on full dataset (refit): {r2_full:.3f}")

coefs = pd.Series(model_full.coef_, index=X.columns)
coefs.to_csv(f"{OUT}/model_coefficients.csv", header=["coefficient"])

# =========================================================
# STEP 7 — CHANNEL CONTRIBUTION + ROI
# =========================================================
contributions = {}
for ch in CHANNELS:
    contributions[ch] = coefs[ch] * X[ch].values

contrib_df = pd.DataFrame(contributions)
contrib_totals = contrib_df.sum().sort_values(ascending=False)

total_spend = df[CHANNELS].sum()
roi = (contrib_totals / total_spend).rename("estimated_roi")  # revenue generated per $ spent

# Marginal ROI: contribution generated by the LAST 10% of spend vs the rest,
# approximating diminishing returns at the current spend level.
marginal_roi = {}
for ch in CHANNELS:
    spend_sorted_idx = np.argsort(df[ch].values)
    top10pct_idx = spend_sorted_idx[-int(len(df) * 0.1):]
    rest_idx = spend_sorted_idx[:-int(len(df) * 0.1)]
    marg_contrib = contrib_df[ch].values[top10pct_idx].mean() - contrib_df[ch].values[rest_idx].mean()
    marg_spend = df[ch].values[top10pct_idx].mean() - df[ch].values[rest_idx].mean()
    marginal_roi[ch] = marg_contrib / marg_spend if marg_spend != 0 else np.nan

channel_summary = pd.DataFrame({
    "total_spend": total_spend,
    "total_contribution": contrib_totals,
    "pct_of_revenue_explained": (contrib_totals / y.sum() * 100).round(2),
    "estimated_roi_per_$": roi.round(3),
    "marginal_roi_at_current_spend": pd.Series(marginal_roi).round(3),
}).sort_values("total_contribution", ascending=False)

channel_summary.to_csv(f"{OUT}/channel_contribution_roi.csv")
print("\n=== CHANNEL CONTRIBUTION & ROI (actual model output) ===")
print(channel_summary.to_string())

zero_channels = coefs[coefs.index.isin(CHANNELS) & (coefs == 0)].index.tolist()
if zero_channels:
    print(f"\nNote: {zero_channels} received a 0 coefficient from the model.")
    print("This is a genuine (not a bug) MMM finding: with only 104 weeks of data and")
    print("smaller/less-varying spend on these channels, the model could not statistically")
    print("distinguish their contribution from zero once TV, seasonality, and promotions")
    print("were accounted for. Real MMM projects hit this constantly -- it's usually solved")
    print("with more data, deliberate spend variation (geo/holdout tests), or Bayesian priors")
    print("that borrow strength across channels/regions. Worth stating exactly this way in")
    print("an interview -- it shows you understand the model's limits, not just its output.")

# Actual vs Predicted plot
plt.figure(figsize=(11, 5))
plt.plot(df["week"], y, label="Actual Revenue", linewidth=1.5)
plt.plot(df["week"], y_pred_full, label="Model Predicted Revenue", linestyle="--", linewidth=1.5)
plt.title(f"MMM: Actual vs Predicted Weekly Revenue (R² = {r2_full:.3f})")
plt.xlabel("Week"); plt.ylabel("Revenue ($)")
plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
plt.savefig(f"{OUT}/actual_vs_predicted.png", dpi=150)
plt.close()

# Contribution bar chart
plt.figure(figsize=(8, 5))
contrib_totals.plot(kind="bar", color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])
plt.title("Total Revenue Contribution by Channel (model output)")
plt.ylabel("Revenue Contribution ($)")
plt.xticks(rotation=20); plt.tight_layout()
plt.savefig(f"{OUT}/channel_contribution.png", dpi=150)
plt.close()

# =========================================================
# STEP 8 — BUDGET SCENARIO ANALYSIS
# =========================================================
# Scenario: reallocate spend from the LOWEST marginal-ROI channel to the
# HIGHEST marginal-ROI channel, holding TOTAL budget constant, and use the
# fitted saturation/adstock response to predict the resulting revenue change.
current_total_budget = total_spend.sum()

best_ch = channel_summary["marginal_roi_at_current_spend"].idxmax()
worst_ch = channel_summary["marginal_roi_at_current_spend"].idxmin()
shift_pct = 0.10  # move 10% of the worst channel's budget to the best channel
shift_amount = total_spend[worst_ch] * shift_pct

def predict_total_revenue_for_spend(spend_dict):
    """Re-run adstock -> saturation -> model for a hypothetical spend scenario
    (using each channel's already-selected decay rate and the fixed alpha),
    holding promotion/seasonality at their historical average."""
    scenario_X = {}
    for ch in CHANNELS:
        weekly_spend = np.full(len(df), spend_dict[ch] / len(df))  # spread evenly across weeks
        adst = apply_adstock(weekly_spend, best_decay[ch])
        scaled = adst / adstocked_data[ch].max()  # use SAME normalization as training
        scenario_X[ch] = np.power(np.clip(scaled, 0, None), SATURATION_ALPHA)
    scenario_X["promotion_flag"] = df["promotion_flag"].mean()
    scenario_X["seasonality_index"] = df["seasonality_index"].mean()
    scenario_df = pd.DataFrame(scenario_X)
    return model_full.predict(scenario_df).sum()

baseline_spend = total_spend.to_dict()
baseline_revenue_est = predict_total_revenue_for_spend(baseline_spend)

scenario_spend = baseline_spend.copy()
scenario_spend[worst_ch] -= shift_amount
scenario_spend[best_ch] += shift_amount
scenario_revenue_est = predict_total_revenue_for_spend(scenario_spend)

revenue_change_pct = (scenario_revenue_est - baseline_revenue_est) / baseline_revenue_est * 100

scenario_result = pd.DataFrame({
    "scenario": ["Baseline (actual spend)", f"Shift 10% of {worst_ch} -> {best_ch}"],
    "total_budget": [current_total_budget, current_total_budget],
    f"{worst_ch}": [baseline_spend[worst_ch], scenario_spend[worst_ch]],
    f"{best_ch}": [baseline_spend[best_ch], scenario_spend[best_ch]],
    "model_estimated_total_revenue": [baseline_revenue_est, scenario_revenue_est],
})
scenario_result.to_csv(f"{OUT}/budget_scenario_analysis.csv", index=False)

print(f"\n=== BUDGET SCENARIO (actual model output, not assumed) ===")
print(f"Lowest marginal ROI channel:  {worst_ch}")
print(f"Highest marginal ROI channel: {best_ch}")
print(f"Scenario: shift 10% of {worst_ch} budget -> {best_ch}, total budget held constant")
print(scenario_result.to_string(index=False))
print(f"\nModel-estimated revenue change: {revenue_change_pct:+.2f}%")
print("\nNote: this is a MODEL-BASED ESTIMATE from a simplified regression on 104 weeks")
print("of synthetic data -- it demonstrates the scenario-analysis METHOD, not a validated")
print("real-world business claim. State it that way on a resume / in interviews.")

print(f"\nAll outputs saved to {OUT}/")
