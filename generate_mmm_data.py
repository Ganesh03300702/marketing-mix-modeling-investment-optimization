"""
generate_mmm_data.py
----------------------
Generates 104 weeks (2 years) of synthetic weekly marketing data:
TV, Digital, Social, Print spend + Promotions + Seasonality -> Revenue.

Ground-truth effects are injected deliberately (known adstock decay rates,
known saturation curves, known channel effectiveness) so later in the
pipeline we can sanity-check whether the model recovers sensible signals.
This is standard practice before applying MMM to real data.
"""

import numpy as np
import pandas as pd

np.random.seed(11)

N_WEEKS = 104
weeks = pd.date_range("2023-01-02", periods=N_WEEKS, freq="W-MON")

# --- True (hidden) channel behavior we're injecting, for our own sanity checks ---
TRUE_PARAMS = {
    "tv":      {"adstock_decay": 0.55, "sat_alpha": 0.65, "effectiveness": 3.8},
    "digital": {"adstock_decay": 0.25, "sat_alpha": 0.55, "effectiveness": 5.2},
    "social":  {"adstock_decay": 0.35, "sat_alpha": 0.60, "effectiveness": 4.0},
    "print":   {"adstock_decay": 0.45, "sat_alpha": 0.70, "effectiveness": 2.0},
}

# --- Weekly spend patterns (different scale/volatility per channel) ---
tv_spend      = np.clip(45000 + 12000*np.sin(np.arange(N_WEEKS)/8) + np.random.normal(0, 6000, N_WEEKS), 5000, None)
digital_spend = np.clip(30000 + 8000*np.sin(np.arange(N_WEEKS)/5 + 1) + np.random.normal(0, 4500, N_WEEKS), 3000, None)
social_spend  = np.clip(18000 + 6000*np.sin(np.arange(N_WEEKS)/6 + 2) + np.random.normal(0, 3000, N_WEEKS), 1000, None)
print_spend   = np.clip(9000 + 2500*np.sin(np.arange(N_WEEKS)/10) + np.random.normal(0, 1500, N_WEEKS), 500, None)

# --- Promotions: binary flag, ~15% of weeks ---
promo_flag = (np.random.rand(N_WEEKS) < 0.15).astype(int)

# --- Seasonality: yearly cycle (weekly granularity -> period 52) ---
seasonality_index = 1 + 0.20 * np.sin(2 * np.pi * np.arange(N_WEEKS) / 52 + 0.5)

def apply_adstock(spend, decay):
    """Geometric adstock: this week's effective spend = spend + decay * previous effective spend.
    Models carryover -- an ad seen this week still influences behavior next week."""
    adstocked = np.zeros_like(spend, dtype=float)
    adstocked[0] = spend[0]
    for t in range(1, len(spend)):
        adstocked[t] = spend[t] + decay * adstocked[t - 1]
    return adstocked

def apply_saturation(adstocked_spend, alpha):
    """Diminishing-returns (saturation) curve using a power transform:
    response = adstocked_spend ** alpha, alpha < 1 => diminishing returns."""
    return np.power(adstocked_spend, alpha)

channel_contributions = {}
raw_spend = {"tv": tv_spend, "digital": digital_spend, "social": social_spend, "print": print_spend}

base_revenue = 180000
total_response = np.zeros(N_WEEKS)

for ch, spend in raw_spend.items():
    p = TRUE_PARAMS[ch]
    adstocked = apply_adstock(spend, p["adstock_decay"])
    saturated = apply_saturation(adstocked, p["sat_alpha"])
    contribution = p["effectiveness"] * saturated
    channel_contributions[ch] = contribution
    total_response += contribution

promo_lift = promo_flag * 15000
noise = np.random.normal(0, 6000, N_WEEKS)

revenue = (base_revenue * seasonality_index) + total_response + promo_lift + noise
revenue = np.clip(revenue, 50000, None)

df = pd.DataFrame({
    "week": weeks,
    "tv_spend": np.round(tv_spend, 2),
    "digital_spend": np.round(digital_spend, 2),
    "social_spend": np.round(social_spend, 2),
    "print_spend": np.round(print_spend, 2),
    "promotion_flag": promo_flag,
    "seasonality_index": np.round(seasonality_index, 4),
    "revenue": np.round(revenue, 2),
})

# Inject minor realistic messiness (a handful of missing spend values) for the cleaning step
dirty_idx = np.random.choice(df.index, size=5, replace=False)
df.loc[dirty_idx, "digital_spend"] = np.nan

df.to_csv("project3_mmm_investment_optimization/data/weekly_marketing_data.csv", index=False)
print(f"Generated {len(df)} weekly rows -> data/weekly_marketing_data.csv")
print(df.head())
