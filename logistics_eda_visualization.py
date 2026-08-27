"""
Week 3 Task - Advanced Data Analysis and Visualization in Logistics
Author: Anam Khan
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# -----------------------------------------------------------------------
# 1. DATA SIMULATION
# -----------------------------------------------------------------------
N = 2000
regions = ["North", "South", "East", "West", "Central"]
modes = ["Road", "Air", "Rail", "Sea"]
categories = ["Electronics", "Apparel", "Groceries", "Furniture", "Pharma"]

region_p = [0.28, 0.22, 0.20, 0.18, 0.12]
mode_base_days = {"Road": 4, "Air": 1.5, "Rail": 6, "Sea": 12}
mode_base_cost = {"Road": 8, "Air": 35, "Rail": 5, "Sea": 3}

df = pd.DataFrame({
    "Shipment_ID": range(1, N + 1),
    "Region": np.random.choice(regions, N, p=region_p),
    "Transport_Mode": np.random.choice(modes, N, p=[0.5, 0.15, 0.15, 0.2]),
    "Product_Category": np.random.choice(categories, N),
    "Shipment_Volume_Units": np.random.gamma(shape=3, scale=40, size=N).round().astype(int) + 1,
})

# Delivery time depends on mode + adds noise; occasional delay spikes (bottlenecks)
df["Delivery_Time_Days"] = df["Transport_Mode"].map(mode_base_days) + np.random.normal(0, 1.2, N)
delay_mask = np.random.rand(N) < 0.06
df.loc[delay_mask, "Delivery_Time_Days"] += np.random.uniform(5, 15, delay_mask.sum())
df["Delivery_Time_Days"] = df["Delivery_Time_Days"].clip(lower=0.5).round(1)

# Transportation cost depends on mode, volume, and distance proxy (region)
region_distance_factor = {"North": 1.0, "South": 1.15, "East": 0.9, "West": 1.05, "Central": 0.8}
df["Transportation_Cost"] = (
    df["Transport_Mode"].map(mode_base_cost) * df["Shipment_Volume_Units"]
    * df["Region"].map(region_distance_factor)
    * np.random.normal(1, 0.15, N)
).round(2)

# On-time flag: delivered within mode-specific SLA
sla = {"Road": 6, "Air": 3, "Rail": 8, "Sea": 15}
df["SLA_Days"] = df["Transport_Mode"].map(sla)
df["On_Time"] = df["Delivery_Time_Days"] <= df["SLA_Days"]

# Inject a few missing values (simulating real-world data gaps, already handled per Week 2 pipeline)
for col in ["Transportation_Cost", "Delivery_Time_Days"]:
    idx = np.random.choice(df.index, size=15, replace=False)
    df.loc[idx, col] = np.nan
df[["Transportation_Cost", "Delivery_Time_Days"]] = df[["Transportation_Cost", "Delivery_Time_Days"]].apply(
    lambda s: s.fillna(s.median())
)

df.to_csv("logistics_dataset_simulated.csv", index=False)
print("Dataset shape:", df.shape)

# -----------------------------------------------------------------------
# 2. EXPLORATORY DATA ANALYSIS
# -----------------------------------------------------------------------
summary_stats = df[["Delivery_Time_Days", "Transportation_Cost", "Shipment_Volume_Units"]].describe()
print("\nSummary statistics:\n", summary_stats)

on_time_rate = df["On_Time"].mean() * 100
print(f"\nOverall on-time delivery rate: {on_time_rate:.1f}%")

corr = df[["Delivery_Time_Days", "Transportation_Cost", "Shipment_Volume_Units"]].corr()
print("\nCorrelation matrix:\n", corr)

mode_summary = df.groupby("Transport_Mode").agg(
    Avg_Delivery_Days=("Delivery_Time_Days", "mean"),
    Avg_Cost=("Transportation_Cost", "mean"),
    On_Time_Rate=("On_Time", "mean"),
    Shipments=("Shipment_ID", "count"),
).round(2)
print("\nBy transport mode:\n", mode_summary)

region_summary = df.groupby("Region").agg(
    Avg_Delivery_Days=("Delivery_Time_Days", "mean"),
    Avg_Cost=("Transportation_Cost", "mean"),
    On_Time_Rate=("On_Time", "mean"),
).round(2)
print("\nBy region:\n", region_summary)

summary_stats.to_csv("charts/summary_stats.csv")
mode_summary.to_csv("charts/mode_summary.csv")
region_summary.to_csv("charts/region_summary.csv")

# -----------------------------------------------------------------------
# 3. VISUALIZATIONS
# -----------------------------------------------------------------------
palette = sns.color_palette("Blues_d", n_colors=5)

# 3.1 Distribution of delivery times
plt.figure(figsize=(7, 4.2))
sns.histplot(df["Delivery_Time_Days"], bins=30, kde=True, color="#1F4E79")
plt.title("Distribution of Delivery Times")
plt.xlabel("Delivery Time (Days)")
plt.ylabel("Number of Shipments")
plt.tight_layout()
plt.savefig("charts/01_delivery_time_distribution.png")
plt.close()

# 3.2 Average delivery time & cost by transport mode (bar)
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
sns.barplot(x=mode_summary.index, y=mode_summary["Avg_Delivery_Days"], ax=axes[0], palette="Blues_d")
axes[0].set_title("Avg Delivery Time by Transport Mode")
axes[0].set_ylabel("Days")
sns.barplot(x=mode_summary.index, y=mode_summary["Avg_Cost"], ax=axes[1], palette="Oranges_d")
axes[1].set_title("Avg Transportation Cost by Mode")
axes[1].set_ylabel("Cost (USD)")
plt.tight_layout()
plt.savefig("charts/02_mode_time_cost.png")
plt.close()

# 3.3 On-time delivery rate by region (bar)
plt.figure(figsize=(7, 4.2))
sns.barplot(x=region_summary.index, y=region_summary["On_Time_Rate"] * 100, palette="Greens_d")
plt.title("On-Time Delivery Rate by Region")
plt.ylabel("On-Time Rate (%)")
plt.xlabel("Region")
plt.axhline(on_time_rate, color="red", linestyle="--", label=f"Overall avg ({on_time_rate:.1f}%)")
plt.legend()
plt.tight_layout()
plt.savefig("charts/03_ontime_by_region.png")
plt.close()

# 3.4 Correlation heatmap
plt.figure(figsize=(5.5, 4.5))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
plt.title("Correlation Between Key Logistics Metrics")
plt.tight_layout()
plt.savefig("charts/04_correlation_heatmap.png")
plt.close()

# 3.5 Cost vs Volume scatter, colored by mode
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df, x="Shipment_Volume_Units", y="Transportation_Cost",
                 hue="Transport_Mode", alpha=0.6, palette="Set2", s=25)
plt.title("Transportation Cost vs. Shipment Volume")
plt.xlabel("Shipment Volume (Units)")
plt.ylabel("Transportation Cost (USD)")
plt.tight_layout()
plt.savefig("charts/05_cost_vs_volume.png")
plt.close()

# 3.6 Boxplot of delivery time by mode (spot bottlenecks/outliers)
plt.figure(figsize=(7, 4.5))
sns.boxplot(data=df, x="Transport_Mode", y="Delivery_Time_Days", palette="Purples")
plt.title("Delivery Time Spread & Outliers by Transport Mode")
plt.ylabel("Delivery Time (Days)")
plt.tight_layout()
plt.savefig("charts/06_delivery_boxplot.png")
plt.close()

# 3.7 Shipment volume share by product category (pie)
cat_share = df.groupby("Product_Category")["Shipment_Volume_Units"].sum()
plt.figure(figsize=(6, 5))
plt.pie(cat_share, labels=cat_share.index, autopct="%1.1f%%",
        colors=sns.color_palette("pastel"), startangle=90)
plt.title("Shipment Volume Share by Product Category")
plt.tight_layout()
plt.savefig("charts/07_category_share_pie.png")
plt.close()

print("\nAll charts saved to charts/")
