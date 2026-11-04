import pandas as pd

# Load master dataset
df = pd.read_csv("ifnd_features_master.csv")

# 1. Inspect top outlets publishing fake vs real news
print("--- Top Outlets Breakdown ---")
outlet_label_counts = (
    df.groupby(["Web", "label_binary"]).size().unstack(fill_value=0)
)
outlet_label_counts.columns = ["Real Count", "Fake Count"]
outlet_label_counts["Total"] = (
    outlet_label_counts["Real Count"] + outlet_label_counts["Fake Count"]
)
print(outlet_label_counts.sort_values(by="Total", ascending=False).head(10))

# 2. Robustness Check: Compare features controlling for outlet
# Let's see the mean word count and VADER sentiment for real vs fake *within* the most common outlets
print("\n--- Robustness Check: Word Count & Sentiment by Outlet ---")
robustness_summary = (
    df.groupby(["Web", "label_binary"])[["word_count", "vader"]]
    .mean()
    .unstack()
)
robustness_summary.columns = [
    "Real_WordCount",
    "Fake_WordCount",
    "Real_Vader",
    "Fake_Vader",
]
robustness_summary["Total_Articles"] = (
    df.groupby("Web").size()
)  # add article counts per outlet

# Sort by outlets with the most articles
robustness_summary = robustness_summary.sort_values(
    by="Total_Articles", ascending=False
)

# Save to CSV for the report
robustness_summary.to_csv("robustness_outlet_control.csv")
print(
    "Robustness check complete! Saved summary table to"
    " 'robustness_outlet_control.csv'."
)
print(robustness_summary.head(10))