import math
import pandas as pd


def mean(lst):
  return sum(lst) / len(lst) if lst else 0.0


def variance(lst):
  if len(lst) < 2:
    return 0.0
  m = mean(lst)
  return sum((x - m) ** 2 for x in lst) / (len(lst) - 1)


def median(lst):
  if not lst:
    return 0.0
  sorted_lst = sorted(lst)
  n = len(sorted_lst)
  mid = n // 2
  if n % 2 == 0:
    return (sorted_lst[mid - 1] + sorted_lst[mid]) / 2.0
  return sorted_lst[mid]


def cohens_d(group1, group2):
  n1, n2 = len(group1), len(group2)
  if n1 < 2 or n2 < 2:
    return 0.0
  var1, var2 = variance(group1), variance(group2)
  pooled_se = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
  if pooled_se == 0:
    return 0.0
  return (mean(group1) - mean(group2)) / pooled_se


# Load master dataset
df = pd.read_csv("ifnd_features_master.csv")

exclude_cols = [
    "row_id",
    "text",
    "label_binary",
    "Web",
    "Category",
    "year",
    "month",
    "year_month",
    "date_quality",
]
feature_cols = [c for c in df.columns if c not in exclude_cols]

results = []
real_df = df[df["label_binary"] == 0]
fake_df = df[df["label_binary"] == 1]

for col in feature_cols:
  r_vals = real_df[col].dropna().tolist()
  f_vals = fake_df[col].dropna().tolist()

  results.append({
      "Feature": col,
      "Real Mean": round(mean(r_vals), 4),
      "Fake Mean": round(mean(f_vals), 4),
      "Real Median": round(median(r_vals), 4),
      "Fake Median": round(median(f_vals), 4),
      "Cohens d": round(cohens_d(r_vals, f_vals), 4),
  })

summary_df = pd.DataFrame(results)
summary_df.to_csv("univariate_summary_results.csv", index=False)
print("Univariate analysis complete (No NumPy)! Saved to csv.")
print(summary_df.to_string())

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Load directly from the CSV you already saved
plot_df = pd.read_csv("model_coefficients.csv").sort_values(
    by="Coefficient", ascending=True
)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, max(6, len(plot_df) * 0.4)))

colors = ["crimson" if c > 0 else "steelblue" for c in plot_df["Coefficient"]]
plt.barh(
    plot_df["Feature"],
    plot_df["Coefficient"],
    color=colors,
    edgecolor="black",
    alpha=0.85,
)
plt.axvline(x=0, color="black", linestyle="-", linewidth=1)

plt.title(
    "Logistic Regression Coefficients Across All Features\n(Red = Predicts"
    " Fake | Blue = Predicts Real)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Standardized Coefficient Value (beta)", fontsize=12)
plt.ylabel("Evaluated Feature", fontsize=12)

plt.tight_layout()
plt.savefig("coefficient_magnitude_all_features.png", dpi=300, bbox_inches="tight")
plt.close()

print("Saved full coefficient plot from CSV as 'coefficient_magnitude_all_features.png'")