import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

print("Loading master dataset...")
df = pd.read_csv("../data/ifnd_features_master.csv")

ignore_cols = {
    "row_id",
    "text",
    "label_binary",
    "Web",
    "Category",
    "year",
    "month",
    "year_month",
    "date_quality",
    "label_str",
}
features = [c for c in df.columns if c not in ignore_cols]

real_group = df[df["label_binary"] == 0]
fake_group = df[df["label_binary"] == 1]

records = []
for feat in features:
  real_vals = real_group[feat]
  fake_vals = fake_group[feat]

  t_stat, p_val = stats.ttest_ind(real_vals, fake_vals, equal_var=False)
  log_p = -np.log10(p_val) if p_val > 0 else 300.0

  records.append(
      {"Feature": feat, "t_stat": t_stat, "p_value": p_val, "neg_log_p": log_p}
  )

results_df = pd.DataFrame(records).sort_values(by="neg_log_p", ascending=False)
results_df.to_csv("all_features_p_values_master.csv", index=False)

print("\n--- Significance Results ---")
print(results_df.to_string(index=False))

sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 7))
sns.barplot(x="neg_log_p", y="Feature", data=results_df, palette="viridis_r")
plt.axvline(
    x=-np.log10(0.05),
    color="crimson",
    linestyle="--",
    label="Significance Alpha (0.05)",
)
plt.title(
    "Feature Significance: Welch t-Test (-log10 p-values)",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("-log10(p-value)", fontsize=11)
plt.ylabel("Dataset Feature", fontsize=11)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("p_values_all_features_plot.png", dpi=300)
plt.close()