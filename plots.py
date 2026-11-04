import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Set plotting style
sns.set_theme(style="whitegrid")

# Load master dataset
df = pd.read_csv("ifnd_features_master.csv")

# Map labels for nicer plot legends
df["label_str"] = df["label_binary"].map({0: "Real News", 1: "Fake News"})

# 1. Figure 1: Word Count Comparison Boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(
    x="label_str",
    y="word_count",
    data=df,
    palette=["#4c72b0", "#c44e52"],
    showfliers=False,
)
plt.title(
    "Headline Word Count Distribution: Real vs Fake News",
    fontsize=14,
    fontweight="bold",
)
plt.xlabel("Headline Type", fontsize=12)
plt.ylabel("Word Count", fontsize=12)
plt.tight_layout()
plt.savefig("boxplot_word_count.png", dpi=300)
plt.close()

# 2. Figure 2: VADER Sentiment Polarity Comparison
plt.figure(figsize=(8, 6))
sns.kdeplot(
    data=df,
    x="vader",
    hue="label_str",
    common_norm=False,
    palette=["#4c72b0", "#c44e52"],
    fill=True,
    alpha=0.4,
)
plt.title(
    "Emotional Polarity (VADER) Distribution", fontsize=14, fontweight="bold"
)
plt.xlabel("VADER Compound Score (-1 to +1)", fontsize=12)
plt.ylabel("Density", fontsize=12)
plt.tight_layout()
plt.savefig("density_vader_sentiment.png", dpi=300)
plt.close()

print(
    "Plots generated successfully! Saved 'boxplot_word_count.png' and"
    " 'density_vader_sentiment.png'."
)