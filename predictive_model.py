import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


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

x = df[feature_cols]
y = df["label_binary"]


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

#Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(x_train, y_train)

#Evaluate accuracy and performance
y_pred = model.predict(x_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Classification Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

coef_df = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": model.coef_[0]}
).sort_values(by="Coefficient", ascending=False)

coef_df.to_csv("model_coefficients.csv", index=False)
print(
    "\nTop predictive"
    " features:"
)
print(coef_df.to_string(index=False))



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
