import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 1. Load the merged master dataset
df = pd.read_csv("ifnd_features_master.csv")

# 2. Define feature columns (exclude metadata and target)
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

# Drop any rows with missing feature values
df_model = df.dropna(subset=feature_cols + ["label_binary"])

X = df_model[feature_cols]
y = df_model["label_binary"]

# 3. Split into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Logistic Regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate accuracy and performance
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Classification Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6. Extract and save feature importance (coefficients)
coef_df = pd.DataFrame(
    {"Feature": feature_cols, "Coefficient": model.coef_[0]}
).sort_values(by="Coefficient", ascending=False)

coef_df.to_csv("model_coefficients.csv", index=False)
print(
    "\nFeature coefficients saved to model_coefficients.csv! Top predictive"
    " features:"
)
print(coef_df.to_string(index=False))