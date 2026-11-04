import pandas as pd

# Load all parts
df_clean = pd.read_csv("ifnd_clean.csv")
df_m1 = pd.read_csv("features_m1.csv")
df_m2 = pd.read_csv("features_m2.csv")
df_m3 = pd.read_csv("features_m3.csv")

# Merge them on row_id
df_master = df_clean.merge(df_m1, on="row_id").merge(df_m2, on="row_id").merge(df_m3, on="row_id")

# Save master dataset for Phase 2 analysis
df_master.to_csv("ifnd_features_master.csv", index=False)
print("Master dataset created with shape:", df_master.shape)