import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("data/baseline_feature.csv")

# Encode AgeGroup (same as train_baseline.py)
if "AgeGroup" in df.columns:
    encoder = LabelEncoder()
    df["AgeGroup"] = encoder.fit_transform(df["AgeGroup"])

# ===========================
# Prepare Features
# ===========================

X = df.drop(columns=["target"], errors="ignore")

print("Feature Columns:")
print(X.columns.tolist())

# ===========================
# Load Trained Model
# ===========================

model = joblib.load("models/baseline_model.pkl")

importance = model.feature_importances_

print("\nNumber of Features :", len(X.columns))
print("Number of Importances :", len(importance))

# ===========================
# Check Feature Mismatch
# ===========================

if len(X.columns) != len(importance):
    print("\nERROR!")
    print("The model was trained with a different number of features.")
    print("Please make sure train_baseline.py and feature_importance.py use the same preprocessing.")
    exit()

# ===========================
# Create DataFrame
# ===========================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

# ===========================
# Create Output Folder
# ===========================

os.makedirs("outputs", exist_ok=True)

# ===========================
# Save CSV
# ===========================

feature_importance.to_csv(
    "outputs/feature_importance.csv",
    index=False
)

# ===========================
# Plot Graph
# ===========================

plt.figure(figsize=(12, 6))

plt.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xticks(rotation=45, ha="right")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.title("Random Forest Feature Importance")

plt.tight_layout()

plt.savefig("outputs/feature_importance.png")

plt.show()
# plt.show()
plt.close()

print("\nFeature Importance Saved Successfully!")
print("CSV   : outputs/feature_importance.csv")
print("Graph : outputs/feature_importance.png")

