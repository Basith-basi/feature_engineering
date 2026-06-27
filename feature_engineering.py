import pandas as pd

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("data/cleaned_data.csv")

print(df.head())
print(df.info())

print(df["target"].value_counts())
print(df["target"].isnull().sum())

# Remove ID column
df.drop(columns=["patientid"], errors="ignore", inplace=True)

# Create AgeGroup feature
df["AgeGroup"] = pd.cut(
    df["age"],
    bins=[0, 25, 45, 100],
    labels=["Young", "Middle", "Senior"]
)

# Save engineered dataset
df.to_csv("data/baseline_feature.csv", index=False)

print("baseline_feature.csv created successfully!")