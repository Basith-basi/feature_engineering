import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("data/baseline_feature.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Convert AgeGroup to numbers
encoder = LabelEncoder()
df["AgeGroup"] = encoder.fit_transform(df["AgeGroup"])
# ==========================
# Define Target Column
# ==========================

target_column = "target"      # Change this if your target column has a different name

X = df.drop(columns=[target_column])
y = df[target_column]

# ==========================
# Split Dataset
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# ==========================
# Train Model
# ==========================
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("data/baseline_feature.csv")

# Encode AgeGroup
encoder = LabelEncoder()
df["AgeGroup"] = encoder.fit_transform(df["AgeGroup"])

# Features and Target
X = df.drop(columns=["target", "patientid"], errors="ignore")
y = df["target"]
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# ==========================
# Prediction
# ==========================

y_pred = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

# ==========================
# Save Model
# ==========================

os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/baseline_model.pkl")

print("\nModel Saved Successfully!")
print("Location : models/baseline_model.pkl")