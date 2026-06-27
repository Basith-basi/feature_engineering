import pandas as pd

# ===========================
# Load Dataset
# ===========================

df = pd.read_csv("data/baseline_feature.csv")

print("=" * 60)
print("          LEAKAGE CHECK")
print("=" * 60)

print("\nColumns in Dataset:\n")

for col in df.columns:
    print("-", col)

print("\nChecking each column...\n")

# ===========================
# Keywords
# ===========================

id_keywords = [
    "patientid",
    "customerid",
    "userid",
    "orderid"
]

leakage_keywords = [
    "status",
    "result",
    "outcome",
    "prediction",
    "label",
    "final",
    "diagnosis"
]

id_found = []
leakage_found = []

# ===========================
# Leakage Check
# ===========================

for col in df.columns:

    column = col.lower()

    # ID columns
    if (
        column in id_keywords
        or column.endswith("id")
        or column == "id"
    ):
        print(f"{col:25} -> REMOVE (Identifier)")
        id_found.append(col)

    # Target column
    elif column == "target":
        print(f"{col:25} -> TARGET COLUMN")

    # Leakage columns
    elif any(keyword in column for keyword in leakage_keywords):
        print(f"{col:25} -> POSSIBLE LEAKAGE")
        leakage_found.append(col)

    # Safe columns
    else:
        print(f"{col:25} -> OK")

# ===========================
# Summary
# ===========================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print(f"Identifier Columns Found : {len(id_found)}")
print(f"Leakage Columns Found    : {len(leakage_found)}")

if id_found:
    print("Identifiers :", ", ".join(id_found))

if leakage_found:
    print("Leakage Columns :", ", ".join(leakage_found))
else:
    print("No leakage features detected.")
print("\n" + "=" * 60)
print("Leakage Check Completed Successfully!")
print("=" * 60)