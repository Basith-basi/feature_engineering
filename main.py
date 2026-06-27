import subprocess

print("=" * 60)
print("TASK 7 : FEATURE ENGINEERING PIPELINE")
print("=" * 60)

print("\nStep 1 : Feature Engineering")
subprocess.run(["python", "feature_engineering.py"], check=True)

print("\nStep 2 : Train Baseline Model")
subprocess.run(["python", "train_baseline.py"], check=True)

print("\nStep 3 : Feature Importance")
subprocess.run(["python", "feature_importance.py"], check=True)

print("\nStep 4 : Leakage Check")
subprocess.run(["python", "leakage_check.py"], check=True)

print("\nAll Tasks Completed Successfully!")