from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Data" / ",id,cons_12m,cons_gas_12m,cons_last_montT4.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)

# Remove the exported CSV index column, if present.
df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")]
df = df.loc[:, df.columns != ""]

if "churn" not in df.columns:
    raise ValueError("Target column 'churn' was not found in the dataset.")

# Keep the original dataset unchanged.
df = df.dropna(subset=["churn"]).copy()
y = df["churn"].astype(int)
X = df.drop(columns=["id", "churn"], errors="ignore")

# Ensure model inputs are numeric.
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([float("inf"), float("-inf")], float("nan"))
X = X.fillna(X.median()).fillna(0)

if y.nunique() < 2:
    raise ValueError("The target column must contain both churn classes.")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

model = RandomForestClassifier(
    n_estimators=1000,
    random_state=42,
    n_jobs=-1,
)

print("Training Random Forest model...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"\nTest accuracy: {accuracy_score(y_test, predictions):.4f}")

if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(X_test)[:, 1]
    print(f"Test ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

print("\nClassification report:")
print(classification_report(y_test, predictions, zero_division=0))

joblib.dump(model, MODEL_DIR / "churn_model.joblib")

with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
    json.dump(X.columns.tolist(), file, indent=2)

print(f"\nModel saved: {MODEL_DIR / 'churn_model.joblib'}")
print(f"Features saved: {MODEL_DIR / 'feature_columns.json'}")
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
