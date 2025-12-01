# ============================================
# 🧠 SMS Spam — End-to-End Train & Export (Single File)
# ============================================

# This script is self-contained:
# - Imports
# - Paths
# - Dataset loading (flexible locations)
# - TF-IDF + Ensemble (NB + Calibrated LinearSVC + RF)
# - Evaluation (accuracy, ROC-AUC, confusion matrix, report)
# - Saves: models/vectorizer.pkl, models/ensemble_model.pkl, models/label_encoder.pkl,
#          models/test_metrics.json, models/test_predictions.csv
# It can be run from terminal or `%run` inside a notebook.

import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# ---------------- Paths ----------------
ROOT = Path(__file__).parent if "__file__" in globals() else Path.cwd()

# Try to find the CSV in common locations relative to where this file is run
possible_paths = [
    ROOT / "data" / "smsspam.csv",
    ROOT.parent / "data" / "smsspam.csv",
    ROOT / "smsspam.csv",
    ROOT.parent / "smsspam.csv",
]

DATA_PATH = None
for p in possible_paths:
    if p.exists():
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise RuntimeError("❌ smsspam.csv not found. Put it in ./data or project root.")

MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print(f"✅ Using dataset: {DATA_PATH.resolve()}")
print(f"📁 Models will be saved to: {MODELS_DIR.resolve()}")

# ---------------- Load dataset ----------------
def load_dataset(path: Path) -> pd.DataFrame:
    # Support typical Kaggle SMS spam formats (first two columns)
    df = None
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            pass
    if df is None:
        raise RuntimeError(f"Could not read CSV at {path}")
    # Map columns
    cols = [c.lower() for c in df.columns]
    if "label" in cols and "message" in cols:
        df = df.rename(columns={df.columns[cols.index('label')]: "label",
                                df.columns[cols.index('message')]: "message"})
    elif "v1" in cols and "v2" in cols:
        df = df.rename(columns={df.columns[cols.index('v1')]: "label",
                                df.columns[cols.index('v2')]: "message"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]
    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    print("Counts:", Counter(df["label"]))
    return df

df = load_dataset(DATA_PATH)

# ---------------- Split ----------------
y = df["label"].map({"ham": 0, "spam": 1})
X_train_text, X_test_text, y_train, y_test = train_test_split(
    df["message"], y, test_size=0.2, random_state=42, stratify=y
)

print(f"Train: {len(X_train_text)}  |  Test: {len(X_test_text)}")

# ---------------- Vectorizer ----------------
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english",
)

X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# ---------------- Models ----------------
nb = MultinomialNB(alpha=0.1)
svm = CalibratedClassifierCV(LinearSVC(), cv=3)  # calibrated SVM to get probabilities
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

ens = VotingClassifier(estimators=[("nb", nb), ("svm", svm), ("rf", rf)], voting="soft", weights=[1.0, 1.2, 1.0])
ens.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = ens.predict(X_test)
try:
    y_prob = ens.predict_proba(X_test)[:, 1]
except Exception:
    y_prob = np.full_like(y_pred, np.nan, dtype=float)

acc = accuracy_score(y_test, y_pred)
try:
    auc = roc_auc_score(y_test, y_prob[~np.isnan(y_prob)])
except Exception:
    auc = float("nan")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\n✅ Training complete.")
print(f"Accuracy: {acc:.4f}  |  ROC-AUC: {auc:.4f}" if np.isfinite(auc) else f"Accuracy: {acc:.4f}  |  ROC-AUC: N/A")
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=["ham", "spam"]))
print("Confusion Matrix:\n", cm)

# ---------------- Save artifacts ----------------
with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODELS_DIR / "ensemble_model.pkl", "wb") as f:
    pickle.dump(ens, f)
with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
    pickle.dump({"label_to_int": {"ham": 0, "spam": 1}, "int_to_label": {0: "ham", 1: "spam"}}, f)

# Save test metrics for the Streamlit app
with open(MODELS_DIR / "test_metrics.json", "w") as f:
    json.dump({"accuracy": float(acc), "roc_auc": (float(auc) if np.isfinite(auc) else None), "confusion_matrix": cm.tolist()}, f, indent=2)

# Save per-row predictions for ROC/inspection
pd.DataFrame({
    "message": X_test_text.reset_index(drop=True),
    "true_label": ["ham" if i == 0 else "spam" for i in y_test],
    "pred_label": ["ham" if i == 0 else "spam" for i in y_pred],
    "spam_probability": y_prob,
}).to_csv(MODELS_DIR / "test_predictions.csv", index=False)

print(f"\n📁 Artifacts saved to: {MODELS_DIR.resolve()}")

# ---------------- Optional: show confusion matrix (when run in notebook) ----------------
try:
    fig = plt.figure(figsize=(4.5, 3.5))
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Test)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0,1], ["ham","spam"]); plt.yticks([0,1],["ham","spam"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
except Exception:
    pass
