import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, roc_auc_score
)

# ---------------- Paths (relative; Windows-friendly)
DATA_PATH = Path("data/smsspam.csv")
MODELS_DIR = Path("models")

# ---------------- Utils
def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise RuntimeError(f"Dataset not found at {path.resolve()}")

    df = None
    for enc in ["utf-8", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip")
            break
        except Exception:
            pass
    if df is None:
        raise RuntimeError(f"Could not read dataset at {path}")

    cols = [c.lower() for c in df.columns]
    if "label" in cols and "message" in cols:
        df = df.rename(columns={df.columns[cols.index("label")]: "label",
                                df.columns[cols.index("message")]: "message"})
    elif "v1" in cols and "v2" in cols:
        df = df.rename(columns={df.columns[cols.index("v1")]: "label",
                                df.columns[cols.index("v2")]: "message"})
    else:
        df = df.iloc[:, :2]
        df.columns = ["label", "message"]

    df = df.dropna(subset=["label", "message"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].replace({"spam ": "spam", "ham ": "ham"})
    df = df[df["label"].isin(["ham", "spam"])].reset_index(drop=True)
    return df

def adaptive_split(df: pd.DataFrame):
    enc = {"ham": 0, "spam": 1}
    df = df.copy()
    df["y"] = df["label"].map(enc)

    counts = Counter(df["label"])
    n = len(df)
    min_class = min(counts.values()) if counts else 0

    # Make a reasonable test size for tiny sets
    if n <= 10 or min_class < 2:
        test_rows = min(max(2, n // 4), max(1, n - 1))
        test_size = test_rows / n
        stratify = None
    else:
        test_size = 0.2
        stratify = df["y"]

    return train_test_split(
        df["message"], df["y"], test_size=test_size, random_state=42, stratify=stratify
    ), counts

def build_vectorizer(n_train_docs: int) -> TfidfVectorizer:
    # Make TF-IDF safe for small datasets
    if n_train_docs <= 1:
        min_df, max_df, ngr = 1, 1.0, (1, 1)
    elif n_train_docs <= 3:
        min_df, max_df, ngr = 1, 1.0, (1, 1)
    elif n_train_docs < 10:
        min_df, max_df, ngr = 1, 0.9, (1, 2)
    else:
        min_df, max_df, ngr = 2, 0.95, (1, 2)

    return TfidfVectorizer(
        max_features=5000,
        ngram_range=ngr,
        min_df=min_df,
        max_df=max_df,
        stop_words="english",
    )

def cv_score(model, X, y, n_splits=5):
    # shrink CV if dataset is small
    n_splits = max(2, min(n_splits, int(np.floor(y.value_counts().min()))))
    if n_splits < 2:
        return np.array([1.0])  # degenerate case; avoid crash
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    return cross_val_score(model, X, y, cv=cv, scoring="accuracy")

# ---------------- Main
def main():
    print("Loading SMS dataset...")
    df = load_dataset(DATA_PATH)

    # Basic stats
    total = len(df)
    ham = int((df["label"] == "ham").sum())
    spam = int((df["label"] == "spam").sum())
    pct_spam = (spam / total * 100) if total else 0.0

    # Encoding detection info (best-effort echo)
    used_enc = "utf-8/latin1/cp1252 (auto-tried)"
    print(f"Dataset loaded with {used_enc} encoding")
    print("Dataset Statistics:")
    print(f"  Total messages: {total:,}")
    print(f"  Ham messages: {ham:,}")
    print(f"  Spam messages: {spam:,}")
    print(f"  Spam percentage: {pct_spam:.1f}%")

    print("Extracting features...")
    (X_train_text, X_test_text, y_train, y_test), counts = adaptive_split(df)

    vect = build_vectorizer(len(X_train_text))
    try:
        X_train = vect.fit_transform(X_train_text)
    except ValueError:
        # absolute fallback
        vect = TfidfVectorizer(stop_words="english")
        X_train = vect.fit_transform(X_train_text)
    X_test = vect.transform(X_test_text)
    print("Feature extraction completed!\n")

    # Models
    print("Training machine learning models...\n")
    nb = MultinomialNB(alpha=0.1)
    lr = LogisticRegression(max_iter=1000, n_jobs=None if hasattr(LogisticRegression, "n_jobs") else None)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    # Train + evaluate each
    print("="*70)
    print("MODEL TRAINING RESULTS")
    print("="*70, "\n")

    results = []

    # Naïve Bayes
    print("Training Naive Bayes...")
    nb.fit(X_train, y_train)
    nb_acc = accuracy_score(y_test, nb.predict(X_test))
    nb_cv = cv_score(MultinomialNB(alpha=0.1), X_train, y_train)
    print(f"  Test Accuracy: {nb_acc:.4f}")
    print(f"  Cross-Validation: {nb_cv.mean():.4f} ± {nb_cv.std():.4f}\n")
    results.append(("Naive Bayes", nb_acc, nb, nb_cv.mean()))

    # Logistic Regression
    print("Training Logistic Regression...")
    lr.fit(X_train, y_train)
    lr_acc = accuracy_score(y_test, lr.predict(X_test))
    lr_cv = cv_score(LogisticRegression(max_iter=1000), X_train, y_train)
    print(f"  Test Accuracy: {lr_acc:.4f}")
    print(f"  Cross-Validation: {lr_cv.mean():.4f} ± {lr_cv.std():.4f}\n")
    results.append(("Logistic Regression", lr_acc, lr, lr_cv.mean()))

    # Random Forest
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    rf_cv = cv_score(RandomForestClassifier(n_estimators=200, random_state=42), X_train, y_train)
    print(f"  Test Accuracy: {rf_acc:.4f}")
    print(f"  Cross-Validation: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}\n")
    results.append(("Random Forest", rf_acc, rf, rf_cv.mean()))

    # Best by CV mean (as in your screenshot wording)
    best_name, _, best_model, best_cv = max(results, key=lambda r: r[3])
    print(f"Best model: {best_name} (CV: {best_cv:.4f})\n")

    # Detailed classification report for NB (to mirror your example)
    print("DETAILED CLASSIFICATION REPORT - Naive Bayes:")
    print("-"*70)
    print(classification_report(y_test, nb.predict(X_test), target_names=["Ham","Spam"]))

    # Save artifacts (use an ensemble too for the UI)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ens = VotingClassifier(
        estimators=[("nb", nb), ("lr", lr), ("rf", rf)],
        voting="soft",
        weights=[1.0, 1.1, 1.0]
    )
    ens.fit(X_train, y_train)

    with open(MODELS_DIR/"vectorizer.pkl", "wb") as f: pickle.dump(vect, f)
    with open(MODELS_DIR/"ensemble_model.pkl", "wb") as f: pickle.dump(ens, f)
    with open(MODELS_DIR/"label_encoder.pkl", "wb") as f:
        pickle.dump({"label_to_int": {"ham":0,"spam":1}, "int_to_label": {0:"ham",1:"spam"}}, f)

    # Save test metrics for the frontend
    try:
        ens_proba = ens.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, ens_proba)
    except Exception:
        ens_proba = None
        auc = None
    cm = confusion_matrix(y_test, ens.predict(X_test), labels=[0,1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, ens.predict(X_test))),
        "roc_auc": float(auc) if auc is not None else None,
        "confusion_matrix": cm.tolist()
    }
    with open(MODELS_DIR/"test_metrics.json", "w") as f: json.dump(metrics, f, indent=2)

    out = pd.DataFrame({
        "message": X_test_text.reset_index(drop=True),
        "true_label": ["ham" if i==0 else "spam" for i in y_test],
        "pred_label": ["ham" if i==0 else "spam" for i in ens.predict(X_test)],
        "spam_probability": ens_proba if ens_proba is not None else np.nan
    })
    out.to_csv(MODELS_DIR/"test_predictions.csv", index=False)

    # Small demo like your screenshot
    print("Extracting features...")
    print("Feature extraction completed!")
    demo = "i love you"
    v = vect.transform([demo])
    pred = best_model.predict(v)[0]
    try:
        if hasattr(best_model, "predict_proba"):
            sp = best_model.predict_proba(v)[0,1]
            conf = 1.0 - sp if pred == 0 else sp
        else:
            sp = np.nan
            conf = 1.0
    except Exception:
        sp = np.nan
        conf = 1.0
    label = "HAM" if pred == 0 else "SPAM"
    print(f"Message: {demo}")
    print(f"Prediction: {label} (Confidence: {conf*100:.1f}%)")
    if np.isfinite(sp):
        print(f"Spam Probability: {sp*100:.1f}%")
    else:
        print("Spam Probability: N/A")

    # crude “key features” summary
    words = demo.split()
    print(f"Key features: length: {len(demo)}, words: {len(words)}")

if __name__ == "__main__":
    main()
