import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==============================================
# ⚙️ Load model + vectorizer + metrics
# ==============================================
MODELS_DIR = Path("models")
METRICS_FILE = MODELS_DIR / "test_metrics.json"
VECTORIZER_FILE = MODELS_DIR / "vectorizer.pkl"

st.set_page_config(page_title="SMS Spam Detector", layout="wide")

st.title("📩 SMS Spam Detection Dashboard")
st.markdown("### Real-time spam classification with live accuracy charts & explanations")

if not METRICS_FILE.exists() or not VECTORIZER_FILE.exists():
    st.error("❌ Model artifacts not found. Please train the model first using `sms_spam_comparison.ipynb`.")
    st.stop()

# Load metrics
with open(METRICS_FILE, "r") as f:
    metrics = json.load(f)

best_model_name = metrics["best_model"]
with open(MODELS_DIR / f"model_{best_model_name.replace(' ', '_').replace('+','plus').replace('(','').replace(')','')}.pkl", "rb") as f:
    model = pickle.load(f)

with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

# ==============================================
# 📊 Sidebar: Metrics
# ==============================================
st.sidebar.header("📈 Model Performance Summary")
st.sidebar.metric("Best Model", best_model_name)
st.sidebar.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
if metrics.get("roc_auc"):
    st.sidebar.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

# Confusion Matrix
if "confusion_matrix" in metrics:
    st.sidebar.markdown("**Confusion Matrix**")
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    st.sidebar.pyplot(fig)

# ==============================================
# 🧠 Main: User input
# ==============================================
st.subheader("🔍 Try the Model Yourself")

example_texts = [
    "Win a FREE iPhone now!!! Click here to claim your prize.",
    "Hey, I’ll call you when I reach home.",
]
default_text = st.text_area("Enter an SMS message:", example_texts[0], height=100)

if st.button("Classify Message"):
    X = vectorizer.transform([default_text])
    pred = model.predict(X)[0]
    try:
        prob = model.predict_proba(X)[0][1]
    except Exception:
        prob = None

    label = "🚨 SPAM" if pred == 1 else "✅ HAM"
    st.markdown(f"### Prediction: {label}")

    if prob is not None:
        st.progress(prob)
        st.markdown(f"**Spam probability:** {prob:.3f}")

# ==============================================
# 💬 Word Influence Visualization (Naive Bayes)
# ==============================================
st.subheader("🧩 Word Influence Analysis")

try:
    nb_path = MODELS_DIR / "model_Naive_Bayes.pkl"
    if nb_path.exists():
        with open(nb_path, "rb") as f:
            nb_model = pickle.load(f)

        feature_names = np.array(vectorizer.get_feature_names_out())
        log_ratio = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]

        def explain_sentence(text, top_k=8):
            X = vectorizer.transform([text])
            contrib = X.toarray()[0] * log_ratio
            top_spam_idx = np.argsort(contrib)[-top_k:][::-1]
            top_ham_idx = np.argsort(contrib)[:top_k]
            top_spam = [(feature_names[i], contrib[i]) for i in top_spam_idx if contrib[i] > 0]
            top_ham = [(feature_names[i], contrib[i]) for i in top_ham_idx if contrib[i] < 0]
            words = [w for w,_ in top_ham] + [w for w,_ in top_spam]
            vals = [v for _,v in top_ham] + [v for _,v in top_spam]
            fig, ax = plt.subplots(figsize=(6,3))
            ax.barh(words, vals, color=["green" if v<0 else "red" for v in vals])
            ax.axvline(0, color="black", linewidth=1)
            ax.set_title("Word Influence (negative → ham, positive → spam)")
            st.pyplot(fig)

        if st.button("Explain Word Influence"):
            explain_sentence(default_text)

except Exception as e:
    st.warning("Word-level analysis only available for Naive Bayes model.")

# ==============================================
# 🧾 Footer
# ==============================================
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit & scikit-learn")
S