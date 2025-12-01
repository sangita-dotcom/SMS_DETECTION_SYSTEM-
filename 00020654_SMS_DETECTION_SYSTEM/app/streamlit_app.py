# app/streamlit_app.py
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import scipy.sparse
from sklearn.metrics import roc_curve, auc  # <-- added for ROC

# ----------------------------- PAGE SETUP -----------------------------
st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="wide",
)

# Minimal style helpers (bigger fonts for presentations)
def h1(txt): st.markdown(f"<div style='font-size:28px;font-weight:700'>{txt}</div>", unsafe_allow_html=True)
def h2(txt): st.markdown(f"<div style='font-size:22px;font-weight:700;margin-top:8px'>{txt}</div>", unsafe_allow_html=True)
def big_metric(label, value): st.markdown(f"<div style='font-size:14px;color:#666'>{label}</div><div style='font-size:28px;font-weight:700'>{value}</div>", unsafe_allow_html=True)

st.title("📩 SMS Spam Detection System")

# ----------------------------- LOAD MODELS -----------------------------
@st.cache_resource
def load_models():
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/ensemble_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/label_encoder.pkl", "rb") as f:
        label_enc = pickle.load(f)
    return vectorizer, model, label_enc

# Guardrails: check files
models_dir = Path("models")
needed = ["vectorizer.pkl", "ensemble_model.pkl", "label_encoder.pkl"]
missing = [n for n in needed if not (models_dir / n).exists()]
if missing:
    st.error(
        "Model files not found: **" + ", ".join(missing) + "**\n\n"
        "➡️ Run the trainer first:\n\n```powershell\npython .\\train_and_export.py\n```"
    )
    st.stop()

vectorizer, model, label_enc = load_models()
label_map = {0: "HAM", 1: "SPAM"}

# ----------------------------- MODEL PERFORMANCE (TOP PANEL) -----------------------------
h1("Model Performance")

metrics_path = Path("models/test_metrics.json")
preds_path = Path("models/test_predictions.csv")

acc_val, auc_val, cm = None, None, None
if metrics_path.exists():
    m = json.loads(metrics_path.read_text())
    acc_val = m.get("accuracy", None)
    auc_val = m.get("roc_auc", None)
    cm = np.array(m.get("confusion_matrix", [[0,0],[0,0]]))
else:
    st.caption("No metrics file found yet. Run the trainer to generate `models/test_metrics.json`.")

cA, cB, cC = st.columns([1.2, 1.2, 2.0])
with cA:
    big_metric("Accuracy", f"{acc_val:.4f}" if isinstance(acc_val, float) else "N/A")
with cB:
    big_metric("ROC-AUC", f"{auc_val:.4f}" if isinstance(auc_val, float) else "N/A")
with cC:
    # Accuracy bar (clean, single bar)
    fig = plt.figure(figsize=(5.4, 0.6))
    ax = plt.gca()
    ax.barh(["Accuracy"], [acc_val if isinstance(acc_val, float) else 0.0])
    ax.set_xlim(0, 1)
    for spine in ["top","right","left","bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([])
    plt.tight_layout()
    st.pyplot(fig)

# Optional: show saved confusion matrix
if isinstance(cm, np.ndarray):
    h2("Saved Test-Set Confusion Matrix")
    fig2 = plt.figure(figsize=(4, 3))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Test Set)")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.colorbar()
    st.pyplot(fig2)

# ---------- NEW: ROC Curve (from saved test predictions) ----------
if preds_path.exists():
    try:
        dfp = pd.read_csv(preds_path)
        # true labels: ham=0, spam=1
        y_true = (dfp["true_label"].astype(str).str.lower() == "spam").astype(int).to_numpy()
        # predicted probabilities (may be NaN if estimator lacked predict_proba)
        y_score = dfp["spam_probability"].to_numpy()
        # filter out NaNs
        mask = np.isfinite(y_score)
        if mask.sum() >= 2 and y_true[mask].sum() > 0 and (1 - y_true[mask]).sum() > 0:
            fpr, tpr, _ = roc_curve(y_true[mask], y_score[mask])
            roc_auc = auc(fpr, tpr)
            h2("ROC Curve (Test Set)")
            fig3 = plt.figure(figsize=(4.8, 3.6))
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0, 1], [0, 1], linestyle="--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC (AUC = {roc_auc:.4f})")
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.caption("ROC curve not available (need valid probabilities and both classes in test set).")
    except Exception as e:
        st.caption(f"Could not render ROC curve: {e}")
else:
    st.caption("No per-row predictions file yet. Run the trainer to create `models/test_predictions.csv`.")

st.markdown("---")

# ----------------------------- QUICK PICKS -----------------------------
h1("Quick Picks")
col_q1, col_q2, col_q3, col_q4 = st.columns(4)
quick_ham_1 = "Hey, are we meeting at 5 pm today?"
quick_ham_2 = "I'll call you later when I finish work."
quick_spam_1 = "WIN a FREE iPhone now!!! Click the link to claim."
quick_spam_2 = "URGENT: Verify your account at http://bad.link to avoid suspension."

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

with col_q1:
    if st.button("HAM #1"): st.session_state.input_text = quick_ham_1
with col_q2:
    if st.button("HAM #2"): st.session_state.input_text = quick_ham_2
with col_q3:
    if st.button("SPAM #1"): st.session_state.input_text = quick_spam_1
with col_q4:
    if st.button("SPAM #2"): st.session_state.input_text = quick_spam_2

# ----------------------------- INPUT -----------------------------
h1("Classify a Message")
text = st.text_area("Message:", height=120, value=st.session_state.input_text, placeholder="Type an SMS message here…")
run_pred = st.button("🔍 Classify Message")

# ----------------------------- PREDICT & EXPLAIN -----------------------------
def classify_and_explain(msg: str):
    X_input = vectorizer.transform([msg])
    pred = model.predict(X_input)[0]
    # Probability (if supported)
    try:
        proba = model.predict_proba(X_input)[:, 1][0]
    except Exception:
        proba = np.nan

    # Find NB for interpretable word contributions
    nb_model = None
    for name, est in getattr(model, "estimators", []):
        if hasattr(est, "feature_log_prob_"):
            nb_model = est
            break

    exp = {"has_nb": nb_model is not None, "top_spam": [], "top_ham": [], "contrib": None, "feature_names": None}
    if nb_model is not None:
        feature_names = np.array(vectorizer.get_feature_names_out())
        X_dense = X_input.toarray() if scipy.sparse.issparse(X_input) else X_input
        # word contribution to spam vs ham (log-odds)
        log_ratio = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
        contrib = X_dense[0] * log_ratio
        exp["feature_names"] = feature_names
        exp["contrib"] = contrib
        top_spam_idx = np.argsort(contrib)[-10:][::-1]
        top_ham_idx = np.argsort(contrib)[:10]
        exp["top_spam"] = [(feature_names[i], float(contrib[i])) for i in top_spam_idx if contrib[i] > 0]
        exp["top_ham"]  = [(feature_names[i], float(contrib[i])) for i in top_ham_idx if contrib[i] < 0]
    return pred, proba, exp

if run_pred and text.strip():
    pred, proba, exp = classify_and_explain(text)

    c1, c2 = st.columns([1, 2])
    with c1:
        # Big numbers
        big_metric("Prediction", "SPAM" if pred == 1 else "HAM")
        if np.isfinite(proba):
            big_metric("Spam Probability", f"{proba:.3f}")
    with c2:
        # Live probability bar (0..1)
        val = proba if np.isfinite(proba) else 0.0
        figp = plt.figure(figsize=(6.4, 0.6))
        axp = plt.gca()
        axp.barh(["Spam Probability"], [val])
        axp.set_xlim(0, 1)
        for spine in ["top","right","left","bottom"]:
            axp.spines[spine].set_visible(False)
        axp.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        axp.set_yticks([])
        plt.tight_layout()
        st.pyplot(figp)

    st.markdown("---")
    h1("Live Explanation")

    if exp["has_nb"] and exp["feature_names"] is not None and exp["contrib"] is not None:
        top_spam, top_ham = exp["top_spam"], exp["top_ham"]

        # Bar chart of word contributions (ham-negative, spam-positive)
        if top_spam or top_ham:
            words = [w for w, _ in top_ham] + [w for w, _ in top_spam]
            values = [v for _, v in top_ham] + [v for _, v in top_spam]
            fig = plt.figure(figsize=(8, 4))
            ax = plt.gca()
            ax.barh(words, values)
            ax.axvline(0, linewidth=1)
            ax.set_title("Word Influence (NB log-odds × TF-IDF)")
            ax.set_xlabel("← ham  |  spam →")
            plt.tight_layout()
            st.pyplot(fig)

        # Highlighted text: color by contribution
        feature_names = exp["feature_names"]
        contrib = exp["contrib"]
        tokens = text.split()
        highlighted = []
        for w in tokens:
            token = w.strip(",.!?:;\"'()[]{}").lower()
            idx = np.where(feature_names == token)[0]
            if len(idx) > 0:
                val = contrib[idx[0]]
                if val > 0:
                    color = f"rgba(255, 0, 0, {min(1.0, abs(val)*3)})"     # spammy
                elif val < 0:
                    color = f"rgba(0, 170, 0, {min(1.0, abs(val)*3)})"     # hammy
                else:
                    color = "inherit"
                highlighted.append(f"<span style='background-color:{color}'>{w}</span>")
            else:
                highlighted.append(w)

        st.markdown(
            f"**Message with word influence:**<br><div style='font-size:1.08em'>{' '.join(highlighted)}</div>",
            unsafe_allow_html=True
        )

        # Lists of top contributors (presenter-friendly)
        if top_spam:
            h2("Top spam-indicative words")
            for w, v in top_spam:
                st.markdown(f"<span style='color:#b00000'>+{v:.3f}</span> — {w}", unsafe_allow_html=True)
        if top_ham:
            h2("Top ham-indicative words")
            for w, v in reversed(top_ham):
                st.markdown(f"<span style='color:#007000'>{v:.3f}</span> — {w}", unsafe_allow_html=True)
    else:
        st.info("Explanation not available (Naïve Bayes base learner not found).")

else:
    st.caption("👆 Enter a message above and click **Classify Message** to see prediction, live bar, and explanation.")

# ----------------------------- TWO CONFUSION MINI-MATRICES -----------------------------
st.markdown("---")
h1("Two ‘confusion-type’ mini-matrices")

def mini_confusion_ui(label_prefix: str, default_a: str, default_b: str):
    c1, c2 = st.columns(2)
    with c1:
        sA = st.text_area(f"{label_prefix} — Sentence A", default_a, key=f"{label_prefix}_A_text")
        yA = st.selectbox(f"True label for A", ["ham", "spam"], index=0, key=f"{label_prefix}_A_y")
    with c2:
        sB = st.text_area(f"{label_prefix} — Sentence B", default_b, key=f"{label_prefix}_B_text")
        yB = st.selectbox(f"True label for B", ["ham", "spam"], index=1, key=f"{label_prefix}_B_y")
    go = st.button(f"Evaluate {label_prefix}")
    return go, sA, sB, yA, yB

def plot_pair_confusion(sentA, sentB, trueA, trueB, title_suffix=""):
    msgs = [sentA, sentB]
    Xp = vectorizer.transform(msgs)
    y_true = np.array([0 if trueA == "ham" else 1, 0 if trueB == "ham" else 1])
    y_pred = model.predict(Xp)

    st.write(
        f"A → **{'SPAM' if y_pred[0]==1 else 'HAM'}**, "
        f"B → **{'SPAM' if y_pred[1]==1 else 'HAM'}**"
    )

    cm2 = np.zeros((2, 2), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm2[t, p] += 1

    fig = plt.figure(figsize=(4, 3))
    plt.imshow(cm2, interpolation="nearest")
    plt.title(f"Confusion (Pair {title_suffix})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks([0, 1], ["ham", "spam"])
    plt.yticks([0, 1], ["ham", "spam"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm2[i, j], ha="center", va="center")
    plt.colorbar()
    st.pyplot(fig)

# Pair 1 (A & B)
go1, sA, sB, yA, yB = mini_confusion_ui("Pair 1", "Hey, can we talk later?", "WIN a FREE iPhone now!!!")
if go1:
    plot_pair_confusion(sA, sB, yA, yB, title_suffix="1")

# Pair 2 (C & D)
go2, sC, sD, yC, yD = mini_confusion_ui("Pair 2", "I'll call you later.", "URGENT: Verify your account at http://bad.link")
if go2:
    plot_pair_confusion(sC, sD, yC, yD, title_suffix="2")

# ----------------------------- SAMPLE OF TEST-SET PREDICTIONS -----------------------------
st.markdown("---")
h1("Sample of Test-Set Predictions")
if preds_path.exists():
    dfp = pd.read_csv(preds_path)
    st.dataframe(dfp.head(25))
else:
    st.caption("No per-row predictions file found. It will be created after training.")
    
