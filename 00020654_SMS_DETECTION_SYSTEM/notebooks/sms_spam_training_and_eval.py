# ============================================
# 🧠 SMS Spam Ensemble - Training & Evaluation
# ============================================

# Split dataset
X = df["message"]
y = df["label"].map({"ham": 0, "spam": 1})

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Vectorization
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    stop_words="english"
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Models
nb = MultinomialNB(alpha=0.1)
svm = CalibratedClassifierCV(LogisticRegression(max_iter=200, C=3), cv=3)
rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

ensemble = VotingClassifier(
    estimators=[("nb", nb), ("svm", svm), ("rf", rf)],
    voting="soft",
    weights=[1.0, 1.2, 1.0]
)

# Train
ensemble.fit(X_train_vec, y_train)

# Predictions
y_pred = ensemble.predict(X_test_vec)
y_prob = ensemble.predict_proba(X_test_vec)[:, 1]

# Evaluation
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("✅ Model Training Completed!")
print(f"Accuracy: {acc:.4f}, ROC-AUC: {auc:.4f}")
print("Confusion Matrix:\n", cm)

# Classification Report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

# ============================================
# 📈 Visualization
# ============================================

plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
plt.xticks([0, 1], ["Ham", "Spam"])
plt.yticks([0, 1], ["Ham", "Spam"])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
plt.tight_layout()
plt.show()

# ============================================
# 💾 Save Artifacts
# ============================================

with open(MODELS_DIR / "vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(MODELS_DIR / "ensemble_model.pkl", "wb") as f:
    pickle.dump(ensemble, f)

metrics = {
    "accuracy": float(acc),
    "roc_auc": float(auc),
    "confusion_matrix": cm.tolist()
}

with open(MODELS_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n📁 Artifacts Saved in:", MODELS_DIR.resolve())
