# SMS Spam Detection — Ensemble (NB + SVM + RF)

This package contains a complete, ready-to-run project for a robust SMS spam detector using an **ensemble** of:
- Naïve Bayes
- Calibrated Linear SVM (for probabilities)
- Random Forest

## Structure
```
sms_spam_ensemble_pkg/
├── app/
│   └── streamlit_app.py         # Frontend (Streamlit)
├── data/
│   └── smsspam.csv              # Dataset (HAM/SPAM)
├── legacy/
│   └── simple_sms_spam_detector.py  # Your old reference code
├── models/
│   ├── vectorizer.pkl
│   ├── ensemble_model.pkl
│   ├── label_encoder.pkl
│   ├── test_metrics.json
│   └── test_predictions.csv
├── notebooks/
│   └── sms_spam_ensemble.ipynb  # Live charts + confusion matrix
├── train_and_export.py          # Retraining script
└── requirements.txt
```

## Quickstart (VS Code or terminal)
1. Create a virtual environment
   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

2. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Retrain and export artifacts
   ```bash
   python train_and_export.py
   ```

4. Run the UI (Streamlit)
   ```bash
   streamlit run app/streamlit_app.py
   ```
   - Classify any SMS
   - Enter exactly two sentences and see a **mini confusion matrix** for just those two
   - View saved **test-set confusion matrix** + **accuracy/ROC-AUC**

## Notebook
Open `notebooks/sms_spam_ensemble.ipynb` and run all cells to see training logs, metrics, and the confusion matrix chart inline.

## Notes
- Uses TF-IDF (1–2 grams, 5k max features, English stop-words).
- Soft-voting ensemble with a slight weight on the SVM for sparse text.
- Models already trained and saved to `models/`. You can regenerate them with the trainer.
