# Readmission-DL — City General Hospital 30-day Readmission Prediction

**Student name:** *Kashyap Vekariya*
**Student ID:** *1144*
**Submission date:** *27/03/2026*

---

## Problem

Predict whether a patient will be readmitted within 30 days of discharge using structured clinical data from City General Hospital (3,800 training records, 950 test records). The positive class (readmitted) represents only 9% of records — a 10:1 class imbalance.

---

## My model

**Architecture:**
3-hidden-layer MLP: `Input(21) → [Linear(64) → BatchNorm → ReLU → Dropout(0.5)] → [Linear(32) → BatchNorm → ReLU → Dropout(0.5)] → [Linear(16) → BatchNorm → ReLU → Dropout(0.5)] → Linear(1)`

Raw logits are output (no final sigmoid); `BCEWithLogitsLoss` with `pos_weight=8.0` handles both the sigmoid and the class-imbalance reweighting in a single numerically stable step. Trained with Adam (lr=3e-3), cosine annealing LR schedule, 80 epochs, batch size 128. Best checkpoint per fold selected by validation AUROC.

**Key preprocessing decisions:**
Three data-quality issues were corrected before modelling: (1) blood pressure values below 30 mmHg were identified as decimal-shift entry errors and multiplied by 10; (2) age sentinel values of 999 were replaced with NaN and imputed with the training-set median via `SimpleImputer`; (3) glucose missingness (17.7%) was preserved as a binary indicator column before median imputation, allowing the model to learn whether the test was ordered at all as a clinical signal. Day-of-week and admission month were cyclic-encoded (sin/cos) to preserve their periodicity.

**How I handled class imbalance:**
`BCEWithLogitsLoss(pos_weight=8.0)` — the positive class loss is upweighted by a factor of 8 during training (capped from the raw 10.1 ratio to avoid collapsing precision). No data augmentation or oversampling was used, keeping all validation folds entirely free of synthetic samples.

---

## Results on validation set

| Metric | Value |
|--------|-------|
| AUROC | **0.9269** |
| F1 (minority class) | **0.6267** |
| Precision (minority) | 0.5984 |
| Recall (minority) | 0.6579 |
| Decision threshold used | 0.295 |

*All metrics computed on out-of-fold (OOF) predictions from 5-fold stratified cross-validation — no data leakage.*

---

## How to run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

Run the notebook top-to-bottom:

```bash
jupyter notebook notebooks/solution.ipynb
```

Or execute as a script:

```bash
cd notebooks && jupyter nbconvert --to script solution.ipynb --stdout | python
```

This trains the model, saves evaluation plots to `outputs/`, and saves `outputs/metrics.json`.

### 3. Run inference on the test set

```bash
python src/predict.py --input data/test.csv
```

With a custom output path or threshold:

```bash
python src/predict.py --input data/test.csv --output predictions.csv --threshold 0.35
```

The output CSV contains two columns: `patient_id` and `readmission_probability`.

---

## Repository structure

```
readmission-dl/
├── data/
│   ├── train.csv
│   └── test.csv
├── notebooks/
│   └── solution.ipynb       ← full pipeline: EDA → training → evaluation
├── outputs/                 ← auto-created by notebook
│   ├── eda.png
│   ├── evaluation.png
│   ├── predictions.csv
│   └── metrics.json
├── src/
│   └── predict.py           ← inference script
├── DECISIONS.md
├── requirements.txt
└── README.md
```

---

## Limitations and honest assessment

**What would improve with more time:**
- Gradient-boosted trees (LightGBM / XGBoost) typically outperform MLPs on tabular clinical data at this sample size; a stacking ensemble of LightGBM + MLP would likely push AUROC above 0.95.
- SHAP values to validate that `glucose_missing` and the cyclic date features contribute genuine signal rather than noise.
- Platt scaling or isotonic regression for probability calibration — `pos_weight` reweighting distorts the output probabilities relative to the true 9% prior, which matters if the probabilities are used for risk scoring rather than binary classification.
- Optuna hyperparameter search over dropout rate, hidden layer sizes, and learning rate.

**Where this model might fail in production:**
- **Temporal shift:** the dataset covers 2020 admissions; admission patterns changed post-COVID. Periodic retraining is essential.
- **BP correction heuristic:** if the data-entry system is fixed (decimal-shift bug removed), the `x < 30 → ×10` correction would corrupt future legitimate low-BP readings (e.g. septic shock patients). The cleaning step should be reviewed before deployment.
- **Threshold recalibration:** the optimal threshold (0.295) was tuned on the training distribution. If the hospital implements readmission-reduction interventions that lower the true readmission rate, the threshold will need to be recalibrated — use `--threshold` in `predict.py`.
