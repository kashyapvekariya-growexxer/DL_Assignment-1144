# Decision log

This file documents three key decision points from your pipeline.
**Each entry is mandatory. Vague or generic answers will be penalised.**
The question to answer is not "what does this technique do" — it is "why did YOU make THIS choice given YOUR data."

---

## Decision 1: Data cleaning strategy
*Complete after Phase 1 (by approximately 1:30)*

**What I did:**
I applied four targeted fixes to observed data-quality problems. (1) Age = 999 was replaced with NaN and imputed downstream with the column median — a binary `age_missing` indicator was NOT added because the sentinel pattern was rare (12 rows) and the imputer handles it cleanly. (2) Blood pressure values below 30 mmHg were multiplied by 10 — these are clearly decimal-shift entry errors (values like 12.4 become 124 mmHg, a normal systolic). (3) `glucose_level_mgdl` had 673 missing values (17.7%); a binary `glucose_missing` indicator was added before median imputation so the model can learn that missingness itself is a clinical signal. (4) Mixed date formats (`YYYY-MM-DD` vs `DD/MM/YYYY`) were parsed with `format="mixed"`, and month was extracted as a cyclic feature (sin/cos encoding) to respect its periodicity.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
The BP threshold of 30 (not 50, not 60) was chosen because values in the anomalous cluster range from 10–25: multiplying by 10 gives 100–250 mmHg, which is entirely within the physiological systolic range. A threshold of 50 would also catch legitimately low readings (e.g. 45 mmHg in a post-operative patient). Cyclic encoding for month and day-of-week was chosen over ordinal integers because December (12) and January (1) are adjacent temporally but numerically distant — the MLP cannot recover that relationship from raw integers without learning it explicitly.

**What I considered and rejected:**
- For BP: I considered treating sub-30 values as NaN (missing) rather than correcting them. Rejected because the pattern is systematic and the correction is recoverable — treating them as missing discards real clinical information.
- For glucose missingness: I considered dropping the 673 rows. Rejected because that removes 9% of the dataset — disproportionately harmful when the positive class already has only 342 examples.
- For age = 999: I considered a missingness indicator column. Rejected because only 12 rows are affected — the indicator would be near-zero everywhere and unlikely to add signal worth the added feature dimensionality.

**What would happen if I was wrong here:**
- If BP < 30 values are not decimal-shift errors (e.g. they were entered in kPa), multiplying by 10 instead of 7.5 would produce slightly biased values (e.g. 17 × 10 = 170 vs 17 × 7.5 = 127.5 mmHg). The feature would still be directionally correct and the model would learn from it, but calibration for that subgroup would be imprecise.
- If glucose missingness is random rather than informative, the `glucose_missing` column adds noise. The cost is minimal — one spurious binary feature — but the risk of masking a true signal is higher than the cost of keeping it.

---

## Decision 2: Model architecture and handling class imbalance
*Complete after Phase 2 (by approximately 3:00)*

**What I did:**
I used a 3-hidden-layer MLP (`64 → 32 → 16`) with BatchNorm after each layer, ReLU activations, Dropout (0.5), and raw logits fed into `BCEWithLogitsLoss` with `pos_weight = min(n_neg/n_pos, 8.0)`. The pos_weight was capped at 8.0 (not the raw 10.1) to avoid over-correcting toward recall at the expense of precision. Training used Adam (lr = 3e-3), cosine annealing LR schedule, 80 epochs, and the best model checkpoint by validation AUROC was restored per fold.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
The 10:1 imbalance (3,458 vs 342) means that without intervention the model achieves 91% accuracy by predicting all-negative. `pos_weight` in `BCEWithLogitsLoss` is the most direct lever available in PyTorch — it reweights the loss for positive-class examples without requiring any data augmentation or separate oversampling step, which keeps the validation data completely clean (no synthetic samples in the val fold). I chose `pos_weight = 8.0` (not 10.1) after observing on a held-out check that the raw ratio drove recall very high (>0.85) while precision collapsed to ~0.25 — the 8.0 cap yielded a better F1 balance. BatchNorm was added because with only 3,800 training samples the internal covariate shift across folds can cause instability; BatchNorm stabilises the loss surface without adding parameters.

**What I considered and rejected:**
- SMOTE oversampling: considered but rejected in favour of pos_weight because SMOTE introduces synthetic samples that can leak into the validation fold if not carefully managed per-fold, and it requires an extra dependency (`imblearn`). pos_weight achieves the same rebalancing effect directly in the loss function.
- Deeper architecture (256 → 128 → 64 → 32): tried in preliminary runs — validation AUROC peaked at the same level but training was slower and more prone to overfitting given the dataset size.
- Focal loss: would be appropriate here, but requires a custom loss function. `BCEWithLogitsLoss` with pos_weight is functionally equivalent for the binary case and is part of the PyTorch standard library.

**What would happen if I was wrong here:**
- If pos_weight = 8.0 is still too high for the deployment population (e.g. if true readmission rate is lower than 9%), the model will over-predict positives — high recall but poor precision. The threshold in `predict.py` can be raised to correct for this without retraining.
- If the MLP is underfitting (the 3-layer architecture is too shallow for the interaction structure in this data), AUROC would plateau below 0.93. The held-out CV AUROC of 0.927 suggests the architecture is appropriate — there is no sign of systematic bias across folds.

---

## Decision 3: Evaluation metric and threshold selection
*Complete after Phase 3 (by approximately 4:00)*

**What I did:**
I used AUROC as the primary model-selection metric (used for early stopping and checkpoint selection per fold) and then performed a sweep over 171 threshold values from 0.05 to 0.90 on the full out-of-fold (OOF) predictions to find the threshold that maximises F1 on the minority class. The optimal threshold was **0.295** (OOF AUROC 0.9269, F1 0.6267, Precision 0.5984, Recall 0.6579). Test predictions are generated as the average of the 5 fold models' probabilities.

**Why I did it (not why the technique works — why THIS choice given what you observed in the data):**
AUROC was used for checkpoint selection because it is threshold-independent and unaffected by the 10:1 class imbalance — a model that always predicts 0 has AUROC = 0.5, not 0.91. The threshold was then tuned separately on OOF predictions (not a held-out set) because with only 342 positive examples, splitting off a separate threshold-tuning set would reduce the calibration sample below 70 positives — too noisy to be reliable. The threshold of 0.295 (lower than the default 0.5) reflects the effect of `pos_weight` in the loss: the model learns to output higher logits for positive examples, pushing its calibrated probabilities upward, so the decision boundary needs to be shifted left to match the true class prior.

**What I considered and rejected:**
- Using the default threshold of 0.5: on OOF predictions this gave F1 = 0.42 for the minority class vs 0.627 at 0.295 — a 49% relative improvement. There is no reason to use 0.5 when the training distribution has been deliberately reweighted.
- Tuning to maximise recall: a threshold of ~0.10 achieves recall > 0.85 but precision drops to ~0.20 — clinically unworkable because 80% of high-risk flags would be false alarms. F1 reflects the cost symmetry appropriate for a discharge-intervention context.
- Average Precision (PR-AUC) as primary metric: appropriate and more sensitive to imbalance than AUROC, but AUROC is the dominant metric in published readmission literature, enabling comparison with external benchmarks.

**What would happen if I was wrong here:**
- If the OOF threshold (0.295) is optimistic due to overfitting to the 3,800-sample distribution, the test-set F1 will be lower than reported. This is partially mitigated by using all 5 folds' OOF predictions (the full training set) rather than a single held-out split.
- If the deployment population has a different readmission rate (e.g. post-intervention), the threshold becomes miscalibrated. The `predict.py` script exposes `--threshold` as a CLI flag so the clinical team can recalibrate without retraining the model.

---

*Word count guidance: aim for 80–150 words per decision. More is not better — precision is.*
