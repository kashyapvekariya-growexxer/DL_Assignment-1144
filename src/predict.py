import argparse
import json
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
MODEL_DIR = os.path.join(REPO_ROOT, "outputs")

ALL_INSURANCE = ["Medicaid", "Medicare", "Private", "Uninsured"]


# ── Preprocessing ─────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["admission_date"] = pd.to_datetime(
        df["admission_date"], format="mixed", dayfirst=False
    )
    df["admission_month"] = df["admission_date"].dt.month
    df.drop(columns=["admission_date"], inplace=True)

    df["age"] = df["age"].replace(999, np.nan)

    df["blood_pressure_systolic"] = df["blood_pressure_systolic"].apply(
        lambda x: x * 10 if pd.notna(x) and x < 30 else x
    )

    df["glucose_missing"] = df["glucose_level_mgdl"].isna().astype(int)

    dow_map = {"Mon": 0, "Tue": 1, "Wed": 2, "Thu": 3,
               "Fri": 4, "Sat": 5, "Sun": 6}
    dow_num = df["discharge_day_of_week"].map(dow_map)
    df["discharge_dow_sin"] = np.sin(2 * np.pi * dow_num / 7)
    df["discharge_dow_cos"] = np.cos(2 * np.pi * dow_num / 7)
    df.drop(columns=["discharge_day_of_week"], inplace=True)

    df["admission_month_sin"] = np.sin(2 * np.pi * df["admission_month"] / 12)
    df["admission_month_cos"] = np.cos(2 * np.pi * df["admission_month"] / 12)
    df.drop(columns=["admission_month"], inplace=True)

    df["gender_enc"] = (df["gender"] == "M").astype(int)
    df.drop(columns=["gender"], inplace=True)

    df["insurance_type"] = pd.Categorical(
        df["insurance_type"], categories=ALL_INSURANCE
    )
    ins_dummies = pd.get_dummies(df["insurance_type"], prefix="ins", drop_first=True)
    df = pd.concat([df.drop(columns=["insurance_type"]), ins_dummies], axis=1)

    return df


# ── PyTorch model  ← UPDATED to match solution_fixed.py ──────────────────────
#   hidden: [64, 32, 16]   
#   dropout: 0.5            
if TORCH_AVAILABLE:
    class ReadmissionMLP(nn.Module):
        def __init__(self, input_dim: int,
                     hidden: list = [64, 32, 16],   # ← updated
                     dropout: float = 0.5):          # ← updated
            super().__init__()
            layers = []
            in_dim = input_dim
            for h in hidden:
                layers += [
                    nn.Linear(in_dim, h),
                    nn.BatchNorm1d(h),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(1)


def load_torch_model(model_path, input_dim, device):
    # Architecture must match what was saved during training
    model = ReadmissionMLP(input_dim, [64, 32, 16], dropout=0.5).to(device)  # ← updated
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_torch(models, X_sc, device) -> np.ndarray:
    X_t = torch.from_numpy(X_sc).to(device)
    proba_sum = np.zeros(len(X_sc))
    with torch.no_grad():
        for m in models:
            proba_sum += torch.sigmoid(m(X_t)).cpu().numpy()
    return proba_sum / len(models)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="30-day hospital readmission inference."
    )
    parser.add_argument("--input",     "-i", required=True,
                        help="Path to input CSV (e.g. data/test.csv)")
    parser.add_argument("--output",    "-o",
                        default=os.path.join(REPO_ROOT, "predictions.csv"),
                        help="Path for output CSV (default: predictions.csv)")
    parser.add_argument("--threshold", "-t", type=float, default=None,
                        help="Override decision threshold (default: read from metrics.json)")
    args = parser.parse_args()

    # ── Load input ────────────────────────────────────────────────────────
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: '{args.input}'")
        sys.exit(1)

    df_raw   = pd.read_csv(args.input)
    test_ids = df_raw["patient_id"].values
    print(f"Input: {args.input}  ({len(df_raw)} records)")

    # ── Clean ─────────────────────────────────────────────────────────────
    df_clean = clean(df_raw)
    df_clean = df_clean.drop(columns=["patient_id", "readmitted_30d"], errors="ignore")

    # ── Imputer + Scaler ──────────────────────────────────────────────────
    imputer_path = os.path.join(MODEL_DIR, "imputer.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")

    if os.path.isfile(imputer_path) and os.path.isfile(scaler_path):
        import pickle
        with open(imputer_path, "rb") as f: imputer = pickle.load(f)
        with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    else:
        train_path = os.path.join(REPO_ROOT, "data", "train.csv")
        if not os.path.isfile(train_path):
            print(f"ERROR: training data not found at {train_path}")
            sys.exit(1)
        train_raw   = pd.read_csv(train_path)
        train_clean = clean(train_raw)
        X_tr_raw    = train_clean.drop(
            columns=["readmitted_30d", "patient_id"], errors="ignore"
        )
        imputer = SimpleImputer(strategy="median")
        imputer.fit(X_tr_raw)
        scaler = StandardScaler()
        scaler.fit(imputer.transform(X_tr_raw))

    # ── Align columns ─────────────────────────────────────────────────────
    try:
        df_clean = df_clean.reindex(columns=imputer.feature_names_in_, fill_value=0)
    except AttributeError:
        pass

    X_imp = imputer.transform(df_clean).astype(np.float32)
    X_sc  = scaler.transform(X_imp).astype(np.float32)
    input_dim = X_sc.shape[1]

    # ── Threshold ─────────────────────────────────────────────────────────
    # Reads the threshold saved by solution_fixed.py (metrics.json).
    # Falls back to 0.5 if metrics.json is missing.
    default_threshold = 0.5
    if args.threshold is not None:
        threshold = args.threshold
        print(f"Threshold: {threshold:.3f}  (user override)")
    elif os.path.isfile(metrics_path):
        with open(metrics_path) as f:
            threshold = json.load(f).get("threshold", default_threshold)
        print(f"Threshold: {threshold:.3f}  (loaded from metrics.json)")
    else:
        threshold = default_threshold
        print(f"Threshold: {threshold:.3f}  (default fallback)")

    # ── Predict ───────────────────────────────────────────────────────────
    proba = None

    # Primary: load saved PyTorch fold models
    if TORCH_AVAILABLE:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fold_models = []
        for fold in range(1, 6):
            mp = os.path.join(MODEL_DIR, f"model_fold{fold}.pt")
            if os.path.isfile(mp):
                fold_models.append(load_torch_model(mp, input_dim, device))
        if fold_models:
            print(f"Loaded {len(fold_models)} PyTorch fold model(s) from {MODEL_DIR}")
            proba = predict_torch(fold_models, X_sc, device)

   
    if proba is None:
        print("No saved PyTorch models found → training sklearn MLPClassifier fallback …")
        train_path  = os.path.join(REPO_ROOT, "data", "train.csv")
        train_raw   = pd.read_csv(train_path)
        train_cl    = clean(train_raw)
        X_tr_raw    = train_cl.drop(columns=["readmitted_30d", "patient_id"], errors="ignore")
        y_train     = train_cl["readmitted_30d"].values.astype(np.float32)
        X_tr_imp    = imputer.transform(X_tr_raw).astype(np.float32)
        X_tr_sc     = scaler.transform(X_tr_imp).astype(np.float32)

        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),  
            activation="relu",
            solver="adam",
            alpha=1e-3,                        
            batch_size=128,
            learning_rate_init=1e-3,           
            max_iter=150,                      
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,               
            random_state=42,
        )
        clf.fit(X_tr_sc, y_train)
        proba = clf.predict_proba(X_sc)[:, 1]
        print("Fallback MLPClassifier training done.")

    # ── Binary predictions ────────────────────────────────────────────────
    binary_preds = (proba >= threshold).astype(int)

    # ── Write output ──────────────────────────────────────────────────────
    out = pd.DataFrame({
        "patient_id":     test_ids,
        "readmitted_30d": binary_preds,
    })
    out.to_csv(args.output, index=False)

    print(f"\nOutput saved → {args.output}")
    print(f"Total predicted as readmitted (1): {binary_preds.sum()} / {len(binary_preds)}")
    print("\nSample predictions:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()