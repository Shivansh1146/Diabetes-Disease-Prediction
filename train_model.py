"""
train_model.py - Diabetes Prediction Model Training Script
Trains Logistic Regression and Random Forest models, selects the best one,
and saves it along with the scaler and metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

# ─────────────────────────────────────────────
# 1. Paths
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

DATASET_PATH = os.path.join(DATASET_DIR, 'diabetes.csv')
MODEL_PATH   = os.path.join(MODELS_DIR, 'diabetes_model.pkl')
SCALER_PATH  = os.path.join(MODELS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics.json')

# ─────────────────────────────────────────────
# 2. Generate dataset if missing
# ─────────────────────────────────────────────
def ensure_dataset():
    if not os.path.exists(DATASET_PATH):
        print("[!] Dataset not found - downloading original Pima Indians dataset...")
        os.chdir(DATASET_DIR)
        exec(open(os.path.join(DATASET_DIR, 'generate_dataset.py')).read())
        os.chdir(BASE_DIR)
        print(f"[OK] Dataset saved to {DATASET_PATH}")
    else:
        print(f"[OK] Dataset found: {DATASET_PATH}")

# ─────────────────────────────────────────────
# 3. Load & Clean Data
# ─────────────────────────────────────────────
def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"\n[Info] Raw dataset shape: {df.shape}")

    # Replace physiologically-impossible zeros with NaN
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)

    # Fill missing values with median (robust to skew)
    for col in zero_cols:
        median = df[col].median()
        if pd.isna(median):
            median = 0
        df[col] = df[col].fillna(median)
        print(f"   Filled NaN in {col:25s} -> median = {median:.2f}")

    # Explicitly drop any remaining rows with NaNs across all columns just to be completely safe
    df = df.dropna()

    # Clip extreme outliers at 1st / 99th percentile
    feature_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    for col in feature_cols:
        lo, hi = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(lo, hi)

    print(f"\n[OK] Cleaned dataset shape: {df.shape}")
    print(f"   Class distribution:\n{df['Outcome'].value_counts()}")
    return df

# ─────────────────────────────────────────────
# 4. Train Models
# ─────────────────────────────────────────────
def train(df):
    FEATURES = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    TARGET   = 'Outcome'

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Logistic Regression ──
    print("\n[*] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    lr_prob = lr.predict_proba(X_test_s)[:, 1]
    lr_acc  = accuracy_score(y_test, lr_pred)
    lr_auc  = roc_auc_score(y_test, lr_prob)

    # ── Random Forest ──
    print("[*] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, class_weight='balanced', n_jobs=-1
    )
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    rf_prob = rf.predict_proba(X_test_s)[:, 1]
    rf_acc  = accuracy_score(y_test, rf_pred)
    rf_auc  = roc_auc_score(y_test, rf_prob)

    print(f"\n[Result] Logistic Regression -> Accuracy: {lr_acc:.4f}  AUC: {lr_auc:.4f}")
    print(f"[Result] Random Forest       -> Accuracy: {rf_acc:.4f}  AUC: {rf_auc:.4f}")

    # ── Choose best model by AUC ──
    if rf_auc >= lr_auc:
        best_model, best_name = rf, "Random Forest"
        best_pred, best_prob  = rf_pred, rf_prob
    else:
        best_model, best_name = lr, "Logistic Regression"
        best_pred, best_prob  = lr_pred, lr_prob

    print(f"\n[Winner] Best model: {best_name}")

    # ── Final metrics ──
    cm  = confusion_matrix(y_test, best_pred)
    acc = accuracy_score(y_test, best_pred)
    pre = precision_score(y_test, best_pred)
    rec = recall_score(y_test, best_pred)
    f1  = f1_score(y_test, best_pred)
    auc = roc_auc_score(y_test, best_prob)

    # ── Cross-validation ──
    cv_scores = cross_val_score(
        best_model, scaler.transform(X), y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='accuracy'
    )

    print(f"\n{'='*50}")
    print(f"  Model        : {best_name}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  Precision    : {pre:.4f}")
    print(f"  Recall       : {rec:.4f}")
    print(f"  F1-Score     : {f1:.4f}")
    print(f"  AUC-ROC      : {auc:.4f}")
    print(f"  CV Accuracy  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"  Confusion Matrix:\n{cm}")
    print(f"{'='*50}\n")

    # ── Model-specific comparison table ──
    comparison = {
        "Logistic Regression": {
            "accuracy": round(lr_acc * 100, 2),
            "auc": round(lr_auc, 4),
            "precision": round(precision_score(y_test, lr_pred), 4),
            "recall": round(recall_score(y_test, lr_pred), 4),
            "f1": round(f1_score(y_test, lr_pred), 4),
        },
        "Random Forest": {
            "accuracy": round(rf_acc * 100, 2),
            "auc": round(rf_auc, 4),
            "precision": round(precision_score(y_test, rf_pred), 4),
            "recall": round(recall_score(y_test, rf_pred), 4),
            "f1": round(f1_score(y_test, rf_pred), 4),
        }
    }

    metrics = {
        "model_name": best_name,
        "accuracy": round(acc * 100, 2),
        "precision": round(pre * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "auc_roc": round(auc, 4),
        "cv_accuracy": round(cv_scores.mean() * 100, 2),
        "cv_std": round(cv_scores.std() * 100, 2),
        "confusion_matrix": cm.tolist(),
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
        "comparison": comparison,
        "feature_names": FEATURES,
        "training_samples": len(X_train),
        "testing_samples": len(X_test),
        "total_samples": len(X),
    }

    # ── Save ──
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[OK] Model  saved -> {MODEL_PATH}")
    print(f"[OK] Scaler saved -> {SCALER_PATH}")
    print(f"[OK] Metrics saved -> {METRICS_PATH}")
    return metrics

# ─────────────────────────────────────────────
# 5. Entry Point
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("  Diabetes Prediction Model Training")
    print("=" * 60)
    ensure_dataset()
    df      = load_and_clean(DATASET_PATH)
    metrics = train(df)
    print("\n[Done] Training complete! You can now run: python app.py")
