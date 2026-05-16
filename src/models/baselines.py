"""
src/models/baselines.py

Scikit-learn / XGBoost baselines operating on concatenated hand-crafted features.
Useful for benchmarking and SHAP-based interpretability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score, balanced_accuracy_score
)
import joblib

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


# ──────────────────────────────────────────────────────────────────────────────
# Feature matrix builder from DataLoader batches
# ──────────────────────────────────────────────────────────────────────────────

def collate_features(
    batches: List[Dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a list of dataset item dicts into (X, y) arrays for sklearn.

    Uses: snore_vec, physio_vec, resp_vec (concatenated).
    """
    X_list, y_list = [], []
    for item in batches:
        parts = []
        for key in ("snore_vec", "physio_vec", "resp_vec"):
            v = item.get(key)
            if v is not None:
                arr = v.numpy() if hasattr(v, "numpy") else np.asarray(v)
                parts.append(arr.ravel())
        if parts:
            X_list.append(np.concatenate(parts))
            y_list.append(int(item["label"]))

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)
    # Replace NaN / Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, y


# ──────────────────────────────────────────────────────────────────────────────
# Random Forest baseline
# ──────────────────────────────────────────────────────────────────────────────

def build_rf_pipeline(
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# XGBoost baseline
# ──────────────────────────────────────────────────────────────────────────────

def build_xgb_pipeline(
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Pipeline:
    if not HAS_XGB:
        raise ImportError("xgboost not installed. Run: pip install xgboost")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_jobs=n_jobs,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric="mlogloss",
        )),
    ])


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_classifier(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    results: Dict = {
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, target_names=class_names),
    }

    if y_prob is not None:
        n_classes = y_prob.shape[1]
        if n_classes == 2:
            results["roc_auc"] = roc_auc_score(y_test, y_prob[:, 1])
        else:
            try:
                results["roc_auc_ovr"] = roc_auc_score(
                    y_test, y_prob, multi_class="ovr", average="macro"
                )
            except Exception:
                pass

    return results


# ──────────────────────────────────────────────────────────────────────────────
# SHAP interpretation helper
# ──────────────────────────────────────────────────────────────────────────────

def compute_shap_values(clf, X: np.ndarray, feature_names: Optional[List[str]] = None):
    """
    Compute SHAP values for a fitted sklearn/XGB pipeline.

    Returns a (shap_values, explainer) tuple.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap not installed. Run: pip install shap")

    # Extract underlying model (after scaler)
    model = clf[-1] if hasattr(clf, "__getitem__") else clf
    X_transformed = clf[:-1].transform(X) if hasattr(clf, "__getitem__") else X

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.Explainer(model, X_transformed)

    shap_values = explainer(X_transformed)
    return shap_values, explainer


# ──────────────────────────────────────────────────────────────────────────────
# Save / load
# ──────────────────────────────────────────────────────────────────────────────

def save_model(clf, path: str) -> None:
    joblib.dump(clf, path)
    print(f"Model saved to {path}")


def load_model(path: str):
    return joblib.load(path)
