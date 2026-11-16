"""Metrik klasifikasi (tanpa dependensi sklearn agar dashboard ringan)."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int = 4) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_report(y_true, y_pred, n_classes: int = 4) -> dict:
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos = cm.sum(axis=0).astype(float)
    actual_pos = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    present = actual_pos > 0
    return {
        "accuracy": float(tp.sum() / max(cm.sum(), 1)),
        "precision_macro": float(precision[present].mean()),
        "recall_macro": float(recall[present].mean()),
        "f1_macro": float(f1[present].mean()),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support": actual_pos.astype(int).tolist(),
        "confusion_matrix": cm.tolist(),
    }


def macro_f1(y_true, y_pred, n_classes: int = 4) -> float:
    return classification_report(y_true, y_pred, n_classes)["f1_macro"]
