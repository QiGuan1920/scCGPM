"""Evaluation utilities for the GSE112274 target domain."""

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .model import ChemGuideModel


@torch.no_grad()
def evaluate_target(
    model: ChemGuideModel,
    target_pathways: np.ndarray,
    target_drug_idx: int,
    target_labels: np.ndarray,
    drug_features_t: torch.Tensor,
    device: torch.device,
):
    model.eval()
    target_x = torch.tensor(target_pathways, dtype=torch.float32, device=device)
    target_feat = drug_features_t[target_drug_idx].unsqueeze(0).expand(
        target_x.shape[0], -1
    )

    latent, gate = model.encode(target_x, target_feat)
    probability = torch.sigmoid(model.logits(latent)).cpu().numpy()
    y = np.asarray(target_labels).astype(int)

    # Retrospective evaluation convention inherited from the supplied code:
    # select a Youden-J threshold from target labels. This is not a label-free
    # deployment threshold.
    if len(np.unique(y)) < 2:
        best_threshold = 0.5
    else:
        fpr, tpr, thresholds = roc_curve(y, probability)
        best_threshold = float(thresholds[np.argmax(tpr - fpr)])
        if not np.isfinite(best_threshold):
            best_threshold = 1.0

    prediction = (probability >= best_threshold).astype(int)

    def safe(metric_fn, **kwargs):
        try:
            return float(metric_fn(y, **kwargs))
        except Exception:
            return float("nan")

    recalls = recall_score(y, prediction, average=None, labels=[0, 1], zero_division=0)
    metrics = {
        "ROC_AUC": safe(roc_auc_score, y_score=probability),
        "PR_AUC": safe(average_precision_score, y_score=probability),
        "Macro_F1": safe(f1_score, y_pred=prediction, average="macro", zero_division=0),
        "Balanced_Acc": safe(balanced_accuracy_score, y_pred=prediction),
        "MCC": safe(matthews_corrcoef, y_pred=prediction),
        "Recall_resistant(0)": float(recalls[0]),
        "Recall_sensitive(1)": float(recalls[1]),
        "best_threshold": best_threshold,
        "n_cells": int(len(y)),
    }

    return metrics, probability, prediction, gate.cpu().numpy(), best_threshold
