"""
Shared evaluation utilities for ISIC classification and segmentation.

Classification helpers:
    compute_clf_metrics   — macro-F1, per-class F1, accuracy
    plot_confusion_matrix — saves a labelled heatmap
    plot_roc_curves       — per-class OvR ROC curves on one figure

Segmentation helpers:
    compute_dice          — Dice coefficient (batch or single)
    compute_iou           — IoU / Jaccard (batch or single)
    compute_seg_metrics   — both at once

Visualization:
    plot_seg_predictions  — image | GT mask | pred mask grid
    plot_clf_predictions  — image | GT label | pred label grid

All plot_* functions return the matplotlib Figure so callers can either
save it (fig.savefig(...)) or log it to W&B (wandb.log({"fig": fig})).
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_curve,
    auc,
)

# ISIC 2018 Task 3 class names (label index order matches splits CSV)
CLF_CLASSES = [
    "MEL",   # Melanoma
    "NV",    # Melanocytic nevus
    "BCC",   # Basal cell carcinoma
    "AKIEC", # Actinic keratosis / Bowen's disease
    "BKL",   # Benign keratosis
    "DF",    # Dermatofibroma
    "VASC",  # Vascular lesion
]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_clf_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] = CLF_CLASSES,
) -> dict:
    """
    Compute macro-F1, per-class F1, and overall accuracy.

    Args:
        y_true:       ground-truth integer labels
        y_pred:       predicted integer labels
        class_names:  label names for the returned dict keys

    Returns:
        dict with keys:
            "macro_f1"        — float
            "accuracy"        — float
            "per_class_f1"    — dict {class_name: float}
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(acc),
        "per_class_f1": {
            name: float(per_class[i])
            for i, name in enumerate(class_names)
            if i < len(per_class)
        },
    }


def plot_confusion_matrix(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str] = CLF_CLASSES,
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """
    Plot a labelled confusion matrix heatmap.

    Args:
        normalize: if True, normalize rows to sum to 1 (shows recall per class)

    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], fmt),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=8,
            )

    fig.tight_layout()
    return fig


def plot_roc_curves(
    y_true: list[int] | np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str] = CLF_CLASSES,
    title: str = "Per-class ROC Curves (OvR)",
) -> plt.Figure:
    """
    Plot one ROC curve per class (one-vs-rest).

    Args:
        y_true: integer ground-truth labels, shape (N,)
        y_prob: softmax probabilities, shape (N, num_classes)

    Returns:
        matplotlib Figure
    """
    y_true = np.asarray(y_true)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=1.5, label=f"{name} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set(
        xlim=[0, 1], ylim=[0, 1.02],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=title,
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------

def compute_dice(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    Dice coefficient for binary segmentation.

    Args:
        pred:      predicted logits or probabilities, shape (N, H, W) or (H, W)
        target:    ground-truth binary mask, same shape as pred
        threshold: binarization threshold (applied to pred)
        eps:       smoothing term to avoid division by zero

    Returns:
        scalar Dice score in [0, 1]
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    intersection = (pred_bin * target).sum()
    return float((2.0 * intersection + eps) / (pred_bin.sum() + target.sum() + eps))


def compute_iou(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> float:
    """
    IoU (Jaccard index) for binary segmentation.

    Args: same as compute_dice
    Returns:
        scalar IoU score in [0, 1]
    """
    pred_bin = (pred > threshold).astype(np.float32)
    target = target.astype(np.float32)

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return float((intersection + eps) / (union + eps))


def compute_seg_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute both Dice and IoU at once.

    Returns:
        dict with keys "dice" and "iou"
    """
    return {
        "dice": compute_dice(pred, target, threshold),
        "iou":  compute_iou(pred, target, threshold),
    }


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_seg_predictions(
    images: list[np.ndarray],
    masks_gt: list[np.ndarray],
    masks_pred: list[np.ndarray],
    n: int = 6,
    title: str = "Segmentation Predictions",
) -> plt.Figure:
    """
    Display a grid of: original image | ground-truth mask | predicted mask.

    Args:
        images:     list of HWC uint8 numpy arrays
        masks_gt:   list of HW float32 arrays (values 0 or 1)
        masks_pred: list of HW float32 arrays (raw probabilities)
        n:          number of samples to show

    Returns:
        matplotlib Figure
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground Truth", "Predicted"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=11, fontweight="bold")

    for i in range(n):
        dice = compute_dice(masks_pred[i], masks_gt[i])
        axes[i, 0].imshow(images[i])
        axes[i, 1].imshow(masks_gt[i],   cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(masks_pred[i] > 0.5, cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_ylabel(f"Dice={dice:.3f}", fontsize=8, rotation=0,
                               labelpad=50, va="center")
        for ax in axes[i]:
            ax.axis("off")

    fig.suptitle(title, fontsize=13, y=1.01)
    fig.tight_layout()
    return fig


def plot_clf_predictions(
    images: list[np.ndarray],
    labels_gt: list[int],
    labels_pred: list[int],
    class_names: list[str] = CLF_CLASSES,
    n: int = 8,
    ncols: int = 4,
    title: str = "Classification Predictions",
) -> plt.Figure:
    """
    Display a grid of images with ground-truth and predicted labels.
    Correct predictions are shown in green, incorrect in red.

    Args:
        images:      list of HWC uint8 numpy arrays
        labels_gt:   list of integer ground-truth labels
        labels_pred: list of integer predicted labels
        n:           number of samples to show

    Returns:
        matplotlib Figure
    """
    n = min(n, len(images))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3.2))
    axes = np.array(axes).flatten()

    for i in range(n):
        ax = axes[i]
        ax.imshow(images[i])
        ax.axis("off")
        gt   = class_names[labels_gt[i]]
        pred = class_names[labels_pred[i]]
        color = "green" if labels_gt[i] == labels_pred[i] else "red"
        ax.set_title(f"GT: {gt}\nPred: {pred}", fontsize=8, color=color)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig
