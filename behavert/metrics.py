"""Evaluation metrics."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def classification_report(
    y_true,
    y_pred,
    num_classes: int,
    ignore_classes: Iterable[int] = (),
    label_names: Optional[Sequence[str]] = None,
    average_classes: Optional[Iterable[int]] = None,
) -> dict:
    """Per-class precision / recall / F1 and their macro average.

    Frames with label ``-1`` are dropped. Frames whose *true* label is in
    ``ignore_classes`` are dropped too, and those classes are left out of the
    average (CalMS21 convention: ``other`` is not scored), but predicting an
    ignored class for a scored frame still counts as an error.

    ``average_classes`` restricts which classes enter ``macro_f1`` without
    dropping any frames (e.g. PAIR-R24M social reports Close / Explore / Chase
    but not None). ``f1_class1`` is the F1 of class 1, the number reported for
    binary behavior tasks.
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    ignore = set(int(c) for c in ignore_classes)
    keep = (y_true >= 0) & ~np.isin(y_true, list(ignore) or [-2])
    y_true, y_pred = y_true[keep], y_pred[keep]
    classes = [c for c in range(num_classes) if c not in ignore]
    names = list(label_names) if label_names else [str(c) for c in range(num_classes)]
    if len(y_true) == 0:
        return {"accuracy": float("nan"), "macro_f1": float("nan"), "per_class": {}, "n": 0}
    p = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    r = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    avg = classes if average_classes is None else [c for c in classes if c in set(average_classes)]
    per = dict(zip(classes, f))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(np.mean([per[c] for c in avg])),
        "f1_class1": float(per.get(1, float("nan"))),
        "per_class": {names[c]: {"precision": float(pi), "recall": float(ri), "f1": float(fi)}
                      for c, pi, ri, fi in zip(classes, p, r, f)},
        "n": int(len(y_true)),
    }


def regression_report(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = y_pred - y_true
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err))), "n": int(len(err)),
            "label_variance": float(np.var(y_true))}


def recording_decision(window_probs: np.ndarray) -> tuple[int, np.ndarray]:
    """Combine window probabilities ``[N, C]`` into one recording-level label.

    The mean probability over windows decides. For two classes this is the
    paper's rule: class 1 if the mean of ``p1 - p0`` is ``>= 0``.
    Returns the decision and the mean probabilities.
    """
    mean = np.asarray(window_probs, dtype=np.float64).mean(0)
    if mean.shape[0] == 2:
        return int(mean[1] - mean[0] >= 0), mean
    return int(mean.argmax()), mean


def format_report(rep: dict, title: str = "") -> str:
    lines = [title] if title else []
    if "mse" in rep:
        lines.append(f"  MSE {rep['mse']:.5f}   MAE {rep['mae']:.5f}   (label variance {rep['label_variance']:.5f}, n={rep['n']})")
        return "\n".join(lines)
    for name, m in rep.get("per_class", {}).items():
        lines.append(f"  {name:>20s}  P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}")
    lines.append(f"  {'macro F1':>20s}  {rep['macro_f1']:.4f}   accuracy {rep['accuracy']:.4f}   (n={rep['n']})")
    if len(rep.get("per_class", {})) == 2:
        lines.append(f"  {'class-1 F1':>20s}  {rep['f1_class1']:.4f}")
    return "\n".join(lines)
