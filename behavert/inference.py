"""Run a trained model on windows or on whole recordings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .data.windows import Windows, sliding_windows, stitch_window_scores
from .metrics import recording_decision
from .model import BehaVERT


def _device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.no_grad()
def predict_windows(model: BehaVERT, windows: Windows, batch_size: int = 256, return_embeddings: bool = False):
    """Softmax probabilities per window: ``[N, L, C]`` (token) or ``[N, C]`` (sequence).

    Regression models return their raw outputs instead.

    With ``return_embeddings=True`` also returns the classifier input
    (per-frame ``[N, L, H]`` or ``[CLS]`` ``[N, H]``).
    """
    model.eval()
    dev = _device(model)
    probs, embs = [], []
    for i in range(0, len(windows), batch_size):
        x = windows.inputs[i : i + batch_size].to(dev)
        m = windows.mask[i : i + batch_size].to(dev)
        out = model(x, attention_mask=m)
        logits = out.logits.float()
        probs.append((logits if model.config.problem == "regression" else logits.softmax(-1)).cpu())
        if return_embeddings:
            embs.append(out.embedding.float().cpu())
    p = torch.cat(probs) if probs else torch.zeros(0)
    if return_embeddings:
        return p, torch.cat(embs)
    return p


@dataclass
class RecordingPrediction:
    probs: np.ndarray
    """Token model: per-frame ``[T, C]`` (overlapping windows averaged). Sequence model: per-window ``[N, C]``."""
    labels: np.ndarray
    """Token model: per-frame ``[T]``. Sequence model: per-window ``[N]``."""
    window_starts: np.ndarray
    recording_label: Optional[int] = None
    """Sequence model only: one label for the whole recording."""
    recording_probs: Optional[np.ndarray] = None


@torch.no_grad()
def predict_recording(
    model: BehaVERT,
    keypoints,
    stride: Optional[int] = None,
    batch_size: int = 256,
) -> RecordingPrediction:
    """Predict a whole ``[T, ...]`` recording.

    The window length is the model's ``max_seq_length``; ``stride`` defaults
    to the one stored in the model's ``preprocessing`` metadata, else half a
    window. Keypoints must be prepared like the model's training data (see
    ``model.config.keypoints`` / ``model.config.preprocessing``).
    """
    cfg = model.config
    L = cfg.max_seq_length
    stride = stride or cfg.preprocessing.get("stride") or L // 2
    if not 0 < stride <= L:
        raise ValueError(f"stride must be between 1 and the window length ({L}), got {stride}")
    x = torch.as_tensor(np.asarray(keypoints), dtype=torch.float32)
    if x.shape[0] == 0:
        raise ValueError("no frames to predict")
    x = x.reshape(x.shape[0], -1)
    if x.shape[1] != cfg.input_dim:
        raise ValueError(f"model expects {cfg.input_dim} values per frame, got {x.shape[1]}")
    w = sliding_windows(x, None, L, stride, pad_remainder=True)
    p = predict_windows(model, w, batch_size)
    if cfg.mode == "token":
        frame_p = stitch_window_scores(p, w.starts, w.mask, x.shape[0]).numpy()
        return RecordingPrediction(frame_p, frame_p.argmax(-1), w.starts.numpy())
    label, mean = recording_decision(p.numpy())
    return RecordingPrediction(p.numpy(), p.numpy().argmax(-1), w.starts.numpy(), label, mean)
