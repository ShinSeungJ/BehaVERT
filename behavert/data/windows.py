"""Cut recordings into fixed-length windows (paper Section 3.3, temporal unfolding).

A recording is a ``[T, F]`` array of flattened keypoints (``F`` = animals x
keypoints x coordinates) with optional per-frame labels ``[T]``. The model sees
windows of ``length`` frames, taken every ``stride`` frames.

* ``stride = length // 2`` gives 50% overlapping windows (temporal unfolding).
* ``stride = length`` gives non-overlapping chunks.

With ``pad_remainder=True`` the frames left after the last full window are
kept as one more window, zero-padded at the end. Padded frames get
``attention_mask = 0`` and label ``-1`` (ignored by losses and metrics).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

IGNORE_INDEX = -1


@dataclass
class Windows:
    inputs: torch.Tensor  # [N, L, F] float32
    mask: torch.Tensor  # [N, L] long, 1 = real frame, 0 = padding
    labels: Optional[torch.Tensor]  # [N, L] long (float for regression), -1 on padding; None if unlabeled
    starts: torch.Tensor  # [N] long, first frame of each window in the recording

    def __len__(self) -> int:
        return self.inputs.shape[0]


def _as_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def sliding_windows(
    frames,
    labels=None,
    length: int = 64,
    stride: Optional[int] = None,
    pad_remainder: bool = True,
) -> Windows:
    """Split one recording into windows.

    Args:
        frames: ``[T, F]`` (or ``[T, ...]``, flattened to ``[T, F]``).
        labels: optional ``[T]`` labels per frame (integer classes, or floats
            for regression; the dtype is kept).
        length: window length in frames.
        stride: step between window starts (default ``length // 2``).
        pad_remainder: when frames remain after the last full window, add one
            more window for them, starting at ``n_full * stride`` and zero-padded
            at the end (mask 0, label ``IGNORE_INDEX``).
    """
    x = _as_tensor(frames, torch.float32)
    x = x.reshape(x.shape[0], -1)
    T, F = x.shape
    stride = stride or max(1, length // 2)
    y = None if labels is None else _as_tensor(labels).reshape(-1)
    if y is not None and not torch.is_floating_point(y):
        y = y.long()
    if y is not None and y.shape[0] != T:
        raise ValueError(f"labels have {y.shape[0]} frames but keypoints have {T}")

    n_full = 0 if T < length else (T - length) // stride + 1
    starts = [i * stride for i in range(n_full)]
    xs = [x[s : s + length] for s in starts]
    ms = [torch.ones(length, dtype=torch.long) for _ in starts]
    ys = [y[s : s + length] for s in starts] if y is not None else None

    rest_start = n_full * stride
    if pad_remainder and _leftover(T, length, stride, n_full):
        n_real = T - rest_start
        pad = length - n_real
        xs.append(torch.cat([x[rest_start:], x.new_zeros(pad, F)]))
        ms.append(torch.cat([torch.ones(n_real, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))
        if ys is not None:
            ys.append(torch.cat([y[rest_start:], y.new_full((pad,), IGNORE_INDEX)]))
        starts.append(rest_start)

    if not xs:
        return Windows(
            inputs=x.new_zeros(0, length, F),
            mask=torch.zeros(0, length, dtype=torch.long),
            labels=None if y is None else y.new_zeros(0, length),
            starts=torch.zeros(0, dtype=torch.long),
        )
    return Windows(
        inputs=torch.stack(xs),
        mask=torch.stack(ms),
        labels=None if ys is None else torch.stack(ys),
        starts=torch.tensor(starts, dtype=torch.long),
    )


def _leftover(T: int, length: int, stride: int, n_full: int) -> bool:
    """Whether frames remain after the last of ``n_full`` full windows."""
    covered = (n_full - 1) * stride + length if n_full else 0
    return covered < T


def window_starts(T: int, length: int, stride: int, pad_remainder: bool) -> tuple[list[int], int]:
    """Start frames of the windows :func:`sliding_windows` would cut, and how
    many of them are full (the rest is the padded leftover window)."""
    n_full = 0 if T < length else (T - length) // stride + 1
    starts = [i * stride for i in range(n_full)]
    if pad_remainder and _leftover(T, length, stride, n_full):
        starts.append(n_full * stride)
    return starts, n_full


def windows_of_many(recordings_frames, recordings_labels, length: int, stride: Optional[int] = None,
                    pad_remainder: bool = True) -> tuple[Windows, torch.Tensor]:
    """Window several recordings into pre-allocated tensors (low peak memory).

    Same windows as calling :func:`sliding_windows` per recording and
    concatenating. Returns the windows and the recording index of each window.
    """
    stride = stride or max(1, length // 2)
    plan = []
    for i, x in enumerate(recordings_frames):
        starts, _ = window_starts(x.shape[0], length, stride, pad_remainder)
        plan += [(i, s) for s in starts]
    F = recordings_frames[0].reshape(recordings_frames[0].shape[0], -1).shape[1] if recordings_frames else 0
    N = len(plan)
    has_labels = recordings_labels is not None and all(y is not None for y in recordings_labels)
    inputs = torch.zeros(N, length, F, dtype=torch.float32)
    mask = torch.zeros(N, length, dtype=torch.long)
    labels = None
    if has_labels:
        dtype = recordings_labels[0].dtype if len(recordings_labels) else torch.long
        labels = torch.full((N, length), IGNORE_INDEX, dtype=dtype)
    for k, (i, s) in enumerate(plan):
        x = recordings_frames[i]
        n = min(length, x.shape[0] - s)
        inputs[k, :n] = x[s : s + n].reshape(n, -1)
        mask[k, :n] = 1
        if has_labels:
            labels[k, :n] = recordings_labels[i][s : s + n]
    owner = torch.tensor([i for i, _ in plan], dtype=torch.long)
    starts = torch.tensor([s for _, s in plan], dtype=torch.long)
    return Windows(inputs=inputs, mask=mask, labels=labels, starts=starts), owner


def concat_windows(parts: list[Windows]) -> Windows:
    parts = [p for p in parts if len(p)]
    if not parts:
        raise ValueError("no windows")
    has_labels = parts[0].labels is not None
    return Windows(
        inputs=torch.cat([p.inputs for p in parts]),
        mask=torch.cat([p.mask for p in parts]),
        labels=torch.cat([p.labels for p in parts]) if has_labels else None,
        starts=torch.cat([p.starts for p in parts]),
    )


def stitch_window_scores(scores: torch.Tensor, starts: torch.Tensor, mask: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Average per-frame scores of overlapping windows back onto the recording.

    Args:
        scores: ``[N, L, C]`` per-window, per-frame scores (e.g. probabilities).
        starts: ``[N]`` window start frames.
        mask: ``[N, L]`` real-frame mask.
        num_frames: length ``T`` of the recording.

    Returns:
        ``[T, C]`` averaged scores (frames never covered are 0).
    """
    N, L, C = scores.shape
    out = scores.new_zeros(num_frames, C)
    count = scores.new_zeros(num_frames, 1)
    for i in range(N):
        n = int(mask[i].sum())
        s = int(starts[i])
        n = min(n, num_frames - s)
        out[s : s + n] += scores[i, :n]
        count[s : s + n] += 1
    return out / count.clamp(min=1)
