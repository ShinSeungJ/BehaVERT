"""PyTorch dataset over windows, with context-aware augmentation."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .recordings import Recording
from .windows import IGNORE_INDEX, Windows, sliding_windows, windows_of_many


def windows_from_recordings(
    recordings: list[Recording],
    length: int,
    stride: Optional[int] = None,
    pad_remainder: bool = True,
    concat: bool = False,
) -> tuple[Windows, torch.Tensor]:
    """Window every recording separately (default) and stack the results.

    ``concat=True`` joins all recordings into one stream first, so windows can
    span two recordings; the paper's CalMS21 experiments did this.

    Returns the windows and, per window, the index of its recording
    (``-1`` everywhere when ``concat=True``).
    """
    if concat:
        frames = torch.cat([r.keypoints for r in recordings])
        lab = [r.labels_per_frame() for r in recordings]
        labels = None if any(l is None for l in lab) else torch.cat(lab)
        w = sliding_windows(frames, labels, length, stride, pad_remainder)
        return w, torch.full((len(w),), -1, dtype=torch.long)
    labels = [r.labels_per_frame() for r in recordings]
    return windows_of_many([r.keypoints for r in recordings], labels, length, stride, pad_remainder)


class WindowDataset(Dataset):
    """Windows plus optional context-aware augmentation.

    Augmentation follows the paper: a window is used as-is the first time it
    is drawn and augmented on every later draw (the stored data is never
    changed). Only windows that contain one of ``augment_classes`` are
    augmented (``None`` = all windows).

    Items are dicts with ``input [L, F]``, ``mask [L]``, ``target`` (``[L]``
    frame labels, or the window's label when ``sequence_labels=True``) and
    ``index``.
    """

    def __init__(
        self,
        windows: Windows,
        augment_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        augment_classes: Optional[Iterable[int]] = None,
        sequence_labels: bool = False,
    ):
        self.w = windows
        self.augment_fn = augment_fn
        self.sequence_labels = sequence_labels
        # shared memory, so the "first use" bookkeeping also works with DataLoader workers
        self.uses = torch.zeros(len(windows), dtype=torch.int32).share_memory_()
        if augment_fn is None or windows.labels is None or augment_classes is None:
            self.augmentable = np.ones(len(windows), dtype=bool)
        else:
            cls = torch.tensor(sorted(set(augment_classes)), dtype=windows.labels.dtype)
            self.augmentable = torch.isin(windows.labels, cls).any(1).numpy()

    def __len__(self) -> int:
        return len(self.w)

    def __getitem__(self, i: int) -> dict:
        x = self.w.inputs[i]
        if self.augment_fn is not None and self.augmentable[i]:
            if self.uses[i] >= 1:
                n = int(self.w.mask[i].sum())  # real frames come first; the padding stays zero
                if n == len(x):
                    x = self.augment_fn(x)
                elif n > 0:
                    x = x.clone()
                    x[:n] = self.augment_fn(x[:n])
            self.uses[i] += 1
        item = {"input": x, "mask": self.w.mask[i], "index": i}
        if self.w.labels is not None:
            y = self.w.labels[i]
            item["target"] = y[0] if self.sequence_labels else y
        return item

    # ------------------------------------------------------------ statistics
    def window_classes(self) -> list[np.ndarray]:
        """Classes present in each window (padding excluded)."""
        if self.w.labels is None:
            raise ValueError("unlabeled dataset")
        if self.sequence_labels:
            return [np.array([int(v)]) for v in self.w.labels[:, 0]]
        return [np.unique(y[y != IGNORE_INDEX].numpy()) for y in self.w.labels]

    def class_proportions(self, num_classes: int) -> list[float]:
        """Fraction of labelled frames per class (window labels in sequence mode)."""
        y = self.w.labels[:, 0] if self.sequence_labels else self.w.labels.reshape(-1)
        y = y[y != IGNORE_INDEX].long()
        counts = torch.bincount(y, minlength=num_classes)[:num_classes].double()
        return (counts / counts.sum()).tolist()


def subset_windows(w: Windows, idx) -> Windows:
    idx = torch.as_tensor(idx, dtype=torch.long)
    return Windows(
        inputs=w.inputs[idx], mask=w.mask[idx],
        labels=None if w.labels is None else w.labels[idx], starts=w.starts[idx],
    )


class ProportionalBatchSampler(Sampler):
    """Class-balanced batches (paper Algorithm 1, Eq. 9).

    For every batch, each class ``c`` contributes ``ceil(p_c * B)`` windows
    drawn with replacement from the windows that contain ``c``; the pool is
    shuffled and cut to ``B``. Rare behaviors therefore appear in every batch.
    One epoch has ``ceil(N / B)`` batches.
    """

    def __init__(self, window_classes: list[np.ndarray], batch_size: int, proportions: list[float]):
        self.batch_size = batch_size
        self.proportions = np.asarray(proportions, dtype=float)
        self.pools = {c: [] for c in range(len(proportions))}
        for i, classes in enumerate(window_classes):
            for c in classes:
                if 0 <= c < len(proportions):
                    self.pools[int(c)].append(i)
        self.pools = {c: np.asarray(v) for c, v in self.pools.items() if len(v)}
        if not self.pools or not any(np.ceil(self.proportions[c] * batch_size) > 0 for c in self.pools):
            raise ValueError("no window contains a labelled class in range; check the labels and num_classes")
        n = len(window_classes)
        self.num_batches = n // batch_size + (1 if n % batch_size else 0)

    def __iter__(self):
        for _ in range(self.num_batches):
            batch: list[int] = []
            while len(batch) < self.batch_size:
                for c, pool in self.pools.items():
                    k = int(np.ceil(self.proportions[c] * self.batch_size))
                    if k > 0:
                        batch.extend(np.random.choice(pool, k, replace=True).tolist())
            np.random.shuffle(batch)
            yield batch[: self.batch_size]

    def __len__(self) -> int:
        return self.num_batches


def collate(batch: list[dict]) -> dict:
    out = {"input": torch.stack([b["input"] for b in batch]), "mask": torch.stack([b["mask"] for b in batch])}
    if "target" in batch[0]:
        out["target"] = torch.stack([torch.as_tensor(b["target"]) for b in batch])
    return out
