"""2D map of window embeddings (PCA then t-SNE) for the synchronized video.

The map holds the windows of the rendered clip and, optionally, a backdrop of
reference windows (one from the middle of each of N recordings of the
dataset) so the clip can be seen against the dataset's overall structure.
How it is drawn comes from the model's ``display["embedding"]`` settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .analysis import Analysis


@dataclass
class Reference:
    embeddings: np.ndarray  # [R, H]
    labels: np.ndarray  # [R] true label of each recording
    probs: np.ndarray  # [R, C]


@dataclass
class EmbeddingMap:
    xy: np.ndarray  # [N, 2] reference points first, then the clip's windows
    color_index: np.ndarray  # [N] class used for the point color
    alpha: np.ndarray  # [N]
    frame_to_point: np.ndarray  # [T_clip] point of the window closest to each clip frame
    point_to_frame: np.ndarray  # [N] centre frame of each clip window (-1 for reference points)
    color_value: Optional[np.ndarray] = None  # [N] binary models: P(class 1)


@torch.no_grad()
def reference_embeddings(model, recordings, count: int, seed: int = 0) -> Reference:
    """One window from the middle of each of ``count`` recordings (sampled by name)."""
    L = model.config.max_seq_length
    dev = next(model.parameters()).device
    by_name = {r.name: r for r in recordings if r.num_frames >= L}
    names = sorted(by_name)
    chosen = np.random.default_rng(seed).choice(names, size=min(count, len(names)), replace=False)
    embs, labels, probs = [], [], []
    model.eval()
    for n in chosen:
        r = by_name[n]
        s = max(0, r.num_frames // 2 - L // 2)
        x = r.keypoints[s : s + L][None].float().to(dev)
        out = model(x, attention_mask=torch.ones(1, L, dtype=torch.long, device=dev))
        e = out.embedding[0] if model.config.mode == "sequence" else out.last_hidden_state[0].mean(0)
        embs.append(e.float().cpu().numpy())
        probs.append(out.logits.float().softmax(-1).reshape(-1, out.logits.shape[-1]).mean(0).cpu().numpy())
        y = r.labels_per_frame()
        labels.append(int(r.label) if r.label is not None else int(torch.mode(y).values) if y is not None else -1)
    return Reference(np.stack(embs), np.array(labels), np.stack(probs))


def embedding_map(res: Analysis, mode: str, num_classes: int, settings: Optional[dict] = None,
                  reference: Optional[Reference] = None) -> EmbeddingMap:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    s = settings or {}
    R = 0 if reference is None else len(reference.embeddings)
    emb = res.window_embeddings if R == 0 else np.concatenate([reference.embeddings, res.window_embeddings])
    n = len(emb)
    if n >= 4:
        reduced = PCA(n_components=min(50, n - 1, emb.shape[1])).fit_transform(emb)
        perplexity = max(float(s.get("perplexity_min", 2)), min(30.0, (n - 1) / 3.0))
        xy = TSNE(n_components=2, perplexity=perplexity, max_iter=int(s.get("max_iter", 400)), init="pca",
                  random_state=int(s.get("seed", 0))).fit_transform(reduced)
    else:
        xy = np.zeros((n, 2))
        xy[:, 0] = np.arange(n)

    starts, lengths = res.window_starts, res.window_lengths
    if mode == "token":  # majority predicted class and mean confidence over the window's frames
        color = np.array([np.bincount(res.labels[a : a + l], minlength=num_classes).argmax()
                          for a, l in zip(starts, lengths)])
        alpha = np.array([res.confidence[a : a + l].mean() for a, l in zip(starts, lengths)])
    else:  # the window's own prediction
        color = res.window_probs.argmax(1)
        alpha = res.window_probs.max(1)
    if s.get("alpha_min") is not None:
        alpha = np.clip(alpha, float(s["alpha_min"]), 1.0)
    value = res.window_probs[:, 1] if (mode == "sequence" and num_classes == 2) else None

    centres = (starts + lengths // 2).astype(np.int64)
    if R:
        color = np.concatenate([reference.labels, color])
        alpha = np.concatenate([np.full(R, float(s.get("reference_alpha", 0.75))), alpha])
        if value is not None:
            value = np.concatenate([reference.probs[:, 1], value])
    T = int(starts[-1] + lengths[-1]) if len(starts) else 0
    t = np.arange(T)
    right = np.clip(np.searchsorted(centres, t), 0, len(centres) - 1)
    left = np.clip(right - 1, 0, len(centres) - 1)
    nearest = np.where(np.abs(t - centres[left]) <= np.abs(centres[right] - t), left, right)
    return EmbeddingMap(
        xy=xy.astype(np.float32), color_index=color.astype(np.int64), alpha=alpha.astype(np.float32),
        frame_to_point=(R + nearest).astype(np.int64),
        point_to_frame=np.concatenate([np.full(R, -1), centres]).astype(np.int64),
        color_value=None if value is None else value.astype(np.float32),
    )
