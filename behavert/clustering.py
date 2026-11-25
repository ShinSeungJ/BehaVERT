"""Embedding map and clustering quality over a set of recordings.

Embeddings are taken per frame for token models (last-layer frame vectors)
and per window for sequence models (the ``[CLS]`` vector), reduced with PCA
and projected to 2D with t-SNE or UMAP.

``embedding.csv``  one row per point: recording, frame or window, x, y, true and predicted class
``clustering.csv`` NMI, purity, accuracy and macro F1 of the model's predictions
``embedding.png`` + ``embedding.svg`` / ``.pdf`` scatter colored by class (text stays editable)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from .data.windows import sliding_windows
from .model import BehaVERT


@torch.no_grad()
def collect_embeddings(model: BehaVERT, recordings, stride: Optional[int] = None, batch_size: int = 64,
                       max_points: Optional[int] = None, seed: int = 0):
    """Returns ``(embeddings [N, H], DataFrame with recording / position / true / pred / confidence)``.

    With ``max_points``, token-model frames are thinned while collecting (to about
    twice ``max_points``) so memory stays bounded on large datasets.
    """
    cfg = model.config
    rng = np.random.default_rng(seed)
    total = sum(r.num_frames for r in recordings)
    keep_p = 1.0 if not max_points else min(1.0, 2.0 * max_points / max(total, 1))
    L = cfg.max_seq_length
    stride = stride or L
    dev = next(model.parameters()).device
    model.eval()
    embs, rows = [], []
    for rec in recordings:
        y = rec.labels_per_frame()
        w = sliding_windows(rec.keypoints, y, L, stride, pad_remainder=True)
        for i in range(0, len(w), batch_size):
            x = w.inputs[i : i + batch_size].to(dev)
            m = w.mask[i : i + batch_size].to(dev)
            out = model(x, attention_mask=m)
            p = out.logits.float().softmax(-1).cpu().numpy()
            if cfg.mode == "sequence":
                e = out.embedding.float().cpu().numpy()
                for k in range(x.shape[0]):
                    j = i + k
                    embs.append(e[k])
                    rows.append({"recording": rec.name, "window": j, "start": int(w.starts[j]),
                                 "true": int(w.labels[j, 0]) if w.labels is not None else -1,
                                 "pred": int(p[k].argmax()), "confidence": float(p[k].max())})
            else:
                h = out.last_hidden_state.float().cpu().numpy()
                for k in range(x.shape[0]):
                    j = i + k
                    n = int(m[k].sum()); s = int(w.starts[j])
                    sel = np.flatnonzero(rng.random(n) < keep_p) if keep_p < 1 else np.arange(n)
                    if not len(sel):
                        continue
                    embs.append(h[k, sel])
                    rows.append(pd.DataFrame({
                        "recording": rec.name, "window": j, "frame": s + sel,
                        "true": w.labels[j, sel].numpy() if w.labels is not None else -1,
                        "pred": p[k, sel].argmax(1), "confidence": p[k, sel].max(1)}))
    if cfg.mode == "sequence":
        return np.stack(embs), pd.DataFrame(rows)
    return np.concatenate(embs), pd.concat(rows, ignore_index=True)


def purity(y_true: np.ndarray, clusters: np.ndarray) -> float:
    """Fraction of points that belong to the most common class of their cluster."""
    total = 0
    for c in np.unique(clusters):
        _, counts = np.unique(y_true[clusters == c], return_counts=True)
        total += counts.max()
    return total / max(len(y_true), 1)


def clustering_metrics(table: pd.DataFrame) -> pd.DataFrame:
    """NMI, purity, accuracy and macro F1 of the model's predictions against the true classes."""
    from sklearn.metrics import accuracy_score, f1_score, normalized_mutual_info_score

    y, pred = table["true"].values, table["pred"].values
    return pd.DataFrame([
        {"clusters": "model predictions", "nmi": normalized_mutual_info_score(y, pred), "purity": purity(y, pred),
         "accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
         "n": len(y)},
    ])


def project(emb: np.ndarray, method: str = "tsne", seed: int = 0, perplexity: float = 30.0) -> np.ndarray:
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=min(50, emb.shape[0] - 1, emb.shape[1]), random_state=seed).fit_transform(emb)
    if method == "umap":
        try:
            import umap
        except ImportError as e:  # pragma: no cover
            raise ImportError("UMAP needs: pip install umap-learn") from e
        return umap.UMAP(n_components=2, random_state=seed).fit_transform(reduced)
    from sklearn.manifold import TSNE

    perplexity = min(perplexity, max(2.0, (len(reduced) - 1) / 3.0))
    return TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(reduced)


def plot_embedding(xy: np.ndarray, classes: np.ndarray, names: Sequence[str], colors: Sequence[str],
                   out_base: Path, title: str = "", method: str = "tsne") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["svg.fonttype"] = "none"  # keep text editable in Illustrator / Inkscape
    plt.rcParams["pdf.fonttype"] = 42
    fig, ax = plt.subplots(figsize=(6, 6))
    for c in np.unique(classes):
        sel = classes == c
        ax.scatter(xy[sel, 0], xy[sel, 1], s=4, alpha=0.7, linewidths=0, color=colors[c % len(colors)],
                   label=names[c] if 0 <= c < len(names) else str(c), rasterized=sel.sum() > 5000)
    label = "UMAP" if method == "umap" else "t-SNE"
    ax.set_xlabel(f"{label} 1"); ax.set_ylabel(f"{label} 2")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_box_aspect(1)
    ax.legend(markerscale=3, frameon=False, loc="best", fontsize=9)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=300 if ext == "png" else None)
    plt.close(fig)


def run_embedding_analysis(model: BehaVERT, recordings, out_dir: str | Path, method: str = "tsne",
                           stride: Optional[int] = None, exclude_classes: Sequence[int] = (),
                           max_points: Optional[int] = 20000, color_by: str = "true", seed: int = 0,
                           title: str = "", batch_size: int = 64) -> pd.DataFrame:
    from .decode.render import behavior_colors

    cfg = model.config
    names = list(cfg.label_names) or [str(c) for c in range(cfg.num_classes)]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb, table = collect_embeddings(model, recordings, stride, batch_size, max_points, seed)
    keep = (table["true"].values >= 0) & ~np.isin(table["true"].values, list(exclude_classes))
    emb, table = emb[keep], table[keep].reset_index(drop=True)
    if max_points and len(table) > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(len(table), max_points, replace=False))
        emb, table = emb[idx], table.iloc[idx].reset_index(drop=True)
    metrics = clustering_metrics(table)
    xy = project(emb, method, seed)
    table.insert(table.columns.get_loc("true"), "x", xy[:, 0])
    table.insert(table.columns.get_loc("true"), "y", xy[:, 1])
    table["true_name"] = [names[c] for c in table["true"]]
    table["pred_name"] = [names[c] for c in table["pred"]]
    table.to_csv(out_dir / "embedding.csv", index=False, float_format="%.6g")
    metrics.to_csv(out_dir / "clustering.csv", index=False, float_format="%.6g")
    plot_embedding(xy, table[color_by].values, names, behavior_colors(names, cfg.label_colors),
                   out_dir / "embedding", title, method)
    return metrics
