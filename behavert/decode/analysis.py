"""Stage 2 — run BehaVERT on a keypoint sequence and keep everything the
visualizations need, from a single pass over sliding windows:

* per-frame class probabilities, predicted label and confidence (token
  models: logits of overlapping windows averaged, then softmax; sequence
  models: each frame gets the mean probabilities of the windows covering it,
  and its confidence is the mean of those windows' top probability),
* per-frame attention from the last layer (heads averaged): for token models
  the attention each frame *receives*, for sequence models the attention from
  ``[CLS]`` to each frame,
* one embedding per window (token models: mean of the frame embeddings;
  sequence models: the pooled ``[CLS]`` embedding).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..data.windows import Windows
from ..metrics import recording_decision
from ..model import BehaVERT


@dataclass
class Analysis:
    probs: np.ndarray  # [T, C] per-frame class probabilities
    labels: np.ndarray  # [T] predicted class
    confidence: np.ndarray  # [T] probability of the predicted class
    attention: np.ndarray  # [T]
    window_starts: np.ndarray  # [N]
    window_lengths: np.ndarray  # [N] real (unpadded) frames
    window_embeddings: np.ndarray  # [N, H]
    window_probs: np.ndarray  # [N, C] (token models: mean over the window's frames)
    recording_probs: Optional[np.ndarray] = None  # [C] sequence models: whole-recording decision
    recording_label: Optional[int] = None

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in fields(self)
                                     if getattr(self, f.name) is not None})

    @classmethod
    def load(cls, path: str | Path) -> "Analysis":
        z = np.load(path)
        kw = {f.name: z[f.name] if f.name in z.files else None for f in fields(cls)}
        if kw["recording_label"] is not None:
            kw["recording_label"] = int(kw["recording_label"])
        return cls(**kw)


@torch.no_grad()
def analyze(model: BehaVERT, frames: np.ndarray, stride: Optional[int] = None, batch_size: int = 64) -> Analysis:
    """Run ``model`` over ``frames`` ``[T, input_dim]``."""
    cfg = model.config
    if cfg.problem != "classification":
        raise ValueError("decode supports classification models")
    if frames.shape[1] != cfg.input_dim:
        raise ValueError(f"model expects {cfg.input_dim} values per frame, got {frames.shape[1]}")
    L = cfg.max_seq_length
    stride = stride or cfg.preprocessing.get("stride") or L // 2
    if not 0 < stride <= L:
        raise ValueError(f"stride must be between 1 and the window length ({L}), got {stride}")
    if len(frames) == 0:
        raise ValueError("no frames to analyse")
    T, C = len(frames), cfg.num_classes
    w = covering_windows(frames, L, stride)
    dev = next(model.parameters()).device
    model.eval()

    score_sum = np.zeros((T, C)); att_sum = np.zeros(T); conf_sum = np.zeros(T); cnt = np.zeros(T)
    embs, wprobs = [], []
    for i in range(0, len(w), batch_size):
        x = w.inputs[i : i + batch_size].to(dev)
        m = w.mask[i : i + batch_size].to(dev)
        out = model(x, attention_mask=m, output_attentions=True)
        p = out.logits.float().softmax(-1)
        att = out.attentions[-1].float().mean(1)  # [B, S, S] heads averaged
        mf = m.float()
        if cfg.mode == "token":
            # attention received by each frame, averaged over the frames attending to it
            recv = (att * mf[:, :, None]).sum(1) / mf.sum(1, keepdim=True)
            emb = (out.last_hidden_state.float() * mf[..., None]).sum(1) / mf.sum(1, keepdim=True)
            wp = (p * mf[..., None]).sum(1) / mf.sum(1, keepdim=True)
            frame_score = out.logits.float()  # logits are averaged over windows
            frame_conf = p.max(-1).values
        else:
            recv = att[:, 0, 1:]  # [CLS] -> frames
            emb = out.embedding.float()
            wp = p
            frame_score = p[:, None, :].expand(-1, L, -1)
            frame_conf = p.max(-1).values[:, None].expand(-1, L)
        recv, frame_score, frame_conf = recv.cpu().numpy(), frame_score.cpu().numpy(), frame_conf.cpu().numpy()
        for k in range(x.shape[0]):
            s = int(w.starts[i + k]); n = int(m[k].sum())
            n = min(n, T - s)
            score_sum[s : s + n] += frame_score[k, :n]
            conf_sum[s : s + n] += frame_conf[k, :n]
            att_sum[s : s + n] += recv[k, :n]
            cnt[s : s + n] += 1
        embs.append(emb.cpu().numpy()); wprobs.append(wp.cpu().numpy())

    cnt = np.maximum(cnt, 1)
    if cfg.mode == "token":
        z = score_sum / cnt[:, None]
        probs = np.exp(z - z.max(1, keepdims=True))
        probs /= probs.sum(1, keepdims=True)
        confidence = probs.max(1)
    else:
        probs = score_sum / cnt[:, None]
        confidence = conf_sum / cnt
    window_probs = np.concatenate(wprobs)
    res = Analysis(
        probs=probs.astype(np.float32),
        labels=probs.argmax(1),
        confidence=confidence.astype(np.float32),
        attention=(att_sum / cnt).astype(np.float32),
        window_starts=w.starts.numpy(),
        window_lengths=w.mask.sum(1).numpy(),
        window_embeddings=np.concatenate(embs).astype(np.float32),
        window_probs=window_probs.astype(np.float32),
    )
    if cfg.mode == "sequence":
        res.recording_label, res.recording_probs = recording_decision(window_probs)
        res.recording_probs = res.recording_probs.astype(np.float32)
    return res


def covering_windows(frames: np.ndarray, length: int, stride: int) -> Windows:
    """Windows every ``stride`` frames plus one ending exactly at the last frame,
    so every frame is covered without padding (one padded window if the
    recording is shorter than ``length``)."""
    x = torch.as_tensor(np.asarray(frames), dtype=torch.float32)
    T, F = x.shape
    if T <= length:
        inputs = torch.zeros(1, length, F); inputs[0, :T] = x
        mask = torch.zeros(1, length, dtype=torch.long); mask[0, :T] = 1
        return Windows(inputs, mask, None, torch.zeros(1, dtype=torch.long))
    starts = list(range(0, T - length + 1, stride))
    if starts[-1] + length < T:
        starts.append(T - length)
    idx = torch.tensor(starts)[:, None] + torch.arange(length)[None]
    return Windows(x[idx], torch.ones(len(starts), length, dtype=torch.long), None,
                   torch.tensor(starts, dtype=torch.long))


def count_episodes(labels: np.ndarray, cls: int) -> int:
    hit = (labels == cls).astype(np.int8)
    return int(((hit[1:] == 1) & (hit[:-1] == 0)).sum() + (hit[:1] == 1).sum()) if len(hit) else 0


def summary_rows(res: Analysis, label_names: list[str], fps: float) -> list[dict]:
    """Frames, seconds, percent and episodes of every class."""
    T = len(res.labels)
    rows = []
    for c, n in enumerate(label_names):
        frames = int((res.labels == c).sum())
        rows.append({"behavior": n, "frames": frames, "seconds": round(frames / fps, 2),
                     "percent": round(100 * frames / max(T, 1), 3), "episodes": count_episodes(res.labels, c)})
    return rows


def write_tables(res: Analysis, label_names: list[str], fps: float, out_dir: Path) -> None:
    """``behavior.csv`` (one row per frame) and ``summary.csv`` (one row per behavior)."""
    import pandas as pd

    names = np.asarray(label_names)
    T = len(res.labels)
    df = pd.DataFrame({
        "frame": np.arange(T),
        "time_s": np.round(np.arange(T) / fps, 4),
        "behavior": names[res.labels],
        "confidence": np.round(res.confidence, 4),
        "attention": res.attention,
    })
    for c, n in enumerate(label_names):
        df[f"p_{n}"] = np.round(res.probs[:, c], 4)
    df.to_csv(out_dir / "behavior.csv", index=False)

    summary = pd.DataFrame(summary_rows(res, label_names, fps))
    if res.recording_label is not None:
        summary["recording_probability"] = np.round(res.recording_probs, 4)
        summary["recording_prediction"] = [n == label_names[res.recording_label] for n in label_names]
    summary.to_csv(out_dir / "summary.csv", index=False)
