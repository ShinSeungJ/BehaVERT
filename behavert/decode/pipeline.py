"""``behavert decode``: videos or keypoints -> behavior, embeddings, attention, videos.

For every input recording, results go to ``<out>/<name>/``::

    keypoints.csv   per-frame keypoints the model reads (written when they come from a video)
    behavior.csv    per-frame behavior, confidence, attention and class probabilities
    summary.csv     frames, seconds, percent and episodes per behavior
    analysis.npz    everything above plus one embedding per window
    overlay.mp4     skeleton + behavior on the video            (--overlay)
    sync.mp4        video, embedding map, attention, confidence  (--sync)

A stage is skipped when its output exists (``force=True`` recomputes).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..model import BehaVERT
from .analysis import Analysis, analyze, write_tables
from .layout import KeypointLayout

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".mpg", ".mpeg", ".m4v", ".wmv")


@dataclass
class Item:
    name: str  # output folder name
    video: Optional[Path] = None
    frames: Optional[np.ndarray] = None  # [T, F] if already known
    ground_truth: Optional[np.ndarray] = None
    key: Optional[str] = None  # recording name inside its dataset (benchmark inputs)


# ------------------------------------------------------------------ inputs
def find_videos(path: str | Path) -> list[tuple[str, Path]]:
    p = Path(path)
    if p.is_file():
        return [(p.stem, p)]
    vids = sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_EXTENSIONS)
    if not vids:
        raise FileNotFoundError(f"no videos under {p}")
    return [(str(v.relative_to(p).with_suffix("")), v) for v in vids]


def read_keypoint_file(path: str | Path, recording: Optional[str] = None) -> tuple[str, np.ndarray, Optional[np.ndarray]]:
    """``(name, frames [T, F], labels or None)`` from .csv / .npy / .npz / .pt / .pkl."""
    import pandas as pd

    from ..data.recordings import _has_header, load_recordings

    p = Path(path)
    if p.suffix == ".csv":
        df = pd.read_csv(p, header=0 if _has_header(p) else None)
        skip = {"frame", "time", "time_s", "timestamp", "label", "behavior"}
        df = df[[c for c in df.columns if str(c).lower() not in skip]]
        return p.stem, df.values.astype(np.float32), None
    if p.suffix == ".npy":
        a = np.load(p)
        return p.stem, a.reshape(len(a), -1).astype(np.float32), None
    recs = load_recordings(p)
    if recording is None and len(recs) > 1:
        raise SystemExit(f"{p} holds {len(recs)} recordings; pick one with --recording "
                         f"(e.g. {', '.join(r.name for r in recs[:3])})")
    rec = recs[0] if recording is None else next((r for r in recs if r.name == recording), None)
    if rec is None:
        raise SystemExit(f"recording {recording!r} not found in {p}")
    y = rec.labels_per_frame()
    return rec.name, rec.keypoints.numpy(), None if y is None else y.numpy()


def read_benchmark_recording(config: str, recording: str, data_root: str, task: Optional[str] = None):
    """A recording (and its ground truth) from a training config's dataset, plus all its recordings."""
    from ..recipes import load_config, load_split_recordings

    cfg = load_config(config, task)
    splits = load_split_recordings(cfg, data_root, cfg.get("train", {}).get("seed", 42))
    pool = {r.name: r for recs in splits.values() for r in recs or []}
    for r in pool.values():
        if r.name == recording or r.name.endswith("/" + recording):
            y = r.labels_per_frame()
            return r.name, r.keypoints.numpy(), None if y is None else y.numpy(), list(pool.values())
    raise SystemExit(f"recording {recording!r} not found in {config}")


def read_extra_strip(spec: dict, key: str, num_frames: int, data_root: str, cache: dict) -> Optional[dict]:
    """Per-frame label strip from dataset files: ``spec["labels"] = [[name, color, path], ...]``
    (paths relative to ``data_root``); where several labels are active the earlier one wins."""
    import torch

    none_name, none_color = spec.get("none", ["none", "#ECECEC"])
    names, colors = [none_name], [none_color]
    labels = np.zeros(num_frames, dtype=np.int64)
    found = False
    for idx, (name, color, rel) in enumerate(spec["labels"], start=1):
        names.append(name)
        colors.append(color)
        path = Path(data_root) / rel
        if not path.exists():
            continue
        if str(path) not in cache:
            cache[str(path)] = torch.load(path, map_location="cpu", weights_only=False)
        v = cache[str(path)].get(key)
        if v is None:
            continue
        found = True
        v = np.asarray(v).astype(np.int64)[:num_frames]
        v = np.pad(v, (0, num_frames - len(v)))
        labels[(v > 0) & (labels == 0)] = idx
    if not found:
        return None
    return {"title": spec.get("title", "Labels"), "labels": labels, "names": names, "colors": colors,
            "legend": spec.get("legend", False)}


def read_labels(path: str | Path, label_names: Sequence[str]) -> np.ndarray:
    """Ground-truth labels: one value per frame (class index or name), .csv/.npy/.txt."""
    import pandas as pd

    p = Path(path)
    if p.suffix == ".npy":
        v = np.load(p)
    else:
        df = pd.read_csv(p)
        col = next((c for c in ("behavior", "label", "labels") if c in df.columns), df.columns[-1])
        v = df[col].values
    if v.dtype.kind in "OUS":
        lookup = {n: i for i, n in enumerate(label_names)}
        unknown = sorted({str(x) for x in v} - set(lookup))
        if unknown:
            raise SystemExit(f"{p}: labels {unknown[:5]} are not classes of this model ({', '.join(label_names)})")
        v = np.array([lookup[str(x)] for x in v])
    return v.astype(np.int64)


# ------------------------------------------------------------------ pipeline
def decode(
    model: str | BehaVERT,
    video: Optional[str | Path] = None,
    keypoints: Optional[str | Path] = None,
    config: Optional[str] = None,
    recording: Optional[str] = None,
    task: Optional[str] = None,
    data_root: Optional[str] = None,
    pose_model: Optional[str] = None,
    labels: Optional[str | Path] = None,
    out: str | Path = "outputs/decode",
    overlay: bool = False,
    sync: bool = False,
    start: int = 0,
    frames: Optional[int] = None,
    sync_frames: int = 900,
    stride: Optional[int] = None,
    fps: Optional[float] = None,
    device: Optional[str] = None,
    batch_size: int = 64,
    limit: Optional[int] = None,
    backdrop: Optional[int] = None,
    force: bool = False,
) -> list[Path]:
    """Run the decode pipeline; returns the output folder of each recording.

    Inputs (one of): ``video`` (a file or a folder of videos; keypoints come
    from ``pose_model``), ``keypoints`` (a keypoint file, optionally with
    ``video`` for drawing), or ``config`` + ``recording`` (a recording of a
    benchmark dataset, with its ground truth). ``limit`` processes only the
    first frames of each input. ``backdrop``
    overrides how many dataset recordings form the embedding-map backdrop
    (0 = none; default from the model's display settings).
    """
    import torch

    from ..recipes import DEFAULT_DATA_ROOT

    data_root = data_root or DEFAULT_DATA_ROOT
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    net = model if isinstance(model, BehaVERT) else BehaVERT.from_pretrained(model, device=dev)
    cfg = net.config
    layout = KeypointLayout.from_config(cfg)
    names = list(cfg.label_names) or [str(i) for i in range(cfg.num_classes)]

    items: list[Item] = []
    pool = None  # all recordings of the dataset (benchmark inputs)
    if config:
        if not recording:
            raise SystemExit("--config needs --recording")
        name, x, y, pool = read_benchmark_recording(config, recording, data_root, task)
        items.append(Item(Path(name).name, Path(video) if video else None, x, y, key=name))
    elif keypoints:
        name, x, y = read_keypoint_file(keypoints, recording)
        items.append(Item(Path(name).name, Path(video) if video else None, x, y, key=name))
    elif video:
        items = [Item(n, v) for n, v in find_videos(video)]
    else:
        raise SystemExit("give --video, --keypoints, or --config with --recording")
    if labels is not None:
        if len(items) != 1:
            raise SystemExit("--labels works with a single recording")
        items[0].ground_truth = read_labels(labels, names)
        n = len(items[0].frames) if items[0].frames is not None else None
        if n is not None and len(items[0].ground_truth) < min(n, limit or n):
            raise SystemExit(f"--labels has {len(items[0].ground_truth)} rows but the input has {n} frames")

    pose = None
    reference = None
    strip_cache: dict = {}
    out_dirs = []
    for it in items:
        d = Path(out) / it.name
        d.mkdir(parents=True, exist_ok=True)
        kp_csv = d / "keypoints.csv"
        rate = fps
        used_pose = None
        # reuse earlier outputs only if they were made the same way
        prev = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        same_input = prev.get("limit") == limit and prev.get("input") == str(it.video or keypoints or config)
        same_model = same_input and prev.get("model") == str(model) and prev.get("stride") == stride
        fresh = force or not same_input

        # 1. keypoints
        if it.frames is None:
            if kp_csv.exists() and not fresh:
                it.frames = read_keypoint_file(kp_csv)[1]
            else:
                from .pose import PoseModel, pose_to_model_frames

                pose_name = pose_model or cfg.keypoints.get("pose_model")
                if pose is None:
                    if not pose_name:
                        raise SystemExit("this model names no pose model; pass --pose-model")
                    pose = PoseModel(pose_name, device=dev)
                used_pose = pose_name
                p = pose.predict_video(it.video, max_frames=limit)
                it.frames = pose_to_model_frames(p, layout)
                rate = rate or p.fps
                _write_keypoints(kp_csv, it.frames, layout)
                print(f"[{it.name}] keypoints: {len(it.frames)} frames, "
                      f"{int((~p.detected).sum())} without detection (interpolated)")
        if limit and len(it.frames) > limit:
            it.frames = it.frames[:limit]
            if it.ground_truth is not None:
                it.ground_truth = it.ground_truth[:limit]
        if it.frames.shape[1] != cfg.input_dim:
            raise SystemExit(f"[{it.name}] {it.frames.shape[1]} values per frame, model expects {cfg.input_dim}")
        rate = rate or _video_fps(it.video) or layout.fps

        # 2. behavior, attention, embeddings
        npz = d / "analysis.npz"
        res = Analysis.load(npz) if npz.exists() and same_model and not force else None
        if res is None or len(res.labels) != len(it.frames):
            res = analyze(net, it.frames, stride=stride, batch_size=batch_size)
            res.save(npz)
            write_tables(res, names, rate, d)
        top = np.bincount(res.labels, minlength=len(names))
        msg = ", ".join(f"{names[c]} {100 * top[c] / len(res.labels):.1f}%" for c in np.argsort(-top)[:3])
        print(f"[{it.name}] {len(res.labels)} frames — {msg}"
              + (f" — recording: {names[res.recording_label]}" if res.recording_label is not None else ""))

        # 3. videos
        from .render import render_overlay, render_sync

        redo_videos = force or not same_model
        if overlay and (redo_videos or not (d / "overlay.mp4").exists()):
            render_overlay(d / "overlay.mp4", it.frames, res, layout, names, it.video, rate, start, frames,
                           label_colors=cfg.label_colors, style=(cfg.display or {}).get("overlay"))
            print(f"[{it.name}] wrote {d / 'overlay.mp4'}")
        if sync and (redo_videos or not (d / "sync.mp4").exists()):
            from .embedding import reference_embeddings

            disp = cfg.display or {}
            n_ref = int(backdrop if backdrop is not None
                        else (disp.get("embedding") or {}).get("reference_recordings", 0))
            if n_ref and pool and reference is None:
                print(f"[{it.name}] embedding backdrop from {n_ref} recordings of the dataset")
                reference = reference_embeddings(net, pool, n_ref)
            extra = None
            if disp.get("extra_strip") and it.key and config:
                extra = read_extra_strip(disp["extra_strip"], it.key, len(it.frames), data_root, strip_cache)
            render_sync(d / "sync.mp4", net, it.frames, layout, names, it.video, rate, start,
                        frames or sync_frames, it.ground_truth, name=it.name, reference=reference,
                        extra_strip=extra, batch_size=batch_size)
            print(f"[{it.name}] wrote {d / 'sync.mp4'}")

        (d / "run.json").write_text(json.dumps({
            "model": str(model) if isinstance(model, str) else "(in-memory model)",
            "input": str(it.video or keypoints or config), "limit": limit, "stride": stride,
            "pose_model": used_pose,
            "video": str(it.video) if it.video else None, "frames": int(len(it.frames)), "fps": rate,
            "labels": names,
        }, indent=2) + "\n")
        out_dirs.append(d)
    return out_dirs


def _write_keypoints(path: Path, frames: np.ndarray, layout: KeypointLayout) -> None:
    import pandas as pd

    cols = layout.column_names()
    pd.DataFrame(frames, columns=cols if len(cols) == frames.shape[1] else None).to_csv(
        path, index=False, float_format="%.6f")


def _video_fps(video) -> Optional[float]:
    if video is None:
        return None
    import cv2

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else None
    cap.release()
    return fps or None
