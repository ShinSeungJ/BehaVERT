"""Stage 1 — video to keypoints with a YOLO pose model (needs ``pip install behavert[pose]``).

The pose model is an Ultralytics checkpoint folder (``model.pt`` +
``config.json`` with ``keypoint_names`` and ``predict`` settings). Keypoints
are kept in normalized image coordinates (x / width, y / height); the
behavior model's ``pose_indices`` select and order the ones it was trained on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ..hub import resolve_pose_model
from .layout import KeypointLayout


@dataclass
class PoseResult:
    keypoints: np.ndarray  # [T, K, 2] normalized x, y (NaN where not detected)
    confidence: np.ndarray  # [T, K]
    detected: np.ndarray  # [T] bool
    names: list[str]
    video_size: tuple[int, int]
    fps: float


class PoseModel:
    def __init__(self, name_or_path: str, device: Optional[str] = None):
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise ImportError("pose estimation needs Ultralytics: pip install 'behavert[pose]'") from e
        cfg_path, weights = resolve_pose_model(name_or_path)
        self.config = json.loads(Path(cfg_path).read_text()) if Path(cfg_path).exists() else {}
        self.model = YOLO(str(weights))
        self.device = device
        self.names = self.config.get("keypoint_names") or []

    def predict_video(self, video: str | Path, max_frames: Optional[int] = None, progress: bool = True) -> PoseResult:
        import cv2

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        if max_frames:
            total = min(total, max_frames) if total else max_frames

        params = dict(self.config.get("predict", {}))
        params.setdefault("max_det", self.config.get("max_animals", 1))
        if self.device is not None:
            params["device"] = self.device
        kps, confs, det = [], [], []
        bar = _progress(total, f"pose {Path(video).name}") if progress else None
        while True:
            if max_frames and len(kps) >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            r = self.model.predict(frame, verbose=False, **params)[0]
            k = r.keypoints
            if k is None or k.xyn is None or len(k.xyn) == 0:
                n = len(self.names) or (kps[-1].shape[0] if kps else 0)
                kps.append(np.full((n, 2), np.nan, np.float32))
                confs.append(np.zeros(n, np.float32))
                det.append(False)
            else:
                kps.append(k.xyn[0].cpu().numpy().astype(np.float32))
                confs.append(k.conf[0].cpu().numpy().astype(np.float32) if k.conf is not None
                             else np.ones(len(k.xyn[0]), np.float32))
                det.append(True)
            if bar:
                bar.update(1)
        cap.release()
        if bar:
            bar.close()
        n = max((len(k) for k in kps), default=0)
        kps = [k if len(k) == n else np.full((n, 2), np.nan, np.float32) for k in kps]
        confs = [c if len(c) == n else np.zeros(n, np.float32) for c in confs]
        return PoseResult(np.stack(kps), np.stack(confs), np.array(det), self.names, (W, H), fps)


def interpolate_missing(frames: np.ndarray) -> np.ndarray:
    """Fill NaN rows of ``[T, F]`` by linear interpolation in time (edges: nearest frame)."""
    out = frames.copy()
    t = np.arange(len(out))
    good = ~np.isnan(out).any(1)
    if not good.any():
        raise ValueError("no frame with a detection")
    for j in range(out.shape[1]):
        out[~good, j] = np.interp(t[~good], t[good], out[good, j])
    return out


def pose_to_model_frames(pose: PoseResult, layout: KeypointLayout) -> np.ndarray:
    """Select / order the pose keypoints the behavior model uses and fill gaps -> ``[T, F]``."""
    if layout.num_animals != 1:
        raise NotImplementedError("video input supports single-animal models; for several animals, "
                                  "track identities yourself and pass --keypoints")
    idx = layout.pose_indices or list(range(layout.num_keypoints))
    if max(idx) >= pose.keypoints.shape[1]:
        raise ValueError(f"the behavior model expects pose keypoints {idx}, "
                         f"but the pose model gives {pose.keypoints.shape[1]}")
    k = pose.keypoints[:, idx, :]
    if layout.units == "pixels":
        k = k * np.array(pose.video_size, np.float32)
    return interpolate_missing(k.reshape(len(k), -1)).astype(np.float32)


def _progress(total, desc):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="frame")
    except ImportError:  # pragma: no cover
        return None
