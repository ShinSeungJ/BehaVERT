"""The keypoint layout a model expects, read from its config."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..model import BehaVERTConfig

ANIMAL_COLORS = ["#FF5733", "#2E86C1", "#27AE60", "#8E44AD", "#F39C12"]


@dataclass
class KeypointLayout:
    num_animals: int
    names: list[str]
    dims: int = 2
    edges: list[tuple[int, int]] = field(default_factory=list)
    units: str = "pixels"  # "pixels" or "normalized" (x / width, y / height)
    frame_size: Optional[tuple[int, int]] = None  # (width, height) for pixel coordinates
    fps: float = 30.0
    pose_indices: Optional[list[int]] = None  # which pose-model keypoints feed the model
    animal_names: list[str] = field(default_factory=list)
    # How coordinates map onto the video when it differs from frame_size:
    # {"swap_xy": bool, "scale": s, "reference_width": w} -> pixel = coord * s * video_width / w
    video_mapping: Optional[dict] = None

    @property
    def num_keypoints(self) -> int:
        return len(self.names)

    @property
    def input_dim(self) -> int:
        return self.num_animals * self.num_keypoints * self.dims

    @classmethod
    def from_config(cls, config: BehaVERTConfig) -> "KeypointLayout":
        k = dict(config.keypoints or {})
        names = k.get("names") or k.get("keypoint_names")
        dims = int(k.get("dims", k.get("num_dims", 2)))
        animals = int(k.get("num_animals", 1))
        if not names:  # unknown layout: treat every coordinate pair as a keypoint
            names = [f"kp{i}" for i in range(config.input_dim // dims)]
            animals = 1
        layout = cls(
            num_animals=animals,
            names=list(names),
            dims=dims,
            edges=[tuple(e) for e in k.get("edges", [])],
            units=k.get("units", "pixels"),
            frame_size=tuple(k["frame_size"]) if k.get("frame_size") else None,
            fps=float(k.get("fps", 30)),
            pose_indices=k.get("pose_indices"),
            animal_names=list(k.get("animal_names", [])),
            video_mapping=k.get("video_mapping"),
        )
        if layout.input_dim != config.input_dim:
            raise ValueError(f"keypoint metadata describes {layout.input_dim} values per frame, "
                             f"but the model takes {config.input_dim}")
        return layout

    def column_names(self) -> list[str]:
        axes = "xyz"[: self.dims]
        prefix = [f"{a}_" for a in (self.animal_names or [f"animal{i + 1}" for i in range(self.num_animals)])]
        if self.num_animals == 1:
            prefix = [""]
        return [f"{p}{n}_{ax}" for p in prefix for n in self.names for ax in axes]

    def unflatten(self, frames: np.ndarray) -> np.ndarray:
        """``[T, F]`` -> ``[T, animals, keypoints, dims]``."""
        return frames.reshape(frames.shape[0], self.num_animals, self.num_keypoints, self.dims)

    def to_pixels(self, frames: np.ndarray, video_size: Optional[tuple[int, int]]) -> np.ndarray:
        """``[T, F]`` model coordinates -> ``[T, animals, keypoints, 2]`` pixel x, y for drawing."""
        pts = self.unflatten(frames)[..., :2].astype(np.float32).copy()
        vm = self.video_mapping
        if vm and video_size:
            if vm.get("swap_xy"):
                pts = pts[..., ::-1].copy()
            pts *= float(vm.get("scale", 1.0)) * video_size[0] / float(vm.get("reference_width", video_size[0]))
            return pts
        if self.units == "normalized":
            w, h = video_size or (1, 1)
            pts[..., 0] *= w
            pts[..., 1] *= h
        elif video_size and self.frame_size and tuple(video_size) != tuple(self.frame_size):
            pts[..., 0] *= video_size[0] / self.frame_size[0]
            pts[..., 1] *= video_size[1] / self.frame_size[1]
        return pts
