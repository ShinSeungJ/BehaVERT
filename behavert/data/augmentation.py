"""Context-aware data augmentation.

Each call picks 1-2 operations at random from {mirror, rotate, rotate_90,
shift, jitter} and applies them to one window. Every operation respects the
arena: a transform that would move keypoints outside it is undone or retried.

Two arena families:

* 2D rectangular arenas in pixel (or normalized) coordinates — CalMS21,
  MABe22, DeepEthogram. Origin at a corner; the arena is ``[0, W] x [0, H]``.
* 3D circular arenas in world coordinates centred on the arena — PAIR-R24M,
  SBeA.
  Spatial operations act on X/Y; Z is never mirrored or rotated.

Use a preset (``AUGMENTATION_PRESETS``) or describe your own arena with
:class:`ArenaConfig`, then call :func:`augment` on a ``[L, F]`` window.
Input windows are never modified in place.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class ArenaConfig:
    """Arena geometry and augmentation strength for one dataset."""

    coords_per_kpt: int  # 2 (x, y) or 3 (x, y, z)
    jitter_scale: float  # max absolute offset added per axis
    # 2D rectangular arena
    width: Optional[float] = None
    height: Optional[float] = None
    # 3D circular arena (centred at the origin)
    radius: Optional[float] = None  # keypoints must stay within this XY radius
    z_min: Optional[float] = None
    z_max: Optional[float] = None
    max_shift: Optional[float] = None  # initial XY shift range (shrinks on retry)
    operations: tuple[str, ...] = ()


    def __post_init__(self) -> None:
        if not self.operations:
            self.operations = (("mirror", "shift", "rotate", "jitter", "rotate_90") if self.coords_per_kpt == 2
                               else ("mirror", "shift", "rotate", "rotate_90", "jitter"))
        self.operations = tuple(self.operations)
        if self.coords_per_kpt == 2 and (self.width is None or self.height is None):
            raise ValueError("2D arenas need width and height")
        if self.coords_per_kpt == 3 and None in (self.radius, self.z_min, self.z_max, self.max_shift):
            raise ValueError("3D arenas need radius, z_min, z_max and max_shift")
        unknown = set(self.operations) - {"mirror", "rotate", "rotate_90", "shift", "jitter"}
        if unknown:
            raise ValueError(f"unknown augmentation operations: {sorted(unknown)}")


# Arena of each benchmark.
AUGMENTATION_PRESETS: dict[str, ArenaConfig] = {
    # 1024 x 570 px, 2 mice x 7 keypoints. No 90-degree rotation (non-square arena).
    "calms21": ArenaConfig(
        coords_per_kpt=2, width=1024, height=570, jitter_scale=0.03,
        operations=("mirror", "shift", "rotate", "jitter"),
    ),
    # 800 x 800 px, 3 mice x 12 keypoints.
    "mabe22": ArenaConfig(coords_per_kpt=2, width=800, height=800, jitter_scale=0.02),
    # coordinates normalized to [0, 1], 1 mouse x 11 keypoints.
    "deepethogram": ArenaConfig(coords_per_kpt=2, width=1.0, height=1.0, jitter_scale=0.02),
    # 3D, circular cage of radius 400 mm.
    "pair24m": ArenaConfig(
        coords_per_kpt=3, radius=400.0, z_min=-60.0, z_max=380.0, jitter_scale=10.0, max_shift=50.0,
    ),
    # 3D, centred SBeA coordinates, circular arena of radius 99.4.
    "sbea": ArenaConfig(
        coords_per_kpt=3, radius=99.4, z_min=-45.4, z_max=45.4, jitter_scale=3.0, max_shift=20.0,
    ),
}


def get_arena(preset_or_config) -> ArenaConfig:
    if isinstance(preset_or_config, ArenaConfig):
        return preset_or_config
    if isinstance(preset_or_config, dict):
        return ArenaConfig(**preset_or_config)
    try:
        return AUGMENTATION_PRESETS[preset_or_config]
    except KeyError:
        raise KeyError(f"unknown augmentation preset {preset_or_config!r}; "
                       f"choose from {sorted(AUGMENTATION_PRESETS)} or pass an ArenaConfig") from None


def augment(window: torch.Tensor, arena, operations: Optional[list[str]] = None) -> torch.Tensor:
    """Return an augmented copy of a ``[L, F]`` keypoint window (the input is not changed).

    ``operations`` forces specific operations (applied in canonical order);
    by default 1-2 are drawn at random from ``arena.operations``.
    """
    arena = get_arena(arena)
    x = window.clone()
    if operations is None:
        enabled = list(arena.operations)
        k = random.randint(1, min(2, len(enabled)))
        operations = random.sample(enabled, k=k)
    if arena.coords_per_kpt == 2:
        return _augment_2d_rect(x, arena, set(operations))
    return _augment_3d_circular(x, arena, set(operations))


# --------------------------------------------------------------------------- 2D
def _in_rect(kpts: torch.Tensor, W: float, H: float) -> bool:
    return bool(kpts[..., 0].min() >= 0 and kpts[..., 0].max() <= W
                and kpts[..., 1].min() >= 0 and kpts[..., 1].max() <= H)


def _rotate_about(kpts: torch.Tensor, theta: torch.Tensor, cx: float, cy: float) -> torch.Tensor:
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).reshape(2, 2).to(kpts)
    centre = kpts.new_tensor([cx, cy])
    return (kpts - centre) @ R.T + centre


def _augment_2d_rect(x: torch.Tensor, a: ArenaConfig, ops: set[str]) -> torch.Tensor:
    L, F = x.shape
    W, H = a.width, a.height
    cx, cy = W / 2, H / 2

    if "mirror" in ops:  # independent X / Y flips, 50% each
        if random.random() < 0.5:
            x[:, 0::2] = W - x[:, 0::2]
        if random.random() < 0.5:
            x[:, 1::2] = H - x[:, 1::2]

    if "rotate_90" in ops:  # discrete rotation about the arena centre; undone if it leaves the arena
        angle = random.choice([0, 90, 180, 270])
        if angle:
            r = _rotate_about(x.view(L, -1, 2), torch.tensor(angle * torch.pi / 180), cx, cy)
            if _in_rect(r, W, H):
                x = r.reshape(L, F)

    if "rotate" in ops:  # free rotation; if it leaves the arena try sliding back, else undo
        r = _rotate_about(x.view(L, -1, 2), torch.rand(()) * 2 * torch.pi, cx, cy)
        if not _in_rect(r, W, H):
            sx = torch.clamp(W - r[..., 0].max(), max=0) + torch.clamp(-r[..., 0].min(), min=0)
            sy = torch.clamp(H - r[..., 1].max(), max=0) + torch.clamp(-r[..., 1].min(), min=0)
            r[..., 0] += sx
            r[..., 1] += sy
        if _in_rect(r, W, H):
            x = r.reshape(L, F)

    if "shift" in ops:  # largest translation that keeps every keypoint inside
        max_xp, max_xn = W - x[:, 0::2].max(), -x[:, 0::2].min()
        max_yp, max_yn = H - x[:, 1::2].max(), -x[:, 1::2].min()
        x[:, 0::2] += torch.rand(()) * (max_xp + max_xn) - max_xn
        x[:, 1::2] += torch.rand(()) * (max_yp + max_yn) - max_yn

    if "jitter" in ops:  # one small offset per axis for the whole window
        s = a.jitter_scale
        x[:, 0::2] += s * (torch.rand(()) * 2 - 1)
        x[:, 1::2] += s * (torch.rand(()) * 2 - 1)
    return x


# --------------------------------------------------------------------------- 3D
def _rotate_xy(x: torch.Tensor, theta: torch.Tensor, radius: float) -> torch.Tensor:
    L, F = x.shape
    k = x.view(L, -1, 3)
    c, s = torch.cos(theta), torch.sin(theta)
    R = torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).reshape(2, 2).to(x)
    xy = k[..., :2] @ R.T
    if torch.sqrt((xy ** 2).sum(-1)).max() <= radius:
        return torch.cat([xy, k[..., 2:]], dim=-1).reshape(L, F)
    return x


def _augment_3d_circular(x: torch.Tensor, a: ArenaConfig, ops: set[str]) -> torch.Tensor:
    if x.shape[1] % 3:
        raise ValueError(f"3D augmentation needs F divisible by 3, got {x.shape[1]}")

    if "mirror" in ops:  # flip X, Y, both, or nothing (Z never mirrored)
        m = random.randint(0, 3)
        if m in (1, 3):
            x[:, 0::3] = -x[:, 0::3]
        if m in (2, 3):
            x[:, 1::3] = -x[:, 1::3]

    if "rotate_90" in ops:
        angle = random.choice([0, 90, 180, 270])
        if angle:
            x = _rotate_xy(x, torch.tensor(angle * torch.pi / 180), a.radius)

    if "rotate" in ops:
        x = _rotate_xy(x, torch.rand(()) * 2 * torch.pi, a.radius)

    if "shift" in ops:  # random XY shift, shrunk until everything stays in the arena; small Z shift
        xs_, ys_ = x[:, 0::3], x[:, 1::3]
        if torch.sqrt(xs_ ** 2 + ys_ ** 2).max() <= a.radius:
            max_shift = a.max_shift
            for _ in range(20):
                dx = (torch.rand(()) * 2 - 1) * max_shift
                dy = (torch.rand(()) * 2 - 1) * max_shift
                if torch.sqrt((xs_ + dx) ** 2 + (ys_ + dy) ** 2).max() <= a.radius:
                    head_pos = a.z_max - x[:, 2::3].max()
                    head_neg = a.z_min - x[:, 2::3].min()
                    z_range = min(abs(float(head_pos)), abs(float(head_neg))) * 0.1
                    x[:, 0::3] += dx
                    x[:, 1::3] += dy
                    x[:, 2::3] += (torch.rand(()) * 2 - 1) * z_range
                    break
                max_shift *= 0.8

    if "jitter" in ops:
        s = a.jitter_scale
        for d in range(3):
            x[:, d::3] += s * (torch.rand(()) * 2 - 1)
    return x
