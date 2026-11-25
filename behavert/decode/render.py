"""Stage 3 — videos: a skeleton + behavior overlay, and the synchronized panel video."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np

from .analysis import Analysis
from .layout import ANIMAL_COLORS, KeypointLayout

BEHAVIOR_COLORS = ["#E74C3C", "#F1C40F", "#3498DB", "#2ECC71", "#9B59B6", "#E67E22", "#1ABC9C",
                   "#D35400", "#34495E", "#16A085", "#C0392B", "#7F8C8D"]
QUIET_NAMES = {"other", "others", "background", "none", "far", "idle"}
QUIET_COLOR = "#BDC3C7"


def behavior_colors(names: list[str], preset: Optional[list[str]] = None) -> list[str]:
    if preset and len(preset) == len(names):
        return list(preset)
    out, k = [], 0
    for n in names:
        if n.lower() in QUIET_NAMES:
            out.append(QUIET_COLOR)
        else:
            out.append(BEHAVIOR_COLORS[k % len(BEHAVIOR_COLORS)])
            k += 1
    return out


def _bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return b, g, r


class VideoReader:
    """Sequential frame reader (``None`` frames when there is no video)."""

    def __init__(self, path: Optional[str | Path], start: int = 0):
        import cv2

        self.cap = None
        self.size = None
        self.fps = None
        if path is not None:
            self.cap = cv2.VideoCapture(str(path))
            if not self.cap.isOpened():
                raise FileNotFoundError(f"cannot open video {path}")
            self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or None
            if start:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    def read(self) -> Optional[np.ndarray]:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        return frame if ok else None

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class VideoWriter:
    """H.264 mp4 through ffmpeg (falls back to OpenCV's mp4v if ffmpeg is missing)."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float):
        import cv2

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = size
        self.proc = self.cv = None
        if shutil.which("ffmpeg"):
            self.proc = subprocess.Popen(
                ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
                 "-r", f"{fps}", "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "fast", str(self.path)],
                stdin=subprocess.PIPE)
        else:
            self.cv = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    def write(self, frame: np.ndarray) -> None:
        if self.proc:
            self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            self.cv.write(frame)

    def close(self) -> None:
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        else:
            self.cv.release()


def _canvas_size(layout: KeypointLayout, video_size) -> tuple[int, int]:
    if video_size:
        return video_size
    if layout.frame_size:
        return tuple(layout.frame_size)
    return (640, 480)


def render_overlay(out_path: Path, frames: np.ndarray, res: Analysis, layout: KeypointLayout,
                   label_names: list[str], video: Optional[str | Path] = None, fps: Optional[float] = None,
                   start: int = 0, num_frames: Optional[int] = None, label_colors: Optional[list[str]] = None,
                   style: Optional[dict] = None) -> Path:
    """Video with the skeleton(s) and the predicted behavior drawn on every frame.

    ``style`` (the model's ``display["overlay"]``) may set ``edge_colors_bgr`` (one per
    skeleton edge), ``label_colors_bgr`` (one per class), ``line_thickness``,
    ``point_radius``, ``point_color_bgr``, ``font_scale`` and ``show_confidence``.
    """
    import cv2

    st = dict(style or {})
    reader = VideoReader(video, start)
    size = _canvas_size(layout, reader.size)
    fps = fps or reader.fps or layout.fps
    end = len(frames) if num_frames is None else min(len(frames), start + num_frames)
    pts = layout.to_pixels(frames[start:end], size)
    label_bgr = [tuple(c) for c in st["label_colors_bgr"]] if st.get("label_colors_bgr") else \
        [_bgr(c) for c in behavior_colors(label_names, label_colors)]
    edge_bgr = [tuple(c) for c in st.get("edge_colors_bgr", [])]
    thick = int(st.get("line_thickness", 2))
    radius = int(st.get("point_radius", 3))
    point_bgr = tuple(st.get("point_color_bgr", (0, 0, 0)))
    font = float(st.get("font_scale", 1.0))
    writer = VideoWriter(out_path, size, fps)
    for i in range(end - start):
        img = reader.read()
        if img is None:
            if reader.cap is not None:
                break
            img = np.full((size[1], size[0], 3), 255, np.uint8)
        for a in range(layout.num_animals):
            animal_bgr = _bgr(ANIMAL_COLORS[a % len(ANIMAL_COLORS)])
            p = pts[i, a]
            for e, (u, v) in enumerate(layout.edges):
                if np.isfinite(p[[u, v]]).all():
                    col = edge_bgr[e] if e < len(edge_bgr) else animal_bgr
                    cv2.line(img, (int(p[u, 0]), int(p[u, 1])), (int(p[v, 0]), int(p[v, 1])), col, thick)
            for x, y in p:
                if np.isfinite(x) and np.isfinite(y):
                    cv2.circle(img, (int(x), int(y)), radius, point_bgr, -1)
        t = start + i
        lab = int(res.labels[t])
        text = label_names[lab] + (f"  {res.confidence[t]:.2f}" if st.get("show_confidence", False) else "")
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font, 2)
        cv2.putText(img, text, ((size[0] - tw) // 2, 40 + th), cv2.FONT_HERSHEY_SIMPLEX, font,
                    label_bgr[lab % len(label_bgr)], 2, cv2.LINE_AA)
        writer.write(img)
    writer.close()
    reader.close()
    return Path(out_path)


def render_sync(out_path: Path, model, frames: np.ndarray, layout: KeypointLayout, label_names: list[str],
                video: Optional[str | Path] = None, fps: Optional[float] = None, start: int = 0,
                num_frames: Optional[int] = 900, ground_truth: Optional[np.ndarray] = None, name: str = "",
                reference=None, extra_strip: Optional[dict] = None, window_seconds: float = 3.0, batch_size: int = 64) -> Path:
    """The synchronized panel video for frames ``[start, start + num_frames)``.

    The clip is analysed on its own, and
    titles, strip labels, colors and the embedding map follow the model's
    ``display`` settings.

    Args:
        reference: optional :class:`~behavert.decode.embedding.Reference` backdrop for the map.
        extra_strip: ``{"title", "labels" [T], "names", "colors", "legend"}`` for one more
            label strip (e.g. MABe22 behaviors), covering the whole recording.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.colors import LinearSegmentedColormap

    from .analysis import analyze
    from .embedding import embedding_map
    from .sync import LabelStrip, SyncBundle, draw_skeleton, render_sync_video

    cfg = model.config
    disp = dict(cfg.display or {})
    T = len(frames)
    end = T if num_frames is None else min(T, start + num_frames)
    clip = frames[start:end]
    res = analyze(model, clip, batch_size=batch_size)
    emb = embedding_map(res, cfg.mode, cfg.num_classes, disp.get("embedding"), reference)

    reader = VideoReader(video, start)
    size = _canvas_size(layout, reader.size)
    fps = fps or reader.fps or layout.fps
    pts = layout.to_pixels(clip, size)
    colors = behavior_colors(label_names, cfg.label_colors)
    edges = [layout.edges] * layout.num_animals
    animal_colors = disp.get("animal_colors") or ANIMAL_COLORS[: layout.num_animals]

    # per-edge / keypoint colors of the model's overlay style, if it has one (BGR -> RGB hex)
    ov = disp.get("overlay") or {}
    to_hex = lambda bgr: "#%02X%02X%02X" % (bgr[2], bgr[1], bgr[0])  # noqa: E731
    edge_colors = [to_hex(c) for c in ov["edge_colors_bgr"]] if ov.get("edge_colors_bgr") else None
    point_color = to_hex(ov["point_color_bgr"]) if ov.get("point_color_bgr") else None

    def draw(i, ax):
        bg = reader.read()
        if bg is not None:
            bg = bg[:, :, ::-1]
        draw_skeleton(ax, pts[i], edges=edges, mouse_colors=animal_colors, bg_image=bg,
                      xlim=(0, size[0]), ylim=(size[1], 0), edge_colors=edge_colors, point_color=point_color)

    gt = None
    if ground_truth is not None:  # frames without a label (short label file) are left blank
        gt = np.full(len(clip), -1, dtype=np.int64)
        part = np.asarray(ground_truth)[start:end].astype(np.int64)
        gt[: len(part)] = part
    title = f"{disp['title']} — {name}" if disp.get("title") else name
    if disp.get("title_shows_truth") and gt is not None and len(gt) and gt[0] >= 0:
        title += f" (true={label_names[int(gt[0])]})"

    kwargs = {}
    if emb.color_value is not None:  # binary sequence model: color the map by P(class 1)
        stops = disp.get("probability_colormap") or [colors[0], "#F5F5F5", colors[1]]
        kwargs = dict(tsne_color_values=emb.color_value,
                      tsne_cmap=LinearSegmentedColormap.from_list("p1", stops),
                      tsne_vmin=0.0, tsne_vmax=1.0, tsne_colorbar_label=f"P({label_names[1]})")
    strips = []
    if extra_strip is not None:
        strips.append(LabelStrip(title=extra_strip["title"], labels=np.asarray(extra_strip["labels"])[start:end],
                                 names=extra_strip["names"], colors=extra_strip["colors"],
                                 show_legend=extra_strip.get("legend", False), legend_skip_first=True))

    bundle = SyncBundle(
        num_frames=len(clip), fps=fps, skeleton_frame_fn=draw,
        attention=res.attention, confidence=res.confidence, gt_labels=gt, pred_labels=res.labels,
        label_names=label_names, label_colors=colors,
        tsne_xy=emb.xy, tsne_color_idx=emb.color_index, tsne_alpha=emb.alpha,
        seq_to_frame=emb.point_to_frame, frame_to_seq=emb.frame_to_point,
        extra_strips=strips, title=title,
        skeleton_title=disp.get("skeleton_title", "Skeleton"),
        tsne_title=disp.get("embedding_title", "Sequence embedding (t-SNE)"),
        attention_label=disp.get("attention_label", "attention value"),
        confidence_label=disp.get("confidence_label", "confidence"),
        gt_label_title=disp.get("truth_title", "GT behav"),
        pred_label_title=disp.get("prediction_title", "Pred"),
        window_seconds=window_seconds,
        behavior_overlay_position=disp.get("label_overlay", "top"),
        show_label_legend=disp.get("label_legend", True),
        **kwargs,
    )
    try:
        return render_sync_video(bundle, out_path)
    finally:
        reader.close()
