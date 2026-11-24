"""Synchronized visualization video (paper Fig. 1c).

Layout::

    +----------------------+----------------------+
    |   skeleton panel     |    embedding map     |
    |   (video background) |  (current window     |
    |                      |   circled in red)    |
    +----------------------+----------------------+
    |             attention strip                 |
    +---------------------------------------------+
    |             confidence strip                |
    +---------------------------------------------+
    |  ground-truth label strip (if available)    |
    +---------------------------------------------+
    |   prediction strip                          |
    +---------------------------------------------+
    |   (optional) extra label strip(s)           |
    +---------------------------------------------+

The red dashed cursor is fixed at the centre (time offset 0) and the strips
flow past it: each frame shows the data in [-window_seconds, +window_seconds]
around the current frame.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


# --------------------------------------------------------------------------
# Optional extra label strip (e.g. MaBe22 behavior groups)
# --------------------------------------------------------------------------
@dataclass
class LabelStrip:
    title: str                                # short label shown on the left
    labels: np.ndarray                        # (num_frames,) int per-frame class index
    names: Sequence[str]                      # class index -> name (for legend)
    colors: Sequence[str]                     # class index -> color
    show_legend: bool = False                 # render color-swatch legend below strip
    legend_skip_first: bool = False           # omit the first ("none") entry from the legend


# --------------------------------------------------------------------------
# Data bundle passed from per-dataset entry scripts to the compositor.
# --------------------------------------------------------------------------
@dataclass
class SyncBundle:
    # Per-frame data --------------------------------------------------------
    num_frames: int                          # total frames in the demo clip
    fps: float                               # output video fps
    skeleton_frame_fn: Callable[[int, plt.Axes], None]
        # callback that draws the skeleton (and optional video background) for
        # frame index `i` into the given Axes. Should call ax.clear() first.

    # Time-series strips (length == num_frames) ----------------------------
    attention: np.ndarray                    # (num_frames,) attention weight
    confidence: np.ndarray                   # (num_frames,) softmax max prob
    gt_labels: Optional[np.ndarray]          # (num_frames,) int class index, or None
    pred_labels: np.ndarray                  # (num_frames,) int class index

    label_names: Sequence[str]               # class index -> name (GT / pred)
    label_colors: Sequence[str]              # class index -> color

    # t-SNE scatter (one point per *sequence*) -----------------------------
    tsne_xy: np.ndarray                      # (num_seq, 2)
    tsne_color_idx: np.ndarray               # (num_seq,) class index for color
    tsne_alpha: np.ndarray                   # (num_seq,) point alpha (0-1)
    seq_to_frame: np.ndarray                 # (num_seq,) representative frame per seq
    frame_to_seq: np.ndarray                 # (num_frames,) seq index for each frame

    # Optional extra categorical strips appended below GT/Pred -------------
    extra_strips: List[LabelStrip] = field(default_factory=list)

    # Headings / cosmetic --------------------------------------------------
    title: str = ""
    skeleton_title: str = "Skeleton"
    tsne_title: str = "Embedding (t-SNE)"
    attention_label: str = "Attention"
    confidence_label: str = "Confidence"
    gt_label_title: str = "Ground truth"
    pred_label_title: str = "Prediction"
    tsne_xlim: Optional[tuple] = None
    tsne_ylim: Optional[tuple] = None
    skeleton_xlim: Optional[tuple] = None
    skeleton_ylim: Optional[tuple] = None
    invert_skeleton_y: bool = False

    # Flowing-window controls ----------------------------------------------
    window_seconds: float = 3.0              # half-width of the visible window

    # Where to place the per-frame behavior name overlay on the skeleton
    # panel. 'top' = top-center (default), 'bottom' = bottom-center.
    behavior_overlay_position: str = 'top'

    # Highlight bands on the attention strip. Each entry is
    # (start_frame, end_frame, color, alpha). The bands slide with the
    # flowing window just like the line data.
    attention_highlights: List[tuple] = field(default_factory=list)

    # When True, render a color-swatch legend (label_names + label_colors)
    # below the prediction strip. Useful for CalMS21 where the GT/Pred bars
    # carry the behavior colors directly without a separate strip.
    show_label_legend: bool = False

    # Optional continuous-color t-SNE (overrides tsne_color_idx when set) --
    tsne_color_values: Optional[np.ndarray] = None     # (num_seq,) float
    # cmap can be a registered colormap name or a Colormap instance
    tsne_cmap: object = 'coolwarm'
    tsne_vmin: float = 0.0
    tsne_vmax: float = 1.0
    tsne_colorbar_label: str = ''


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _set_strip_xticks(ax: plt.Axes, window_frames: int, fps: float,
                      show_labels: bool, tick_step_s: float = 1.0) -> None:
    """Tick marks every `tick_step_s` seconds across the [-W, +W] window."""
    n_ticks = int(np.floor(window_frames / fps / tick_step_s))
    ticks = [int(round(i * tick_step_s * fps))
             for i in range(-n_ticks, n_ticks + 1)]
    ax.set_xticks(ticks)
    if show_labels:
        ax.set_xticklabels([f"{t/fps:+.1f}s" if t else "0.0s" for t in ticks],
                           fontsize=7)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(-window_frames, window_frames)


def _pad_for_window(arr: np.ndarray, window_frames: int,
                    pad_value=np.nan) -> np.ndarray:
    """Pad a (T, ...) array with `window_frames` of `pad_value` on each side."""
    if arr.ndim == 1:
        pad = np.full((window_frames,), pad_value, dtype=arr.dtype)
        return np.concatenate([pad, arr, pad], axis=0)
    pad_shape = (window_frames,) + arr.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=arr.dtype)
    return np.concatenate([pad, arr, pad], axis=0)


# --------------------------------------------------------------------------
# Main compositor
# --------------------------------------------------------------------------
def render_sync_video(bundle: SyncBundle, output_path: str | Path,
                      dpi: int = 110) -> Path:
    """Render the synchronized visualization video to `output_path` (mp4)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    window_frames = int(round(bundle.window_seconds * bundle.fps))
    W = window_frames

    # ---- figure / grid spec ----
    n_extra = len(bundle.extra_strips)
    has_extra_legend = any(s.show_legend for s in bundle.extra_strips)
    # legend space is reserved if any strip wants one
    has_legend = has_extra_legend or bundle.show_label_legend
    has_gt = bundle.gt_labels is not None
    # rows: top panels, attn, conf, [gt], pred, *extras
    row_height_ratios = [6.0, 1.0, 1.0] + ([0.45] if has_gt else []) + [0.45] + [0.45] * n_extra
    fig_h = 9.0 + 0.6 * n_extra + (0.5 if has_legend else 0.0) - (0.0 if has_gt else 0.45)
    fig_w = 12.0
    # libx264 + yuv420p requires even pixel dimensions; snap fig size to
    # the next-larger even pixel count at our chosen dpi.
    def _even_inches(inches: float, dpi_: int) -> float:
        px = int(round(inches * dpi_))
        if px % 2:
            px += 1
        return px / dpi_
    fig_w = _even_inches(fig_w, dpi)
    fig_h = _even_inches(fig_h, dpi)
    fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=False)
    if bundle.title:
        fig.suptitle(bundle.title, fontsize=12, y=0.995)
    bottom_margin = 0.11 if has_legend else 0.06
    use_colorbar = bundle.tsne_color_values is not None
    # Leave horizontal room for the t-SNE colorbar's label
    right_margin = 0.94 if use_colorbar else 0.985
    gs = GridSpec(
        nrows=len(row_height_ratios), ncols=2,
        height_ratios=row_height_ratios,
        width_ratios=[1, 1],
        left=0.08, right=right_margin, top=0.94, bottom=bottom_margin,
        hspace=0.22, wspace=0.14,
    )
    ax_sk = fig.add_subplot(gs[0, 0])
    ax_ts = fig.add_subplot(gs[0, 1])
    ax_att = fig.add_subplot(gs[1, :])
    ax_conf = fig.add_subplot(gs[2, :])
    r = 3
    ax_gt = fig.add_subplot(gs[r, :]) if has_gt else None
    r += 1 if has_gt else 0
    ax_pr = fig.add_subplot(gs[r, :])
    ax_extras = [fig.add_subplot(gs[r + 1 + i, :]) for i in range(n_extra)]

    # ---- t-SNE scatter (static; only the red highlight moves) ----
    TSNE_DOT_S = 40  # scatter s parameter (area in pt^2); diameter = sqrt(s)
    if use_colorbar:
        sc = ax_ts.scatter(
            bundle.tsne_xy[:, 0], bundle.tsne_xy[:, 1],
            c=bundle.tsne_color_values, cmap=bundle.tsne_cmap,
            vmin=bundle.tsne_vmin, vmax=bundle.tsne_vmax,
            s=TSNE_DOT_S, alpha=np.clip(bundle.tsne_alpha, 0.5, 1.0),
            linewidths=0, edgecolors='none',
        )
    else:
        base_colors = np.array(
            [bundle.label_colors[i % len(bundle.label_colors)]
             for i in bundle.tsne_color_idx], dtype=object)
        sc = ax_ts.scatter(
            bundle.tsne_xy[:, 0], bundle.tsne_xy[:, 1],
            c=base_colors.tolist(), s=TSNE_DOT_S,
            alpha=np.clip(bundle.tsne_alpha, 0.5, 1.0),
            linewidths=0, edgecolors='none',
        )
    if bundle.tsne_xlim:
        ax_ts.set_xlim(*bundle.tsne_xlim)
    if bundle.tsne_ylim:
        ax_ts.set_ylim(*bundle.tsne_ylim)
    ax_ts.set_xlabel('TSNE-1', fontsize=9)
    ax_ts.set_ylabel('TSNE-2', fontsize=9)
    ax_ts.set_title(bundle.tsne_title, fontsize=10)
    ax_ts.set_xticks([]); ax_ts.set_yticks([])
    # Force the t-SNE plot box to be a perfect square (in physical units).
    ax_ts.set_box_aspect(1.0)
    # Class legend in top-left corner of the t-SNE axes. When a colorbar is
    # in use, recolor the legend markers to match the cmap endpoints so the
    # legend stays consistent with the actual point colors.
    if use_colorbar:
        if isinstance(bundle.tsne_cmap, Colormap):
            cm = bundle.tsne_cmap
        else:
            cm = matplotlib.colormaps[bundle.tsne_cmap]
        legend_marker_colors = [cm(0.0), cm(1.0)] + [
            bundle.label_colors[i] for i in range(2, len(bundle.label_names))]
    else:
        legend_marker_colors = list(bundle.label_colors)
    handles = [
        Line2D([0], [0], marker='o', linestyle='',
               color=legend_marker_colors[i],
               markersize=9, label=bundle.label_names[i])
        for i in range(len(bundle.label_names))
    ]
    ax_ts.legend(handles=handles, loc='upper left', fontsize=10,
                 frameon=True, markerscale=1.0)
    if use_colorbar:
        # vertical colorbar to the right of the t-SNE panel
        cbar = fig.colorbar(sc, ax=ax_ts, pad=0.03, fraction=0.04,
                            ticks=[bundle.tsne_vmin,
                                   (bundle.tsne_vmin + bundle.tsne_vmax) / 2,
                                   bundle.tsne_vmax])
        if bundle.tsne_colorbar_label:
            cbar.set_label(bundle.tsne_colorbar_label, fontsize=9,
                           labelpad=4)
        cbar.ax.tick_params(labelsize=7)
    hl, = ax_ts.plot([], [], 'o', markersize=TSNE_DOT_S ** 0.5 + 5,
                     markerfacecolor='none', markeredgecolor='red',
                     markeredgewidth=1.0, zorder=10)

    # ---- skeleton axis style ----
    ax_sk.set_title(bundle.skeleton_title, fontsize=10)
    if bundle.skeleton_xlim:
        ax_sk.set_xlim(*bundle.skeleton_xlim)
    if bundle.skeleton_ylim:
        ax_sk.set_ylim(*bundle.skeleton_ylim)
    ax_sk.set_xticks([]); ax_sk.set_yticks([])

    # ---- pad data arrays so windowed slicing is trivial ----
    att_pad = _pad_for_window(bundle.attention.astype(np.float64), W,
                              pad_value=np.nan)
    conf_pad = _pad_for_window(bundle.confidence.astype(np.float64), W,
                               pad_value=np.nan)
    # categorical strips: pad with sentinel value (will render transparent)
    NAN_LABEL = -1
    gt_pad = (_pad_for_window(bundle.gt_labels.astype(np.int64), W, pad_value=NAN_LABEL)
              if has_gt else None)
    pr_pad = _pad_for_window(bundle.pred_labels.astype(np.int64), W,
                             pad_value=NAN_LABEL)
    extras_pad = [
        (s, _pad_for_window(s.labels.astype(np.int64), W,
                            pad_value=NAN_LABEL))
        for s in bundle.extra_strips
    ]

    # ---- compute fixed y-limits from full data ----
    def _yrange(a: np.ndarray) -> tuple:
        valid = a[np.isfinite(a)]
        if valid.size == 0:
            return 0.0, 1.0
        lo, hi = float(valid.min()), float(valid.max())
        pad = max(0.02, 0.05 * (hi - lo))
        return lo - pad, hi + pad

    att_ylim = _yrange(bundle.attention.astype(np.float64))
    conf_ylim = _yrange(bundle.confidence.astype(np.float64))

    # ---- create persistent strip artists (updated each frame) ----
    xs = np.arange(-W, W + 1)
    line_att, = ax_att.plot(xs, np.full_like(xs, np.nan, dtype=np.float64),
                            color='#1f77b4', linewidth=1.2)
    line_conf, = ax_conf.plot(xs, np.full_like(xs, np.nan, dtype=np.float64),
                              color='#d62728', linewidth=1.2)

    # Highlight bands on the attention strip (sliding with the window)
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import blended_transform_factory
    att_hl_trans = blended_transform_factory(ax_att.transData,
                                             ax_att.transAxes)
    att_hl_rects = []
    for hl_spec in bundle.attention_highlights:
        # support (start, end, color) or (start, end, color, alpha)
        if len(hl_spec) == 4:
            hl_start, hl_end, hl_color, hl_alpha = hl_spec
        else:
            hl_start, hl_end, hl_color = hl_spec
            hl_alpha = 0.4
        rect = Rectangle((0, 0), 0, 1, transform=att_hl_trans,
                         facecolor=hl_color, edgecolor='none',
                         alpha=hl_alpha, zorder=1)
        ax_att.add_patch(rect)
        att_hl_rects.append((int(hl_start), int(hl_end), rect))

    def _style_line_strip(ax, ylabel, ylim, show_xticklabels=False,
                          show_xlabel=False):
        ax.set_ylim(*ylim)
        _set_strip_xticks(ax, W, bundle.fps, show_labels=show_xticklabels)
        # vertical y-axis label, 
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2, zorder=10)
        ax.tick_params(axis='y', labelsize=7)
        # full rectangle box: keep all four spines
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        # grey gridlines (vertical + horizontal).
        # Major grid only so the line strips share an identical set of
        # vertical gridlines (one per major x-tick).
        ax.minorticks_off()
        ax.grid(True, which='major', color='#cdcdcd', linewidth=0.6,
                linestyle='-', alpha=1.0, zorder=0)
        ax.set_axisbelow(True)
        if show_xlabel:
            ax.set_xlabel('Time Offset (s)', fontsize=8)

    _style_line_strip(ax_att, bundle.attention_label, att_ylim)
    _style_line_strip(ax_conf, bundle.confidence_label, conf_ylim)

    # Categorical strips: build extended color lists where index 0 = "no
    # data" (white/transparent) so we can encode the NAN_LABEL via a shifted
    # index. Use a per-strip ListedColormap with N+1 colors.
    def _setup_cat_strip(ax, names, colors, title, show_xticklabels=False,
                         show_xlabel=False):
        ax.set_yticks([])
        _set_strip_xticks(ax, W, bundle.fps, show_labels=show_xticklabels)
        ax.set_ylim(0, 1)
        ax.set_ylabel(title, fontsize=8)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.2, zorder=10)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.8)
        if show_xlabel:
            ax.set_xlabel('Time Offset (s)', fontsize=8)

    if has_gt:
        _setup_cat_strip(ax_gt, bundle.label_names, bundle.label_colors,
                         bundle.gt_label_title)
    is_last_cat = (n_extra == 0)
    _setup_cat_strip(ax_pr, bundle.label_names, bundle.label_colors,
                     bundle.pred_label_title,
                     show_xticklabels=is_last_cat,
                     show_xlabel=is_last_cat)
    for k, ax_e in enumerate(ax_extras):
        is_last = (k == n_extra - 1)
        _setup_cat_strip(ax_e, bundle.extra_strips[k].names,
                         bundle.extra_strips[k].colors,
                         bundle.extra_strips[k].title,
                         show_xticklabels=is_last,
                         show_xlabel=is_last)

    # Categorical imshow handles — we keep one image artist per strip and
    # update its data each frame.
    def _cmap_for(colors):
        # prepend a white color for the NAN sentinel (-1 shifted to 0)
        return ListedColormap(['#FFFFFF'] + list(colors))

    def _make_cat_image(ax, n_classes):
        # initial blank slice
        blank = np.zeros((1, 2 * W + 1), dtype=np.int64)
        im = ax.imshow(blank, aspect='auto',
                       cmap=ListedColormap(['#FFFFFF']),
                       extent=(-W - 0.5, W + 0.5, 0, 1),
                       interpolation='nearest', zorder=1)
        return im

    im_gt = ax_gt.imshow(
        np.zeros((1, 2 * W + 1)), aspect='auto',
        cmap=_cmap_for(bundle.label_colors), vmin=0,
        vmax=len(bundle.label_colors),
        extent=(-W - 0.5, W + 0.5, 0, 1),
        interpolation='nearest', zorder=1,
    ) if has_gt else None
    im_pr = ax_pr.imshow(
        np.zeros((1, 2 * W + 1)), aspect='auto',
        cmap=_cmap_for(bundle.label_colors), vmin=0,
        vmax=len(bundle.label_colors),
        extent=(-W - 0.5, W + 0.5, 0, 1),
        interpolation='nearest', zorder=1,
    )
    extras_imgs = []
    for ax_e, (s, _arr) in zip(ax_extras, extras_pad):
        im = ax_e.imshow(
            np.zeros((1, 2 * W + 1)), aspect='auto',
            cmap=_cmap_for(s.colors), vmin=0, vmax=len(s.colors),
            extent=(-W - 0.5, W + 0.5, 0, 1),
            interpolation='nearest', zorder=1,
        )
        extras_imgs.append(im)

    # Color-swatch legend(s) for any extra strip that opts in.
    from matplotlib.patches import Patch  # local import to keep header tidy
    for ax_e, s in zip(ax_extras, bundle.extra_strips):
        if not s.show_legend:
            continue
        skip = 1 if s.legend_skip_first else 0
        patches = [Patch(facecolor=s.colors[i], edgecolor='black',
                         linewidth=0.4, label=s.names[i])
                   for i in range(skip, len(s.names))]
        ncol = min(len(patches), 9)
        ax_e.legend(handles=patches, loc='upper center',
                    bbox_to_anchor=(0.5, -1.6), ncol=ncol,
                    fontsize=10, frameon=False, handlelength=1.4,
                    handleheight=1.1, columnspacing=1.0, borderpad=0.2)

    # Color-swatch legend for the GT/Pred class colors (CalMS21 path: there
    # are no extra strips, but the user still wants a behavior-color legend).
    if bundle.show_label_legend:
        host_ax = ax_extras[-1] if ax_extras else ax_pr
        patches = [Patch(facecolor=bundle.label_colors[i], edgecolor='black',
                         linewidth=0.4, label=bundle.label_names[i])
                   for i in range(len(bundle.label_names))]
        ncol = min(len(patches), 9)
        host_ax.legend(handles=patches, loc='upper center',
                       bbox_to_anchor=(0.5, -1.6), ncol=ncol,
                       fontsize=10, frameon=False, handlelength=1.4,
                       handleheight=1.1, columnspacing=1.0, borderpad=0.2)

    # 

    # ---- per-frame update ----
    def update(i: int):
        # skeleton (callback clears the axis itself)
        bundle.skeleton_frame_fn(i, ax_sk)
        ax_sk.set_title(
            f"{bundle.skeleton_title}  •  frame {i+1}/{bundle.num_frames}",
            fontsize=10)

        # ---- behavior overlay on top of the video panel ----
        # Build a stack of "Title: name" lines for GT label + any extra strips,
        # colored by the active class. The text sits in the upper-left of the
        # skeleton axes in axes-fraction coords so it survives the ax.clear().
        overlay_lines = []
        gt_idx = int(bundle.gt_labels[i]) if has_gt else int(bundle.pred_labels[i])
        if 0 <= gt_idx < len(bundle.label_names):
            overlay_lines.append((
                bundle.label_names[gt_idx],
                bundle.label_colors[gt_idx % len(bundle.label_colors)],
            ))
        for s in bundle.extra_strips:
            lab = int(s.labels[i])
            if 0 <= lab < len(s.names):
                overlay_lines.append((
                    s.names[lab],
                    s.colors[lab % len(s.colors)],
                ))
        on_bottom = bundle.behavior_overlay_position == 'bottom'
        if on_bottom:
            y, va, step = 0.03, 'bottom', +0.075
            overlay_lines = list(reversed(overlay_lines))
        else:
            y, va, step = 0.97, 'top', -0.075
        for text, color in overlay_lines:
            ax_sk.text(
                0.5, y, text, transform=ax_sk.transAxes,
                ha='center', va=va, fontsize=11, fontweight='bold',
                color=color,
                bbox=dict(facecolor='white', edgecolor='black',
                          linewidth=0.6, alpha=0.85, pad=2.0),
                zorder=20,
            )
            y += step

        # padded indices: padded position of frame i is W+i; window spans
        # [W+i - W : W+i + W + 1] = [i : i + 2W+1]
        lo = i
        hi = i + 2 * W + 1

        line_att.set_ydata(att_pad[lo:hi])
        line_conf.set_ydata(conf_pad[lo:hi])

        # Slide attention highlight bands relative to current frame i.
        # A highlight on frames [a, b] sits at window coords [a-i, b-i].
        for a, b, rect in att_hl_rects:
            x0 = (a - i) - 0.5
            x1 = (b - i) + 0.5
            # clip to visible window so the patch doesn't overshoot
            vx0 = max(x0, -W - 0.5)
            vx1 = min(x1, W + 0.5)
            if vx1 <= vx0:
                rect.set_x(-W - 0.5)
                rect.set_width(0)
            else:
                rect.set_x(vx0)
                rect.set_width(vx1 - vx0)

        # Categorical: shift int labels by +1 so the sentinel becomes 0
        pr_slice = pr_pad[lo:hi] + 1
        if has_gt:
            im_gt.set_data((gt_pad[lo:hi] + 1)[None, :])
        im_pr.set_data(pr_slice[None, :])
        for (s, arr), im in zip(extras_pad, extras_imgs):
            im.set_data((arr[lo:hi] + 1)[None, :])

        # t-SNE highlight
        seq_idx = int(bundle.frame_to_seq[i])
        hl.set_data([bundle.tsne_xy[seq_idx, 0]],
                    [bundle.tsne_xy[seq_idx, 1]])

        return [a for a in (hl, line_att, line_conf, im_gt, im_pr, *extras_imgs) if a is not None]

    writer = animation.FFMpegWriter(
        fps=bundle.fps, codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p', '-preset', 'fast', '-crf', '20'],
    )
    bar = _progress(bundle.num_frames, f"sync video {output_path.name}")
    with writer.saving(fig, str(output_path), dpi=dpi):
        for i in range(bundle.num_frames):
            update(i)
            writer.grab_frame()
            if bar:
                bar.update(1)
    if bar:
        bar.close()
    plt.close(fig)
    return output_path


def _progress(total, desc):
    try:
        from tqdm import tqdm
        return tqdm(total=total, desc=desc, unit="frame")
    except ImportError:  # pragma: no cover
        return None


# --------------------------------------------------------------------------
# Skeleton drawing helpers
# --------------------------------------------------------------------------
def draw_skeleton(ax: plt.Axes, kpts: np.ndarray, edges: Sequence[Sequence[tuple]],
                  mouse_colors: Sequence[str], bg_image: Optional[np.ndarray] = None,
                  point_size: float = 12.0, linewidth: float = 2.0,
                  xlim=None, ylim=None, invert_y: bool = False,
                  edge_colors: Optional[Sequence[str]] = None,
                  point_color: Optional[str] = None) -> None:
    """Draw a multi-mouse skeleton on `ax`.

    Args:
        kpts:   (num_mice, num_keypoints, 2) array of x,y coordinates.
        edges:  list of edge lists, one per mouse.
        mouse_colors: list of colors, one per mouse.
        bg_image: optional HxWx3 RGB array drawn as background.
        edge_colors: optional color per edge (instead of the mouse color).
        point_color: optional keypoint color (instead of the mouse color).
    """
    ax.clear()
    if bg_image is not None:
        ax.imshow(bg_image)
    for m in range(kpts.shape[0]):
        col = mouse_colors[m % len(mouse_colors)]
        pts = kpts[m]
        ax.scatter(pts[:, 0], pts[:, 1], s=point_size, c=point_color or col, zorder=3,
                   edgecolors='black', linewidths=0.4)
        for e, (a, b) in enumerate(edges[m % len(edges)]):
            ec = edge_colors[e] if edge_colors and e < len(edge_colors) else col
            ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]],
                    color=ec, linewidth=linewidth, zorder=2)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if invert_y:
        ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])
