"""Load recordings into one common form, and split them.

Every loader returns a list of :class:`Recording`: a name, a ``[T, F]`` float
array of flattened keypoints, and either per-frame labels (behavior) or one
label for the whole recording (state, e.g. strain; a float for regression).

Bring your own data with :func:`load_recordings`: a ``.pt`` / ``.pkl`` /
``.npz`` file holding ``{name: {"keypoints": [T, ...], "labels": [T] or int}}``
(``labels`` optional). Keypoints are flattened per frame in C order, so a
``[T, animals, keypoints, coords]`` array becomes animal-major, then keypoint,
then coordinate — the order the released models expect.

Loaders for the five paper benchmarks follow the curated files of the
BehaVERT dataset release (see each dataset's README there).
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class Recording:
    name: str
    keypoints: torch.Tensor  # [T, F] float32
    frame_labels: Optional[torch.Tensor] = None  # [T] long
    label: Optional[int | float] = None  # one label for the whole recording (int class, float target)

    @property
    def num_frames(self) -> int:
        return self.keypoints.shape[0]

    def labels_per_frame(self) -> Optional[torch.Tensor]:
        if self.frame_labels is not None:
            return self.frame_labels
        if self.label is None:
            return None
        if isinstance(self.label, float):
            return torch.full((self.num_frames,), self.label, dtype=torch.float32)
        return torch.full((self.num_frames,), int(self.label), dtype=torch.long)


def _flatten(kp) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(kp), dtype=torch.float32)
    return t.reshape(t.shape[0], -1)


def _as_label(v, regression: bool):
    v = v.item() if hasattr(v, "item") else v
    return float(v) if regression else int(v)


# ----------------------------------------------------------------- CalMS21
CALMS21_TASK1_LABELS = ["attack", "investigation", "mount", "other"]
CALMS21_TASK3_BEHAVIORS = ["approach", "disengaged", "groom", "intromission", "mount_attempt", "sniff_face",
                           "whiterearing"]


def load_calms21(json_file: str | Path, group: Optional[str] = None) -> list[Recording]:
    """CalMS21 task JSON (``{group: {session: {"keypoints", "annotations"}}}``).

    Keypoints are stored as ``[T, mouse, (x, y), keypoint]``; each frame is
    flattened to ``mouse -> keypoint -> (x, y)`` (28 values). ``group``
    selects one top-level key — for Task 3 that is the behavior
    (``CALMS21_TASK3_BEHAVIORS``), whose sessions are labelled
    ``0 = other, 1 = behavior``.
    """
    with open(json_file) as f:
        data = json.load(f)
    if group is not None:
        if group not in data:
            raise KeyError(f"{group!r} not in {json_file}; available: {sorted(data)}")
        data = {group: data[group]}
    recs = []
    for sessions in data.values():
        for session_name, session in sessions.items():
            kp = np.asarray(session["keypoints"], dtype=np.float32).transpose(0, 1, 3, 2)  # [T, M, K, 2]
            labels = session.get("annotations")
            recs.append(Recording(
                name=session_name,
                keypoints=_flatten(kp),
                frame_labels=None if labels is None else torch.as_tensor(labels, dtype=torch.long),
            ))
    return recs


def load_calms21_pretraining(pt_file: str | Path) -> list[Recording]:
    """Unlabeled CalMS21 keypoints for self-supervised pretraining.

    ``calms21_all_keypoints.pt``: ``{"sessions": [[T, 28], ...], "metadata": {...}}``,
    every session of every CalMS21 file (tasks 1-3 and the unlabeled videos).
    """
    d = torch.load(pt_file, map_location="cpu", weights_only=False)
    names = d.get("metadata", {}).get("session_names") or [f"session_{i}" for i in range(len(d["sessions"]))]
    return [Recording(name=str(n), keypoints=_flatten(s)) for n, s in zip(names, d["sessions"])]


# ----------------------------------------------------------------- MABe22
MABE22_STRAIN_LABELS = ["WT", "TG"]

_MABE22_KPTS_CACHE: dict[str, dict] = {}


def _mabe22_keypoints(path) -> dict:
    key = str(Path(path).resolve())
    if key not in _MABE22_KPTS_CACHE:
        _MABE22_KPTS_CACHE.clear()
        _MABE22_KPTS_CACHE[key] = torch.load(path, map_location="cpu", weights_only=False)
    return _MABE22_KPTS_CACHE[key]


def load_mabe22_state(keypoints_file: str | Path, labels_file: str | Path, regression: bool = False) -> list[Recording]:
    """MABe22 videos with one label each (strain, lights, experiment day, time of day).

    ``keypoints_file``: ``{video: [1800, 3 mice, 12 keypoints, 2]}`` in pixels,
    missing keypoints = 0. ``labels_file``: ``{video: value}``. Only videos in
    both are kept, sorted by name (the order :func:`stratified_split` expects).
    """
    kps = _mabe22_keypoints(keypoints_file)
    labels = torch.load(labels_file, map_location="cpu", weights_only=False)
    return [Recording(name=n, keypoints=_flatten(kps[n]), label=_as_label(labels[n], regression))
            for n in sorted(set(kps) & set(labels))]


def load_mabe22_frames(keypoints_file: str | Path, labels_file: str | Path) -> list[Recording]:
    """MABe22 videos with per-frame binary behavior labels (``{video: [1800]}``)."""
    kps = _mabe22_keypoints(keypoints_file)
    labels = torch.load(labels_file, map_location="cpu", weights_only=False)
    return [Recording(name=n, keypoints=_flatten(kps[n]),
                      frame_labels=torch.as_tensor(np.asarray(labels[n])).long())
            for n in sorted(set(kps) & set(labels))]


# ----------------------------------------------------------------- DeepEthogram
DEEPETHOGRAM_LABELS = ["background", "Locomote", "Rear", "FaceGroom", "BodyGroom", "Defecate"]


DEEPETHOGRAM_KEYPOINTS = ["nose", "left_ear", "right_ear", "base_neck", "left_forepaw", "right_forepaw",
                          "tail_base", "left_hindpaw", "right_hindpaw", "tail_tip", "body_center"]


def load_deepethogram(keypoints_dir: str | Path, labels_dir: str | Path) -> list[Recording]:
    """``keypoints/mouseXX.csv`` (22 normalized values ``nose_x, nose_y, ...``;
    a header row is optional) and ``labels/mouseXX_labels.csv`` (one-hot
    behavior columns).

    A frame's class is its active column; if several are active the first
    non-background one wins, and if none is active it is background.
    """
    import pandas as pd

    recs = []
    for kp_file in sorted(Path(keypoints_dir).glob("mouse*.csv")):
        name = kp_file.stem
        kp = pd.read_csv(kp_file, header=0 if _has_header(kp_file) else None).values.astype(np.float32)
        onehot = pd.read_csv(Path(labels_dir) / f"{name}_labels.csv", index_col=0)[DEEPETHOGRAM_LABELS].values > 0
        y = np.zeros(len(onehot), dtype=np.int64)
        for i, row in enumerate(onehot):
            active = np.flatnonzero(row)
            if len(active) == 1:
                y[i] = active[0]
            elif len(active) > 1:
                y[i] = next((a for a in active if a != 0), 0)
        if len(kp) != len(y):
            raise ValueError(f"{name}: {len(kp)} keypoint rows vs {len(y)} label rows")
        recs.append(Recording(name=name, keypoints=torch.from_numpy(kp), frame_labels=torch.from_numpy(y)))
    return recs


# ----------------------------------------------------------------- PAIR-R24M
PAIR24M_SINGLE_LABELS = ["Idle", "SmallMovement", "HeadTilt", "Groom", "Sniff", "Investigate", "RearUp",
                         "RearDown", "CrouchExplore", "Amble", "Locomotion"]
PAIR24M_SOCIAL_LABELS = ["None", "Close", "Explore", "Chase"]


def load_pair24m_single(single_dir: str | Path, multi_dir: Optional[str | Path] = None) -> list[Recording]:
    """Single-animal tracks for per-animal behavior (``behaviorCoarse`` 1-11 -> 0-10).

    Every file in ``single_dir`` gives one track (``single/<file>``); every
    file in ``multi_dir`` gives two (``multi/<file>/an1``, ``.../an2``). Frames
    without a valid label are dropped. 36 values per frame (12 markers x 3).
    """
    import pandas as pd

    recs = []
    sources = [(single_dir, "single", ["an1"])] + ([(multi_dir, "multi", ["an1", "an2"])] if multi_dir else [])
    for folder, prefix, animals in sources:
        for f in sorted(Path(folder).glob("*_markerDataset.csv")):
            df = pd.read_csv(f)
            for an in animals:
                cols = [c for c in df.columns if c.startswith(f"absolutePosition_{an}_")]
                lab = df[f"behaviorCoarse_{an}"].values
                ok = pd.notna(lab) & (lab >= 1) & (lab <= 11)
                name = f"{prefix}/{f.stem}" + (f"/{an}" if prefix == "multi" else "")
                recs.append(Recording(
                    name=name,
                    keypoints=torch.as_tensor(df[cols].values[ok], dtype=torch.float32),
                    frame_labels=torch.as_tensor((lab[ok] - 1).astype(np.int64)),
                ))
    return recs


def load_pair24m_social(multi_dir: str | Path) -> list[Recording]:
    """Two-animal recordings for social interactions (``interactionCat`` 0-3).

    72 values per frame: animal 1's 36 coordinates, then animal 2's.
    """
    import pandas as pd

    recs = []
    for f in sorted(Path(multi_dir).glob("*_markerDataset.csv")):
        df = pd.read_csv(f)
        cols = [c for c in df.columns if c.startswith("absolutePosition_")]
        recs.append(Recording(
            name=f.stem,
            keypoints=torch.as_tensor(df[cols].values, dtype=torch.float32),
            frame_labels=torch.as_tensor(df["interactionCat"].values.astype(np.int64)),
        ))
    return recs


# ----------------------------------------------------------------- SBeA
SBEA_GENOTYPES = ["ko_ko", "wt_ko", "wt_wt"]
SBEA_GENOTYPE_LABELS = ["KO-KO", "KO-WT", "WT-WT"]
SBEA_SINGLE_LABELS = ["crouch_explore", "locomotion", "groom", "rearing", "other"]
SBEA_SOCIAL_LABELS = ["close", "explore", "approach", "far"]


def _sbea_files(keypoints_root):
    for g, genotype in enumerate(SBEA_GENOTYPES):
        for f in sorted(Path(keypoints_root, genotype).glob("*.pt")):
            yield g, f, torch.load(f, map_location="cpu", weights_only=False)


def load_sbea_genotype(keypoints_root: str | Path) -> list[Recording]:
    """Whole recordings labelled by genotype pair: 0 KO-KO, 1 KO-WT, 2 WT-WT.

    ``<root>/{ko_ko,wt_ko,wt_wt}/*.pt`` with ``coords3d [27000, 96]``.
    """
    return [Recording(name=f.stem.removesuffix("-id3d"), keypoints=_flatten(d["coords3d"]), label=g)
            for g, f, d in _sbea_files(keypoints_root)]


def load_sbea_behavior(keypoints_root: str | Path, behaviors_dir: str | Path, mode: str = "social") -> list[Recording]:
    """The first 900 frames of each recording with our behavior annotations.

    ``mode="social"``: both mice (96 values per frame), interaction labels.
    ``mode="single"``: each mouse separately (48 values), individual labels;
    recordings are named ``<session>_<mouse id>``.
    """
    import pandas as pd

    if mode not in ("social", "single"):
        raise ValueError("mode must be 'social' or 'single'")
    recs = []
    for _, f, d in _sbea_files(keypoints_root):
        session = f.stem.removesuffix("-id3d")
        coords = _flatten(d["coords3d"])[:900]
        df = pd.read_csv(Path(behaviors_dir) / f"{session}_behaviors.csv")
        if mode == "social":
            recs.append(Recording(session, coords, torch.tensor(df["interact"].values, dtype=torch.long)))
        else:
            for k, mouse in enumerate(d["name3d"]):
                recs.append(Recording(f"{session}_{mouse}", coords[:, 48 * k: 48 * (k + 1)].contiguous(),
                                      torch.tensor(df[mouse].values, dtype=torch.long)))
    return recs


# ----------------------------------------------------------------- generic
def load_recordings(path: str | Path) -> list[Recording]:
    """Load ``{name: {"keypoints": [T, ...], "labels": [T] | int}}`` from .pt/.pkl/.npz.

    For ``.npz`` use keys ``<name>/keypoints`` and optionally ``<name>/labels``.
    """
    path = Path(path)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        data: dict = defaultdict(dict)
        for key in z.files:
            name, _, field = key.rpartition("/")
            data[name][field] = z[key]
    elif path.suffix in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        data = torch.load(path, map_location="cpu", weights_only=False)

    recs = []
    for name, item in data.items():
        if not isinstance(item, dict):
            item = {"keypoints": item}
        lab = item.get("labels")
        frame_labels, label = None, None
        if lab is not None:
            arr = np.asarray(lab)
            if arr.ndim == 0:
                label = arr.item()
            else:
                frame_labels = torch.as_tensor(arr, dtype=torch.long).reshape(-1)
        rec = Recording(name=str(name), keypoints=_flatten(item["keypoints"]), frame_labels=frame_labels, label=label)
        if frame_labels is not None and frame_labels.shape[0] != rec.num_frames:
            raise ValueError(f"{name}: {frame_labels.shape[0]} labels for {rec.num_frames} frames")
        recs.append(rec)
    return recs


def load_csv_recordings(
    path: str | Path,
    label_column: Optional[str] = "label",
    recording_column: Optional[str] = None,
    label_names: Optional[Sequence[str]] = None,
    ignore_columns: Sequence[str] = ("frame", "time", "time_s", "timestamp", "behavior"),
) -> list[Recording]:
    """Recordings from CSV files: a folder of CSVs (one recording each, named after the
    file) or a single CSV.

    Every column except ``label_column``, ``recording_column`` and ``ignore_columns``
    is a keypoint coordinate, in the order the model should read them (e.g.
    ``nose_x, nose_y, ...``). Labels may be class numbers or class names (names are
    matched against ``label_names``). With ``recording_column``, one CSV can hold
    several recordings. Files without a header row are read as keypoints only.
    """
    import pandas as pd

    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no .csv files in {path}")
    lookup = {str(n): i for i, n in enumerate(label_names or [])}
    recs = []
    for f in files:
        if not _has_header(f):
            df = pd.read_csv(f, header=None)
            recs.append(Recording(f.stem, torch.as_tensor(df.values, dtype=torch.float32)))
            continue
        df = pd.read_csv(f)
        groups = df.groupby(recording_column, sort=True) if recording_column else [(f.stem, df)]
        for name, g in groups:
            drop = {label_column, recording_column, *ignore_columns}
            cols = [c for c in g.columns if c not in drop]
            frame_labels = None
            if label_column and label_column in g.columns:
                v = g[label_column].values
                if v.dtype.kind in "OUS":
                    v = np.array(["" if (isinstance(x, float) and np.isnan(x)) else x for x in v], dtype=object)
                    lookup_ = dict(lookup, **{"": -1})
                    unknown = sorted(set(map(str, v)) - set(lookup_))
                    if unknown:
                        raise ValueError(f"{f.name}: labels {unknown[:5]} are not in label_names {list(lookup)}")
                    v = np.array([lookup_[str(x)] for x in v])
                v = pd.to_numeric(pd.Series(v), errors="coerce").fillna(-1).values  # blank -> -1 (ignored)
                frame_labels = torch.as_tensor(np.asarray(v, dtype=np.int64))
            name = str(name) if recording_column else f.stem
            recs.append(Recording(name, torch.as_tensor(g[cols].values.astype(np.float32)), frame_labels))
    return recs


def _has_header(csv_file) -> bool:
    with open(csv_file) as fh:
        first = fh.readline().split(",")[0].strip()
    try:
        float(first)
        return False
    except ValueError:
        return True


# ----------------------------------------------------------------- splits
Split = tuple[list[Recording], list[Recording], list[Recording]]


def stratified_split(
    recordings: list[Recording],
    train: float = 0.6,
    valid: float = 0.2,
    seed: int = 42,
    num_bins: int = 10,
    by: str = "positive_ratio",
) -> Split:
    """Video-level split stratified in bins.

    Videos are sorted by their fraction of positive frames
    (``by="positive_ratio"``) or by their label value (``by="value"``, for
    regression), cut into ``num_bins`` bins, and each bin is shuffled and
    split ``train : valid : rest``. Shuffling uses NumPy's global RNG seeded
    with ``seed``.
    """
    stats = []
    for i, r in enumerate(recordings):
        if by == "value":
            key = float(r.label)
        else:
            y = r.labels_per_frame()
            key = float((y > 0).sum()) / len(y) if y is not None and len(y) else 0.0
        stats.append((key, i))
    stats.sort(key=lambda s: s[0])  # stable: ties keep input order

    np.random.seed(seed)
    per_bin = len(stats) // num_bins
    tr, va, te = [], [], []
    for b in range(num_bins):
        chunk = list(stats[b * per_bin : (len(stats) if b == num_bins - 1 else (b + 1) * per_bin)])
        np.random.shuffle(chunk)
        n_tr = int(len(chunk) * train)
        n_va = int(len(chunk) * valid)
        tr += [recordings[i] for _, i in chunk[:n_tr]]
        va += [recordings[i] for _, i in chunk[n_tr : n_tr + n_va]]
        te += [recordings[i] for _, i in chunk[n_tr + n_va :]]
    return tr, va, te


def _recording_class(r: Recording) -> int:
    """A recording's single class: its ``label``, or the label shared by all its frames."""
    if r.label is not None:
        return int(r.label)
    y = r.labels_per_frame()
    values = [] if y is None else torch.unique(y[y >= 0]).tolist()
    if len(values) != 1:
        raise ValueError(f"recording {r.name!r} needs one label for the whole recording "
                         f"for a per-class split; found {values or 'none'}")
    return int(values[0])


def per_class_split(recordings: list[Recording], train: float = 0.6, valid: float = 0.2, seed: int = 42) -> Split:
    """Shuffle the recordings of each class (in class order) and split each class.

    Used for SBeA genotype (10 recordings per class -> 6 / 2 / 2).
    """
    np.random.seed(seed)
    by_class: dict[int, list[Recording]] = defaultdict(list)
    for r in recordings:
        by_class[_recording_class(r)].append(r)
    tr, va, te = [], [], []
    for c in sorted(by_class):
        items = list(by_class[c])
        np.random.shuffle(items)
        n_tr = int(len(items) * train)
        n_va = int(len(items) * valid)
        tr += items[:n_tr]
        va += items[n_tr : n_tr + n_va]
        te += items[n_tr + n_va :]
    return tr, va, te


def balanced_split(recordings: list[Recording], num_classes: int, train: float = 0.6, valid: float = 0.2) -> Split:
    """Deterministic greedy split that keeps each split's class mix close to the overall mix.

    Recordings are taken largest first and each goes to the split (with room
    left) whose class ratios would deviate least from the overall ratios
    (sum of absolute differences). Used for DeepEthogram and PAIR-R24M.
    """
    counts = []
    for r in recordings:
        y = r.labels_per_frame()
        counts.append(np.bincount(y[y >= 0].numpy(), minlength=num_classes)[:num_classes].astype(float))
    total = np.sum(counts, axis=0)
    target = total / total.sum()
    n = len(recordings)
    caps = [int(n * train), int(n * valid), n - int(n * train) - int(n * valid)]
    order = sorted(range(n), key=lambda i: -len(recordings[i].labels_per_frame()))
    members: list[list[int]] = [[], [], []]
    frames = [np.zeros(num_classes) for _ in range(3)]
    sizes = [0, 0, 0]
    for i in order:
        size = len(recordings[i].labels_per_frame())
        div = []
        for s in range(3):
            if len(members[s]) < caps[s]:
                div.append(np.abs((frames[s] + counts[i]) / (sizes[s] + size) - target).sum())
            else:
                div.append(np.inf)
        if div[0] <= div[1] and div[0] <= div[2] and len(members[0]) < caps[0]:
            s = 0
        elif div[1] <= div[2] and len(members[1]) < caps[1]:
            s = 1
        elif len(members[2]) < caps[2]:
            s = 2
        else:
            s = int(np.argmin([len(m) for m in members]))
        members[s].append(i)
        frames[s] += counts[i]
        sizes[s] += size
    return tuple([recordings[i] for i in m] for m in members)  # type: ignore[return-value]


def fixed_split(recordings: list[Recording], train: Sequence[str], valid: Sequence[str], test: Sequence[str]) -> Split:
    """Split by explicit recording names (as stored in the experiment configs)."""
    by_name = {r.name: r for r in recordings}
    out = []
    for names in (train, valid, test):
        missing = [n for n in names if n not in by_name]
        if missing:
            raise KeyError(f"recordings not found: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
        out.append([by_name[n] for n in names])
    return tuple(out)  # type: ignore[return-value]
