"""Build data, model, loss and sampler from a YAML experiment config.

See ``configs/`` for the paper experiments, and ``configs/custom_token.yaml``
(per-frame labels) or ``configs/custom_state.yaml`` (one label per recording) for
your own data. A config may start with ``base: <other.yaml>`` (relative to
itself) and may offer several variants under ``tasks:``, chosen with
``--task``. Data paths are resolved against ``data_root`` unless absolute.
"""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from sklearn.model_selection import train_test_split

from . import report
from .data import recordings as R
from .data.augmentation import augment, get_arena
from .data.dataset import ProportionalBatchSampler, WindowDataset, subset_windows, windows_from_recordings
from .data.windows import window_starts
from .inference import predict_windows
from .losses import WEIGHTED_LOSSES, build_loss
from .metrics import classification_report, recording_decision, regression_report
from .model import BehaVERT, BehaVERTConfig

# where `behavert download` puts the benchmark data (run commands from the repository root)
DEFAULT_DATA_ROOT = os.environ.get("BEHAVERT_DATA", "dataset")
DATASET_REPO = "ShinSeungJ/behavert-data"
BENCHMARKS = ("calms21", "mabe22", "pair24m", "deepethogram", "sbea")


# ------------------------------------------------------------------ config files
def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _read_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = cfg.pop("base", None)
    if base:
        cfg = _merge(_read_config(path.parent / base), cfg)
    return cfg


def config_tasks(path: str | Path) -> list[str]:
    """Names of the tasks a config offers (empty if it has a single setup)."""
    return list((_read_config(Path(path)).get("tasks") or {}).keys())


def load_config(path: str | Path, task: Optional[str] = None) -> dict:
    """Read a config; for a multi-task config, apply the chosen task.

    A config may list variants under ``tasks: {name: {...overrides...}}``.
    ``task`` (or the config's own ``task:`` key) picks one; its keys are
    deep-merged over the rest, and ``{task}`` in ``output_dir`` is filled in.
    """
    cfg = _read_config(Path(path))
    tasks = cfg.pop("tasks", None)
    if tasks:
        task = task or cfg.get("task")
        if task not in tasks:
            raise SystemExit(f"{path}: choose a task with --task, one of: {', '.join(tasks)}")
        cfg = _merge(cfg, tasks[task] or {})
        cfg["task"] = task
    elif task:
        raise SystemExit(f"{path} has no tasks; drop --task")
    if "output_dir" in cfg and cfg.get("task"):
        cfg["output_dir"] = str(cfg["output_dir"]).format(task=cfg["task"])
    return cfg


def _path(p: str, root: str | Path) -> Path:
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else Path(root) / p


def is_regression(cfg: dict) -> bool:
    return cfg.get("model", {}).get("problem", "classification") == "regression"


# ------------------------------------------------------------------ recordings
def _load_pool(data: dict, root, regression: bool) -> list[R.Recording]:
    fmt = data["format"]
    P = lambda key: _path(data[key], root)  # noqa: E731
    if fmt == "mabe22_state":
        return R.load_mabe22_state(P("keypoints"), P("labels"), regression=regression)
    if fmt == "mabe22_frames":
        return R.load_mabe22_frames(P("keypoints"), P("labels"))
    if fmt == "deepethogram":
        return R.load_deepethogram(P("keypoints"), P("labels"))
    if fmt == "pair24m_single":
        return R.load_pair24m_single(P("single"), P("multi") if data.get("multi") else None)
    if fmt == "pair24m_social":
        return R.load_pair24m_social(P("multi"))
    if fmt == "sbea_genotype":
        return R.load_sbea_genotype(P("keypoints"))
    if fmt == "sbea_behavior":
        return R.load_sbea_behavior(P("keypoints"), P("behaviors"), data.get("mode", "social"))
    raise ValueError(f"unknown data format {fmt!r}")


def _count(folder, pattern: str) -> int:
    return len(list(Path(folder).glob(pattern)))


def _load_pool_checked(cfg: dict, root) -> list[R.Recording]:
    """``_load_pool`` with the loading check: each source, then what was matched."""
    data = cfg["data"]
    fmt = data["format"]
    P = lambda key: _path(data[key], root)  # noqa: E731
    names = cfg.get("model", {}).get("label_names") or []
    if fmt in ("mabe22_state", "mabe22_frames"):
        report.loading("keypoints", P("keypoints"))
        report.loaded(f"Loaded {len(R._mabe22_keypoints(P('keypoints'))):,} recordings")
        task = cfg.get("task")
        report.loading(f"{task} labels" if task else "labels", P("labels"))
        report.loaded(f"Loaded {len(torch.load(P('labels'), map_location='cpu', weights_only=False)):,} recordings")
        recs = _load_pool(data, root, is_regression(cfg))
        report.loaded(report.recordings_summary(recs, "Matched"))
    elif fmt == "deepethogram":
        report.loading("keypoints", P("keypoints"))
        report.loaded(f"Loaded {_count(P('keypoints'), 'mouse*.csv'):,} recordings")
        report.loading("labels", P("labels"))
        report.loaded(f"Loaded {_count(P('labels'), '*_labels.csv'):,} recordings")
        recs = _load_pool(data, root, False)
        report.loaded(report.recordings_summary(recs, "Matched"))
    elif fmt == "pair24m_single":
        report.loading("single-animal data", P("single"))
        report.loaded(f"Loaded {_count(P('single'), '*_markerDataset.csv'):,} recordings")
        if data.get("multi"):
            n = _count(P("multi"), "*_markerDataset.csv")
            report.loading("two-animal data", P("multi"), "each animal is its own recording")
            report.loaded(f"Loaded {n:,} files, {2 * n:,} recordings")
        recs = _load_pool(data, root, False)
        report.loaded(report.recordings_summary(recs, "In total"))
    elif fmt == "pair24m_social":
        report.loading("two-animal data", P("multi"))
        recs = _load_pool(data, root, False)
        report.loaded(report.recordings_summary(recs))
    elif fmt == "sbea_genotype":
        report.loading("keypoints", P("keypoints"))
        recs = _load_pool(data, root, False)
        counts = np.bincount([int(r.label) for r in recs], minlength=len(names))
        groups = ", ".join(f"{n}: {c}" for n, c in zip(names, counts))
        summary = report.recordings_summary(recs)
        head, tail = summary.split(" recordings, ", 1)
        report.loaded(f"{head} recordings ({groups}), {tail}")
    elif fmt == "sbea_behavior":
        report.loading("keypoints", P("keypoints"))
        report.loading("behavior labels", P("behaviors"), data.get("mode", "social"))
        recs = _load_pool(data, root, False)
        report.loaded(report.recordings_summary(recs))
    else:
        recs = _load_pool(data, root, is_regression(cfg))
    return recs


FILE_KEYS = (("train", "data"), ("valid", "validation data"), ("test", "test data"))


def _load_file(data: dict, key: str, root, label_names=None) -> Optional[list[R.Recording]]:
    if not data.get(key):
        return None
    p = _path(data[key], root)
    fmt = data["format"]
    if fmt == "calms21":
        return R.load_calms21(p, group=data.get("group"))
    if fmt == "calms21_pretraining":
        return R.load_calms21_pretraining(p)
    if fmt == "recordings":
        return R.load_recordings(p)
    if fmt == "csv":
        return R.load_csv_recordings(p, label_column=data.get("label_column", "label"),
                                     recording_column=data.get("recording_column"),
                                     label_names=label_names)
    raise ValueError(f"format {fmt!r} does not use '{key}' files")


FILE_FORMATS = ("calms21", "calms21_pretraining", "recordings", "csv")


def load_split_recordings(cfg: dict, root, seed: int) -> dict[str, Optional[list[R.Recording]]]:
    """Return ``{"train", "valid", "test"}`` recording lists.

    ``valid`` is ``None`` when validation windows are carved out of the
    training windows later (``split.method: random_windows``).
    """
    data = cfg["data"]
    split = data.get("split", {"method": "random_windows", "valid": 0.2})
    method = split.get("method", "random_windows")
    n_cls = cfg.get("model", {}).get("num_classes", 2)

    if data["format"] in FILE_FORMATS:
        names = cfg.get("model", {}).get("label_names")
        files = {}
        for key, what in FILE_KEYS:
            if data.get(key):
                report.loading(what, _path(data[key], root), f"behavior: {data['group']}" if data.get("group") else "")
                files[key] = _load_file(data, key, root, names)
                report.loaded(report.recordings_summary(files[key]))
        train, valid, test = files.get("train"), files.get("valid"), files.get("test")
        if method == "test_as_valid":  # paper CalMS21 Task 3: no separate validation set
            return {"train": train, "valid": test, "test": test}
        if method == "random_windows" or valid is not None:
            return {"train": train, "valid": valid, "test": test}
        pool, given_test = train, test
    else:
        pool, given_test = _load_pool_checked(cfg, root), None

    tr_frac, va_frac = split.get("train", 0.6), split.get("valid", 0.2)
    if given_test:  # a separate test file: the pool only provides train and valid
        tr_frac = 1.0 - va_frac
    if method == "stratified":
        tr, va, te = R.stratified_split(pool, tr_frac, va_frac, seed, by=split.get("by", "positive_ratio"))
    elif method == "per_class":
        tr, va, te = R.per_class_split(pool, tr_frac, va_frac, seed)
    elif method == "balanced":
        tr, va, te = R.balanced_split(pool, n_cls, tr_frac, va_frac)
    elif method == "fixed":
        tr, va, te = R.fixed_split(pool, split["train"], split["valid"], split["test"])
    elif method == "random_recordings":
        idx = np.random.RandomState(seed).permutation(len(pool))
        n_tr, n_va = int(len(pool) * tr_frac), int(len(pool) * va_frac)
        tr, va, te = ([pool[i] for i in idx[:n_tr]], [pool[i] for i in idx[n_tr:n_tr + n_va]],
                      [pool[i] for i in idx[n_tr + n_va:]])
    else:
        raise ValueError(f"unknown split method {method!r}")
    test = given_test or te
    report.step_line(f"Splitting recordings: {_split_description(cfg, method, split, tr_frac, va_frac, seed, given_test)}",
                     f"train: {len(tr):,} recordings, valid: {len(va):,}, test: {len(test):,}")
    return {"train": tr, "valid": va, "test": test}


def _split_description(cfg: dict, method: str, split: dict, tr: float, va: float, seed: int, given_test) -> str:
    if method == "fixed":
        return "fixed lists in the config"
    shares = f"train {tr:g}, valid {va:g}" + ("" if given_test else f", test {1 - tr - va:.2g}")
    if method == "stratified":
        if split.get("by") == "value":
            how = "stratified by value"
        elif cfg.get("model", {}).get("mode") == "sequence":
            how = "stratified by label"
        else:
            how = f"stratified by {cfg.get('task') or 'labelled'} frames"
    elif method == "per_class":
        how = "per class"
    elif method == "balanced":
        return f"balanced by class frames ({shares})"
    else:
        how = "random"
    return f"{how} ({shares}; seed {seed})"


def count_windows(recs: list[R.Recording], window: dict) -> int:
    """Number of windows ``make_windows`` would build, without building them."""
    L = window["length"]
    s = resolve_stride(window)
    pad = window.get("pad_remainder", True)
    lengths = [sum(r.num_frames for r in recs)] if window.get("concat_recordings") else [r.num_frames for r in recs]
    return sum(len(window_starts(T, L, s, pad)[0]) for T in lengths)


def resolve_stride(window: dict) -> int:
    """``data.window.stride`` in frames.

    A value in (0, 1] (``0.5``, ``"1/2"``, ``0.25``, ``"1/4"``; ``1`` = the whole
    window) is a share of the window length; a whole number above 1 is a number
    of frames. Unset means half a window.
    """
    from fractions import Fraction

    L = window["length"]
    spec = window.get("stride")
    if spec is None:
        return max(1, L // 2)
    try:
        value = Fraction(str(spec).strip())
    except (ValueError, ZeroDivisionError):
        raise ValueError(f"data.window.stride must be a fraction (0.5, 1/2) or a number of frames (32), "
                         f"got {spec!r}") from None
    if 0 < value <= 1:
        frames = max(1, int(round(float(value) * L)))
    elif value > 1 and value.denominator == 1:
        frames = int(value)
    else:
        raise ValueError(f"data.window.stride must be a fraction in (0, 1] or a whole number of frames, got {spec!r}")
    if frames > L:
        raise ValueError(f"data.window.stride={spec} is {frames} frames, more than the window length ({L})")
    return frames


def stride_description(window: dict) -> str:
    """The stride for the log: ``32``, or ``32 (1/2 of the window)`` when given as a share."""
    frames, spec = resolve_stride(window), window.get("stride")
    if spec is None or (isinstance(spec, int) and spec > 1):
        return str(frames)
    share = "the whole window" if frames == window["length"] else f"{spec} of the window"
    return f"{frames} ({share})"


def make_windows(recs: list[R.Recording], window: dict):
    return windows_from_recordings(
        recs,
        length=window["length"],
        stride=resolve_stride(window),
        pad_remainder=window.get("pad_remainder", True),
        concat=window.get("concat_recordings", False),
    )


@dataclass
class Datasets:
    train: WindowDataset
    valid: Optional[WindowDataset]
    test_recordings: Optional[list[R.Recording]]
    train_recordings: Optional[int] = None  # None when train / valid were split by windows
    valid_recordings: Optional[int] = None


def build_augmentation(aug_cfg: Optional[dict]):
    """``(augment_fn, classes)`` from the ``data.augmentation`` block (``(None, None)`` if off)."""
    if not aug_cfg or not aug_cfg.get("enabled", True):
        return None, None
    arena = get_arena(aug_cfg["arena"])
    return (lambda x: augment(x, arena)), aug_cfg.get("classes")


def build_datasets(cfg: dict, root, seed: int, sequence_labels: bool) -> Datasets:
    data = cfg["data"]
    window = data["window"]
    recs = load_split_recordings(cfg, root, seed)
    augment_fn, augment_classes = build_augmentation(data.get("augmentation"))
    method = data.get("split", {}).get("method", "random_windows")
    how = f"{window['length']} frames, stride {resolve_stride(window)}"
    if window.get("concat_recordings"):
        how += ", recordings joined"
    if sequence_labels:
        how += "; one value per sequence" if is_regression(cfg) else "; one label per sequence"

    train_w, _ = make_windows(recs["train"], window)
    n_tr_rec = n_va_rec = None
    if recs["valid"] is None:
        report.step_line(f"Generating sequences ({how})", f"{len(train_w):,} sequences")
        frac = data.get("split", {}).get("valid", 0.2)
        if frac:
            tr_idx, va_idx = train_test_split(np.arange(len(train_w)), test_size=frac, random_state=seed)
            valid_w = subset_windows(train_w, va_idx)
            train_w = subset_windows(train_w, tr_idx)
            report.step_line(f"Splitting sequences: random {frac * 100:g}% for validation (seed {seed})",
                             f"train: {len(train_w):,} sequences, valid: {len(valid_w):,} sequences")
        else:
            valid_w = None
    else:
        valid_w, _ = make_windows(recs["valid"], window)
        if method == "test_as_valid":
            report.step_line(f"Generating sequences ({how})",
                             f"train: {len(train_w):,} sequences, test: {len(valid_w):,} sequences")
            report.step_line("Splitting: the test recordings are used for validation",
                             f"train: {len(train_w):,} sequences, valid: {len(valid_w):,} sequences")
        else:
            n_tr_rec, n_va_rec = len(recs["train"]), len(recs["valid"])
            n_te = count_windows(recs["test"], window) if recs["test"] else 0
            report.step_line(f"Generating sequences ({how})",
                             f"train: {len(train_w):,} sequences, valid: {len(valid_w):,}, test: {n_te:,}")

    train = WindowDataset(train_w, augment_fn, augment_classes, sequence_labels)
    valid = None if valid_w is None else WindowDataset(valid_w, sequence_labels=sequence_labels)
    return Datasets(train, valid, recs["test"], n_tr_rec, n_va_rec)


def print_distribution(title: str, windows, cfg: dict, sequence_labels: bool, recordings: Optional[int] = None,
                       pretraining: bool = False) -> None:
    """``<title> Dataset Distribution`` for a set of windows."""
    names = cfg.get("model", {}).get("label_names") or []
    if pretraining or windows.labels is None:
        report.size_distribution(f"{title} Dataset Distribution", int(windows.mask.sum()), len(windows.inputs))
    elif is_regression(cfg):
        report.value_distribution(f"{title} Dataset Distribution", windows.labels[:, 0].numpy(), "sequences",
                                  recordings)
    elif sequence_labels:
        y = windows.labels[:, 0].long()
        counts = torch.bincount(y[y >= 0], minlength=len(names)).tolist()
        report.class_distribution(f"{title} Dataset Distribution", counts, names, "sequences", recordings)
    else:
        y = windows.labels.reshape(-1)
        y = y[y >= 0].long()
        counts = torch.bincount(y, minlength=len(names)).tolist()
        report.class_distribution(f"{title} Dataset Distribution", counts, names, "frames")


# ------------------------------------------------------------------ model / loss
def build_model(cfg: dict, input_dim: int) -> BehaVERT:
    m = dict(cfg["model"])
    init_from = m.pop("init_from", None)
    w = cfg["data"]["window"]
    preprocessing = m.pop("preprocessing", {})
    preprocessing.setdefault("window", w["length"])
    preprocessing.setdefault("stride", resolve_stride(w))
    config = BehaVERTConfig(input_dim=input_dim, max_seq_length=w["length"], preprocessing=preprocessing, **m)
    model = BehaVERT(config)
    if init_from:
        model.load_encoder(init_from)
    return model


def build_loss_fn(cfg: dict, train: WindowDataset, num_classes: int):
    loss_cfg = dict(cfg.get("loss", {"name": "hybrid"}))
    name = loss_cfg.pop("name")
    # the weighted losses weight classes by the training data's inverse class proportions
    # unless the config gives class_weights; class_weight_multipliers scale either
    props = train.class_proportions(num_classes) if name in WEIGHTED_LOSSES and "class_weights" not in loss_cfg else None
    return build_loss(name, props, **loss_cfg)


def build_sampler(cfg: dict, train: WindowDataset, batch_size: int, num_classes: int):
    if cfg.get("sampler", "proportional") != "proportional" or is_regression(cfg):
        return None
    return ProportionalBatchSampler(train.window_classes(), batch_size, train.class_proportions(num_classes))


# ------------------------------------------------------------------ test evaluation
def test_windows(cfg: dict, test_recs: list[R.Recording]):
    """The test windows (and their recording index), built once for every checkpoint."""
    return make_windows(test_recs, cfg["data"]["window"])


def evaluate_test(model: BehaVERT, cfg: dict, test_recs: list[R.Recording], batch_size: int = 256,
                  windows=None) -> dict:
    """Test metrics using the same windowing as training.

    * Token models: frame-level report over all windows (with overlapping
      windows a frame counts once per window, as in the paper).
    * Sequence classifiers: window-level report, plus a recording-level report
      where each recording is labelled from its averaged window probabilities.
    * Regression: window-level MSE.
    """
    mc = model.config
    window = cfg["data"]["window"]
    tcfg = cfg.get("train", {})
    ignore = tcfg.get("metric_ignore_classes", [])
    avg = tcfg.get("metric_average_classes")
    w, owner = windows if windows is not None else make_windows(test_recs, window)
    out = {}
    if mc.problem == "regression":
        pred = predict_windows(model, w, batch_size).reshape(-1)
        out["windows"] = regression_report(w.labels[:, 0].numpy(), pred.numpy())
        return out
    probs = predict_windows(model, w, batch_size)
    if mc.mode == "token":
        out["frames"] = classification_report(w.labels.numpy(), probs.argmax(-1).numpy(),
                                              mc.num_classes, ignore, mc.label_names, avg)
        return out
    out["windows"] = classification_report(w.labels[:, 0].numpy(), probs.argmax(-1).numpy(),
                                           mc.num_classes, ignore, mc.label_names, avg)
    y_true, y_pred = [], []
    for i, r in enumerate(test_recs):
        decision, _ = recording_decision(probs[(owner == i).numpy()].numpy())
        y_true.append(int(r.label) if r.label is not None else int(r.labels_per_frame()[0]))
        y_pred.append(decision)
    out["recordings"] = classification_report(np.array(y_true), np.array(y_pred),
                                              mc.num_classes, ignore, mc.label_names, avg)
    return out


def resolve_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
