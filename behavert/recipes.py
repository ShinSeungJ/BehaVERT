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

from .data import recordings as R
from .data.augmentation import augment, get_arena
from .data.dataset import ProportionalBatchSampler, WindowDataset, subset_windows, windows_from_recordings
from .inference import predict_windows
from .losses import WEIGHTED_LOSSES, build_loss
from .metrics import classification_report, format_report, recording_decision, regression_report
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
        train, test = _load_file(data, "train", root, names), _load_file(data, "test", root, names)
        valid = _load_file(data, "valid", root, names)
        if method == "test_as_valid":  # paper CalMS21 Task 3: no separate validation set
            return {"train": train, "valid": test, "test": test}
        if method == "random_windows" or valid is not None:
            return {"train": train, "valid": valid, "test": test}
        pool, given_test = train, test
    else:
        pool, given_test = _load_pool(data, root, is_regression(cfg)), None

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
    return {"train": tr, "valid": va, "test": given_test or te}


def make_windows(recs: list[R.Recording], window: dict):
    return windows_from_recordings(
        recs,
        length=window["length"],
        stride=window.get("stride"),
        pad_remainder=window.get("pad_remainder", True),
        concat=window.get("concat_recordings", False),
    )


@dataclass
class Datasets:
    train: WindowDataset
    valid: Optional[WindowDataset]
    test_recordings: Optional[list[R.Recording]]


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

    train_w, _ = make_windows(recs["train"], window)
    if recs["valid"] is None:
        frac = data.get("split", {}).get("valid", 0.2)
        if frac:
            tr_idx, va_idx = train_test_split(np.arange(len(train_w)), test_size=frac, random_state=seed)
            valid_w = subset_windows(train_w, va_idx)
            train_w = subset_windows(train_w, tr_idx)
        else:
            valid_w = None
    else:
        valid_w, _ = make_windows(recs["valid"], window)

    train = WindowDataset(train_w, augment_fn, augment_classes, sequence_labels)
    valid = None if valid_w is None else WindowDataset(valid_w, sequence_labels=sequence_labels)
    return Datasets(train, valid, recs["test"])


# ------------------------------------------------------------------ model / loss
def build_model(cfg: dict, input_dim: int) -> BehaVERT:
    m = dict(cfg["model"])
    init_from = m.pop("init_from", None)
    w = cfg["data"]["window"]
    preprocessing = m.pop("preprocessing", {})
    preprocessing.setdefault("window", w["length"])
    preprocessing.setdefault("stride", w.get("stride") or w["length"] // 2)
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
    loss_fn = build_loss(name, props, **loss_cfg)
    weights = getattr(loss_fn, "weights", None)
    shown = {k: v for k, v in loss_cfg.items() if k not in ("class_weights", "class_weight_multipliers")}
    if weights is not None:
        print(f"loss: {name}  class weights: {np.round(weights.tolist(), 4).tolist()}  {shown}")
    else:
        print(f"loss: {name}  {loss_cfg}")
    return loss_fn


def build_sampler(cfg: dict, train: WindowDataset, batch_size: int, num_classes: int):
    if cfg.get("sampler", "proportional") != "proportional" or is_regression(cfg):
        return None
    return ProportionalBatchSampler(train.window_classes(), batch_size, train.class_proportions(num_classes))


# ------------------------------------------------------------------ test evaluation
def evaluate_test(model: BehaVERT, cfg: dict, test_recs: list[R.Recording], batch_size: int = 256) -> dict:
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
    w, owner = make_windows(test_recs, window)
    out = {}
    if mc.problem == "regression":
        pred = predict_windows(model, w, batch_size).reshape(-1)
        out["windows"] = regression_report(w.labels[:, 0].numpy(), pred.numpy())
        print(format_report(out["windows"], "test (windows):"))
        return out
    probs = predict_windows(model, w, batch_size)
    if mc.mode == "token":
        out["frames"] = classification_report(w.labels.numpy(), probs.argmax(-1).numpy(),
                                              mc.num_classes, ignore, mc.label_names, avg)
        print(format_report(out["frames"], "test (frames):"))
        return out
    out["windows"] = classification_report(w.labels[:, 0].numpy(), probs.argmax(-1).numpy(),
                                           mc.num_classes, ignore, mc.label_names, avg)
    print(format_report(out["windows"], "test (windows):"))
    y_true, y_pred = [], []
    for i, r in enumerate(test_recs):
        decision, _ = recording_decision(probs[(owner == i).numpy()].numpy())
        y_true.append(int(r.label) if r.label is not None else int(r.labels_per_frame()[0]))
        y_pred.append(decision)
    out["recordings"] = classification_report(np.array(y_true), np.array(y_pred),
                                              mc.num_classes, ignore, mc.label_names, avg)
    print(format_report(out["recordings"], "test (recordings):"))
    return out


def resolve_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
