"""Terminal output of ``behavert train`` / ``pretrain`` / ``evaluate``.

All the printing of a training run lives here: the configuration block, the
dataset loading check and class distributions, the training and validation
lines, the hyperparameters, and the test table.
"""

from __future__ import annotations

import math
import textwrap
from typing import Iterable, Optional, Sequence

import numpy as np

WIDE = 92      # training-step rules and the test table
NARROW = 60    # configuration and hyperparameter banners
KEY = 21       # width of the "  key : value" labels
ITEM = 27      # width of the "  Class i (name) : count" labels


def _p(text: str = "") -> None:
    print(text, flush=True)


def banner(title: str, width: int = NARROW) -> None:
    _p("=" * width)
    _p(title)
    _p("=" * width)


def key_values(rows: Iterable[tuple[str, object]]) -> None:
    """``  Key                  : value`` lines; long values wrap under the value column."""
    indent = " " * (2 + KEY + 2)
    for key, value in rows:
        text = f"  {key:<{KEY}}: {value}"
        if len(text) <= WIDE:
            _p(text)
        else:
            _p(textwrap.fill(str(value), width=WIDE, initial_indent=f"  {key:<{KEY}}: ",
                             subsequent_indent=indent, break_on_hyphens=False, break_long_words=False))


def block(title: str, rows: Iterable[tuple[str, object]], close: bool = True) -> None:
    """A banner and aligned ``key : value`` rows; ``close`` adds the closing rule."""
    banner(title)
    key_values(rows)
    if close:
        _p("=" * NARROW)
        _p()


def number(v) -> str:
    """Config numbers as Python writes them (1e-05, 0.0001, 512)."""
    return repr(v) if isinstance(v, float) else str(v)


def named(ids: Sequence[int], names: Optional[Sequence[str]]) -> str:
    if names:
        return f"{list(ids)} ({', '.join(names[i] for i in ids)})"
    return str(list(ids))


# ------------------------------------------------------------------ loading check
def loading(what: str, path, note: str = "") -> None:
    _p(f"Loading {what} from: {path}" + (f" ({note})" if note else ""))


def loaded(text: str) -> None:
    _p(f"  ✅ {text}")


def skipped(name: str, reason: str) -> None:
    _p(f"  ⚠️ Skipping {name}: {reason}")


def step_line(title: str, result: str) -> None:
    _p(title)
    _p(f"  → {result}")


def recordings_summary(recs, prefix: str = "Loaded") -> str:
    frames = sum(r.num_frames for r in recs)
    values = recs[0].keypoints.reshape(recs[0].num_frames, -1).shape[1] if recs else 0
    return f"{prefix} {len(recs):,} recordings, {frames:,} frames, {values} values per frame"


# ------------------------------------------------------------------ distributions
def class_distribution(title: str, counts: Sequence[int], names: Sequence[str], unit: str = "frames",
                       recordings: Optional[int] = None) -> None:
    total = int(sum(counts))
    _p(f"{title}:")
    where = f" from {recordings:,} recordings" if recordings is not None else ""
    _p(f"  {'Total samples':<{ITEM}}: {total:,} {unit}{where}")
    for i, c in enumerate(counts):
        name = names[i] if i < len(names) else str(i)
        share = c / total if total else 0.0
        _p(f"  {f'Class {i} ({name})':<{ITEM}}: {int(c):>9,}   Proportion: {share:.5f}")


def value_distribution(title: str, values: np.ndarray, unit: str, recordings: Optional[int] = None) -> None:
    v = np.asarray(values, dtype=np.float64)
    _p(f"{title}:")
    where = f" from {recordings:,} recordings" if recordings is not None else ""
    _p(f"  {'Total samples':<{ITEM}}: {len(v):,} {unit}{where}")
    if len(v):
        _p(f"  {'Value range':<{ITEM}}: [{v.min():.3f}, {v.max():.3f}]")
        _p(f"  {'Mean ± std':<{ITEM}}: {v.mean():.3f} ± {v.std():.3f}")


def size_distribution(title: str, frames: int, sequences: int) -> None:
    _p(f"{title}:")
    _p(f"  {'Total samples':<{ITEM}}: {frames:,} frames in {sequences:,} sequences")


# ------------------------------------------------------------------ training loop
class TrainingReport:
    """Train / validation lines between rules, and the checkpoint messages.

    ``kind`` is ``"classification"``, ``"regression"`` or ``"pretraining"``.
    ``label_title`` heads the validation table (``behavior``, ``state``, ...);
    ``units`` is appended to pretraining errors (e.g. ``px``).
    """

    def __init__(self, kind: str = "classification", label_title: str = "behavior", units: str = ""):
        self.kind = kind
        self.label_title = label_title
        self.units = f" {units}" if units else ""
        self._open_step: Optional[int] = None

    def _open(self, step: int) -> None:
        if self._open_step != step:
            _p("-" * WIDE)
            self._open_step = step

    def close(self) -> None:
        if self._open_step is not None:
            _p("-" * WIDE)
            self._open_step = None

    def train(self, epoch: int, step: int, loss: float, lr: float) -> None:
        self._open(step)
        head = f"epoch: {epoch}, (step: {step})"
        if self.kind == "pretraining":
            _p(f"{head} | train loss (MSE): {loss:.2f} | RMSE: {math.sqrt(max(loss, 0.0)):.2f}{self.units} "
               f"| lr: {lr:.8f}")
        elif self.kind == "regression":
            _p(f"{head} | train loss (MSE): {loss:.8f} | lr: {lr:.8f}")
        else:
            _p(f"{head} | train loss: {loss:.8f} | lr: {lr:.8f}")

    def valid(self, epoch: int, step: int, rep: dict) -> None:
        self._open(step)
        head = f"epoch: {epoch}, (step: {step})"
        if self.kind == "pretraining":
            _p(f"{head} | valid loss (MSE): {rep['loss']:.2f} | RMSE: {rep['rmse']:.2f}{self.units} "
               f"| MAE: {rep['mae']:.2f}{self.units}")
            return
        if self.kind == "regression":
            _p(f"{head} | valid loss (MSE): {rep['loss']:.8f} | MAE: {rep['mae']:.6f} "
               f"| RMSE: {math.sqrt(rep['mse']):.6f} | R²: {r_squared(rep):.6f}")
            return
        _p(f"{head} | valid loss: {rep['loss']:.8f}")
        _p(f"  {self.label_title:<20}{'precision':>11}{'recall':>11}{'f1':>11}")
        for name, m in rep.get("per_class", {}).items():
            _p(f"  {name:<20}{m['precision']:>11.6f}{m['recall']:>11.6f}{m['f1']:>11.6f}")
        _p(f"Overall Validation Metrics - Accuracy: {rep['accuracy']:.6f}, "
           f"Class-Averaged F1: {rep['macro_f1']:.6f}")

    @staticmethod
    def saved(name: str, path) -> None:
        _p(f"Model checkpoint '{name}' saved successfully to {path}.")

    def early_stop(self, epoch: int, epochs: int) -> None:
        _p(f"Early stop at epoch {epoch}/{epochs}.")


def r_squared(rep: dict) -> float:
    var = rep.get("label_variance", 0.0)
    return 1.0 - rep["mse"] / var if var > 0 else float("nan")


# ------------------------------------------------------------------ test table
CHECKPOINT_TITLES = {"best_loss": "Best Loss", "best_f1": "Best F1", "best_f1_class1": "Best F1", "last": "Last"}


def checkpoint_title(name: str) -> str:
    return CHECKPOINT_TITLES.get(name, name)


def _rule(indent: int = 0) -> str:
    return " " * indent + "-" * (WIDE - indent)


def _class_rows(first: str, indent: int, class_width: int, rep: dict) -> list[str]:
    lines = []
    pad = " " * indent
    for i, (name, m) in enumerate(rep.get("per_class", {}).items()):
        lead = first if i == 0 else pad
        lines.append(f"{lead}{name:<{class_width}}{m['precision']:>11.4f}{m['recall']:>11.4f}{m['f1']:>11.4f}")
    lines.append(_rule(indent))
    lines.append(f"{pad}Accuracy: {rep['accuracy']:.4f}   Class-Averaged F1: {rep['macro_f1']:.4f}")
    return lines


def inference_table(results: dict[str, dict], kind: str, units: str = "") -> None:
    """One block per checkpoint.

    ``kind``: ``"token"`` (results[name]["frames"]), ``"sequence"``
    (["windows"] and ["recordings"]), ``"regression"`` (["windows"]) or
    ``"pretraining"`` (MSE / RMSE / MAE).
    """
    _p()
    banner("INFERENCE ON TEST SET", WIDE)
    u = f" {units}" if units else ""
    if kind == "token":
        _p(f"{'Model':<15}{'Class':<26}{'Precision':>11}{'Recall':>11}{'F1':>11}")
    elif kind == "sequence":
        _p(f"{'Model':<15}{'Level':<13}{'Class':<21}{'Precision':>11}{'Recall':>11}{'F1':>11}")
    elif kind == "regression":
        _p(f"{'Model':<15}{'MSE':>15}{'MAE':>13}{'RMSE':>13}{'R²':>13}")
    else:
        _p(f"{'Model':<15}{'MSE':>15}{'RMSE':>13}{'MAE':>13}")
    _p(_rule())
    for name, res in results.items():
        title = f"{checkpoint_title(name):<15}"
        if kind == "token":
            for line in _class_rows(title, 15, 26, res["frames"]):
                _p(line)
        elif kind == "sequence":
            first = _class_rows(f"{title}{'sequence':<13}", 28, 21, res["windows"])
            second = _class_rows(f"{'':<15}{'recording':<13}", 28, 21, res["recordings"])
            for line in first + [_rule(15)] + second:
                _p(line)
        elif kind == "regression":
            r = res["windows"]
            _p(f"{title}{r['mse']:>15.6f}{r['mae']:>13.6f}{math.sqrt(r['mse']):>13.6f}{r_squared(r):>13.6f}")
        else:
            rmse = f"{res['rmse']:.2f}{u}"
            mae = f"{res['mae']:.2f}{u}"
            _p(f"{title}{res['loss']:>15.2f}{rmse:>13}{mae:>13}")
        _p(_rule())
    _p("=" * WIDE)
