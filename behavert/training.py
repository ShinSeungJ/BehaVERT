"""Training loop shared by fine-tuning and pretraining."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data.dataset import WindowDataset, collate
from .metrics import classification_report, regression_report
from .report import TrainingReport


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    eval_batch_size: int = 256
    grad_accum: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup: Optional[int | str] = None  # steps (e.g. 4000) or a share of all steps (e.g. "10%"); default 10%
    max_grad_norm: float = 1.0
    eval_every: int | str = 100  # validate every N optimizer steps, or "epoch" (at the end of each epoch)
    log_every: int | str = 20    # print the training loss every N steps, or "epoch" (mean over the epoch)
    seed: int = 42
    max_steps: Optional[int] = None  # stop early (debugging / smoke tests)
    # early stopping: stop once neither the validation loss nor the selection metric has
    # improved for this many epochs (default: 20% of `epochs`, never below 10%; 0 = off)
    patience: Optional[int] = None
    metric_ignore_classes: list[int] = field(default_factory=list)  # frames with these true labels are not scored
    metric_average_classes: Optional[list[int]] = None  # classes averaged into macro_f1 (default: all scored)
    select_on: str = "macro_f1"  # checkpoint selection: "macro_f1", "f1_class1" or "loss"
    amp: Optional[str] = None  # "bf16" | "fp16" | None
    num_workers: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown train options: {sorted(unknown)}")
        return cls(**d)

    def __post_init__(self) -> None:
        for name in ("eval_every", "log_every"):
            value = getattr(self, name)
            if isinstance(value, str) and value.strip().lower() == "epoch":
                setattr(self, name, "epoch")
            elif isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"train.{name} must be a positive number of steps or 'epoch', got {value!r}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _autocast(amp: Optional[str], device: torch.device):
    if not amp or device.type != "cuda":
        return torch.autocast(device_type=device.type, enabled=False)
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16 if amp == "bf16" else torch.float16)


@torch.no_grad()
def evaluate_classifier(model, loader, loss_fn, cfg: TrainConfig, device, label_names=None) -> dict:
    """Window-level loss and metrics (frame-level for token models; MSE for regression)."""
    model.eval()
    regression = model.config.problem == "regression"
    losses, preds, labels = [], [], []
    for batch in loader:
        x, m, y = batch["input"].to(device), batch["mask"].to(device), batch["target"].to(device)
        with _autocast(cfg.amp, device):
            logits = model(x, attention_mask=m).logits
        losses.append(loss_fn(logits, y).item())
        out = logits.float().reshape(y.shape) if regression else logits.argmax(-1)
        preds.append(out.cpu().numpy().reshape(-1))
        labels.append(y.cpu().numpy().reshape(-1))
    if regression:
        rep = regression_report(np.concatenate(labels), np.concatenate(preds))
    else:
        rep = classification_report(np.concatenate(labels), np.concatenate(preds), model.config.num_classes,
                                    cfg.metric_ignore_classes, label_names, cfg.metric_average_classes)
    rep["loss"] = float(np.mean(losses))
    return rep


def resolve_warmup(warmup: Optional[int | str], total_steps: int) -> int:
    """Warm-up length in optimizer steps: an int is a number of steps, a string such
    as ``"10%"`` a share of ``total_steps``; ``None`` means 10%."""
    if warmup is None:
        warmup = "10%"
    if isinstance(warmup, str):
        text = warmup.strip()
        if not text.endswith("%"):
            raise ValueError(f"train.warmup: give a number of steps or a percentage such as '10%', got {warmup!r}")
        share = float(text[:-1]) / 100
        if not 0 <= share <= 1:
            raise ValueError(f"train.warmup must be between 0% and 100%, got {warmup!r}")
        return int(round(share * total_steps))
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"train.warmup: give a number of steps or a percentage such as '10%', got {warmup!r}")
    return warmup


def resolve_patience(patience: Optional[int], epochs: int) -> int:
    """Early-stopping patience in epochs: 20% of ``epochs`` by default and never below
    10% (at least 1); 0 turns early stopping off."""
    if patience == 0:
        return 0
    floor = max(1, epochs // 10)
    return max(floor, max(1, epochs * 2 // 10) if patience is None else patience)


CHECKPOINT_NAMES = {"macro_f1": "best_f1", "f1_class1": "best_f1_class1", "loss": "best_loss"}


def checkpoint_name(select_on: str) -> str:
    """Folder of the checkpoint kept for ``select_on`` (``macro_f1`` -> ``best_f1``)."""
    return CHECKPOINT_NAMES.get(select_on, f"best_{select_on}")


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    out_dir: str | Path,
    loss_step: Callable[[torch.nn.Module, dict], torch.Tensor],
    evaluate: Optional[Callable[[torch.nn.Module], dict]] = None,
    save: Optional[Callable[[torch.nn.Module, Path], None]] = None,
    select_on: Optional[str] = "macro_f1",
    report: Optional[TrainingReport] = None,
) -> dict:
    """Generic loop: AdamW, linear warm-up/decay, gradient clipping and accumulation.

    Saves ``best_loss/``, the checkpoint for ``select_on`` (``best_f1/`` for macro F1,
    ``best_f1_class1/`` for the class-1 F1) and ``last/`` under ``out_dir``
    (via ``save(model, dir)``), plus ``history.jsonl``. With validation, training
    stops early once neither the validation loss nor ``select_on`` has improved for
    ``cfg.patience`` epochs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = report or TrainingReport()
    device = next(model.parameters()).device
    save = save or (lambda mdl, d: mdl.save_pretrained(d))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # one optimizer step per `grad_accum` batches, counted across epochs
    total = max(1, math.ceil(len(train_loader) * cfg.epochs / cfg.grad_accum))
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    warmup_steps = resolve_warmup(cfg.warmup, total)
    sched = get_linear_schedule_with_warmup(opt, warmup_steps, total)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16" and device.type == "cuda"))

    best = {"loss": math.inf}
    if select_on and select_on != "loss":
        best[select_on] = -math.inf
    patience = resolve_patience(cfg.patience, cfg.epochs)
    epoch, best_epoch = 0, 0  # current epoch; last epoch in which a tracked metric improved
    step, last_eval = 0, -1
    history = open(out_dir / "history.jsonl", "a")
    running: list[float] = []
    done = False

    def run_eval():
        nonlocal best_epoch, last_eval
        if evaluate is None or last_eval == step:
            return
        last_eval = step
        rep = evaluate(model)
        model.train()
        rec = {"step": step, **{k: v for k, v in rep.items() if k != "per_class"}, "per_class": rep.get("per_class")}
        history.write(json.dumps(rec) + "\n")
        history.flush()
        report.valid(epoch, step, rep)
        if rep["loss"] < best["loss"]:
            best["loss"] = rep["loss"]
            best_epoch = epoch
            save(model, out_dir / "best_loss")
            report.saved("best_loss", out_dir / "best_loss")
        if select_on and select_on != "loss" and select_on in rep and rep[select_on] > best[select_on]:
            best[select_on] = rep[select_on]
            best_epoch = epoch
            name = checkpoint_name(select_on)
            save(model, out_dir / name)
            report.saved(name, out_dir / name)

    def optimizer_step():
        nonlocal step, pending
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
        scaler.step(opt)
        scaler.update()
        sched.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        pending = 0

    model.train()
    pending = 0  # batches accumulated since the last optimizer step
    stopped_early = False
    for epoch in range(1, cfg.epochs + 1):
        for batch in train_loader:
            with _autocast(cfg.amp, device):
                loss = loss_step(model, {k: v.to(device) for k, v in batch.items()})
            scaler.scale(loss / cfg.grad_accum).backward()
            running.append(loss.item())
            pending += 1
            if pending < cfg.grad_accum:
                continue
            optimizer_step()
            if cfg.log_every != "epoch" and step % cfg.log_every == 0:
                report.train(epoch, step, float(np.mean(running)), sched.get_last_lr()[0])
                running = []
            if cfg.eval_every != "epoch" and step % cfg.eval_every == 0:
                run_eval()
            if cfg.max_steps and step >= cfg.max_steps:
                done = True
                break
        if cfg.log_every == "epoch" and running:
            report.train(epoch, step, float(np.mean(running)), sched.get_last_lr()[0])
            running = []
        if cfg.eval_every == "epoch":
            run_eval()
        if done:
            break
        if evaluate is not None and last_eval >= 0 and patience > 0 and epoch - best_epoch >= patience:
            stopped_early = True
            break
    if pending:  # gradients of the last, incomplete accumulation
        optimizer_step()
    run_eval()
    report.close()
    if stopped_early:
        report.early_stop(epoch, cfg.epochs)
    save(model, out_dir / "last")
    report.saved("last", out_dir / "last")
    history.close()
    summary = {"steps": step, "total_steps": total, "warmup_steps": warmup_steps, "epochs": epoch,
               "stopped_early": stopped_early, "best": best,
               "patience": patience, "config": asdict(cfg)}
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def classifier_loss_step(loss_fn):
    def step(model, batch):
        logits = model(batch["input"], attention_mask=batch["mask"]).logits
        return loss_fn(logits, batch["target"])
    return step


def make_loader(ds: WindowDataset, batch_size: int, sampler=None, shuffle=False, num_workers=0) -> DataLoader:
    if sampler is not None:
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collate, num_workers=num_workers)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collate, num_workers=num_workers)
