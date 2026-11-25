"""Training loop shared by fine-tuning and pretraining."""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from .data.dataset import WindowDataset, collate
from .metrics import classification_report, format_report, regression_report


@dataclass
class TrainConfig:
    epochs: int = 10
    batch_size: int = 64
    eval_batch_size: int = 256
    grad_accum: int = 1
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_grad_norm: float = 1.0
    eval_every: int = 100  # optimizer steps between validations
    log_every: int = 20
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


def resolve_patience(patience: Optional[int], epochs: int) -> int:
    """Early-stopping patience in epochs: 20% of ``epochs`` by default and never below
    10% (at least 1); 0 turns early stopping off."""
    if patience == 0:
        return 0
    floor = max(1, epochs // 10)
    return max(floor, max(1, epochs * 2 // 10) if patience is None else patience)


def fit(
    model: torch.nn.Module,
    train_loader: DataLoader,
    cfg: TrainConfig,
    out_dir: str | Path,
    loss_step: Callable[[torch.nn.Module, dict], torch.Tensor],
    evaluate: Optional[Callable[[torch.nn.Module], dict]] = None,
    save: Optional[Callable[[torch.nn.Module, Path], None]] = None,
    select_on: Optional[str] = "macro_f1",
) -> dict:
    """Generic loop: AdamW, linear warm-up/decay, gradient clipping and accumulation.

    Saves ``best_<select_on>/``, ``best_loss/`` and ``last/`` under ``out_dir``
    (via ``save(model, dir)``), plus ``history.jsonl``. With validation, training
    stops early once neither the validation loss nor ``select_on`` has improved for
    ``cfg.patience`` epochs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    save = save or (lambda mdl, d: mdl.save_pretrained(d))

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    # one optimizer step per `grad_accum` batches, counted across epochs
    total = max(1, math.ceil(len(train_loader) * cfg.epochs / cfg.grad_accum))
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    sched = get_linear_schedule_with_warmup(opt, cfg.warmup_steps, total)
    scaler = torch.amp.GradScaler("cuda", enabled=(cfg.amp == "fp16" and device.type == "cuda"))

    best = {"loss": math.inf}
    if select_on and select_on != "loss":
        best[select_on] = -math.inf
    patience = resolve_patience(cfg.patience, cfg.epochs)
    no_gain = (f"neither validation loss nor {select_on} improved" if select_on and select_on != "loss"
               else "validation loss did not improve")
    epoch, best_epoch = 0, 0  # current epoch; last epoch in which a tracked metric improved
    step, t0, last_eval = 0, time.time(), -1
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
        msg = f"[eval] step {step}  loss {rep['loss']:.5f}"
        if select_on and select_on != "loss" and select_on in rep:
            msg += f"  {select_on} {rep[select_on]:.4f}"
        print(msg, flush=True)
        if rep["loss"] < best["loss"]:
            best["loss"] = rep["loss"]
            best_epoch = epoch
            save(model, out_dir / "best_loss")
        if select_on and select_on != "loss" and select_on in rep and rep[select_on] > best[select_on]:
            best[select_on] = rep[select_on]
            best_epoch = epoch
            save(model, out_dir / f"best_{select_on}")
            if "per_class" in rep and rep["per_class"]:
                print(format_report(rep, f"  new best {select_on}"), flush=True)

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
            if step % cfg.log_every == 0:
                print(f"epoch {epoch}  step {step}/{total}  loss {np.mean(running):.5f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
                running = []
            if step % cfg.eval_every == 0:
                run_eval()
            if cfg.max_steps and step >= cfg.max_steps:
                done = True
                break
        if done:
            break
        if evaluate is not None and last_eval >= 0 and patience > 0 and epoch - best_epoch >= patience:
            print(f"early stop at epoch {epoch}/{cfg.epochs}: {no_gain} for {epoch - best_epoch} epochs "
                  f"(last improvement in epoch {best_epoch})", flush=True)
            stopped_early = True
            break
    if pending:  # gradients of the last, incomplete accumulation
        optimizer_step()
    run_eval()
    save(model, out_dir / "last")
    history.close()
    summary = {"steps": step, "epochs": epoch, "stopped_early": stopped_early, "best": best,
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
