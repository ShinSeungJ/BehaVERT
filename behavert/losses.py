"""Losses for imbalanced behavior data (paper Section 3.4).

All losses take logits ``[..., C]`` and integer targets ``[...]``; targets of
``-1`` (padding) are ignored. The weighted losses (weighted_ce, focal, hybrid)
weight class ``c`` by the normalized inverse class proportion of the training data,
``w_c = (1 / p_c) / sum_k (1 / p_k)``. In a config, ``class_weights`` gives the
weights directly instead, and ``class_weight_multipliers`` scales either per class
(e.g. ``[2, 1, 1]`` doubles the share of class 0); weights are normalized to sum to 1.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_proportion_weights(proportions: Sequence[float]) -> torch.Tensor:
    p = torch.as_tensor(proportions, dtype=torch.float32).clamp(min=1e-8)
    w = 1.0 / p
    return w / w.sum()


def resolve_class_weights(
    class_proportions: Optional[Sequence[float]] = None,
    class_weights: Optional[Sequence[float]] = None,
    class_weight_multipliers: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Per-class weights summing to 1: ``class_weights`` if given, else the inverse
    ``class_proportions``, then multiplied by ``class_weight_multipliers``."""
    if class_weights is not None:
        w = torch.as_tensor(class_weights, dtype=torch.float32)
    elif class_proportions is not None:
        w = inverse_proportion_weights(class_proportions)
    else:
        raise ValueError("class weights need class_weights or the class proportions of the training data")
    if class_weight_multipliers is not None:
        m = torch.as_tensor(class_weight_multipliers, dtype=torch.float32)
        if m.shape != w.shape:
            raise ValueError(f"class_weight_multipliers has {m.numel()} values for {w.numel()} classes")
        w = w * m
    if (w < 0).any() or w.sum() <= 0:
        raise ValueError("class weights must be non-negative and not all zero")
    return w / w.sum()


class _Masked(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(reduction)
        self.reduction = reduction

    def _reduce(self, loss: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        loss = loss * valid
        if self.reduction == "mean":
            return loss.sum() / valid.sum().clamp(min=1)
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCrossEntropyLoss(_Masked):
    """Cross-entropy with inverse-proportion class weights (``Lwce``)."""

    def __init__(self, class_proportions: Optional[Sequence[float]] = None, reduction: str = "mean",
                 class_weights: Optional[Sequence[float]] = None,
                 class_weight_multipliers: Optional[Sequence[float]] = None):
        super().__init__(reduction)
        self.register_buffer("weights", resolve_class_weights(class_proportions, class_weights,
                                                              class_weight_multipliers))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets >= 0).float()
        t = targets.clamp(min=0).long()
        ce = -torch.gather(F.log_softmax(logits.float(), -1), -1, t.unsqueeze(-1)).squeeze(-1)
        return self._reduce(ce * self.weights[t], valid)


class FocalLoss(_Masked):
    """One-vs-rest focal loss, ``alpha_c (1 - p_t)^gamma BCE``, averaged over classes."""

    def __init__(self, class_proportions: Optional[Sequence[float]] = None, gamma: float = 2.0,
                 reduction: str = "mean", class_weights: Optional[Sequence[float]] = None,
                 class_weight_multipliers: Optional[Sequence[float]] = None):
        super().__init__(reduction)
        self.gamma = gamma
        self.register_buffer("weights", resolve_class_weights(class_proportions, class_weights,
                                                              class_weight_multipliers))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        valid = (targets >= 0).float()
        t = targets.clamp(min=0).long()
        onehot = F.one_hot(t, logits.shape[-1]).float()
        bce = F.binary_cross_entropy_with_logits(logits.float(), onehot, reduction="none")
        focal = self.weights * (1 - torch.exp(-bce)) ** self.gamma * bce
        return self._reduce(focal.mean(-1), valid)


class HybridLoss(_Masked):
    """``(1 - alpha) * weighted CE + alpha * focal`` (paper Eq. 10).

    The class weights apply to the cross-entropy term; the focal term is unweighted.
    """

    def __init__(
        self,
        class_proportions: Optional[Sequence[float]] = None,
        gamma: float = 2.0,
        alpha: float = 0.5,
        reduction: str = "mean",
        class_weights: Optional[Sequence[float]] = None,
        class_weight_multipliers: Optional[Sequence[float]] = None,
    ):
        super().__init__(reduction)
        self.gamma = gamma
        self.alpha = alpha
        self.register_buffer("weights", resolve_class_weights(class_proportions, class_weights,
                                                              class_weight_multipliers))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        valid = (targets >= 0).float()
        t = targets.clamp(min=0).long()
        ce = -torch.gather(F.log_softmax(logits, -1), -1, t.unsqueeze(-1)).squeeze(-1)
        wce = ce * self.weights[t]
        onehot = F.one_hot(t, logits.shape[-1]).float()
        bce = F.binary_cross_entropy_with_logits(logits, onehot, reduction="none")
        focal = ((1 - torch.exp(-bce)) ** self.gamma * bce).mean(-1)
        return self._reduce((1 - self.alpha) * wce + self.alpha * focal, valid)


class CrossEntropyLoss(nn.CrossEntropyLoss):
    def __init__(self, class_proportions=None, reduction: str = "mean"):
        super().__init__(ignore_index=-1, reduction=reduction)

    def forward(self, logits, targets):
        return super().forward(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1).long())


class SoftF1Loss(nn.Module):
    """``1 - macro F1`` computed from softmax probabilities (paper Eq. 14).

    Frame targets of ``-1`` are ignored. Optional ``class_weights`` (and/or
    ``class_weight_multipliers``) turn the macro average into a weighted one.
    Class proportions are not used.
    """

    def __init__(self, class_proportions=None, class_weights: Sequence[float] | None = None, eps: float = 1e-7,
                 class_weight_multipliers: Sequence[float] | None = None):
        super().__init__()
        self.eps = eps
        w = class_weights
        if class_weight_multipliers is not None:
            w = torch.as_tensor(class_weights if class_weights is not None else [1.0] * len(class_weight_multipliers),
                                dtype=torch.float32) * torch.as_tensor(class_weight_multipliers, dtype=torch.float32)
        self.class_weights = None if w is None else torch.as_tensor(w, dtype=torch.float32)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.shape[-1]
        logits = logits.float().reshape(-1, C)
        targets = targets.reshape(-1)
        keep = targets >= 0
        logits, targets = logits[keep], targets[keep].long()
        if targets.numel() == 0:
            return logits.sum() * 0.0
        onehot = F.one_hot(targets, C).float()
        p = logits.softmax(-1)
        tp = (onehot * p).sum(0)
        fp = ((1 - onehot) * p).sum(0)
        fn = (onehot * (1 - p)).sum(0)
        precision = tp / (tp + fp + self.eps)
        recall = tp / (tp + fn + self.eps)
        f1 = (2 * precision * recall / (precision + recall + self.eps)).clamp(self.eps, 1 - self.eps)
        if self.class_weights is not None:
            w = self.class_weights.to(f1.device)
            return 1 - (f1 * w / w.sum()).sum()
        return 1 - f1.mean()


class MSELoss(nn.Module):
    """Mean squared error for regression; outputs ``[B, 1]`` against targets ``[B]``."""

    def __init__(self, class_proportions=None):
        super().__init__()

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(outputs.float().reshape(targets.shape), targets.float())


WEIGHTED_LOSSES = ("weighted_ce", "focal", "hybrid")  # weighted by class (see the module docstring)

LOSSES = {
    "ce": CrossEntropyLoss,
    "weighted_ce": WeightedCrossEntropyLoss,
    "focal": FocalLoss,
    "hybrid": HybridLoss,
    "f1": SoftF1Loss,
    "mse": MSELoss,
}


def build_loss(name: str, class_proportions: Sequence[float], **kwargs) -> nn.Module:
    try:
        cls = LOSSES[name]
    except KeyError:
        raise KeyError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from None
    return cls(class_proportions, **kwargs)
