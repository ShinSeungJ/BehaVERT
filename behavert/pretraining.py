"""Self-supervised pretraining by masked keypoint modeling.

A random ``mask_ratio`` of the individual coordinates in each window is set to
0 and the encoder, through a small decoder, reconstructs them. The loss is the
mean squared error on the masked coordinates only. Afterwards the encoder is
fine-tuned with :meth:`behavert.BehaVERT.load_encoder`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from .model import BehaVERT, BehaVERTConfig

DECODER_NAME = "decoder.safetensors"


class MaskedKeypointModel(nn.Module):
    def __init__(self, config: BehaVERTConfig):
        super().__init__()
        if config.mode != "token":
            raise ValueError("pretraining uses the token (per-frame) encoder")
        self.encoder = BehaVERT(config)
        H = config.hidden_size
        self.decoder = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.LayerNorm(H), nn.Linear(H, config.input_dim))

    @property
    def config(self) -> BehaVERTConfig:
        return self.encoder.config

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(x, attention_mask=attention_mask).last_hidden_state
        return self.decoder(h)

    def save_pretrained(self, directory: str | Path) -> None:
        """Encoder as a regular BehaVERT checkpoint, decoder alongside it."""
        from safetensors.torch import save_file

        self.encoder.save_pretrained(directory)
        state = {k: v.detach().contiguous().cpu() for k, v in self.decoder.state_dict().items()}
        save_file(state, str(Path(directory) / DECODER_NAME))

    @classmethod
    def from_pretrained(cls, directory: str | Path, device=None) -> "MaskedKeypointModel":
        """Encoder and decoder written by :meth:`save_pretrained`."""
        from safetensors.torch import load_file

        encoder = BehaVERT.from_pretrained(directory, device="cpu")
        model = cls(encoder.config)
        model.encoder = encoder
        model.decoder.load_state_dict(load_file(str(Path(directory) / DECODER_NAME)))
        return model.to(device or "cpu").eval()


def mask_coordinates(x: torch.Tensor, ratio: float, frame_mask: Optional[torch.Tensor] = None):
    """Zero a random ``ratio`` of the coordinates of every window.

    ``x``: ``[B, L, F]``. Padding frames (``frame_mask == 0``) are never
    selected. Returns the masked input and the boolean positions ``[B, L, F]``.
    """
    B, L, F = x.shape
    scores = torch.rand(B, L, F, device=x.device)
    if frame_mask is not None:
        scores = scores.masked_fill(frame_mask[..., None] == 0, 2.0)
    n = int(L * F * ratio)
    flat = scores.reshape(B, -1)
    positions = torch.zeros_like(flat, dtype=torch.bool)
    if n > 0:
        idx = flat.topk(n, dim=1, largest=False).indices
        positions.scatter_(1, idx, True)
        positions &= flat < 2.0
    positions = positions.reshape(B, L, F)
    return x.masked_fill(positions, 0.0), positions


def masked_mse(pred: torch.Tensor, target: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    n = positions.sum()
    if n == 0:
        return pred.sum() * 0.0
    return (((pred.float() - target.float()) ** 2) * positions).sum() / n


def pretrain_loss_step(mask_ratio: float):
    def step(model: MaskedKeypointModel, batch: dict) -> torch.Tensor:
        x, m = batch["input"], batch["mask"]
        masked, pos = mask_coordinates(x, mask_ratio, m)
        return masked_mse(model(masked, attention_mask=m), x, pos)
    return step


@torch.no_grad()
def evaluate_pretraining(model: MaskedKeypointModel, loader, mask_ratio: float, seed: int = 0) -> dict:
    """Masked-coordinate MSE, RMSE and MAE (in input units) with a fixed masking seed."""
    model.eval()
    dev = next(model.parameters()).device
    se, ae, n = 0.0, 0.0, 0
    with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
        torch.manual_seed(seed)
        for batch in loader:
            x, m = batch["input"].to(dev), batch["mask"].to(dev)
            masked, pos = mask_coordinates(x, mask_ratio, m)
            err = (model(masked, attention_mask=m).float() - x)[pos]
            se += float((err ** 2).sum())
            ae += float(err.abs().sum())
            n += int(pos.sum())
    mse = se / max(n, 1)
    return {"loss": mse, "rmse": mse ** 0.5, "mae": ae / max(n, 1)}
