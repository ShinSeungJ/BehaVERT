"""BehaVERT: a BERT encoder over keypoint frames.

Each video frame is one token. A frame's keypoints are flattened into a vector
and linearly projected to the BERT hidden size; positional information is
added, and a 12-layer BERT encoder contextualises the sequence.

Two prediction heads (paper Section 3.2):

* ``mode="token"``    — one label per frame (behavior classification).
* ``mode="sequence"`` — one label per window via a learnable ``[CLS]`` token
  (state classification, e.g. strain).

Example
-------
>>> from behavert import BehaVERT, BehaVERTConfig
>>> cfg = BehaVERTConfig(input_dim=28, num_classes=4, mode="token", max_seq_length=128)
>>> model = BehaVERT(cfg)
>>> out = model(torch.randn(8, 128, 28))
>>> out.logits.shape
torch.Size([8, 128, 4])
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "model.safetensors"


@dataclass
class BehaVERTConfig:
    """Architecture plus the metadata needed to use a trained model correctly.

    Only the architecture fields affect the network. ``label_names``,
    ``keypoints`` and ``preprocessing`` travel with released checkpoints so that
    input can be prepared the same way the model was trained.
    """

    input_dim: int
    num_classes: int
    mode: str = "token"  # "token" | "sequence"
    max_seq_length: int = 64
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    # "sinusoidal" (fixed) or "learned"; added on top of BERT's own position embeddings.
    positional_encoding: str = "sinusoidal"
    embedding_dropout: float = 0.0
    classifier_dropout: float = 0.1
    # 0: linear classifier. >0: Linear(H, head_hidden) -> ReLU -> Dropout -> Linear (used for regression).
    head_hidden: int = 0
    # "classification" (softmax over num_classes) or "regression" (num_classes real outputs)
    problem: str = "classification"
    max_position_embeddings: int = 512
    # --- metadata (not used by the network) ---
    label_names: list[str] = field(default_factory=list)
    label_colors: list[str] = field(default_factory=list)  # optional display color per class
    keypoints: dict[str, Any] = field(default_factory=dict)
    preprocessing: dict[str, Any] = field(default_factory=dict)
    display: dict[str, Any] = field(default_factory=dict)  # how `behavert decode` draws this model's videos
    description: str = ""

    def __post_init__(self) -> None:
        if self.mode not in ("token", "sequence"):
            raise ValueError(f"mode must be 'token' or 'sequence', got {self.mode!r}")
        if self.positional_encoding not in ("sinusoidal", "learned"):
            raise ValueError(
                f"positional_encoding must be 'sinusoidal' or 'learned', got {self.positional_encoding!r}"
            )
        n_tokens = self.max_seq_length + (1 if self.mode == "sequence" else 0)
        if n_tokens > self.max_position_embeddings:
            raise ValueError(
                f"max_seq_length={self.max_seq_length} needs {n_tokens} positions but "
                f"max_position_embeddings={self.max_position_embeddings}; raise it."
            )
        if self.problem not in ("classification", "regression"):
            raise ValueError(f"problem must be 'classification' or 'regression', got {self.problem!r}")
        if self.problem == "regression" and self.mode != "sequence":
            raise ValueError("regression is supported with mode='sequence' (one value per window)")
        if self.label_colors and len(self.label_colors) != self.num_classes:
            raise ValueError("label_colors must have num_classes entries")
        if self.label_names and len(self.label_names) != self.num_classes:
            raise ValueError("label_names must have num_classes entries")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BehaVERTConfig":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "BehaVERTConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


@dataclass
class BehaVERTOutput:
    logits: torch.Tensor
    """``[B, L, C]`` in token mode, ``[B, C]`` in sequence mode."""
    last_hidden_state: torch.Tensor
    """``[B, L, H]`` (token) or ``[B, 1+L, H]`` (sequence, index 0 is ``[CLS]``)."""
    embedding: torch.Tensor
    """Representation used by the classifier: per-frame ``[B, L, H]`` (token) or pooled ``[CLS]`` ``[B, H]`` (sequence)."""
    attentions: Optional[tuple[torch.Tensor, ...]] = None
    """One ``[B, heads, T, T]`` tensor per layer when ``output_attentions=True``."""


def sinusoidal_encoding(length: int, dim: int, device=None) -> torch.Tensor:
    """Standard Transformer sin/cos positional encoding, shape ``[length, dim]``."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(0, length, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, dim, 2, device=device).float() * -(math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class BehaVERT(nn.Module):
    def __init__(self, config: BehaVERTConfig):
        super().__init__()
        self.config = config
        H = config.hidden_size
        is_seq = config.mode == "sequence"

        self.embedding = nn.Linear(config.input_dim, H)
        if is_seq:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, H))
        if config.positional_encoding == "learned":
            n = config.max_seq_length + (1 if is_seq else 0)
            self.positional_encodings = nn.Parameter(torch.randn(1, n, H))
        self.dropout = nn.Dropout(config.embedding_dropout)

        bert_cfg = BertConfig(
            vocab_size=1,  # inputs are keypoint embeddings, never word ids
            hidden_size=H,
            num_hidden_layers=config.num_layers,
            num_attention_heads=config.num_heads,
            intermediate_size=config.intermediate_size,
            max_position_embeddings=config.max_position_embeddings,
        )
        self.bert = BertModel(bert_cfg, add_pooling_layer=is_seq)
        if config.head_hidden:
            self.classifier_dropout = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(H, config.head_hidden), nn.ReLU(), nn.Dropout(config.classifier_dropout),
                nn.Linear(config.head_hidden, config.num_classes),
            )
        else:
            self.classifier_dropout = nn.Dropout(config.classifier_dropout)
            self.classifier = nn.Linear(H, config.num_classes)

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> BehaVERTOutput:
        """
        Args:
            x: ``[B, L, input_dim]`` flattened keypoints per frame.
            attention_mask: ``[B, L]`` with 1 for real frames and 0 for padding.
                In sequence mode a ``[B, 1+L]`` mask that already covers
                ``[CLS]`` is also accepted.
            output_attentions: also return per-layer attention maps.
        """
        B, L, _ = x.shape
        is_seq = self.config.mode == "sequence"
        h = self.embedding(x)

        if is_seq:
            h = torch.cat([self.cls_token.expand(B, -1, -1), h], dim=1)
            if attention_mask is not None and attention_mask.shape[1] == L:
                ones = attention_mask.new_ones(B, 1)
                attention_mask = torch.cat([ones, attention_mask], dim=1)

        T = h.shape[1]
        if self.config.positional_encoding == "learned":
            if T > self.positional_encodings.shape[1]:
                raise ValueError(f"sequence length {L} exceeds max_seq_length={self.config.max_seq_length}")
            h = h + self.positional_encodings[:, :T, :]
        else:
            h = h + sinusoidal_encoding(T, h.shape[-1], device=h.device)
        h = self.dropout(h)

        bert_out = self._run_bert(h, attention_mask, output_attentions)

        if is_seq:
            embedding = bert_out.pooler_output
        else:
            embedding = bert_out.last_hidden_state
        logits = self.classifier(self.classifier_dropout(embedding))
        return BehaVERTOutput(
            logits=logits,
            last_hidden_state=bert_out.last_hidden_state,
            embedding=embedding,
            attentions=bert_out.attentions if output_attentions else None,
        )

    def _run_bert(self, h, attention_mask, output_attentions):
        # Fused attention kernels (sdpa/flash) cannot return attention maps, and
        # recent transformers versions then return an empty tuple. Switch to the
        # eager implementation only for calls that need the maps.
        cfg = self.bert.config
        previous = getattr(cfg, "_attn_implementation", None)
        if output_attentions and previous not in (None, "eager"):
            cfg._attn_implementation = "eager"
        try:
            return self.bert(
                inputs_embeds=h,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
            )
        finally:
            if output_attentions and previous not in (None, "eager"):
                cfg._attn_implementation = previous

    # --------------------------------------------------------- save / load
    def save_pretrained(self, directory: str | Path) -> None:
        """Write ``config.json`` and ``model.safetensors`` to ``directory``."""
        from safetensors.torch import save_file

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.config.save(directory / CONFIG_NAME)
        state = {k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()}
        save_file(state, str(directory / WEIGHTS_NAME), metadata={"format": "pt"})

    def load_encoder(self, name_or_path: str | Path, revision: str | None = None, verbose: bool = True) -> list[str]:
        """Initialise from another checkpoint's encoder, e.g. after pretraining.

        Copies every weight whose name and shape match, except the classifier.
        Returns the names of parameters left at their fresh initialisation.
        """
        from safetensors.torch import load_file

        from .hub import resolve_model_files

        _, weights_file = resolve_model_files(str(name_or_path), revision=revision)
        source = load_file(str(weights_file))
        own = self.state_dict()
        take = {k: v for k, v in source.items()
                if not k.startswith("classifier.") and k in own and own[k].shape == v.shape}
        self.load_state_dict(take, strict=False)
        fresh = sorted(k for k in own if k not in take)
        if verbose:
            print(f"loaded {len(take)} tensors from {name_or_path}; newly initialised: {fresh}")
        return fresh

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str | Path,
        device: str | torch.device | None = None,
        revision: str | None = None,
    ) -> "BehaVERT":
        """Load from a local directory or a Hugging Face Hub repo id.

        ``name_or_path`` can also be a short alias from
        :data:`behavert.hub.PRETRAINED_MODELS` (e.g. ``"calms21-task1"``).
        """
        from safetensors.torch import load_file

        from .hub import resolve_model_files

        config_file, weights_file = resolve_model_files(str(name_or_path), revision=revision)
        model = cls(BehaVERTConfig.load(config_file))
        model.load_state_dict(load_file(str(weights_file)))
        model.eval()
        if device is not None:
            model.to(device)
        return model
