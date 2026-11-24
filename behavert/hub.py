"""Locating released checkpoints (local folders or the Hugging Face Hub).

Released models live in two Hub repos, one folder per model:

* ``ShinSeungJ/behavert``      — BehaVERT models (MIT): calms21-task1, mabe22-strain, oft-behavior
* ``ShinSeungJ/behavert-pose`` — YOLO pose models (AGPL-3.0): oft-pose, sbea-pose

and are referred to by their short name, e.g. ``BehaVERT.from_pretrained("calms21-task1")``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .model import CONFIG_NAME, WEIGHTS_NAME

MODEL_REPO = "ShinSeungJ/behavert"
POSE_REPO = "ShinSeungJ/behavert-pose"

# short name -> (Hub repo id, folder inside the repo)
PRETRAINED_MODELS: dict[str, tuple[str, str]] = {
    "calms21-task1": (MODEL_REPO, "calms21-task1"),
    "mabe22-strain": (MODEL_REPO, "mabe22-strain"),
    "oft-behavior": (MODEL_REPO, "oft-behavior"),
}

# pose-estimation models (Ultralytics YOLO) used by ``behavert decode --video``
POSE_MODELS: dict[str, tuple[str, str]] = {
    "oft-pose": (POSE_REPO, "oft-pose"),
    "sbea-pose": (POSE_REPO, "sbea-pose"),
}
POSE_WEIGHTS_NAME = "model.pt"


def _local_copy(name: str, aliases: dict) -> Path | None:
    """A local copy of a released model under ``$BEHAVERT_CHECKPOINTS``, laid out like the
    Hub (``<repo name>/<model>/``) or flat (``<model>/``)."""
    root = os.environ.get("BEHAVERT_CHECKPOINTS")
    if not root or name not in aliases:
        return None
    repo, sub = aliases[name]
    for cand in (Path(root).expanduser() / repo.split("/")[-1] / sub, Path(root).expanduser() / name):
        if cand.is_dir():
            return cand
    return None


def resolve_files(name_or_path: str, filenames: list[str], aliases: dict[str, tuple[str, str]],
                  revision: str | None = None) -> list[Path]:
    """Local paths to ``filenames`` of a checkpoint folder.

    ``name_or_path`` is a local directory, a short name from ``aliases`` or a Hub
    repo id whose root holds the files. Hub downloads are cached by ``huggingface_hub``.
    """
    local = Path(name_or_path).expanduser()
    if not local.is_dir():
        local = _local_copy(name_or_path, aliases) or local
    if local.is_dir():
        paths = [local / f for f in filenames]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"{local} is missing {', '.join(missing)}")
        return paths

    if name_or_path in aliases:
        repo_id, sub = aliases[name_or_path]
    else:
        looks_local = (name_or_path.startswith((".", "/", "~")) or name_or_path.count("/") > 1
                       or Path(name_or_path).suffix in (".pt", ".pth", ".safetensors", ".json"))
        if looks_local:
            raise FileNotFoundError(f"{name_or_path} not found (expected a checkpoint folder)")
        if "/" not in name_or_path:
            known = ", ".join(sorted(aliases))
            raise ValueError(f"{name_or_path!r} is not a local folder, a released model ({known}) or a Hub repo id")
        repo_id, sub = name_or_path, None

    from huggingface_hub import hf_hub_download

    return [Path(hf_hub_download(repo_id, f, subfolder=sub, revision=revision)) for f in filenames]


def resolve_model_files(name_or_path: str, revision: str | None = None) -> tuple[Path, Path]:
    """Return local paths to a BehaVERT checkpoint's ``(config.json, model.safetensors)``."""
    cfg, w = resolve_files(name_or_path, [CONFIG_NAME, WEIGHTS_NAME], PRETRAINED_MODELS, revision)
    return cfg, w


def resolve_pose_model(name_or_path: str, revision: str | None = None) -> tuple[Path, Path]:
    """Return local paths to a pose model's ``(config.json, model.pt)``.

    A bare ``.pt`` file is also accepted (then no config is available).
    """
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p.parent / CONFIG_NAME, p
    cfg, w = resolve_files(name_or_path, [CONFIG_NAME, POSE_WEIGHTS_NAME], POSE_MODELS, revision)
    return cfg, w
