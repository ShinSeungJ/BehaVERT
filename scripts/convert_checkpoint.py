"""Convert a training checkpoint (``checkpoint_*.pth.tar``) to a release folder.

The output folder contains ``config.json`` + ``model.safetensors`` and loads
with ``BehaVERT.from_pretrained(folder)``. Optimizer/scheduler state is
dropped, and BERT's unused 30,522-word vocabulary table is removed (inputs are
keypoint embeddings, never word ids), so the result is ~1/3 the size.

Usage
-----
    python scripts/convert_checkpoint.py OLD.pth.tar OUT_DIR --preset calms21-task1
    python scripts/convert_checkpoint.py OLD.pth.tar OUT_DIR --mode token

Architecture (input size, classes, sequence length) is read from the weights;
``--preset`` additionally fills in label names, keypoint layout and
preprocessing for the released models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from behavert import BehaVERT, BehaVERTConfig  # noqa: E402

# ``keypoints`` describes the model's input so ``behavert decode`` can check and
# draw it: per-animal keypoint names, skeleton edges (indices into names),
# coordinate units ("pixels" or "normalized" to [0, 1] by frame width / height)
# and, for pixels, the frame size the coordinates refer to.
DISPLAY = {
    "calms21-task1": {
        "title": "CalMS21",
        "skeleton_title": "Skeleton",
        "embedding_title": "Sequence embedding (t-SNE)",
        "attention_label": "attention value",
        "confidence_label": "confidence",
        "truth_title": "GT behav",
        "prediction_title": "Pred",
        "label_overlay": "top",
        "label_legend": True,
        "animal_colors": [
            "#FF5733",
            "#2E86C1"
        ],
        "embedding": {
            "perplexity_min": 2,
            "max_iter": 400
        }
    },
    "mabe22-strain": {
        "title": "MaBe22 strain",
        "title_shows_truth": True,
        "skeleton_title": "3-mouse skeleton",
        "embedding_title": "CLS embedding (t-SNE)",
        "attention_label": "attention value",
        "confidence_label": "confidence",
        "truth_title": "GT strain",
        "prediction_title": "Pred strain",
        "label_overlay": "bottom",
        "label_legend": False,
        "animal_colors": [
            "#FF5733",
            "#27AE60",
            "#2E86C1"
        ],
        "probability_colormap": [
            "#6FA8DC",
            "#F2E5E2",
            "#E5847A"
        ],
        "embedding": {
            "reference_recordings": 200,
            "perplexity_min": 5,
            "max_iter": 600,
            "reference_alpha": 0.75,
            "alpha_min": 0.7
        },
        "extra_strip": {
            "title": "Behavior",
            "none": [
                "none",
                "#ECECEC"
            ],
            "legend": True,
            "labels": [
                [
                    "oral-oral",
                    "#8B0000",
                    "mabe22/task/behavior/contact/oral_oral_contact/mabe22_oral_oral_contact.pt"
                ],
                [
                    "oral-genital",
                    "#D62728",
                    "mabe22/task/behavior/contact/oral_genital_contact/mabe22_oral_genital_contact.pt"
                ],
                [
                    "oral-ear",
                    "#E377C2",
                    "mabe22/task/behavior/contact/oral_ear_contact/mabe22_oral_ear_contact.pt"
                ],
                [
                    "chase",
                    "#FF7F0E",
                    "mabe22/task/behavior/movement/chase/mabe22_chase.pt"
                ],
                [
                    "approach",
                    "#FFB347",
                    "mabe22/task/behavior/movement/approach/mabe22_approach.pt"
                ],
                [
                    "huddles",
                    "#2CA02C",
                    "mabe22/task/behavior/contact/huddles/mabe22_huddles.pt"
                ],
                [
                    "watching",
                    "#17BECF",
                    "mabe22/task/behavior/watching/mabe22_watching.pt"
                ],
                [
                    "close",
                    "#9467BD",
                    "mabe22/task/behavior/contact/close/mabe22_close.pt"
                ]
            ]
        }
    },
    "oft-behavior": {
        "title": "OFT",
        "skeleton_title": "Skeleton",
        "embedding_title": "Sequence embedding (t-SNE)",
        "attention_label": "attention value",
        "confidence_label": "confidence",
        "truth_title": "GT behav",
        "prediction_title": "Pred",
        "label_overlay": "top",
        "label_legend": True,
        "animal_colors": [
            "#FF5733"
        ],
        "embedding": {
            "perplexity_min": 2,
            "max_iter": 400
        }
    }
}

PRESETS = {
    "calms21-task1": dict(
        mode="token",
        positional_encoding="sinusoidal",
        embedding_dropout=0.0,
        label_names=["attack", "investigation", "mount", "other"],
        label_colors=["#E74C3C", "#F1C40F", "#3498DB", "#BDC3C7"],
        keypoints={
            "dataset": "CalMS21 (Sun et al., 2021)",
            "num_animals": 2,
            "animal_names": ["resident", "intruder"],
            "names": ["nose", "left_ear", "right_ear", "neck", "left_hip", "right_hip", "tail_base"],
            "edges": [[0, 1], [0, 2], [1, 3], [2, 3], [3, 4], [3, 5], [4, 6], [5, 6]],
            "dims": 2,
            "units": "pixels",
            "frame_size": [1024, 570],
            "fps": 30,
        },
        preprocessing={
            "coordinates": "raw MARS pixel coordinates (1024x570 frame), no normalization",
            "window": 128,
            "stride": 64,
            "ignored_label_for_metrics": "other",
        },
        description=(
            "BehaVERT token model for CalMS21 Task 1 (attack / investigation / mount / other), "
            "masked-keypoint-modeling pretraining then supervised fine-tuning."
        ),
    ),
    "mabe22-strain": dict(
        mode="sequence",
        positional_encoding="learned",
        embedding_dropout=0.1,
        label_names=["WT", "TG"],
        label_colors=["#6FA8DC", "#E5847A"],
        keypoints={
            "dataset": "MABe22 mouse triplets (Sun et al., 2023)",
            "num_animals": 3,
            "names": ["nose", "left_ear", "right_ear", "neck", "left_forepaw", "right_forepaw",
                      "center_back", "left_hindpaw", "right_hindpaw", "tail_base", "tail_middle", "tail_tip"],
            "edges": [[0, 1], [0, 2], [1, 3], [2, 3], [3, 6], [3, 4], [3, 5], [6, 7], [6, 8],
                      [6, 9], [9, 10], [10, 11]],
            "dims": 2,
            "units": "pixels",
            "frame_size": [800, 800],
            "fps": 30,
            # official MABe22 overlay rule for the released (resized) videos
            "video_mapping": {"swap_xy": True, "scale": 0.44, "reference_width": 224},
        },
        preprocessing={
            "coordinates": "raw pixel coordinates as distributed, missing keypoints = 0, no normalization",
            "window": 64,
            "stride": 32,
            "video_level_prediction": "average of (p1 - p0) over windows, TG if >= 0",
        },
        description=(
            "BehaVERT [CLS] sequence model for MABe22 binary strain classification."
        ),
    ),
    "oft-behavior": dict(
        mode="token",
        positional_encoding="sinusoidal",
        embedding_dropout=0.0,
        label_names=["supported_rearing", "unsupported_rearing", "grooming", "freezing", "jumping",
                     "backwards_walking", "others"],
        keypoints={
            "dataset": "open-field test (OFT), single mouse, top view",
            "num_animals": 1,
            "names": ["nose", "left_ear", "right_ear", "left_forelimb", "right_forelimb", "tail_base",
                      "left_hindlimb", "right_hindlimb", "tail_middle", "tail_tip", "body_center"],
            "edges": [[0, 1], [0, 2], [1, 10], [2, 10], [10, 3], [10, 4], [10, 5], [5, 6], [5, 7],
                      [5, 8], [8, 9]],
            "dims": 2,
            "units": "normalized",
            "fps": 30,
            # oft-pose predicts 12 keypoints; the model uses these 11, in this order
            # (keypoint 3, thoracic_center, is not used).
            "pose_model": "oft-pose",
            "pose_indices": [0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        preprocessing={
            "coordinates": "x / frame width, y / frame height; frames without a detection are "
                           "linearly interpolated",
            "window": 64,
            "stride": 32,
        },
        description="BehaVERT token model for seven open-field behaviors of a single mouse.",
    ),
}


# overlay.mp4 style of oft-behavior (per-edge skeleton colors and label colors, BGR)
DISPLAY["oft-behavior"]["overlay"] = {"edge_colors_bgr": [[38, 32, 193], [38, 77, 229], [35, 142, 247], [21, 192, 254], [110, 37, 109], [160, 37, 209], [68, 174, 97], [176, 141, 13], [97, 97, 255], [176, 97, 13], [120, 120, 120]], "label_colors_bgr": [[0, 0, 255], [0, 165, 255], [0, 255, 255], [0, 255, 0], [255, 0, 0], [255, 0, 255], [200, 200, 200]], "line_thickness": 2, "point_radius": 3, "point_color_bgr": [0, 0, 0], "font_scale": 1.0, "show_confidence": False}

for _name, _display in DISPLAY.items():
    PRESETS[_name]["display"] = _display


def remap_state_dict(sd: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        if k.startswith("bert.bert."):  # old token model wrapped BertForTokenClassification
            k = "bert." + k[len("bert.bert."):]
        elif k.startswith("bert.classifier."):
            k = k[len("bert."):]
        if k.endswith("embeddings.word_embeddings.weight"):
            v = v[:1].clone()  # vocabulary is unused; keep a single row
        out[k] = v
    return out


def infer_config(sd: dict[str, torch.Tensor], mode: str, saved_args, overrides: dict) -> BehaVERTConfig:
    hidden, input_dim = sd["embedding.weight"].shape
    num_classes = sd["classifier.weight"].shape[0]
    num_layers = 1 + max(int(k.split(".")[3]) for k in sd if k.startswith("bert.encoder.layer."))
    max_pos = sd["bert.embeddings.position_embeddings.weight"].shape[0]
    intermediate = sd["bert.encoder.layer.0.intermediate.dense.weight"].shape[0]
    if "positional_encodings" in sd:
        pe = "learned"
        seq_len = sd["positional_encodings"].shape[1] - (1 if mode == "sequence" else 0)
    else:
        pe = "sinusoidal"
        seq_len = getattr(saved_args, "max_seq_length", None) or overrides.get("max_seq_length")
        if seq_len is None:
            raise SystemExit("cannot infer max_seq_length from a sinusoidal model; pass --max-seq-length")
    fields = dict(
        input_dim=int(input_dim),
        num_classes=int(num_classes),
        mode=mode,
        max_seq_length=int(seq_len),
        hidden_size=int(hidden),
        num_layers=num_layers,
        num_heads=overrides.pop("num_heads", 12),
        intermediate_size=int(intermediate),
        positional_encoding=pe,
        embedding_dropout=0.1 if mode == "sequence" else 0.0,
        max_position_embeddings=int(max_pos),
    )
    fields.update(overrides)
    return BehaVERTConfig(**fields)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--preset", choices=sorted(PRESETS))
    p.add_argument("--mode", choices=["token", "sequence"])
    p.add_argument("--max-seq-length", type=int)
    p.add_argument("--fp16", action="store_true", help="store weights in float16 (half the size)")
    a = p.parse_args()

    overrides = dict(PRESETS[a.preset]) if a.preset else {}
    mode = a.mode or overrides.get("mode")
    if mode is None:
        raise SystemExit("pass --preset or --mode")
    overrides["mode"] = mode
    if a.max_seq_length:
        overrides["max_seq_length"] = a.max_seq_length

    ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    sd = remap_state_dict(ckpt.get("model_state_dict", ckpt))
    cfg = infer_config(sd, mode, ckpt.get("args"), overrides)

    model = BehaVERT(cfg)
    model.load_state_dict(sd, strict=True)
    if a.fp16:
        model.half()
    model.save_pretrained(a.out_dir)

    size = (a.out_dir / "model.safetensors").stat().st_size / 2**20
    n = sum(t.numel() for t in model.parameters()) / 1e6
    print(f"wrote {a.out_dir}  ({n:.1f}M parameters, {size:.0f} MiB)")
    print(f"  mode={cfg.mode} input_dim={cfg.input_dim} classes={cfg.num_classes} "
          f"seq_len={cfg.max_seq_length} pe={cfg.positional_encoding}")


if __name__ == "__main__":
    main()
