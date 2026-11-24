"""The ``behavert`` command.

    behavert decode   --video demo/video/oft.mp4 --model oft-behavior --overlay
    behavert decode   --config configs/calms21/task1.yaml --recording mouse071_task1_annotator1 \\
                      --model calms21-task1 --video mouse071.mp4 --sync
    behavert train    configs/mabe22/behavior.yaml --task chase
    behavert evaluate configs/mabe22/state.yaml mabe22-strain --task strain
    behavert attention configs/mabe22/state.yaml mabe22-strain --task strain
    behavert embed    configs/calms21/task1.yaml calms21-task1
    behavert pretrain configs/calms21/pretrain.yaml

    behavert download calms21 mabe22     # benchmark data into dataset/

Run ``behavert <command> -h`` for the options of each command.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def _apply_overrides(cfg: dict, items: list[str]) -> dict:
    """``--set a.b=value`` (value parsed as YAML)."""
    import yaml

    for item in items:
        key, _, value = item.partition("=")
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def _config_args(p: argparse.ArgumentParser, data_root_default: str) -> None:
    p.add_argument("--task", help="variant to run, for configs that list several under 'tasks'")
    p.add_argument("--data-root", default=data_root_default, help="base folder for relative data paths (default: dataset, or env BEHAVERT_DATA)")
    p.add_argument("--device", help="e.g. cuda, cuda:1, cpu (default: cuda if available)")


def _check_model_fits_config(model, cfg, command: str, classification_only: bool = False) -> None:
    """Stop early when a checkpoint and a config disagree (window length, task type)."""
    c = model.config
    L = cfg["data"]["window"]["length"]
    if L != c.max_seq_length:
        raise SystemExit(f"the config uses {L}-frame windows but the checkpoint was trained on "
                         f"{c.max_seq_length}; use the config the checkpoint was trained with")
    if (cfg["model"].get("mode", "token")) != c.mode:
        raise SystemExit(f"the config is for a {cfg['model'].get('mode', 'token')} model, the checkpoint is {c.mode}")
    if classification_only and c.problem != "classification":
        raise SystemExit(f"`behavert {command}` works with classification models")


# ------------------------------------------------------------------ train
def cmd_train(args) -> None:
    import yaml

    from .model import BehaVERT
    from .recipes import (build_datasets, build_loss_fn, build_model, build_sampler, evaluate_test, load_config,
                          resolve_device)
    from .training import TrainConfig, classifier_loss_step, evaluate_classifier, fit, make_loader, set_seed

    cfg = _apply_overrides(load_config(args.config, args.task), args.set)
    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    out_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(tcfg.seed)
    device = resolve_device(args.device)

    sequence = cfg["model"].get("mode", "token") == "sequence"
    ds = build_datasets(cfg, args.data_root, tcfg.seed, sequence_labels=sequence)
    print(f"windows: train {len(ds.train)}, valid {len(ds.valid) if ds.valid else 0}, "
          f"test recordings {len(ds.test_recordings) if ds.test_recordings else 0}")

    model = build_model(cfg, ds.train.w.inputs.shape[-1]).to(device)
    n_cls = model.config.num_classes
    loss_fn = build_loss_fn(cfg, ds.train, n_cls).to(device)
    sampler = build_sampler(cfg, ds.train, tcfg.batch_size, n_cls)
    train_loader = make_loader(ds.train, tcfg.batch_size, sampler, shuffle=sampler is None, num_workers=tcfg.num_workers)
    valid_loader = make_loader(ds.valid, tcfg.eval_batch_size) if ds.valid else None
    evaluate = None
    if valid_loader is not None:
        evaluate = lambda m: evaluate_classifier(m, valid_loader, loss_fn, tcfg, device, m.config.label_names)  # noqa: E731

    summary = fit(model, train_loader, tcfg, out_dir, classifier_loss_step(loss_fn), evaluate, select_on=tcfg.select_on)
    print(json.dumps(summary["best"], indent=2))

    test_recordings = ds.test_recordings
    del ds, train_loader, valid_loader, evaluate, sampler  # free the training windows before testing
    gc.collect()
    if test_recordings and not args.skip_test:
        results = {}
        for name in dict.fromkeys((f"best_{tcfg.select_on}", "best_loss", "last")):
            d = out_dir / name
            if d.exists():
                print(f"\n=== {name} ===")
                results[name] = evaluate_test(BehaVERT.from_pretrained(d, device=device), cfg, test_recordings,
                                              tcfg.eval_batch_size)
        (out_dir / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")


# ------------------------------------------------------------------ pretrain
def cmd_pretrain(args) -> None:
    import yaml

    from .model import BehaVERTConfig
    from .pretraining import MaskedKeypointModel, evaluate_pretraining, pretrain_loss_step
    from .recipes import build_datasets, load_config, resolve_device
    from .training import TrainConfig, fit, make_loader, set_seed

    cfg = _apply_overrides(load_config(args.config, args.task), args.set)
    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    out_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(tcfg.seed)
    device = resolve_device(args.device)
    ratio = cfg["pretraining"]["mask_ratio"]

    ds = build_datasets(cfg, args.data_root, tcfg.seed, sequence_labels=False)
    print(f"windows: train {len(ds.train)}, valid {len(ds.valid) if ds.valid else 0}")
    m = dict(cfg.get("model", {}))
    m.pop("init_from", None)
    config = BehaVERTConfig(
        input_dim=ds.train.w.inputs.shape[-1], num_classes=1, mode="token",
        max_seq_length=cfg["data"]["window"]["length"],
        description="encoder pretrained by masked keypoint modeling (classifier unused)",
        preprocessing={"pretraining": cfg["pretraining"], "window": cfg["data"]["window"]["length"]},
        **m,
    )
    model = MaskedKeypointModel(config).to(device)
    loader = make_loader(ds.train, tcfg.batch_size, shuffle=True, num_workers=tcfg.num_workers)
    valid = make_loader(ds.valid, tcfg.eval_batch_size) if ds.valid else None
    evaluate = (lambda mdl: evaluate_pretraining(mdl, valid, ratio, tcfg.seed)) if valid else None
    summary = fit(model, loader, tcfg, out_dir, pretrain_loss_step(ratio), evaluate, select_on=None)
    print(json.dumps(summary["best"], indent=2))


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args) -> None:
    from .model import BehaVERT
    from .recipes import evaluate_test, load_config, load_split_recordings, resolve_device

    cfg = load_config(args.config, args.task)
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))
    if not recs["test"]:
        raise SystemExit("this config defines no test set")
    model = BehaVERT.from_pretrained(args.checkpoint, device=resolve_device(args.device))
    _check_model_fits_config(model, cfg, "evaluate")
    print(f"{args.checkpoint}: {len(recs['test'])} test recordings")
    results = evaluate_test(model, cfg, recs["test"], args.batch_size)
    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


# ------------------------------------------------------------------ attention
def cmd_attention(args) -> None:
    import numpy as np
    import torch

    from .attention import run_attention_analysis
    from .model import BehaVERT
    from .recipes import load_config, load_split_recordings, resolve_device

    cfg = load_config(args.config, args.task)
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))[args.split]
    if not recs:
        raise SystemExit(f"this config has no {args.split} recordings")
    recs = sorted(recs, key=lambda r: r.name)[: args.recordings or None]
    model = BehaVERT.from_pretrained(args.checkpoint, device=resolve_device(args.device))
    _check_model_fits_config(model, cfg, "attention", classification_only=True)

    # per-frame behavior labels to relate attention to: the model's display strip files, if present
    behaviors = {}
    spec = (model.config.display or {}).get("extra_strip")
    if spec and not args.no_behaviors:
        for name, _color, rel in spec["labels"]:
            path = Path(args.data_root) / rel
            if path.exists():
                d = torch.load(path, map_location="cpu", weights_only=False)
                behaviors[name] = {r.name: np.asarray(d[r.name]) for r in recs if r.name in d}
    out = Path(args.out or f"outputs/attention/{Path(args.config).stem}" + (f"_{args.task}" if args.task else ""))
    print(f"{len(recs)} {args.split} recordings, {len(behaviors)} behavior label sets -> {out}")
    tables = run_attention_analysis(model, recs, out, stride=args.stride, behaviors=behaviors or None,
                                    batch_size=args.batch_size)
    t = tables["attention_tests"]
    with __import__("pandas").option_context("display.width", 160, "display.max_colwidth", 48):
        print(t[["analysis", "group_a", "group_b", "mean_a", "mean_b", "p_value"]].to_string(index=False))
    print(f"wrote {', '.join(n + '.csv' for n in tables)} to {out}")


# ------------------------------------------------------------------ embed
def cmd_embed(args) -> None:
    from .clustering import run_embedding_analysis
    from .model import BehaVERT
    from .recipes import load_config, load_split_recordings, resolve_device

    cfg = load_config(args.config, args.task)
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))[args.split]
    if not recs:
        raise SystemExit(f"this config has no {args.split} recordings")
    recs = sorted(recs, key=lambda r: r.name)[: args.recordings or None]
    model = BehaVERT.from_pretrained(args.checkpoint, device=resolve_device(args.device))
    _check_model_fits_config(model, cfg, "embed", classification_only=True)
    out = Path(args.out or f"outputs/embedding/{Path(args.config).stem}" + (f"_{args.task}" if args.task else ""))
    exclude = args.exclude if args.exclude is not None else cfg.get("train", {}).get("metric_ignore_classes", [])
    print(f"{len(recs)} {args.split} recordings -> {out}")
    metrics = run_embedding_analysis(model, recs, out, method=args.method, stride=args.stride,
                                     exclude_classes=exclude, max_points=args.max_points or None,
                                     color_by=args.color_by, title=args.title or "", batch_size=args.batch_size)
    print(metrics.to_string(index=False))
    print(f"wrote embedding.csv, clustering.csv and embedding.png/.svg/.pdf to {out}")


# ------------------------------------------------------------------ decode
def cmd_decode(args) -> None:
    from .decode import decode

    decode(
        model=args.model, video=args.video, keypoints=args.keypoints, config=args.config,
        recording=args.recording, task=args.task, data_root=args.data_root, pose_model=args.pose_model,
        labels=args.labels, out=args.out, overlay=args.overlay, sync=args.sync, start=args.start,
        frames=args.frames, sync_frames=args.sync_frames, stride=args.stride, fps=args.fps,
        device=args.device, batch_size=args.batch_size, limit=args.limit, force=args.force,
        backdrop=args.backdrop,
    )


# ------------------------------------------------------------------ download
def cmd_download(args) -> None:
    from huggingface_hub import snapshot_download

    from .recipes import BENCHMARKS, DATASET_REPO

    names = args.benchmarks or list(BENCHMARKS)
    unknown = sorted(set(names) - set(BENCHMARKS))
    if unknown:
        raise SystemExit(f"unknown benchmark(s) {unknown}; choose from {', '.join(BENCHMARKS)}")
    # only the benchmark folders: dataset/README.md and LICENSE are part of the repository
    patterns = [f"{n}/**" for n in names]
    print(f"downloading {', '.join(names)} from {DATASET_REPO} to {args.dir}/ ...")
    snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=args.dir, allow_patterns=patterns,
                      revision=args.revision)
    print(f"done; commands read {args.dir}/ by default")


# ------------------------------------------------------------------ main
def main(argv=None) -> None:
    from .recipes import DEFAULT_DATA_ROOT

    ap = argparse.ArgumentParser(prog="behavert", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("download", help="download the benchmark datasets from Hugging Face",
                       description="Download the BehaVERT benchmark data (all, or the ones named) into dataset/.")
    p.add_argument("benchmarks", nargs="*", metavar="benchmark",
                   help="calms21, mabe22, pair24m, deepethogram, sbea (default: all)")
    p.add_argument("--dir", default=DEFAULT_DATA_ROOT, help="target folder (default: dataset)")
    p.add_argument("--revision", help="dataset revision (branch, tag or commit)")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("decode", help="videos or keypoints -> behavior, embeddings, attention, videos",
                       description="Decode behavior from videos (with a pose model) or keypoints, and render "
                                   "overlay and synchronized videos. Outputs go to <out>/<recording>/.")
    src = p.add_argument_group("input (one of)")
    src.add_argument("--video", help="video file or folder; keypoints come from the pose model "
                                     "(with --keypoints / --config: the video to draw on)")
    src.add_argument("--keypoints", help="keypoint file (.csv, .npy, or a recordings .pt/.npz/.pkl)")
    src.add_argument("--config", help="training config whose dataset holds --recording (gives ground truth)")
    src.add_argument("--recording", help="recording name inside --keypoints or --config")
    p.add_argument("--model", required=True, help="BehaVERT checkpoint: folder, alias (calms21-task1, "
                                                  "mabe22-strain, oft-behavior) or Hub repo id")
    p.add_argument("--pose-model", help="pose model for --video (default: the one named by the behavior model)")
    p.add_argument("--labels", help="ground-truth labels per frame (.csv / .npy) to show in the sync video")
    p.add_argument("--out", default="outputs/decode", help="output folder (default: outputs/decode)")
    p.add_argument("--overlay", action="store_true", help="write overlay.mp4 (skeleton + behavior on the video)")
    p.add_argument("--sync", action="store_true", help="write sync.mp4 (video, embedding map, attention, labels)")
    p.add_argument("--start", type=int, default=0, help="first frame of the rendered videos")
    p.add_argument("--frames", type=int, help="number of frames to render (overlay default: all)")
    p.add_argument("--sync-frames", type=int, default=900, help="frames in sync.mp4 if --frames is not set (default 900)")
    p.add_argument("--backdrop", type=int, help="dataset recordings in the embedding-map backdrop of sync.mp4 "
                                                "(needs --config; 0 = clip only; default: the model's setting)")
    p.add_argument("--stride", type=int, help="window step (default: the model's)")
    p.add_argument("--fps", type=float, help="frame rate (default: from the video, else the model)")
    p.add_argument("--limit", type=int, help="process only the first N frames of each input (quick tries)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--force", action="store_true", help="recompute stages whose outputs exist")
    _config_args(p, DEFAULT_DATA_ROOT)
    p.set_defaults(func=cmd_decode)

    for name, func, helptext in (("train", cmd_train, "train a classifier from a config, then test it"),
                                 ("pretrain", cmd_pretrain, "self-supervised pretraining (masked keypoint modeling)")):
        p = sub.add_parser(name, help=helptext, description=helptext)
        p.add_argument("config")
        _config_args(p, DEFAULT_DATA_ROOT)
        p.add_argument("--output-dir", help="overrides output_dir in the config")
        p.add_argument("--set", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                       help="override config entries (may be repeated)")
        if name == "train":
            p.add_argument("--skip-test", action="store_true")
        p.set_defaults(func=func)

    p = sub.add_parser("attention", help="attention vs confidence, correctness and behaviors, as CSV tables",
                       description="Collect per-frame attention over a config's recordings and relate it to "
                                   "confidence, correctness and (if available) behavior labels. Writes CSV tables.")
    p.add_argument("config")
    p.add_argument("checkpoint", help="folder, alias or Hub repo id")
    _config_args(p, DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--recordings", type=int, help="use only the first N recordings (by name)")
    p.add_argument("--stride", type=int, help="window step (default: the window length, i.e. no overlap)")
    p.add_argument("--no-behaviors", action="store_true", help="skip the behavior tables")
    p.add_argument("--out", help="output folder (default: outputs/attention/<config>[_<task>])")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_attention)

    p = sub.add_parser("embed", help="embedding map (t-SNE / UMAP) and clustering metrics over a dataset",
                       description="Project a dataset's embeddings to 2D, plot them colored by class, and score "
                                   "how well they cluster (NMI, purity, accuracy, F1).")
    p.add_argument("config")
    p.add_argument("checkpoint", help="folder, alias or Hub repo id")
    _config_args(p, DEFAULT_DATA_ROOT)
    p.add_argument("--split", default="test", choices=["train", "valid", "test"])
    p.add_argument("--recordings", type=int, help="use only the first N recordings (by name)")
    p.add_argument("--method", default="tsne", choices=["tsne", "umap"])
    p.add_argument("--color-by", default="true", choices=["true", "pred"], help="color points by true or predicted class")
    p.add_argument("--exclude", type=int, nargs="*", help="classes to leave out (default: the config's "
                                                          "metric_ignore_classes, e.g. CalMS21 'other')")
    p.add_argument("--max-points", type=int, default=20000, help="random subsample for the projection (0 = all)")
    p.add_argument("--stride", type=int, help="window step (default: the window length)")
    p.add_argument("--title")
    p.add_argument("--out", help="output folder (default: outputs/embedding/<config>[_<task>])")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("evaluate", help="test a checkpoint on a config's test set",
                       description="Evaluate a checkpoint on the test split defined by a config.")
    p.add_argument("config")
    p.add_argument("checkpoint", help="folder, alias or Hub repo id")
    _config_args(p, DEFAULT_DATA_ROOT)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--output", help="write metrics to this JSON file")
    p.set_defaults(func=cmd_evaluate)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
