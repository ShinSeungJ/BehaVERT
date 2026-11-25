"""The ``behavert`` command.

    behavert decode   --video demo_data/video/oft.mp4 --model oft-behavior --overlay
    behavert decode   --config configs/calms21/task1.yaml --recording mouse071_task1_annotator1 \\
                      --model calms21-task1 --video mouse071.mp4 --sync
    behavert train    configs/mabe22/behavior.yaml --task chase
    behavert evaluate configs/mabe22/state.yaml mabe22-strain --task strain
    behavert attention configs/mabe22/state.yaml mabe22-strain --task strain
    behavert embed    configs/calms21/task1_pretrained.yaml calms21-task1
    behavert pretrain configs/calms21/pretrain.yaml

    behavert download calms21 mabe22     # benchmark data into dataset/

Run ``behavert <command> -h`` for the options of each command.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path


def _true_false(value: str) -> bool:
    v = value.strip().lower()
    if v in ("true", "1", "yes"):
        return True
    if v in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


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


# ------------------------------------------------------------------ run settings shown in the log
def _config_name(args) -> str:
    return f"{args.config} (task: {args.task})" if getattr(args, "task", None) else str(args.config)


def _model_desc(cfg: dict, short: bool = False) -> str:
    m = cfg.get("model", {})
    mode = m.get("mode", "token")
    if m.get("problem") == "regression":
        return "sequence regression"
    if short:
        return mode
    names = m.get("label_names") or []
    return f"{mode}, {m.get('num_classes', len(names))} classes" + (f" ({', '.join(names)})" if names else "")


def _loss_rows(cfg: dict) -> list:
    loss = dict(cfg.get("loss") or {"name": "hybrid"})
    name = loss.pop("name")
    mult = loss.pop("class_weight_multipliers", None)
    loss.pop("class_weights", None)
    params = ", ".join(f"{k} {v}" for k, v in loss.items())
    rows = [("Loss function", f"{name} ({params})" if params else name)]
    if mult is not None:
        rows.append(("Weight multipliers", str(list(mult))))
    return rows


def _augmentation_rows(cfg: dict) -> list:
    from . import report

    aug = cfg["data"].get("augmentation")
    if not aug or not aug.get("enabled", True):
        return [("Augmentation", "off")]
    arena = aug.get("arena")
    classes = aug.get("classes")
    names = cfg.get("model", {}).get("label_names")
    return [("Augmentation", f"{arena} arena" if isinstance(arena, str) else "custom arena"),
            ("Augment classes", "all windows" if classes is None else report.named(classes, names))]


def _window_rows(cfg: dict) -> list:
    from .recipes import stride_description

    w = cfg["data"]["window"]
    return [("Max seq length", w["length"]), ("Sequence stride", stride_description(w))]


def _start_rows(args, cfg: dict, tcfg, out_dir, pretraining: bool = False) -> list:
    from . import report
    from .training import resolve_patience

    rows = [("Config", _config_name(args)), ("Output dir", out_dir)]
    if cfg.get("model", {}).get("init_from"):
        rows.append(("Pretrain checkpoint", cfg["model"]["init_from"]))
    if pretraining:
        if "use_all_data" in cfg:
            rows.append(("Use all data", bool(cfg["use_all_data"])))
        rows.append(("Mask ratio", cfg["pretraining"]["mask_ratio"]))
    else:
        rows.append(("Model", _model_desc(cfg)))
    rows += _window_rows(cfg) + _augmentation_rows(cfg)
    if not pretraining:
        rows += _loss_rows(cfg)
    rows += [("Batch size", tcfg.batch_size), ("Learning rate", report.number(tcfg.learning_rate)),
             ("Epochs", tcfg.epochs), ("Early stop patience", resolve_patience(tcfg.patience, tcfg.epochs)),
             ("Seed", tcfg.seed)]
    return rows


def _hyperparameter_rows(args, cfg: dict, tcfg, out_dir, summary: dict, pretraining: bool = False) -> list:
    from . import report

    rows = [("Config", _config_name(args)), ("Output dir", out_dir)]
    if cfg.get("model", {}).get("init_from"):
        rows.append(("Pretrain checkpoint", cfg["model"]["init_from"]))
    if pretraining:
        if "use_all_data" in cfg:
            rows.append(("Use all data", bool(cfg["use_all_data"])))
        rows.append(("Mask ratio", cfg["pretraining"]["mask_ratio"]))
    else:
        rows.append(("Model", _model_desc(cfg, short=True)))
        rows += _loss_rows(cfg)
        regression = cfg.get("model", {}).get("problem") == "regression"
        rows.append(("Sampler", "none" if regression else cfg.get("sampler", "proportional")))
    epochs = f"{tcfg.epochs}" + (f" (early stop at {summary['epochs']})" if summary.get("stopped_early") else "")
    steps, total = summary["warmup_steps"], summary["total_steps"]
    n = f"{steps} step" + ("" if steps == 1 else "s")
    warmup = (f"{n} ({100 * steps / total:.2f}%)" if isinstance(tcfg.warmup, int)
              else f"{tcfg.warmup or '10%'} ({n})")
    rows += [("Epochs", epochs), ("Batch size", tcfg.batch_size), ("Learning rate", report.number(tcfg.learning_rate)),
             ("Weight decay", report.number(tcfg.weight_decay)), ("Warmup", warmup), ("Total steps", total),
             ("Grad accum steps", tcfg.grad_accum)]
    rows += _window_rows(cfg) + [("Max grad norm", report.number(tcfg.max_grad_norm))] + _augmentation_rows(cfg)
    every = lambda v: v if v == "epoch" else f"{v} step" + ("" if v == 1 else "s")  # noqa: E731
    rows += [("Report every", every(tcfg.log_every)), ("Eval every", every(tcfg.eval_every)),
             ("Early stop patience", summary["patience"]), ("Seed", tcfg.seed)]
    return rows


def _checkpoint_desc(name, model) -> str:
    c = model.config
    return f"{name} ({c.mode}, {c.num_classes} classes)"


def _table_kind(cfg: dict) -> str:
    m = cfg.get("model", {})
    if m.get("problem") == "regression":
        return "regression"
    return "sequence" if m.get("mode") == "sequence" else "token"


# ------------------------------------------------------------------ train
def cmd_train(args) -> None:
    import yaml

    from . import report
    from .model import BehaVERT
    from .recipes import (build_datasets, build_loss_fn, build_model, build_sampler, default_select_on,
                          evaluate_test, load_config, print_distribution, resolve_device, test_windows)
    from .training import (TrainConfig, checkpoint_name, classifier_loss_step, evaluate_classifier, fit,
                           make_loader, set_seed)

    cfg = _apply_overrides(load_config(args.config, args.task), args.set)
    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    tcfg.select_on = tcfg.select_on or default_select_on(cfg)
    out_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(tcfg.seed)
    device = resolve_device(args.device)
    kind = _table_kind(cfg)
    sequence = cfg["model"].get("mode", "token") == "sequence"

    title = "FINE-TUNING FROM PRETRAINED MODEL" if cfg["model"].get("init_from") else "TRAINING CONFIGURATION"
    report.block(title, _start_rows(args, cfg, tcfg, out_dir))
    ds = build_datasets(cfg, args.data_root, tcfg.seed, sequence_labels=sequence)
    model = build_model(cfg, ds.train.w.inputs.shape[-1]).to(device)
    print()
    print_distribution("Train", ds.train.w, cfg, sequence, ds.train_recordings)
    if ds.valid is not None:
        print()
        print_distribution("Validation", ds.valid.w, cfg, sequence, ds.valid_recordings)
    print()

    n_cls = model.config.num_classes
    loss_fn = build_loss_fn(cfg, ds.train, n_cls).to(device)
    sampler = build_sampler(cfg, ds.train, tcfg.batch_size, n_cls)
    train_loader = make_loader(ds.train, tcfg.batch_size, sampler, shuffle=sampler is None, num_workers=tcfg.num_workers)
    valid_loader = make_loader(ds.valid, tcfg.eval_batch_size) if ds.valid else None
    evaluate = None
    if valid_loader is not None:
        evaluate = lambda m: evaluate_classifier(m, valid_loader, loss_fn, tcfg, device, m.config.label_names)  # noqa: E731

    log = report.TrainingReport("regression" if kind == "regression" else "classification",
                                "behavior" if kind == "token" else cfg.get("label_title", "class"))
    summary = fit(model, train_loader, tcfg, out_dir, classifier_loss_step(loss_fn), evaluate,
                  select_on=tcfg.select_on, report=log)

    test_recordings = ds.test_recordings
    del ds, train_loader, valid_loader, evaluate, sampler  # free the training windows before testing
    gc.collect()
    print()
    if not test_recordings or args.skip_test:
        report.block("HYPERPARAMETERS", _hyperparameter_rows(args, cfg, tcfg, out_dir, summary), close=False)
        return
    windows = test_windows(cfg, test_recordings)
    print_distribution("Test", windows[0], cfg, sequence, len(test_recordings))
    print()
    report.block("HYPERPARAMETERS", _hyperparameter_rows(args, cfg, tcfg, out_dir, summary), close=False)
    results = {}
    for name in dict.fromkeys(("best_loss", checkpoint_name(tcfg.select_on), "last")):
        d = out_dir / name
        if d.exists():
            results[name] = evaluate_test(BehaVERT.from_pretrained(d, device=device), cfg, test_recordings,
                                          tcfg.eval_batch_size, windows)
    report.inference_table(results, kind)
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")


# ------------------------------------------------------------------ pretrain
def cmd_pretrain(args) -> None:
    import yaml

    from . import report
    from .data.dataset import WindowDataset
    from .model import BehaVERTConfig
    from .pretraining import MaskedKeypointModel, evaluate_pretraining, pretrain_loss_step
    from .recipes import _merge, build_datasets, load_config, print_distribution, resolve_device, test_windows
    from .training import TrainConfig, fit, make_loader, set_seed

    cfg = _apply_overrides(load_config(args.config, args.task), args.set)
    # use_all_data (config or --use-all-data): false swaps in the task_data_only settings
    if args.use_all_data is not None:
        cfg["use_all_data"] = args.use_all_data
    task_only = cfg.pop("task_data_only", None)
    if not cfg.get("use_all_data", True):
        if not task_only:
            raise SystemExit(f"{args.config} has no task_data_only settings for use_all_data: false")
        cfg = _merge(cfg, task_only)
    tcfg = TrainConfig.from_dict(cfg.get("train", {}))
    out_dir = Path(args.output_dir or cfg.get("output_dir", "outputs/pretrain"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    set_seed(tcfg.seed)
    device = resolve_device(args.device)
    ratio = cfg["pretraining"]["mask_ratio"]
    units = cfg["pretraining"].get("units", "")

    report.block("PRETRAINING CONFIGURATION", _start_rows(args, cfg, tcfg, out_dir, pretraining=True))
    ds = build_datasets(cfg, args.data_root, tcfg.seed, sequence_labels=False)
    print()
    print_distribution("Train", ds.train.w, cfg, False, pretraining=True)
    if ds.valid is not None:
        print()
        print_distribution("Validation", ds.valid.w, cfg, False, pretraining=True)
    print()
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
    summary = fit(model, loader, tcfg, out_dir, pretrain_loss_step(ratio), evaluate, select_on=None,
                  report=report.TrainingReport("pretraining", units=units))

    print()
    rows = _hyperparameter_rows(args, cfg, tcfg, out_dir, summary, pretraining=True)
    # with a data.test file (e.g. CalMS21 with --use-all-data false) the encoder is
    # scored on it: reconstruction error of the masked coordinates, in keypoint units
    if not ds.test_recordings or args.skip_test:
        report.block("HYPERPARAMETERS", rows, close=False)
        print()
        report.banner("TRAINING COMPLETE")
        if not ds.test_recordings:
            print("No test set: all the data was used for pretraining.")
        print(f"The encoder in {out_dir / 'best_loss'} is ready for fine-tuning.")
        return
    test_w, _ = test_windows(cfg, ds.test_recordings)
    print_distribution("Test", test_w, cfg, False, pretraining=True)
    print()
    report.block("HYPERPARAMETERS", rows, close=False)
    test_loader = make_loader(WindowDataset(test_w), tcfg.eval_batch_size)
    results = {}
    for name in ("best_loss", "last"):
        d = out_dir / name
        if d.exists():
            results[name] = evaluate_pretraining(MaskedKeypointModel.from_pretrained(d, device), test_loader,
                                                 ratio, tcfg.seed)
    report.inference_table(results, "pretraining", units)
    (out_dir / "test_results.json").write_text(json.dumps(results, indent=2) + "\n")


# ------------------------------------------------------------------ evaluate
def cmd_evaluate(args) -> None:
    from . import report
    from .model import BehaVERT
    from .recipes import evaluate_test, load_config, load_split_recordings, print_distribution, resolve_device, test_windows

    cfg = load_config(args.config, args.task)
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))
    if not recs["test"]:
        raise SystemExit("this config defines no test set")
    model = BehaVERT.from_pretrained(args.checkpoint, device=resolve_device(args.device))
    _check_model_fits_config(model, cfg, "evaluate")
    windows = test_windows(cfg, recs["test"])
    print()
    print_distribution("Test", windows[0], cfg, cfg["model"].get("mode") == "sequence", len(recs["test"]))
    results = {Path(str(args.checkpoint)).name: evaluate_test(model, cfg, recs["test"], args.batch_size, windows)}
    report.inference_table(results, _table_kind(cfg))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        report.saved("Metrics", args.output)


# ------------------------------------------------------------------ attention
def cmd_attention(args) -> None:
    import numpy as np
    import torch

    from .attention import run_attention_analysis
    from .model import BehaVERT
    from .recipes import load_config, load_split_recordings, resolve_device

    from . import report

    cfg = load_config(args.config, args.task)
    device = resolve_device(args.device)
    model = BehaVERT.from_pretrained(args.checkpoint, device=device)
    _check_model_fits_config(model, cfg, "attention", classification_only=True)
    out = Path(args.out or f"outputs/attention/{Path(args.config).stem}" + (f"_{args.task}" if args.task else ""))
    spec = (model.config.display or {}).get("extra_strip")
    n_sets = len(spec["labels"]) if spec and not args.no_behaviors else 0
    report.block("ATTENTION CONFIGURATION", [
        ("Config", _config_name(args)), ("Model", _checkpoint_desc(args.checkpoint, model)),
        ("Split", args.split + (f" (first {args.recordings} recordings)" if args.recordings else "")),
        ("Sequence stride", args.stride or f"{model.config.max_seq_length} (the whole window)"),
        ("Behavior labels", f"{n_sets} sets" if n_sets else "none"), ("Output dir", out), ("Device", device)])
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))[args.split]
    if not recs:
        raise SystemExit(f"this config has no {args.split} recordings")
    recs = sorted(recs, key=lambda r: r.name)[: args.recordings or None]

    # per-frame behavior labels to relate attention to: the model's display strip files, if present
    behaviors = {}
    spec = (model.config.display or {}).get("extra_strip")
    if spec and not args.no_behaviors:
        for name, _color, rel in spec["labels"]:
            path = Path(args.data_root) / rel
            if path.exists():
                d = torch.load(path, map_location="cpu", weights_only=False)
                behaviors[name] = {r.name: np.asarray(d[r.name]) for r in recs if r.name in d}
    L = model.config.max_seq_length
    s = args.stride or L
    tables = run_attention_analysis(model, recs, out, stride=args.stride, behaviors=behaviors or None,
                                    batch_size=args.batch_size)
    report.step_line(f"Collecting attention over the {args.split} split ({len(recs):,} recordings, {L} frames, stride {s})",
                     f"{len(tables['attention_windows']):,} sequences, {len(tables['attention_frames']):,} frames; "
                     "confidence split at its median")
    report.attention_tests_table(tables["attention_tests"])
    labels = {"attention_frames": "Attention per frame", "attention_windows": "Attention per window",
              "attention_tests": "Attention tests", "behavior_attention": "Behavior attention",
              "behavior_peaks": "Behavior peaks", "behavior_success_failure": "Behavior success / failure"}
    for name in tables:
        report.saved(labels.get(name, name.replace("_", " ").capitalize()), out / f"{name}.csv")


# ------------------------------------------------------------------ embed
def cmd_embed(args) -> None:
    from .clustering import run_embedding_analysis
    from .model import BehaVERT
    from .recipes import load_config, load_split_recordings, resolve_device

    from . import report

    cfg = load_config(args.config, args.task)
    device = resolve_device(args.device)
    model = BehaVERT.from_pretrained(args.checkpoint, device=device)
    _check_model_fits_config(model, cfg, "embed", classification_only=True)
    out = Path(args.out or f"outputs/embedding/{Path(args.config).stem}" + (f"_{args.task}" if args.task else ""))
    exclude = args.exclude if args.exclude is not None else cfg.get("train", {}).get("metric_ignore_classes", [])
    names = list(model.config.label_names or [])
    L = model.config.max_seq_length
    report.block("EMBEDDING CONFIGURATION", [
        ("Config", _config_name(args)), ("Model", _checkpoint_desc(args.checkpoint, model)),
        ("Split", args.split + (f" (first {args.recordings} recordings)" if args.recordings else "")),
        ("Method", {"tsne": "t-SNE", "umap": "UMAP"}.get(args.method, args.method)),
        ("Color by", "true class" if args.color_by == "true" else "predicted class"),
        ("Excluded classes", report.named(sorted(exclude), names) if exclude else "none"),
        ("Max points", args.max_points or "all"),
        ("Sequence stride", args.stride or f"{L} (the whole window)"), ("Output dir", out), ("Device", device)])
    recs = load_split_recordings(cfg, args.data_root, cfg.get("train", {}).get("seed", 42))[args.split]
    if not recs:
        raise SystemExit(f"this config has no {args.split} recordings")
    recs = sorted(recs, key=lambda r: r.name)[: args.recordings or None]
    metrics = run_embedding_analysis(model, recs, out, method=args.method, stride=args.stride,
                                     exclude_classes=exclude, max_points=args.max_points or None,
                                     color_by=args.color_by, title=args.title or "", batch_size=args.batch_size)
    method = {"tsne": "t-SNE", "umap": "UMAP"}.get(args.method, args.method)
    report.step_line(f"Embedding the {args.split} split ({len(recs):,} recordings, {L} frames, stride {args.stride or L})",
                     f"{int(metrics['n'].iloc[0]):,} points of the scored classes; projected with {method}")
    report.clustering_table(metrics)
    report.saved("Embedding", out / "embedding.csv")
    report.saved("Clustering metrics", out / "clustering.csv")
    report.saved("Embedding map", f"{out / 'embedding.png'} (.svg, .pdf)")


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
    print(f"Downloading {', '.join(names)} from {DATASET_REPO} to {args.dir}/", flush=True)
    snapshot_download(DATASET_REPO, repo_type="dataset", local_dir=args.dir, allow_patterns=patterns,
                      revision=args.revision)
    print(f"Download complete. Every command reads {args.dir}/ by default.")


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
        p.add_argument("--skip-test", action="store_true", help="skip scoring on the config's test set")
        if name == "pretrain":
            p.add_argument("--use-all-data", type=_true_false, metavar="{true,false}",
                           help="pretrain on all data (true) or on the task's training data only, then score "
                                "the encoder on its test data (false); default: use_all_data in the config")
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
