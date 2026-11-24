"""Dataset-level attention analysis (paper Fig. 8), written as CSV tables.

For every window of the chosen recordings, the attention each frame gets from
the last layer (heads averaged; ``[CLS]`` -> frame for sequence models, the
attention a frame receives for token models) and from all layers averaged is
recorded with the prediction, its confidence and correctness. From these:

``attention_frames.csv``   one row per frame of every window (+ behavior labels if given)
``attention_windows.csv``  one row per window: confidence, correctness, attention mean and
                           variance (peakedness) for the last layer and all layers
``attention_tests.csv``    median-split comparisons:
    confidence -> peakedness, peakedness -> confidence, correctness -> peakedness,
    peakedness -> correctness (last layer and all layers), confidence -> correctness
``behavior_attention.csv``        attention with vs without each behavior (success state)
``behavior_peaks.csv``            behavior rate in peak vs other frames (success state)
``behavior_success_failure.csv``  attention on each behavior, success vs failure state

The success state is high-confidence (>= median) correct windows, the failure
state high-confidence wrong windows; a peak frame has attention above its
window's mean + 1 std.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .data.windows import sliding_windows
from .model import BehaVERT


@torch.no_grad()
def collect_attention(model: BehaVERT, recordings, stride: Optional[int] = None,
                      behaviors: Optional[dict[str, dict]] = None, batch_size: int = 64) -> pd.DataFrame:
    """One row per real frame of every window of ``recordings``.

    ``behaviors``: ``{name: {recording_name: per-frame 0/1}}`` adds ``behavior_<name>`` columns.
    """
    cfg = model.config
    L = cfg.max_seq_length
    stride = stride or L
    dev = next(model.parameters()).device
    model.eval()
    names = list(cfg.label_names) or [str(c) for c in range(cfg.num_classes)]
    rows = []
    for rec in recordings:
        y = rec.labels_per_frame()
        w = sliding_windows(rec.keypoints, y, L, stride, pad_remainder=True)
        for i in range(0, len(w), batch_size):
            x = w.inputs[i : i + batch_size].to(dev)
            m = w.mask[i : i + batch_size].to(dev)
            out = model(x, attention_mask=m, output_attentions=True)
            p = out.logits.float().softmax(-1)
            att = torch.stack([a.float().mean(1) for a in out.attentions])  # [layers, B, S, S]
            mf = m.float()
            if cfg.mode == "sequence":
                a_last, a_all = att[-1, :, 0, 1:], att[:, :, 0, 1:].mean(0)
                frame_p = p[:, None, :].expand(-1, L, -1)
            else:
                recv = (att * mf[None, :, :, None]).sum(2) / mf.sum(1)[None, :, None]  # [layers, B, L]
                a_last, a_all = recv[-1], recv.mean(0)
                frame_p = p
            a_last, a_all, frame_p = a_last.cpu().numpy(), a_all.cpu().numpy(), frame_p.cpu().numpy()
            for k in range(x.shape[0]):
                j = i + k
                n = int(m[k].sum()); s = int(w.starts[j])
                fp = frame_p[k, :n]
                pred = fp.argmax(1)
                d = {"recording": rec.name, "window": j, "frame": np.arange(s, s + n), "token": np.arange(n),
                     "true": w.labels[j, :n].numpy() if w.labels is not None else -1,
                     "pred": pred, "confidence": fp.max(1),
                     "attention_last_layer": a_last[k, :n], "attention_all_layers": a_all[k, :n]}
                for c, nm in enumerate(names):
                    d[f"p_{nm}"] = fp[:, c]
                for bn, per_rec in (behaviors or {}).items():
                    v = per_rec.get(rec.name)
                    d[f"behavior_{bn}"] = (np.asarray(v)[s : s + n] > 0).astype(int) if v is not None else 0
                rows.append(pd.DataFrame(d))
    df = pd.concat(rows, ignore_index=True)
    df["correct"] = (df["true"] == df["pred"]).astype(int)
    return df


def window_table(frames: pd.DataFrame) -> pd.DataFrame:
    g = frames.groupby(["recording", "window"], sort=False)
    t = g.agg(start=("frame", "min"), frames=("frame", "size"), true=("true", "first"), pred=("pred", "first"),
              confidence=("confidence", "mean"), correct=("correct", "mean"),
              attention_mean=("attention_last_layer", "mean"), attention_var=("attention_last_layer", "var"),
              attention_mean_all_layers=("attention_all_layers", "mean"),
              attention_var_all_layers=("attention_all_layers", "var")).reset_index()
    t["correct"] = (t["correct"] >= 0.5).astype(int)
    return t


def _compare(name: str, a: pd.Series, b: pd.Series, labels=("high", "low"), measure: str = "") -> dict:
    from scipy.stats import ttest_ind

    a, b = a.dropna(), b.dropna()
    t, p = ttest_ind(a, b) if len(a) > 1 and len(b) > 1 else (np.nan, np.nan)
    return {"analysis": name, "measure": measure, "group_a": labels[0], "group_b": labels[1],
            "n_a": len(a), "n_b": len(b), "mean_a": a.mean(), "mean_b": b.mean(),
            "ratio_a_b": a.mean() / b.mean() if b.mean() else np.nan, "statistic": t, "p_value": p, "test": "t-test"}


def tests_table(frames: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import chi2_contingency, levene

    rows = []
    med_conf = windows["confidence"].median()
    hi_w, lo_w = windows[windows.confidence >= med_conf], windows[windows.confidence < med_conf]
    for layer, col, var in (("last layer", "attention_last_layer", "attention_var"),
                            ("all layers", "attention_all_layers", "attention_var_all_layers")):
        key = frames.set_index(["recording", "window"]).index
        hi_f = frames[key.isin(hi_w.set_index(["recording", "window"]).index)]
        lo_f = frames[key.isin(lo_w.set_index(["recording", "window"]).index)]
        # confidence -> attention
        rows.append(_compare(f"confidence -> peakedness ({layer})", hi_w[var], lo_w[var],
                             ("high confidence", "low confidence"), "window attention variance"))
        rows.append(_compare(f"confidence -> attention ({layer})", hi_f[col], lo_f[col],
                             ("high confidence", "low confidence"), "frame attention"))
        lv = levene(hi_f[col], lo_f[col]) if len(hi_f) > 1 and len(lo_f) > 1 else (np.nan, np.nan)
        rows.append({"analysis": f"confidence -> attention spread ({layer})", "measure": "frame attention variance",
                     "group_a": "high confidence", "group_b": "low confidence", "n_a": len(hi_f), "n_b": len(lo_f),
                     "mean_a": hi_f[col].var(), "mean_b": lo_f[col].var(),
                     "ratio_a_b": hi_f[col].var() / lo_f[col].var() if lo_f[col].var() else np.nan,
                     "statistic": lv[0], "p_value": lv[1], "test": "Levene"})
        # peakedness -> confidence
        med_var = windows[var].median()
        pk, un = windows[windows[var] >= med_var], windows[windows[var] < med_var]
        rows.append(_compare(f"peakedness -> confidence ({layer})", pk["confidence"], un["confidence"],
                             ("peaked", "uniform"), "window confidence"))
        rows.append(_compare(f"peakedness -> correctness ({layer})", pk["correct"], un["correct"],
                             ("peaked", "uniform"), "window accuracy"))
        # correctness -> attention
        ok, bad = windows[windows.correct == 1], windows[windows.correct == 0]
        rows.append(_compare(f"correctness -> peakedness ({layer})", ok[var], bad[var],
                             ("correct", "incorrect"), "window attention variance"))
    # confidence -> correctness
    hi = windows.confidence >= med_conf
    table = pd.crosstab(hi, windows.correct)
    chi2, p = (chi2_contingency(table)[:2] if table.shape == (2, 2) else (np.nan, np.nan))
    rows.append({"analysis": "confidence -> correctness", "measure": "window accuracy",
                 "group_a": "high confidence", "group_b": "low confidence", "n_a": int(hi.sum()), "n_b": int((~hi).sum()),
                 "mean_a": windows.correct[hi].mean(), "mean_b": windows.correct[~hi].mean(),
                 "ratio_a_b": windows.correct[hi].mean() / windows.correct[~hi].mean() if windows.correct[~hi].mean() else np.nan,
                 "statistic": chi2, "p_value": p, "test": "chi-square"})
    out = pd.DataFrame(rows)
    out.insert(0, "median_confidence", med_conf)
    return out


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    s = np.sqrt(((len(a) - 1) * a.std() ** 2 + (len(b) - 1) * b.std() ** 2) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0


def behavior_tables(frames: pd.DataFrame, windows: pd.DataFrame) -> dict[str, pd.DataFrame]:
    from scipy.stats import chi2_contingency, ttest_ind

    beh = [c[len("behavior_"):] for c in frames.columns if c.startswith("behavior_")]
    if not beh:
        return {}
    med = windows.confidence.median()
    sel = lambda w: frames.set_index(["recording", "window"]).index.isin(w.set_index(["recording", "window"]).index)  # noqa: E731
    success = frames[sel(windows[(windows.confidence >= med) & (windows.correct == 1)])]
    failure = frames[sel(windows[(windows.confidence >= med) & (windows.correct == 0)])]
    col = "attention_last_layer"

    presence, peaks, contrast = [], [], []
    stats = success.groupby(["recording", "window"])[col].agg(["mean", "std"]).reset_index()
    s2 = success.merge(stats, on=["recording", "window"], how="left")
    is_peak = s2[col] > s2["mean"] + s2["std"]
    for b in beh:
        c = f"behavior_{b}"
        pres, absn = success[success[c] == 1][col], success[success[c] == 0][col]
        t, p = ttest_ind(pres, absn) if len(pres) > 1 and len(absn) > 1 else (np.nan, np.nan)
        presence.append({"behavior": b, "n_present": len(pres), "n_absent": len(absn),
                         "attention_present": pres.mean(), "attention_absent": absn.mean(),
                         "difference": pres.mean() - absn.mean(), "t": t, "p_value": p,
                         "cohens_d": _cohens_d(pres, absn)})
        pk, npk = s2[is_peak][c], s2[~is_peak][c]
        table = np.array([[(pk == 1).sum(), (pk == 0).sum()], [(npk == 1).sum(), (npk == 0).sum()]])
        if table[:, 0].sum() > 0 and len(pk) and len(npk):
            chi2, pv = chi2_contingency(table)[:2]
            v = float(np.sqrt(chi2 / table.sum()))
        else:
            chi2, pv, v = np.nan, np.nan, np.nan
        peaks.append({"behavior": b, "peak_rate": pk.mean() if len(pk) else np.nan,
                      "non_peak_rate": npk.mean() if len(npk) else np.nan,
                      "difference": (pk.mean() - npk.mean()) if len(pk) and len(npk) else np.nan,
                      "n_peak_frames_with_behavior": int((pk == 1).sum()),
                      "n_other_frames_with_behavior": int((npk == 1).sum()),
                      "chi2": chi2, "p_value": pv, "cramers_v": v})
        sp, fp = success[success[c] == 1][col], failure[failure[c] == 1][col]
        t, p = ttest_ind(sp, fp) if len(sp) > 1 and len(fp) > 1 else (np.nan, np.nan)
        contrast.append({"behavior": b, "n_success": len(sp), "n_failure": len(fp),
                         "attention_success": sp.mean(), "attention_failure": fp.mean(),
                         "difference": sp.mean() - fp.mean(), "t": t, "p_value": p, "cohens_d": _cohens_d(sp, fp)})
    return {
        "behavior_attention": pd.DataFrame(presence).sort_values("cohens_d", key=abs, ascending=False),
        "behavior_peaks": pd.DataFrame(peaks).sort_values("difference", key=abs, ascending=False),
        "behavior_success_failure": pd.DataFrame(contrast).sort_values("cohens_d", key=abs, ascending=False),
    }


def run_attention_analysis(model: BehaVERT, recordings, out_dir: str | Path, stride: Optional[int] = None,
                           behaviors: Optional[dict[str, dict]] = None, batch_size: int = 64) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = collect_attention(model, recordings, stride, behaviors, batch_size)
    windows = window_table(frames)
    tables = {"attention_frames": frames, "attention_windows": windows, "attention_tests": tests_table(frames, windows)}
    tables.update(behavior_tables(frames, windows))
    for name, df in tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, float_format="%.6g")
    return tables
