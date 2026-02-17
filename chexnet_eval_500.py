#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import time
import json
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# -----------------------
# Labels NIH14 + No Finding
# -----------------------
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"


# -----------------------
# Device / load utils
# -----------------------
def pick_device(force_cpu: bool = False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _torch_load_safe(path: str, device):
    """
    Safe checkpoint load across torch versions.
    """
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


# -----------------------
# NIH labels parsing (same spirit as your PYLON pipeline)
# - "No Finding" => empty list (normal image)
# -----------------------
def parse_labels(label_str: str):
    s = str(label_str).strip()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return []
    parts = [p.replace("Pleural Thickening", "Pleural_Thickening") for p in parts]
    if len(parts) == 1 and parts[0] == NO_FINDING:
        return []
    return parts


def labels_to_vec(labels_list):
    v = np.zeros(len(NIH14), dtype=np.int32)
    for lab in labels_list:
        if lab in NIH14:
            v[NIH14.index(lab)] = 1
    return v


# -----------------------
# Build filename index (same as PYLON)
# -----------------------
def build_filename_index(images_root: Path):
    """
    Index NIH archive:
      - images_*/images/*.(png/jpg/jpeg)
      - images/*.(png/jpg/jpeg)
      - fallback: recursive
    map: basename -> full path (first found)
    """
    exts = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}
    idx = {}
    images_root = images_root.resolve()

    for p in images_root.glob("images_*/images/*"):
        if p.is_file() and p.suffix in exts:
            idx.setdefault(p.name, p.resolve())

    images_dir = images_root / "images"
    if images_dir.exists():
        for p in images_dir.glob("*"):
            if p.is_file() and p.suffix in exts:
                idx.setdefault(p.name, p.resolve())

    if len(idx) == 0:
        for p in images_root.rglob("*"):
            if p.is_file() and p.suffix in exts:
                idx.setdefault(p.name, p.resolve())

    return idx


# -----------------------
# CheXNet model + checkpoint loading
# -----------------------
def build_chexnet(num_classes=14):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def load_chexnet_checkpoint(model, ckpt_path: Path, device):
    """
    Handles common CheXNet checkpoints: dict with 'state_dict' or raw state_dict,
    with possible prefixes: module., module.densenet121., etc.
    """
    ckpt = _torch_load_safe(str(ckpt_path), device=device)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.densenet121."):
            k = k.replace("module.densenet121.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        # Fix some older naming patterns (seen in some repos)
        k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
        k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)

        if k.startswith("classifier.0."):
            k = k.replace("classifier.0.", "classifier.", 1)

        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


# -----------------------
# CheXNet input transform (standard ImageNet normalization)
# -----------------------
TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # [0,1], 3-ch
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image_chexnet(img_path: Path):
    """
    CheXNet expects 3-channel input.
    NIH images can be grayscale; we convert to RGB (3 identical channels).
    Returns x: [1,3,224,224] float tensor.
    """
    img = Image.open(img_path).convert("RGB")
    x = TF(img).unsqueeze(0)
    return x


# -----------------------
# Metrics (same as your PYLON script)
# -----------------------
def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    y_pred = (y_prob >= thr).astype(np.int32)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    p_micro = tp / (tp + fp) if (tp + fp) else 0.0
    r_micro = tp / (tp + fn) if (tp + fn) else 0.0
    f1_micro = 2 * p_micro * r_micro / (p_micro + r_micro) if (p_micro + r_micro) else 0.0

    ps, rs, f1s = [], [], []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tpj = int(((yp == 1) & (yt == 1)).sum())
        fpj = int(((yp == 1) & (yt == 0)).sum())
        fnj = int(((yp == 0) & (yt == 1)).sum())
        pj = tpj / (tpj + fpj) if (tpj + fpj) else 0.0
        rj = tpj / (tpj + fnj) if (tpj + fnj) else 0.0
        f1j = 2 * pj * rj / (pj + rj) if (pj + rj) else 0.0
        ps.append(pj); rs.append(rj); f1s.append(f1j)

    p_macro = float(np.mean(ps)) if ps else 0.0
    r_macro = float(np.mean(rs)) if rs else 0.0
    f1_macro = float(np.mean(f1s)) if f1s else 0.0

    macro_auc = float("nan")
    macro_ap = float("nan")
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        aucs, aps = [], []
        for j in range(y_true.shape[1]):
            yt = y_true[:, j]
            yp = y_prob[:, j]
            if yt.max() == yt.min():
                continue
            aucs.append(float(roc_auc_score(yt, yp)))
            aps.append(float(average_precision_score(yt, yp)))
        if aucs:
            macro_auc = float(np.mean(aucs))
        if aps:
            macro_ap = float(np.mean(aps))
    except Exception:
        pass

    return {
        "micro_precision": p_micro,
        "micro_recall": r_micro,
        "micro_f1": f1_micro,
        "macro_precision": p_macro,
        "macro_recall": r_macro,
        "macro_f1": f1_macro,
        "macro_auroc": macro_auc,
        "macro_map": macro_ap,
    }


def dominant_pred_15(prob14: np.ndarray, tau_nf: float):
    mx = float(prob14.max())
    if mx < tau_nf:
        return NO_FINDING
    return NIH14[int(np.argmax(prob14))]


def dominant_gt_15(gt_labels, prob14: np.ndarray):
    if not gt_labels:
        return NO_FINDING
    best = None
    bestp = -1.0
    for lab in gt_labels:
        if lab in NIH14:
            p = float(prob14[NIH14.index(lab)])
            if p > bestp:
                bestp = p
                best = lab
    return best if best is not None else NO_FINDING


def format_top5(prob14: np.ndarray):
    idx = np.argsort(-prob14)[:5]
    return ",".join([f"{NIH14[i]}:{prob14[i]:.4f}" for i in idx])


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="CheXNet checkpoint path (.pth/.pth.tar)")
    ap.add_argument("--csv", required=True, help="NIH Data_Entry_2017.csv (or compatible)")
    ap.add_argument("--images_root", required=True, help="NIH archive root (contains images_*/images)")
    ap.add_argument("--split_list", required=True, help="test_list.txt (one filename per line)")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--thr", type=float, default=0.1, help="multilabel threshold")
    ap.add_argument("--tau_nf", type=float, default=0.07, help="No Finding if max(prob14)<tau_nf")

    ap.add_argument("--sample_n", type=int, default=500)
    ap.add_argument("--sample_mode", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(force_cpu=args.cpu)
    print("[INFO] Device:", device)
    print(f"[INFO] thr={args.thr} tau_nf={args.tau_nf} sample_n={args.sample_n} mode={args.sample_mode} seed={args.seed}")

    # Load split list and sample
    split_path = Path(args.split_list)
    all_fns = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
    print(f"[INFO] split_list loaded: {split_path} (n={len(all_fns)})")

    if args.sample_mode == "first":
        sampled = all_fns[:args.sample_n]
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.sample_n, len(all_fns))
        sampled = list(rng.choice(all_fns, size=n, replace=False))
    sampled_set = set(sampled)
    print(f"[INFO] sampled {len(sampled)} images.")

    # Read CSV
    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]
    img_col = "Image Index" if "Image Index" in df.columns else None
    lab_col = "Finding Labels" if "Finding Labels" in df.columns else None
    if img_col is None or lab_col is None:
        raise ValueError("CSV must contain columns: 'Image Index' and 'Finding Labels'.")

    # Filter rows to sampled (preserve sampled order)
    df = df[df[img_col].astype(str).isin(sampled_set)]
    order = {fn: i for i, fn in enumerate(sampled)}
    df["__ord"] = df[img_col].astype(str).map(order)
    df = df.sort_values("__ord").drop(columns=["__ord"]).reset_index(drop=True)
    print(f"[INFO] CSV filtered rows: {len(df)}")

    # Build file index
    images_root = Path(args.images_root)
    print("[INFO] building filename index ...")
    idx = build_filename_index(images_root)
    print(f"[INFO] index size: {len(idx)}")

    # Build model
    model = build_chexnet(num_classes=14)
    model = load_chexnet_checkpoint(model, Path(args.ckpt), device=device).to(device)
    model.eval()

    rows = []
    y_true, y_prob = [], []
    skipped_missing = 0

    for i in range(len(df)):
        fn = str(df.loc[i, img_col]).strip()
        gt_labels = parse_labels(df.loc[i, lab_col])
        gt_vec = labels_to_vec(gt_labels)

        img_path = idx.get(fn, None)
        if img_path is None or not Path(img_path).exists():
            skipped_missing += 1
            continue

        x = load_image_chexnet(Path(img_path)).to(device)

        with torch.no_grad():
            logits = model(x)  # [1,14]
            prob = torch.sigmoid(logits)[0].detach().cpu().numpy()

        y_true.append(gt_vec)
        y_prob.append(prob)

        pred_idx = np.where(prob >= args.thr)[0].tolist()
        pred_labels_thr = [NIH14[j] for j in pred_idx]
        pred_labels_thr_str = "|".join(pred_labels_thr) if pred_labels_thr else NO_FINDING

        dom_pred = dominant_pred_15(prob, tau_nf=args.tau_nf)
        dom_gt = dominant_gt_15(gt_labels, prob)

        top1_j = int(np.argmax(prob))
        top1_name = NIH14[top1_j]
        top1_score = float(prob[top1_j])

        rows.append({
            "Image": fn,
            "img_path": str(img_path),
            "GT_labels": "|".join(gt_labels) if gt_labels else NO_FINDING,
            "Pred_labels@thr": pred_labels_thr_str,
            "thr": args.thr,
            "tau_nf": args.tau_nf,
            "Dominant_GT_15": dom_gt,
            "Dominant_Pred_15": dom_pred,
            "Top1_disease": top1_name,
            "Top1_score": top1_score,
            "probs_top5": format_top5(prob),
            "GT_vec_sum": int(gt_vec.sum()),
        })

        if (i + 1) % 25 == 0:
            print(f"[INFO] processed {i+1}/{len(df)}...")

    if not y_true:
        raise RuntimeError("0 image processed. Check split_list/csv/images_root.")

    y_true = np.stack(y_true, axis=0).astype(np.int32)
    y_prob = np.stack(y_prob, axis=0).astype(np.float32)

    m = multilabel_metrics(y_true, y_prob, thr=args.thr)

    print("\n[MULTILABEL NIH14 METRICS]")
    print(f"thr={args.thr:.3f}")
    print(f"micro  P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"macro  P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} F1={m['macro_f1']:.3f}")
    print(f"macro AUROC={m['macro_auroc']:.3f} | macro mAP={m['macro_map']:.3f}")

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "preds_with_chexnet_sample.csv"
    df_out.to_csv(out_csv, index=False)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "multilabel": m,
        "args": vars(args),
        "processed": int(len(df_out)),
        "skipped_missing": int(skipped_missing),
        "elapsed_sec": float(time.time() - t0),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(f"\n[DONE] Saved CSV: {out_csv}")
    print(f"[DONE] Saved metrics: {metrics_path}")
    print(f"[INFO] processed={len(df_out)} skipped_missing={skipped_missing} elapsed={payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
"""
/home/shakib/Desktop/S5/projet/chexnet/models/m-25012018-123527.pth.tar

CUDA_VISIBLE_DEVICES= \
python3 chexnet_eval_500.py \
  --ckpt /home/shakib/Desktop/S5/projet/chexnet/models/m-25012018-123527.pth.tar \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --out_dir /home/shakib/Desktop/S5/projet/xai_outputs/chexnet_eval_random500_seed0 \
  --thr 0.1 \
  --tau_nf 0.07 \
  --sample_n 500 --sample_mode random --seed 0 \
  --cpu



"""