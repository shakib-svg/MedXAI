#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CheXNet (DenseNet121 NIH14) + LRP (Captum)

Outputs:
  overlays/<img>__lrp_LABEL.png         overlay
  overlays/<img>__lrp_LABEL__heat.png   heat-only
  raw/<img>__lrp_LABEL__raw.png         raw (optional)
  preds_with_chexnet_lrp_sample.csv
  metrics.json
"""

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
import torch.nn.functional as F
from torchvision import models, transforms
import cv2

# ---- Captum LRP ----
try:
    from captum.attr import LRP
except Exception as e:
    raise RuntimeError(
        "Captum n'est pas installé. Fais: pip install captum\n"
        f"Erreur import: {e}"
    )

NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"


# -----------------------
# Device / torch.load compat
# -----------------------
def pick_device(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def torch_load_compat(path: str, device):
    import inspect
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=device, weights_only=False)
    return torch.load(path, map_location=device)


# -----------------------
# Labels
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


def dominant_pred_15(prob14: np.ndarray, tau_nf: float):
    mx = float(prob14.max())
    if mx < tau_nf:
        return NO_FINDING
    return NIH14[int(np.argmax(prob14))]


def dominant_gt_15(gt_labels, prob14: np.ndarray):
    if not gt_labels:
        return NO_FINDING
    best, bestp = None, -1.0
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
# Build filename index
# -----------------------
def build_filename_index(images_root: Path):
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
# CheXNet model + transform
# -----------------------
def build_chexnet():
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, 14)
    return m


TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def load_image_chexnet(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = TF(img).unsqueeze(0)  # [1,3,224,224]
    rgb = np.asarray(img).astype(np.uint8)
    rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
    return rgb, x


# -----------------------
# Checkpoint loading (robust)
# -----------------------
def _remap_key(k: str) -> str:
    for pref in ("module.", "densenet121.", "model."):
        if k.startswith(pref):
            k = k[len(pref):]

    m = re.match(r"^classifier\.(\d+)\.(weight|bias)$", k)
    if m:
        idx = int(m.group(1))
        if idx in (0, 1, 2):
            k = f"classifier.{m.group(2)}"

    k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
    k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)
    return k


def load_chexnet_checkpoint(model, ckpt_path: Path, device):
    ckpt = torch_load_compat(str(ckpt_path), device=device)

    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            sd = ckpt
    else:
        sd = ckpt

    new_sd = {_remap_key(k): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    miss = set(missing)
    if ("classifier.weight" in miss) or ("classifier.bias" in miss):
        print("[LOAD] Missing:", missing)
        print("[LOAD] Unexpected:", unexpected)
        raise RuntimeError("Classifier NOT loaded (classifier.weight/bias missing).")

    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


# -----------------------
# LRP utilities
# -----------------------
def lrp_attribution_map(lrp_obj: LRP, model: nn.Module, x: torch.Tensor, class_idx: int,
                        mode: str = "pos", smooth_sigma: float = 0.0) -> np.ndarray:
    """
    Returns a 2D map in [0,1] for visualization.
    mode:
      - "pos": positive relevance only
      - "abs": absolute relevance
      - "raw": signed relevance normalized for viz (not recommended)
    """
    # captum nécessite souvent requires_grad
    x = x.requires_grad_(True)

    # attribution shape: [1,3,224,224]
    attr = lrp_obj.attribute(x, target=class_idx)
    a = attr[0].detach().cpu().numpy().astype(np.float32)  # [3,H,W]

    # combine channels
    m = a.sum(axis=0)  # [H,W] signed

    if mode == "pos":
        m = np.maximum(m, 0.0)
    elif mode == "abs":
        m = np.abs(m)
    elif mode == "raw":
        pass
    else:
        raise ValueError("mode must be in {pos, abs, raw}")

    # optional smoothing
    if smooth_sigma and smooth_sigma > 0:
        k = int(max(3, round(6 * smooth_sigma + 1)))
        if k % 2 == 0:
            k += 1
        m = cv2.GaussianBlur(m, (k, k), smooth_sigma)

    # normalize to [0,1] robustly
    if float(m.max()) <= 1e-8:
        return np.zeros_like(m, dtype=np.float32)
    m = m - float(m.min())
    m = m / (float(m.max()) + 1e-8)
    return m.astype(np.float32)


# -----------------------
# Save heat/overlay/raw (heatmap visible)
# -----------------------
def save_heat_and_overlay(rgb_u8: np.ndarray,
                          cam01: np.ndarray,
                          out_overlay_path: Path,
                          alpha_heat: float = 0.60,
                          amplify: float = 1.0,
                          colormap=cv2.COLORMAP_TURBO):
    H, W = rgb_u8.shape[:2]
    if cam01.shape[0] != H or cam01.shape[1] != W:
        cam01 = cv2.resize(cam01, (W, H), interpolation=cv2.INTER_CUBIC)

    cam01 = np.clip(cam01, 0, 1).astype(np.float32)
    cam01 = np.clip(cam01 * float(amplify), 0, 1)

    if float(cam01.max()) <= 1e-8:
        cam_u8 = np.zeros((H, W), dtype=np.uint8)
    else:
        cam_u8 = cv2.normalize(cam01, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    heat_bgr = cv2.applyColorMap(cam_u8, colormap)

    heat_path = out_overlay_path.with_name(out_overlay_path.stem + "__heat.png")
    cv2.imwrite(str(heat_path), heat_bgr)

    img_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    a = float(np.clip(alpha_heat, 0.0, 1.0))
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - a, heat_bgr, a, 0.0)
    cv2.imwrite(str(out_overlay_path), overlay_bgr)


def save_raw_gray(cam01: np.ndarray, out_path: Path):
    cam01 = np.clip(cam01, 0, 1).astype(np.float32)
    if float(cam01.max()) <= 1e-8:
        cam_u8 = np.zeros(cam01.shape, dtype=np.uint8)
    else:
        cam_u8 = cv2.normalize(cam01, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(str(out_path), cam_u8)


# -----------------------
# Metrics
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


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--split_list", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--tau_nf", type=float, default=0.07)

    ap.add_argument("--sample_n", type=int, default=500)
    ap.add_argument("--sample_mode", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)

    # LRP settings
    ap.add_argument("--xai_on_predicted", action="store_true",
                    help="Explique les labels predits (prob>=thr). Sinon fallback top-k.")
    ap.add_argument("--topk_xai", type=int, default=1)
    ap.add_argument("--xai_min_prob", type=float, default=0.20,
                    help="Skip LRP si prob < xai_min_prob (évite cartes diffuses).")
    ap.add_argument("--lrp_mode", choices=["pos", "abs", "raw"], default="pos")
    ap.add_argument("--lrp_smooth_sigma", type=float, default=0.0)

    # Visualization
    ap.add_argument("--xai_alpha", type=float, default=0.60)
    ap.add_argument("--xai_amplify", type=float, default=2.0)
    ap.add_argument("--save_raw", action="store_true")

    ap.add_argument("--debug_first", type=int, default=5)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    device = pick_device(force_cpu=args.cpu)
    print("[INFO] Device:", device)
    print(f"[INFO] thr={args.thr} xai_on_predicted={args.xai_on_predicted} topk_xai={args.topk_xai} xai_min_prob={args.xai_min_prob}")
    print(f"[INFO] lrp_mode={args.lrp_mode} smooth_sigma={args.lrp_smooth_sigma}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    overlays_dir = out_dir / "overlays"
    raw_dir = out_dir / "raw"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw:
        raw_dir.mkdir(parents=True, exist_ok=True)

    # sample split list
    split_path = Path(args.split_list)
    all_fns = [ln.strip() for ln in split_path.read_text().splitlines() if ln.strip()]
    if args.sample_mode == "first":
        sampled = all_fns[:args.sample_n]
    else:
        rng = np.random.default_rng(args.seed)
        n = min(args.sample_n, len(all_fns))
        sampled = list(rng.choice(all_fns, size=n, replace=False))
    sampled = [str(x) for x in sampled]
    sampled_set = set(sampled)
    print(f"[INFO] sampled {len(sampled)} images.")

    # csv filter
    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]
    img_col = "Image Index"
    lab_col = "Finding Labels"
    df = df[df[img_col].astype(str).isin(sampled_set)]
    order = {fn: i for i, fn in enumerate(sampled)}
    df["__ord"] = df[img_col].astype(str).map(order)
    df = df.sort_values("__ord").drop(columns=["__ord"]).reset_index(drop=True)
    print(f"[INFO] CSV filtered rows: {len(df)}")

    # image index
    idx = build_filename_index(Path(args.images_root))
    print(f"[INFO] filename index size: {len(idx)}")
    if len(idx) == 0:
        raise RuntimeError("Empty image index: check --images_root")

    # model + LRP object
    model = build_chexnet()
    model = load_chexnet_checkpoint(model, Path(args.ckpt), device=device).to(device)
    model.eval()
    lrp = LRP(model)

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

        rgb_u8, x = load_image_chexnet(Path(img_path))
        x = x.to(device)

        # forward for probs/metrics
        with torch.no_grad():
            logits = model(x)
            prob = torch.sigmoid(logits)[0].detach().cpu().numpy().astype(np.float32)

        if i < args.debug_first:
            print(f"[DBG] {fn} prob min/mean/max = {prob.min():.4f} {prob.mean():.4f} {prob.max():.4f}")

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

        # choose classes to explain
        if args.xai_on_predicted and len(pred_idx) > 0:
            pred_sorted = sorted(pred_idx, key=lambda j: float(prob[j]), reverse=True)
            xai_classes = pred_sorted[:min(args.topk_xai, len(pred_sorted))]
        else:
            k = min(args.topk_xai, 14)
            xai_classes = np.argsort(-prob)[:k].tolist()

        overlay_files, heat_files, raw_files = [], [], []

        # LRP per selected class (skip low prob)
        for j in xai_classes:
            if float(prob[j]) < args.xai_min_prob:
                continue

            # LRP needs grads => no torch.no_grad here
            x_lrp = x.detach().clone()
            m01 = lrp_attribution_map(
                lrp_obj=lrp,
                model=model,
                x=x_lrp,
                class_idx=j,
                mode=args.lrp_mode,
                smooth_sigma=args.lrp_smooth_sigma
            )

            base = f"{Path(fn).stem}__lrp_{NIH14[j]}"
            overlay_path = overlays_dir / f"{base}.png"
            save_heat_and_overlay(
                rgb_u8=rgb_u8,
                cam01=m01,
                out_overlay_path=overlay_path,
                alpha_heat=args.xai_alpha,
                amplify=args.xai_amplify,
                colormap=cv2.COLORMAP_TURBO
            )
            overlay_files.append(f"{base}.png")
            heat_files.append(f"{base}__heat.png")

            if args.save_raw:
                raw_path = raw_dir / f"{base}__raw.png"
                save_raw_gray(m01, raw_path)
                raw_files.append(f"{base}__raw.png")

        prob_cols = {f"p_{lab}": float(prob[k]) for k, lab in enumerate(NIH14)}

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
            "Overlay_files": "|".join(overlay_files),
            "Heat_files": "|".join(heat_files),
            "Raw_files": "|".join(raw_files),
            "probs_top5": format_top5(prob),
            "GT_vec_sum": int(gt_vec.sum()),
            "n_pred@thr": int(len(pred_idx)),
            **prob_cols,
        })

        if (i + 1) % 25 == 0:
            print(f"[INFO] processed {i+1}/{len(df)}...")

    if not y_true:
        raise RuntimeError("0 image processed. Check split_list/csv/images_root matching.")

    y_true = np.stack(y_true, axis=0).astype(np.int32)
    y_prob = np.stack(y_prob, axis=0).astype(np.float32)

    m = multilabel_metrics(y_true, y_prob, thr=args.thr)

    print("\n[MULTILABEL NIH14 METRICS]")
    print(f"thr={args.thr:.3f}")
    print(f"micro  P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"macro  P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} F1={m['macro_f1']:.3f}")
    print(f"macro AUROC={m['macro_auroc']:.3f} | macro mAP={m['macro_map']:.3f}")

    df_out = pd.DataFrame(rows)
    out_csv = out_dir / "preds_with_chexnet_lrp_sample.csv"
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

    print(f"\n[DONE] CSV: {out_csv}")
    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] overlays: {overlays_dir}")
    if args.save_raw:
        print(f"[DONE] raw: {raw_dir}")
    print(f"[INFO] processed={len(df_out)} skipped_missing={skipped_missing} elapsed={payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
"""
CUDA_VISIBLE_DEVICES=0 python3 chexnet_lrp_500.py \
  --ckpt /home/shakib/Desktop/S5/projet/chexnet/models/m-25012018-123527.pth.tar \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --out_dir /home/shakib/Desktop/S5/projet/xai_outputs/chexnet_lrp_random500_seed0 \
  --sample_n 500 --sample_mode random --seed 0 \
  --thr 0.5 \
  --xai_on_predicted --topk_xai 2 \
  --xai_min_prob 0.20 \
  --lrp_mode pos \
  --xai_alpha 0.65 --xai_amplify 2.0 \
  --save_raw






"""