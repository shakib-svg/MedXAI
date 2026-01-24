#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

import cv2


# NIH14 classes (CheXNet output order)
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)


# -----------------------------
# Model
# -----------------------------
def build_chexnet(num_classes=14):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def _torch_load_safe(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_chexnet_checkpoint(model, ckpt_path):
    ckpt = _torch_load_safe(ckpt_path)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.densenet121."):
            k = k.replace("module.densenet121.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
        k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)

        if k.startswith("classifier.0."):
            k = k.replace("classifier.0.", "classifier.", 1)

        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


# -----------------------------
# Utility: overlay heatmap
# -----------------------------
def overlay_heatmap_on_rgb(rgb_224, heat_224, alpha=0.4):
    """
    rgb_224: float [H,W,3] in [0,1], RGB
    heat_224: float [H,W] in [0,1]
    returns: BGR uint8 for cv2.imwrite
    """
    heat = (heat_224 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip((1 - alpha) * rgb_224 + alpha * heat, 0, 1)
    return (overlay[..., ::-1] * 255).astype(np.uint8)


# -----------------------------
# Labels + metrics helpers
# -----------------------------
def parse_labels(label_str: str):
    s = str(label_str).strip()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return [NO_FINDING]
    parts = [p.replace("Pleural Thickening", "Pleural_Thickening") for p in parts]
    return parts


def labels_to_vec(labels_list):
    v = np.zeros(len(NIH14), dtype=np.int32)
    for lab in labels_list:
        if lab in NIH14:
            v[NIH14.index(lab)] = 1
    return v


def micro_precision_recall_f1(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


def single_label_metrics(gt_ids, pr_ids, class_names):
    gt = np.asarray(gt_ids, dtype=np.int64)
    pr = np.asarray(pr_ids, dtype=np.int64)
    K = len(class_names)

    acc = float((gt == pr).mean()) if len(gt) else float("nan")

    f1s = []
    per = {}
    for c in range(K):
        tp = int(((gt == c) & (pr == c)).sum())
        fp = int(((gt != c) & (pr == c)).sum())
        fn = int(((gt == c) & (pr != c)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        per[class_names[c]] = {"precision": prec, "recall": rec, "f1": f1, "support": int((gt == c).sum())}
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return acc, macro_f1, per


# -----------------------------
# Explain policy (Pred vs GT)
# -----------------------------
def choose_target_classes(explain_mode, gt_primary, pred_primary, top1_idx):
    """
    Returns list of class indices in NIH14 to explain.
    If label is No Finding -> fallback to top1_idx (best proxy).
    """
    def idx_from_label(lab: str):
        return NIH14.index(lab) if lab in NIH14 else None

    targets = []

    if explain_mode == "pred":
        if pred_primary == NO_FINDING:
            targets = [top1_idx]
        else:
            targets = [idx_from_label(pred_primary)]

    elif explain_mode == "gt":
        if gt_primary == NO_FINDING:
            targets = [top1_idx]
        else:
            targets = [idx_from_label(gt_primary)]

    else:  # hybrid
        # always explain Pred
        if pred_primary == NO_FINDING:
            targets.append(top1_idx)
        else:
            targets.append(idx_from_label(pred_primary))

        # if wrong, also explain GT (if pathology)
        if pred_primary != gt_primary and gt_primary in NIH14:
            targets.append(idx_from_label(gt_primary))

    targets = [t for t in targets if t is not None]
    seen = set()
    targets = [t for t in targets if not (t in seen or seen.add(t))]
    return targets


# -----------------------------
# Integrated Gradients
# -----------------------------
def make_baseline(x: torch.Tensor, baseline: str):
    """
    x: normalized input [1,3,224,224]
    baseline:
      - "zero": zeros in normalized space (recommended)
      - "black": raw black image normalized -> (0-mean)/std
    """
    if baseline == "zero":
        return torch.zeros_like(x)
    elif baseline == "black":
        # raw black (0) -> normalized: (0 - mean)/std
        device = x.device
        mean = IMAGENET_MEAN.to(device).view(1, 3, 1, 1)
        std = IMAGENET_STD.to(device).view(1, 3, 1, 1)
        return ((torch.zeros_like(x) * std) + 0.0 - mean) / std  # equivalent to -mean/std
    else:
        raise ValueError("baseline must be 'zero' or 'black'")


@torch.no_grad()
def get_probs(model, x):
    return torch.sigmoid(model(x))[0].detach().cpu().numpy()


def integrated_gradients(model, x, target_idx: int, baseline="zero", ig_steps=50):
    """
    x: normalized input [1,3,224,224]
    returns:
      - attr_map: [224,224] in [0,1] (saliency from IG)
    """
    device = x.device
    x0 = make_baseline(x, baseline).to(device)

    # scaled inputs: x0 + alpha*(x-x0)
    alphas = torch.linspace(0.0, 1.0, steps=ig_steps, device=device).view(-1, 1, 1, 1)

    # accumulate gradients
    total_grad = torch.zeros_like(x, device=device)

    for a in alphas:
        xi = x0 + a * (x - x0)
        xi.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(xi)
        score = logits[:, target_idx].sum()

        grads = torch.autograd.grad(score, xi, retain_graph=False, create_graph=False)[0]
        total_grad += grads.detach()

    avg_grad = total_grad / float(ig_steps)
    ig = (x - x0) * avg_grad  # [1,3,224,224]

    # Convert to 2D saliency: sum abs over channels
    sal = ig[0].abs().sum(dim=0)  # [224,224]
    sal = sal.detach().cpu().numpy()

    # normalize to [0,1]
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    return sal


# -----------------------------
# XAI metrics: deletion / insertion AUC
# -----------------------------
def deletion_insertion_auc(model, x, class_idx, heat_224, steps=10, mode="deletion"):
    device = x.device
    x0 = x.detach().clone()

    flat = heat_224.flatten()
    order = np.argsort(-flat)  # descending
    H, W = heat_224.shape
    total = H * W

    scores = []
    for s in range(steps + 1):
        k = int((s / steps) * total)
        mask = np.zeros(total, dtype=np.float32)
        mask[order[:k]] = 1.0
        mask = mask.reshape(H, W)

        mask_t = torch.tensor(mask, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_t = mask_t.repeat(1, 3, 1, 1)

        if mode == "insertion":
            current = x0 * mask_t
        else:
            current = x0 * (1 - mask_t)

        with torch.no_grad():
            p = torch.sigmoid(model(current))[0, class_idx].item()
        scores.append(p)

    x_axis = np.linspace(0, 1, len(scores))
    auc = float(np.trapz(scores, x_axis))
    return auc


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images_dir", default=None)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--task", choices=["single_label", "multilabel"], default="single_label")
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--nf_threshold", type=float, default=0.10)

    ap.add_argument("--explain", choices=["pred", "gt", "hybrid"], default="pred")

    # IG params
    ap.add_argument("--ig_steps", type=int, default=50, help="nb pas pour Integrated Gradients")
    ap.add_argument("--baseline", choices=["zero", "black"], default="zero",
                    help="baseline IG: zero (reco) ou black")

    # Output / XAI eval
    ap.add_argument("--alpha", type=float, default=0.4)
    ap.add_argument("--steps", type=int, default=10, help="steps deletion/insertion")

    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    (out_dir / "overlays").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]

    img_col = "Image Index" if "Image Index" in df.columns else ("Image_Index" if "Image_Index" in df.columns else None)
    lab_col = "Finding Labels" if "Finding Labels" in df.columns else ("Finding_Labels" if "Finding_Labels" in df.columns else None)
    if img_col is None or lab_col is None:
        raise ValueError("CSV: colonnes attendues introuvables (Image Index / Finding Labels).")

    has_src = "src_path" in df.columns
    csv_dir = Path(args.csv).expanduser().resolve().parent
    base_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else csv_dir

    model = build_chexnet(14)
    model = load_chexnet_checkpoint(model, args.ckpt).to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ])

    rows_out = []

    # Model bookkeeping
    y_true_all = []
    y_pred_all = []
    nofinding_total = 0
    nofinding_fp = 0

    SL_CLASSES = NIH14 + [NO_FINDING]
    sl_gt, sl_pred = [], []

    for _, row in df.iterrows():
        img_name = str(row[img_col]).strip()
        gt_str = str(row[lab_col]).strip()
        gt_labels = parse_labels(gt_str)
        gt_primary = gt_labels[0] if gt_labels else NO_FINDING

        # path
        if has_src:
            p = Path(str(row["src_path"]).strip())
            img_path = p if p.is_absolute() else (base_dir / p).resolve()
        else:
            img_path = (base_dir / img_name).resolve()

        if not img_path.exists():
            print(f"[SKIP] not found: {img_path}")
            continue

        raw = Image.open(img_path).convert("RGB")
        x = tf(raw).unsqueeze(0).to(device)

        probs = get_probs(model, x)
        top1_idx = int(np.argmax(probs))
        top1_score = float(np.max(probs))

        # predict
        if args.task == "multilabel":
            pred_idx = np.where(probs >= args.threshold)[0].tolist()
            pred_labels = [NIH14[j] for j in pred_idx]
            pred_primary = pred_labels[0] if pred_labels else NO_FINDING
        else:
            if top1_score < args.nf_threshold:
                pred_primary = NO_FINDING
                pred_idx = []
            else:
                pred_primary = NIH14[top1_idx]
                pred_idx = [top1_idx]

        # bookkeeping
        gt_vec = labels_to_vec(gt_labels)
        pred_vec = np.zeros(len(NIH14), dtype=np.int32)
        for j in pred_idx:
            pred_vec[j] = 1
        y_true_all.append(gt_vec)
        y_pred_all.append(pred_vec)

        if gt_primary == NO_FINDING:
            nofinding_total += 1
            if len(pred_idx) > 0:
                nofinding_fp += 1

        gt_id = NIH14.index(gt_primary) if gt_primary in NIH14 else len(NIH14)
        pr_id = NIH14.index(pred_primary) if pred_primary in NIH14 else len(NIH14)
        sl_gt.append(gt_id)
        sl_pred.append(pr_id)

        # targets to explain
        targets = choose_target_classes(args.explain, gt_primary, pred_primary, top1_idx)

        # overlay base RGB
        raw_224 = raw.resize((224, 224))
        rgb_np = np.asarray(raw_224).astype(np.float32) / 255.0

        overlay_files = []
        del_aucs, ins_aucs = [], []

        for ti in targets:
            # IG heatmap
            heat = integrated_gradients(model, x, ti, baseline=args.baseline, ig_steps=args.ig_steps)
            overlay = overlay_heatmap_on_rgb(rgb_np, heat, alpha=args.alpha)

            out_img = out_dir / "overlays" / f"{Path(img_name).stem}__{args.explain}__ig_{NIH14[ti]}.png"
            cv2.imwrite(str(out_img), overlay)
            overlay_files.append(out_img.name)

            del_auc = deletion_insertion_auc(model, x, ti, heat, steps=args.steps, mode="deletion")
            ins_auc = deletion_insertion_auc(model, x, ti, heat, steps=args.steps, mode="insertion")
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)

        rows_out.append({
            "Image": img_name,
            "GT": gt_primary,
            "Pred": pred_primary,
            "Top1": NIH14[top1_idx],
            "Top1_score": top1_score,
            "explain": args.explain,
            "method": "integrated_gradients",
            "baseline": args.baseline,
            "ig_steps": int(args.ig_steps),
            "Targets": "|".join([NIH14[t] for t in targets]) if targets else "",
            "DeletionAUC_mean": float(np.mean(del_aucs)) if del_aucs else np.nan,
            "InsertionAUC_mean": float(np.mean(ins_aucs)) if ins_aucs else np.nan,
            "Overlay_Files": "|".join(overlay_files),
            "img_path": str(img_path),
        })

        print(f"[{len(rows_out)}/{len(df)}] {img_name} | GT={gt_primary} | Pred={pred_primary} | explain={args.explain} | IG baseline={args.baseline}")

    if len(rows_out) == 0:
        print("\n[ERROR] 0 image traitée. Vérifie src_path / base_dir.")
        return

    results = pd.DataFrame(rows_out)
    out_csv = out_dir / "results_integrated_gradients.csv"
    results.to_csv(out_csv, index=False)

    print("\n[DONE] Saved overlays ->", (out_dir / "overlays"))
    print("[DONE] Saved results ->", out_csv)

    # metrics
    y_true = np.stack(y_true_all, axis=0) if y_true_all else np.zeros((0, 14), dtype=np.int32)
    y_pred = np.stack(y_pred_all, axis=0) if y_pred_all else np.zeros((0, 14), dtype=np.int32)
    prec, rec, f1 = micro_precision_recall_f1(y_true, y_pred)
    fpr_nf = (nofinding_fp / (nofinding_total + 1e-9)) if nofinding_total > 0 else np.nan

    print("\n[MODEL METRICS]")
    print(f"Micro-Precision (multilabel view): {prec:.3f} | Micro-Recall: {rec:.3f} | Micro-F1: {f1:.3f}")
    print(f"NoFinding FPR (>=1 predicted label): {fpr_nf:.3f}  (on {nofinding_total} No Finding images)")

    if args.task == "single_label":
        acc, macro_f1, per = single_label_metrics(sl_gt, sl_pred, SL_CLASSES)
        print("\n[SINGLE-LABEL METRICS]")
        print(f"Accuracy: {acc:.3f} | Macro-F1: {macro_f1:.3f}")

    print("\n[XAI METRICS (mean over explained classes)]")
    if results["DeletionAUC_mean"].notna().any():
        print(f"Mean DeletionAUC: {results['DeletionAUC_mean'].mean():.3f}")
    else:
        print("Mean DeletionAUC: n/a")
    if results["InsertionAUC_mean"].notna().any():
        print(f"Mean InsertionAUC: {results['InsertionAUC_mean'].mean():.3f}")
    else:
        print("Mean InsertionAUC: n/a")


if __name__ == "__main__":
    main()
