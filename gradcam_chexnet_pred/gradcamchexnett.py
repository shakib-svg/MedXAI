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


# -----------------------------
# Model
# -----------------------------
def build_chexnet(num_classes=14):
    m = models.densenet121(weights=None)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def _torch_load_safe(path: str):
    """
    Try to load with weights_only=True (new torch). Fallback otherwise.
    """
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_chexnet_checkpoint(model, ckpt_path):
    ckpt = _torch_load_safe(ckpt_path)

    # CheXNet commonly stores {"state_dict": ...}
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module.densenet121."):
            k = k.replace("module.densenet121.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        # Fix older naming patterns
        k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
        k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)

        if k.startswith("classifier.0."):
            k = k.replace("classifier.0.", "classifier.", 1)

        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


# -----------------------------
# Grad-CAM
# -----------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.h1 = target_layer.register_forward_hook(self._fh)
        self.h2 = target_layer.register_full_backward_hook(self._bh)

    def _fh(self, m, inp, out):
        self.activations = out

    def _bh(self, m, gin, gout):
        self.gradients = gout[0]

    def __call__(self, x, class_idx: int):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)                # [1,14] (raw logits)
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (w * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


def overlay_cam_on_rgb(rgb_224, cam_224, alpha=0.4):
    """
    rgb_224: float [H,W,3] in [0,1], RGB
    cam_224: float [H,W] in [0,1]
    returns: BGR uint8 for cv2.imwrite
    """
    heat = (cam_224 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip((1 - alpha) * rgb_224 + alpha * heat, 0, 1)
    return (overlay[..., ::-1] * 255).astype(np.uint8)  # BGR uint8


# -----------------------------
# Labels + metrics helpers
# -----------------------------
def parse_labels(label_str: str):
    s = str(label_str).strip()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return [NO_FINDING]
    # normalize typical NIH spelling
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
        per[class_names[c]] = {
            "precision": prec, "recall": rec, "f1": f1, "support": int((gt == c).sum())
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else float("nan")
    return acc, macro_f1, per


# -----------------------------
# Simple XAI metrics: deletion / insertion AUC
# -----------------------------
def deletion_insertion_auc(model, x, class_idx, cam_224, steps=10, mode="deletion"):
    """
    deletion: mask top-cam pixels to 0
    insertion: start from black, insert top-cam pixels
    """
    device = x.device
    x0 = x.detach().clone()

    flat = cam_224.flatten()
    order = np.argsort(-flat)
    H, W = cam_224.shape
    total = H * W

    scores = []
    for s in range(steps + 1):
        k = int((s / steps) * total)
        mask = np.zeros(total, dtype=np.float32)
        mask[order[:k]] = 1.0
        mask = mask.reshape(H, W)

        mask_t = torch.tensor(mask, device=device).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        mask_t = mask_t.repeat(1, 3, 1, 1)  # [1,3,H,W]

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
# Explain policy (Pred vs GT)
# -----------------------------
def choose_cam_classes(explain_mode, gt_primary, pred_primary, top1_idx):
    """
    Returns list of class indices (in NIH14 index space) to explain.
    - If label is No Finding, we fall back to top1_idx (best proxy).
    """
    def idx_from_label(lab: str):
        return NIH14.index(lab) if lab in NIH14 else None

    cam = []

    if explain_mode == "pred":
        if pred_primary == NO_FINDING:
            cam = [top1_idx]
        else:
            cam = [idx_from_label(pred_primary)]

    elif explain_mode == "gt":
        if gt_primary == NO_FINDING:
            cam = [top1_idx]
        else:
            cam = [idx_from_label(gt_primary)]

    else:  # hybrid
        # always explain Pred
        if pred_primary == NO_FINDING:
            cam.append(top1_idx)
        else:
            cam.append(idx_from_label(pred_primary))

        # if wrong, also explain GT (if pathology)
        if pred_primary != gt_primary and gt_primary in NIH14:
            cam.append(idx_from_label(gt_primary))

    # remove None and duplicates preserving order
    cam = [c for c in cam if c is not None]
    seen = set()
    cam = [c for c in cam if not (c in seen or seen.add(c))]
    return cam


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Chemin checkpoint CheXNet .pth.tar")
    ap.add_argument("--csv", required=True, help="CSV subset (50 images)")
    ap.add_argument("--images_dir", default=None,
                    help="Base images si src_path absent OU pour résoudre src_path relatif (optionnel)")
    ap.add_argument("--out_dir", required=True, help="Sorties: overlays + results.csv")

    ap.add_argument("--task", choices=["single_label", "multilabel"], default="single_label",
                    help="Ton dataset est single-label: utilise single_label")
    ap.add_argument("--threshold", type=float, default=0.1, help="Seuil multi-label (task=multilabel)")
    ap.add_argument("--nf_threshold", type=float, default=0.10,
                    help="(task=single_label) si max(proba)<nf_threshold => Pred='No Finding'")

    ap.add_argument("--explain", choices=["pred", "gt", "hybrid"], default="pred",
                    help="Règle d’explication: pred | gt | hybrid (pred + gt si erreur)")
    ap.add_argument("--alpha", type=float, default=0.4, help="Alpha overlay heatmap")
    ap.add_argument("--steps", type=int, default=10, help="Steps insertion/deletion")

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

    # Robust base dir: csv_dir by default, can be overridden by --images_dir
    csv_dir = Path(args.csv).expanduser().resolve().parent
    base_dir = Path(args.images_dir).expanduser().resolve() if args.images_dir else csv_dir

    model = build_chexnet(14)
    model = load_chexnet_checkpoint(model, args.ckpt).to(device).eval()

    # DenseNet121 last block is a common Grad-CAM target
    gc = GradCAM(model, model.features.denseblock4)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    rows_out = []

    # Multi-label bookkeeping
    y_true_all = []
    y_pred_all = []
    nofinding_total = 0
    nofinding_fp = 0

    # Single-label bookkeeping (add No Finding as extra class)
    SL_CLASSES = NIH14 + [NO_FINDING]
    sl_gt = []
    sl_pred = []

    for _, row in df.iterrows():
        img_name = str(row[img_col]).strip()
        gt_str = str(row[lab_col]).strip()
        gt_labels = parse_labels(gt_str)
        gt_primary = gt_labels[0] if len(gt_labels) else NO_FINDING

        # Resolve path
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

        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].detach().cpu().numpy()

        top1_idx = int(np.argmax(probs))
        top1_score = float(np.max(probs))

        # ----- Predict -----
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

        # ----- Metrics bookkeeping -----
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

        # ----- Explain policy (Pred vs GT) -----
        cam_classes = choose_cam_classes(args.explain, gt_primary, pred_primary, top1_idx)

        # Prepare RGB 224 for overlay
        raw_224 = raw.resize((224, 224))
        rgb_np = np.asarray(raw_224).astype(np.float32) / 255.0

        overlay_files = []
        del_aucs = []
        ins_aucs = []

        for ci in cam_classes:
            cam = gc(x, ci)
            overlay = overlay_cam_on_rgb(rgb_np, cam, alpha=args.alpha)

            # include explain tag to avoid overwriting when hybrid
            out_img = out_dir / "overlays" / f"{Path(img_name).stem}__{args.explain}__cam_{NIH14[ci]}.png"
            cv2.imwrite(str(out_img), overlay)
            overlay_files.append(out_img.name)

            del_auc = deletion_insertion_auc(model, x, ci, cam, steps=args.steps, mode="deletion")
            ins_auc = deletion_insertion_auc(model, x, ci, cam, steps=args.steps, mode="insertion")
            del_aucs.append(del_auc)
            ins_aucs.append(ins_auc)

        rows_out.append({
            "Image": img_name,
            "GT": gt_primary,
            "Pred": pred_primary,
            "Top1": NIH14[top1_idx],
            "Top1_score": top1_score,
            "explain": args.explain,
            "CAM_Classes": "|".join([NIH14[c] for c in cam_classes]) if cam_classes else "",
            "DeletionAUC_mean": float(np.mean(del_aucs)) if del_aucs else np.nan,
            "InsertionAUC_mean": float(np.mean(ins_aucs)) if ins_aucs else np.nan,
            "Overlay_Files": "|".join(overlay_files),
            "img_path": str(img_path),
        })

        print(f"[{len(rows_out)}/{len(df)}] {img_name} | GT={gt_primary} | Pred={pred_primary} | explain={args.explain}")

    gc.close()

    if len(rows_out) == 0:
        print("\n[ERROR] 0 image traitée. Vérifie src_path / base_dir.")
        print("Astuce: ce script résout src_path relatif par rapport au dossier du CSV (ou --images_dir).")
        return

    results = pd.DataFrame(rows_out)
    out_csv = out_dir / "results_gradcam.csv"
    results.to_csv(out_csv, index=False)

    print("\n[DONE] Saved overlays ->", (out_dir / "overlays"))
    print("[DONE] Saved results ->", out_csv)

    # ----- Model metrics -----
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
        print("\nPer-class (support>0):")
        for k, v in per.items():
            if v["support"] > 0:
                print(f"  {k:>18s} | support={v['support']:2d} | P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")

    # ----- XAI metrics -----
    print("\n[XAI METRICS (mean over explained classes)]")
    if "DeletionAUC_mean" in results.columns and results["DeletionAUC_mean"].notna().any():
        print(f"Mean DeletionAUC: {results['DeletionAUC_mean'].mean():.3f}")
    else:
        print("Mean DeletionAUC: n/a")

    if "InsertionAUC_mean" in results.columns and results["InsertionAUC_mean"].notna().any():
        print(f"Mean InsertionAUC: {results['InsertionAUC_mean'].mean():.3f}")
    else:
        print("Mean InsertionAUC: n/a")


if __name__ == "__main__":
    main()
