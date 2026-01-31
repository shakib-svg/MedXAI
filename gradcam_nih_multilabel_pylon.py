#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
import sys
import inspect
import importlib

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn.functional as F
import cv2


NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"


# -------------------------
# Utils compat / device
# -------------------------
def torch_load_compat(path: str):
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


def pick_device(force_cpu: bool = False):
    if force_cpu or (not torch.cuda.is_available()):
        return torch.device("cpu")
    dev = torch.device("cuda")
    # test simple pour éviter GPU incompatible
    try:
        x = torch.randn(1, 1, 8, 8, device=dev)
        _ = x.sum().item()
        return dev
    except Exception:
        return torch.device("cpu")


# -------------------------
# NIH labels
# -------------------------
def parse_labels(label_str: str):
    s = str(label_str).strip()
    if s == "" or s.lower() == "nan":
        return [NO_FINDING]
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


# -------------------------
# Image I/O
# -------------------------
def load_rgb_as_gray_tensor(img_path: Path, size=256):
    img = Image.open(img_path).convert("RGB")
    rgb0 = np.asarray(img).astype(np.float32) / 255.0  # H,W,3
    rgb_256 = cv2.resize(rgb0, (size, size), interpolation=cv2.INTER_LINEAR)

    x_rgb = torch.from_numpy(rgb_256).permute(2, 0, 1).unsqueeze(0)  # 1,3,H,W
    x_gray = x_rgb.mean(dim=1, keepdim=True)  # 1,1,H,W
    return x_gray, rgb_256, rgb0


# -------------------------
# Grad-CAM
# -------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._fh)
        self.h2 = target_layer.register_full_backward_hook(self._bh)

    def _fh(self, m, inp, out):
        self.activations = out

    def _bh(self, m, gin, gout):
        self.gradients = gout[0]

    @torch.no_grad()
    def _forward_logits(self, x):
        out = self.model(x)
        return out.pred  # [B,14]

    def __call__(self, x, class_idx: int):
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = out.pred
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        A = self.activations
        dA = self.gradients

        w = dA.mean(dim=(2, 3), keepdim=True)
        w_abs_max = w.abs().max()
        if w_abs_max > 0 and w_abs_max < 1e-3:
            w = w / (w_abs_max + 1e-8) * 0.1

        cam = (w * A).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()

        # normalize [0,1]
        mn, mx = float(cam.min()), float(cam.max())
        if mx > mn:
            cam = (cam - mn) / (mx - mn)
        else:
            cam = np.zeros_like(cam)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


def overlay_cam_on_rgb(rgb_float01, cam01, alpha=0.4):
    heat = (cam01 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay = np.clip((1 - alpha) * rgb_float01 + alpha * heat, 0, 1)
    return (overlay[..., ::-1] * 255).astype(np.uint8)  # BGR uint8 for cv2.imwrite


def suppress_cam_border(cam, frac=0.0):
    if frac <= 0:
        return cam
    h, w = cam.shape
    bh = int(round(h * frac))
    bw = int(round(w * frac))
    out = cam.copy()
    if bh > 0:
        out[:bh, :] = 0
        out[-bh:, :] = 0
    if bw > 0:
        out[:, :bw] = 0
        out[:, -bw:] = 0
    return out


# -------------------------
# Build & patch Pylon
# -------------------------
def build_pylon_from_repo(pylon_repo: str, n_in=1, n_out=14):
    """
    Robust import:
    - pylon_repo is root repo: .../projet/pylon
    - actual code is .../pylon/pylon/pylon.py
    We insert .../pylon/pylon into sys.path then import module 'pylon'.
    """
    repo = Path(pylon_repo).expanduser().resolve()
    pkg = repo / "pylon"  # contains pylon.py, trainer/, model/, etc.
    if not (pkg / "pylon.py").exists():
        raise FileNotFoundError(f"Expected {pkg/'pylon.py'} not found. Check --pylon_repo.")

    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))

    # Avoid collision with a different installed module named "pylon"
    if "pylon" in sys.modules:
        del sys.modules["pylon"]

    pylon = importlib.import_module("pylon")
    print("[DEBUG] pylon loaded from:", getattr(pylon, "__file__", None))

    conf = pylon.PylonConfig(n_in=n_in, n_out=n_out)
    model = pylon.Pylon(conf)
    return model


def load_checkpoint(model, ckpt_path: str):
    ckpt = torch_load_compat(ckpt_path)
    sd = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_sd[k] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


def patch_pylon_decoder(model, up_max=3, verbose=True):
    """
    Patch that worked for you:
    - custom decoder.forward using pa + up3/up2/up1
    - makes Pylon compatible with current segmentation_models_pytorch behavior
    """
    dec = model.net.decoder

    # Patch pa.forward just in case it receives list/tuple
    if hasattr(dec, "pa"):
        pa = dec.pa
        orig_pa_forward = pa.forward

        def pa_forward_patched(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                # take last = bottleneck tensor
                x = x[-1]
            return orig_pa_forward(x, *args, **kwargs)

        pa.forward = pa_forward_patched
        if verbose:
            print("[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)")

    # Custom decoder.forward
    # expected features list length >= 5 (input + 4 skips + bottleneck), but we’ll be defensive.
    ups = []
    for name in ["up5", "up4", "up3", "up2", "up1"]:
        if hasattr(dec, name):
            ups.append(name)

    # We will use only up3, up2, up1 as in your working setup
    use_ups = [u for u in ["up3", "up2", "up1"] if hasattr(dec, u)]
    if len(use_ups) == 0:
        raise RuntimeError("Decoder has no up blocks (up1/up2/up3 not found).")

    orig_dec_forward = dec.forward

    def decoder_forward_custom(features, *args, **kwargs):
        # normalize features to list[Tensor]
        if isinstance(features, tuple) and len(features) == 1 and isinstance(features[0], (list, tuple)):
            features = features[0]
        if isinstance(features, tuple):
            features = list(features)
        if not isinstance(features, list):
            # last resort
            features = [features]

        # ensure tensors only
        # typical: [x0, f1, f2, f3, f4, bottleneck]
        if len(features) < 2:
            return orig_dec_forward(features, *args, **kwargs)

        bottleneck = features[-1]
        x = dec.pa(bottleneck)

        # map ups to skips from the end:
        # up3 uses features[-2], up2 uses features[-3], up1 uses features[-4]
        for up_name, skip_offset in [("up3", 2), ("up2", 3), ("up1", 4)]:
            if hasattr(dec, up_name) and len(features) >= skip_offset:
                skip = features[-skip_offset]
                x = getattr(dec, up_name)(skip, x)

        return x

    dec.forward = decoder_forward_custom
    if verbose:
        print(f"[DEBUG] Patched decoder.forward (custom, up_max={up_max}, ups={use_ups})")


# -------------------------
# Locate images
# -------------------------
def load_split_list(split_list: Path):
    lines = []
    with open(split_list, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                lines.append(s)
    return lines


def build_filename_index(images_root: Path):
    """
    Build dict: filename -> full path
    Only if split list contains bare filenames (no subpath).
    """
    idx = {}
    # NIH structure in your archive: images_001/images/*.png, images_002/images/*.png, ...
    for sub in sorted(images_root.glob("images_*")):
        p = sub / "images"
        if not p.exists():
            continue
        for fp in p.glob("*.png"):
            idx[fp.name] = fp
    return idx


def resolve_img_path(images_root: Path, item: str, name_index=None):
    # item can be subpath like "images_001/images/0000.png" or bare filename
    if "/" in item or "\\" in item:
        p = (images_root / item).resolve()
        return p
    # bare filename
    if name_index is not None and item in name_index:
        return name_index[item]
    # fallbacks
    p1 = (images_root / item).resolve()
    if p1.exists():
        return p1
    p2 = (images_root / "images" / item).resolve()
    if p2.exists():
        return p2
    # last resort (slow): search in images_*/images
    for sub in sorted(images_root.glob("images_*")):
        p = (sub / "images" / item).resolve()
        if p.exists():
            return p
    return None


# -------------------------
# Metrics (quick)
# -------------------------
def micro_precision_recall_f1(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return prec, rec, f1


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylon_repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--split_list", required=True)

    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--alpha", type=float, default=0.4)

    # choose what to explain
    ap.add_argument("--thr", type=float, default=None,
                    help="If set: explain all classes with prob>=thr (limited by --max_cam_per_image).")
    ap.add_argument("--topk", type=int, default=1,
                    help="If --thr is not set: explain top-k classes per image.")
    ap.add_argument("--max_cam_per_image", type=int, default=3)

    ap.add_argument("--target_layer", choices=["layer4_first", "layer4_last", "layer3_last"],
                    default="layer4_first")
    ap.add_argument("--cam_border_suppress", type=float, default=0.0)

    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--out_csv", default=None)

    ap.add_argument("--save_orig", action="store_true",
                    help="Also save overlays on original resolution (much heavier).")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--debug_every", type=int, default=0)

    args = ap.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    overlays_dir = out_dir / "overlays"
    overlays_dir.mkdir(parents=True, exist_ok=True)

    out_csv = Path(args.out_csv).expanduser().resolve() if args.out_csv else (out_dir / "preds_with_cam.csv")

    device = pick_device(force_cpu=args.cpu)
    print("[INFO] Device:", device)
    print("[INFO] split_list:", args.split_list)
    print("[INFO] thr:", args.thr, "| topk:", args.topk, "| max_cam_per_image:", args.max_cam_per_image)

    # Load CSV (labels)
    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError("CSV must contain columns: 'Image Index' and 'Finding Labels'.")

    # Map filename -> GT labels
    gt_map = dict(zip(df["Image Index"].astype(str), df["Finding Labels"].astype(str)))

    # Load split list
    split_list = load_split_list(Path(args.split_list).expanduser().resolve())
    print(f"[INFO] split_list loaded: {args.split_list} (n={len(split_list)})")

    images_root = Path(args.images_root).expanduser().resolve()

    # if split list has bare filenames, build index
    need_index = all(("/" not in s and "\\" not in s) for s in split_list[:100])
    name_index = None
    if need_index:
        print("[INFO] building filename index (images_*/images/*.png)...")
        name_index = build_filename_index(images_root)
        print("[INFO] index size:", len(name_index))

    # Build model
    model = build_pylon_from_repo(args.pylon_repo, n_in=1, n_out=14)
    model = load_checkpoint(model, args.ckpt).to(device).eval()
    patch_pylon_decoder(model, up_max=3, verbose=True)

    # target layer for Grad-CAM
    if args.target_layer == "layer4_first":
        target_layer = model.net.encoder.layer4[0]
    elif args.target_layer == "layer3_last":
        target_layer = model.net.encoder.layer3[-1]
    else:
        target_layer = model.net.encoder.layer4[-1]

    gc = GradCAM(model, target_layer)

    rows = []
    y_true_all = []
    y_pred_all = []

    processed = 0
    skipped_missing = 0

    for i, item in enumerate(split_list, start=1):
        filename = Path(item).name  # for GT lookup
        img_path = resolve_img_path(images_root, item, name_index=name_index)
        if img_path is None or (not Path(img_path).exists()):
            skipped_missing += 1
            continue

        # GT labels
        raw_gt = gt_map.get(filename, "")
        gt_labels = parse_labels(raw_gt)
        gt_vec = labels_to_vec(gt_labels)
        gt_clean = [l for l in gt_labels if l != NO_FINDING]
        gt_str = "|".join(gt_clean) if len(gt_clean) > 0 else NO_FINDING

        # Load img
        x, rgb_256, rgb0 = load_rgb_as_gray_tensor(Path(img_path), size=args.img_size)
        x = x.to(device)

        # Predict
        with torch.no_grad():
            out = model(x)
            probs = torch.sigmoid(out.pred)[0].detach().cpu().numpy()  # (14,)

        # Choose classes to explain
        top1_idx = int(np.argmax(probs))
        top1_lab = NIH14[top1_idx]
        top1_score = float(probs[top1_idx])

        if args.thr is not None:
            cand = np.where(probs >= float(args.thr))[0].tolist()
            if len(cand) == 0:
                # no class passes thr -> explain top1 (optional, but useful)
                cand = [top1_idx]
            # limit
            cand = sorted(cand, key=lambda k: float(probs[k]), reverse=True)[: int(args.max_cam_per_image)]
        else:
            k = max(1, int(args.topk))
            cand = list(np.argsort(-probs)[:k])

        pred_labels = [NIH14[j] for j in cand]
        pred_str = "|".join(pred_labels) if len(pred_labels) > 0 else NO_FINDING
        pred_vec = np.zeros(14, dtype=np.int32)
        for j in cand:
            pred_vec[j] = 1

        # Save CAM overlays
        overlay_files = []

        for ci in cand:
            cam = gc(x, ci)
            cam = suppress_cam_border(cam, frac=args.cam_border_suppress)

            overlay_256 = overlay_cam_on_rgb(rgb_256, cam, alpha=args.alpha)
            out_name = f"{Path(filename).stem}__cam_{NIH14[ci]}.png"
            out_path = overlays_dir / out_name
            cv2.imwrite(str(out_path), overlay_256)
            overlay_files.append(out_name)

            if args.save_orig:
                # Resize cam to original for overlay
                H0, W0, _ = rgb0.shape
                cam0 = cv2.resize(cam, (W0, H0), interpolation=cv2.INTER_LINEAR)
                cam0 = suppress_cam_border(cam0, frac=args.cam_border_suppress)
                overlay0 = overlay_cam_on_rgb(rgb0, cam0, alpha=args.alpha)
                out_name0 = f"{Path(filename).stem}__ORIG__cam_{NIH14[ci]}.png"
                out_path0 = overlays_dir / out_name0
                cv2.imwrite(str(out_path0), overlay0)
                overlay_files.append(out_name0)

        # compact top probs json (for debug like your sample)
        top5 = list(np.argsort(-probs)[:5])
        probs_json = ",".join([f"{NIH14[j]}:{probs[j]:.4f}" for j in top5])

        rows.append({
            "Image": filename,
            "img_path": str(img_path),
            "GT_labels": gt_str,
            "Pred_labels": pred_str,
            "thr": (float(args.thr) if args.thr is not None else ""),
            "topk": (int(args.topk) if args.thr is None else 0),
            "Overlay_files": "|".join(overlay_files),
            "Top1": top1_lab,
            "Top1_score": top1_score,
            "probs_top5": probs_json,
            "GT_vec_sum": int(gt_vec.sum()),
        })

        y_true_all.append(gt_vec)
        y_pred_all.append(pred_vec)

        processed += 1
        if processed % 500 == 0:
            print(f"[INFO] processed {processed} images...")

        if args.debug_every and (processed % args.debug_every == 0):
            print(f"[DEBUG] {filename} GT={gt_str} PRED={pred_str} top1={top1_lab}({top1_score:.3f})")

    gc.close()

    # Save CSV
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print("\n[DONE] Saved preds+cams CSV:", out_csv)
    print("[DONE] Saved overlays dir:", overlays_dir)
    print(f"[INFO] processed={processed} skipped_missing={skipped_missing}")

    # quick multilabel P/R/F1 with the chosen pred_vec (cand)
    if len(y_true_all) > 0:
        y_true = np.stack(y_true_all, axis=0)
        y_pred = np.stack(y_pred_all, axis=0)
        p, r, f1 = micro_precision_recall_f1(y_true, y_pred)
        print("\n[MULTILABEL (based on explained classes)]")
        print(f"micro P={p:.3f} R={r:.3f} F1={f1:.3f}")
        if args.thr is not None:
            print(f"(thr={args.thr}, max_cam_per_image={args.max_cam_per_image})")
        else:
            print(f"(topk={args.topk})")


if __name__ == "__main__":
    main()
"""
CUDA_VISIBLE_DEVICES= python gradcam_nih_multilabel_export.py \
  --pylon_repo /home/shakib/Desktop/S5/projet/pylon \
  --ckpt /home/shakib/Desktop/S5/projet/pylon_ckpt/pylon_nih_256.pkl \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --topk 1 \
  --target_layer layer4_first \
  --cam_border_suppress 0.05 \
  --out_dir /home/shakib/Desktop/S5/projet/xai_outputs/gradcam_test_top1 \
  --cpu



"""