#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PYLON + Grad-CAM++ (NIH14) — FINAL PRO (500 images)
Improvements vs previous:
1) Robust image indexing (.png/.jpg/.jpeg) across NIH layouts (+ recursive fallback)
2) Grad-CAM++ with auto fallback to Grad-CAM when positive grads vanish (prevents black CAM)
3) Cleaner heatmaps:
   - percentile normalization (p_low/p_high)
   - optional gamma
   - NO aggressive saturation by default (amplify=1.0)
   - optional center-ellipse mask to reduce corner/marker artifacts
4) Saves:
   overlays/<img>__cam_LABEL.png         (overlay)
   overlays/<img>__cam_LABEL__heat.png   (heat-only)
   raw/<img>__cam_LABEL__raw.png         (optional grayscale)
   preds_with_cam_sample.csv
   metrics.json
"""

import argparse
from pathlib import Path
import sys
import importlib.util
import time
import json

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


# -----------------------
# Basic utils
# -----------------------
def pick_device(force_cpu: bool = False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def torch_load_compat(path: str, device):
    import inspect
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location=device, weights_only=False)
    return torch.load(path, map_location=device)


def build_filename_index(images_root: Path):
    """
    Robust basename->fullpath index.
    Supports .png/.jpg/.jpeg and common NIH layouts:
      - images_*/images/*
      - images/*
      - recursive fallback
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


def load_image_rgb_and_gray(img_path: Path, size: int = 256):
    """
    Returns:
      - rgb_u8: (H,W,3) uint8 resized
      - x: torch Tensor [1,1,H,W] float32 in [0,1]
    """
    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img).astype(np.float32)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb_u8 = np.clip(rgb, 0, 255).astype(np.uint8)

    x = torch.from_numpy(rgb / 255.0).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x.mean(dim=1, keepdim=True)                                  # [1,1,H,W]
    return rgb_u8, x


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
# Import PYLON
# -----------------------
def import_pylon_from_repo(pylon_repo: Path):
    """
    pylon_repo=/.../projet/pylon
    file: pylon_repo/pylon/pylon.py
    add pylon_repo/pylon to sys.path for trainer imports.
    """
    pylon_repo = pylon_repo.resolve()
    pkg_root = pylon_repo / "pylon"
    pylon_py = pkg_root / "pylon.py"
    if not pylon_py.exists():
        raise FileNotFoundError(f"Not found: {pylon_py}")

    if str(pkg_root) not in sys.path:
        sys.path.insert(0, str(pkg_root))

    spec = importlib.util.spec_from_file_location("pylon_local", str(pylon_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    print(f"[DEBUG] pylon loaded from: {pylon_py}")
    return mod


def build_and_load_model(pylon_repo: Path, ckpt: Path, device, n_in: int = 1, n_out: int = 14):
    pylon_mod = import_pylon_from_repo(pylon_repo)
    if not hasattr(pylon_mod, "PylonConfig") or not hasattr(pylon_mod, "Pylon"):
        raise RuntimeError("Loaded module doesn't contain PylonConfig/Pylon.")

    conf = pylon_mod.PylonConfig(n_in=n_in, n_out=n_out)
    model = pylon_mod.Pylon(conf).to(device)

    ckpt_obj = torch_load_compat(str(ckpt), device=device)
    sd = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


def patch_pylon_for_inference(model, up_max: int = 3):
    """
    Minimal patch (matches your working inference):
    - decoder.pa.forward : list/tuple -> last tensor
    - decoder.forward : custom chain of up blocks
    """
    net = model.net
    dec = net.decoder

    if hasattr(dec, "pa") and hasattr(dec.pa, "forward"):
        orig_pa_fwd = dec.pa.forward

        def pa_forward_patched(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                x = x[-1]
            return orig_pa_fwd(x, *args, **kwargs)

        dec.pa.forward = pa_forward_patched
        print("[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)")

    chosen = []
    for cand in ["up3", "up2", "up1", "up0", "up4"]:
        if hasattr(dec, cand):
            chosen.append(cand)
    chosen = chosen[:up_max]

    def decoder_forward_custom(features, *args, **kwargs):
        if isinstance(features, tuple) and len(features) == 1:
            features = features[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError(f"decoder expected list/tuple, got {type(features)}")

        feats = list(features)
        bottleneck = feats[-1]
        x = dec.pa(bottleneck) if hasattr(dec, "pa") else bottleneck

        for i, up_name in enumerate(chosen):
            up = getattr(dec, up_name)
            skip = feats[-(i + 2)]
            x = up(skip, x)
        return x

    dec.forward = decoder_forward_custom
    print(f"[DEBUG] Patched decoder.forward (custom, ups={chosen})")


def get_target_layer(model, name: str):
    """
    name="layer4_first" => model.net.encoder.layer4[0] if exists
    else exact named_modules key, else suffix match.
    """
    if name == "layer4_first":
        enc = getattr(getattr(model, "net", None), "encoder", None)
        if enc is not None and hasattr(enc, "layer4"):
            try:
                return enc.layer4[0]
            except Exception:
                pass
        for n, m in model.named_modules():
            if n.endswith("encoder.layer4.0") or n.endswith("layer4.0"):
                return m
        raise ValueError("layer4_first not found. Use --target_layer with a real module name.")

    for n, m in model.named_modules():
        if n == name:
            return m
    for n, m in model.named_modules():
        if n.endswith(name):
            return m
    raise ValueError(f"Target layer '{name}' not found.")


# -----------------------
# CAM hooks + CAM++ (with fallback)
# -----------------------
class ActivationsAndGradients:
    def __init__(self, target_module: torch.nn.Module):
        self.activations = None
        self.gradients = None
        self.hook_fwd = target_module.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

        def _grad_hook(grad):
            self.gradients = grad

        if hasattr(out, "register_hook"):
            out.register_hook(_grad_hook)

    def close(self):
        try:
            self.hook_fwd.remove()
        except Exception:
            pass


def campp_or_cam_from_logits(model, hooks: ActivationsAndGradients, logits: torch.Tensor, class_idx: int):
    """
    Returns cam [H,W] in [0,1].
    Uses Grad-CAM++; if ReLU(grad) vanishes, falls back to Grad-CAM weights.
    """
    model.zero_grad(set_to_none=True)
    score = logits[0, class_idx]
    score.backward(retain_graph=True)

    A = hooks.activations  # [1,C,h,w]
    dY = hooks.gradients   # [1,C,h,w]
    if A is None or dY is None:
        raise RuntimeError("No activations/gradients captured. Check target_layer.")
    if A.dim() != 4 or dY.dim() != 4:
        raise RuntimeError(f"Unexpected dims: A={A.dim()} dY={dY.dim()}")

    with torch.no_grad():
        relu_grad = F.relu(dY)

        if relu_grad.max() > 0:
            grad2 = dY * dY
            grad3 = grad2 * dY
            sum_A_grad3 = (A * grad3).sum(dim=(2, 3), keepdim=True)  # [1,C,1,1]
            denom = 2.0 * grad2 + sum_A_grad3
            denom = torch.where(denom != 0.0, denom, torch.ones_like(denom))
            alpha = grad2 / denom
            weights = (alpha * relu_grad).sum(dim=(2, 3), keepdim=True)  # [1,C,1,1]
        else:
            weights = dY.mean(dim=(2, 3), keepdim=True)  # Grad-CAM fallback

        cam = (weights * A).sum(dim=1, keepdim=False)[0]  # [h,w]
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam.detach().cpu().numpy()


# -----------------------
# CAM post-processing (reduce artifacts)
# -----------------------
def suppress_border(cam: np.ndarray, frac: float):
    if frac <= 0:
        return cam
    h, w = cam.shape
    bh = int(round(frac * h))
    bw = int(round(frac * w))
    out = cam.copy()
    out[:bh, :] = 0
    out[-bh:, :] = 0
    out[:, :bw] = 0
    out[:, -bw:] = 0
    return out


def normalize_cam_percentile(cam01: np.ndarray, p_low=10.0, p_high=99.5, gamma=1.0):
    cam = cam01.astype(np.float32)

    lo = float(np.percentile(cam, p_low))
    hi = float(np.percentile(cam, p_high))

    # fallback when percentiles collapse (common on 8x8)
    if (hi - lo) < 1e-6:
        lo2 = float(cam.min())
        hi2 = float(cam.max())
        if (hi2 - lo2) < 1e-6:
            return np.zeros_like(cam, dtype=np.float32)
        cam = (cam - lo2) / (hi2 - lo2 + 1e-8)
    else:
        cam = (cam - lo) / (hi - lo + 1e-8)

    cam = np.clip(cam, 0, 1)
    cam = cam ** float(gamma)
    return cam


def center_ellipse_mask(h, w, scale=0.92):
    """
    Simple lung-focused mask (no segmentation). Reduces corner/marker artifacts.
    """
    Y, X = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    ry, rx = (h * scale) / 2.0, (w * scale) / 2.0
    m = ((Y - cy) ** 2) / (ry ** 2 + 1e-8) + ((X - cx) ** 2) / (rx ** 2 + 1e-8) <= 1.0
    return m.astype(np.float32)


def apply_optional_center_mask(cam01: np.ndarray, enable: bool, scale: float):
    if not enable:
        return cam01
    h, w = cam01.shape
    m = center_ellipse_mask(h, w, scale=scale)
    out = cam01 * m
    mx = out.max()
    if mx > 0:
        out = out / mx
    return out


# -----------------------
# Save images (heat-only + overlay)
# -----------------------
def save_heat_and_overlay(rgb_u8: np.ndarray,
                          cam01: np.ndarray,
                          out_overlay_path: Path,
                          alpha_heat: float = 0.35,
                          amplify: float = 1.0,
                          colormap=cv2.COLORMAP_TURBO):
    """
    Writes:
      - overlay: out_overlay_path
      - heat-only: out_overlay_path stem + '__heat.png'
    """
    H, W = rgb_u8.shape[:2]
    if cam01.shape[0] != H or cam01.shape[1] != W:
        cam01 = cv2.resize(cam01, (W, H), interpolation=cv2.INTER_CUBIC)

    cam01 = np.clip(cam01, 0, 1).astype(np.float32)
    cam01 = np.clip(cam01 * float(amplify), 0, 1)

    cam_u8 = (cam01 * 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(cam_u8, colormap)

    heat_path = out_overlay_path.with_name(out_overlay_path.stem + "__heat.png")
    cv2.imwrite(str(heat_path), heat_bgr)

    img_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    a = float(np.clip(alpha_heat, 0.0, 1.0))
    overlay_bgr = cv2.addWeighted(img_bgr, 1.0 - a, heat_bgr, a, 0.0)
    cv2.imwrite(str(out_overlay_path), overlay_bgr)


def save_raw_gray(cam01: np.ndarray, out_path: Path):
    cam_u8 = (np.clip(cam01, 0, 1) * 255).astype(np.uint8)
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
    ap.add_argument("--pylon_repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--split_list", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--tau_nf", type=float, default=0.07)

    ap.add_argument("--sample_n", type=int, default=500)
    ap.add_argument("--sample_mode", choices=["first", "random"], default="random")
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--topk_cam", type=int, default=1)
    ap.add_argument("--target_layer", type=str, default="layer4_first")
    ap.add_argument("--up_max", type=int, default=3)
    ap.add_argument("--cpu", action="store_true")

    # CAM cleanup
    ap.add_argument("--cam_border_suppress", type=float, default=0.03)
    ap.add_argument("--cam_norm_low", type=float, default=10.0)
    ap.add_argument("--cam_norm_high", type=float, default=99.5)
    ap.add_argument("--cam_gamma", type=float, default=1.0)
    ap.add_argument("--cam_alpha", type=float, default=0.35)
    ap.add_argument("--cam_amplify", type=float, default=1.0)

    ap.add_argument("--use_center_mask", action="store_true")
    ap.add_argument("--center_mask_scale", type=float, default=0.92)

    ap.add_argument("--save_raw_cam", action="store_true")

    args = ap.parse_args()

    t0 = time.time()
    out_dir = Path(args.out_dir)
    overlays_dir = out_dir / "overlays"
    raw_dir = out_dir / "raw"
    overlays_dir.mkdir(parents=True, exist_ok=True)
    if args.save_raw_cam:
        raw_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(force_cpu=args.cpu)
    print("[INFO] Device:", device)
    print(f"[INFO] sample_n={args.sample_n} mode={args.sample_mode} seed={args.seed}")
    print(f"[INFO] thr={args.thr} tau_nf={args.tau_nf} topk_cam={args.topk_cam}")
    print(f"[INFO] target_layer={args.target_layer}")
    print(f"[INFO] cam: alpha={args.cam_alpha} amplify={args.cam_amplify} norm=[{args.cam_norm_low},{args.cam_norm_high}] gamma={args.cam_gamma}")
    print(f"[INFO] cleanup: border={args.cam_border_suppress} center_mask={args.use_center_mask} scale={args.center_mask_scale}")

    # Split
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

    # CSV
    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]
    img_col = "Image Index" if "Image Index" in df.columns else None
    lab_col = "Finding Labels" if "Finding Labels" in df.columns else None
    if img_col is None or lab_col is None:
        raise ValueError("CSV must contain: 'Image Index' and 'Finding Labels'.")

    df = df[df[img_col].astype(str).isin(sampled_set)]
    order = {fn: i for i, fn in enumerate(sampled)}
    df["__ord"] = df[img_col].astype(str).map(order)
    df = df.sort_values("__ord").drop(columns=["__ord"]).reset_index(drop=True)
    print(f"[INFO] CSV filtered rows: {len(df)}")

    # Image index
    images_root = Path(args.images_root)
    print("[INFO] building filename index (robust)...")
    idx = build_filename_index(images_root)
    print(f"[INFO] index size: {len(idx)}")
    if len(idx) == 0:
        raise RuntimeError("Image index is empty. Check --images_root.")
    k = next(iter(idx.keys()))
    print(f"[INFO] index example: {k} -> {idx[k]}")

    # Model
    model = build_and_load_model(Path(args.pylon_repo), Path(args.ckpt), device=device, n_in=1, n_out=14)
    patch_pylon_for_inference(model, up_max=args.up_max)
    model.eval()

    target_layer = get_target_layer(model, args.target_layer)
    hooks = ActivationsAndGradients(target_layer)

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
            if skipped_missing <= 5:
                print(f"[MISS] {fn} not found under {images_root}")
            continue

        rgb_u8, x = load_image_rgb_and_gray(Path(img_path), size=args.img_size)
        x = x.to(device)

        # forward with graph
        model.zero_grad(set_to_none=True)
        out = model(x)

        if hasattr(out, "pred"):
            logits = out.pred
        elif isinstance(out, (tuple, list)) and len(out) > 0:
            logits = out[0]
        else:
            raise RuntimeError("Model output doesn't have .pred and isn't tuple/list with logits.")

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

        topk = min(args.topk_cam, 14)
        topk_idx = np.argsort(-prob)[:topk].tolist()

        overlay_files, heat_files, raw_files = [], [], []

        for j in topk_idx:
            cam01 = campp_or_cam_from_logits(model, hooks, logits, class_idx=j)

            # cleanup on CAM grid (e.g., 8x8)
            cam01 = suppress_border(cam01, args.cam_border_suppress)

            # normalization (conservative)
            cam_viz = normalize_cam_percentile(cam01, args.cam_norm_low, args.cam_norm_high, args.cam_gamma)

            # (optional) reduce corners/markers artifacts
            cam_viz = apply_optional_center_mask(cam_viz, args.use_center_mask, args.center_mask_scale)

            base = f"{Path(fn).stem}__cam_{NIH14[j]}"
            overlay_path = overlays_dir / f"{base}.png"
            save_heat_and_overlay(
                rgb_u8=rgb_u8,
                cam01=cam_viz,
                out_overlay_path=overlay_path,
                alpha_heat=args.cam_alpha,
                amplify=args.cam_amplify,
                colormap=cv2.COLORMAP_TURBO
            )
            overlay_files.append(f"{base}.png")
            heat_files.append(f"{base}__heat.png")

            if args.save_raw_cam:
                raw_path = raw_dir / f"{base}__raw.png"
                save_raw_gray(cam_viz, raw_path)
                raw_files.append(f"{base}__raw.png")

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
        })

        if (i + 1) % 10 == 0:
            print(f"[INFO] processed {i+1}/{len(df)}...")

    hooks.close()

    if not y_true:
        raise RuntimeError("0 image processed. Check split_list/csv/images_root matching.")

    y_true = np.stack(y_true, axis=0).astype(np.int32)
    y_prob = np.stack(y_prob, axis=0).astype(np.float32)

    m = multilabel_metrics(y_true, y_prob, thr=args.thr)
    df_out = pd.DataFrame(rows)

    # dominant 15 accuracy + macro-f1
    acc15 = float((df_out["Dominant_Pred_15"] == df_out["Dominant_GT_15"]).mean()) if len(df_out) else 0.0
    labels15 = NIH14 + [NO_FINDING]
    f1s = []
    for lab in labels15:
        gt = (df_out["Dominant_GT_15"] == lab).values
        pr = (df_out["Dominant_Pred_15"] == lab).values
        tp = int((gt & pr).sum())
        fp = int((~gt & pr).sum())
        fn = int((gt & ~pr).sum())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2*p*r/(p+r) if (p+r) else 0.0
        f1s.append(f1)
    macro_f1_15 = float(np.mean(f1s))

    print("\n[MULTILABEL NIH14 METRICS]")
    print(f"thr={args.thr:.3f}")
    print(f"micro  P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"macro  P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} F1={m['macro_f1']:.3f}")
    print(f"macro AUROC={m['macro_auroc']:.3f} | macro mAP={m['macro_map']:.3f}")

    print("\n[SINGLE-LABEL 15 (Dominant GT vs Pred incl. No Finding)]")
    print(f"Accuracy15={acc15:.3f} | Macro-F1-15={macro_f1_15:.3f}")

    out_dir = Path(args.out_dir)
    out_csv = out_dir / "preds_with_cam_sample.csv"
    df_out.to_csv(out_csv, index=False)

    metrics_path = out_dir / "metrics.json"
    payload = {
        "multilabel": m,
        "dominant15": {"accuracy15": acc15, "macro_f1_15": macro_f1_15},
        "args": vars(args),
        "processed": int(len(df_out)),
        "skipped_missing": int(skipped_missing),
        "elapsed_sec": float(time.time() - t0),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))

    print(f"\n[DONE] CSV: {out_csv}")
    print(f"[DONE] metrics: {metrics_path}")
    print(f"[DONE] overlays: {Path(args.out_dir) / 'overlays'}")
    if args.save_raw_cam:
        print(f"[DONE] raw: {Path(args.out_dir) / 'raw'}")
    print(f"[INFO] processed={len(df_out)} skipped_missing={skipped_missing} elapsed={payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES= \
python3 pylon_gradcampp_pro_final_500.py \
  --pylon_repo /home/shakib/Desktop/S5/projet/pylon \
  --ckpt /home/shakib/Desktop/S5/projet/pylon_ckpt/pylon_nih_256.pkl \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --out_dir /home/shakib/Desktop/S5/projet/xai_outputs/campp_final_500_fixed \
  --img_size 256 \
  --thr 0.1 \
  --tau_nf 0.07 \
  --topk_cam 1 \
  --target_layer layer4_first \
  --cam_border_suppress 0 \
  --cam_norm_low 5 --cam_norm_high 99 --cam_gamma 1.0 \
  --cam_alpha 0.35 --cam_amplify 1.0 \
  --sample_n 500 --sample_mode random --seed 0 \
  --cpu \
  --save_raw_cam

[MULTILABEL NIH14 METRICS]
thr=0.100
micro  P=0.317 R=0.573 F1=0.408
macro  P=0.331 R=0.428 F1=0.342
macro AUROC=0.780 | macro mAP=0.317

[SINGLE-LABEL 15 (Dominant GT vs Pred incl. No Finding)]
Accuracy15=0.478 | Macro-F1-15=0.387
Sur 500 images, tes chiffres sont cohérents et plutôt “dans la norme” pour du NIH14 multi-label sans calibration fine, mais il faut bien interpréter ce que chaque métrique dit. Les valeurs exactes sont aussi dans ton metrics.json 

metrics

.

Lecture rapide de tes résultats
1) Multi-label (NIH14)

AUROC macro = 0.780 

metrics


→ C’est un bon signal : en ranking (trier les positifs au-dessus des négatifs), le modèle se débrouille correctement.

mAP macro = 0.317 

metrics


→ mAP pénalise fort le déséquilibre + les faux positifs : 0.31 est typique/raisonnable sur NIH14 si tu n’es pas en configuration “paper-level tuning”.

F1 micro = 0.408 (P=0.317, R=0.573) 

metrics


→ Ton rappel est élevé, mais précision faible : tu déclares “malade” souvent (thr=0.1), donc tu récupères pas mal de vrais positifs, mais tu génères beaucoup de faux positifs.

👉 Conclusion multi-label : le modèle classe assez bien (AUROC), mais le point de fonctionnement (thr=0.1) est très “rappel” et pas “précision”.

2) “Single-label 15” (dominant + No Finding)

Accuracy15 = 0.478, Macro-F1-15 = 0.387 

metrics


→ C’est “OK” mais pas étonnant : tu forces un problème multi-label (où plusieurs maladies peuvent coexister) en une seule étiquette dominante, ce qui est intrinsèquement plus dur et parfois artificiel (surtout si la GT a plusieurs labels).

👉 Conclusion “dominant15” : utile comme indicateur, mais pas la métrique principale pour NIH14.
"""