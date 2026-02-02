#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import sys
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
import torch.nn.functional as F

# -----------------------
# Labels NIH14 + NoFinding (post-rule)
# -----------------------
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"


# -----------------------
# Device / torch.load compat
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


# -----------------------
# Image loading
# -----------------------
def load_image_gray_tensor(img_path: Path, size: int = 256):
    """
    Read PNG -> RGB -> resize -> grayscale(mean) -> Tensor [1,1,H,W] in [0,1]
    """
    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x.mean(dim=1, keepdim=True)                          # [1,1,H,W]
    return x


# -----------------------
# Labels parsing
# -----------------------
def parse_labels(label_val):
    """
    NIH CSV: "Finding Labels" is a string like "Hernia|Infiltration" or "No Finding".
    Return a LIST[str] of disease labels (NIH14 names). No Finding => [].
    """
    if label_val is None or (isinstance(label_val, float) and np.isnan(label_val)):
        return []

    # If already list-like, normalize
    if isinstance(label_val, (list, tuple)):
        parts = [str(x).strip() for x in label_val if str(x).strip()]
    else:
        s = str(label_val).strip()
        if not s:
            return []
        parts = [p.strip() for p in s.split("|") if p.strip()]

    # normalize NIH variants
    parts = [p.replace("Pleural Thickening", "Pleural_Thickening") for p in parts]

    # NIH "No Finding" => empty list
    if len(parts) == 1 and parts[0] == NO_FINDING:
        return []

    # keep only NIH14 known labels
    parts = [p for p in parts if p in NIH14]
    return parts


def labels_to_vec(labels_list):
    v = np.zeros(len(NIH14), dtype=np.int32)
    for lab in labels_list:
        if lab in NIH14:
            v[NIH14.index(lab)] = 1
    return v


# -----------------------
# Find images (fast index)
# -----------------------
def build_filename_index(images_root: Path):
    """
    Map filename -> full path by scanning images_root/images_*/images/*.png
    """
    print("[INFO] building filename index (images_*/images/*.png)...")
    index = {}
    for p in images_root.glob("images_*/images/*.png"):
        index[p.name] = p.resolve()
    print(f"[INFO] index size: {len(index)}")
    return index


def find_image(images_root: Path, index: dict, filename: str):
    # direct path case (if images_root points to folder with images)
    direct = (images_root / filename)
    if direct.exists():
        return direct.resolve()
    return index.get(filename, None)


# -----------------------
# Import Pylon (robuste)
# -----------------------
def import_pylon_from_repo(pylon_repo: Path):
    """
    Real file: pylon_repo/pylon/pylon.py
    trainer package: pylon_repo/pylon/trainer/...
    So add pylon_repo/pylon to sys.path and load module via importlib.
    """
    pylon_repo = pylon_repo.resolve()
    pkg_root = pylon_repo / "pylon"
    pylon_py = pkg_root / "pylon.py"
    if not pylon_py.exists():
        raise FileNotFoundError(f"Missing: {pylon_py}")

    # allow "from trainer.start import *"
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
        raise RuntimeError("Loaded pylon module has no PylonConfig/Pylon")

    conf = pylon_mod.PylonConfig(n_in=n_in, n_out=n_out)
    model = pylon_mod.Pylon(conf)

    ckpt_obj = torch_load_compat(str(ckpt), device)
    sd = ckpt_obj["state_dict"] if isinstance(ckpt_obj, dict) and "state_dict" in ckpt_obj else ckpt_obj

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing={len(missing)} Unexpected={len(unexpected)}")
    return model


# -----------------------
# Patch inference (your decoder fix)
# -----------------------
def patch_pylon_for_inference(model, up_max: int = 3):
    net = model.net
    dec = net.decoder

    # Patch PA: list/tuple -> bottleneck tensor
    if hasattr(dec, "pa") and hasattr(dec.pa, "forward"):
        orig_pa_fwd = dec.pa.forward

        def pa_forward_patched(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                x = x[-1]
            return orig_pa_fwd(x, *args, **kwargs)

        dec.pa.forward = pa_forward_patched
        print("[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)")

    # Choose up blocks (stable for your repo)
    chosen = []
    for cand in ["up3", "up2", "up1", "up0", "up4"]:
        if hasattr(dec, cand):
            chosen.append(cand)
    chosen = chosen[:up_max]

    def decoder_forward_custom(features, *args, **kwargs):
        # unwrap common smp tuple
        if isinstance(features, tuple) and len(features) == 1:
            features = features[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError(f"decoder expected list/tuple, got {type(features)}")

        feats = list(features)
        bottleneck = feats[-1]
        x = dec.pa(bottleneck) if hasattr(dec, "pa") else bottleneck

        # up3 uses skip feats[-2], up2 uses feats[-3], up1 uses feats[-4]...
        for i, up_name in enumerate(chosen):
            up = getattr(dec, up_name)
            skip = feats[-(i + 2)]
            x = up(skip, x)

        return x

    dec.forward = decoder_forward_custom
    print(f"[DEBUG] Patched decoder.forward (custom, ups={chosen})")


# -----------------------
# Target layer selection
# -----------------------
def get_target_layer(model, name: str):
    """
    Support:
      - exact module name (from model.named_modules())
      - shorthand: layer4_first -> first submodule in encoder.layer4 if exists
    """
    # Exact name
    named = dict(model.named_modules())
    if name in named:
        return named[name]

    # Shorthand: layer4_first
    if name == "layer4_first":
        # try common paths
        candidates = []
        for n, m in model.named_modules():
            # look for encoder layer4 blocks
            if ("encoder" in n or "backbone" in n) and "layer4" in n:
                candidates.append((n, m))
        # sort shortest name first
        candidates.sort(key=lambda x: len(x[0]))
        if candidates:
            # pick the first block-like module under layer4
            # heuristic: first candidate that is a conv/bn/relu or the first child
            n0, m0 = candidates[0]
            # if it has children, pick first child
            kids = list(m0.children())
            return kids[0] if kids else m0

    # If not found: print helpful suggestions
    print("[ERROR] Target layer not found. Some module names samples:")
    for i, n in enumerate(list(named.keys())[:50]):
        print("  ", n)
    raise ValueError(f"Target layer '{name}' not found in model modules.")


# -----------------------
# Grad-CAM (vanilla)
# -----------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out[0] corresponds to grad wrt output of layer
        self.gradients = grad_out[0]

    def remove(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx: int):
        """
        x: [1,1,H,W]
        class_idx: 0..13 (NIH14 index)
        returns cam: [H,W] float in [0,1]
        """
        self.model.zero_grad(set_to_none=True)

        out = self.model(x)
        logits = out.pred  # [1,14]
        score = logits[0, class_idx]
        score.backward(retain_graph=False)

        # activations: [1,C,h,w], gradients: [1,C,h,w]
        A = self.activations
        G = self.gradients
        if A is None or G is None:
            raise RuntimeError("GradCAM hooks did not capture activations/gradients")

        weights = G.mean(dim=(2, 3), keepdim=True)       # [1,C,1,1]
        cam = (weights * A).sum(dim=1, keepdim=False)    # [1,h,w]
        cam = F.relu(cam)
        cam = cam[0].detach().cpu().numpy()

        # normalize
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam


def overlay_cam_on_image(img_rgb_0_1: np.ndarray, cam_0_1: np.ndarray, border_suppress: float = 0.0):
    """
    img_rgb_0_1: HxWx3 float in [0,1]
    cam_0_1: hxw float in [0,1] -> resized to HxW
    border_suppress: suppress border artifacts by zeroing a border percent
    """
    H, W, _ = img_rgb_0_1.shape
    cam = cv2.resize(cam_0_1, (W, H), interpolation=cv2.INTER_LINEAR)

    if border_suppress and border_suppress > 0:
        b = int(min(H, W) * border_suppress)
        cam[:b, :] = 0
        cam[-b:, :] = 0
        cam[:, :b] = 0
        cam[:, -b:] = 0

    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)     # BGR uint8
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    out = 0.55 * img_rgb_0_1 + 0.45 * heat
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


# -----------------------
# Metrics (multilabel on sample)
# -----------------------
def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support

    y_pred = (y_prob >= thr).astype(np.int32)

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average="binary", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    aurocs = {}
    aps = {}
    for j, name in enumerate(NIH14):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.max() == yt.min():
            continue
        aurocs[name] = float(roc_auc_score(yt, yp))
        aps[name] = float(average_precision_score(yt, yp))

    macro_auc = float(np.mean(list(aurocs.values()))) if aurocs else float("nan")
    macro_ap = float(np.mean(list(aps.values()))) if aps else float("nan")

    return {
        "micro_p": float(p_micro), "micro_r": float(r_micro), "micro_f1": float(f1_micro),
        "macro_p": float(p_macro), "macro_r": float(r_macro), "macro_f1": float(f1_macro),
        "macro_auc": macro_auc, "macro_map": macro_ap,
    }


def singlelabel15_metrics(dominant_gt_15: list, dominant_pred_15: list):
    from sklearn.metrics import f1_score, accuracy_score

    labels15 = NIH14 + [NO_FINDING]
    # map label->id
    m = {lab: i for i, lab in enumerate(labels15)}
    y_true = np.array([m[x] for x in dominant_gt_15], dtype=np.int64)
    y_pred = np.array([m[x] for x in dominant_pred_15], dtype=np.int64)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return acc, macro_f1


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylon_repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--csv", required=True, help="NIH Data_Entry_2017.csv")
    ap.add_argument("--images_root", required=True, help="NIH archive root (contains images_001..)")
    ap.add_argument("--split_list", required=True, help="test_list.txt")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.10, help="multilabel threshold for NIH14 preds")
    ap.add_argument("--tau_nf", type=float, default=0.07, help="No Finding if max(prob)<tau_nf")
    ap.add_argument("--topk_cam", type=int, default=1, help="how many disease cams per image (usually 1)")
    ap.add_argument("--target_layer", default="layer4_first")
    ap.add_argument("--cam_border_suppress", type=float, default=0.05)

    ap.add_argument("--sample_n", type=int, default=50)
    ap.add_argument("--sample_mode", choices=["first", "random"], default="first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--up_max", type=int, default=3)

    args = ap.parse_args()

    device = pick_device(args.cpu)
    print("[INFO] Device:", device)
    print(f"[INFO] thr: {args.thr} | tau_nf: {args.tau_nf} | sample_n: {args.sample_n} | mode: {args.sample_mode}")
    print(f"[INFO] topk CAM: {args.topk_cam} | target_layer: {args.target_layer}")

    out_dir = Path(args.out_dir)
    overlays_dir = out_dir / "overlays"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    # Load split list and sample filenames
    split_list = [x.strip() for x in Path(args.split_list).read_text().splitlines() if x.strip()]
    print(f"[INFO] split_list loaded: {args.split_list} (n={len(split_list)})")

    if args.sample_mode == "random":
        random.seed(args.seed)
        sampled = random.sample(split_list, k=min(args.sample_n, len(split_list)))
    else:
        sampled = split_list[:min(args.sample_n, len(split_list))]
    print(f"[INFO] sampled {len(sampled)} images.")

    # Load CSV and filter to sampled filenames
    df = pd.read_csv(args.csv, engine="python")
    df.columns = [c.strip() for c in df.columns]
    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError("CSV must contain columns: 'Image Index' and 'Finding Labels'")

    df = df[df["Image Index"].isin(sampled)].copy()
    print(f"[INFO] CSV filtered rows: {len(df)}")

    # Build image index
    images_root = Path(args.images_root)
    fname_index = build_filename_index(images_root)

    # Model
    model = build_and_load_model(Path(args.pylon_repo), Path(args.ckpt), device, n_in=1, n_out=14)
    patch_pylon_for_inference(model, up_max=args.up_max)
    model = model.to(device).eval()

    # Target layer + GradCAM
    target_layer = get_target_layer(model, args.target_layer)
    cam_engine = GradCAM(model, target_layer)

    rows = []
    y_true = []
    y_prob = []
    dom_gt_15 = []
    dom_pred_15 = []

    processed = 0
    skipped_missing = 0

    # Iterate
    for _, r in df.iterrows():
        fn = str(r["Image Index"]).strip()
        gt_labels = parse_labels(r["Finding Labels"])
        gt_vec = labels_to_vec(gt_labels)

        img_path = find_image(images_root, fname_index, fn)
        if img_path is None or not Path(img_path).exists():
            skipped_missing += 1
            continue

        # forward
        x = load_image_gray_tensor(Path(img_path), size=args.img_size).to(device)

        with torch.no_grad():
            out = model(x)
            logits = out.pred[0]  # [14]
            prob = torch.sigmoid(logits).detach().cpu().numpy()

        # Pred multilabel @thr
        pred_idx_thr = np.where(prob >= args.thr)[0].tolist()
        pred_labels_thr = [NIH14[i] for i in pred_idx_thr]

        # Pred15 with NoFinding rule
        top1_idx = int(np.argmax(prob))
        top1_lab = NIH14[top1_idx]
        top1_score = float(prob[top1_idx])

        if top1_score < args.tau_nf:
            dominant_pred = NO_FINDING
        else:
            dominant_pred = top1_lab

        # Dominant GT15:
        # - if GT empty => No Finding
        # - else choose the GT label that the MODEL scores highest (more stable when GT multi)
        if len(gt_labels) == 0:
            dominant_gt = NO_FINDING
        else:
            best = max(gt_labels, key=lambda lab: prob[NIH14.index(lab)] if lab in NIH14 else -1)
            dominant_gt = best if best in NIH14 else gt_labels[0]

        # CAM targets:
        # If dominant_pred is No Finding, we still explain top1 disease (because model has no NF neuron).
        cam_targets = []
        if dominant_pred == NO_FINDING:
            cam_targets = [top1_idx]
        else:
            cam_targets = [NIH14.index(dominant_pred)]

        # If topk_cam>1, add next best diseases (excluding duplicates)
        if args.topk_cam > 1:
            order = np.argsort(-prob).tolist()
            for j in order:
                if j not in cam_targets:
                    cam_targets.append(j)
                if len(cam_targets) >= args.topk_cam:
                    break

        # Save overlays for each target
        img_rgb = np.asarray(Image.open(img_path).convert("RGB")).astype(np.float32) / 255.0
        img_rgb = cv2.resize(img_rgb, (args.img_size, args.img_size), interpolation=cv2.INTER_LINEAR)

        overlay_files = []
        for j in cam_targets:
            # run gradcam (needs gradients)
            cam = cam_engine(x, j)
            overlay = overlay_cam_on_image(img_rgb, cam, border_suppress=args.cam_border_suppress)
            out_name = f"{Path(fn).stem}__cam_{NIH14[j]}.png"
            cv2.imwrite(str(overlays_dir / out_name), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            overlay_files.append(out_name)

        # record
        rows.append({
            "Image": fn,
            "img_path": str(img_path),
            "GT_labels": "|".join(gt_labels) if len(gt_labels) else NO_FINDING,
            "Pred_labels@thr": "|".join(pred_labels_thr) if len(pred_labels_thr) else NO_FINDING,
            "thr": args.thr,
            "tau_nf": args.tau_nf,
            "Dominant_GT_15": dominant_gt,
            "Dominant_Pred_15": dominant_pred,
            "Top1_disease": top1_lab,
            "Top1_score": top1_score,
            "Overlay_files": "|".join(overlay_files),
            "probs_top5": ",".join([f"{NIH14[i]}:{prob[i]:.4f}" for i in np.argsort(-prob)[:5]]),
            "GT_vec_sum": int(gt_vec.sum()),
        })

        y_true.append(gt_vec)
        y_prob.append(prob)
        dom_gt_15.append(dominant_gt)
        dom_pred_15.append(dominant_pred)

        processed += 1
        if processed % 10 == 0:
            print(f"[INFO] processed {processed}/{len(df)}...")

    cam_engine.remove()

    if processed == 0:
        print("[ERROR] processed=0. Check paths/split/csv.")
        return

    y_true = np.stack(y_true, axis=0).astype(np.int32)
    y_prob = np.stack(y_prob, axis=0).astype(np.float32)

    # Metrics multilabel NIH14
    m = multilabel_metrics(y_true, y_prob, thr=args.thr)
    print("\n[MULTILABEL NIH14 METRICS]")
    print(f"thr={args.thr:.3f}")
    print(f"micro  P={m['micro_p']:.3f} R={m['micro_r']:.3f} F1={m['micro_f1']:.3f}")
    print(f"macro  P={m['macro_p']:.3f} R={m['macro_r']:.3f} F1={m['macro_f1']:.3f}")
    print(f"macro AUROC={m['macro_auc']:.3f} | macro mAP={m['macro_map']:.3f}")

    # Metrics single-label 15
    acc15, macrof1_15 = singlelabel15_metrics(dom_gt_15, dom_pred_15)
    print("\n[SINGLE-LABEL 15 (Dominant GT vs Dominant Pred incl. No Finding)]")
    print(f"Accuracy15={acc15:.3f} | Macro-F1-15={macrof1_15:.3f}")
    print("Note: Dominant_GT uses 'best GT label by model prob' when GT is multilabel.")

    # Save CSV
    out_csv = out_dir / "preds_with_cam_sample.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[DONE] Saved CSV: {out_csv}")
    print(f"[DONE] Saved overlays dir: {overlays_dir}")
    print(f"[INFO] processed={processed} (sample) skipped_missing={skipped_missing}")


if __name__ == "__main__":
    main()
"""
(PYtorch) shakib@shakibhp:~/Desktop/S5/projet/code/PYLON$ CUDA_VISIBLE_DEVICES= python gradcam_nf_multilabel_sample50.py \
  --pylon_repo /home/shakib/Desktop/S5/projet/pylon \
  --ckpt /home/shakib/Desktop/S5/projet/pylon_ckpt/pylon_nih_256.pkl \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --sample_n 50 --sample_mode first \
  --thr 0.10 --tau_nf 0.07 \
  --topk_cam 1 \
  --target_layer layer4_first \
  --cam_border_suppress 0.05 \
  --out_dir /home/shakib/Desktop/S5/projet/xai_outputs/gradcam_nf_sample50 \
  --cpu
[INFO] Device: cpu
[INFO] thr: 0.1 | tau_nf: 0.07 | sample_n: 50 | mode: first
[INFO] topk CAM: 1 | target_layer: layer4_first
[INFO] split_list loaded: /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt (n=25596)
[INFO] sampled 50 images.
[INFO] CSV filtered rows: 50
[INFO] building filename index (images_*/images/*.png)...
[INFO] index size: 112120
[DEBUG] pylon loaded from: /home/shakib/Desktop/S5/projet/pylon/pylon/pylon.py
[LOAD] Missing=0 Unexpected=0
[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)
[DEBUG] Patched decoder.forward (custom, ups=['up3', 'up2', 'up1'])
[INFO] processed 10/50...
[INFO] processed 20/50...
[INFO] processed 30/50...
[INFO] processed 40/50...
[INFO] processed 50/50...

[MULTILABEL NIH14 METRICS]
thr=0.100
micro  P=0.413 R=0.506 F1=0.455
macro  P=0.243 R=0.249 F1=0.230
macro AUROC=0.651 | macro mAP=0.389

[SINGLE-LABEL 15 (Dominant GT vs Dominant Pred incl. No Finding)]
Accuracy15=0.480 | Macro-F1-15=0.285
Note: Dominant_GT uses 'best GT label by model prob' when GT is multilabel.

[DONE] Saved CSV: /home/shakib/Desktop/S5/projet/xai_outputs/gradcam_nf_sample50/preds_with_cam_sample.csv
[DONE] Saved overlays dir: /home/shakib/Desktop/S5/projet/xai_outputs/gradcam_nf_sample50/overlays
[INFO] processed=50 (sample) skipped_missing=0



"""