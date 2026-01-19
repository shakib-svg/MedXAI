import os
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


# ---------------------------------------------------------------------
# Labels NIH14 (ordre standard)
# ---------------------------------------------------------------------
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]


# ---------------------------------------------------------------------
# Model: CheXNet = DenseNet-121 + classifier(14)
# ---------------------------------------------------------------------
def build_chexnet(num_classes: int = 14) -> torch.nn.Module:
    m = models.densenet121(weights=None)
    # IMPORTANT: ton checkpoint est compatible avec classifier Linear direct
    # (on remappe classifier.0 -> classifier.* au chargement)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    return m


def load_chexnet_checkpoint(model: torch.nn.Module, ckpt_path: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise ValueError("Checkpoint invalide: attendu dict avec clé 'state_dict'.")

    sd = ckpt["state_dict"]

    new_sd = {}
    for k, v in sd.items():
        # Enlever préfixes
        if k.startswith("module.densenet121."):
            k = k.replace("module.densenet121.", "", 1)
        if k.startswith("module."):
            k = k.replace("module.", "", 1)

        # Renommer norm.1 -> norm1 ; conv.2 -> conv2 (DenseNet impl différente)
        k = re.sub(r"\.norm\.(\d)\.", r".norm\1.", k)
        k = re.sub(r"\.conv\.(\d)\.", r".conv\1.", k)

        # Tête: classifier.0.* -> classifier.*
        if k.startswith("classifier.0."):
            k = k.replace("classifier.0.", "classifier.", 1)

        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"[LOAD] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    if missing[:5]:
        print("[LOAD] Sample missing:", missing[:5])
    if unexpected[:5]:
        print("[LOAD] Sample unexpected:", unexpected[:5])

    if len(missing) != 0 or len(unexpected) != 0:
        print("[WARN] Le modèle peut fonctionner, mais idéalement Missing/Unexpected = 0.")

    return model


# ---------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
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

    def __call__(self, x: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)  # [B,14]
        score = logits[:, class_idx].sum()
        score.backward(retain_graph=True)

        w = self.gradients.mean(dim=(2, 3), keepdim=True)         # [B,C,1,1]
        cam = (w * self.activations).sum(dim=1, keepdim=True)     # [B,1,h,w]
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        self.h1.remove()
        self.h2.remove()


# ---------------------------------------------------------------------
# Utilities: labels / thresholds / visualization
# ---------------------------------------------------------------------
def parse_findings_label(label_str: str):
    """
    label_str exemple:
      "No Finding"
      "Pneumonia|Pneumothorax"
    """
    if label_str is None or (isinstance(label_str, float) and np.isnan(label_str)):
        return []
    s = str(label_str).strip()
    if s.lower() == "no finding":
        return ["No Finding"]
    parts = [p.strip() for p in s.split("|") if p.strip()]
    return parts


def default_thresholds(n: int, t: float) -> np.ndarray:
    return np.full(n, float(t), dtype=np.float32)


def topk_indices(probs: np.ndarray, k: int):
    k = max(1, min(int(k), len(probs)))
    return np.argsort(-probs)[:k]


def decide_predicted_classes(
    probs: np.ndarray,
    thresholds: np.ndarray,
    allow_no_finding: bool = True
):
    """
    Multi-label decision:
      predicted = {i | prob[i] >= threshold[i]}
      If empty and allow_no_finding: return empty list meaning "No Finding"
    """
    idx = np.where(probs >= thresholds)[0].tolist()
    if len(idx) == 0 and allow_no_finding:
        return []
    return idx


def overlay_cam_on_image(rgb_224: np.ndarray, cam_224: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """
    rgb_224: float [0,1] shape (224,224,3)
    cam_224: float [0,1] shape (224,224)
    returns overlay uint8 BGR for cv2.imwrite
    """
    heat = (cam_224 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    overlay = np.clip((1 - alpha) * rgb_224 + alpha * heat, 0, 1)
    overlay_bgr = (overlay[..., ::-1] * 255).astype(np.uint8)
    return overlay_bgr


# ---------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CheXNet + Multi-label + Grad-CAM (ChestX-ray14)")
    parser.add_argument("--ckpt", required=True, help="Chemin checkpoint CheXNet .pth.tar")
    parser.add_argument("--image", default=None, help="Chemin vers une image (png/jpg/jpeg)")
    parser.add_argument("--img_dir", default=None, help="Dossier d'images (png/jpg/jpeg)")
    parser.add_argument("--out_dir", required=True, help="Dossier de sortie overlays")
    parser.add_argument("--csv", default=None, help="(Optionnel) Data_Entry_2017.csv pour ground truth")
    parser.add_argument("--img_root", default=None, help="(Optionnel) Racine des images si CSV utilisé")
    parser.add_argument("--threshold", type=float, default=0.5, help="Seuil global (si pas de thresholds par classe)")
    parser.add_argument("--topk", type=int, default=5, help="Afficher top-k scores")
    parser.add_argument("--cam_mode", choices=["predicted", "topk", "fixed"], default="predicted",
                        help="Grad-CAM pour classes prédites / topk / une classe fixée")
    parser.add_argument("--fixed_class", default="Pneumonia",
                        help="Si cam_mode=fixed : nom classe à expliquer (ex: Pneumonia)")
    parser.add_argument("--max_images", type=int, default=0,
                        help="Limiter le nombre d'images traitées (0 = pas de limite)")
    args = parser.parse_args()

    if args.image is None and args.img_dir is None:
        raise ValueError("Tu dois fournir --image OU --img_dir.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[INFO] Device:", device)

    # CSV map (optionnel)
    df = None
    label_map = None
    if args.csv is not None:
        df = pd.read_csv(args.csv)
        if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
            raise ValueError("CSV invalide: colonnes attendues: 'Image Index' et 'Finding Labels'")
        label_map = dict(zip(df["Image Index"].astype(str), df["Finding Labels"].astype(str)))
        print("[INFO] CSV loaded:", args.csv, "| rows:", len(df))

    # Model
    model = build_chexnet(14)
    model = load_chexnet_checkpoint(model, args.ckpt).to(device).eval()

    # Grad-CAM target layer (stable)
    target_layer = model.features.denseblock4
    gc = GradCAM(model, target_layer)

    # Preprocess
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    thresholds = default_thresholds(len(NIH14), args.threshold)

    # Build list of image paths
    image_paths = []
    if args.image is not None:
        image_paths = [Path(args.image)]
    else:
        p = Path(args.img_dir)
        exts = {".png", ".jpg", ".jpeg", ".bmp"}
        image_paths = [x for x in sorted(p.rglob("*")) if x.suffix.lower() in exts]

    if args.max_images and args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    print("[INFO] #images:", len(image_paths))

    # Fixed class handling
    fixed_idx = None
    if args.cam_mode == "fixed":
        if args.fixed_class not in NIH14:
            raise ValueError(f"fixed_class invalide. Choisir parmi: {NIH14}")
        fixed_idx = NIH14.index(args.fixed_class)

    # Process loop
    for n, img_path in enumerate(image_paths, 1):
        try:
            raw = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[SKIP] Cannot open {img_path}: {e}")
            continue

        x = tf(raw).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = torch.sigmoid(model(x))[0].detach().cpu().numpy()

        # Show top-k scores
        order = topk_indices(probs, args.topk)
        print(f"\n[{n}/{len(image_paths)}] {img_path.name}")
        for i in order:
            print(f"  {NIH14[i]:<20} {float(probs[i]):.3f}")

        # Ground truth (optionnel)
        gt = None
        if label_map is not None:
            key = img_path.name  # Image Index = filename
            if key in label_map:
                gt = parse_findings_label(label_map[key])
                print("  GT:", gt)
            else:
                print("  GT: (not found in CSV)")

        # Multi-label decision
        predicted_idx = decide_predicted_classes(probs, thresholds, allow_no_finding=True)
        if len(predicted_idx) == 0:
            print("  Pred:", ["No Finding"])
        else:
            print("  Pred:", [NIH14[i] for i in predicted_idx])

        # Choix des classes à expliquer
        if args.cam_mode == "predicted":
            cam_classes = predicted_idx if len(predicted_idx) > 0 else list(order[:1])  # si vide, expliquer top1
        elif args.cam_mode == "topk":
            cam_classes = list(order[:args.topk])
        else:  # fixed
            cam_classes = [fixed_idx]

        # Prépare image 224 pour overlay
        raw_224 = raw.resize((224, 224))
        raw_np = np.asarray(raw_224).astype(np.float32) / 255.0  # RGB [0,1]

        # Générer overlays par classe
        for ci in cam_classes:
            cam = gc(x, ci)
            overlay_bgr = overlay_cam_on_image(raw_np, cam, alpha=0.4)

            stem = img_path.stem
            cname = NIH14[ci]
            out_name = f"{stem}__cam_{cname}.png"
            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), overlay_bgr)

        print(f"  Saved overlays: {len(cam_classes)} -> {out_dir}")

    gc.close()
    print("\n[DONE] Finished.")


if __name__ == "__main__":
    main()



