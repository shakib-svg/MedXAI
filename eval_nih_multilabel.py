#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import sys
import importlib.util
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import cv2

# -----------------------
# Labels NIH14
# -----------------------
NIH14 = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
    "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
    "Pleural_Thickening", "Hernia"
]
NO_FINDING = "No Finding"


# -----------------------
# Utils
# -----------------------
def pick_device(force_cpu: bool = False):
    if force_cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def torch_load_compat(path: str):
    import inspect
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        return torch.load(path, map_location="cpu", weights_only=False)
    return torch.load(path, map_location="cpu")


def load_image_gray_tensor(img_path: Path, size: int = 256):
    """
    Charge une image RGB, resize (size,size), puis convertit en grayscale (mean des canaux).
    Sort: Tensor [1,1,H,W] float32 dans [0,1]
    """
    img = Image.open(img_path).convert("RGB")
    rgb = np.asarray(img).astype(np.float32) / 255.0
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    x = x.mean(dim=1, keepdim=True)                           # [1,1,H,W]
    return x


def parse_labels(label_str: str):
    s = str(label_str).strip()
    parts = [p.strip() for p in s.split("|") if p.strip()]
    if not parts:
        return []
    parts = [p.replace("Pleural Thickening", "Pleural_Thickening") for p in parts]
    # NIH CSV utilise "No Finding" comme label explicite pour normal
    if len(parts) == 1 and parts[0] == NO_FINDING:
        return []
    return parts


def labels_to_vec(labels_list):
    v = np.zeros(len(NIH14), dtype=np.int32)
    for lab in labels_list:
        if lab in NIH14:
            v[NIH14.index(lab)] = 1
    return v


def find_image_in_archive(images_root: Path, filename: str) -> Path:
    """
    NIH: images_root contient images_001/images/, images_002/images/, etc.
    On cherche filename dans images_root/**/images/filename.
    (Optimisable avec index map, mais OK pour démarrer.)
    """
    # cas où images_root pointe déjà vers un dossier "images" (ex subset50/images)
    direct = images_root / filename
    if direct.exists():
        return direct.resolve()

    # cas NIH archive: images_###/images/...
    hits = list(images_root.glob(f"images_*/images/{filename}"))
    if hits:
        return hits[0].resolve()

    # fallback plus large (lent)
    hits = list(images_root.rglob(filename))
    if hits:
        return hits[0].resolve()

    return None


# -----------------------
# Import Pylon from repo (robuste)
# -----------------------
def import_pylon_from_repo(pylon_repo: Path):
    """
    pylon_repo = /home/.../projet/pylon
    Le fichier réel est : pylon_repo/pylon/pylon.py
    Et "trainer" est : pylon_repo/pylon/trainer/...
    Donc on ajoute pylon_repo/pylon dans sys.path pour résoudre "trainer".
    Puis on charge pylon.py via importlib (évite conflits de nom).
    """
    pylon_repo = pylon_repo.resolve()
    pkg_root = pylon_repo / "pylon"
    pylon_py = pkg_root / "pylon.py"
    if not pylon_py.exists():
        raise FileNotFoundError(f"Introuvable: {pylon_py}")

    # Pour que "from trainer.start import *" marche
    sys.path.insert(0, str(pkg_root))

    spec = importlib.util.spec_from_file_location("pylon_local", str(pylon_py))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    print(f"[DEBUG] pylon loaded from: {pylon_py}")
    return mod


def build_and_load_model(pylon_repo: Path, ckpt: Path, n_in: int = 1, n_out: int = 14):
    pylon_mod = import_pylon_from_repo(pylon_repo)
    if not hasattr(pylon_mod, "PylonConfig") or not hasattr(pylon_mod, "Pylon"):
        raise RuntimeError("Le module pylon chargé ne contient pas PylonConfig/Pylon (import cassé).")

    conf = pylon_mod.PylonConfig(n_in=n_in, n_out=n_out)
    model = pylon_mod.Pylon(conf)

    ckpt_obj = torch_load_compat(str(ckpt))
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
# Patch inference (ton cas)
# -----------------------
def patch_pylon_for_inference(model, up_max: int = 3):
    """
    Patch minimal pour ton repo:
    - decoder.pa.forward : si on reçoit list/tuple -> on prend bottleneck = last tensor
    - decoder.forward : version custom qui utilise up3/up2/up1 (up_max=3) avec skips features[-2],[-3],[-4]
    """
    net = model.net
    dec = net.decoder

    # Patch PA: list->tensor
    if hasattr(dec, "pa") and hasattr(dec.pa, "forward"):
        orig_pa_fwd = dec.pa.forward

        def pa_forward_patched(x, *args, **kwargs):
            if isinstance(x, (list, tuple)):
                x = x[-1]  # bottleneck tensor
            return orig_pa_fwd(x, *args, **kwargs)

        dec.pa.forward = pa_forward_patched
        print("[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)")

    # Custom decoder forward
    ups = []
    for name in ["up4", "up3", "up2", "up1", "up0"]:
        if hasattr(dec, name):
            ups.append(name)

    # on prend les "up" les plus hautes comme dans ton fix (up3, up2, up1)
    # si up4 existe mais casse, on l'ignore par défaut (up_max=3)
    chosen = []
    for cand in ["up3", "up2", "up1", "up0", "up4"]:
        if hasattr(dec, cand):
            chosen.append(cand)
    chosen = chosen[:up_max]

    def decoder_forward_custom(features, *args, **kwargs):
        # Normalise features -> list[Tensor]
        if isinstance(features, tuple) and len(features) == 1:
            features = features[0]
        if not isinstance(features, (list, tuple)):
            raise TypeError(f"decoder expected list/tuple of tensors, got {type(features)}")

        feats = list(features)
        # Some encoders include input image at feats[0] (c=1). C'est OK: on utilise les derniers.
        bottleneck = feats[-1]
        x = dec.pa(bottleneck) if hasattr(dec, "pa") else bottleneck

        # up modules: up3 prend skip feats[-2], up2 -> feats[-3], up1 -> feats[-4], etc.
        for i, up_name in enumerate(chosen):
            up = getattr(dec, up_name)
            skip = feats[-(i + 2)]
            x = up(skip, x)

        return x

    dec.forward = decoder_forward_custom
    print(f"[DEBUG] Patched decoder.forward (custom, up_max={up_max}, ups={chosen})")


# -----------------------
# Metrics (multilabel)
# -----------------------
def multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr: float):
    """
    y_true: [N,14] {0,1}
    y_prob: [N,14] [0,1]
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score,
        precision_recall_fscore_support
    )

    y_pred = (y_prob >= thr).astype(np.int32)

    # micro/macro P/R/F1
    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true.ravel(), y_pred.ravel(), average="binary", zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    # AUROC + AP par classe (skip si pas de positifs ou pas de négatifs)
    aurocs = {}
    aps = {}
    for j, name in enumerate(NIH14):
        yt = y_true[:, j]
        yp = y_prob[:, j]
        if yt.max() == yt.min():
            # tous 0 ou tous 1 => ROC AUC indéfini
            continue
        aurocs[name] = float(roc_auc_score(yt, yp))
        aps[name] = float(average_precision_score(yt, yp))

    macro_auc = float(np.mean(list(aurocs.values()))) if aurocs else float("nan")
    macro_ap = float(np.mean(list(aps.values()))) if aps else float("nan")

    return {
        "thr": thr,
        "micro_precision": float(p_micro),
        "micro_recall": float(r_micro),
        "micro_f1": float(f1_micro),
        "macro_precision": float(p_macro),
        "macro_recall": float(r_macro),
        "macro_f1": float(f1_macro),
        "macro_auroc": macro_auc,
        "macro_map": macro_ap,
        "per_class_auroc": aurocs,
        "per_class_ap": aps,
    }


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pylon_repo", required=True, help="/.../projet/pylon")
    ap.add_argument("--ckpt", required=True, help="pylon_nih_256.pkl")
    ap.add_argument("--csv", required=True, help="Data_Entry_2017.csv ou subset")
    ap.add_argument("--images_root", required=True, help="/.../Dataset/archive (contient images_001..)")
    ap.add_argument("--split_list", default=None, help="Optionnel: test_list.txt ou train_val_list.txt (1 filename/ligne)")
    ap.add_argument("--img_size", type=int, default=256)
    ap.add_argument("--thr", type=float, default=0.1, help="seuil global multilabel")
    ap.add_argument("--up_max", type=int, default=3, help="patch decoder: nb up blocks (3 marche chez toi)")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out_pred_csv", default=None, help="Optionnel: CSV avec probs/preds")
    ap.add_argument("--cache_npz", default=None, help="Optionnel: sauvegarde y_true/y_prob/filenames")
    args = ap.parse_args()

    pylon_repo = Path(args.pylon_repo)
    ckpt = Path(args.ckpt)
    csv_path = Path(args.csv)
    images_root = Path(args.images_root)

    device = pick_device(force_cpu=args.cpu)
    print("[INFO] Device:", device)
    print("[INFO] thr:", args.thr)

    df = pd.read_csv(csv_path, engine="python")
    df.columns = [c.strip() for c in df.columns]

    img_col = "Image Index" if "Image Index" in df.columns else ("Image_Index" if "Image_Index" in df.columns else None)
    lab_col = "Finding Labels" if "Finding Labels" in df.columns else ("Finding_Labels" if "Finding_Labels" in df.columns else None)
    if img_col is None or lab_col is None:
        raise ValueError("CSV: colonnes attendues introuvables (Image Index / Finding Labels)")

    # Filtre split (optionnel)
    allow = None
    if args.split_list:
        split_path = Path(args.split_list)
        allow = set([line.strip() for line in split_path.read_text().splitlines() if line.strip()])
        print(f"[INFO] split_list loaded: {split_path} (n={len(allow)})")

    # modèle
    model = build_and_load_model(pylon_repo, ckpt, n_in=1, n_out=14)
    patch_pylon_for_inference(model, up_max=args.up_max)
    model = model.to(device).eval()

    filenames = []
    y_true = []
    y_prob = []
    rows = []

    skipped_missing = 0
    processed = 0

    for _, row in df.iterrows():
        fn = str(row[img_col]).strip()
        if allow is not None and fn not in allow:
            continue

        img_path = find_image_in_archive(images_root, fn)
        if img_path is None or not Path(img_path).exists():
            skipped_missing += 1
            continue

        labels = parse_labels(str(row[lab_col]))
        gt = labels_to_vec(labels)

        x = load_image_gray_tensor(Path(img_path), size=args.img_size).to(device)

        with torch.no_grad():
            out = model(x)
            logits = out.pred  # [1,14]
            prob = torch.sigmoid(logits)[0].detach().cpu().numpy()

        filenames.append(fn)
        y_true.append(gt)
        y_prob.append(prob)

        if args.out_pred_csv:
            pred_idx = np.where(prob >= args.thr)[0].tolist()
            pred_labels = [NIH14[i] for i in pred_idx]
            rows.append({
                "Image": fn,
                "GT_labels": "|".join(labels) if labels else NO_FINDING,
                "Pred_labels": "|".join(pred_labels) if pred_labels else NO_FINDING,
                **{f"p_{NIH14[j]}": float(prob[j]) for j in range(14)},
            })

        processed += 1
        if processed % 5000 == 0:
            print(f"[INFO] processed {processed} images...")

    if processed == 0:
        print("[ERROR] 0 image traitée. Vérifie images_root / split_list.")
        return

    y_true = np.stack(y_true, axis=0).astype(np.int32)
    y_prob = np.stack(y_prob, axis=0).astype(np.float32)

    # métriques
    m = multilabel_metrics(y_true, y_prob, thr=args.thr)

    print("\n[MULTILABEL METRICS]")
    print(f"thr={m['thr']:.3f}")
    print(f"micro  P={m['micro_precision']:.3f} R={m['micro_recall']:.3f} F1={m['micro_f1']:.3f}")
    print(f"macro  P={m['macro_precision']:.3f} R={m['macro_recall']:.3f} F1={m['macro_f1']:.3f}")
    print(f"macro AUROC={m['macro_auroc']:.3f} | macro mAP={m['macro_map']:.3f}")

    # affichage auroc/ap par classe (tri AUROC)
    if m["per_class_auroc"]:
        print("\nPer-class AUROC (top->low):")
        for k, v in sorted(m["per_class_auroc"].items(), key=lambda kv: -kv[1]):
            apv = m["per_class_ap"].get(k, float("nan"))
            print(f"  {k:>18s}  AUROC={v:.3f}  AP={apv:.3f}")

    # sorties
    if args.out_pred_csv:
        out_csv = Path(args.out_pred_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print("\n[DONE] Saved preds CSV:", out_csv)

    if args.cache_npz:
        cache = Path(args.cache_npz)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, filenames=np.array(filenames), y_true=y_true, y_prob=y_prob)
        print("[DONE] Saved cache NPZ:", cache)

    print(f"\n[INFO] processed={processed} skipped_missing={skipped_missing}")


if __name__ == "__main__":
    main()


"""
(PYtorch) shakib@shakibhp:~/Desktop/S5/projet/code/PYLON$ CUDA_VISIBLE_DEVICES= python eval_nih_multilabel.py \
  --pylon_repo /home/shakib/Desktop/S5/projet/pylon \
  --ckpt /home/shakib/Desktop/S5/projet/pylon_ckpt/pylon_nih_256.pkl \
  --csv /home/shakib/Desktop/S5/projet/Dataset/archive/Data_Entry_2017.csv \
  --images_root /home/shakib/Desktop/S5/projet/Dataset/archive \
  --split_list /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt \
  --thr 0.1 \
  --out_pred_csv /home/shakib/Desktop/S5/projet/xai_outputs/nih_test_preds_thr010.csv \
  --cache_npz /home/shakib/Desktop/S5/projet/xai_outputs/nih_test_cache.npz \
  --cpu
[INFO] Device: cpu
[INFO] thr: 0.1
[INFO] split_list loaded: /home/shakib/Desktop/S5/projet/Dataset/archive/test_list.txt (n=25596)
[DEBUG] pylon loaded from: /home/shakib/Desktop/S5/projet/pylon/pylon/pylon.py
[LOAD] Missing=0 Unexpected=0
[DEBUG] Patched decoder.pa.forward (list/tuple->bottleneck tensor)
[DEBUG] Patched decoder.forward (custom, up_max=3, ups=['up3', 'up2', 'up1'])
[INFO] processed 5000 images...
[INFO] processed 10000 images...
[INFO] processed 15000 images...
[INFO] processed 20000 images...
[INFO] processed 25000 images...

[MULTILABEL METRICS]
thr=0.100
micro  P=0.310 R=0.590 F1=0.406
macro  P=0.287 R=0.431 F1=0.328
macro AUROC=0.802 | macro mAP=0.281

Per-class AUROC (top->low):
              Hernia  AUROC=0.905  AP=0.403
           Emphysema  AUROC=0.890  AP=0.400
        Cardiomegaly  AUROC=0.882  AP=0.344
        Pneumothorax  AUROC=0.849  AP=0.425
               Edema  AUROC=0.834  AP=0.137
            Effusion  AUROC=0.819  AP=0.507
                Mass  AUROC=0.807  AP=0.324
            Fibrosis  AUROC=0.801  AP=0.079
              Nodule  AUROC=0.768  AP=0.249
         Atelectasis  AUROC=0.761  AP=0.335
  Pleural_Thickening  AUROC=0.757  AP=0.126
       Consolidation  AUROC=0.740  AP=0.153
           Pneumonia  AUROC=0.709  AP=0.049
        Infiltration  AUROC=0.706  AP=0.400

[DONE] Saved preds CSV: /home/shakib/Desktop/S5/projet/xai_outputs/nih_test_preds_thr010.csv
[DONE] Saved cache NPZ: /home/shakib/Desktop/S5/projet/xai_outputs/nih_test_cache.npz

[INFO] processed=25596 skipped_missing=0



"""