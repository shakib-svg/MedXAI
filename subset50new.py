#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


NIH_FINDINGS = [
    "No Finding",
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax",
]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\ufeff", "") for c in df.columns]
    # Harmonisation : certains exports ont des espaces multiples
    rename_map = {}
    for c in df.columns:
        cs = " ".join(c.split())
        rename_map[c] = cs
    df = df.rename(columns=rename_map)

    # S'assure que les colonnes clés existent (format NIH standard)
    # Colonnes attendues : "Image Index", "Finding Labels"
    if "Image Index" not in df.columns:
        # tentative de récupération si colonnes mal parsées
        # (dans certains cas, le séparateur ou l'encodage casse les entêtes)
        possible = [c for c in df.columns if "Image" in c and "Index" in c]
        if possible:
            df = df.rename(columns={possible[0]: "Image Index"})
    if "Finding Labels" not in df.columns:
        possible = [c for c in df.columns if "Finding" in c and "Label" in c]
        if possible:
            df = df.rename(columns={possible[0]: "Finding Labels"})

    if "Image Index" not in df.columns or "Finding Labels" not in df.columns:
        raise ValueError(
            f"Colonnes introuvables. Colonnes détectées: {list(df.columns)}. "
            f"Le CSV NIH attendu contient au minimum 'Image Index' et 'Finding Labels'."
        )

    return df


def build_image_index(data_root: Path, cache_json: Path | None = None) -> dict[str, str]:
    """
    Construit un mapping: filename -> chemin absolu
    en scannant images_*/images/*.png
    """
    if cache_json and cache_json.exists():
        return json.loads(cache_json.read_text())

    image_map: dict[str, str] = {}
    for images_dir in sorted(data_root.glob("images_*/images")):
        if images_dir.is_dir():
            for p in images_dir.glob("*.png"):
                image_map[p.name] = str(p.resolve())

    if not image_map:
        raise FileNotFoundError(
            f"Aucune image trouvée sous {data_root}/images_*/images/*.png"
        )

    if cache_json:
        cache_json.parent.mkdir(parents=True, exist_ok=True)
        cache_json.write_text(json.dumps(image_map, indent=2))

    return image_map


def parse_labels(label_str: str) -> list[str]:
    parts = [p.strip() for p in str(label_str).split("|")]
    # normalisation mineure
    parts = [p.replace("Pleural Thickening", "Pleural_Thickening") for p in parts]
    return parts


def select_indices(df: pd.DataFrame, n: int, mode: str, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)

    labels_list = df["Finding Labels"].apply(parse_labels)

    if mode == "single_label_balanced":
        single_mask = labels_list.apply(lambda x: len(x) == 1)
        df2 = df[single_mask].copy()
        labels2 = labels_list[single_mask].apply(lambda x: x[0])

    elif mode == "primary_label_balanced":
        df2 = df.copy()
        labels2 = labels_list.apply(lambda x: x[0] if len(x) else "No Finding")

    else:
        raise ValueError("mode doit être 'single_label_balanced' ou 'primary_label_balanced'")

    # classes disponibles (intersection avec liste NIH standard si possible)
    available = sorted(set(labels2.unique()))
    # Optionnel: ordonner selon NIH_FINDINGS (si présent) + le reste
    ordered = [c for c in NIH_FINDINGS if c in available] + [c for c in available if c not in NIH_FINDINGS]

    per_class = {c: df2.index[labels2 == c].to_numpy() for c in ordered}

    # allocation équilibrée
    k_base = max(1, n // max(1, len(ordered)))
    selected = []

    for c in ordered:
        candidates = per_class[c]
        if len(candidates) == 0:
            continue
        k = min(k_base, len(candidates))
        pick = rng.choice(candidates, size=k, replace=False).tolist()
        selected.extend(pick)

    # complète si on n'a pas atteint n
    if len(selected) < n:
        remaining_pool = np.array(list(set(df2.index.to_numpy()) - set(selected)))
        if len(remaining_pool) < (n - len(selected)):
            raise RuntimeError(
                f"Pool insuffisant: sélection {len(selected)} + remaining {len(remaining_pool)} < {n}."
            )
        extra = rng.choice(remaining_pool, size=(n - len(selected)), replace=False).tolist()
        selected.extend(extra)

    # si dépassement (rare), on coupe
    selected = selected[:n]

    return pd.Index(selected)


def subset_bbox(bbox_csv: Path, selected_filenames: set[str], out_path: Path) -> None:
    bbox = pd.read_csv(bbox_csv)
    bbox.columns = [c.strip().replace("\ufeff", "") for c in bbox.columns]
    # NIH BBox a souvent une colonne 'Image Index'
    if "Image Index" not in bbox.columns:
        possible = [c for c in bbox.columns if "Image" in c and "Index" in c]
        if possible:
            bbox = bbox.rename(columns={possible[0]: "Image Index"})
    if "Image Index" not in bbox.columns:
        raise ValueError(f"Colonne 'Image Index' introuvable dans {bbox_csv}")

    bbox_sub = bbox[bbox["Image Index"].isin(selected_filenames)].copy()
    bbox_sub.to_csv(out_path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True,
                    help="Racine du dataset NIH (contient images_001, images_002, ..., Data_Entry_2017.csv)")
    ap.add_argument("--csv", type=str, required=True, help="Chemin vers Data_Entry_2017.csv")
    ap.add_argument("--out_dir", type=str, required=True, help="Dossier de sortie pour la mini-dataset")
    ap.add_argument("--n", type=int, default=50, help="Nombre d'images à sélectionner")
    ap.add_argument("--seed", type=int, default=42, help="Seed aléatoire")
    ap.add_argument("--mode", type=str, default="single_label_balanced",
                    choices=["single_label_balanced", "primary_label_balanced"],
                    help="Mode de stratification")
    ap.add_argument("--bbox_csv", type=str, default=None,
                    help="(Optionnel) chemin vers BBox_List_2017.csv pour produire un subset bbox")
    ap.add_argument("--cache_index", action="store_true",
                    help="Cache l'index des images pour accélérer les runs suivants")
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser().resolve()
    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    out_images = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    # 1) Charger CSV
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)

    # 2) Index images
    cache_json = (out_dir / "_cache" / "image_index.json") if args.cache_index else None
    image_map = build_image_index(data_root, cache_json=cache_json)

    # 3) Sélection stratifiée
    idx = select_indices(df, n=args.n, mode=args.mode, seed=args.seed)
    sub = df.loc[idx].copy()

    # 4) Copier images + écrire src_path
    copied = 0
    missing = []
    src_paths = []
    for fn in sub["Image Index"].tolist():
        if fn not in image_map:
            missing.append(fn)
            src_paths.append("")
            continue
        src = Path(image_map[fn])
        dst = out_images / fn
        shutil.copy2(src, dst)
        src_paths.append(str(dst.resolve()))
        copied += 1

    sub["src_path"] = src_paths

    # 5) Sauver CSV subset
    out_csv = out_dir / "Data_Entry_2017_subset.csv"
    sub.to_csv(out_csv, index=False)

    # 6) Stats par classe
    labels = sub["Finding Labels"].apply(parse_labels)
    if args.mode == "single_label_balanced":
        primary = labels.apply(lambda x: x[0] if len(x) else "No Finding")
    else:
        primary = labels.apply(lambda x: x[0] if len(x) else "No Finding")

    stats = primary.value_counts().sort_index()
    stats_path = out_dir / "class_counts.txt"
    stats_path.write_text(stats.to_string() + "\n")

    # 7) Optionnel: subset BBox
    if args.bbox_csv:
        bbox_csv = Path(args.bbox_csv).expanduser().resolve()
        out_bbox = out_dir / "BBox_List_2017_subset.csv"
        subset_bbox(bbox_csv, set(sub["Image Index"].tolist()), out_bbox)

    # 8) Log
    log = [
        f"data_root={data_root}",
        f"csv={csv_path}",
        f"out_dir={out_dir}",
        f"n={args.n}",
        f"seed={args.seed}",
        f"mode={args.mode}",
        f"copied={copied}",
        f"missing={len(missing)}",
    ]
    (out_dir / "subset_log.txt").write_text("\n".join(log) + "\n")
    if missing:
        (out_dir / "missing_images.txt").write_text("\n".join(missing) + "\n")

    print("[DONE] Subset créé")
    print(f"  Images copiées: {copied}/{args.n}")
    print(f"  CSV subset: {out_csv}")
    print(f"  Counts: {stats_path}")
    if args.bbox_csv:
        print(f"  BBox subset: {out_dir / 'BBox_List_2017_subset.csv'}")


if __name__ == "__main__":
    main()




#python3 subset50new.py \
#  --data_root . \
  #--csv Data_Entry_2017.csv \
  #--out_dir ~/Desktop/S5/projet/subset50 \
  #--n 50 \
  #--seed 42 \
  #--mode single_label_balanced \
  #--cache_index
