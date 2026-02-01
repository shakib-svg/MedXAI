#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
from pathlib import Path
import pandas as pd


def pick(df, n, sort_col="Top1_score", descending=True):
    if len(df) == 0:
        return df
    return df.sort_values(sort_col, ascending=not descending).head(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_csv", required=True, help="results_gradcam.csv")
    ap.add_argument("--overlays_dir", required=True, help="dossier overlays/")
    ap.add_argument("--out_dir", required=True, help="dossier de sortie (examples)")
    ap.add_argument("--k", type=int, default=4, help="nb d'exemples par catégorie (default=4 -> 12 total)")
    args = ap.parse_args()

    res = Path(args.results_csv).expanduser().resolve()
    ov = Path(args.overlays_dir).expanduser().resolve()
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(res)
    for c in ["GT", "Pred", "Top1_score", "Overlay_Files"]:
        if c not in df.columns:
            raise ValueError(f"Colonne manquante: {c}")

    # catégories
    correct = df[df["GT"] == df["Pred"]].copy()
    wrong = df[df["GT"] != df["Pred"]].copy()
    nf_fp = df[(df["GT"] == "No Finding") & (df["Pred"] != "No Finding")].copy()

    # on prend des exemples “confiants” (Top1_score élevé)
    sel_correct = pick(correct, args.k)
    sel_wrong = pick(wrong, args.k)
    sel_nf_fp = pick(nf_fp, args.k)

    selected = pd.concat([sel_correct.assign(group="correct"),
                          sel_wrong.assign(group="wrong"),
                          sel_nf_fp.assign(group="nofinding_fp")], ignore_index=True)

    # copie overlays
    copied = 0
    kept_rows = []
    for _, r in selected.iterrows():
        files = str(r["Overlay_Files"]).split("|") if pd.notna(r["Overlay_Files"]) else []
        files = [f for f in files if f.strip()]
        if not files:
            continue

        # pred -> généralement 1 overlay
        f0 = files[0]
        src = ov / f0
        if not src.exists():
            continue

        dst = out / f"{r['group']}__{f0}"
        shutil.copy2(src, dst)
        copied += 1

        kept_rows.append({
            "group": r["group"],
            "Image": r["Image"],
            "GT": r["GT"],
            "Pred": r["Pred"],
            "Top1": r.get("Top1", ""),
            "Top1_score": r["Top1_score"],
            "overlay_file": dst.name
        })

    out_csv = out / "selected_examples.csv"
    pd.DataFrame(kept_rows).to_csv(out_csv, index=False)

    print("[DONE] Copied overlays:", copied)
    print("[DONE] Summary CSV:", out_csv)
    print("Open folder:", out)


if __name__ == "__main__":
    main()
