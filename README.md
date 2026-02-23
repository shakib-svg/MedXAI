# MedXAI

**MedXAI** is a script-based collection for **evaluation + explainability (XAI)** on **multi-label chest X-ray classification** using the **NIH ChestX-ray14 (NIH14)** label set. It includes utilities to generate:

- **Predictions CSVs** (per-image outputs + probabilities)
- **Metrics JSON** (micro/macro P/R/F1, AUROC, mAP, etc.)
- **Explanation maps / overlays** (Grad-CAM, Grad-CAM++, Integrated Gradients, LRP)

Two model “tracks” are covered:

1. **CheXNet-style DenseNet121** (14 NIH labels) — “CheXNet” scripts.
2. A **Pylon** model loaded from an external local repo — “Pylon” scripts.

> ⚠️ **Research / educational code only. Not for clinical use.**  
> Outputs are not medical advice and should not be used for diagnosis or treatment.

---

## Table of contents

- [What’s inside](#whats-inside)
- [Methods](#methods)
- [Dataset expectations](#dataset-expectations)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Scripts](#scripts)
  - [CheXNet](#chexnet)
  - [Pylon](#pylon)
  - [Subset / utilities](#subset--utilities)
- [Outputs](#outputs)
- [Repository layout](#repository-layout)
- [Included example outputs](#included-example-outputs)
- [Notes on metrics](#notes-on-metrics)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## What’s inside

This repo is **script-first** (no Python package). It includes:

- Evaluation on NIH14 (+ a “No Finding” rule)  
- Sampling (e.g., random 500, or a curated/stratified subset)
- Explanation map generation and saving:
  - overlayed on the image
  - optionally raw maps / heat-only images

Many folders in the repo are **saved run outputs** (CSV, metrics, and overlay images).

---

## Methods

Implemented/used across scripts:

- **Grad-CAM**
- **Grad-CAM++**
- **Integrated Gradients**
- **Layer-wise Relevance Propagation (LRP)** (multiple variants / settings)

---

## Dataset expectations

Most scripts assume the NIH archive layout such as:

- `Data_Entry_2017.csv`
- `test_list.txt` (one filename per line)
- Images under either:
  - `images_*/images/*.png` (NIH14 standard archive layout)
  - or a flat `images/` directory
  - scripts usually include a fallback recursive scan if needed

**NIH14 labels** used in this repo:

`Atelectasis, Cardiomegaly, Effusion, Infiltration, Mass, Nodule, Pneumonia, Pneumothorax, Consolidation, Edema, Emphysema, Fibrosis, Pleural_Thickening, Hernia`

A **“No Finding”** label is derived with a simple rule:
- If the **maximum** predicted probability across NIH14 is below `tau_nf`, the predicted dominant class is set to **No Finding**.

---

## Installation

Create a virtual environment (recommended), then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# core
pip install numpy pandas pillow opencv-python scikit-learn

# deep learning (install the right build for your CUDA/CPU)
pip install torch torchvision

```
1) CheXNet evaluation (random sample of 500)
```bash
python3 chexnet_eval_500.py \
  --ckpt /path/to/chexnet_checkpoint.pth.tar \
  --csv /path/to/Data_Entry_2017.csv \
  --images_root /path/to/NIH/archive \
  --split_list /path/to/test_list.txt \
  --out_dir ./xai_outputs/chexnet_eval_random500_seed0 \
  --thr 0.1 \
  --tau_nf 0.07 \
  --sample_n 500 --sample_mode random --seed 0
```
2) CheXNet Grad-CAM (with overlays)
```bash
python3 chexnet_gradcam_500.py \
  --ckpt /path/to/chexnet_checkpoint.pth.tar \
  --csv /path/to/Data_Entry_2017.csv \
  --images_root /path/to/NIH/archive \
  --split_list /path/to/test_list.txt \
  --out_dir ./xai_outputs/chexnet_gradcam_random500_seed0 \
  --sample_n 500 --sample_mode random --seed 0 \
  --topk_cam 1 \
  --thr 0.1 \
  --target_layer features.denseblock4
```
3) Pylon evaluation
```bash
python3 eval_nih_multilabel.py \
  --pylon_repo /path/to/pylon_repo \
  --ckpt /path/to/pylon_checkpoint.pkl \
  --csv /path/to/Data_Entry_2017.csv \
  --images_root /path/to/NIH/archive \
  --split_list /path/to/test_list.txt \
  --out_dir ./xai_outputs/output_pylon \
  --img_size 256
```
## Scripts

### CheXNet

- `chexnet_eval_500.py`  
  Evaluate CheXNet on NIH test split (optionally sampled). Saves CSV + metrics.

- `chexnet_gradcam_500.py`  
  CheXNet Grad-CAM on a sample. Saves overlays and debug CSV with per-class probabilities.

- `chexnet_campp_500.py`  
  CheXNet Grad-CAM++ (CAM++). Saves overlays + CSV + metrics.

- `chexnet_ig_500.py`  
  CheXNet Integrated Gradients on a sample. Saves IG maps + overlays + CSV + metrics.

- `chexnet_lrp_500.py`, `chexnet_lrp.py`  
  CheXNet LRP variants.

There are also older / helper files:

- `chexnet_gradcampp.py`
- `chexnet_integrated_gradients.py`
- `laod_chexnet.py` (helper; filename has a typo)

### Pylon

These scripts load a local Pylon repo dynamically (via a `--pylon_repo` path) and run evaluation / explainability.

- `eval_nih_multilabel.py`  
  Full evaluation for Pylon, saves CSV + `.npz` outputs.

- `pylon_gradcampp_eval_500.py`  
  Grad-CAM++ for Pylon + evaluation + overlays (sample 500).

- `pylon_integrated_gradients_500.py`  
  Integrated Gradients for Pylon (sample 500).

- `pylon_lrp_500.py`  
  LRP for Pylon (sample 500).

- `gradcam_nih_multilabel_pylon.py`  
  Pylon Grad-CAM pipeline (multi-label NIH).

### Subset / utilities

- `subset50new.py`  
  Build a balanced NIH subset (default 50 images), copy images, write a subset CSV, and optionally a subset of bounding boxes.

- `effacer50.py`  
  Utility to delete images not in a hard-coded keep-list (use carefully).

- `inspect_ckpt.py`  
  Placeholder / helper (currently minimal in this repo snapshot).

- `gradcam_nf_multilabel_sample50.py`  
  A Grad-CAM pipeline that runs on a 50-image subset and writes overlays + metrics.

---

## Outputs

Most evaluation/XAI scripts write:

- `preds_with_*.csv` — per-image rows, GT labels, predicted labels, top-1, etc.
- `metrics.json` — summary metrics + args + runtime
- `overlays/` — PNG overlays (original + heatmap)
- sometimes `raw/` — raw attribution maps or heat-only images (optional)

---

## Repository layout

Top-level (abridged):

```text
.
├── campp_pylon_500/                         # saved run output(s)
├── chexnet_eval_random500_seed0/            # saved run output(s)
├── chexnet_gradcam_fixed_random500_seed0/   # saved run output(s)
├── chexnet_campp_random500_seed0/           # saved run output(s)
├── chexnet_ig_random500_seed0/              # saved run output(s)
├── chexnet_lrp_random500_seed0/             # saved run output(s)
├── gradcam_nf_sample50/                     # saved run output(s)
├── gradcam_pylon_final_random500_seed0_tau007/
├── gradcam_pylon_sample50_best_tau012/
├── iG_Pylon_500/
├── lrp_approx_final_random500_seed0/
├── output_pylon/
├── chexnet_eval_500.py
├── chexnet_gradcam_500.py
├── chexnet_campp_500.py
├── chexnet_ig_500.py
├── chexnet_lrp_500.py
├── eval_nih_multilabel.py
├── pylon_gradcampp_eval_500.py
├── pylon_integrated_gradients_500.py
├── pylon_lrp_500.py
├── subset50new.py
└── ...

```
## Included example outputs

This repo already contains several output folders (CSV + metrics + overlays).  
For example, a committed CheXNet evaluation run (`chexnet_eval_random500_seed0/metrics.json`) shows:

- `thr=0.1`, `tau_nf=0.07`, processed `500`
- macro AUROC and mAP reported in `metrics.json`

(Your results will vary depending on checkpoint + environment.)

---

## Notes on metrics

Most scripts report:

- Micro precision/recall/F1 (global)
- Macro precision/recall/F1 (per-class average)
- Macro AUROC and macro mAP (when scikit-learn is available)

Some scripts also compute an auxiliary “dominant-15” score:

- treat prediction as one “dominant label” among 14 + “No Finding”
- compare to a “dominant GT” derived from GT labels and model probabilities

This is useful for sanity checks, but NIH14 is inherently multi-label.

---

## Troubleshooting

### 1) “Missing keys / classifier not loaded”

Some CheXNet scripts include robust key-remapping for older checkpoints and will error if the classifier is not properly loaded. Verify:

- you used the correct checkpoint
- key prefixes (e.g., `module.`) are handled
- classifier weights are present

### 2) Images not found

Make sure `--images_root` points to the NIH archive root containing `images_*/images/`.  
If you use a custom subset layout, keep the filenames consistent with `Data_Entry_2017.csv` / `test_list.txt`.

### 3) OpenCV import issues

If `import cv2` fails, reinstall:

```bash
pip install --force-reinstall opencv-python
::contentReference[oaicite:0]{index=0}
```
