# One-click demo — OCT volume inference (CNN / MoE / CNN+Transformer)

This repo lets you run **inference** on macular OCT volumes with 3 models:
- CNN (per-slice) + majority vote
- CNN + Mixture-of-Experts (MoE)
- CNN + Transformer fusion

**Goal:** readers of our paper can test the models in a few minutes.

---

## 0) What you need
- Python 3.10+  
- (Recommended) a GPU with CUDA; CPU works too (slower)

---

## 1) Quick start (the simplest way)
### A) Clone
```bash
git clone https://github.com/KMairot/TestAI.git
cd TestAI
```
---

## 2) Install
```bash
pip install -r requirements.txt
```

## 3) Run the notebook
```bash
jupyter notebook
```
Open the notebook of your choice : 
1. demo_cnn_vote.ipynb (CNN+vote)

2. demo_moe.ipynb (MoE)

3. demo_hybrid.ipynb (hybride) and click Run All.

The notebook will:

1. download pretrained .pt files from GitHub Releases (v1.0)

2. run inference on your OCT slices

3. save outputs/predictions_<model>.csv

---

## Recommended (to avoid filename/volume bugs): convert `.E2E` → PNG first

Our inference notebooks expect **19 PNG slices per volume** named like:

`<volume_id>_<L|R>_<slice>.png` where slice goes from `01` to `19`  
Example: `371_4646_28891_R_01.png ... 371_4646_28891_R_19.png`

Because these filenames are **directly produced by OCT-Converter from `.E2E` files**, the most reliable workflow is:

### 1) Put your `.E2E` files in label folders
Create a folder (e.g. `E2E_ROOT/`) with subfolders named exactly like the labels:

E2E_ROOT/
  CHM/
    *.E2E
  Healthy/
    *.E2E
  USH2A/
    *.E2E



### 2) Convert `.E2E` to PNGs (224×224, 19 slices) using our script
```bash
pip install oct-converter pillow numpy
python tools/export_e2e_to_png.py --e2e-root "C:/path/to/E2E_ROOT" --out-root "C:/path/to/dataset_png"
```
This creates:
```bash
dataset_png/
  CHM/      <volume_id>_<L|R>_01.png ... _19.png
  Healthy/  ...
  USH2A/    ...
  patients_metadata.csv
```
If your OCT files are not .E2E

Please use OCT-Converter to export your volumes to images, then rename them to match the expected format above.
See OCT-Converter documentation for supported file formats and export options.


