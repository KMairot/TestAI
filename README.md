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

Open demo_inference.ipynb and click Run All.

The notebook will:

1. download pretrained .pt files from GitHub Releases (v1.0)

2. run inference on your OCT slices

3. save outputs/predictions_<model>.csv

