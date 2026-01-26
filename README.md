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
git clone https://github.com/TODO_ORG/TODO_REPO.git
cd TODO_REPO

