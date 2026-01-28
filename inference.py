from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

from data.volume import build_volume_index, load_volume_tensor
from models.cnn_vote import ResNetSliceClassifier
from models.moe import MoEWrapper
from models.hybrid import build_hybrid_model


def _to_device(device: Union[str, torch.device]) -> torch.device:
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


@torch.no_grad()
def run_inference(
    input_dir: Union[str, Path],
    model_name: str,
    weights_path: Optional[Union[str, Path]] = None,
    device: Union[str, torch.device] = "cpu",
    image_size: int = 224,
    class_names: Optional[List[str]] = None,
    strict_19: bool = True,
) -> pd.DataFrame:
    """
    Run inference on a folder containing OCT slice images named like:
        <volume_id>_<L|R>_<slice>.PNG  (slice 01..19)
    where <volume_id> can itself contain underscores, e.g. 371_4646_28891.

    The folder may be:
      - a split folder containing class subfolders (CHM/USH2A/Healthy), OR
      - any folder containing images (recursively)

    Parameters
    ----------
    model_name : "cnn_vote" | "moe" | "hybrid"
    weights_path : path to .pt checkpoint. If None, will try to find in ./weights/ by convention.
    class_names : optional list used only for pretty labels in outputs (index -> name).
                 IMPORTANT: must match the training class order.
    strict_19 : if True, only volumes with exactly 19 slices are processed.

    Returns
    -------
    pd.DataFrame with one row per volume (patient+eye).
    """
    input_dir = Path(input_dir)
    device = _to_device(device)

    model_name = model_name.lower().strip()
    if model_name in ["cnn", "cnn_vote", "vote"]:
        model_name = "cnn_vote"
    elif model_name in ["moe"]:
        model_name = "moe"
    elif model_name in ["hybrid", "transformer", "xformer"]:
        model_name = "hybrid"
    else:
        raise ValueError(f"Unknown model_name={model_name}. Use: cnn_vote | moe | hybrid")

    # Index volumes
    volumes = build_volume_index(input_dir=input_dir, strict_19=strict_19)
    if len(volumes) == 0:
        raise RuntimeError(f"No valid volumes found under {input_dir}. Check your filenames and structure.")

    # Infer num_classes from class_names if provided, else default 3
    num_classes = len(class_names) if class_names else 3

    # Resolve weights path
    if weights_path is None:
        # Conventional filenames (adjust if you rename checkpoints)
        default_map = {
           "cnn_vote": "cnn_resnet50_2025-10-20_best.pt",
    "moe": "cnn_resnet50_MOE_2025-11-09_best.pt",
    "hybrid": "hybrid_resnet50_xformer_2025-11-04_best.pt",
        }
        weights_path = Path("weights") / default_map[model_name]
    weights_path = Path(weights_path)

    # Build model
    if model_name == "cnn_vote":
        model = ResNetSliceClassifier(num_classes=num_classes)
    elif model_name == "moe":
        model = MoEWrapper(num_classes=num_classes)
    else:  # hybrid
        model = build_hybrid_model(num_classes=num_classes)

    # Load checkpoint (state_dict only is supported)
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Some of your training scripts used a wrapper "core." prefix
    # We'll try a few common normalizations.
    loaded = False
    load_errors = []

    def _try_load(sd: dict, strict: bool):
        nonlocal loaded
        if loaded:
            return
        try:
            model.load_state_dict(sd, strict=strict)
            loaded = True
        except Exception as e:
            load_errors.append(str(e))

    _try_load(state, strict=True)

    if not loaded:
        # Strip a top-level "core." prefix if present
        if any(k.startswith("core.") for k in state.keys()):
            stripped = {k.replace("core.", "", 1): v for k, v in state.items()}
            _try_load(stripped, strict=True)
            _try_load(stripped, strict=False)

    if not loaded:
        _try_load(state, strict=False)

    if not loaded:
        raise RuntimeError(
            "Failed to load checkpoint. Here are the last errors:\n"
            + "\n---\n".join(load_errors[-3:])
        )

    model.to(device)
    model.eval()

    rows = []
    for v in volumes:
        x = load_volume_tensor(v["paths"], image_size=image_size).unsqueeze(0).to(device)  # [1,19,3,H,W]

        if model_name == "cnn_vote":
            logits_slices = model.forward_slices(x)  # [1,19,C]
            probs_slices = torch.softmax(logits_slices, dim=-1).squeeze(0)  # [19,C]
            pred_slices = probs_slices.argmax(dim=-1).cpu().numpy()  # [19]
            # majority vote
            counts = np.bincount(pred_slices, minlength=num_classes)
            top = int(counts.argmax())
            # tie-break by mean prob
            ties = np.where(counts == counts.max())[0]
            if len(ties) > 1:
                mean_probs = probs_slices.mean(dim=0).cpu().numpy()
                top = int(ties[np.argmax(mean_probs[ties])])
            probs = probs_slices.mean(dim=0).cpu().numpy()
            pred_idx = top

        elif model_name == "moe":
            logits, weights = model(x)   # logits [1,C], weights [1,19]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            # keep router weights for interpretability
            v["router_weights"] = weights.squeeze(0).detach().cpu().numpy().tolist()

        else:  # hybrid
            logits = model(x)            # [1,C]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())

        pred_label = class_names[pred_idx] if class_names and pred_idx < len(class_names) else str(pred_idx)

        row = {
            "volume_id": v["vid"],
            "laterality": v["lat"],
            "n_slices": len(v["paths"]),
            "pred_idx": pred_idx,
            "pred_label": pred_label,
        }
        # Optional true label if detected from folder name
        if v.get("true_label") is not None:
            row["true_label"] = v["true_label"]

        # probabilities
        for i in range(num_classes):
            col = f"prob_{class_names[i]}" if class_names and i < len(class_names) else f"prob_{i}"
            row[col] = float(probs[i])

        # optional MoE weights
        if "router_weights" in v:
            row["router_weights"] = v["router_weights"]

        rows.append(row)

    return pd.DataFrame(rows)
