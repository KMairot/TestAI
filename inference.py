from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Union

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


def _strip_prefix(sd: dict, prefix: str) -> dict:
    """Remove a common prefix from state_dict keys (module., core., etc.)."""
    if any(k.startswith(prefix) for k in sd.keys()):
        return {k.replace(prefix, "", 1) if k.startswith(prefix) else k: v for k, v in sd.items()}
    return sd


def _extract_state_dict(ckpt) -> dict:
    """Support common checkpoint conventions."""
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    # already a state_dict (OrderedDict)
    return ckpt


def infer_hybrid_num_layers_from_state_dict(sd: dict) -> int:
    """
    Infer num_encoder_layers from keys like:
        transformer_fusion.encoder.layers.<i>.*
    """
    pat = re.compile(r"transformer_fusion\.encoder\.layers\.(\d+)\.")
    ids = []
    for k in sd.keys():
        m = pat.search(k)
        if m:
            ids.append(int(m.group(1)))
    return (max(ids) + 1) if ids else 1


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
    where <volume_id> can itself contain underscores.

    Folder can be:
      - a split folder containing class subfolders (CHM/USH2A/Healthy), OR
      - any folder containing images (recursively)

    Parameters
    ----------
    model_name : "cnn_vote" | "moe" | "hybrid"
    weights_path : path to .pt checkpoint. If None, will try ./weights/ convention.
    class_names : optional list for pretty labels (index -> name). MUST match training order.
    strict_19 : if True, only volumes with exactly 19 slices are processed.

    Returns
    -------
    pd.DataFrame with one row per volume (patient+eye).
    """
    input_dir = Path(input_dir)
    device = _to_device(device)

    # normalize model name
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
        raise RuntimeError(f"No valid volumes found under {input_dir}. Check filenames/structure.")

    # Num classes
    num_classes = len(class_names) if class_names else 3

    # Resolve weights path
    if weights_path is None:
        default_map = {
            "cnn_vote": "cnn_resnet50_2025-10-20_best.pt",
            "moe": "cnn_resnet50_MOE_2025-11-09_best.pt",
            "hybrid": "hybrid_resnet50_xformer_2025-11-04_best.pt",
        }
        weights_path = Path("weights") / default_map[model_name]
    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path.resolve()}")

    # ----------------------------
    # Load checkpoint FIRST
    # ----------------------------
    ckpt = torch.load(weights_path, map_location="cpu")
    state = _extract_state_dict(ckpt)

    state = _strip_prefix(state, "module.")

    if model_name == "moe":
        # MoEWrapper attend des clés "core.*"
        if not any(k.startswith("core.") for k in state.keys()):
            state = {f"core.{k}": v for k, v in state.items()}
    else:
        # CNN / Hybrid n'ont pas de wrapper core => on strip si présent
        state = _strip_prefix(state, "core.")


    # ----------------------------
    # Build model
    #   - hybrid: infer num layers from checkpoint
    # ----------------------------
    if model_name == "cnn_vote":
        model = ResNetSliceClassifier(num_classes=num_classes)

    elif model_name == "moe":
        model = MoEWrapper(num_classes=num_classes)

    else:  # hybrid
        inferred_layers = infer_hybrid_num_layers_from_state_dict(state)
        print(f"[hybrid] inferred num_encoder_layers = {inferred_layers}")
        model = build_hybrid_model(num_classes=num_classes, num_encoder_layers=inferred_layers)

    # ----------------------------
    # Load weights (refuse silent mismatch)
    # ----------------------------
    incompat = model.load_state_dict(state, strict=False)
    if incompat.missing_keys or incompat.unexpected_keys:
        raise RuntimeError(
            "Checkpoint incompatible.\n"
            f"Missing keys: {len(incompat.missing_keys)} (e.g. {incompat.missing_keys[:10]})\n"
            f"Unexpected keys: {len(incompat.unexpected_keys)} (e.g. {incompat.unexpected_keys[:10]})"
        )

    model.to(device)
    model.eval()

    # ----------------------------
    # Inference loop
    # ----------------------------
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
            logits, weights = model(x)  # logits [1,C], weights [1,19]
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            pred_idx = int(probs.argmax())
            v["router_weights"] = weights.squeeze(0).detach().cpu().numpy().tolist()

        else:  # hybrid
            logits = model(x)  # [1,C]
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
