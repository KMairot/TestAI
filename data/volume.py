from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms import InterpolationMode
from PIL import Image

# Match: <volume_id>_<L|R>_<slice>.<ext>
# volume_id can contain underscores, e.g. 371_4646_28891
_PAT = re.compile(r"^(?P<vid>.+?)_(?P<lat>[LR])_(?P<slice>\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def parse_name(p: Path) -> Optional[Dict]:
    m = _PAT.match(p.name)
    if not m:
        return None
    return {"vid": m.group("vid"), "lat": m.group("lat").upper(), "slice": int(m.group("slice"))}

def _detect_true_label(p: Path) -> Optional[str]:
    # If dataset is structured as split/Label/*.png, label is parent folder name
    parent = p.parent.name
    if parent.lower() in ["chm", "ush2a", "healthy"]:
        return parent
    return None

def build_volume_index(input_dir: Union[str, Path], strict_19: bool = True) -> List[Dict]:
    """
    Recursively scan input_dir and group slices into volumes by (vid, lat).
    Returns list of dicts: {vid, lat, paths, true_label?}
    """
    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)

    files = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
    groups: Dict[Tuple[str, str], Dict] = {}

    for p in files:
        info = parse_name(p)
        if not info:
            continue
        key = (info["vid"], info["lat"])
        g = groups.get(key)
        if g is None:
            g = {"vid": info["vid"], "lat": info["lat"], "paths": [], "true_label": _detect_true_label(p)}
            groups[key] = g
        g["paths"].append((info["slice"], p))

    out = []
    for (vid, lat), g in groups.items():
        g["paths"].sort(key=lambda t: t[0])
        paths = [p for _, p in g["paths"]]
        if strict_19 and len(paths) != 19:
            continue
        out.append({"vid": vid, "lat": lat, "paths": paths[:19], "true_label": g.get("true_label")})

    return sorted(out, key=lambda d: (d["vid"], d["lat"]))

def load_volume_tensor(paths: List[Path], image_size: int = 224, normalize_imagenet: bool = True) -> torch.Tensor:
    """
    Load 19 slices -> tensor [19, 3, H, W]
    """
    tf = [
        T.ToImage(),
        T.Grayscale(num_output_channels=3),
        T.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
        T.ToDtype(torch.float32, scale=True),
    ]
    if normalize_imagenet:
        tf.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    tf = T.Compose(tf)

    imgs = []
    for p in paths:
        img = Image.open(p).convert("L")
        imgs.append(tf(img))
    return torch.stack(imgs, dim=0)
