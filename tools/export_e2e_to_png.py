# -*- coding: utf-8 -*-
"""
Export OCT PNG slices from .E2E files (external validation / end-users)

Expected input structure:
E2E_ROOT/
  CHM/
    *.E2E
  Healthy/
    *.E2E
  USH2A/
    *.E2E
  (any other label folder name is accepted)

Output structure:
OUT_ROOT/
  CHM/
    <volume_id>_<L|R>_01.png ... _19.png
  Healthy/
  USH2A/
  patients_metadata.csv

Usage (example):
python tools/export_e2e_to_png.py --e2e-root "C:/path/to/E2E" --out-root "C:/path/to/dataset"

Dependencies:
pip install oct-converter pillow numpy
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple, Dict
from datetime import datetime, date
import csv
import re
import argparse

import numpy as np
from PIL import Image

from oct_converter.readers import E2E


# ------------------- utils -------------------

def sanitize_filename(s: str) -> str:
    """Remove characters that can break filenames on Windows/Linux."""
    s = str(s)
    s = s.strip().replace(" ", "_")
    return re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)

def to_uint8(slice_arr: np.ndarray, clip_pct: float = 1.0) -> np.ndarray:
    """Convert slice to uint8 [0,255] with optional percentile clipping."""
    a = np.asarray(slice_arr, dtype=np.float32)
    if clip_pct > 0:
        lo, hi = np.percentile(a, [clip_pct, 100.0 - clip_pct])
        if hi <= lo:
            lo, hi = float(np.min(a)), float(np.max(a))
    else:
        lo, hi = float(np.min(a)), float(np.max(a))
    if hi <= lo:
        return np.zeros_like(a, dtype=np.uint8)
    a = np.clip(a, lo, hi)
    a = (a - lo) / (hi - lo + 1e-12)
    return (a * 255.0 + 0.5).astype(np.uint8)

def get_n_bscans(vol) -> Optional[int]:
    for attr in ("num_slices", "n_slices", "num_bscans", "n_bscans"):
        if hasattr(vol, attr):
            try:
                v = getattr(vol, attr)
                if v is not None and int(v) > 0:
                    return int(v)
            except Exception:
                pass
    arr = getattr(vol, "volume", None)
    if arr is not None and hasattr(arr, "shape"):
        try:
            return int(arr.shape[0])
        except Exception:
            pass
    return None

def central_19_indices(n: int) -> Optional[List[int]]:
    if n == 19:
        return list(range(19))
    if n == 25:
        return list(range(3, 22))          # 3..21
    if n == 61:
        return list(range(0, 61, 3))[:19]  # 0,3,...,54
    return None

def compute_even_19_indices(n: int) -> List[int]:
    if n < 19:
        raise ValueError("n must be >= 19")
    xs = np.linspace(0, n - 1, 19)
    idxs = sorted(set(int(round(x)) for x in xs))
    while len(idxs) < 19:
        for i in range(n):
            if i not in idxs:
                idxs.append(i)
                if len(idxs) == 19:
                    break
    return sorted(idxs)[:19]

def list_e2e_files(label_dir: Path) -> List[Path]:
    return [p for p in label_dir.rglob("*") if p.suffix.lower() == ".e2e"]

def _to_date(obj) -> Optional[date]:
    if obj is None:
        return None
    if isinstance(obj, date):
        return obj
    if isinstance(obj, datetime):
        return obj.date()
    s = str(obj).strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None

def compute_age_years(dob, acquisition_date) -> Optional[float]:
    d_dob = _to_date(dob)
    d_exam = _to_date(acquisition_date)
    if d_dob is None or d_exam is None:
        return None
    days = (d_exam - d_dob).days
    return round(days / 365.25, 2)


# ------------------- export -------------------

def export_label_dir(
    label_dir: Path,
    out_root: Path,
    clip_percent: float,
    image_size: int,
    unknown_laterality: str = "skip",
) -> Tuple[List[Path], List[Path], List[Dict]]:
    label = label_dir.name
    out_label_dir = out_root / label
    out_label_dir.mkdir(parents=True, exist_ok=True)

    files = list_e2e_files(label_dir)
    print(f"\n== Label {label} : {len(files)} fichier(s) .E2E ==")

    files_with_exports: List[Path] = []
    files_without_exports: List[Path] = []
    patient_rows: List[Dict] = []

    for idx_file, fpath in enumerate(files, 1):
        print(f"[{idx_file}/{len(files)}] {fpath}")
        stem = fpath.stem

        sanga_2_used = False
        sanga_7_used = False

        n_vol_exported = 0

        try:
            reader = E2E(str(fpath))
            vols = reader.read_oct_volume() or []
        except Exception as e:
            print(f"  ERROR lecture E2E : {e!r}")
            files_without_exports.append(fpath)
            continue

        for v_idx, vol in enumerate(vols):
            n = get_n_bscans(vol)
            arr = getattr(vol, "volume", None)
            lat = getattr(vol, "laterality", None)  # 'L' / 'R' / None
            vol_id = getattr(vol, "volume_id", None) or f"{fpath.stem}_vol{v_idx:02d}"

            vol_id = sanitize_filename(vol_id)

            lat_char = str(lat).upper() if lat is not None else ""
            if lat_char not in {"L", "R"}:
                if unknown_laterality.lower() == "skip":
                    # if laterality is unknown, skip to avoid breaking the filename convention
                    row_lat = "U"
                else:
                    row_lat = unknown_laterality.upper()
                lat_char = row_lat

            # patient metadata
            dob = getattr(vol, "DOB", None)
            sex = getattr(vol, "sex", None)
            acq_date = getattr(vol, "acquisition_date", None)
            first_name = getattr(vol, "first_name", None)
            surname = getattr(vol, "surname", None)
            age_years = compute_age_years(dob, acq_date)

            row = {
                "label": label,
                "e2e_file": str(fpath),
                "volume_index": v_idx,
                "volume_id": vol_id,
                "laterality": lat_char,
                "n_slices": n if n is not None else "",
                "DOB": str(dob) if dob is not None else "",
                "sex": str(sex) if sex is not None else "",
                "acquisition_date": str(acq_date) if acq_date is not None else "",
                "first_name": str(first_name) if first_name is not None else "",
                "surname": str(surname) if surname is not None else "",
                "age_at_exam_years": age_years if age_years is not None else "",
                "exported": 0,
                "export_status": "",
            }

            # if laterality is still not L/R, enforce policy
            if lat_char not in {"L", "R"}:
                row["export_status"] = "unknown_laterality_skipped"
                patient_rows.append(row)
                continue

            sub_slices = None
            export_status = ""
            handled_special = False

            # ---- special cases ----
            if stem == "BELHO01S" and n in (3, 4) and arr is not None and n >= 1:
                sub_slices = [arr[0]] * 19
                export_status = f"BELHO01S_dup_first_from_{n}"
                handled_special = True

            if not handled_special and stem in {"COLLE01A", "HUBER01L", "HUBER01P"} and v_idx in (2, 7):
                if arr is not None and n is not None and n >= 1:
                    sub_slices = [arr[0]] * 19
                    export_status = f"{stem}_vol{v_idx}_dup_first"
                    handled_special = True

            if not handled_special and stem == "LOR01M" and v_idx in (0, 1):
                if arr is not None and n is not None and n >= 1:
                    sub_slices = [arr[0]] * 19
                    export_status = f"LOR01M_vol{v_idx}_dup_first"
                    handled_special = True

            if not handled_special and stem in {"GAUTI01P", "MEFTO01A"} and n == 49 and arr is not None:
                idxs = compute_even_19_indices(n)
                sub_slices = [arr[i] for i in idxs]
                export_status = f"{stem}_49slice_downsample_to_19"
                handled_special = True

            if not handled_special and stem == "SANGA01D" and arr is not None and n is not None:
                if n == 2 and not sanga_2_used and n >= 1:
                    sub_slices = [arr[0]] * 19
                    export_status = "SANGA01D_first_2slice_dup_first"
                    sanga_2_used = True
                    handled_special = True
                elif n == 7 and not sanga_7_used and n >= 1:
                    src_slices = [arr[i] for i in range(min(7, n))]
                    tiled = []
                    while len(tiled) < 19:
                        tiled.extend(src_slices)
                    sub_slices = tiled[:19]
                    export_status = "SANGA01D_7slice_tiled_to_19"
                    sanga_7_used = True
                    handled_special = True

            if not handled_special and stem == "SARDA01S" and arr is not None and n is not None and n >= 2:
                sub_slices = [arr[1]] * 19
                export_status = "SARDA01S_dup_second_slice"
                handled_special = True

            if not handled_special and stem == "ZAFRA02C" and v_idx in (1, 3):
                if arr is not None and n is not None and n >= 1:
                    sub_slices = [arr[0]] * 19
                    export_status = f"ZAFRA02C_vol{v_idx}_dup_first"
                    handled_special = True

            # ---- general cases ----
            if not handled_special:
                if arr is None:
                    row["export_status"] = "no_volume_array"
                    patient_rows.append(row)
                    continue

                keep_idx = central_19_indices(n if n is not None else -1)
                if keep_idx is None:
                    row["export_status"] = f"unsupported_n_slices_{n}"
                    patient_rows.append(row)
                    continue

                sub_slices = [arr[i] for i in keep_idx]
                export_status = (
                    "ok_19" if n == 19
                    else "ok_25_to_19" if n == 25
                    else "ok_61_step3_to_19" if n == 61
                    else "ok_generic"
                )

            if sub_slices is None or len(sub_slices) != 19:
                row["export_status"] = f"incomplete_{len(sub_slices) if sub_slices is not None else 0}"
                patient_rows.append(row)
                continue

            flip = (lat_char == "L")

            for s_rel, sl in enumerate(sub_slices, 1):  # 1..19
                sl_u8 = to_uint8(sl, clip_pct=clip_percent)
                if flip:
                    sl_u8 = np.fliplr(sl_u8)

                im = Image.fromarray(sl_u8)
                im = im.resize((image_size, image_size), Image.BILINEAR)

                out_name = f"{vol_id}_{lat_char}_{s_rel:02d}.png"
                im.save(out_label_dir / out_name)

            n_vol_exported += 1
            row["exported"] = 1
            row["export_status"] = export_status
            patient_rows.append(row)

        print(f"  -> {n_vol_exported} volume(s) exporté(s) pour ce fichier")

        if n_vol_exported > 0:
            files_with_exports.append(fpath)
        else:
            files_without_exports.append(fpath)

    return files_with_exports, files_without_exports, patient_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2e-root", type=str, required=True, help="Root folder containing label subfolders with .E2E files")
    ap.add_argument("--out-root", type=str, required=True, help="Output folder where PNGs will be written")
    ap.add_argument("--clip-percent", type=float, default=1.0)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--unknown-laterality", type=str, default="skip", choices=["skip","L","R"],
                    help="What to do if laterality is missing. Default: skip volumes. Or force L/R.")
    args = ap.parse_args()

    e2e_root = Path(args.e2e_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    label_dirs = [d for d in e2e_root.iterdir() if d.is_dir()]
    if not label_dirs:
        print(f"No label subfolders found in {e2e_root}")
        return

    print("Found label folders:")
    for d in label_dirs:
        print(" -", d.name)

    all_with_exports: List[Path] = []
    all_without_exports: List[Path] = []
    all_patient_rows: List[Dict] = []

    for label_dir in label_dirs:
        files_with, files_without, patient_rows = export_label_dir(
            label_dir=label_dir,
            out_root=out_root,
            clip_percent=args.clip_percent,
            image_size=args.image_size,
            unknown_laterality=args.unknown_laterality
        )
        all_with_exports.extend(files_with)
        all_without_exports.extend(files_without)
        all_patient_rows.extend(patient_rows)

    print("\n=== EXPORT DONE ===")
    print(f"PNGs saved to: {out_root}")

    print("\nFiles with at least one exported volume:")
    for p in all_with_exports[:200]:
        print(" -", p)
    if len(all_with_exports) > 200:
        print(f" ... ({len(all_with_exports)-200} more)")

    print("\nFiles without any exported volume:")
    for p in all_without_exports[:200]:
        print(" -", p)
    if len(all_without_exports) > 200:
        print(f" ... ({len(all_without_exports)-200} more)")

    if all_patient_rows:
        csv_path = out_root / "patients_metadata.csv"
        fieldnames = sorted({k for r in all_patient_rows for k in r.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_patient_rows:
                writer.writerow(row)
        print(f"\nMetadata CSV written: {csv_path}")


if __name__ == "__main__":
    main()
