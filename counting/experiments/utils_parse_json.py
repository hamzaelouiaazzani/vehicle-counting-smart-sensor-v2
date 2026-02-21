#!/usr/bin/env python3
"""
Build results DataFrame from either:
 - old single big JSON file: {"meta":..., "sequences": {...}}
 - OR new converter output folder: <results_dir>/meta.json and <results_dir>/sequences/*.json

Maintains previous IMAROC_1 / IMAROC_2 logic (van handling, grouping, adapters).
This version reads per-sequence files one-by-one (streaming-friendly) and converts Decimal values.

Fixes:
 - When meta does not include `default_classes_used` and `detector_class_names`,
   the code now falls back to `meta["class_map"]` (if present) and uses the integer
   class ids there as the indices into `counts_by_class` (this matches the example
   where counts_by_class is indexed by the detector's class id).
 - Supports both "dense/compact" detector outputs (counts_by_class is a compact
   array where index 0..N-1 correspond to detector classes) AND "sparse" outputs
   where counts_by_class is indexed by the detector class id (e.g. index 7 for class 7).
 - Supports counts_by_class provided as a dict (class_id -> count) or list.
"""

from pathlib import Path
from typing import Dict, Any, Callable, List, Optional, Sequence, Union, Tuple, Iterable
import json
from decimal import Decimal
import pandas as pd
import re
import ijson

# -------------------------
# Defaults (adjust if needed)
# -------------------------
DEFAULT_COCO_CLASS_MAP = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_CLASS_ORDER = sorted(DEFAULT_COCO_CLASS_MAP.keys())  # [1,2,3,5,7]

DATASET_REGISTRY = {
    "IMAROC_1": {
        "adapter": None,  # will be filled later
        "annotation_csv": r"C:\Users\hamza\Datasets\TrafficDatasets\IMAROC_1\actual_counts.csv"
    },
    "IMAROC_2": {
        "adapter": None,  # will be filled later
        "annotation_csv": r"C:\Users\hamza\Datasets\TrafficDatasets\IMAROC_2\actual_counts.csv"
    }
}

# -------------------------
# Utilities
# -------------------------
def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value) if value is not None and value != "" else default
    except Exception:
        return default

def _convert_decimals(obj: Any, decimal_to_str: bool = False) -> Any:
    """
    Recursively convert Decimal instances to float (or str if decimal_to_str True).
    Also handles nested lists/dicts.
    """
    if isinstance(obj, Decimal):
        return str(obj) if decimal_to_str else float(obj)
    if isinstance(obj, dict):
        return {k: _convert_decimals(v, decimal_to_str) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_decimals(v, decimal_to_str) for v in obj]
    return obj

def _load_json_file(path: Union[str, Path], decimal_to_str: bool = False) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    return _convert_decimals(data, decimal_to_str)

def load_results_like_old_or_new(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Accept either:
      - a file path to the legacy big JSON ({"meta":..., "sequences": {...}})
      - OR a directory path which contains meta.json and a sequences/ folder.
    Returns a dict in memory containing:
      {"meta": {...}, "sequences_iter": iterable_of_(seq_name, seq_obj)}
    Note: sequences are not loaded all at once when results_path is a directory.
    """
    p = Path(results_path)
    if p.is_file():
        # legacy big JSON: load into memory (only call if you actually have a file)
        data = _load_json_file(p)
        meta = data.get("meta", {}) or {}
        sequences = data.get("sequences", {}) or {}
        # produce an iterator that yields from the dict
        def seq_iter():
            for k, v in sequences.items():
                yield k, _convert_decimals(v)
        return {"meta": meta, "sequences_iter": seq_iter()}
    elif p.is_dir():
        # new converter output dir
        meta_path = p / "meta.json"
        sequences_dir = p / "sequences"
        if not meta_path.exists():
            # try parent (some users place meta at parent)
            raise FileNotFoundError(f"meta.json not found in folder: {p}")
        meta = _load_json_file(meta_path)
        if not sequences_dir.exists() or not sequences_dir.is_dir():
            raise FileNotFoundError(f"sequences directory not found: {sequences_dir}")
        # create an iterator that loads each file one by one
        def seq_iter():
            for seq_file in sorted(sequences_dir.glob("*.json")):
                try:
                    seq_obj = _load_json_file(seq_file)
                    # derive name: prefer explicit inside name keys, fallback to file stem
                    seq_name = None
                    for candidate in ("sequence", "name", "video_name", "video", "filename"):
                        if isinstance(seq_obj, dict) and candidate in seq_obj:
                            val = seq_obj.get(candidate)
                            if isinstance(val, str) and val.strip():
                                seq_name = val
                                break
                    if not seq_name:
                        seq_name = seq_file.stem
                    yield seq_name, seq_obj
                except Exception as e:
                    # skip corrupted sequence files but notify user
                    print(f"[WARN] Failed to load sequence file {seq_file}: {e}")
                    continue
        return {"meta": meta, "sequences_iter": seq_iter()}
    else:
        raise FileNotFoundError(f"Path not found: {p}")

# -------------------------
# Annotation CSV helpers
# -------------------------
def _load_annotation_df(annotation_path: Optional[str]) -> pd.DataFrame:
    if not annotation_path:
        return pd.DataFrame()
    df = pd.read_csv(annotation_path, dtype=str).replace({pd.NA: None, "": None})
    return df

def _find_annotation_row_dict(ann_df: pd.DataFrame, sequence: str) -> Dict[str, Any]:
    if ann_df is None or ann_df.empty:
        return {}
    seq = str(sequence)

    # exact match on common column names
    for col in ("sequence", "video", "video_id", "filename", "name", "video_name"):
        if col in ann_df.columns:
            mask = ann_df[col].astype(str) == seq
            if mask.any():
                return ann_df.loc[mask].iloc[0].to_dict()

    # exact match on index
    try:
        if seq in ann_df.index:
            row = ann_df.loc[seq]
            if isinstance(row, pd.Series):
                return row.to_dict()
    except Exception:
        pass

    # substring match in object columns
    for col in ann_df.columns:
        if ann_df[col].dtype == object:
            matches = ann_df[ann_df[col].str.contains(seq, na=False)]
            if not matches.empty:
                return matches.iloc[0].to_dict()

    return {}

# -------------------------
# Dataset adapters (same as before)
# -------------------------
def annotated_from_imaroc1_csv(ann_df: pd.DataFrame, sequence: str, keep_van_separate: bool = False) -> Dict[str, Any]:
    row = _find_annotation_row_dict(ann_df, sequence)
    out = {
        "actual_bicycle_counts": _safe_int(row.get("actual_bicycle_counts")),
        "actual_car_counts": _safe_int(row.get("actual_car_counts")),
        "actual_motorcycle_counts": _safe_int(row.get("actual_motorcycle_counts")),
        "actual_bus_counts": _safe_int(row.get("actual_bus_counts")),
        "actual_truck_counts": _safe_int(row.get("actual_truck_counts")),
        "actual_total_vehicles": None if row.get("actual_total_vehicles") in (None, "") else _safe_int(row.get("actual_total_vehicles")),
    }
    if not out["actual_total_vehicles"]:
        out["actual_total_vehicles"] = (
            out["actual_car_counts"] + out["actual_bicycle_counts"] +
            out["actual_motorcycle_counts"] + out["actual_bus_counts"] + out["actual_truck_counts"]
        )
    out["actual_van_counts"] = 0
    return out

def annotated_from_imaroc2_csv(ann_df: pd.DataFrame, sequence: str, keep_van_separate: bool = False) -> Dict[str, Any]:
    row = _find_annotation_row_dict(ann_df, sequence)
    actual_car = _safe_int(row.get("actual_car_counts"))
    actual_van = _safe_int(row.get("actual_van_counts"))
    actual_bicycle = _safe_int(row.get("actual_bicycle_counts"))
    actual_motorcycle = _safe_int(row.get("actual_motorcycle_counts"))
    actual_bus = _safe_int(row.get("actual_bus_counts"))
    actual_truck = _safe_int(row.get("actual_truck_counts"))

    if keep_van_separate:
        combined_car = actual_car
    else:
        combined_car = actual_car + actual_van

    total = combined_car + actual_bicycle + actual_motorcycle + actual_bus + actual_truck

    return {
        "actual_bicycle_counts": actual_bicycle,
        "actual_car_counts": combined_car,
        "actual_motorcycle_counts": actual_motorcycle,
        "actual_bus_counts": actual_bus,
        "actual_truck_counts": actual_truck,
        "actual_total_vehicles": total,
        "actual_van_counts": actual_van,
        "direction_relative_to_camera": row.get("direction_relative_to_camera"),
        "road_directionality": row.get("road_directionality"),
        "lighting": row.get("lighting"),
    }

# register adapters
DATASET_REGISTRY["IMAROC_1"]["adapter"] = annotated_from_imaroc1_csv
DATASET_REGISTRY["IMAROC_2"]["adapter"] = annotated_from_imaroc2_csv

# -------------------------
# Core builder (adapted to stream sequences)
# -------------------------
def build_results_generic(
    results_path: Union[str, Path],
    dataset: Optional[str] = None,
    class_names: Optional[Sequence[str]] = None,
    class_order: Optional[Sequence[int]] = None,
    group_2_3: Optional[Sequence[Union[int, str]]] = None,
    group_4: Optional[Sequence[Union[int, str]]] = None,
    annotate_adapter: Optional[Callable[..., Dict[str, Any]]] = None,
    annotation_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build results from either:
      - big JSON file path, or
      - folder with meta.json and sequences/*.json
    Streams sequence files one at a time when using the new folder layout.
    """
    container = load_results_like_old_or_new(results_path)
    meta = container.get("meta", {}) or {}

    # --- detect class names and class-id mapping
    # class_id_map is a list where class_id_map[i] is the detector class id (int)
    # that corresponds to detected_class_names[i].
    detected_class_names: List[str] = []
    class_id_map: List[int] = []

    if class_names:
        # user explicitly supplied class names -> assume dense compact mapping 0..N-1
        detected_class_names = list(class_names)
        class_id_map = list(range(len(detected_class_names)))
    else:
        detector_names_meta = meta.get("detector_class_names")
        default_classes_used_meta = meta.get("default_classes_used")
        class_map_meta = meta.get("class_map")

        if detector_names_meta and isinstance(detector_names_meta, list) and len(detector_names_meta) > 0:
            # If detector_class_names exist in meta, prefer them.
            detected_class_names = list(detector_names_meta)
            # If default_classes_used provided and matches length, honor it as class ids
            if default_classes_used_meta and isinstance(default_classes_used_meta, list) and len(default_classes_used_meta) == len(detected_class_names):
                try:
                    class_id_map = [int(x) for x in default_classes_used_meta]
                except Exception:
                    # fallback to dense mapping in case of bad types
                    class_id_map = list(range(len(detected_class_names)))
            else:
                # no explicit class ids -> assume compact dense mapping 0..N-1
                class_id_map = list(range(len(detected_class_names)))
        elif class_map_meta and isinstance(class_map_meta, dict) and len(class_map_meta) > 0:
            # meta provides class_map: keys are detector class ids (likely strings) -> map to names
            try:
                class_map_int = {int(k): v for k, v in class_map_meta.items()}
            except Exception:
                # if keys aren't convertible, fallback to defaults
                class_map_int = {}
            if class_map_int:
                class_id_map = sorted(class_map_int.keys())
                detected_class_names = [class_map_int[k] for k in class_id_map]
            else:
                # fallback
                class_id_map = DEFAULT_CLASS_ORDER.copy()
                detected_class_names = [DEFAULT_COCO_CLASS_MAP[k] for k in class_id_map]
        else:
            # final fallback to built-in default mapping
            class_id_map = DEFAULT_CLASS_ORDER.copy()
            detected_class_names = [DEFAULT_COCO_CLASS_MAP[k] for k in class_id_map]

    # class_order indices default: user can pass indices relative to detected_class_names
    if class_order:
        class_order_idx = list(class_order)
    else:
        class_order_idx = list(range(len(detected_class_names)))

    # determine if detector predicts 'van' (case-insensitive match)
    detector_has_van = any((name or "").strip().lower() == "van" for name in detected_class_names)

    # normalize groups to indices (accept names or indices)
    name_to_idx = {name: idx for idx, name in enumerate(detected_class_names)}
    def _normalize_group(g):
        if g is None:
            return []
        out = []
        for item in g:
            if isinstance(item, int):
                out.append(item)
            else:
                if item in name_to_idx:
                    out.append(name_to_idx[item])
                else:
                    lowered = item.lower()
                    for n, idx in name_to_idx.items():
                        if n and n.lower() == lowered:
                            out.append(idx)
                            break
        return sorted(set(out))

    # default sensible groups for IMAROC datasets (name matching)
    if group_2_3 is None and dataset in ("IMAROC_1", "IMAROC_2"):
        candidate_names_2_3 = {"bicycle", "motorcycle", "bike", "motorbike"}
        group_2_3 = [n for n in detected_class_names if n and n.lower() in candidate_names_2_3]
    if group_4 is None and dataset in ("IMAROC_1", "IMAROC_2"):
        candidate_names_4 = {"car", "truck", "bus", "van"}
        group_4 = [n for n in detected_class_names if n and n.lower() in candidate_names_4]

    group_2_3_idx = _normalize_group(group_2_3)
    group_4_idx = _normalize_group(group_4)

    # annotation adapter selection
    if annotate_adapter is None:
        if dataset in DATASET_REGISTRY:
            adapter_fn = DATASET_REGISTRY[dataset]["adapter"]
            annotation_csv = annotation_csv or DATASET_REGISTRY[dataset]["annotation_csv"]
        else:
            adapter_fn = lambda df, seq, **kw: {}
    else:
        adapter_fn = annotate_adapter

    ann_df = _load_annotation_df(annotation_csv) if annotation_csv else pd.DataFrame()

    # helper to convert counts_by_class which may be list or dict into a list (sparse allowed)
    def _pred_bycls_to_list(pred_bycls) -> List[int]:
        if pred_bycls is None:
            return []
        if isinstance(pred_bycls, dict):
            # keys might be strings
            try:
                mapping = {int(k): int(v) for k, v in pred_bycls.items()}
            except Exception:
                # fall back to attempt without int-conversion
                mapping = {}
                for k, v in pred_bycls.items():
                    try:
                        mapping[int(k)] = int(v)
                    except Exception:
                        continue
            if not mapping:
                return []
            maxk = max(mapping.keys())
            arr = [0] * (maxk + 1)
            for k, v in mapping.items():
                if 0 <= k <= maxk:
                    arr[k] = int(v)
            return arr
        if isinstance(pred_bycls, list):
            return [int(x) if x is not None else 0 for x in pred_bycls]
        # unknown type -> try to coerce to list
        try:
            return [int(x) for x in list(pred_bycls)]
        except Exception:
            return []

    # helper to get predicted count for a detected_class index i using class_id_map
    def _get_pred_for_class_index(pred_arr: List[int], class_id_map: List[int], idx: int) -> int:
        if idx < 0 or idx >= len(class_id_map):
            return 0
        class_id = class_id_map[idx]
        if 0 <= class_id < len(pred_arr):
            return int(pred_arr[class_id])
        # sometimes pred_arr might be compact (class ids are 0..len-1) and class_id_map is also 0..N-1,
        # this case handled above. Otherwise, return 0.
        return 0

    # build rows
    rows = []
    seq_iter = container.get("sequences_iter", [])
    seq_count = 0
    for seq_name, seq_entry in seq_iter:
        seq_count += 1
        # seq_entry should be a dict similar to previous "seq_entry"
        # pass keep_van_separate flag to adapter so it can return actual_van_counts separately if needed
        annotated_norm = adapter_fn(ann_df, seq_name, keep_van_separate=detector_has_van)

        actual_bicycle = _safe_int(annotated_norm.get("actual_bicycle_counts"))
        actual_car = _safe_int(annotated_norm.get("actual_car_counts"))
        actual_motor = _safe_int(annotated_norm.get("actual_motorcycle_counts"))
        actual_bus = _safe_int(annotated_norm.get("actual_bus_counts"))
        actual_truck = _safe_int(annotated_norm.get("actual_truck_counts"))
        actual_van = _safe_int(annotated_norm.get("actual_van_counts"))
        actual_total = _safe_int(annotated_norm.get("actual_total_vehicles"))

        actual_2_3 = actual_bicycle + actual_motor
        actual_4 = actual_car + actual_bus + actual_truck + actual_van

        # counts structure: try multiple common field names
        counts_dict = {}
        if isinstance(seq_entry, dict):
            counts_dict = seq_entry.get("counts", {}) or seq_entry.get("results", {}) or seq_entry.get("predictions", {}) or {}
        else:
            counts_dict = {}

        for exp_id, pred_obj in counts_dict.items():
            if pred_obj is None:
                continue

            predicted_total = pred_obj.get("total_count", None)
            pred_bycls_raw = pred_obj.get("counts_by_class", []) or pred_obj.get("by_class", []) or pred_obj.get("per_class", []) or []
            # normalize to list (sparse index supported)
            pred_list = _pred_bycls_to_list(pred_bycls_raw)

            cfg = parse_exp_id(exp_id)
            counter_family = cfg.get("counter")
            tracker_name = cfg.get("tracker")
            vicinity = cfg.get("vicinity")
            stride = cfg.get("stride")

            # predicted group aggregates from normalized indices: use class_id_map to map indices -> class ids
            pred_2_3 = sum(_get_pred_for_class_index(pred_list, class_id_map, idx) for idx in group_2_3_idx)
            pred_4 = sum(_get_pred_for_class_index(pred_list, class_id_map, idx) for idx in group_4_idx)

            base_row = {
                "exp_id": exp_id,
                "sequence": seq_name,
                "counter_family": counter_family,
                "tracker": tracker_name,
                "vicinity": vicinity,
                "stride": stride,
                "actual_total_vehicles": actual_total,
                "actual_bicycle_counts": actual_bicycle,
                "actual_car_counts": actual_car,
                "actual_motorcycle_counts": actual_motor,
                "actual_bus_counts": actual_bus,
                "actual_truck_counts": actual_truck,
                "actual_van_counts": actual_van,
                "actual_2_3_wheels": int(actual_2_3),
                "actual_4_wheels": int(actual_4),
                "predicted_total": int(predicted_total) if predicted_total is not None else None,
                "predicted_2_3_wheels": int(pred_2_3),
                "predicted_4_wheels": int(pred_4)
            }

            # IMAROC_2 metadata fields
            if dataset == "IMAROC_2":
                base_row.update({
                    "direction_relative_to_camera": annotated_norm.get("direction_relative_to_camera"),
                    "road_directionality": annotated_norm.get("road_directionality"),
                    "lighting": annotated_norm.get("lighting"),
                })

            # per-class predicted columns using detected_class_names ordering and class_id_map mapping
            for idx in class_order_idx:
                name = detected_class_names[idx] if 0 <= idx < len(detected_class_names) else f"class_{idx}"
                val = _get_pred_for_class_index(pred_list, class_id_map, idx)
                col_name = f"predicted_{name}"
                base_row[col_name] = int(val)

            rows.append(base_row)

        # small progress feedback
        if seq_count % 100 == 0:
            print(f"[INFO] Processed {seq_count} sequence files...")

    df = pd.DataFrame(rows)
    if not df.empty:
        df["algorithm_id"] = df["exp_id"].apply(get_algorithm_id_from_exp)

    # order columns for convenience (same order as original)
    meta_cols = ["exp_id", "algorithm_id", "sequence", "counter_family", "tracker", "vicinity", "stride"]
    dataset_meta_cols = []
    if dataset == "IMAROC_2":
        dataset_meta_cols = ["direction_relative_to_camera", "road_directionality", "lighting"]
    annotated_cols = [
        "actual_total_vehicles",
        "actual_bicycle_counts", "actual_car_counts", "actual_motorcycle_counts",
        "actual_bus_counts", "actual_truck_counts", "actual_van_counts",
        "actual_2_3_wheels", "actual_4_wheels"
    ]
    predicted_class_cols = [f"predicted_{name}" for name in detected_class_names]
    predicted_group_cols = ["predicted_2_3_wheels", "predicted_4_wheels"]

    cols = meta_cols + dataset_meta_cols + annotated_cols + ["predicted_total"] + predicted_class_cols + predicted_group_cols
    cols = [c for c in cols if c in df.columns]
    return df[cols]

# -------------------------
# Helper functions from original
# -------------------------
def parse_exp_id(exp_id: str) -> Dict[str, Any]:
    parts = exp_id.split("__")
    return {
        "sequence": parts[0] if len(parts) > 0 else None,
        "counter": parts[1] if len(parts) > 1 else None,
        "stride": int(parts[2].lstrip("s")) if len(parts) > 2 and parts[2].startswith("s") else None,
        "tracker": None if len(parts) <= 3 or (len(parts) > 3 and parts[3].startswith("tracker-") and parts[3].split("tracker-")[1] == "none") else (parts[3].split("tracker-")[1] if len(parts) > 3 and parts[3].startswith("tracker-") else None),
        "vicinity": None if len(parts) <= 4 or (len(parts) > 4 and parts[4].startswith("vic-") and parts[4].split("vic-")[1] == "none") else (float(parts[4].split("vic-")[1]) if len(parts) > 4 and parts[4].startswith("vic-") else None),
    }

def get_algorithm_id_from_exp(exp_id: str) -> str:
    parts = exp_id.split("__", 1)
    return parts[1] if len(parts) > 1 else exp_id

# -------------------------
# Convenience wrappers
# -------------------------
def build_results_IMAROC1(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return build_results_generic(path, dataset="IMAROC_1", **kwargs)

def build_results_IMAROC2(path: Union[str, Path], **kwargs) -> pd.DataFrame:
    return build_results_generic(path, dataset="IMAROC_2", **kwargs)















# -------------------------
# Parsing latency tables
# -------------------------
def parse_detection_latencies(seq_data: Dict[str, Any], sequence_name: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns: sequence, frame_idx, pre, inf, post, n_detections, latency
    Times are converted to milliseconds (pre, inf, post, latency).
    """
    per = seq_data.get("per_frame_pipeline_timing", {}) or {}
    pre = list(per.get("pre", []))
    inf = list(per.get("inf", []))
    post = list(per.get("post", []))
    n_det = list(per.get("n_detections", []))
    n = max(len(pre), len(inf), len(post), len(n_det))
    rows = []
    for i in range(n):
        pre_ms = float(pre[i]) * 1000 if i < len(pre) and pre[i] is not None else pd.NA
        inf_ms = float(inf[i]) * 1000 if i < len(inf) and inf[i] is not None else pd.NA
        post_ms = float(post[i]) * 1000 if i < len(post) and post[i] is not None else pd.NA
        n_d = int(n_det[i]) if i < len(n_det) and n_det[i] is not None else pd.NA
        rows.append({
            "sequence": sequence_name,
            "frame_idx": i,
            "pre": pre_ms,
            "inf": inf_ms,
            "post": post_ms,
            "n_detections": n_d
        })


    detection_df = pd.DataFrame(rows)
    # latency will be NaN if any component is NA; that's ok
    detection_df["latency"] = detection_df["pre"] + detection_df["inf"] + detection_df["post"]
    return detection_df

def parse_tracking_latencies(seq_data: Dict[str, Any], sequence_name: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      sequence, tracker, stride, stride_index, frame_idx_estimated, latency
    - stride_index: index within the stride-sampled frames (0..)
    - frame_idx_estimated: estimated original frame index (stride_index * stride)
    """
    per = seq_data.get("per_frame_pipeline_timing", {}) or {}
    track = per.get("track", {}) or {}
    rows = []
    for tracker_name, tracker_data in track.items():
        if not isinstance(tracker_data, dict):
            continue
        for stride_key, latency_list in tracker_data.items():
            try:
                stride = int(stride_key)
            except Exception:
                # If stride key is non-int, skip
                continue
            if not isinstance(latency_list, Iterable):
                continue
            for idx, latency in enumerate(latency_list):
                latency_ms = float(latency) * 1000 if latency is not None else pd.NA
                rows.append({
                    "sequence": sequence_name,
                    "tracker": tracker_name,
                    "stride": stride,
                    "stride_index": idx,
                    "frame_idx_estimated": idx * stride,
                    "latency": latency_ms
                })
    if not rows:
        return pd.DataFrame(columns=["sequence","tracker","stride","stride_index","frame_idx_estimated","latency"])
    return pd.DataFrame(rows)

def _extract_stride_from_counter_key(key: str) -> int:
    m = re.search(r"_s(\d+)", key)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return 1
    return 1

def _extract_counter_id(key: str) -> str:
    m = re.search(r"(counter_[0-9]+)", key)
    if m:
        return m.group(1)
    return key

def parse_count_latencies(seq_data: Dict[str, Any], sequence_name: str) -> pd.DataFrame:
    """
    Returns DataFrame with columns:
      sequence, counter_key, counter_id, stride, stride_index, frame_idx_estimated, latency
    """
    per = seq_data.get("per_frame_pipeline_timing", {}) or {}
    count = per.get("count", {}) or {}
    rows = []
    for counter_key, latency_list in count.items():
        stride = _extract_stride_from_counter_key(counter_key)
        counter_id = _extract_counter_id(counter_key)
        if not isinstance(latency_list, Iterable):
            continue
        for idx, latency in enumerate(latency_list):
            latency_ms = float(latency) * 1000 if latency is not None else pd.NA
            rows.append({
                "sequence": sequence_name,
                "counter_key": counter_key,
                "counter_id": counter_id,
                "stride": stride,
                "stride_index": idx,
                "frame_idx_estimated": idx * stride,
                "latency": latency_ms
            })
    if not rows:
        return pd.DataFrame(columns=["sequence","counter_key","counter_id","stride","stride_index","frame_idx_estimated","latency"])
    return pd.DataFrame(rows)

def parse_sequence(seq_name: str, seq_data: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse one sequence dictionary and return three DataFrames:
      detection_df, tracking_df, counting_df
    """
    det = parse_detection_latencies(seq_data, seq_name)
    trk = parse_tracking_latencies(seq_data, seq_name)
    cnt = parse_count_latencies(seq_data, seq_name)
    return det, trk, cnt

def parse_latencies(results_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse the entire JSON root (big JSON file) OR a folder (meta.json + sequences/*.json).

    Returns combined detection_df, tracking_df, counting_df across all sequences.
    Streaming-friendly for the folder case.
    """
    container = load_results_like_old_or_new(results_path)
    seq_iter = container.get("sequences_iter", [])
    det_list, trk_list, cnt_list = [], [], []
    seq_count = 0
    for seq_name, seq_data in seq_iter:
        seq_count += 1
        det, trk, cnt = parse_sequence(seq_name, seq_data)
        if not det.empty:
            det_list.append(det)
        if not trk.empty:
            trk_list.append(trk)
        if not cnt.empty:
            cnt_list.append(cnt)
        if seq_count % 100 == 0:
            print(f"[INFO] Parsed {seq_count} sequences...")
    detection_df = pd.concat(det_list, ignore_index=True) if det_list else pd.DataFrame()
    tracking_df = pd.concat(trk_list, ignore_index=True) if trk_list else pd.DataFrame()
    counting_df = pd.concat(cnt_list, ignore_index=True) if cnt_list else pd.DataFrame()
    return detection_df, tracking_df, counting_df