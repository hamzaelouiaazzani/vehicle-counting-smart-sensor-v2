
#!/usr/bin/env python3
"""
Modified counting pipeline script.
Changes: add CLI args to control saving of heavy/per-frame data:
  --save-per-frame-counts         (default: False)
  --save-per-frame-timing         (default: False)
  --no-save-video-characteristics (default: keep saving video characteristics = True)

Behavior summary:
- When --save-per-frame-counts is NOT provided (default False) the script will NOT allocate
  or append per-frame counts (total or per-class) and they will not be included in output.
- When --save-per-frame-timing is NOT provided (default False) the script will NOT allocate
  or append per-frame timing data and it will not be included in output.
- By default video characteristics are saved. Use --no-save-video-characteristics to disable
  writing the video metadata block; note the code still probes minimal width/height needed
  to compute the counting line.

I kept the original pipeline logic and only guarded allocations/appends/serialization
with the boolean flags.
"""

from itertools import product
from pathlib import Path
import json
import logging
from tqdm import tqdm
from datetime import datetime
import copy
import gc
import psutil
import argparse
from ast import literal_eval

import numpy as np
import pandas as pd

# local packages (keep your project structure)
from framegrabber.frame_grabber import FrameGrabber
from detection.ultralytics_detectors import UltralyticsDetector
from tracking.track import Tracker
from counting.count_config_loader import CountingConfigLoader
from utils.profilers import Profile

# -----------------------
# Defaults / constants
# -----------------------
logging.getLogger().setLevel(logging.WARNING)

# base datasets root (fixed, not a CLI arg)
BASE_DATASET_ROOT = Path(r"/home/hamza")
BASE_DATASET_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets")

DEFAULT_DATASET = "IMAROC_2"

# These will be set in main() via make_paths()
RESULTS_DIR = None
RESULTS_FILE = None
FPS = 30

STRIDES = [1, 2, 3, 4]
TRACKLESS_VICINITIES = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
TRACK_VICINITIES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
TRACKERS_AVAILABLE = ['bytetrack', 'ocsort', 'strongsort', 'botsort', 'deepocsort', 'boosttrack', 'hybridsort']

# Full superset counting map (six counters)
COUNTING_MAP_DEFAULT = {
    "counter_0": ["by_vicinity"],
    "counter_1": ["by_id"],
    "counter_2": ["by_cross_id"],
    "counter_3": ["by_id", "by_vicinity"],
    "counter_4": ["by_cross_id", "by_vicinity"],
    "counter_5": ["by_cross_id", "by_id", "by_vicinity"],
}

# COCO-like vehicle class mapping kept as before
COCO_VEHICLES = [1, 2, 3, 5, 7]
CLASS_MAP = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# global counter loader (will be initialized in main so frame_size can be configurable)
counter_load = None

# -----------------------
# Path helpers
# -----------------------

def make_paths(dataset_name: str, model_name: str):
    if dataset_name == "IMAROC_1":
        dataset_path = BASE_DATASET_ROOT / "IMAROC_1"
        results_dir = Path("counting") / "experiments" / "IMAROC1"
    elif dataset_name == "IMAROC_2":
        dataset_path = BASE_DATASET_ROOT / "IMAROC_2"
        results_dir = Path("counting") / "experiments" / "IMAROC2"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    results_dir.mkdir(parents=True, exist_ok=True)
    model_stem = Path(model_name).stem.replace(".", "_")
    results_file = results_dir / f"{model_stem}_{dataset_name}.json"
    return dataset_path, results_dir, results_file

# -----------------------
# Lightweight result-storage helpers (resume-friendly)
# -----------------------

def init_results_storage(results_dir: Path, results_file: Path):
    """Prepare meta.json, sequences dir and index file. Return meta dict and paths."""
    sequences_dir = results_dir / "sequences"
    sequences_dir.mkdir(parents=True, exist_ok=True)
    meta_file = results_dir / "meta.json"
    index_file = results_dir / "sequences_index.jsonl"

    if meta_file.exists():
        try:
            with meta_file.open("r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}
    else:
        meta = {}

    # ensure index exists (will be append-only)
    if not index_file.exists():
        index_file.touch()

    return {"meta": meta, "meta_file": meta_file, "sequences_dir": sequences_dir, "index_file": index_file, "results_file": results_file}


def is_sequence_done(sequences_dir: Path, seq_key: str) -> bool:
    return (sequences_dir / f"{seq_key}.json").exists()


def write_sequence_result(sequences_dir: Path, index_file: Path, seq_key: str, seq_entry: dict):
    """Atomically write per-sequence JSON and append an index line."""
    path = sequences_dir / f"{seq_key}.json"
    # atomic write using tmp -> replace
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(seq_entry, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)

    # append to index (one JSON object per line)
    with index_file.open("a", encoding="utf-8") as fh:
        json.dump({"sequence": seq_key, "timestamp_utc": datetime.utcnow().isoformat() + "Z"}, fh, ensure_ascii=False)
        fh.write("\n")


def get_processed_sequences(sequences_dir: Path):
    """Return set of processed sequence keys (by listing files)."""
    if not sequences_dir.exists():
        return set()
    return {p.stem for p in sequences_dir.glob("*.json")}

# Reuse existing atomic writer for small meta files
def _atomic_write_json(path: Path, data):
    tmp = path.with_suffix(".tmp.json")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    tmp.replace(path)

# -----------------------
# Helpers (unchanged names where possible)
# -----------------------

def get_sequences(path: Path):
    return sorted([p.name for p in path.glob("*.mp4")], key=lambda x: int(Path(x).stem.replace("kech", "")))


def exp_id_for(seq, counter, stride, tracker=None, vicinity=None):
    parts = [str(seq), counter, f"s{stride}"]
    parts.append(f"tracker-{tracker}" if tracker is not None else "tracker-none")
    parts.append(f"vic-{vicinity}" if vicinity is not None else "vic-none")
    return "__".join(parts)


def parse_exp_id(exp_id):
    parts = exp_id.split("__")
    return {
        "sequence": parts[0],
        "counter": parts[1],
        "stride": int(parts[2].lstrip("s")),
        "tracker": None if parts[3].split("tracker-")[1] == "none" else parts[3].split("tracker-")[1],
        "vicinity": None if parts[4].split("vic-")[1] == "none" else float(parts[4].split("vic-")[1]),
    }


def _load_or_init_results(path: Path):
    # keep for backward compat but do not rely on it for sequences
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            backup = path.with_suffix(".broken.json")
            path.rename(backup)
            return {"meta": {}, "sequences": {}}
    return {"meta": {}, "sequences": {}}


# -----------------------
# Detector / model init
# -----------------------

def init_detector(model_path: str = "rtdetr-l.pt", conf: float = 0.10):
    model = UltralyticsDetector(model_path, conf=conf)
    try:
        device = model.predictor.device
    except Exception:
        device = getattr(model, "device", None)
    pipeline_meta = {
        "detector": {"name": str(model_path), "repo": "https://github.com/ultralytics/ultralytics", "input_size": "640x640", "conf_thresh": conf, "nms_iou": 0.7}
    }
    return model, device, pipeline_meta


# -----------------------
# Counters factory (full superset)
# -----------------------

def make_all_counter_instances(seq, counter_load_local, strides, trackless_vics, track_vics, all_trackers, line, counting_map):
    instances = {c: {} for c in counting_map.keys()}

    def _exp_id(counter, s, tracker, vic):
        return exp_id_for(seq, counter, s, tracker, vic)

    base = {"enable": True, "box_in_polygon_mode": "center_in_polygon", "line": line}

    if "counter_0" in counting_map:
        for s, v in product(strides, trackless_vics):
            cid = _exp_id("counter_0", s, None, v)
            cfg = copy.deepcopy(base)
            cfg.update({"name": cid, "counting_logic": counting_map["counter_0"], "line_vicinity": v, "tracker": None})
            instances["counter_0"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

    if "counter_1" in counting_map:
        for s, tr in product(strides, all_trackers):
            cid = _exp_id("counter_1", s, tr, None)
            cfg = copy.deepcopy(base)
            cfg.update({"name": cid, "counting_logic": counting_map["counter_1"], "line_vicinity": None, "tracker": tr})
            instances["counter_1"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

    if "counter_2" in counting_map:
        for s, tr in product(strides, all_trackers):
            cid = _exp_id("counter_2", s, tr, None)
            cfg = copy.deepcopy(base)
            cfg.update({"name": cid, "counting_logic": counting_map["counter_2"], "line_vicinity": None, "tracker": tr})
            instances["counter_2"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

    for cname in ("counter_3", "counter_4", "counter_5"):
        if cname not in counting_map:
            continue
        for s, tr, v in product(strides, all_trackers, track_vics):
            cid = _exp_id(cname, s, tr, v)
            cfg = copy.deepcopy(base)
            cfg.update({"name": cid, "counting_logic": counting_map[cname], "line_vicinity": v, "tracker": tr})
            instances[cname][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

    return instances


# -----------------------
# Video / annotation helpers (dataset aware)
# -----------------------

def get_video_characteristics(sequence: str, dataset_path: Path):
    source = str(dataset_path / sequence)
    fg = FrameGrabber(source=source, stride=1)
    if not fg.open():
        raise RuntimeError(f"Failed to open source {source} for probing")
    try:
        fps = getattr(fg, "fps", None) or getattr(fg, "_fps", None)
        frame = fg.get_frame(timeout=0.5)
        if frame is None:
            raise RuntimeError(f"No frames available in {source}")
        h, w = frame.data.shape[:2]
        if fps is None:
            try:
                cap = getattr(fg, "_cap", None)
                if cap is not None:
                    fps_val = cap.get(5)  # CV_CAP_PROP_FPS == 5
                    fps = fps_val if fps_val and fps_val > 0 else FPS
                else:
                    fps = FPS
            except Exception:
                fps = FPS
    finally:
        fg.release()

    return int(w), int(h), float(fps)


def _line_pixels_imaroc1(raw_str: str, width: int, height: int):
    raw = literal_eval(raw_str)
    (x1f, y1f), (x2f, y2f) = raw
    x1 = int(round(float(x1f) * width))
    y1 = int(round(float(y1f) * height))
    x2 = int(round(float(x2f) * width))
    y2 = int(round(float(y2f) * height))
    return ((x1, y1), (x2, y2))


def _line_pixels_imaroc2(raw_str: str):
    raw = literal_eval(raw_str)
    (x1, y1), (x2, y2) = raw
    return ((int(x1), int(y1)), (int(x2), int(y2)))


def get_line_video(dataset_name: str, video_name: str, annotation_df: pd.DataFrame, width: int, height: int):
    if dataset_name == "IMAROC_1":
        row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower(), "line_of_counting"]
        if row.empty:
            raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_1)")
        return _line_pixels_imaroc1(row.iloc[0], width, height)
    else:
        row = annotation_df.loc[annotation_df["video_name"] == video_name, "line_of_counting"]
        if row.empty:
            raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_2)")
        return _line_pixels_imaroc2(row.iloc[0])


def _extract_counts_only(annotated_dict: dict, dataset_name: str):
    if not annotated_dict:
        return {}
    counts_keys = [k for k in annotated_dict.keys() if k.startswith("actual_")]
    counts = {}
    for k in counts_keys:
        try:
            counts[k] = int(annotated_dict.get(k, 0))
        except Exception:
            counts[k] = annotated_dict.get(k)
    return counts


def get_annotated_counts(dataset_name: str, video_name: str, annotation_df: pd.DataFrame):
    if dataset_name == "IMAROC_1":
        row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower()]
        if row.empty:
            return None
        r = row.iloc[0]
        annotated = {
            "actual_total_vehicles": int(r.get("actual_total_vehicles", 0)),
            "actual_car_counts": int(r.get("actual_car_counts", 0)),
            "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
            "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
            "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
            "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
            "camera_perspective": r.get("camera_perspective", None),
            "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
        }
        return annotated
    else:
        row = annotation_df.loc[annotation_df["video_name"] == video_name]
        if row.empty:
            return None
        r = row.iloc[0]
        annotated = {
            "recording_time": r.get("recording_time", None),
            "annotated_width": int(r.get("width", 0)) if pd.notna(r.get("width", None)) else None,
            "annotated_height": int(r.get("height", 0)) if pd.notna(r.get("height", None)) else None,
            "direction_relative_to_camera": r.get("direction_relative_to_camera", None),
            "road_directionality": r.get("road_directionality", None),
            "lighting": r.get("lighting", None),
            "actual_car_counts": int(r.get("actual_car_counts", 0)),
            "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
            "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
            "actual_van_counts": int(r.get("actual_van_counts", 0)),
            "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
            "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
            "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
        }
        return annotated

# -----------------------
# Per-frame pipeline (unchanged logic), adapted to per-sequence output file
# -----------------------

def process_sequence_per_stride_trackers_diagnose(
    sequence, my_model, device,
    dataset_name, dataset_path, annotation_df,
    periodic_collect=200, empty_cache_every=200,
    counting_map=None, selected_trackers=None,
    counter_load_local=None, storage=None,
    save_per_frame_counts=False, save_per_frame_timing=False, save_video_characteristics=True,
):
    """
    storage is the dict returned by init_results_storage:
      { "meta":..., "meta_file":Path, "sequences_dir":Path, "index_file":Path, "results_file":Path }

    The three boolean flags control whether to allocate/collect/write heavy per-frame
    data. When False (default for counts/timing) the pipeline will skip per-frame
    storage and the associated list appends to reduce memory/IO overhead.
    """
    seq_key = str(sequence)
    sequences_dir = storage["sequences_dir"]
    index_file = storage["index_file"]
    meta = storage["meta"]

    if is_sequence_done(sequences_dir, seq_key):
        print(f"Sequence {seq_key} already processed  skipping.")
        return

    # local seq_entry (no global all_results mutated; avoids loading all sequences)
    seq_entry = {}
    seq_entry.setdefault("counts", {})
    # only create timing container if requested
    if save_per_frame_timing:
        seq_entry.setdefault("per_frame_pipeline_timing", {})
    else:
        seq_entry.setdefault("per_frame_pipeline_timing", {})  # keep empty dict for compatibility
    seq_entry.setdefault("video", {})

    # probe video characteristics (we still probe minimal info; saving is controlled by flag)
    width, height, fps_vid = get_video_characteristics(sequence, dataset_path)

    # dataset-specific line and annotated counts
    counting_line = get_line_video(dataset_name, Path(sequence).name, annotation_df, width=width, height=height)
    annotated_full = get_annotated_counts(dataset_name, Path(sequence).name, annotation_df)

    total_frames_annot = annotated_full.get("total_frames") if annotated_full is not None else None

    # build video metadata; separate annotated_counts (only counts) from video_metadata (other fields)
    annotated_counts = _extract_counts_only(annotated_full, dataset_name) if annotated_full is not None else {}

    video_metadata = {"width": int(width), "height": int(height), "fps": float(fps_vid), "line_of_counting": counting_line}
    # add dataset-provided extra metadata (non-counts)
    if annotated_full is not None:
        for k, v in annotated_full.items():
            if not k.startswith("actual_") and k != "total_frames":
                video_metadata[k] = v

    video_meta = {
        "annotated_counts": annotated_counts,
        # include video_metadata only when requested
        "video_metadata": video_metadata if save_video_characteristics else {},
        "n_frames": int(total_frames_annot) if total_frames_annot is not None else None,
    }

    seq_entry["video"] = video_meta

    # prepare counters & exp_map (use provided counting_map and selected_trackers)
    counter_instances = make_all_counter_instances(
        Path(sequence).stem,
        counter_load_local,
        strides=STRIDES,
        trackless_vics=TRACKLESS_VICINITIES,
        track_vics=TRACK_VICINITIES,
        all_trackers=selected_trackers,
        line=counting_line,
        counting_map=counting_map,
    )

    exp_map = {}
    for fam, exps in counter_instances.items():
        for eid, counter in exps.items():
            cfg = parse_exp_id(eid)
            exp_map[eid] = {"counter": counter, "cfg": cfg}

    # per-exp state
    per_exp_prev_total = {}
    per_exp_prev_bycls = {}
    # only allocate per-frame lists if requested
    per_exp_frame_totals = {eid: [] for eid in exp_map.keys()} if save_per_frame_counts else None
    per_exp_frame_bycls = {eid: [] for eid in exp_map.keys()} if save_per_frame_counts else None

    for eid, v in exp_map.items():
        prev_total = int(getattr(v["counter"].count_result, "total_count", 0))
        arr = getattr(v["counter"].count_result, "counts_by_class", None)
        prev_bycls = np.asarray(arr, dtype=int) if arr is not None else np.zeros(0, dtype=int)
        per_exp_prev_total[eid] = prev_total
        per_exp_prev_bycls[eid] = prev_bycls.copy()

    # timing structures (only prepare containers if requested)
    per_frame_pipeline_timing = {"pre": [], "inf": [], "post": [], "n_detections": [], "track": {}, "count": {}} if save_per_frame_timing else {}
    if save_per_frame_timing:
        for tr in selected_trackers:
            per_frame_pipeline_timing.setdefault("track", {})
            per_frame_pipeline_timing["track"].setdefault(tr, {})
            for s in STRIDES:
                per_frame_pipeline_timing["track"][tr].setdefault(s, [])
        for eid in exp_map.keys():
            per_frame_pipeline_timing.setdefault("count", {})
            per_frame_pipeline_timing["count"].setdefault(eid, [])

    # create tracker instances per (tracker,stride)
    trackers = {}
    track_profiles = {}
    for tr in selected_trackers:
        for s in STRIDES:
            key = (tr, s)
            trackers[key] = Tracker(tr)
            track_profiles[key] = Profile() if tr in ("ocsort", "bytetrack") else Profile(device=device)

    # open FrameGrabber and setup tqdm
    source = str(dataset_path / sequence)
    fg = FrameGrabber(source=source, stride=1)
    if not fg.open():
        raise RuntimeError(f"Failed to open source {source}")
    if getattr(fg, "_grabber_mode", None) == "queue":
        fg.start()

    pre_profile = Profile(device=device)
    inf_profile = Profile(device=device)
    post_profile = Profile(device=device)

    total_for_tqdm = int(total_frames_annot) if total_frames_annot is not None and total_frames_annot > 0 else None
    frame_pbar = tqdm(total=total_for_tqdm, desc=f"Frames {sequence}", unit="frame")

    frame_counter = 0
    print(f"Starting diagnosed per-frame pipeline for {sequence} (total={total_for_tqdm})")
    try:
        while True:
            frame = fg.get_frame(timeout=0.1)
            if frame is None:
                break
            idx = frame.read_idx

            # detection
            with pre_profile:
                pre = my_model.preprocess(frame.data)
            with inf_profile:
                raw = my_model.infer(pre)
            with post_profile:
                ready = my_model.postprocess(raw, pre, frame.data)

            if save_per_frame_timing:
                per_frame_pipeline_timing["pre"].append(float(pre_profile.dt))
                per_frame_pipeline_timing["inf"].append(float(inf_profile.dt))
                per_frame_pipeline_timing["post"].append(float(post_profile.dt))
                per_frame_pipeline_timing["n_detections"].append(int(len(ready)))

            # update trackers for which this frame is a sample (idx % s == 0)
            current_tracks = {}
            for tr in selected_trackers:
                for s in STRIDES:
                    if (idx % s) != 0:
                        continue
                    key = (tr, s)
                    tp = track_profiles[key]
                    with tp:
                        try:
                            tracks = trackers[key].update(ready, frame.data)
                        except Exception as e:
                            print(f"[WARN] tracker {tr} s{s} failed at frame {idx}: {e}")
                            tracks = []
                    current_tracks[key] = tracks
                    if save_per_frame_timing:
                        per_frame_pipeline_timing["track"][tr][s].append(float(tp.dt))

            # counting
            for eid, v in exp_map.items():
                cfg = v["cfg"]
                stride = cfg["stride"]
                tracker_name = cfg["tracker"]
                if (idx % stride) != 0:
                    continue

                # proceed to counting
                count_obj = v["counter"]
                if tracker_name is None:
                    count_input = ready
                else:
                    key = (tracker_name, stride)
                    count_input = current_tracks.get(key, [])

                cp = Profile()
                with cp:
                    try:
                        res = count_obj.count(count_input)
                    except Exception as e:
                        print(f"[WARN] counter {eid} failed at frame {idx}: {e}")
                        res = None
                if save_per_frame_timing:
                    per_frame_pipeline_timing["count"][eid].append(float(cp.dt))

                if res is None:
                    if save_per_frame_counts:
                        per_exp_frame_totals[eid].append(0)
                        per_exp_frame_bycls[eid].append([])
                    continue

                curr_total = int(getattr(res, "total_count", 0))
                prev_total = per_exp_prev_total[eid]
                if save_per_frame_counts:
                    per_exp_frame_totals[eid].append(int(curr_total - prev_total))
                per_exp_prev_total[eid] = curr_total

                curr_bycls = getattr(res, "counts_by_class", None)
                curr_bycls = np.asarray(curr_bycls, dtype=int) if curr_bycls is not None else np.zeros(0, dtype=int)
                prev_bycls = per_exp_prev_bycls[eid]
                max_len = max(prev_bycls.size, curr_bycls.size)
                if max_len == 0:
                    if save_per_frame_counts:
                        per_exp_frame_bycls[eid].append([])
                    per_exp_prev_bycls[eid] = np.zeros(0, dtype=int)
                else:
                    prev_pad = np.pad(prev_bycls, (0, max_len - prev_bycls.size), mode='constant')
                    curr_pad = np.pad(curr_bycls, (0, max_len - curr_bycls.size), mode='constant')
                    if save_per_frame_counts:
                        per_exp_frame_bycls[eid].append((curr_pad - prev_pad).tolist())
                    per_exp_prev_bycls[eid] = curr_pad.copy()

                try:
                    del res
                except Exception:
                    pass

            # update progress and show concise postfix
            frame_counter += 1
            rss_gb = psutil.Process().memory_info().rss / 1e9
            frame_pbar.set_postfix_str(f"seq={sequence} idx={idx} RAM={rss_gb:.2f}GB")
            frame_pbar.update(1)

            # periodic regulation
            if (frame_counter % periodic_collect) == 0:
                gc.collect()
            if (frame_counter % empty_cache_every) == 0:
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    finally:
        try:
            fg.release()
        except Exception:
            pass
        frame_pbar.close()

    # finalize counts
    for eid in exp_map.keys():
        total_final = per_exp_prev_total[eid]
        counts_by_class = per_exp_prev_bycls[eid].tolist() if isinstance(per_exp_prev_bycls[eid], np.ndarray) else []
        counts_obj = {
            "total_count": int(total_final),
            "counts_by_class": counts_by_class,
        }
        # include per-frame lists only when requested
        if save_per_frame_counts:
            counts_obj["per_frame_counts"] = per_exp_frame_totals[eid]
            counts_obj["per_frame_counts_by_class"] = per_exp_frame_bycls[eid]
        seq_entry["counts"][eid] = counts_obj

    # include per-frame timing only when requested
    seq_entry["per_frame_pipeline_timing"] = per_frame_pipeline_timing if save_per_frame_timing else {}

    # update meta timestamp in storage (small file)
    storage["meta"]["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

    # cleanup
    try:
        for t in trackers.values():
            del t
    except Exception:
        pass
    try:
        del exp_map
    except Exception:
        pass
    gc.collect()

    # write per-sequence result atomically and append index
    write_sequence_result(sequences_dir, index_file, Path(sequence).stem, seq_entry)
    # update meta.json atomically
    try:
        _atomic_write_json(storage["meta_file"], storage["meta"])
    except Exception:
        pass

    print(f"Completed sequence {sequence} (diagnosed per-(tracker,stride) pipeline).")

# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run per-frame diagnosed counting experiments on IMAROC_1 or IMAROC_2 dataset."
    )
    parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET, choices=["IMAROC_1", "IMAROC_2"],
                        help="Dataset to run (IMAROC_1 or IMAROC_2) (default: IMAROC_1)")
    parser.add_argument("--model", "-m", default="rtdetr-l.pt",
                        help="Detector model path (default: rtdetr-l.pt). Example: --model yolov8n.pt")
    parser.add_argument("--trackers", "-t", default="bytetrack,ocsort",
                        help="Comma-separated trackers subset to use (default: bytetrack,ocsort). "
                             f"Available: {', '.join(TRACKERS_AVAILABLE)}")
    parser.add_argument("--counters", "-c", default=",".join(sorted(COUNTING_MAP_DEFAULT.keys())),
                        help="Comma-separated counter families to run (default: all 6 counters). "
                             f"Available: {', '.join(sorted(COUNTING_MAP_DEFAULT.keys()))}")
    parser.add_argument("--frame-size", default=1000, type=int,
                        help="Frame size passed to CountingConfigLoader (default 1000)")

    # new flags controlling saving
    parser.add_argument("--save-per-frame-counts", action="store_true", dest="save_per_frame_counts",
                        help="Save per-frame total and per-class counts into sequence JSON (default: False)")
    parser.add_argument("--save-per-frame-timing", action="store_true", dest="save_per_frame_timing",
                        help="Save per-frame pipeline timing into sequence JSON (default: False)")
    # video characteristics default True; provide a --no-save-video-characteristics flag to disable
    parser.add_argument("--no-save-video-characteristics", action="store_false", dest="save_video_characteristics",
                        help="Do not save video characteristics (width/height/fps/line) into sequence JSON (default: save them)")
    parser.set_defaults(save_per_frame_counts=False, save_per_frame_timing=False, save_video_characteristics=True)

    args = parser.parse_args()

    dataset_name = args.dataset

    # compute dataset_path/results_dir/results_file from function (not CLI args)
    dataset_path, results_dir, results_file = make_paths(dataset_name=dataset_name, model_name=args.model)
    if not dataset_path.exists():
        raise RuntimeError(f"Provided dataset path does not exist: {dataset_path}")

    # init storage (meta, sequences dir, index)
    storage = init_results_storage(results_dir, results_file)

    # load annotation CSV (name is same in both: actual_counts.csv)
    annotation_path = dataset_path / "actual_counts.csv"
    if not annotation_path.exists():
        raise RuntimeError(f"annotation file not found at {annotation_path}")
    annotation_df = pd.read_csv(annotation_path)

    # parse trackers and counters
    requested_trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
    selected_trackers = [t for t in requested_trackers if t in TRACKERS_AVAILABLE]
    if not selected_trackers:
        selected_trackers = ['bytetrack', 'ocsort']

    requested_counters = [c.strip() for c in args.counters.split(",") if c.strip()]
    counting_map = {k: v for k, v in COUNTING_MAP_DEFAULT.items() if k in requested_counters}
    if not counting_map:
        counting_map = COUNTING_MAP_DEFAULT.copy()

    # initialize model and counter loader
    my_model, device, pipeline_meta = init_detector(model_path=args.model)
    global counter_load
    counter_load = CountingConfigLoader(default_classes=COCO_VEHICLES, frame_size=args.frame_size)

    # prepare meta (small) and write initial meta.json (kept small)
    storage["meta"].update({
        "tool": f"Evaluation of vision-based vehicle counting algorithms on {dataset_name} Video Dataset (diagnosed per-(tracker,stride) trackers)",
        "dataset": dataset_name,
        "model": str(args.model),
        "version": "1.0",
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "fps": FPS,
        "class_map": CLASS_MAP,
        "counting_map": counting_map,
        "pipeline": pipeline_meta,
        "description": f"Dataset={dataset_name}; Model={args.model}; Selected trackers={selected_trackers}; Selected counters={list(counting_map.keys())}. Per-frame detection + per-(tracker,stride) tracker instances; memory regulation and diagnostics. Results stored per-sequence in {storage['sequences_dir']}",
    })
    _atomic_write_json(storage["meta_file"], storage["meta"])

    sequences = get_sequences(dataset_path)
    seq_pbar = tqdm(sequences, desc="Sequences", unit="seq")
    for seq_idx, sequence in enumerate(seq_pbar):
        seq_pbar.set_description(f"Seq: {sequence}")
        process_sequence_per_stride_trackers_diagnose(
            sequence, my_model, device,
            dataset_name=dataset_name, dataset_path=dataset_path, annotation_df=annotation_df,
            counting_map=counting_map, selected_trackers=selected_trackers,
            counter_load_local=counter_load, storage=storage,
            save_per_frame_counts=args.save_per_frame_counts,
            save_per_frame_timing=args.save_per_frame_timing,
            save_video_characteristics=args.save_video_characteristics,
        )
        print(f"Sequence {sequence} is fully processed")

    print(f"All experiments completed (diagnosed per-(tracker,stride) trackers) on {dataset_name}.")


if __name__ == '__main__':
    main()






























































# #!/usr/bin/env python3
# from itertools import product
# from pathlib import Path
# import json
# import logging
# from tqdm import tqdm
# from datetime import datetime
# import copy
# import gc
# import psutil
# import argparse
# from ast import literal_eval

# import numpy as np
# import pandas as pd

# # local packages (keep your project structure)
# from framegrabber.frame_grabber import FrameGrabber
# from detection.ultralytics_detectors import UltralyticsDetector
# from tracking.track import Tracker
# from counting.count_config_loader import CountingConfigLoader
# from utils.profilers import Profile

# # -----------------------
# # Defaults / constants
# # -----------------------
# logging.getLogger().setLevel(logging.WARNING)

# # base datasets root (fixed, not a CLI arg)
# BASE_DATASET_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets")

# DEFAULT_DATASET = "IMAROC_2"

# # These will be set in main() via make_paths()
# RESULTS_DIR = None
# RESULTS_FILE = None
# FPS = 30

# STRIDES = [1, 2, 3, 4]
# TRACKLESS_VICINITIES = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
# TRACK_VICINITIES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# TRACKERS_AVAILABLE = ['bytetrack', 'ocsort', 'strongsort', 'botsort', 'deepocsort', 'boosttrack', 'hybridsort']

# # Full superset counting map (six counters)
# COUNTING_MAP_DEFAULT = {
#     "counter_0": ["by_vicinity"],
#     "counter_1": ["by_id"],
#     "counter_2": ["by_cross_id"],
#     "counter_3": ["by_id", "by_vicinity"],
#     "counter_4": ["by_cross_id", "by_vicinity"],
#     "counter_5": ["by_cross_id", "by_id", "by_vicinity"],
# }

# # COCO-like vehicle class mapping kept as before
# COCO_VEHICLES = [1, 2, 3, 5, 7]
# CLASS_MAP = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # global counter loader (will be initialized in main so frame_size can be configurable)
# counter_load = None

# # -----------------------
# # Path helpers
# # -----------------------

# def make_paths(dataset_name: str, model_name: str):
#     if dataset_name == "IMAROC_1":
#         dataset_path = BASE_DATASET_ROOT / "IMAROC_1"
#         results_dir = Path("counting") / "experiments" / "IMAROC1"
#     elif dataset_name == "IMAROC_2":
#         dataset_path = BASE_DATASET_ROOT / "IMAROC_2"
#         results_dir = Path("counting") / "experiments" / "IMAROC2"
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     results_dir.mkdir(parents=True, exist_ok=True)
#     model_stem = Path(model_name).stem.replace(".", "_")
#     results_file = results_dir / f"{model_stem}_{dataset_name}.json"
#     return dataset_path, results_dir, results_file

# # -----------------------
# # Lightweight result-storage helpers (resume-friendly)
# # -----------------------

# def init_results_storage(results_dir: Path, results_file: Path):
#     """Prepare meta.json, sequences dir and index file. Return meta dict and paths."""
#     sequences_dir = results_dir / "sequences"
#     sequences_dir.mkdir(parents=True, exist_ok=True)
#     meta_file = results_dir / "meta.json"
#     index_file = results_dir / "sequences_index.jsonl"

#     if meta_file.exists():
#         try:
#             with meta_file.open("r", encoding="utf-8") as fh:
#                 meta = json.load(fh)
#         except Exception:
#             meta = {}
#     else:
#         meta = {}

#     # ensure index exists (will be append-only)
#     if not index_file.exists():
#         index_file.touch()

#     return {"meta": meta, "meta_file": meta_file, "sequences_dir": sequences_dir, "index_file": index_file, "results_file": results_file}


# def is_sequence_done(sequences_dir: Path, seq_key: str) -> bool:
#     return (sequences_dir / f"{seq_key}.json").exists()


# def write_sequence_result(sequences_dir: Path, index_file: Path, seq_key: str, seq_entry: dict):
#     """Atomically write per-sequence JSON and append an index line."""
#     path = sequences_dir / f"{seq_key}.json"
#     # atomic write using tmp -> replace
#     tmp = path.with_suffix(".tmp.json")
#     with tmp.open("w", encoding="utf-8") as fh:
#         json.dump(seq_entry, fh, ensure_ascii=False, indent=2)
#     tmp.replace(path)

#     # append to index (one JSON object per line)
#     with index_file.open("a", encoding="utf-8") as fh:
#         json.dump({"sequence": seq_key, "timestamp_utc": datetime.utcnow().isoformat() + "Z"}, fh, ensure_ascii=False)
#         fh.write("\n")


# def get_processed_sequences(sequences_dir: Path):
#     """Return set of processed sequence keys (by listing files)."""
#     if not sequences_dir.exists():
#         return set()
#     return {p.stem for p in sequences_dir.glob("*.json")}

# # Reuse existing atomic writer for small meta files
# def _atomic_write_json(path: Path, data):
#     tmp = path.with_suffix(".tmp.json")
#     with tmp.open("w", encoding="utf-8") as fh:
#         json.dump(data, fh, ensure_ascii=False, indent=2)
#     tmp.replace(path)

# # -----------------------
# # Helpers (unchanged names where possible)
# # -----------------------

# def get_sequences(path: Path):
#     return sorted([p.name for p in path.glob("*.mp4")], key=lambda x: int(Path(x).stem.replace("kech", "")))


# def exp_id_for(seq, counter, stride, tracker=None, vicinity=None):
#     parts = [str(seq), counter, f"s{stride}"]
#     parts.append(f"tracker-{tracker}" if tracker is not None else "tracker-none")
#     parts.append(f"vic-{vicinity}" if vicinity is not None else "vic-none")
#     return "__".join(parts)


# def parse_exp_id(exp_id):
#     parts = exp_id.split("__")
#     return {
#         "sequence": parts[0],
#         "counter": parts[1],
#         "stride": int(parts[2].lstrip("s")),
#         "tracker": None if parts[3].split("tracker-")[1] == "none" else parts[3].split("tracker-")[1],
#         "vicinity": None if parts[4].split("vic-")[1] == "none" else float(parts[4].split("vic-")[1]),
#     }


# def _load_or_init_results(path: Path):
#     # keep for backward compat but do not rely on it for sequences
#     if path.exists():
#         try:
#             with path.open("r", encoding="utf-8") as fh:
#                 return json.load(fh)
#         except Exception:
#             backup = path.with_suffix(".broken.json")
#             path.rename(backup)
#             return {"meta": {}, "sequences": {}}
#     return {"meta": {}, "sequences": {}}


# # -----------------------
# # Detector / model init
# # -----------------------

# def init_detector(model_path: str = "rtdetr-l.pt", conf: float = 0.10):
#     model = UltralyticsDetector(model_path, conf=conf)
#     try:
#         device = model.predictor.device
#     except Exception:
#         device = getattr(model, "device", None)
#     pipeline_meta = {
#         "detector": {"name": str(model_path), "repo": "https://github.com/ultralytics/ultralytics", "input_size": "640x640", "conf_thresh": conf, "nms_iou": 0.7}
#     }
#     return model, device, pipeline_meta

# # -----------------------
# # Counters factory (full superset)
# # -----------------------

# def make_all_counter_instances(seq, counter_load_local, strides, trackless_vics, track_vics, all_trackers, line, counting_map):
#     instances = {c: {} for c in counting_map.keys()}

#     def _exp_id(counter, s, tracker, vic):
#         return exp_id_for(seq, counter, s, tracker, vic)

#     base = {"enable": True, "box_in_polygon_mode": "center_in_polygon", "line": line}

#     if "counter_0" in counting_map:
#         for s, v in product(strides, trackless_vics):
#             cid = _exp_id("counter_0", s, None, v)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_0"], "line_vicinity": v, "tracker": None})
#             instances["counter_0"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     if "counter_1" in counting_map:
#         for s, tr in product(strides, all_trackers):
#             cid = _exp_id("counter_1", s, tr, None)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_1"], "line_vicinity": None, "tracker": tr})
#             instances["counter_1"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     if "counter_2" in counting_map:
#         for s, tr in product(strides, all_trackers):
#             cid = _exp_id("counter_2", s, tr, None)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_2"], "line_vicinity": None, "tracker": tr})
#             instances["counter_2"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     for cname in ("counter_3", "counter_4", "counter_5"):
#         if cname not in counting_map:
#             continue
#         for s, tr, v in product(strides, all_trackers, track_vics):
#             cid = _exp_id(cname, s, tr, v)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map[cname], "line_vicinity": v, "tracker": tr})
#             instances[cname][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     return instances

# # -----------------------
# # Video / annotation helpers (dataset aware)
# # -----------------------

# def get_video_characteristics(sequence: str, dataset_path: Path):
#     source = str(dataset_path / sequence)
#     fg = FrameGrabber(source=source, stride=1)
#     if not fg.open():
#         raise RuntimeError(f"Failed to open source {source} for probing")
#     try:
#         fps = getattr(fg, "fps", None) or getattr(fg, "_fps", None)
#         frame = fg.get_frame(timeout=0.5)
#         if frame is None:
#             raise RuntimeError(f"No frames available in {source}")
#         h, w = frame.data.shape[:2]
#         if fps is None:
#             try:
#                 cap = getattr(fg, "_cap", None)
#                 if cap is not None:
#                     fps_val = cap.get(5)  # CV_CAP_PROP_FPS == 5
#                     fps = fps_val if fps_val and fps_val > 0 else FPS
#                 else:
#                     fps = FPS
#             except Exception:
#                 fps = FPS
#     finally:
#         fg.release()

#     return int(w), int(h), float(fps)


# def _line_pixels_imaroc1(raw_str: str, width: int, height: int):
#     raw = literal_eval(raw_str)
#     (x1f, y1f), (x2f, y2f) = raw
#     x1 = int(round(float(x1f) * width))
#     y1 = int(round(float(y1f) * height))
#     x2 = int(round(float(x2f) * width))
#     y2 = int(round(float(y2f) * height))
#     return ((x1, y1), (x2, y2))


# def _line_pixels_imaroc2(raw_str: str):
#     raw = literal_eval(raw_str)
#     (x1, y1), (x2, y2) = raw
#     return ((int(x1), int(y1)), (int(x2), int(y2)))


# def get_line_video(dataset_name: str, video_name: str, annotation_df: pd.DataFrame, width: int, height: int):
#     if dataset_name == "IMAROC_1":
#         row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower(), "line_of_counting"]
#         if row.empty:
#             raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_1)")
#         return _line_pixels_imaroc1(row.iloc[0], width, height)
#     else:
#         row = annotation_df.loc[annotation_df["video_name"] == video_name, "line_of_counting"]
#         if row.empty:
#             raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_2)")
#         return _line_pixels_imaroc2(row.iloc[0])


# def _extract_counts_only(annotated_dict: dict, dataset_name: str):
#     if not annotated_dict:
#         return {}
#     counts_keys = [k for k in annotated_dict.keys() if k.startswith("actual_")]
#     counts = {}
#     for k in counts_keys:
#         try:
#             counts[k] = int(annotated_dict.get(k, 0))
#         except Exception:
#             counts[k] = annotated_dict.get(k)
#     return counts


# def get_annotated_counts(dataset_name: str, video_name: str, annotation_df: pd.DataFrame):
#     if dataset_name == "IMAROC_1":
#         row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower()]
#         if row.empty:
#             return None
#         r = row.iloc[0]
#         annotated = {
#             "actual_total_vehicles": int(r.get("actual_total_vehicles", 0)),
#             "actual_car_counts": int(r.get("actual_car_counts", 0)),
#             "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
#             "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
#             "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
#             "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
#             "camera_perspective": r.get("camera_perspective", None),
#             "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
#         }
#         return annotated
#     else:
#         row = annotation_df.loc[annotation_df["video_name"] == video_name]
#         if row.empty:
#             return None
#         r = row.iloc[0]
#         annotated = {
#             "recording_time": r.get("recording_time", None),
#             "annotated_width": int(r.get("width", 0)) if pd.notna(r.get("width", None)) else None,
#             "annotated_height": int(r.get("height", 0)) if pd.notna(r.get("height", None)) else None,
#             "direction_relative_to_camera": r.get("direction_relative_to_camera", None),
#             "road_directionality": r.get("road_directionality", None),
#             "lighting": r.get("lighting", None),
#             "actual_car_counts": int(r.get("actual_car_counts", 0)),
#             "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
#             "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
#             "actual_van_counts": int(r.get("actual_van_counts", 0)),
#             "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
#             "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
#             "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
#         }
#         return annotated

# # -----------------------
# # Per-frame pipeline (unchanged logic), adapted to per-sequence output file
# # -----------------------

# def process_sequence_per_stride_trackers_diagnose(sequence, my_model, device,
#                                                    dataset_name, dataset_path, annotation_df,
#                                                    periodic_collect=200, empty_cache_every=200,
#                                                    counting_map=None, selected_trackers=None,
#                                                    counter_load_local=None, storage=None):
#     """
#     storage is the dict returned by init_results_storage:
#       { "meta":..., "meta_file":Path, "sequences_dir":Path, "index_file":Path, "results_file":Path }
#     """
#     seq_key = str(sequence)
#     sequences_dir = storage["sequences_dir"]
#     index_file = storage["index_file"]
#     meta = storage["meta"]

#     if is_sequence_done(sequences_dir, seq_key):
#         print(f"Sequence {seq_key} already processed — skipping.")
#         return

#     # local seq_entry (no global all_results mutated; avoids loading all sequences)
#     seq_entry = {}
#     seq_entry.setdefault("counts", {})
#     seq_entry.setdefault("per_frame_pipeline_timing", {})
#     seq_entry.setdefault("video", {})

#     # probe video characteristics
#     width, height, fps_vid = get_video_characteristics(sequence, dataset_path)

#     # dataset-specific line and annotated counts
#     counting_line = get_line_video(dataset_name, Path(sequence).name, annotation_df, width=width, height=height)
#     annotated_full = get_annotated_counts(dataset_name, Path(sequence).name, annotation_df)

#     total_frames_annot = annotated_full.get("total_frames") if annotated_full is not None else None

#     # build video metadata; separate annotated_counts (only counts) from video_metadata (other fields)
#     annotated_counts = _extract_counts_only(annotated_full, dataset_name) if annotated_full is not None else {}

#     video_metadata = {"width": int(width), "height": int(height), "fps": float(fps_vid), "line_of_counting": counting_line}
#     # add dataset-provided extra metadata (non-counts)
#     if annotated_full is not None:
#         for k, v in annotated_full.items():
#             if not k.startswith("actual_") and k != "total_frames":
#                 video_metadata[k] = v

#     video_meta = {
#         "annotated_counts": annotated_counts,
#         "video_metadata": video_metadata,
#         "n_frames": int(total_frames_annot) if total_frames_annot is not None else None,
#     }

#     seq_entry["video"] = video_meta

#     # prepare counters & exp_map (use provided counting_map and selected_trackers)
#     counter_instances = make_all_counter_instances(
#         Path(sequence).stem,
#         counter_load_local,
#         strides=STRIDES,
#         trackless_vics=TRACKLESS_VICINITIES,
#         track_vics=TRACK_VICINITIES,
#         all_trackers=selected_trackers,
#         line=counting_line,
#         counting_map=counting_map,
#     )

#     exp_map = {}
#     for fam, exps in counter_instances.items():
#         for eid, counter in exps.items():
#             cfg = parse_exp_id(eid)
#             exp_map[eid] = {"counter": counter, "cfg": cfg}

#     # per-exp state
#     per_exp_prev_total = {}
#     per_exp_prev_bycls = {}
#     per_exp_frame_totals = {}
#     per_exp_frame_bycls = {}
#     for eid, v in exp_map.items():
#         prev_total = int(getattr(v["counter"].count_result, "total_count", 0))
#         arr = getattr(v["counter"].count_result, "counts_by_class", None)
#         prev_bycls = np.asarray(arr, dtype=int) if arr is not None else np.zeros(0, dtype=int)
#         per_exp_prev_total[eid] = prev_total
#         per_exp_prev_bycls[eid] = prev_bycls.copy()
#         per_exp_frame_totals[eid] = []
#         per_exp_frame_bycls[eid] = []

#     # timing structures
#     per_frame_pipeline_timing = {"pre": [], "inf": [], "post": [], "n_detections": [], "track": {}, "count": {}}
#     for tr in selected_trackers:
#         per_frame_pipeline_timing["track"].setdefault(tr, {})
#         for s in STRIDES:
#             per_frame_pipeline_timing["track"][tr].setdefault(s, [])
#     for eid in exp_map.keys():
#         per_frame_pipeline_timing["count"].setdefault(eid, [])

#     # create tracker instances per (tracker,stride)
#     trackers = {}
#     track_profiles = {}
#     for tr in selected_trackers:
#         for s in STRIDES:
#             key = (tr, s)
#             trackers[key] = Tracker(tr)
#             track_profiles[key] = Profile() if tr in ("ocsort", "bytetrack") else Profile(device=device)

#     # open FrameGrabber and setup tqdm
#     source = str(dataset_path / sequence)
#     fg = FrameGrabber(source=source, stride=1)
#     if not fg.open():
#         raise RuntimeError(f"Failed to open source {source}")
#     if getattr(fg, "_grabber_mode", None) == "queue":
#         fg.start()

#     pre_profile = Profile(device=device)
#     inf_profile = Profile(device=device)
#     post_profile = Profile(device=device)

#     total_for_tqdm = int(total_frames_annot) if total_frames_annot is not None and total_frames_annot > 0 else None
#     frame_pbar = tqdm(total=total_for_tqdm, desc=f"Frames {sequence}", unit="frame")

#     frame_counter = 0
#     print(f"Starting diagnosed per-frame pipeline for {sequence} (total={total_for_tqdm})")
#     try:
#         while True:
#             frame = fg.get_frame(timeout=0.1)
#             if frame is None:
#                 break
#             idx = frame.read_idx

#             # detection
#             with pre_profile:
#                 pre = my_model.preprocess(frame.data)
#             with inf_profile:
#                 raw = my_model.infer(pre)
#             with post_profile:
#                 ready = my_model.postprocess(raw, pre, frame.data)

#             per_frame_pipeline_timing["pre"].append(float(pre_profile.dt))
#             per_frame_pipeline_timing["inf"].append(float(inf_profile.dt))
#             per_frame_pipeline_timing["post"].append(float(post_profile.dt))
#             per_frame_pipeline_timing["n_detections"].append(int(len(ready)))

#             # update trackers for which this frame is a sample (idx % s == 0)
#             current_tracks = {}
#             for tr in selected_trackers:
#                 for s in STRIDES:
#                     if (idx % s) != 0:
#                         continue
#                     key = (tr, s)
#                     tp = track_profiles[key]
#                     with tp:
#                         try:
#                             tracks = trackers[key].update(ready, frame.data)
#                         except Exception as e:
#                             print(f"[WARN] tracker {tr} s{s} failed at frame {idx}: {e}")
#                             tracks = []
#                     current_tracks[key] = tracks
#                     per_frame_pipeline_timing["track"][tr][s].append(float(tp.dt))

#             # counting
#             for eid, v in exp_map.items():
#                 cfg = v["cfg"]
#                 stride = cfg["stride"]
#                 tracker_name = cfg["tracker"]
#                 if (idx % stride) != 0:
#                     continue

#                 # proceed to counting
#                 count_obj = v["counter"]
#                 if tracker_name is None:
#                     count_input = ready
#                 else:
#                     key = (tracker_name, stride)
#                     count_input = current_tracks.get(key, [])

#                 cp = Profile()
#                 with cp:
#                     try:
#                         res = count_obj.count(count_input)
#                     except Exception as e:
#                         print(f"[WARN] counter {eid} failed at frame {idx}: {e}")
#                         res = None
#                 per_frame_pipeline_timing["count"][eid].append(float(cp.dt))

#                 if res is None:
#                     per_exp_frame_totals[eid].append(0)
#                     per_exp_frame_bycls[eid].append([])
#                     continue

#                 curr_total = int(getattr(res, "total_count", 0))
#                 prev_total = per_exp_prev_total[eid]
#                 per_exp_frame_totals[eid].append(int(curr_total - prev_total))
#                 per_exp_prev_total[eid] = curr_total

#                 curr_bycls = getattr(res, "counts_by_class", None)
#                 curr_bycls = np.asarray(curr_bycls, dtype=int) if curr_bycls is not None else np.zeros(0, dtype=int)
#                 prev_bycls = per_exp_prev_bycls[eid]
#                 max_len = max(prev_bycls.size, curr_bycls.size)
#                 if max_len == 0:
#                     per_exp_frame_bycls[eid].append([])
#                     per_exp_prev_bycls[eid] = np.zeros(0, dtype=int)
#                 else:
#                     prev_pad = np.pad(prev_bycls, (0, max_len - prev_bycls.size), mode='constant')
#                     curr_pad = np.pad(curr_bycls, (0, max_len - curr_bycls.size), mode='constant')
#                     per_exp_frame_bycls[eid].append((curr_pad - prev_pad).tolist())
#                     per_exp_prev_bycls[eid] = curr_pad.copy()

#                 try:
#                     del res
#                 except Exception:
#                     pass

#             # update progress and show concise postfix
#             frame_counter += 1
#             rss_gb = psutil.Process().memory_info().rss / 1e9
#             frame_pbar.set_postfix_str(f"seq={sequence} idx={idx} RAM={rss_gb:.2f}GB")
#             frame_pbar.update(1)

#             # periodic regulation
#             if (frame_counter % periodic_collect) == 0:
#                 gc.collect()
#             if (frame_counter % empty_cache_every) == 0:
#                 try:
#                     import torch
#                     torch.cuda.empty_cache()
#                 except Exception:
#                     pass

#     finally:
#         try:
#             fg.release()
#         except Exception:
#             pass
#         frame_pbar.close()

#     # finalize counts
#     for eid in exp_map.keys():
#         total_final = per_exp_prev_total[eid]
#         counts_by_class = per_exp_prev_bycls[eid].tolist() if isinstance(per_exp_prev_bycls[eid], np.ndarray) else []
#         counts_obj = {
#             "total_count": int(total_final),
#             "counts_by_class": counts_by_class,
#             "per_frame_counts": per_exp_frame_totals[eid],
#             "per_frame_counts_by_class": per_exp_frame_bycls[eid]
#         }
#         seq_entry["counts"][eid] = counts_obj

#     seq_entry["per_frame_pipeline_timing"] = per_frame_pipeline_timing
#     # update meta timestamp in storage (small file)
#     storage["meta"]["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

#     # cleanup
#     try:
#         for t in trackers.values():
#             del t
#     except Exception:
#         pass
#     try:
#         del exp_map
#     except Exception:
#         pass
#     gc.collect()

#     # write per-sequence result atomically and append index
#     write_sequence_result(sequences_dir, index_file, Path(sequence).stem, seq_entry)
#     # update meta.json atomically
#     try:
#         _atomic_write_json(storage["meta_file"], storage["meta"])
#     except Exception:
#         pass

#     print(f"Completed sequence {sequence} (diagnosed per-(tracker,stride) pipeline).")

# # -----------------------
# # Main
# # -----------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="Run per-frame diagnosed counting experiments on IMAROC_1 or IMAROC_2 dataset."
#     )
#     parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET, choices=["IMAROC_1", "IMAROC_2"],
#                         help="Dataset to run (IMAROC_1 or IMAROC_2) (default: IMAROC_1)")
#     parser.add_argument("--model", "-m", default="rtdetr-l.pt",
#                         help="Detector model path (default: rtdetr-l.pt). Example: --model yolov8n.pt")
#     parser.add_argument("--trackers", "-t", default="bytetrack,ocsort",
#                         help="Comma-separated trackers subset to use (default: bytetrack,ocsort). "
#                              f"Available: {', '.join(TRACKERS_AVAILABLE)}")
#     parser.add_argument("--counters", "-c", default=",".join(sorted(COUNTING_MAP_DEFAULT.keys())),
#                         help="Comma-separated counter families to run (default: all 6 counters). "
#                              f"Available: {', '.join(sorted(COUNTING_MAP_DEFAULT.keys()))}")
#     parser.add_argument("--frame-size", default=1000, type=int,
#                         help="Frame size passed to CountingConfigLoader (default 1000)")
#     args = parser.parse_args()

#     dataset_name = args.dataset

#     # compute dataset_path/results_dir/results_file from function (not CLI args)
#     dataset_path, results_dir, results_file = make_paths(dataset_name=dataset_name, model_name=args.model)
#     if not dataset_path.exists():
#         raise RuntimeError(f"Provided dataset path does not exist: {dataset_path}")

#     # init storage (meta, sequences dir, index)
#     storage = init_results_storage(results_dir, results_file)

#     # load annotation CSV (name is same in both: actual_counts.csv)
#     annotation_path = dataset_path / "actual_counts.csv"
#     if not annotation_path.exists():
#         raise RuntimeError(f"annotation file not found at {annotation_path}")
#     annotation_df = pd.read_csv(annotation_path)

#     # parse trackers and counters
#     requested_trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
#     selected_trackers = [t for t in requested_trackers if t in TRACKERS_AVAILABLE]
#     if not selected_trackers:
#         selected_trackers = ['bytetrack', 'ocsort']

#     requested_counters = [c.strip() for c in args.counters.split(",") if c.strip()]
#     counting_map = {k: v for k, v in COUNTING_MAP_DEFAULT.items() if k in requested_counters}
#     if not counting_map:
#         counting_map = COUNTING_MAP_DEFAULT.copy()

#     # initialize model and counter loader
#     my_model, device, pipeline_meta = init_detector(model_path=args.model)
#     global counter_load
#     counter_load = CountingConfigLoader(default_classes=COCO_VEHICLES, frame_size=args.frame_size)

#     # prepare meta (small) and write initial meta.json (kept small)
#     storage["meta"].update({
#         "tool": f"Evaluation of vision-based vehicle counting algorithms on {dataset_name} Video Dataset (diagnosed per-(tracker,stride) trackers)",
#         "dataset": dataset_name,
#         "model": str(args.model),
#         "version": "1.0",
#         "timestamp_utc": datetime.utcnow().isoformat() + "Z",
#         "fps": FPS,
#         "class_map": CLASS_MAP,
#         "counting_map": counting_map,
#         "pipeline": pipeline_meta,
#         "description": f"Dataset={dataset_name}; Model={args.model}; Selected trackers={selected_trackers}; Selected counters={list(counting_map.keys())}. Per-frame detection + per-(tracker,stride) tracker instances; memory regulation and diagnostics. Results stored per-sequence in {storage['sequences_dir']}",
#     })
#     _atomic_write_json(storage["meta_file"], storage["meta"])

#     sequences = get_sequences(dataset_path)
#     seq_pbar = tqdm(sequences, desc="Sequences", unit="seq")
#     for seq_idx, sequence in enumerate(seq_pbar):
#         seq_pbar.set_description(f"Seq: {sequence}")
#         process_sequence_per_stride_trackers_diagnose(sequence, my_model, device,
#                                                       dataset_name=dataset_name, dataset_path=dataset_path, annotation_df=annotation_df,
#                                                       counting_map=counting_map, selected_trackers=selected_trackers,
#                                                       counter_load_local=counter_load, storage=storage)
#         print(f"Sequence {sequence} is fully processed")

#     print(f"All experiments completed (diagnosed per-(tracker,stride) trackers) on {dataset_name}.")

# if __name__ == '__main__':
#     main()





# #!/usr/bin/env python3
# from itertools import product
# from pathlib import Path
# import json
# import logging
# from tqdm import tqdm
# from datetime import datetime
# import copy
# import gc
# import psutil
# import argparse
# from ast import literal_eval

# import numpy as np
# import pandas as pd

# # local packages (keep your project structure)
# from framegrabber.frame_grabber import FrameGrabber
# from detection.ultralytics_detectors import UltralyticsDetector
# from tracking.track import Tracker
# from counting.count_config_loader import CountingConfigLoader
# from utils.profilers import Profile

# # -----------------------
# # Defaults / constants
# # -----------------------
# logging.getLogger().setLevel(logging.WARNING)

# # base datasets root (fixed, not a CLI arg)
# BASE_DATASET_ROOT = Path(r"C:\Users\hamza\Datasets\TrafficDatasets")

# DEFAULT_DATASET = "IMAROC_1"

# # These will be set in main() via make_paths()
# RESULTS_DIR = None
# RESULTS_FILE = None
# FPS = 30

# STRIDES = [1, 2, 3, 4]
# TRACKLESS_VICINITIES = [0.01, 0.025, 0.05, 0.075, 0.1, 0.2]
# TRACK_VICINITIES = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
# TRACKERS_AVAILABLE = ['bytetrack', 'ocsort', 'strongsort', 'botsort', 'deepocsort', 'boosttrack', 'hybridsort']

# # Full superset counting map (six counters)
# COUNTING_MAP_DEFAULT = {
#     "counter_0": ["by_vicinity"],
#     "counter_1": ["by_id"],
#     "counter_2": ["by_cross_id"],
#     "counter_3": ["by_id", "by_vicinity"],
#     "counter_4": ["by_cross_id", "by_vicinity"],
#     "counter_5": ["by_cross_id", "by_id", "by_vicinity"],
# }

# # COCO-like vehicle class mapping kept as before
# COCO_VEHICLES = [1, 2, 3, 5, 7]
# CLASS_MAP = {1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# # global counter loader (will be initialized in main so frame_size can be configurable)
# counter_load = None

# # -----------------------
# # Path helpers
# # -----------------------

# def make_paths(dataset_name: str, model_name: str):
#     """Return (dataset_path, results_dir, results_file) based on dataset_name and model_name.

#     Rules (per your request):
#       - DATASET_PATH is fixed based on dataset (not an arg):
#           IMAROC_1 -> C:\\Users\\hamza\\Datasets\\TrafficDatasets\\IMAROC_1
#           IMAROC_2 -> C:\\Users\\hamza\\Datasets\\TrafficDatasets\\IMAROC_2
#       - RESULTS_DIR is counting/experiments/IMAROC1 or IMAROC2 accordingly (not an arg)
#       - RESULTS_FILE is RESULTS_DIR / "<model>_<dataset>.json" where <model> is the model stem
#     """
#     if dataset_name == "IMAROC_1":
#         dataset_path = BASE_DATASET_ROOT / "IMAROC_1"
#         results_dir = Path("counting") / "experiments" / "IMAROC1"
#     elif dataset_name == "IMAROC_2":
#         dataset_path = BASE_DATASET_ROOT / "IMAROC_2"
#         results_dir = Path("counting") / "experiments" / "IMAROC2"
#     else:
#         raise ValueError(f"Unknown dataset: {dataset_name}")

#     results_dir.mkdir(parents=True, exist_ok=True)
#     model_stem = Path(model_name).stem.replace(".", "_")
#     results_file = results_dir / f"{model_stem}_{dataset_name}.json"
#     return dataset_path, results_dir, results_file

# # -----------------------
# # Helpers (unchanged names where possible)
# # -----------------------

# def get_sequences(path: Path):
#     return sorted([p.name for p in path.glob("*.mp4")], key=lambda x: int(Path(x).stem.replace("kech", "")))


# def exp_id_for(seq, counter, stride, tracker=None, vicinity=None):
#     parts = [str(seq), counter, f"s{stride}"]
#     parts.append(f"tracker-{tracker}" if tracker is not None else "tracker-none")
#     parts.append(f"vic-{vicinity}" if vicinity is not None else "vic-none")
#     return "__".join(parts)


# def parse_exp_id(exp_id):
#     parts = exp_id.split("__")
#     return {
#         "sequence": parts[0],
#         "counter": parts[1],
#         "stride": int(parts[2].lstrip("s")),
#         "tracker": None if parts[3].split("tracker-")[1] == "none" else parts[3].split("tracker-")[1],
#         "vicinity": None if parts[4].split("vic-")[1] == "none" else float(parts[4].split("vic-")[1]),
#     }


# def _load_or_init_results(path: Path):
#     if path.exists():
#         try:
#             with path.open("r", encoding="utf-8") as fh:
#                 return json.load(fh)
#         except Exception:
#             backup = path.with_suffix(".broken.json")
#             path.rename(backup)
#             return {"meta": {}, "sequences": {}}
#     return {"meta": {}, "sequences": {}}


# def _atomic_write_json(path: Path, data):
#     tmp = path.with_suffix(".tmp.json")
#     with tmp.open("w", encoding="utf-8") as fh:
#         json.dump(data, fh, ensure_ascii=False, indent=2)
#     tmp.replace(path)

# # -----------------------
# # Detector / model init
# # -----------------------

# def init_detector(model_path: str = "rtdetr-l.pt", conf: float = 0.10):
#     model = UltralyticsDetector(model_path, conf=conf)
#     try:
#         device = model.predictor.device
#     except Exception:
#         device = getattr(model, "device", None)
#     pipeline_meta = {
#         "detector": {"name": str(model_path), "repo": "https://github.com/ultralytics/ultralytics", "input_size": "640x640", "conf_thresh": conf, "nms_iou": 0.7}
#     }
#     return model, device, pipeline_meta

# # -----------------------
# # Counters factory (full superset)
# # -----------------------

# def make_all_counter_instances(seq, counter_load_local, strides, trackless_vics, track_vics, all_trackers, line, counting_map):
#     instances = {c: {} for c in counting_map.keys()}

#     def _exp_id(counter, s, tracker, vic):
#         return exp_id_for(seq, counter, s, tracker, vic)

#     base = {"enable": True, "box_in_polygon_mode": "center_in_polygon", "line": line}

#     if "counter_0" in counting_map:
#         for s, v in product(strides, trackless_vics):
#             cid = _exp_id("counter_0", s, None, v)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_0"], "line_vicinity": v, "tracker": None})
#             instances["counter_0"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     if "counter_1" in counting_map:
#         for s, tr in product(strides, all_trackers):
#             cid = _exp_id("counter_1", s, tr, None)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_1"], "line_vicinity": None, "tracker": tr})
#             instances["counter_1"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     if "counter_2" in counting_map:
#         for s, tr in product(strides, all_trackers):
#             cid = _exp_id("counter_2", s, tr, None)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map["counter_2"], "line_vicinity": None, "tracker": tr})
#             instances["counter_2"][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     for cname in ("counter_3", "counter_4", "counter_5"):
#         if cname not in counting_map:
#             continue
#         for s, tr, v in product(strides, all_trackers, track_vics):
#             cid = _exp_id(cname, s, tr, v)
#             cfg = copy.deepcopy(base)
#             cfg.update({"name": cid, "counting_logic": counting_map[cname], "line_vicinity": v, "tracker": tr})
#             instances[cname][cid] = counter_load_local._normalize_global(copy.deepcopy(cfg))

#     return instances

# # -----------------------
# # Video / annotation helpers (dataset aware)
# # -----------------------

# def get_video_characteristics(sequence: str, dataset_path: Path):
#     source = str(dataset_path / sequence)
#     fg = FrameGrabber(source=source, stride=1)
#     if not fg.open():
#         raise RuntimeError(f"Failed to open source {source} for probing")
#     try:
#         fps = getattr(fg, "fps", None) or getattr(fg, "_fps", None)
#         frame = fg.get_frame(timeout=0.5)
#         if frame is None:
#             raise RuntimeError(f"No frames available in {source}")
#         h, w = frame.data.shape[:2]
#         if fps is None:
#             try:
#                 cap = getattr(fg, "_cap", None)
#                 if cap is not None:
#                     fps_val = cap.get(5)  # CV_CAP_PROP_FPS == 5
#                     fps = fps_val if fps_val and fps_val > 0 else FPS
#                 else:
#                     fps = FPS
#             except Exception:
#                 fps = FPS
#     finally:
#         fg.release()

#     return int(w), int(h), float(fps)


# def _line_pixels_imaroc1(raw_str: str, width: int, height: int):
#     raw = literal_eval(raw_str)
#     (x1f, y1f), (x2f, y2f) = raw
#     x1 = int(round(float(x1f) * width))
#     y1 = int(round(float(y1f) * height))
#     x2 = int(round(float(x2f) * width))
#     y2 = int(round(float(y2f) * height))
#     return ((x1, y1), (x2, y2))


# def _line_pixels_imaroc2(raw_str: str):
#     raw = literal_eval(raw_str)
#     (x1, y1), (x2, y2) = raw
#     return ((int(x1), int(y1)), (int(x2), int(y2)))


# def get_line_video(dataset_name: str, video_name: str, annotation_df: pd.DataFrame, width: int, height: int):
#     if dataset_name == "IMAROC_1":
#         row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower(), "line_of_counting"]
#         if row.empty:
#             raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_1)")
#         return _line_pixels_imaroc1(row.iloc[0], width, height)
#     else:
#         row = annotation_df.loc[annotation_df["video_name"] == video_name, "line_of_counting"]
#         if row.empty:
#             raise ValueError(f"No line_of_counting found for '{video_name}' (IMAROC_2)")
#         return _line_pixels_imaroc2(row.iloc[0])


# def _extract_counts_only(annotated_dict: dict, dataset_name: str):
#     if not annotated_dict:
#         return {}
#     counts_keys = [k for k in annotated_dict.keys() if k.startswith("actual_")]
#     counts = {}
#     for k in counts_keys:
#         try:
#             counts[k] = int(annotated_dict.get(k, 0))
#         except Exception:
#             counts[k] = annotated_dict.get(k)
#     return counts


# def get_annotated_counts(dataset_name: str, video_name: str, annotation_df: pd.DataFrame):
#     if dataset_name == "IMAROC_1":
#         row = annotation_df.loc[annotation_df["video_name"].str.lower() == str(video_name).lower()]
#         if row.empty:
#             return None
#         r = row.iloc[0]
#         annotated = {
#             "actual_total_vehicles": int(r.get("actual_total_vehicles", 0)),
#             "actual_car_counts": int(r.get("actual_car_counts", 0)),
#             "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
#             "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
#             "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
#             "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
#             "camera_perspective": r.get("camera_perspective", None),
#             "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
#         }
#         return annotated
#     else:
#         row = annotation_df.loc[annotation_df["video_name"] == video_name]
#         if row.empty:
#             return None
#         r = row.iloc[0]
#         annotated = {
#             "recording_time": r.get("recording_time", None),
#             "annotated_width": int(r.get("width", 0)) if pd.notna(r.get("width", None)) else None,
#             "annotated_height": int(r.get("height", 0)) if pd.notna(r.get("height", None)) else None,
#             "direction_relative_to_camera": r.get("direction_relative_to_camera", None),
#             "road_directionality": r.get("road_directionality", None),
#             "lighting": r.get("lighting", None),
#             "actual_car_counts": int(r.get("actual_car_counts", 0)),
#             "actual_bus_counts": int(r.get("actual_bus_counts", 0)),
#             "actual_truck_counts": int(r.get("actual_truck_counts", 0)),
#             "actual_van_counts": int(r.get("actual_van_counts", 0)),
#             "actual_motorcycle_counts": int(r.get("actual_motorcycle_counts", 0)),
#             "actual_bicycle_counts": int(r.get("actual_bicycle_counts", 0)),
#             "total_frames": int(r.get("total_frames", 0)) if pd.notna(r.get("total_frames", None)) else None,
#         }
#         return annotated

# # -----------------------
# # Per-frame pipeline (unchanged logic) - dataset aware wrappers used by main loop
# # -----------------------

# def process_sequence_per_stride_trackers_diagnose(sequence, all_results, my_model, device,
#                                                    dataset_name, dataset_path, annotation_df,
#                                                    periodic_collect=200, empty_cache_every=200,
#                                                    counting_map=None, selected_trackers=None,
#                                                    counter_load_local=None, results_file=None):
#     seq_key = str(sequence)
#     if seq_key in all_results.get("sequences", {}):
#         return

#     seq_entry = all_results["sequences"].setdefault(seq_key, {})
#     seq_entry.setdefault("counts", {})
#     seq_entry.setdefault("per_frame_pipeline_timing", {})
#     seq_entry.setdefault("video", {})

#     # probe video characteristics
#     width, height, fps_vid = get_video_characteristics(sequence, dataset_path)

#     # dataset-specific line and annotated counts
#     counting_line = get_line_video(dataset_name, Path(sequence).name, annotation_df, width=width, height=height)
#     annotated_full = get_annotated_counts(dataset_name, Path(sequence).name, annotation_df)

#     total_frames_annot = annotated_full.get("total_frames") if annotated_full is not None else None

#     # build video metadata; separate annotated_counts (only counts) from video_metadata (other fields)
#     annotated_counts = _extract_counts_only(annotated_full, dataset_name) if annotated_full is not None else {}

#     video_metadata = {"width": int(width), "height": int(height), "fps": float(fps_vid), "line_of_counting": counting_line}
#     # add dataset-provided extra metadata (non-counts)
#     if annotated_full is not None:
#         for k, v in annotated_full.items():
#             if not k.startswith("actual_") and k != "total_frames":
#                 video_metadata[k] = v

#     video_meta = {
#         "annotated_counts": annotated_counts,
#         "video_metadata": video_metadata,
#         "n_frames": int(total_frames_annot) if total_frames_annot is not None else None,
#     }

#     seq_entry["video"] = video_meta

#     # prepare counters & exp_map (use provided counting_map and selected_trackers)
#     counter_instances = make_all_counter_instances(
#         Path(sequence).stem,
#         counter_load_local,
#         strides=STRIDES,
#         trackless_vics=TRACKLESS_VICINITIES,
#         track_vics=TRACK_VICINITIES,
#         all_trackers=selected_trackers,
#         line=counting_line,
#         counting_map=counting_map,
#     )

#     exp_map = {}
#     for fam, exps in counter_instances.items():
#         for eid, counter in exps.items():
#             cfg = parse_exp_id(eid)
#             exp_map[eid] = {"counter": counter, "cfg": cfg}

#     # per-exp state
#     per_exp_prev_total = {}
#     per_exp_prev_bycls = {}
#     per_exp_frame_totals = {}
#     per_exp_frame_bycls = {}
#     for eid, v in exp_map.items():
#         prev_total = int(getattr(v["counter"].count_result, "total_count", 0))
#         arr = getattr(v["counter"].count_result, "counts_by_class", None)
#         prev_bycls = np.asarray(arr, dtype=int) if arr is not None else np.zeros(0, dtype=int)
#         per_exp_prev_total[eid] = prev_total
#         per_exp_prev_bycls[eid] = prev_bycls.copy()
#         per_exp_frame_totals[eid] = []
#         per_exp_frame_bycls[eid] = []

#     # timing structures
#     per_frame_pipeline_timing = {"pre": [], "inf": [], "post": [], "n_detections": [], "track": {}, "count": {}}
#     for tr in selected_trackers:
#         per_frame_pipeline_timing["track"].setdefault(tr, {})
#         for s in STRIDES:
#             per_frame_pipeline_timing["track"][tr].setdefault(s, [])
#     for eid in exp_map.keys():
#         per_frame_pipeline_timing["count"].setdefault(eid, [])

#     # create tracker instances per (tracker,stride)
#     trackers = {}
#     track_profiles = {}
#     for tr in selected_trackers:
#         for s in STRIDES:
#             key = (tr, s)
#             trackers[key] = Tracker(tr)
#             track_profiles[key] = Profile() if tr in ("ocsort", "bytetrack") else Profile(device=device)

#     # open FrameGrabber and setup tqdm
#     source = str(dataset_path / sequence)
#     fg = FrameGrabber(source=source, stride=1)
#     if not fg.open():
#         raise RuntimeError(f"Failed to open source {source}")
#     if getattr(fg, "_grabber_mode", None) == "queue":
#         fg.start()

#     pre_profile = Profile(device=device)
#     inf_profile = Profile(device=device)
#     post_profile = Profile(device=device)

#     total_for_tqdm = int(total_frames_annot) if total_frames_annot is not None and total_frames_annot > 0 else None
#     frame_pbar = tqdm(total=total_for_tqdm, desc=f"Frames {sequence}", unit="frame")

#     frame_counter = 0
#     print(f"Starting diagnosed per-frame pipeline for {sequence} (total={total_for_tqdm})")
#     try:
#         while True:
#             frame = fg.get_frame(timeout=0.1)
#             if frame is None:
#                 break
#             idx = frame.read_idx

#             # detection
#             with pre_profile:
#                 pre = my_model.preprocess(frame.data)
#             with inf_profile:
#                 raw = my_model.infer(pre)
#             with post_profile:
#                 ready = my_model.postprocess(raw, pre, frame.data)

#             per_frame_pipeline_timing["pre"].append(float(pre_profile.dt))
#             per_frame_pipeline_timing["inf"].append(float(inf_profile.dt))
#             per_frame_pipeline_timing["post"].append(float(post_profile.dt))
#             per_frame_pipeline_timing["n_detections"].append(int(len(ready)))

#             # update trackers for which this frame is a sample (idx % s == 0)
#             current_tracks = {}
#             for tr in selected_trackers:
#                 for s in STRIDES:
#                     if (idx % s) != 0:
#                         continue
#                     key = (tr, s)
#                     tp = track_profiles[key]
#                     with tp:
#                         try:
#                             tracks = trackers[key].update(ready, frame.data)
#                         except Exception as e:
#                             print(f"[WARN] tracker {tr} s{s} failed at frame {idx}: {e}")
#                             tracks = []
#                     current_tracks[key] = tracks
#                     per_frame_pipeline_timing["track"][tr][s].append(float(tp.dt))

#             # counting
#             for eid, v in exp_map.items():
#                 cfg = v["cfg"]
#                 stride = cfg["stride"]
#                 tracker_name = cfg["tracker"]
#                 if (idx % stride) != 0:
#                     continue

#                                     # proceed to counting
#                 count_obj = v["counter"]
#                 if tracker_name is None:
#                     count_input = ready
#                 else:
#                     key = (tracker_name, stride)
#                     count_input = current_tracks.get(key, [])

#                 cp = Profile()
#                 with cp:
#                     try:
#                         res = count_obj.count(count_input)
#                     except Exception as e:
#                         print(f"[WARN] counter {eid} failed at frame {idx}: {e}")
#                         res = None
#                 per_frame_pipeline_timing["count"][eid].append(float(cp.dt))

#                 if res is None:
#                     per_exp_frame_totals[eid].append(0)
#                     per_exp_frame_bycls[eid].append([])
#                     continue

#                 curr_total = int(getattr(res, "total_count", 0))
#                 prev_total = per_exp_prev_total[eid]
#                 per_exp_frame_totals[eid].append(int(curr_total - prev_total))
#                 per_exp_prev_total[eid] = curr_total

#                 curr_bycls = getattr(res, "counts_by_class", None)
#                 curr_bycls = np.asarray(curr_bycls, dtype=int) if curr_bycls is not None else np.zeros(0, dtype=int)
#                 prev_bycls = per_exp_prev_bycls[eid]
#                 max_len = max(prev_bycls.size, curr_bycls.size)
#                 if max_len == 0:
#                     per_exp_frame_bycls[eid].append([])
#                     per_exp_prev_bycls[eid] = np.zeros(0, dtype=int)
#                 else:
#                     prev_pad = np.pad(prev_bycls, (0, max_len - prev_bycls.size), mode='constant')
#                     curr_pad = np.pad(curr_bycls, (0, max_len - curr_bycls.size), mode='constant')
#                     per_exp_frame_bycls[eid].append((curr_pad - prev_pad).tolist())
#                     per_exp_prev_bycls[eid] = curr_pad.copy()

#                 try:
#                     del res
#                 except Exception:
#                     pass

#             # update progress and show concise postfix
#             frame_counter += 1
#             rss_gb = psutil.Process().memory_info().rss / 1e9
#             frame_pbar.set_postfix_str(f"seq={sequence} idx={idx} RAM={rss_gb:.2f}GB")
#             frame_pbar.update(1)

#             # periodic regulation
#             if (frame_counter % periodic_collect) == 0:
#                 gc.collect()
#             if (frame_counter % empty_cache_every) == 0:
#                 try:
#                     import torch
#                     torch.cuda.empty_cache()
#                 except Exception:
#                     pass

#     finally:
#         try:
#             fg.release()
#         except Exception:
#             pass
#         frame_pbar.close()

#     # finalize counts
#     for eid in exp_map.keys():
#         total_final = per_exp_prev_total[eid]
#         counts_by_class = per_exp_prev_bycls[eid].tolist() if isinstance(per_exp_prev_bycls[eid], np.ndarray) else []
#         counts_obj = {
#             "total_count": int(total_final),
#             "counts_by_class": counts_by_class,
#             "per_frame_counts": per_exp_frame_totals[eid],
#             "per_frame_counts_by_class": per_exp_frame_bycls[eid]
#         }
#         seq_entry["counts"][eid] = counts_obj

#     seq_entry["per_frame_pipeline_timing"] = per_frame_pipeline_timing
#     if all_results.get("meta") is None:
#         all_results["meta"] = {}
#     all_results["meta"]["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

#     # cleanup
#     try:
#         for t in trackers.values():
#             del t
#     except Exception:
#         pass
#     try:
#         del exp_map
#     except Exception:
#         pass
#     gc.collect()

#     # write results
#     if results_file is None:
#         raise RuntimeError("results_file must be provided to write results")
#     _atomic_write_json(results_file, all_results)
#     print(f"Completed sequence {sequence} (diagnosed per-(tracker,stride) pipeline).")

# # -----------------------
# # Main
# # -----------------------

# def main():
#     parser = argparse.ArgumentParser(
#         description="Run per-frame diagnosed counting experiments on IMAROC_1 or IMAROC_2 dataset."
#     )
#     parser.add_argument("--dataset", "-d", default=DEFAULT_DATASET, choices=["IMAROC_1", "IMAROC_2"],
#                         help="Dataset to run (IMAROC_1 or IMAROC_2) (default: IMAROC_1)")
#     parser.add_argument("--model", "-m", default="rtdetr-l.pt",
#                         help="Detector model path (default: rtdetr-l.pt). Example: --model yolov8n.pt")
#     parser.add_argument("--trackers", "-t", default="bytetrack,ocsort",
#                         help="Comma-separated trackers subset to use (default: bytetrack,ocsort). "
#                              f"Available: {', '.join(TRACKERS_AVAILABLE)}")
#     parser.add_argument("--counters", "-c", default=",".join(sorted(COUNTING_MAP_DEFAULT.keys())),
#                         help="Comma-separated counter families to run (default: all 6 counters). "
#                              f"Available: {', '.join(sorted(COUNTING_MAP_DEFAULT.keys()))}")
#     parser.add_argument("--frame-size", default=1000, type=int,
#                         help="Frame size passed to CountingConfigLoader (default 1000)")
#     args = parser.parse_args()

#     dataset_name = args.dataset

#     # compute dataset_path/results_dir/results_file from function (not CLI args)
#     dataset_path, results_dir, results_file = make_paths(dataset_name=dataset_name, model_name=args.model)
#     if not dataset_path.exists():
#         raise RuntimeError(f"Provided dataset path does not exist: {dataset_path}")

#     # load annotation CSV (name is same in both: actual_counts.csv)
#     annotation_path = dataset_path / "actual_counts.csv"
#     if not annotation_path.exists():
#         raise RuntimeError(f"annotation file not found at {annotation_path}")
#     annotation_df = pd.read_csv(annotation_path)

#     # parse trackers and counters
#     requested_trackers = [t.strip() for t in args.trackers.split(",") if t.strip()]
#     selected_trackers = [t for t in requested_trackers if t in TRACKERS_AVAILABLE]
#     if not selected_trackers:
#         selected_trackers = ['bytetrack', 'ocsort']

#     requested_counters = [c.strip() for c in args.counters.split(",") if c.strip()]
#     counting_map = {k: v for k, v in COUNTING_MAP_DEFAULT.items() if k in requested_counters}
#     if not counting_map:
#         counting_map = COUNTING_MAP_DEFAULT.copy()

#     # initialize model and counter loader
#     my_model, device, pipeline_meta = init_detector(model_path=args.model)
#     global counter_load
#     counter_load = CountingConfigLoader(default_classes=COCO_VEHICLES, frame_size=args.frame_size)

#     # prepare results structure and write initial meta
#     all_results = _load_or_init_results(results_file)
#     all_results["meta"] = {
#         "tool": f"Evaluation of vision-based vehicle counting algorithms on {dataset_name} Video Dataset (diagnosed per-(tracker,stride) trackers)",
#         "dataset": dataset_name,
#         "model": str(args.model),
#         "version": "1.0",
#         "timestamp_utc": datetime.utcnow().isoformat() + "Z",
#         "fps": FPS,
#         "class_map": CLASS_MAP,
#         "counting_map": counting_map,
#         "pipeline": pipeline_meta,
#         "description": f"Dataset={dataset_name}; Model={args.model}; Selected trackers={selected_trackers}; Selected counters={list(counting_map.keys())}. Per-frame detection + per-(tracker,stride) tracker instances; memory regulation and diagnostics. Results file: {results_file}",
#     }
#     all_results.setdefault("sequences", {})

#     sequences = get_sequences(dataset_path)
#     seq_pbar = tqdm(sequences, desc="Sequences", unit="seq")
#     for seq_idx, sequence in enumerate(seq_pbar):
#         seq_pbar.set_description(f"Seq: {sequence}")
#         process_sequence_per_stride_trackers_diagnose(sequence, all_results, my_model, device,
#                                                       dataset_name=dataset_name, dataset_path=dataset_path, annotation_df=annotation_df,
#                                                       counting_map=counting_map, selected_trackers=selected_trackers,
#                                                       counter_load_local=counter_load, results_file=results_file)
#         print(f"Sequence {sequence} is fully processed")

#     print(f"All experiments completed (diagnosed per-(tracker,stride) trackers) on {dataset_name}.")


# if __name__ == '__main__':
#     main()
