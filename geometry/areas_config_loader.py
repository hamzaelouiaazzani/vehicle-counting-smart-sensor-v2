# --- Standard library ---
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# --- Third-party ---
import numpy as np
import yaml

# --- Local modules ---
from utils.helper_functions import (
    polygon_to_mask,
    bbox_center_in_polygon,
    bbox_corners_in_polygon,
    bbox_any_in_polygon,
)
from geometry.areas import AreaBase, GlobalArea, ROI

logger = logging.getLogger(__name__)



# --- Config loader that builds ROI and GlobalArea instances ---
class ConfigLoader:
    def __init__(self,
                 default_config_path: Path | str = "configs/counter_config.yaml",
                 default_polygon_mode: str = "intersects_polygon",
                 default_line_vicinity: float = 0.20,
                 frame_size: int = 1000):
        self.config_path = Path(default_config_path)
        self.default_polygon_mode = default_polygon_mode
        self.default_line_vicinity = float(default_line_vicinity)
        self.frame_size = int(frame_size)

    # public API
    def load(self) -> Tuple[List[ROI], Optional[GlobalArea]]:
        cfg = self._read_yaml(self.config_path)
        rois_raw = cfg.get("rois") or []
        rois: List[ROI] = []
        for i, raw in enumerate(rois_raw):
            roi = self._normalize_roi(raw, i)
            if roi is not None:
                rois.append(roi)

        global_obj: Optional[GlobalArea] = None
        raw_global = cfg.get("global")
        if raw_global:
            global_obj = self._normalize_global(raw_global)

        return rois, global_obj



    def load_area_info(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Load ROIs and global area, but return their area info dicts
        instead of the full objects.
        """
        rois, global_obj = self.load()

        rois_info = [roi.get_area_info() for roi in rois]
        global_info = global_obj.get_area_info() if global_obj else None

        return rois_info, global_info

    
    # helpers
    def _read_yaml(self, path: Path) -> dict:
        if not path.exists():
            logger.info("Config %s not found — loading empty config.", path)
            return {}
        try:
            with open(path, "r") as fh:
                return yaml.safe_load(fh) or {}
        except Exception as e:
            raise RuntimeError(f"Failed to read config '{path}': {e}")

    def _validate_polygon(self, polygon_raw: Any, name: str) -> Optional[np.ndarray]:
        if polygon_raw in (None, "null"):
            return None
        try:
            arr = np.asarray(polygon_raw, dtype=float)
            if arr.ndim != 2 or arr.shape[1] != 2:
                raise ValueError("polygon must be shape (N,2)")
            return arr
        except Exception as ex:
            logger.warning("%s: invalid polygon provided (%s) — ignoring polygon: %s", name, polygon_raw, ex)
            return None

    def _validate_line(self, line_raw: Any) -> Optional[Tuple[Tuple[float,float], Tuple[float,float]]]:
        if line_raw in (None, "null"):
            return None
        try:
            p0 = tuple(map(float, line_raw[0]))
            p1 = tuple(map(float, line_raw[1]))
            return (p0, p1)
        except Exception:
            logger.warning("Invalid line provided: %s — ignoring", line_raw)
            return None

    def _compute_mask_safe(self, polygon_arr: np.ndarray, name: str):
        if polygon_arr is None:
            return None, None, None
        try:
            mask, x_min, y_min = polygon_to_mask(polygon_arr)
            return mask, x_min, y_min
        except Exception as ex:
            logger.warning("%s: failed to rasterize polygon -> ignoring polygon: %s", name, ex)
            return None, None, None

    def _select_polygon_filter(self, box_mode: Optional[str]) -> ROI.FilterFn:
        mode = (box_mode or "").lower()
        if mode in ("center", "center_in_polygon"):
            return bbox_center_in_polygon
        if mode in ("corners", "corners_in_polygon"):
            return bbox_corners_in_polygon
        if mode in ("any", "any_in_polygon"):
            return bbox_any_in_polygon
        logger.info(f"Unrecognized polygon filter mode {box_mode} — falling back to default bbox_any_in_polygon")
        return bbox_any_in_polygon
    

    # normalizers -> construct domain objects
    def _normalize_roi(self, raw: Dict[str, Any], idx: int) -> Optional[ROI]:
        name = raw.get("name") or f"ROI_{idx}"
        enabled = raw.get("enable") or False
        if enabled is not True:
            return None

        polygon_raw = raw.get("polygon")
        if polygon_raw in (None, "null"):
            return None  # skip

        polygon_arr = self._validate_polygon(polygon_raw, name)
        if polygon_arr is None:
            raise ValueError(f"{name}: invalid polygon format, expected list of [x,y] pairs.")

        box_mode = raw.get("box_in_polygon_mode") or self.default_polygon_mode
        filter_fn = self._select_polygon_filter(box_mode)

        mask, mask_x_min, mask_y_min = self._compute_mask_safe(polygon_arr, name)

        line = self._validate_line(raw.get("line"))
        if line is None:
            line_vicinity = None
        else:
            lv = raw.get("line_vicinity")
            if not isinstance(lv, (int, float)) or not (0 <= float(lv) <= 1):
                logger.info("Invalid or missing 'line_vicinity' for %s — using default %.2f", name, self.default_line_vicinity)
                line_vicinity = self.default_line_vicinity
            else:
                line_vicinity = float(lv)

        return ROI(
            name=name,
            enabled=True,
            polygon=polygon_arr,
            filter_fn=filter_fn,
            mask=mask,
            mask_x_min=mask_x_min,
            mask_y_min=mask_y_min,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=self.frame_size
        )

    def _normalize_global(self, raw_global: Dict[str, Any]) -> Optional[GlobalArea]:
        name = "global"
        enabled = raw_global.get("enable") or False
        if enabled is not True:
            return None

        line = self._validate_line(raw_global.get("line"))
        if line is None:
            line_vicinity = None
        else:
            lv = raw_global.get("line_vicinity")
            if not isinstance(lv, (int, float)) or not (0 <= float(lv) <= 1):
                logger.info("Invalid or missing 'line_vicinity' for %s — using default %.2f", name, self.default_line_vicinity)
                line_vicinity = self.default_line_vicinity
            else:
                line_vicinity = float(lv)

        return GlobalArea(enabled=enabled, line=line, line_vicinity=line_vicinity, frame_size=self.frame_size)