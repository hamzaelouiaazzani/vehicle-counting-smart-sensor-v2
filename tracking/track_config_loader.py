# --- Standard library ---
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

# --- Third-party ---
import numpy as np
import yaml

# --- Local imports (adjust paths as needed) ---
from geometry.areas_config_loader import ConfigLoader
from tracking.track import ROITracker, GlobalTracker  

logger = logging.getLogger(__name__)


class TrackerConfigLoader(ConfigLoader):
    """
    Extends ConfigLoader to produce tracker-enabled area objects (ROITracker, GlobalTracker).
    - Normalizes and validates tracker configuration.
    - Ensures reid_model is disabled for pure motion-based trackers.
    """

    _ALLOWED_TRACKERS = [
        "bytetrack",
        "ocsort",
        "strongsort",
        "botsort",
        "deepocsort",
        "hybridsort",
        "boosttrack",
    ]

    def __init__(
        self,
        default_config_path: Path | str = "configs/tracker_config.yaml",
        default_polygon_mode: str = "intersects_polygon",
        default_line_vicinity: float = 0.20,
        frame_size: int = 1000,
        default_tracker: str = "ocsort",
        default_reid_model: Optional[str] = "osnet_x0_25_market1501.pt",
        classes = None
        
    ):
        super().__init__(
            default_config_path=default_config_path,
            default_polygon_mode=default_polygon_mode,
            default_line_vicinity=default_line_vicinity,
            frame_size=frame_size,
        )
        self.default_tracker = default_tracker
        self.default_reid_model = default_reid_model
        self.target_classes = classes

            
    # --- normalize ROI to ROITracker ---
    def _normalize_roi(self, raw: Dict[str, Any], idx: int):
        name = raw.get("name") or f"ROI_{idx}"
        enabled = raw.get("enable") or False
        if not enabled:
            return None

        polygon_raw = raw.get("polygon")
        if polygon_raw in (None, "null"):
            return None
        polygon_arr = self._validate_polygon(polygon_raw, name)
        if polygon_arr is None:
            raise ValueError(f"{name}: invalid polygon format, expected list of [x,y] pairs.")

        # polygon mode → filter function
        box_mode = raw.get("box_in_polygon_mode") or self.default_polygon_mode
        filter_fn = self._select_polygon_filter(box_mode)
        mask, mask_x_min, mask_y_min = self._compute_mask_safe(polygon_arr, name)

        # line vicinity
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

        # tracker config
        tracker_cfg = raw.get("tracker", {}) or {}
        method = tracker_cfg.get("method", self.default_tracker)
        reid_model = tracker_cfg.get("reid_model", self.default_reid_model)

        if method not in self._ALLOWED_TRACKERS:
            logger.warning(
                "Tracker method %r not recognized for %s. Using default %r.",
                method, name, self.default_tracker
            )
            method = self.default_tracker

        # motion-only trackers => disable reid
        if method in ("bytetrack", "ocsort"):
            reid_model = ""

            
        return ROITracker(
            name=name,
            enabled=True,
            polygon=polygon_arr,
            filter_fn=filter_fn,
            tracker_method=method,
            reid_model=reid_model,
            classes=self.target_classes,
            mask=mask,
            mask_x_min=mask_x_min,
            mask_y_min=mask_y_min,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=self.frame_size,
        )

    # --- normalize global to GlobalTracker ---
    def _normalize_global(self, raw_global: Dict[str, Any]):
        name = "global"
        enabled = raw_global.get("enable") or False
        if not enabled:
            return None

        # line vicinity
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

        # tracker config
        tracker_cfg = raw_global.get("tracker", {}) or {}
        method = tracker_cfg.get("method", self.default_tracker)
        reid_model = tracker_cfg.get("reid_model", self.default_reid_model)

        if method not in self._ALLOWED_TRACKERS:
            logger.warning(
                "Tracker method %r not recognized for %s. Using default %r.",
                method, name, self.default_tracker
            )
            method = self.default_tracker

        if method in ("bytetrack", "ocsort"):
            reid_model = ""
        
        return GlobalTracker(
            enabled=True,
            tracker_method=method,
            reid_model=reid_model,
            classes=self.target_classes,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=self.frame_size,
        )

    # --- public API ---
    def load_tracker_areas(self) -> Tuple[List[Any], Optional[Any]]:
        """Return (roi_trackers, global_tracker)"""
        return self.load()
