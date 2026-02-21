from typing import Optional, Tuple, List, Any
from pathlib import Path
from boxmot.tracker_zoo import create_tracker, get_tracker_config
from boxmot.trackers.bytetrack.basetrack import BaseTrack
from geometry.areas import AreaBase, GlobalArea, ROI
import numpy as np


import logging
logger = logging.getLogger(__name__)


# --- Base Tracker ---
class Tracker:
    def __init__(self,
                 method: str = "ocsort",
                 reid_model: str = "osnet_x0_25_market1501.pt",
                 classes: Optional[List[int]] = None,  # user-facing param name
                 device: str = "",
                 half: bool = False,
                 per_class: bool = True):
        """
        Base Tracker.

        Args:
            method: tracker method name
            reid_model: path to reid model
            classes: list of class ids to keep (None => keep all)
            device, half, per_class: forwarded to create_tracker
        """
        BaseTrack.clear_count()
        self.method = method
        self.config_file = get_tracker_config(method)
        self.reid_model = Path(reid_model)
        self.device = device
        self.half = half
        self.per_class = per_class

        # accept lists/tuples/sets of ints (including numpy int types); otherwise skip class filtering
        if not isinstance(classes, (list, tuple, set)) \
           or not all(isinstance(c, (int, np.integer)) for c in classes):
            logger.info(
                "`classes` parameter is not a list/tuple/set of integers; "
                "disabling class filtering and using default (no class filtering)."
            )
            self.target_classes = None
        else:
            self.target_classes = [int(c) for c in classes]  # e.g. [0,4,6] or None for all

        # create underlying tracker implementation
        self.tracker = create_tracker(
            self.method,
            self.config_file,
            self.reid_model,
            self.device,
            self.half,
            self.per_class
        )

    # --- default geometric filter (no-op) ---
    def filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Default geometric filter: return all-True mask so that base Tracker
        applies only class filtering unless subclass overrides this.
        Returns:
            mask: np.ndarray of bool with shape (N,)
        """
        detections = np.asarray(detections)
        n = detections.shape[0] if detections.ndim > 0 else 0
        return np.ones(n, dtype=bool)

    # --- centralized class filter ---
    def filter_by_classes(self, detections: np.ndarray) -> np.ndarray:
        """
        Filter detections by class IDs.

        Expected detections shape: (N, 6) -> (x1, y1, x2, y2, conf, cls)

        Returns:
            mask: boolean ndarray (N,) where True means keep that detection.
        """
        detections = np.asarray(detections)
        if detections.size == 0:
            return np.zeros(0, dtype=bool)

        # number of detections
        n = detections.shape[0]

        if self.target_classes is None:
            return np.ones(n, dtype=bool)

        # ensure enough columns to read class id
        if detections.ndim == 1 or detections.shape[1] <= 5:
            logger.warning("filter_by_classes(): detections array has unexpected shape %s; denying all.", detections.shape)
            return np.zeros(n, dtype=bool)

        # get class column (6th column index 5)
        cls_col = detections[:, 5].astype(int)
        mask = np.isin(cls_col, self.target_classes)
        return mask.reshape(-1)

    # --- main update: normalize, apply class + geometric filters, then forward to tracker ---
    def update(self, detections: np.ndarray, frame_data: Any):
        """
        Update tracker with filtered detections.

        Steps:
          * Normalize detector output into (N,6) array: (x1,y1,x2,y2,conf,cls)
          * Apply class filter (centralized)
          * Apply geometric filter (self.filter_detections; children may override)
          * Logical AND of both masks
          * Pass filtered detections to underlying tracker

        Returns:
            result of self.tracker.update(filtered_detections, frame_data)
        """
        if self.tracker is None:
            raise RuntimeError("Tracker not initialized.")

        # --- normalize input into ndarray ---
        dets = np.asarray(detections) if detections is not None else np.empty((0, 6))
        if dets.size == 0:
            dets = np.empty((0, 6), dtype=float)
        else:
            # 1D single-row -> reshape
            if dets.ndim == 1:
                # defensive: if exactly 6 elements, reshape; otherwise attempt trim/pad
                if dets.shape[0] == 6:
                    dets = dets.reshape(1, 6)
                elif dets.shape[0] > 6:
                    logger.warning("Single-row detections has %d elements; trimming to first 6.", dets.shape[0])
                    dets = dets[:6].reshape(1, 6)
                else:
                    logger.warning("Single-row detections has too few elements (%d); passing empty to tracker.", dets.shape[0])
                    dets = np.empty((0, 6), dtype=float)
            else:
                # multiple rows: ensure columns >= 6
                if dets.shape[1] < 6:
                    logger.warning("Detections have %d columns (<6); passing empty to tracker.", dets.shape[1])
                    dets = np.empty((0, 6), dtype=float)
                elif dets.shape[1] > 6:
                    logger.warning("Detections have %d columns (>6); trimming to first 6 for tracker.", dets.shape[1])
                    dets = dets[:, :6]

        # ensure numeric dtype
        try:
            dets = dets.astype(float)
        except Exception:
            logger.exception("Failed to cast detections to float -- passing empty to tracker.")
            dets = np.empty((0, 6), dtype=float)

        n = dets.shape[0]

        # --- class mask ---
        mask_cls = self.filter_by_classes(dets)

        # --- geometric mask (child override) ---
        mask_geo = self.filter_detections(dets)
        mask_geo = np.asarray(mask_geo, dtype=bool).reshape(-1) if mask_geo.size else np.ones(0, dtype=bool)

        # normalize shapes before combining
        if mask_cls.shape[0] != n:
            mask_cls = np.resize(mask_cls, n)
        if mask_geo.shape[0] != n:
            mask_geo = np.resize(mask_geo, n)

        mask = np.logical_and(mask_cls, mask_geo)

        # apply mask and pass to underlying tracker
        filtered = dets[mask]
        # ensure shape (M,6) for tracker
        if filtered.size == 0:
            filtered = np.empty((0, 6), dtype=float)

        # final call to underlying tracker
        tracked = self.tracker.update(filtered, frame_data)
        return tracked



# # --- Base Tracker ---
# class Tracker:
#     def __init__(self,
#                  method: str = "ocsort",
#                  reid_model: str = "osnet_x0_25_market1501.pt",
#                  classes: Optional[List[int]] = None,  # user-facing param name
#                  device: str = "",
#                  half: bool = False,
#                  per_class: bool = True):
#         """
#         Base Tracker.

#         Args:
#             method: tracker method name
#             reid_model: path to reid model
#             classes: list of class ids to keep (None => keep all)
#             device, half, per_class: forwarded to create_tracker
#         """
#         BaseTrack.clear_count()
#         self.method = method
#         self.config_file = get_tracker_config(method)
#         self.reid_model = Path(reid_model)
#         self.device = device
#         self.half = half
#         self.per_class = per_class

#         # accept lists/tuples/sets of ints (including numpy int types); otherwise skip class filtering
#         if not isinstance(classes, (list, tuple, set)) \
#            or not all(isinstance(c, (int, np.integer)) for c in classes):
#             logger.info(
#                 "`classes` parameter is not a list/tuple/set of integers; "
#                 "disabling class filtering and using default (no class filtering)."
#             )
#             self.target_classes = None 
#         else:    
#             self.target_classes = [int(c) for c in classes]  # e.g. [0,4,6] or None for all

#         self.tracker = create_tracker(
#             self.method,
#             self.config_file,
#             self.reid_model,
#             self.device,
#             self.half,
#             self.per_class
#         )

#     # --- default geometric filter (no-op) ---
#     def filter_detections(self, detections: np.ndarray) -> np.ndarray:
#         """
#         Default geometric filter: return all-True mask so that base Tracker
#         applies only class filtering unless subclass overrides this.
#         Returns:
#             mask: np.ndarray of bool with shape (N,)
#         """
#         detections = np.asarray(detections)
#         n = detections.shape[0] if detections.ndim > 0 else 0
#         return np.ones(n, dtype=bool)

#     # --- centralized class filter ---
#     def filter_by_classes(self, detections: np.ndarray) -> np.ndarray:
#         """
#         Filter detections by class IDs.

#         Expected detections shape: (N, 6) -> (x1, y1, x2, y2, conf, cls)

#         Returns:
#             mask: boolean ndarray (N,) where True means keep that detection.
#         """
#         detections = np.asarray(detections)
#         if detections.size == 0:
#             return np.zeros(0, dtype=bool)

#         # number of detections
#         n = detections.shape[0]

#         if self.target_classes is None:
#             return np.ones(n, dtype=bool)

#         # get class column (last or 5th index)
#         try:
#             cls_col = detections[:, 5]
#         except IndexError:
#             # malformed detections -> default to keep none
#             return np.zeros(n, dtype=bool)

#         # ensure integer class ids
#         cls_col = cls_col.astype(int)
#         mask = np.isin(cls_col, self.target_classes)
#         return mask

#     # --- main update: apply class + geometric filters, then forward to tracker ---
#     def update(self, detections: np.ndarray, frame_data: Any):
#         """
#         Update tracker with filtered detections.

#         Filtering order:
#           1) class filter (centralized)
#           2) geometric filter (self.filter_detections, overridden by children)
#           3) logical AND of both masks

#         Returns:
#             result of self.tracker.update(filtered_detections, frame_data)
#         """
#         if self.tracker is None:
#             raise RuntimeError("Tracker not initialized.")

#         detections = np.asarray(detections)
#         n = detections.shape[0] if detections.ndim > 0 else 0

#         # class mask
#         mask_cls = self.filter_by_classes(detections)

#         # geometric mask (child override)
#         mask_geo = self.filter_detections(detections)
#         mask_geo = np.asarray(mask_geo, dtype=bool).reshape(-1) if mask_geo.size else np.ones(0, dtype=bool)

#         # normalize shapes and combine
#         if mask_cls.shape[0] != n:
#             mask_cls = np.resize(mask_cls, n)
#         if mask_geo.shape[0] != n:
#             mask_geo = np.resize(mask_geo, n)

#         mask = np.logical_and(mask_cls, mask_geo)

#         # apply mask and pass to underlying tracker
#         filtered = detections[mask]
#         tracked = self.tracker.update(filtered, frame_data)
#         return tracked


# --- Combined tracker with ROI ---
class ROITracker(Tracker, ROI):
    def __init__(self,
                 name: str,
                 enabled: bool,
                 polygon: np.ndarray,
                 filter_fn: ROI.FilterFn,
                 tracker_method: str = "ocsort",
                 reid_model: str = "osnet_x0_25_market1501.pt",
                 classes: Optional[List[int]] = None,
                 device: str = "",
                 half: bool = False,
                 per_class: bool = True,
                 mask: Optional[np.ndarray] = None,
                 mask_x_min: Optional[int] = None,
                 mask_y_min: Optional[int] = None,
                 line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 line_vicinity: Optional[float] = None,
                 frame_size: int = 1000):

        # Init Tracker (passes classes)
        Tracker.__init__(self,
                         method=tracker_method,
                         reid_model=reid_model,
                         classes=classes,
                         device=device,
                         half=half,
                         per_class=per_class)

        # Init ROI area
        ROI.__init__(self,
                     name=name,
                     enabled=enabled,
                     polygon=polygon,
                     filter_fn=filter_fn,
                     mask=mask,
                     mask_x_min=mask_x_min,
                     mask_y_min=mask_y_min,
                     line=line,
                     line_vicinity=line_vicinity,
                     frame_size=frame_size)

    def filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        ROI geometric filtering only (no class filtering here).
        Returns boolean mask (N,).
        """
        detections = np.asarray(detections)
        n = detections.shape[0] if detections.ndim > 0 else 0
        if n == 0:
            return np.zeros(0, dtype=bool)

        mask_poly = self.filter_by_polygon(detections)
        mask_poly = np.asarray(mask_poly, dtype=bool).reshape(-1)

        if self.line:
            mask_line = self.filter_by_vicinity(detections)
            mask_line = np.asarray(mask_line, dtype=bool).reshape(-1)
            return np.logical_and(mask_poly, mask_line)
        return mask_poly


# --- Combined tracker with Global area ---
class GlobalTracker(Tracker, GlobalArea):
    def __init__(self,
                 enabled: bool,
                 tracker_method: str = "ocsort",
                 reid_model: str = "osnet_x0_25_market1501.pt",
                 classes: Optional[List[int]] = None,
                 device: str = "",
                 half: bool = False,
                 per_class: bool = True,
                 line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
                 line_vicinity: Optional[float] = None,
                 frame_size: int = 1000):

        Tracker.__init__(self,
                         method=tracker_method,
                         reid_model=reid_model,
                         classes=classes,
                         device=device,
                         half=half,
                         per_class=per_class)

        GlobalArea.__init__(self,
                            enabled=enabled,
                            line=line,
                            line_vicinity=line_vicinity,
                            frame_size=frame_size)

    def filter_detections(self, detections: np.ndarray) -> np.ndarray:
        """
        Global geometric filter (line vicinity) only.
        Returns boolean mask (N,).
        """
        detections = np.asarray(detections)
        n = detections.shape[0] if detections.ndim > 0 else 0
        if n == 0:
            return np.zeros(0, dtype=bool)

        if self.line:
            mask_line = self.filter_by_vicinity(detections)
            return np.asarray(mask_line, dtype=bool).reshape(-1)

        # no geometric restriction -> all-True (class filtering still applied by parent)
        return np.ones(n, dtype=bool)
