# count.py
"""
Counting module with central class-filtering and ID history tracking (ready to paste).

Key features:
 - Maintains HISTORY_BY_ID: Dict[int, Dict[str, bool]] shared across counters.
 - KNOWN_AREAS: Set[str] of registered area names; new areas register themselves.
 - ID-aware counters set HISTORY_BY_ID[id][area_name] = True when an ID is seen in that area.
 - Provides helpers to inspect/clear history.
"""

from typing import Optional, Tuple, List, Dict, Any, Set, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging
import copy

import numpy as np

# Local modules expected to exist in your project
from geometry.areas import AreaBase, GlobalArea, ROI
from geometry.areas_config_loader import ConfigLoader  # kept if you use it elsewhere

logger = logging.getLogger(__name__)

# === Shared ID history and area registry ===
# HISTORY_BY_ID: id -> { area_name: bool, ... }
HISTORY_BY_ID: Dict[int, Dict[str, bool]] = {}

# Known area names so we can keep consistent keys across id entries
KNOWN_AREAS: Set[str] = set()


def register_area_name(name: str):
    """
    Register a new area name. Add a False entry for this name to all existing IDs.
    """
    if name in KNOWN_AREAS:
        return
    KNOWN_AREAS.add(name)
    for id_, mapping in HISTORY_BY_ID.items():
        # set explicit False for this new area unless it already exists
        if name not in mapping:
            mapping[name] = False


def ensure_id_entry(id_: int):
    """
    Ensure HISTORY_BY_ID has an entry for id_. Initialize all known areas to False.
    """
    if id_ not in HISTORY_BY_ID:
        HISTORY_BY_ID[id_] = {name: False for name in KNOWN_AREAS}


def get_id_history(id_: int) -> Optional[Dict[str, bool]]:
    """
    Return a copy of the history mapping (area_name -> bool) for given id (or None).
    """
    if id_ not in HISTORY_BY_ID:
        return None
    return copy.deepcopy(HISTORY_BY_ID[id_])


def get_all_history() -> Dict[int, Dict[str, bool]]:
    """
    Return a deep copy of the full history mapping.
    """
    return copy.deepcopy(HISTORY_BY_ID)


def clear_history():
    """
    Clear the shared history store (keep KNOWN_AREAS).
    """
    HISTORY_BY_ID.clear()


@dataclass
class CountResult:
    total_count: int = 0
    counts_by_class: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


# ---------------- Base (shared) ----------------
class CountingBaseMixin:
    """Base mixin for counting logic. Holds CountResult and builds combined filters.

    Key features:
      - `classes` argument to enable centralized class filtering (like Tracker.filter_by_classes)
      - `overall_filter` built by subclasses (geometry / id / vicinity)
      - `count()` applies class filter first, then overall_filter, then delegates to `update()`
    """

    def __init__(
        self,
        counting_logic: Optional[set[str]] = None,
        by_polygon: bool = True,
        classes: Optional[List[int]] = None,
    ):
        self.count_result: CountResult = CountResult()
        self.counting_logic: set[str] = counting_logic or {"by_id"}
        self.by_polygon = by_polygon
        # track_ids is True if counting involves object IDs
        self.track_ids = any(k in self.counting_logic for k in ("by_id", "by_cross_id"))

        # default column indices (may be updated by callers if they use a different format)
        if self.track_ids:
            self.id_col = 4
            self.cls_col = 6
            self.n_cols = 8
        else:
            self.id_col = None
            self.cls_col = 5
            self.n_cols = 6

        # class filtering (mirror of Tracker logic)
        self.target_classes: Optional[List[int]] = None
        if classes is None:
            self.target_classes = None
        else:
            if not isinstance(classes, (list, tuple, set)) or not all(isinstance(c, (int, np.integer)) for c in classes):
                logger.info(
                    "`classes` parameter is not a list/tuple/set of integers; disabling class filtering and using default (no class filtering)."
                )
                self.target_classes = None
            else:
                self.target_classes = [int(c) for c in classes]

        # overall_filter should be set by area-specializations using _build_overall_filter
        self.overall_filter: Optional[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = None

    def _build_overall_filter(
        self,
        poly_fn: Optional[Callable] = None,
        vic_fn: Optional[Callable] = None,
        id_fn: Optional[Callable] = None,
        cross_fn: Optional[Callable] = None,
    ) -> Callable:
        """
        Build a combined filter function from available component functions.

        Each provided function should accept (bboxes, ids) and return a boolean mask of shape (N,) or (N,1).
        The returned combined function has signature combined(bboxes, ids) -> np.ndarray(bool, shape (N,)).

        Special-case behaviour (if cross + id + vicinity are requested):
            mask = id & (cross | vicinity)
        Otherwise all enabled filters are AND-ed together.
        """
        logic_set = self.counting_logic
        funcs: List[Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]] = []

        # --- wrappers (normalize signatures + shapes) ---
        if self.by_polygon and poly_fn is not None:
            def _wrap_poly(bboxes, ids):
                return np.asarray(poly_fn(bboxes)).reshape(-1).astype(bool)
            funcs.append(_wrap_poly)

        if "by_vicinity" in logic_set and vic_fn is not None:
            def _wrap_vic(bboxes, ids):
                return np.asarray(vic_fn(bboxes)).reshape(-1).astype(bool)

        if self.track_ids:
            if "by_id" in logic_set and id_fn is not None:
                def _wrap_id(bboxes, ids):
                    return np.asarray(id_fn(ids)).reshape(-1).astype(bool)

            if "by_cross_id" in logic_set and cross_fn is not None:
                def _wrap_cross(bboxes, ids):
                    return np.asarray(cross_fn(bboxes, ids)).reshape(-1).astype(bool)

        # --- 🔴 FIXED SPECIAL CASE ---
        trio = {"by_cross_id", "by_id", "by_vicinity"}
        if trio.issubset(logic_set) and cross_fn and id_fn and vic_fn:
            def _special_or(bboxes, ids):
                m_id = _wrap_id(bboxes, ids)
                m_cross = _wrap_cross(bboxes, ids)
                m_vic = _wrap_vic(bboxes, ids)

                # require id-filter for BOTH cross and vicinity
                return (m_id & (m_cross | m_vic)).astype(bool)

            funcs.append(_special_or)

        else:
            if self.track_ids:
                if "by_cross_id" in logic_set and cross_fn is not None:
                    funcs.append(_wrap_cross)
                if "by_id" in logic_set and id_fn is not None:
                    funcs.append(_wrap_id)

            if "by_vicinity" in logic_set and vic_fn is not None:
                funcs.append(_wrap_vic)

        def combined_filter(bboxes, ids=None):
            if bboxes is None or bboxes.size == 0:
                return np.zeros((0,), dtype=bool)

            if not funcs:
                return np.ones((bboxes.shape[0],), dtype=bool)

            mask = None
            for fn in funcs:
                m = fn(bboxes, ids)
                mask = m if mask is None else (mask & m)

            return mask.astype(bool)

        return combined_filter

    def filter_by_classes(self, objects: np.ndarray) -> np.ndarray:
        """
        Filter rows by class id. Accepts the full detection array (same format passed to count).
        Returns mask of shape (N,) boolean: True -> keep.
        """
        arr = np.asarray(objects) if objects is not None else np.empty((0, self.n_cols))
        if arr.size == 0:
            return np.zeros(0, dtype=bool)

        n = arr.shape[0]
        if self.target_classes is None:
            return np.ones(n, dtype=bool)

        if arr.shape[1] <= self.cls_col:
            # malformed rows: keep none
            return np.zeros(n, dtype=bool)

        cls_col = arr[:, self.cls_col].astype(int)
        mask = np.isin(cls_col, self.target_classes)
        return mask.reshape(-1)

    def update(self, objects: Optional[np.ndarray] = None):
        """Abstract update method; must be implemented in subclasses."""
        raise NotImplementedError("update() must be implemented in subclasses")

    def count(self, objects: Optional[np.ndarray]) -> CountResult:
        """
        Apply class filter first, then overall_filter (geometry/ID/vicinity),
        then call update() and return CountResult.
        """
        # normalize input
        arr = np.asarray(objects) if objects is not None else np.empty((0, self.n_cols))
        if arr.size == 0:
            return self.count_result
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # basic shape checks
        if arr.shape[1] != self.n_cols:
            logger.warning(f"Unexpected array shape: expected {self.n_cols} cols, got {arr.shape[1]}")
            if arr.shape[1] == 6:
                return self.count_result
            if arr.shape[1] == 8:
                # adapt to 8-col format
                self.id_col, self.cls_col, self.n_cols = 4, 6, 8

        # --- central class filter (applied first) ---
        class_mask = self.filter_by_classes(arr)  # shape (N,)
        if class_mask.size != arr.shape[0]:
            logger.warning("filter_by_classes returned mask with unexpected length; defaulting to all rows.")
            class_mask = np.ones((arr.shape[0],), dtype=bool)
        class_mask = np.asarray(class_mask).reshape(-1).astype(bool)

        # extract bboxes (first 4 cols) and ids (if present)
        bboxes = arr[:, :4].astype(float)
        ids = None
        if self.track_ids:
            ids = arr[:, self.id_col].astype(int)

        # call overall_filter (if present)
        filter_fn = self.overall_filter
        if filter_fn is None:
            geo_mask = np.ones((arr.shape[0],), dtype=bool)
        else:
            m = filter_fn(bboxes, ids)
            if m is None:
                geo_mask = np.ones((arr.shape[0],), dtype=bool)
            else:
                geo_mask = np.asarray(m).reshape(-1).astype(bool)

        # combine class + geometry masks
        combined_mask = class_mask & geo_mask

        # build filtered objects (respect expected number of cols)
        if combined_mask.size != arr.shape[0]:
            logger.warning("count(): combined filter produced unexpected length; using all rows.")
            filtered = arr
        else:
            filtered = arr[combined_mask]
            if filtered.size == 0:
                filtered = np.empty((0, self.n_cols))

        # delegate to subclass update() which updates and returns CountResult
        result = self.update(filtered)
        return result


class CountingWithIdsMixin(CountingBaseMixin):
    """
    Counting mixin with ID tracking.
    Provides:
      - filter_by_id
      - filter_by_cross_id
      - update (with counted_ids management and shared HISTORY_BY_ID)
    """

    def __init__(self, counting_logic: Optional[set[str]] = None, by_polygon: bool = True, classes: Optional[List[int]] = None):
        logic = counting_logic or {"by_id"}
        super().__init__(counting_logic=logic, by_polygon=by_polygon, classes=classes)

        # Enforce correct usage
        if not self.track_ids:
            raise ValueError(
                "CountingWithIdsMixin requires counting logic to contain 'by_id' or 'by_cross_id'."
            )

        # State for tracking IDs
        self.counted_ids: Set[int] = set()
        self.prev_signs: Dict[int, int] = {}

        # NOTE: area name is set by AreaBase.__init__ later; registration happens in area constructors
        # Access to shared HISTORY_BY_ID via module-level variable
        self.history_id = HISTORY_BY_ID
        
    def filter_by_id(self, ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(ids).reshape(-1)
        if not self.counted_ids:
            return np.ones((ids.shape[0],), dtype=bool)
        mask = ~np.isin(ids, np.fromiter(self.counted_ids, dtype=int))
        return mask.reshape(-1)

    def update(self, filtered_objects: Optional[np.ndarray] = None):
        """
        Update counts, counted_ids, per-class counts, AND the shared HISTORY_BY_ID.

        For each ID present in filtered_objects:
          - counted_ids is updated (so future calls can filter out duplicates if desired)
          - HISTORY_BY_ID[id][self.name] is set to True (initializing the id entry if needed)
        """
        result = self.count_result
        counted_ids = self.counted_ids

        arr = np.asarray(filtered_objects) if filtered_objects is not None else np.empty((0, self.n_cols))
        if arr.size == 0:
            return result
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.n_cols:
            logger.warning(f"Unexpected array shape: expected {self.n_cols} cols, got {arr.shape[1]}")

        # total count
        result.total_count += arr.shape[0]

        # class counts
        cls_vals = arr[:, self.cls_col].astype(int) if arr.shape[1] > self.cls_col else np.array([], dtype=int)
        if cls_vals.size:
            max_cls = int(cls_vals.max())
            if result.counts_by_class.size <= max_cls:
                new_counts = np.zeros(max_cls + 1, dtype=np.int64)
                new_counts[: result.counts_by_class.size] = result.counts_by_class
                result.counts_by_class = new_counts
            result.counts_by_class += np.bincount(cls_vals, minlength=result.counts_by_class.size)

        # counted IDs
        ids_col = arr[:, self.id_col].astype(int) if arr.shape[1] > self.id_col else np.array([], dtype=int)
        # update counted_ids set
        counted_ids.update(map(int, ids_col))

        # --- Update shared HISTORY_BY_ID for every seen id ---
        area_name = getattr(self, "name", None)
        if area_name is None:
            # Name should be set by AreaBase; warn if not present
            logger.warning("Area name not set while updating HISTORY_BY_ID; ID history will not be updated for this area.")
        else:
            # ensure this area is registered (register_area_name will add False entries for existing ids)
            register_area_name(area_name)
            for raw_id in ids_col:
                id_int = int(raw_id)
                ensure_id_entry(id_int)
                HISTORY_BY_ID[id_int][area_name] = True

        # persist updates
        self.count_result = result
        self.counted_ids = counted_ids
        return result


# ---------------- No IDs ----------------
class CountingWithoutIdsMixin(CountingBaseMixin):
    """
    Counting mixin without ID tracking.
    Provides:
      - update (only total_count and per-class counts)
    """

    def __init__(self, counting_logic: Optional[set[str]] = None, by_polygon: bool = True, classes: Optional[List[int]] = None):
        logic = counting_logic or {"by_vicinity"}
        super().__init__(counting_logic=logic, by_polygon=by_polygon, classes=classes)

        # Enforce correct usage
        if self.track_ids:
            raise ValueError(
                "CountingWithoutIdsMixin cannot be used with 'by_id' or 'by_cross_id'. Use CountingWithIdsMixin instead."
            )

    def update(self, filtered_objects: Optional[np.ndarray] = None):
        result = self.count_result
        arr = np.asarray(filtered_objects) if filtered_objects is not None else np.empty((0, self.n_cols))
        if arr.size == 0:
            return result
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != self.n_cols:
            logger.warning(f"Unexpected array shape: expected {self.n_cols} cols, got {arr.shape[1]}")

        # total count
        result.total_count += arr.shape[0]

        # class counts
        cls_vals = arr[:, self.cls_col].astype(int) if arr.shape[1] > self.cls_col else np.array([], dtype=int)
        if cls_vals.size:
            max_cls = int(cls_vals.max())
            if result.counts_by_class.size <= max_cls:
                new_counts = np.zeros(max_cls + 1, dtype=np.int64)
                new_counts[: result.counts_by_class.size] = result.counts_by_class
                result.counts_by_class = new_counts
            result.counts_by_class += np.bincount(cls_vals, minlength=result.counts_by_class.size)

        self.count_result = result
        return result


# ---------------- Area / ROI concrete classes (forward classes param) ----------------
class CountingAreaWithIds(CountingWithIdsMixin, AreaBase):
    """
    Combines CountingWithIdsMixin (ID-aware counting) and AreaBase (geometric filters).
    Registers its name in the KNOWN_AREAS registry so history keys stay consistent.
    """

    def __init__(
        self,
        name: str,
        enabled: bool,
        counting_logic: Optional[set[str]] = None,
        by_polygon: bool = False,
        line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line_vicinity: Optional[float] = None,
        frame_size: int = 1000,
        classes: Optional[List[int]] = None,
    ):
        CountingWithIdsMixin.__init__(self, counting_logic=counting_logic, by_polygon=by_polygon, classes=classes)
        AreaBase.__init__(
            self,
            name=name,
            enabled=enabled,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
        )

        # register area name globally so HISTORY_BY_ID rows include this column (default False)
        register_area_name(self.name)


    def filter_by_cross_id(
        self,
        bboxes: np.ndarray,
        ids: np.ndarray,
        counted_ids: Optional[Set[int]] = None,
        prev_signs: Optional[Dict[int, int]] = None,
    ) -> np.ndarray:
        """
        Crossing-only filter: detect *all* objects that crossed the line (based on sign changes),
        intentionally decoupled from counted_ids; uniqueness/newness should be handled by filter_by_id.
        Signature kept for backwards compatibility (counted_ids param is accepted but ignored).
        """
        if bboxes.size == 0:
            return np.zeros((0,), dtype=bool)

        ids = np.asarray(ids, dtype=int)
        # counted_ids intentionally ignored here to decouple crossing detection from new-ID filtering
        prev_signs = prev_signs if prev_signs is not None else self.prev_signs

        # Centers
        cx = (bboxes[:, 0] + bboxes[:, 2]) / 2
        cy = (bboxes[:, 1] + bboxes[:, 3]) / 2

        # Line vector
        (x1, y1), (x2, y2) = self.line
        lx, ly = x2 - x1, y2 - y1

        # Side detection
        det = lx * (cy - y1) - ly * (cx - x1)
        sign = np.sign(det).astype(int)

        crossed = np.zeros_like(sign, dtype=bool)

        for i, id_ in enumerate(ids):
            prev = prev_signs.get(int(id_), 0)
            if (prev != 0 and sign[i] != prev) or sign[i] == 0:
                crossed[i] = True
            prev_signs[int(id_)] = sign[i] if sign[i] != 0 else prev

        self.prev_signs = prev_signs
        return crossed.reshape(-1)


class CountingGlobalAreaWithIds(CountingAreaWithIds, GlobalArea):
    def __init__(
        self,
        enabled: bool,
        counting_logic: Optional[set[str]] = None,
        line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line_vicinity: Optional[float] = None,
        frame_size: int = 1000,
        classes: Optional[List[int]] = None,
    ):
        CountingAreaWithIds.__init__(
            self,
            name="global",
            enabled=enabled,
            counting_logic=counting_logic,
            by_polygon=False,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
            classes=classes,
        )
        GlobalArea.__init__(
            self,
            enabled=enabled,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
        )

        # build overall_filter using available area functions
        self.overall_filter = self._build_overall_filter(vic_fn=self.filter_by_vicinity, id_fn=self.filter_by_id, cross_fn=self.filter_by_cross_id)


class CountingROIWithIds(CountingAreaWithIds, ROI):
    def __init__(
        self,
        name: str,
        enabled: bool,
        polygon: np.ndarray,
        filter_fn: ROI.FilterFn,
        counting_logic: Optional[set[str]] = None,
        mask: Optional[np.ndarray] = None,
        mask_x_min: Optional[int] = None,
        mask_y_min: Optional[int] = None,
        line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line_vicinity: Optional[float] = None,
        frame_size: int = 1000,
        classes: Optional[List[int]] = None,
    ):
        CountingAreaWithIds.__init__(
            self,
            name=name,
            enabled=enabled,
            counting_logic=counting_logic,
            by_polygon=True,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
            classes=classes,
        )

        ROI.__init__(
            self,
            name=name,
            enabled=enabled,
            polygon=polygon,
            filter_fn=filter_fn,
            mask=mask,
            mask_x_min=mask_x_min,
            mask_y_min=mask_y_min,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
        )

        self.overall_filter = self._build_overall_filter(poly_fn=self.filter_by_polygon, vic_fn=self.filter_by_vicinity, id_fn=self.filter_by_id, cross_fn=self.filter_by_cross_id)


# ---------------- GlobalArea variants (without IDs) ----------------
class CountingGlobalAreaWithoutIds(CountingWithoutIdsMixin, GlobalArea):
    def __init__(
        self,
        enabled: bool,
        counting_logic: Optional[set[str]] = None,
        line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line_vicinity: Optional[float] = None,
        frame_size: int = 1000,
        classes: Optional[List[int]] = None,
    ):
        CountingWithoutIdsMixin.__init__(self, counting_logic=counting_logic, by_polygon=False, classes=classes)
        GlobalArea.__init__(self, enabled=enabled, line=line, line_vicinity=line_vicinity, frame_size=frame_size)

        self.overall_filter = self._build_overall_filter(vic_fn=self.filter_by_vicinity)


# ---------------- ROI variants (without IDs) ----------------
class CountingROIWithoutIds(CountingWithoutIdsMixin, ROI):
    def __init__(
        self,
        name: str,
        enabled: bool,
        polygon: np.ndarray,
        filter_fn,
        counting_logic: Optional[set[str]] = None,
        mask: Optional[np.ndarray] = None,
        mask_x_min: Optional[int] = None,
        mask_y_min: Optional[int] = None,
        line: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
        line_vicinity: Optional[float] = None,
        frame_size: int = 1000,
        classes: Optional[List[int]] = None,
    ):
        CountingWithoutIdsMixin.__init__(self, counting_logic=counting_logic, by_polygon=True, classes=classes)
        ROI.__init__(
            self,
            name=name,
            enabled=enabled,
            polygon=polygon,
            filter_fn=filter_fn,
            mask=mask,
            mask_x_min=mask_x_min,
            mask_y_min=mask_y_min,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=frame_size,
        )

        # register area name for consistency with ID-aware areas too
        register_area_name(name)

        self.overall_filter = self._build_overall_filter(poly_fn=self.filter_by_polygon, vic_fn=self.filter_by_vicinity)


# End of file
