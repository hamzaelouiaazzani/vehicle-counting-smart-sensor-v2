# counting_config_loader.py
# Ready-to-paste CountingConfigLoader that supports `classes` per-ROI/global
# If YAML doesn't specify classes, falls back to default_classes passed to constructor.

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Set, Iterable

# Local modules - adjust import paths to your repo layout if needed
from geometry.areas_config_loader import ConfigLoader
from counting.count import (
    CountingGlobalAreaWithIds,
    CountingGlobalAreaWithoutIds,
    CountingROIWithIds,
    CountingROIWithoutIds,
)

logger = logging.getLogger(__name__)


class CountingConfigLoader(ConfigLoader):
    """
    Extends ConfigLoader to produce *counting* area objects.

    Features:
      - Normalizes counting_logic values (list -> set)
      - Validates allowed combinations
      - Chooses Counting*WithIds vs Counting*WithoutIds classes
      - Supports `classes` attribute per-ROI/global in YAML; falls back to default_classes
    """

    # allowed sets (use frozenset for hashing/comparison clarity)
    _ALLOWED_LOGICS = {
        frozenset({"by_vicinity"}),
        frozenset({"by_id"}),
        frozenset({"by_cross_id"}),
        frozenset({"by_id", "by_vicinity"}),
        frozenset({"by_cross_id", "by_vicinity"}),
        frozenset({"by_cross_id", "by_id", "by_vicinity"}),
    }

    def __init__(
        self,
        default_config_path: Path | str = "configs/counter_config.yaml",
        default_polygon_mode: str = "intersects_polygon",
        default_line_vicinity: float = 0.20,
        frame_size: int = 1000,
        default_counting_logic: Optional[Iterable[str]] = None,
        default_classes: Optional[Iterable[int]] = None,
    ):
        super().__init__(
            default_config_path=default_config_path,
            default_polygon_mode=default_polygon_mode,
            default_line_vicinity=default_line_vicinity,
            frame_size=frame_size,
        )
        # default counting logic as a normalized set (fallback)
        if default_counting_logic is None:
            default_counting_logic = ["by_vicinity"]
        self.default_counting_logic: Set[str] = self._to_logic_set(default_counting_logic)

        # default classes: None means "no class filtering"
        self.default_classes: Optional[List[int]] = self._to_classes(default_classes)

    # ------- helpers for counting logic -------
    def _to_logic_set(self, raw: Optional[Iterable[str] | str]) -> Set[str]:
        """Convert incoming raw counting_logic (list/tuple/set/str) to normalized set and validate."""
        if raw is None:
            return set(self.default_counting_logic)

        # accept str or iterable
        if isinstance(raw, str):
            s = {raw}
        else:
            try:
                s = set(map(str, raw))
            except Exception:
                logger.warning(
                    "Invalid counting_logic value %r — falling back to default %s",
                    raw,
                    self.default_counting_logic,
                )
                return set(self.default_counting_logic)

        fr = frozenset(s)
        if fr not in self._ALLOWED_LOGICS:
            logger.warning(
                "Unrecognized counting_logic %s — allowed: %s — falling back to default %s",
                s,
                [set(x) for x in self._ALLOWED_LOGICS],
                self.default_counting_logic,
            )
            return set(self.default_counting_logic)
        return set(s)

    # ------- helpers for classes -------
    def _to_classes(self, raw: Optional[Iterable[int] | int]) -> Optional[List[int]]:
        """
        Normalize classes field (None, int, list/tuple/set of ints) -> Optional[List[int]]
        Returns None if raw is None or invalid (meaning: no class filtering).
        """
        if raw is None:
            return None

        # single integer
        if isinstance(raw, (int,)) or (hasattr(raw, "__index__") and not isinstance(raw, (str, bytes))):
            try:
                return [int(raw)]
            except Exception:
                logger.warning("Invalid classes value %r — expected int/list of ints. Disabling class filter.", raw)
                return None

        # iterable of ints
        try:
            cls_list = list(map(int, raw))
            if not cls_list:
                return None
            return cls_list
        except Exception:
            logger.warning("Invalid classes value %r — expected int/list of ints. Disabling class filter.", raw)
            return None

    def _normalize_counting_logic(
        self,
        logic_set: set,
        line: Optional[Any],
        line_vicinity: Optional[float],
        name: str = "global",
    ) -> Tuple[set, Optional[Any], Optional[float]]:
        """
        Normalize counting logic rules for global area.

        Ensures line/line_vicinity consistency with the given logic_set.
        Raises ValueError if line is missing when required.
        """
        if "by_vicinity" not in logic_set:
            if "by_cross_id" not in logic_set:
                line = None
            line_vicinity = None

        if (("by_vicinity" in logic_set) or ("by_cross_id" in logic_set)) and line is None:
            raise ValueError(
                f"Config error: 'line' must be provided when using {logic_set} logic in '{name}'."
            )

        if "by_vicinity" in logic_set:
            if not isinstance(line_vicinity, (int, float)) or not (0 <= float(line_vicinity) <= 1):
                logger.info(
                    "Invalid or missing 'line_vicinity' for %s — using default %.2f",
                    name,
                    self.default_line_vicinity,
                )
                line_vicinity = self.default_line_vicinity

        return logic_set, line, line_vicinity

    def _normalize_roi(self, raw: Dict[str, Any], idx: int):
        """
        Same behavior as ConfigLoader._normalize_roi but returns a CountingROI* instance
        (or None if disabled/invalid).
        Expects YAML `raw` may contain a key `classes:` (int or list of ints).
        """
        name = raw.get("name") or f"ROI_{idx}"
        enabled = raw.get("enable") or False
        if enabled is not True:
            return None

        polygon_raw = raw.get("polygon")
        if polygon_raw in (None, "null"):
            return None  # skip invalid/no polygon

        polygon_arr = self._validate_polygon(polygon_raw, name)
        if polygon_arr is None:
            raise ValueError(f"{name}: invalid polygon format, expected list of [x,y] pairs.")

        # line / line_vicinity handling: only parse here, validation/defaulting done by helper
        line = self._validate_line(raw.get("line"))
        line_vicinity = raw.get("line_vicinity")

        # counting logic normalization (list -> set, validated)
        logic_set = self._to_logic_set(raw.get("counting_logic") or self.default_counting_logic)

        # Validate logic + line requirements early (pass name for logging)
        logic_set, line, line_vicinity = self._normalize_counting_logic(
            logic_set, line, line_vicinity, name=name
        )

        # Now compute mask (done after config validation so heavy work is avoided on invalid config)
        box_mode = raw.get("box_in_polygon_mode") or self.default_polygon_mode
        filter_fn = self._select_polygon_filter(box_mode)
        mask, mask_x_min, mask_y_min = self._compute_mask_safe(polygon_arr, name)

        # classes: prefer YAML value, else loader default_classes, else None (no class filtering)
        classes_raw = raw.get("classes", None)
        classes = self._to_classes(classes_raw) if classes_raw is not None else self.default_classes

        # choose the class based on logic
        if logic_set == {"by_vicinity"}:
            cls = CountingROIWithoutIds
        else:
            cls = CountingROIWithIds

        return cls(
            name=name,
            enabled=True,
            polygon=polygon_arr,
            filter_fn=filter_fn,
            counting_logic=logic_set,
            mask=mask,
            mask_x_min=mask_x_min,
            mask_y_min=mask_y_min,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=self.frame_size,
            classes=classes,
        )

    def _normalize_global(self, raw_global: Dict[str, Any]):
        """
        Same behavior as ConfigLoader._normalize_global but returns a CountingGlobalArea* instance
        (or None if disabled/invalid). Expects optional `classes:` key.
        """
        name = "global"
        enabled = raw_global.get("enable") or False
        if enabled is not True:
            return None

        # parse raw values (validation/defaulting happens in helper)
        line = self._validate_line(raw_global.get("line"))
        line_vicinity = raw_global.get("line_vicinity")

        logic_set = self._to_logic_set(raw_global.get("counting_logic") or self.default_counting_logic)

        # validate/normalize logic, line and line_vicinity (may raise ValueError if required line is missing)
        logic_set, line, line_vicinity = self._normalize_counting_logic(
            logic_set, line, line_vicinity, name=name
        )

        # classes: prefer YAML value, else loader default_classes, else None
        classes_raw = raw_global.get("classes", None)
        classes = self._to_classes(classes_raw) if classes_raw is not None else self.default_classes

        # choose class based on logic
        if logic_set == {"by_vicinity"}:
            cls = CountingGlobalAreaWithoutIds
        else:
            cls = CountingGlobalAreaWithIds

        return cls(
            enabled=True,
            counting_logic=logic_set,
            line=line,
            line_vicinity=line_vicinity,
            frame_size=self.frame_size,
            classes=classes,
        )

    # optional convenience wrapper: returns counting-area objects
    def load_counting_areas(self) -> Tuple[List[Any], Optional[Any]]:
        """Public API returning (rois, global_area) as counting area objects."""
        return self.load()  # ConfigLoader.load is already compatible with our overrides








































# # --- Standard library ---
# import logging
# from pathlib import Path
# from typing import Optional, Tuple, List, Dict, Any, Set, Iterable

# # --- Local modules ---
# from geometry.areas_config_loader import ConfigLoader
# from counting.count import (
#     CountingGlobalAreaWithIds,
#     CountingGlobalAreaWithoutIds,
#     CountingROIWithIds,
#     CountingROIWithoutIds,
# )

# logger = logging.getLogger(__name__)



# class CountingConfigLoader(ConfigLoader):
#     """
#     Extends ConfigLoader to produce *counting* area objects.
#     - Normalizes counting_logic values (list -> set)
#     - Validates allowed combinations
#     - Chooses Counting*WithIds vs Counting*WithoutIds classes
#     """

#     # allowed sets (use frozenset for hashing/comparison clarity)
#     _ALLOWED_LOGICS = {
#         frozenset({"by_vicinity"}),
#         frozenset({"by_id"}),
#         frozenset({"by_cross_id"}),
#         frozenset({"by_id", "by_vicinity"}),
#         frozenset({"by_cross_id", "by_vicinity"}),
#         frozenset({"by_cross_id", "by_id", "by_vicinity"}),
#     }

#     def __init__(
#         self,
#         default_config_path: Path | str = "configs/counter_config.yaml",
#         default_polygon_mode: str = "intersects_polygon",
#         default_line_vicinity: float = 0.20,
#         frame_size: int = 1000,
#         default_counting_logic: Optional[Iterable[str]] = None,
#     ):
#         super().__init__(
#             default_config_path=default_config_path,
#             default_polygon_mode=default_polygon_mode,
#             default_line_vicinity=default_line_vicinity,
#             frame_size=frame_size,
#         )
#         # default counting logic as a normalized set (fallback)
#         if default_counting_logic is None:
#             default_counting_logic = ["by_vicinity"]
#         self.default_counting_logic: Set[str] = self._to_logic_set(default_counting_logic)

#     # ------- helpers for counting logic -------
#     def _to_logic_set(self, raw: Optional[Iterable[str] | str]) -> Set[str]:
#         """Convert incoming raw counting_logic (list/tuple/set/str) to normalized set and validate."""
#         if raw is None:
#             return set(self.default_counting_logic)

#         # accept str or iterable
#         if isinstance(raw, str):
#             s = {raw}
#         else:
#             try:
#                 s = set(map(str, raw))
#             except Exception:
#                 logger.warning("Invalid counting_logic value %r — falling back to default %s", raw, self.default_counting_logic)
#                 return set(self.default_counting_logic)

#         fr = frozenset(s)
#         if fr not in self._ALLOWED_LOGICS:
#             logger.warning("Unrecognized counting_logic %s — allowed: %s — falling back to default %s",
#                            s, [set(x) for x in self._ALLOWED_LOGICS], self.default_counting_logic)
#             return set(self.default_counting_logic)
#         return set(s)




#     def _normalize_counting_logic(
#         self,
#         logic_set: set,
#         line: Optional[Any],
#         line_vicinity: Optional[float],
#         name: str = "global",
#     ) -> Tuple[set, Optional[Any], Optional[float]]:
#         """
#         Normalize counting logic rules for global area.
    
#         Ensures line/line_vicinity consistency with the given logic_set.
#         Raises ValueError if line is missing when required.
#         """
#         if "by_vicinity" not in logic_set:
#             if "by_cross_id" not in logic_set:
#                 line = None
#             line_vicinity = None
    
#         if (("by_vicinity" in logic_set) or ("by_cross_id" in logic_set)) and line is None:
#             raise ValueError(
#                 f"Config error: 'line' must be provided when using {logic_set} logic in '{name}'."
#             )
    
#         if "by_vicinity" in logic_set:
#             if not isinstance(line_vicinity, (int, float)) or not (0 <= float(line_vicinity) <= 1):
#                 logger.info(
#                     "Invalid or missing 'line_vicinity' for %s — using default %.2f",
#                     name,
#                     self.default_line_vicinity,
#                 )
#                 line_vicinity = self.default_line_vicinity
    
#         return logic_set, line, line_vicinity

    
#     def _normalize_roi(self, raw: Dict[str, Any], idx: int):
#         """
#         Same behavior as ConfigLoader._normalize_roi but returns a CountingROI* instance
#         (or None if disabled/invalid).
#         """
#         name = raw.get("name") or f"ROI_{idx}"
#         enabled = raw.get("enable") or False
#         if enabled is not True:
#             return None
    
#         polygon_raw = raw.get("polygon")
#         if polygon_raw in (None, "null"):
#             return None  # skip invalid/no polygon
    
#         polygon_arr = self._validate_polygon(polygon_raw, name)
#         if polygon_arr is None:
#             raise ValueError(f"{name}: invalid polygon format, expected list of [x,y] pairs.")
    
#         # line / line_vicinity handling: only parse here, validation/defaulting done by helper
#         line = self._validate_line(raw.get("line"))
#         line_vicinity = raw.get("line_vicinity")
    
#         # counting logic normalization (list -> set, validated)
#         logic_set = self._to_logic_set(raw.get("counting_logic") or self.default_counting_logic)
    
#         # Validate logic + line requirements early (pass name for logging)
#         logic_set, line, line_vicinity = self._normalize_counting_logic(
#             logic_set, line, line_vicinity, name=name
#         )
    
#         # Now compute mask (done after config validation so heavy work is avoided on invalid config)
#         box_mode = raw.get("box_in_polygon_mode") or self.default_polygon_mode
#         filter_fn = self._select_polygon_filter(box_mode)
#         mask, mask_x_min, mask_y_min = self._compute_mask_safe(polygon_arr, name)
    
#         # choose the class based on logic
#         if logic_set == {"by_vicinity"}:
#             cls = CountingROIWithoutIds
#         else:
#             cls = CountingROIWithIds
    
#         return cls(
#             name=name,
#             enabled=True,
#             polygon=polygon_arr,
#             filter_fn=filter_fn,
#             counting_logic=logic_set,
#             mask=mask,
#             mask_x_min=mask_x_min,
#             mask_y_min=mask_y_min,
#             line=line,
#             line_vicinity=line_vicinity,
#             frame_size=self.frame_size,
#         )
    
#     def _normalize_global(self, raw_global: Dict[str, Any]):
#         """
#         Same behavior as ConfigLoader._normalize_global but returns a CountingGlobalArea* instance
#         (or None if disabled/invalid).
#         """
#         name = "global"
#         enabled = raw_global.get("enable") or False
#         if enabled is not True:
#             return None
    
#         # parse raw values (validation/defaulting happens in helper)
#         line = self._validate_line(raw_global.get("line"))
#         line_vicinity = raw_global.get("line_vicinity")
    
#         logic_set = self._to_logic_set(raw_global.get("counting_logic") or self.default_counting_logic)
    
#         # validate/normalize logic, line and line_vicinity (may raise ValueError if required line is missing)
#         logic_set, line, line_vicinity = self._normalize_counting_logic(
#             logic_set, line, line_vicinity, name=name
#         )
    
#         # choose class based on logic
#         if logic_set == {"by_vicinity"}:
#             cls = CountingGlobalAreaWithoutIds
#         else:
#             cls = CountingGlobalAreaWithIds
    
#         return cls(
#             enabled=True,
#             counting_logic=logic_set,
#             line=line,
#             line_vicinity=line_vicinity,
#             frame_size=self.frame_size,
#         )
    

#     # optional convenience wrapper: returns counting-area objects
#     def load_counting_areas(self) -> Tuple[List[Any], Optional[Any]]:
#         """Public API returning (rois, global_area) as counting area objects."""
#         return self.load()  # ConfigLoader.load is already compatible with our overrides

        