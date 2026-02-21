from typing import Optional, Tuple, Callable
import numpy as np
import logging

logger = logging.getLogger(__name__)


# --- your domain classes (slightly adjusted to accept frame_size) ---
class AreaBase:
    def __init__(self,
                 name: str,
                 enabled: bool,
                 line: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
                 line_vicinity: Optional[float] = None,
                 frame_size: int = 1000):
        self.name = name
        self.enabled = enabled
        self.line = line
        self.line_vicinity = float(line_vicinity) if line_vicinity is not None else None
        self.frame_size = int(frame_size)
        self.polygon = None

    def get_area_info(self) -> dict:
        """Return polygon, line, and line_vicinity as a dictionary."""
        return {
            "name": self.name,
            "polygon": self.polygon,
            "line": self.line,
            "line_vicinity": self.line_vicinity,
        }


    def filter_by_vicinity(self, bboxes: np.ndarray) -> np.ndarray:
        if self.line is None or self.line_vicinity is None:
            return np.zeros((bboxes.shape[0], 1), dtype=bool)

        centers_x = (bboxes[:,0] + bboxes[:,2]) / 2
        centers_y = (bboxes[:,1] + bboxes[:,3]) / 2
        centers = np.stack([centers_x, centers_y], axis=1)

        (x1,y1), (x2,y2) = self.line
        line_vec = np.array([x2-x1, y2-y1])
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.zeros((bboxes.shape[0],1), dtype=bool)

        vecs = centers - np.array([x1,y1])
        dists = np.abs(np.cross(line_vec, vecs) / line_len)
        threshold = self.line_vicinity * self.frame_size
        return (dists < threshold).reshape(-1,1)




    
class GlobalArea(AreaBase):
    def __init__(self,
                 enabled: bool,
                 line: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
                 line_vicinity: Optional[float] = None,
                 frame_size: int = 1000):
        super().__init__(name="global", enabled=enabled, line=line,
                         line_vicinity=line_vicinity, frame_size=frame_size)


class ROI(AreaBase):
    FilterFn = Callable[[np.ndarray, np.ndarray, int, int], np.ndarray]

    def __init__(self,
                 name: str,
                 enabled: bool,
                 polygon: np.ndarray,
                 filter_fn: FilterFn,
                 mask: Optional[np.ndarray] = None,
                 mask_x_min: Optional[int] = None,
                 mask_y_min: Optional[int] = None,
                 line: Optional[Tuple[Tuple[float,float], Tuple[float,float]]] = None,
                 line_vicinity: Optional[float] = None,
                 frame_size: int = 1000):
        super().__init__(name=name, enabled=enabled, line=line,
                         line_vicinity=line_vicinity, frame_size=frame_size)
        self.polygon = polygon
        self.filter_fn = filter_fn
        self.mask = mask
        self.mask_x_min = mask_x_min
        self.mask_y_min = mask_y_min

    def filter_by_polygon(self, bboxes: np.ndarray) -> np.ndarray:
        if self.filter_fn is None or self.mask is None:
            return np.zeros((bboxes.shape[0],1), dtype=bool)
        return self.filter_fn(bboxes, self.mask, self.mask_x_min, self.mask_y_min)