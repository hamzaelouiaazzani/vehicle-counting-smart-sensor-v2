# detectors/interface.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class DetectorError(Exception):
    """Raised for detector-specific failures."""
    pass


class IDetector(ABC):
    """
    Unified detector interface aligned with Ultralytics-style detectors.

    Canonical detector output:
        np.ndarray of shape (N, 6)
        [x1, y1, x2, y2, score, class_id]

    All detectors (Ultralytics, torchvision, TensorRT, custom)
    MUST be able to produce this format.
    """

    # -------------------------
    # Lifecycle
    # -------------------------

    @abstractmethod
    def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
        """Initialize detector resources (model, device, precision, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def warmup(self, imgsz: Any = None) -> None:
        """Optional warmup to reduce first-inference latency."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Release model / GPU / TensorRT resources."""
        raise NotImplementedError

    # -------------------------
    # Pipeline hooks (logical)
    # -------------------------

    @abstractmethod
    def preprocess(self, array_frame: np.ndarray):
        """
        Prepare input for inference.
        Return type is framework-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def infer(self, preprocessed_input, **kwargs):
        """
        Run model forward pass.
        Return type is framework-specific.
        """
        raise NotImplementedError

    @abstractmethod
    def postprocess(
        self,
        raw_output,
        preprocessed_input,
        array_frame: np.ndarray,
    ) -> np.ndarray:
        """
        Convert raw output to canonical detector format.

        MUST return:
            np.ndarray (N, 6)
            columns = [x1, y1, x2, y2, score, class_id]
        """
        raise NotImplementedError

    # -------------------------
    # Public API (PRIMARY)
    # -------------------------

    @abstractmethod
    def detect_to_track(self, array_frame: np.ndarray, **kwargs) -> np.ndarray:
        """
        Run full detection pipeline on a single frame.

        MUST return:
            np.ndarray (N, 6)
            [x1, y1, x2, y2, score, class_id]

        This output is tracker-ready (e.g., BoxMOT).
        """
        raise NotImplementedError



# # detectors/interface.py
# from abc import ABC, abstractmethod
# from typing import Any, Dict, List, Tuple, Optional, Protocol
# import numpy as np

# # Lightweight typed DTOs used by the interface (implementations can extend)
# class BoundingBox:
#     """Axis-aligned box in image coords (x1,y1,x2,y2)."""
#     x1: float
#     y1: float
#     x2: float
#     y2: float
#     score: float
#     class_id: int
#     extra: Dict[str, Any]  # for embeddings, masks, etc.

#     def __init__(self, x1, y1, x2, y2, score, class_id, extra=None):
#         self.x1, self.y1, self.x2, self.y2 = float(x1), float(y1), float(x2), float(y2)
#         self.score = float(score)
#         self.class_id = int(class_id)
#         self.extra = extra or {}

# class Detections:
#     """
#     Container for postprocessed detections:
#       - boxes: List[BoundingBox]
#       - frame_size: (H, W)
#       - timestamp: Optional[float]
#       - raw_meta: Optional[dict]
#     """
#     def __init__(self, boxes: List[BoundingBox], frame_size: Tuple[int,int], timestamp: Optional[float]=None, raw_meta: Optional[dict]=None):
#         self.boxes = boxes
#         self.frame_size = frame_size
#         self.timestamp = timestamp
#         self.raw_meta = raw_meta or {}

# class ModelInput:
#     """Abstract container for whatever the model requires (tensor, batch, dict...)."""
#     def __init__(self, data: Any, meta: Optional[dict]=None):
#         self.data = data
#         self.meta = meta or {}

# class RawOutput:
#     """Raw output from the inference engine (framework-specific)."""
#     def __init__(self, data: Any, meta: Optional[dict]=None):
#         self.data = data
#         self.meta = meta or {}

# class DetectorError(Exception):
#     """Raised for detector-specific failures (init, inference, postprocess)."""
#     pass

# class IDetector(ABC):
#     """
#     Detector interface (abstract) for object detection models.

#     Implementers must:
#       - initialize model resources in __init__
#       - provide preprocess/infer/postprocess pipeline
#       - provide to_tracker_input() to convert to tracker-ready list

#     Keep implementations single-responsibility: do NOT handle tracking or metric computation here.
#     """

#     @abstractmethod
#     def __init__(self, model_name: str , **kwargs: Dict[str,Any]):
#         """Initialize model. Config contains model_path, device, half, classes, batch_size, etc."""
#         raise NotImplementedError

#     @abstractmethod
#     def warmup(self, imgsz: Any = None) -> None:
#         """Optional: perform model warmup to reduce first-inference latency."""
#         raise NotImplementedError

#     @abstractmethod
#     def preprocess(self, array_frame: np.ndarray = None) -> ModelInput:
#         """
#         Convert raw frame (BGR uint8 image) into model input.
#         - MUST NOT modify original frame in-place (information hiding).
#         - Return ModelInput containing tensors/batches used by infer().
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def infer(self, preprocessed_input , **kwargs) -> RawOutput:
#         """
#         Run model forward pass and return framework-specific raw output.
#         - Keep inference deterministic and minimal side effects.
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def postprocess(self, raw_output, preprocessed_input , array_frame) -> Detections:
#         """
#         Convert raw model outputs to normalized Detections in image coordinates.
#         - Apply NMS, score thresholding, class filtering here.
#         - Return Detections with BoundingBox objects, frame_size, timestamp.
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def detect_to_track(self, ready_to_track_array) -> List[Dict[str,Any]]:
#         """
#         Prepare list of dicts for tracker consumption. Each dict usually contains:
#            {"bbox": [x1,y1,x2,y2], "score": float, "class_id": int, "feature": Optional[np.ndarray], "detection_id": Optional[int]}
#         - Feature/embedding must be present only if tracker requires it (e.g., reid).
#         """
#         raise NotImplementedError

#     @abstractmethod
#     def close(self) -> None:
#         """Release model and GPU resources."""
#         raise NotImplementedError

#     # optional: provide __doc__ describing supported models and how to implement a new one.
