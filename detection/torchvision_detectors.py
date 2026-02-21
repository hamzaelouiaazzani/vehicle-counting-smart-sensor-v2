from typing import Any, Dict
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image

from detection.interface import IDetector, DetectorError
import cv2


from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
    FCOS_ResNet50_FPN_Weights,
    SSD300_VGG16_Weights,
    SSDLite320_MobileNet_V3_Large_Weights,
)

_TV_MODELS = {
    "fasterrcnn_resnet50_fpn": torchvision.models.detection.fasterrcnn_resnet50_fpn,
    "fasterrcnn_resnet50_fpn_v2": torchvision.models.detection.fasterrcnn_resnet50_fpn_v2,
    "retinanet_resnet50_fpn": torchvision.models.detection.retinanet_resnet50_fpn,
    "ssd300_vgg16": torchvision.models.detection.ssd300_vgg16,
    "ssdlite320_mobilenet_v3_large": torchvision.models.detection.ssdlite320_mobilenet_v3_large,
    "fcos_resnet50_fpn": torchvision.models.detection.fcos_resnet50_fpn,
}


_TV_WEIGHTS = {
    "fasterrcnn_resnet50_fpn": FasterRCNN_ResNet50_FPN_Weights,
    "fasterrcnn_resnet50_fpn_v2": FasterRCNN_ResNet50_FPN_V2_Weights,
    "retinanet_resnet50_fpn": RetinaNet_ResNet50_FPN_Weights,
    "fcos_resnet50_fpn": FCOS_ResNet50_FPN_Weights,
    "ssd300_vgg16": SSD300_VGG16_Weights,
    "ssdlite320_mobilenet_v3_large": SSDLite320_MobileNet_V3_Large_Weights,
}



class TorchvisionDetector(IDetector):
    """
    Torchvision detector adapter.

    Canonical output:
        np.ndarray (N, 6)
        [x1, y1, x2, y2, score, class_id]
    """

    def __init__(self, model_name: str, **kwargs: Dict[str, Any]):
        if model_name not in _TV_MODELS:
            raise DetectorError(f"Unsupported torchvision model: {model_name}")

        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.score_thr = kwargs.get("conf", 0.25)

        weights = kwargs.get("weights", "DEFAULT")
        self.model = _TV_MODELS[model_name](weights=weights)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.ToTensor()

        weights_enum = _TV_WEIGHTS[model_name].DEFAULT
        self.class_names = weights_enum.meta["categories"]
        

    # -------------------------
    # Lifecycle
    # -------------------------

    def warmup(self, imgsz=None):
        # Optional dummy forward
        h, w = imgsz if imgsz is not None else (640, 640)
        dummy = torch.zeros((3, h, w), device=self.device)
        with torch.no_grad():
            _ = self.model([dummy])

    def close(self):
        if hasattr(self, "model"):
            del self.model
        torch.cuda.empty_cache()

    # -------------------------
    # Pipeline hooks
    # -------------------------

    def preprocess(self, array_frame: np.ndarray):
        # np.ndarray (H,W,3) -> torch.Tensor (3,H,W)
        tensor = self.transform(array_frame).to(self.device)
        return tensor

    def infer(self, preprocessed_input, **kwargs):
        with torch.no_grad():
            outputs = self.model([preprocessed_input])
        return outputs

    def postprocess(self, raw_output, preprocessed_input, array_frame):
        out = raw_output[0]

        boxes = out["boxes"].detach().cpu().numpy()
        scores = out["scores"].detach().cpu().numpy()
        labels = out["labels"].detach().cpu().numpy()

        keep = scores >= self.score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if boxes.size == 0:
            return np.zeros((0, 6), dtype=np.float32)

        return np.concatenate(
            [boxes, scores[:, None], labels[:, None]],
            axis=1,
        ).astype(np.float32)

    # -------------------------
    # Public API
    # -------------------------

    def detect_to_track(self, array_frame: np.ndarray, **kwargs) -> np.ndarray:
        pre = self.preprocess(array_frame)
        raw = self.infer(pre, **kwargs)
        return self.postprocess(raw, pre, array_frame)



    
    
    def plot(self, frame: np.ndarray, detections: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes, class labels, and confidence scores.
    
        Args:
            frame: np.ndarray (H,W,3) BGR or RGB
            detections: np.ndarray (N,6) [x1,y1,x2,y2,score,class_id]
    
        Returns:
            Annotated frame (np.ndarray)
        """
        if detections is None or len(detections) == 0:
            return frame.copy()
    
        img = frame.copy()
        h, w = img.shape[:2]
    
        for det in detections:
            x1, y1, x2, y2, score, class_id = det
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_id = int(class_id)
    
            label = (
                self.class_names[class_id]
                if class_id < len(self.class_names)
                else f"id:{class_id}"
            )
    
            text = f"{label} {score:.2f}"
    
            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
            # Label background
            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                img,
                (x1, y1 - th - 4),
                (x1 + tw + 2, y1),
                (0, 255, 0),
                -1,
            )
    
            # Label text
            cv2.putText(
                img,
                text,
                (x1 + 1, y1 - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    
        return img
