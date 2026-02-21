from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

from ultralytics import YOLO , RTDETR

from detection.interface import IDetector
from ultralytics.utils.checks import check_imgsz


class UltralyticsDetector(IDetector):
    def __init__(self, model_name: str, **kwargs):
        if "yolo" in model_name:
            detector_family = YOLO
        elif "rtdetr" in model_name:
            detector_family = RTDETR
        else:
            raise DetectorError(f"Unsupported model family for {model_name}")

        model = detector_family(Path(model_name))
        
        custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict", "rect": True, "verbose": True , "imgsz": [640]}
        args = {**model.overrides, **custom, **kwargs}
        
        self.predictor = model._smart_load("predictor")(overrides=args, _callbacks=model.callbacks)
        self.predictor.setup_model(model=model.model, verbose=args["verbose"])
        self.predictor.batch = [["_"]]
        self.result = None

        # Warmup Model
        imgsz = args["imgsz"]
        self.warmup(imgsz)
        print(f"Successfully {model_name} model is initialized and warmedup !")



        
    # --- Required abstract methods (stubs for now) ---
    def warmup(self, imgsz = [460,680]):
        imgsz = check_imgsz(imgsz , stride=self.predictor.model.stride , min_dim=2)
        self.predictor.imgsz = imgsz
        self.predictor.model.warmup(imgsz=(1, self.predictor.model.ch, *imgsz))
        
    def preprocess(self, array_frame): 
        # frame np.ndarray of shape (H, W, 3)
        preprocessed_input = self.predictor.preprocess([array_frame])
        return preprocessed_input

    def infer(self, preprocessed_input , **kwargs): 
        # image tensor of shape (1, 3, H, W)
        raw_output = self.predictor.inference(preprocessed_input, **kwargs)
        return raw_output

    def postprocess(self, raw_output, preprocessed_input , array_frame): 
        result_output = self.predictor.postprocess(raw_output, preprocessed_input, [array_frame])[0]
        self.result = result_output
        scores = result_output.boxes.conf.cpu().numpy()
        boxes = result_output.boxes.xyxy.cpu().numpy()
        labels = result_output.boxes.cls.cpu().numpy()
        ready_to_track_array = np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1)        
        return ready_to_track_array

    def detect_to_track(self, array_frame , **kwargs): 
        preprocessed_input = self.preprocess(array_frame)
        raw_output = self.infer(preprocessed_input , **kwargs)
        ready_to_track_array = self.postprocess(raw_output , preprocessed_input , array_frame)
        return ready_to_track_array

    def plot(self):
        det_array_plot = self.result.plot()
        return det_array_plot

    def show(self):
        det_array_plot = self.result.show()
        return det_array_plot

    def save(self):
        det_array_plot = self.result.save()
        return det_array_plot    
        
    def close(self): 
        if hasattr(self, "model") and self.model:
            del self.model
