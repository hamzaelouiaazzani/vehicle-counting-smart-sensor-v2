# 1. 🎥 Demo

[Click here to watch the demo](demo.mp4)

# 2. Vehicle Counting Repository Setup Guide

## Credits:

Before we begin, it's important to acknowledge the foundations upon which this project is built. Please take a moment to review the credits and licensing information below. Respecting the AGPL license is crucial before using, consuming, editing, or sharing this repository.

**Credits:**

* **Object Detection:** This project utilizes the [ultralytics](https://github.com/mikel-brostrom/ultralytics) repository for object detection, licensed under the AGPL license.
* **Object Tracking:** This project utilizes the [boxmot](https://github.com/mikel-brostrom/boxmot) repository for object tracking, also licensed under the AGPL license.

This notebook provides step-by-step instructions to set up and run the vehicle counting application on four different platforms: Google Colab, Jupyter Notebooks, and via Bash/Linux commands and in  NANO JETSON Kit.
## 1.1. Google Colab
**Note:** Please don't forget to set the runtime type to **GPU (T4)** in Colab for optimal performance.

### Setting the Runtime to GPU (T4):

1. After the notebook opens, navigate to the top menu and select **Runtime** > **Change runtime type**.
2. In the popup window, set **Hardware accelerator** to **GPU**.
3. If available, select **T4** as the GPU type.
4. Run the below code cells after setting .yaml config file with target rois, lines and params

## 1.2. Jupyter Notebooks

Note: In case you want to use the Geoforce GPU in your computer to accelerate to speed up processing, kindly install CUDA in your computer 
Follow these steps to set up and run the application in Jupyter Notebooks.

### Using a GeForce GPU in your computer for Accelerated Processing  

To utilize your computer's GeForce GPU to speed up processing, follow these steps:  

1. **Install CUDA:**  
   Download and install the CUDA toolkit compatible with your GPU from the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-12-4-0-download-archive).  

2. **Install PyTorch with GPU Support:**  
   Visit [PyTorch's Get Started Guide](https://pytorch.org/get-started/locally/) to install the appropriate version of PyTorch for your system with GPU (CUDA) support.

#### Notes  
- Ensure your GPU driver is up-to-date before installing CUDA.  
- Follow the instructions on the linked pages carefully to avoid compatibility issues.

### Step 1: Create Virtual Environment (Bash/Anaconda Prompt)
Open a Bash or Anaconda Prompt and run the following commands to create and activate a virtual environment named `vehicle_counter`:
```bash
conda create --name vehicle_counter python=3.8
conda activate vehicle_counter
```
This step assumes you have already installed Anaconda in your computer

> **Note:** You can neglect the above two instructions if you are NOT working in a virtual environment.

### Step 2: Clone the Vehicle Counting Repository
Clone the repository and ensure that vehicle-counting-smart-sensor-v2 is set as your working directory if you haven't done so already.

```python
!git clone https://github.com/hamzaelouiaazzani/vehicle-counting-smart-sensor-v2.git
```

### Step 3: Upgrade pip and Install Dependencies
Download/clone the repository and run the following cell to upgrade pip, setuptools, and wheel, and install the repository dependencies.

```python
!pip install --upgrade pip setuptools wheel
!pip install -e .
```
> **Note:** Once you run this cell, comment it out and do not run it again because the packages are already installed in your environment.

### Step 4: Verify Torch Installation
Run the following cell to confirm that NumPy version 1.24.4 is installed, PyTorch is set up, and CUDA is available for GPU support.

```python
import numpy as np
print("NumPy Version:", np.__version__)

from IPython.display import Image, clear_output  # to display images in Jupyter notebooks
clear_output()

import torch
print(f"Cuda availaibility: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```

### Step 5: set the .yaml config file with target rois, lines and params

### Step 6: Verify Torch Installation
Run the below engine code cell.


## 1.3. Running the Repository on NANO JETSON Kit

Follow these steps to set up and run the repository on a NANO JETSON Developer Kit with GPU support.

### Step 1: Download cuSPARSElt
Download cuSPARSElt to enable GPU usage with PyTorch and TorchVision:
- Visit the following link: [cuSPARSElt Downloads](https://developer.nvidia.com/cusparselt-downloads?target_os=Linux&target_arch=aarch64-jetson&Compilation=Native&Distribution=Ubuntu&target_version=22.04&target_type=deb_network)
- Ensure the file is compatible with your setup.
- Make sure the file `cuda-keyring_1.1-1_all.deb` (or a similar one) is successfully downloaded to your JETSON NANO Kit root: 

### Step 2: Create and Activate a Virtual Environment
1. Create a new virtual environment:
   ```bash
   python3 -m venv vehicle_counter
   ```
2. Activate the virtual environment:
   ```bash
   source vehicle_counter/bin/activate
   ```

### Step 3: Check the NVIDIA Forum
Refer to the following NVIDIA forum for compatible PyTorch and TorchVision Wheel files:
- [PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

### Step 4: Download Required Files
Download the following files:
- PyTorch Wheel: `torch-2.3.0-cp310-cp310-linux_aarch64.whl`
- TorchVision Wheel: `torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl`
- These files are compatible with JetPack 6.1, Ubuntu 22.04, CUDA 12.6, and Jetson Linux L4T R36.4.

If these versions do not work, refer back to the forum for other compatible versions. If no prebuilt versions are suitable, you can build PyTorch and TorchVision from source by using these repositories:
- [PyTorch Source Repository]( https://github.com/pytorch/pytorch)
- [TorchVision Source Repository](https://github.com/pytorch/vision)


### Step 5: Clone the Repository
Clone the repository to your Jetson Kit:
```bash
!git clone https://github.com/hamzaelouiaazzani/vehicle-counting-smart-sensor-v2.git
cd vehicle-counting-smart-sensor-v2
```

### Step 6: Install Dependencies
Upgrade pip, setuptools, and wheel, then install the repository dependencies:
```bash
pip install --upgrade pip setuptools wheel
pip install -e .
cd ..
```

### Step 7: Verify Installation
Run the following commands to confirm installation:
```bash
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
python3 -c "import torchvision; print('TorchVision version:', torchvision.__version__)"
```
The installation of PyTorch and TorchVision using the previous instructions enables only CPU functionality on your NANO Jetson Kit, not GPU support. To confirm this, run the following command. If GPU is not enabled, it will display “CUDA available: False”.

### Step 8: Enable GPU Support
Install the appropriate PyTorch and TorchVision versions for GPU support:
```bash
pip3 install torch-2.3.0-cp310-cp310-linux_aarch64.whl
pip3 install torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl
pip install numpy==1.24.4
```
### Step 9: set the .yaml config file with target rois, lines and params

# 3. Vehicle Counting — Interactive Demo & Visualiser

This notebook provides an interactive demo and runner for the **Vehicle Counting Smart Sensor** pipeline (BoXMOT + Ultralytics detector). It preserves all original code cells unchanged — the code below is the original work; these top cells are documentation only so you can use this notebook as the project README.

Use this notebook to:
- Visualise and set geometric shapes (ROIs, lines, polygons) using the interactive selectors.
- Run the end-to-end pipeline: frame grabbing → detection (Ultralytics) → tracking (BoXMOT) → counting → video output.
- Produce `output_counting.mp4` containing the visualised counting demo.

Keep the code cells unchanged when running to ensure reproducibility.


## Quick Start (short)

1. Activate your conda environment (example name `smart_sensor`):
   ```bash
   conda activate smart_sensor
   ```
2. Ensure required Python packages are installed (see detailed list below).
3. Edit the `source` path in the notebook to point to your video file (example: `C:\Users\hamza\Datasets\TrafficDatasets\IMAROC_2\kech37.mp4`).
4. Run the notebook cells from top to bottom (or run the main execution block that starts the pipeline).
5. The demo writes `output_counting.mp4` in the working directory.

Press `q` in the OpenCV window to stop the run safely (the loop polls `cv2.waitKey(1)` and honors 'q').


## Environment & Dependencies (recommended)

This notebook was developed and tested with the following environment (as reported in outputs):
- Python 3.10
- Ultralytics `8.4.5` (YOLO11n model example)
- PyTorch (as reported by `ultralytics` output; e.g. `torch-2.5.1+cu121`)
- CUDA-enabled GPU recommended for real-time performance (example used NVIDIA GeForce RTX 3050)

Suggested install (example using pip in the conda env):
```bash
pip install ultralytics==8.4.5 boxmot opencv-python torch torchvision numpy pandas tqdm
# plus any project-specific dependencies found in requirements.txt
```

If you use `conda`, make sure CUDA toolkit and compatible `torch` build are installed for GPU acceleration.


## Inputs & Outputs

- **Input video file**: set `source` variable to your video path (e.g. `your_traffic_video.mp4`).
- **Output video**: `output_counting.mp4` (written using `cv2.VideoWriter`).
- **Interactive selectors**: polygon, rectangle, lines, OBB, points — use the selectors in `utils.shape_setter` to create ROIs/lines.

Notes:
- The notebook contains an interactive selection snippet (using OpenCV GUI). If you run this on a remote or headless server, those interactive selectors will not function — use a local GUI session.
- Ensure the `source` path is readable and `cv2.VideoCapture(source)` works before selecting shapes.


## Cell-by-cell explanation (map to original cells)

I kept the original code cells unchanged. Here is a short description for each block so you can use this notebook as documentation/README:

1. `pwd` — convenience cell to check the current working directory (original cell preserved).
2. `cd ..` — example command to change directory (original cell preserved).
3. Interactive selector script — demonstrates how to open the first frame and run various selectors (PolygonSelector is enabled in the example). Use this to define ROIs, lines, rectangles, OBBs, or single points interactively.
4. `### Imprt packages` — header cell (markdown) in the original notebook indicating the start of imports.
5. Imports and model initialisation — imports OpenCV, NumPy, Torch and project modules; initialises `UltralyticsDetector("yolo11n.pt", conf=0.50)`, sets COCO vehicle classes and device.
6. A small introspection cell that prints model predictor args (confidence and image size).
7. `TorchvisionDetector` snippet — shows how to initialize an alternative detector (Faster R-CNN) if desired.
8. Ordered counters helper function and counting configs — builds `CountVisualizer`, loads counting areas via `CountingConfigLoader`, orders counters for rendering.
9. Profilers setup — initialises profiling helpers for different pipeline stages (inference, pre/post, tracking, counting, etc.).
10. Tracker selection and initialisation — example shows `tracking_method = "bytetrack"` and creates a `Tracker`.
11. Main pipeline runner — this is the main loop: opens `FrameGrabber`, initialises `VideoWriter`, iterates over frames, runs `my_model.detect_to_track()`, updates the tracker, counts with configured counters, renders the visualisation using `CountVisualizer`, writes frames to `output_counting.mp4` and shows them in an OpenCV window. The loop is interruptible with the `q` key and properly releases resources in the `finally` block.
12. Final small introspection cells that display counts (`g_count`, etc.).

If you want me to produce a dedicated `README.md` (Markdown file) extracted from these explanations, tell me and I will generate it as a separate file too.


## Important Tips & Troubleshooting

- **Interactive GUI:** The selectors and `cv2.imshow` require a desktop session. They will not work on headless servers unless you use a virtual display (e.g., Xvfb on Linux).
- **VideoWriter:** The notebook uses `mp4v` fourcc. If output fails to open, confirm codecs are available on your platform. The code asserts `video_writer.isOpened()`.
- **Stopping:** Press **`q`** in the OpenCV window to stop early. The `finally` block ensures `frame_grabber.release()`, `video_writer.release()` and `cv2.destroyAllWindows()` are called.
- **Paths:** Use absolute paths for `source` to avoid accidental wrong working directory problems.
- **GPU:** For real-time performance use CUDA-enabled PyTorch and Ultralytics. Confirm `my_model.predictor.device` shows `cuda`.
- **Versioning:** The notebook was run with `Ultralytics 8.4.5`. If you use another version, some APIs or model names might differ.

If you encounter errors when running the pipeline, copy the full traceback here and I will help you debug quickly.


## License & Attribution

Keep the notebook's original license headers (if any) and the repository LICENSE file. This documentation is intended to accompany the project and does not change original code authorship.

— End of documentation header — original code cells follow unchanged below —



```python
pwd
```


```python
cd ..
```

### Use the following script to visulaize and set the geometric shapes (rois polygons, lines,...) you want to set in the con


```python
from utils.shape_setter import PointSelector , LineSelector , TwoLineSelector , PolygonSelector , RectangleSelector , OBBSelector
import cv2
import numpy as np
source = r"you_target_video.mp4" 

stride = 1
stride_method = "periodic_stride"             # "burst_stride", "periodic_stride", "random_sampling"

cap = cv2.VideoCapture(source)
ok, first_frame = cap.read()
cap.release()

if not ok or first_frame is None:
    print("Failed to read example frame; please provide a valid path ('kech.mp4' used in example).")
else:
    # # line
    # line_sel = LineSelector(max_display_size=900, auto_confirm=True, preview_wait_secs=None)
    # line = line_sel.select_line(first_frame)
    # print("Selected line:", line)

    # # two lines
    # two_sel = TwoLineSelector(max_display_size=900, auto_confirm=True, preview_wait_secs=None)
    # two = two_sel.select_two_lines(first_frame)
    # print("Selected two lines:", two)

    #polygon
    poly_sel = PolygonSelector(max_display_size=900, min_points=4, auto_close_on_click_near_first=True,
                               close_pixel_radius=12, preview_wait_secs=None)
    poly = poly_sel.select_polygon(first_frame)
    print("Selected polygon:", poly)

    # # rectangle
    # rect_sel = RectangleSelector(max_display_size=900, auto_confirm=True, preview_wait_secs=None)
    # rect = rect_sel.select_rectangle(first_frame)
    # if rect is None:
    #     print("Cancelled")
    # else:
    #     print("Selected rectangle:", rect)


    # # # obb£
    # obb_selector = OBBSelector()
    # obb_points = obb_selector.select_obb(first_frame)
    # if obb_points is not None:
    #     print("Selected OBB points:", obb_points)
    # else:
    #     print("Selection cancelled")


    # selector = PointSelector()
    # pt = selector.select_point(first_frame)
    # print("Selected point:", pt)

```

### Imprt packages


```python
import time

#####################################################################################################################################

import cv2
import numpy as np
import torch

#####################################################################################################################################

from framegrabber.frame_grabber import FrameGrabber

from detection.ultralytics_detectors import UltralyticsDetector

from tracking.track import Tracker

from counting.count_config_loader import CountingConfigLoader
from counting.count_visualizer import CountVisualizer

#####################################################################################################################################

# Check the ultralytics repo/website/blogs to see all availaible detectors: just put the name here to use it for vehicle counting
my_model = UltralyticsDetector("yolo11n.pt" , conf=0.50)         # rtdetr-l.pt  yolo11n.pt yolo26n.pt yolo11n_finetuned

# target classes to be filtered later (ex: vehicles with 4 wheels)
coco_vehicles = [1, 2, 3, 5, 7]                              # Bicycle, Car, Motorcycle, Bus and Truck
vehicles_4_wheels = [2, 5, 7]                                # Car, Bus and Truck
device = my_model.predictor.device
```

    Ultralytics 8.4.5  Python-3.10.18 torch-2.5.1+cu121 CUDA:0 (NVIDIA GeForce RTX 3050 6GB Laptop GPU, 6144MiB)
    YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs
    Successfully yolo11n.pt model is initialized and warmedup !
    


```python
my_model.predictor.args.conf , my_model.predictor.args.imgsz
```




    (0.5, [640])




```python
# In case you want to use torchivision detectors
from detection.torchvision_detectors import TorchvisionDetector

det = TorchvisionDetector(
    "fasterrcnn_resnet50_fpn_v2",
    conf=0.6,
    device="cuda"
)

# detections = det.detect_to_track(frame.data)
```

* yolo26n: YOLO26n summary (fused): 122 layers, 2,408,932 parameters, 0 gradients, 5.4 GFLOPs
* YOLOv8n summary (fused): 72 layers, 3,151,904 parameters, 0 gradients, 8.7 GFLOPs
* YOLO11n summary (fused): 100 layers, 2,616,248 parameters, 0 gradients, 6.5 GFLOPs


```python
def ordered_counters(*counters):
    """
    Accepts:
      ([roi1, roi2], global_ctr)
      or
      ([roi1, roi2, global_ctr],)

    Returns:
      flat, ordered list of counter objects
    """

    # ---- flatten ----
    flat = []
    for c in counters:
        if isinstance(c, (list, tuple)):
            flat.extend(c)
        else:
            flat.append(c)

    # ---- semantic order ----
    def key(ctr):
        info = ctr.get_area_info()
        # ROI / line first, global last
        if info.get("polygon") is not None or info.get("line") is not None:
            return (0, info.get("name", ""))
        return (1, info.get("name", ""))

    return sorted(flat, key=key)


# You can set region of interest polygons, lines of counting, paramters of counting, logic of counting and other hyperparameters from the .yaml in the config folder
count_vis = CountVisualizer(
    show_legend=False,   # hide class legend
    show_summary=True    # keep total summary box
)

# Load counting params as set in the counting yaml config file.
counter_load = CountingConfigLoader(default_classes = vehicles_4_wheels)

# set counters
counters = counter_load.load_counting_areas()

# order counters for visualisation
counters = ordered_counters(*counters)
counters
```




    [<counting.count.CountingROIWithIds at 0x1cf3f276d10>,
     <counting.count.CountingROIWithIds at 0x1cf3f2f5f90>,
     <counting.count.CountingGlobalAreaWithIds at 0x1cf3f18bac0>]




```python
# Profiles to diagnonise multiple stages latencies
from utils.profilers import Profile

device = my_model.predictor.device
inf_profile = Profile(device=device)
pre_profile = Profile(device=device)
post_profile = Profile(device=device)
track_profile = Profile()
count_profile = Profile()
grabber_profile = Profile()
plot_profile = Profile()
```


```python
# Availaible trackers: ocsort, bytetrack, strongsort, deepocsort, hybridsort, boosttrack, botsort
tracking_method = "bytetrack"
tracker = Tracker(tracking_method)
```


```python
source = r"your_traffic_video.mp4"                   # 0 "kech.mp4" , "vid1.mp4"
source = r"C:\Users\hamza\Datasets\TrafficDatasets\IMAROC_2\kech37.mp4"
# --- Video writer setup ---
output_path = "output_counting.mp4"

fps = 30

# Get frame size (wait until first frame if needed)
ret, test_cap = cv2.VideoCapture(source).read()
h, w = test_cap.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # widely supported
video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

assert video_writer.isOpened(), "Failed to open VideoWriter"



stride = 2
stride_method = "periodic_stride"             # "burst_stride", "periodic_stride", "random_sampling"

frame_grabber = FrameGrabber(source, stride=stride, stride_method=stride_method)


if not frame_grabber.open():
    raise RuntimeError("Failed to open source")
# ensure window exists (main thread)
cv2.namedWindow('BoXMOT + ultralytics', cv2.WINDOW_NORMAL)
if frame_grabber._grabber_mode=="queue":
# start producer
    frame_grabber.start()

try:
    with torch.inference_mode():
        while True:
            with grabber_profile:
                # try to get a frame but don't block forever
                frame = frame_grabber.get_frame(timeout=0.1)  # <-- short timeout keeps loop responsive
                    
            if frame is not None:
                print(f"frame_grabber index: {frame.read_idx}")

                with inf_profile:
                    ready_to_track_array = my_model.detect_to_track(frame.data)
   
                with track_profile:
                    res = tracker.update(ready_to_track_array , frame.data)
                    # print(f"tracking array: {res} for counter: {counters[2]}")

                ## Plot detection for ultralytics models
                # det_array_plot = my_model.plot()
                ## Plot detection for Torchvision models
                # det_array_plot = det.plot(frame.data, detections)

                # Plot tracks
                track_array_plot = tracker.tracker.plot_results(frame.data, show_trajectories=True)

                with count_profile:
                    g_count = counters[2].count(res)
                    roi1_count = counters[0].count(res)
                    roi2_count = counters[1].count(res)
                    
                with plot_profile:
                    count_plot = count_vis.render(track_array_plot, *counters)


                # --- WRITE FRAME TO VIDEO ---
                video_writer.write(count_plot)
                    
                    
                # mark processed & show
                frame_grabber.mark_processed(frame)

                cv2.imshow('BoXMOT + ultralytics', count_plot)
                
            else:
                # no frame this iteration (timeout), you may choose to display a placeholder
                # or simply continue — but still poll for key events below
                pass

            # ALWAYS poll keyboard events so 'q' is detected even when no frame was available
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                # stop producer and break loop
                if frame_grabber._grabber_mode=="queue":
                    frame_grabber.stop(wait=True)
                break
    
                # optional: also break when producer finished (sentinel)
                if frame_grabber._grabber_mode=="queue":
                    if frame is None and frame_grabber._stop_event.is_set():
                        break
finally:
    frame_grabber.release()
    video_writer.release()   # <-- ADD THIS
    cv2.destroyAllWindows()
```

    frame_grabber index: 0
    frame_grabber index: 2
    frame_grabber index: 4
    frame_grabber index: 6
    frame_grabber index: 8
    frame_grabber index: 10
    frame_grabber index: 12
    frame_grabber index: 14
    frame_grabber index: 16
    frame_grabber index: 18
    frame_grabber index: 20
    frame_grabber index: 22
    frame_grabber index: 24
    frame_grabber index: 26
    frame_grabber index: 28
    frame_grabber index: 30
    frame_grabber index: 32
    frame_grabber index: 34
    frame_grabber index: 36
    frame_grabber index: 38
    frame_grabber index: 40
    frame_grabber index: 42
    frame_grabber index: 44
    frame_grabber index: 46
    frame_grabber index: 48
    frame_grabber index: 50
    frame_grabber index: 52
    frame_grabber index: 54
    frame_grabber index: 56
    frame_grabber index: 58
    frame_grabber index: 60
    frame_grabber index: 62
    frame_grabber index: 64
    frame_grabber index: 66
    frame_grabber index: 68
    frame_grabber index: 70
    frame_grabber index: 72
    frame_grabber index: 74
    frame_grabber index: 76
    frame_grabber index: 78
    frame_grabber index: 80
    frame_grabber index: 82
    frame_grabber index: 84
    frame_grabber index: 86
    frame_grabber index: 88
    frame_grabber index: 90
    frame_grabber index: 92
    frame_grabber index: 94
    frame_grabber index: 96
    frame_grabber index: 98
    frame_grabber index: 100
    frame_grabber index: 102
    frame_grabber index: 104
    frame_grabber index: 106
    frame_grabber index: 108
    frame_grabber index: 110
    frame_grabber index: 112
    frame_grabber index: 114
    frame_grabber index: 116
    frame_grabber index: 118
    frame_grabber index: 120
    frame_grabber index: 122
    frame_grabber index: 124
    frame_grabber index: 126
    frame_grabber index: 128
    frame_grabber index: 130
    frame_grabber index: 132
    frame_grabber index: 134
    frame_grabber index: 136
    frame_grabber index: 138
    frame_grabber index: 140
    frame_grabber index: 142
    frame_grabber index: 144
    frame_grabber index: 146
    frame_grabber index: 148
    frame_grabber index: 150
    frame_grabber index: 152
    frame_grabber index: 154
    frame_grabber index: 156
    frame_grabber index: 158
    frame_grabber index: 160
    frame_grabber index: 162
    frame_grabber index: 164
    frame_grabber index: 166
    frame_grabber index: 168
    frame_grabber index: 170
    frame_grabber index: 172
    frame_grabber index: 174
    frame_grabber index: 176
    frame_grabber index: 178
    frame_grabber index: 180
    frame_grabber index: 182
    frame_grabber index: 184
    frame_grabber index: 186
    frame_grabber index: 188
    frame_grabber index: 190
    frame_grabber index: 192
    frame_grabber index: 194
    frame_grabber index: 196
    frame_grabber index: 198
    frame_grabber index: 200
    frame_grabber index: 202
    frame_grabber index: 204
    frame_grabber index: 206
    frame_grabber index: 208
    frame_grabber index: 210
    frame_grabber index: 212
    frame_grabber index: 214
    frame_grabber index: 216
    frame_grabber index: 218
    frame_grabber index: 220
    frame_grabber index: 222
    frame_grabber index: 224
    frame_grabber index: 226
    frame_grabber index: 228
    frame_grabber index: 230
    frame_grabber index: 232
    frame_grabber index: 234
    frame_grabber index: 236
    frame_grabber index: 238
    frame_grabber index: 240
    frame_grabber index: 242
    frame_grabber index: 244
    frame_grabber index: 246
    frame_grabber index: 248
    frame_grabber index: 250
    frame_grabber index: 252
    frame_grabber index: 254
    frame_grabber index: 256
    frame_grabber index: 258
    frame_grabber index: 260
    frame_grabber index: 262
    frame_grabber index: 264
    frame_grabber index: 266
    frame_grabber index: 268
    frame_grabber index: 270
    frame_grabber index: 272
    frame_grabber index: 274
    frame_grabber index: 276
    frame_grabber index: 278
    frame_grabber index: 280
    frame_grabber index: 282
    frame_grabber index: 284
    frame_grabber index: 286
    frame_grabber index: 288
    frame_grabber index: 290
    frame_grabber index: 292
    frame_grabber index: 294
    frame_grabber index: 296
    frame_grabber index: 298
    frame_grabber index: 300
    frame_grabber index: 302
    frame_grabber index: 304
    frame_grabber index: 306
    frame_grabber index: 308
    frame_grabber index: 310
    frame_grabber index: 312
    frame_grabber index: 314
    frame_grabber index: 316
    frame_grabber index: 318
    frame_grabber index: 320
    frame_grabber index: 322
    frame_grabber index: 324
    frame_grabber index: 326
    frame_grabber index: 328
    frame_grabber index: 330
    frame_grabber index: 332
    frame_grabber index: 334
    frame_grabber index: 336
    frame_grabber index: 338
    frame_grabber index: 340
    frame_grabber index: 342
    frame_grabber index: 344
    frame_grabber index: 346
    frame_grabber index: 348
    frame_grabber index: 350
    frame_grabber index: 352
    frame_grabber index: 354
    frame_grabber index: 356
    frame_grabber index: 358
    frame_grabber index: 360
    frame_grabber index: 362
    frame_grabber index: 364
    frame_grabber index: 366
    frame_grabber index: 368
    frame_grabber index: 370
    frame_grabber index: 372
    frame_grabber index: 374
    frame_grabber index: 376
    frame_grabber index: 378
    frame_grabber index: 380
    frame_grabber index: 382
    frame_grabber index: 384
    frame_grabber index: 386
    frame_grabber index: 388
    frame_grabber index: 390
    frame_grabber index: 392
    frame_grabber index: 394
    frame_grabber index: 396
    frame_grabber index: 398
    frame_grabber index: 400
    frame_grabber index: 402
    frame_grabber index: 404
    frame_grabber index: 406
    frame_grabber index: 408
    frame_grabber index: 410
    frame_grabber index: 412
    frame_grabber index: 414
    frame_grabber index: 416
    frame_grabber index: 418
    frame_grabber index: 420
    frame_grabber index: 422
    frame_grabber index: 424
    frame_grabber index: 426
    frame_grabber index: 428
    frame_grabber index: 430
    frame_grabber index: 432
    frame_grabber index: 434
    frame_grabber index: 436
    frame_grabber index: 438
    frame_grabber index: 440
    frame_grabber index: 442
    frame_grabber index: 444
    frame_grabber index: 446
    frame_grabber index: 448
    frame_grabber index: 450
    frame_grabber index: 452
    frame_grabber index: 454
    frame_grabber index: 456
    frame_grabber index: 458
    frame_grabber index: 460
    frame_grabber index: 462
    frame_grabber index: 464
    frame_grabber index: 466
    frame_grabber index: 468
    frame_grabber index: 470
    frame_grabber index: 472
    frame_grabber index: 474
    frame_grabber index: 476
    frame_grabber index: 478
    frame_grabber index: 480
    frame_grabber index: 482
    frame_grabber index: 484
    frame_grabber index: 486
    frame_grabber index: 488
    frame_grabber index: 490
    frame_grabber index: 492
    frame_grabber index: 494
    frame_grabber index: 496
    frame_grabber index: 498
    frame_grabber index: 500
    frame_grabber index: 502
    frame_grabber index: 504
    frame_grabber index: 506
    frame_grabber index: 508
    frame_grabber index: 510
    frame_grabber index: 512
    frame_grabber index: 514
    frame_grabber index: 516
    frame_grabber index: 518
    frame_grabber index: 520
    frame_grabber index: 522
    frame_grabber index: 524
    frame_grabber index: 526
    frame_grabber index: 528
    frame_grabber index: 530
    frame_grabber index: 532
    frame_grabber index: 534
    frame_grabber index: 536
    frame_grabber index: 538
    frame_grabber index: 540
    frame_grabber index: 542
    frame_grabber index: 544
    frame_grabber index: 546
    frame_grabber index: 548
    


```python
g_count , roi1_count , roi2_count
```


```python
g_count.total_count , g_count.counts_by_class
```
