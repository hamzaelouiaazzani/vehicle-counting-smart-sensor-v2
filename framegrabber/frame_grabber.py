import threading
import time
import random
import queue
from dataclasses import dataclass
from typing import Optional, Union, List, Tuple, Iterator
from pathlib import Path

import numpy as np
import cv2


@dataclass
class Frame:
    """Container for a captured frame and its metadata."""
    data: any                 # numpy.ndarray
    timestamp: float
    read_idx: int             # index at capture time
    processed_idx: Optional[int]  # index assigned when processed
    width: Optional[int]
    height: Optional[int]
    frame_rate: Optional[float]  # None if unknown/unreliable


class FrameGrabber:
    """Video/camera frame grabber with support for stride sampling and producer-consumer queue."""

    def __init__(self,
                 source: Union[str, int],
                 stride: Union[int, float] = 1,
                 stride_method: str = "periodic_stride",
                 window_size: Optional[int] = None,
                 grabber_mode: str = "latest",
                 queue_maxsize: int = 4,
                 fallback_fps: Optional[float] = None,
                 allow_dynamic_resolution: bool = False):
        """
        Initialize the frame grabber.

        Args:
            source: Path to video file (str) or camera index (int).
            stride: Frame sampling interval (int) or probability (float in [0, 1]).
            stride_method: Strategy for stride ("periodic_stride", "burst_stride").
            window_size: Window size for float stride sampling.
            grabber_mode: "latest" (read synchronously) or "queue" (producer-consumer).
            queue_maxsize: Max queue size for buffered mode.
            fallback_fps: Default FPS if source reports none.
            allow_dynamic_resolution: Whether to update resolution dynamically.

        Raises:
            TypeError: If stride is not int or float.
            ValueError: If arguments are inconsistent or invalid.
        """
        
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None

        # counters
        self._capture_idx = 0   # increments for every captured frame
        self._processed_count = 0  # increments a frame is processed

        # meta
        self.fps: Optional[float] = fallback_fps
        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fallback_fps = fallback_fps
        self.allow_dynamic_resolution = allow_dynamic_resolution

        # stride setup
        if not isinstance(stride, (int, float)):
            raise TypeError("stride must be int or float")

        self.stride = stride
        self.window_size = window_size
        self.stride_method = stride_method

        if isinstance(self.stride, float):
            if not (0.0 <= self.stride <= 1.0):
                raise ValueError("Random stride (float) must be between 0 and 1.")
            if not (isinstance(self.window_size, int) and self.window_size > 0):
                raise ValueError("window_size must be a positive int when using float stride.")
            keep_n = int(round(self.stride * self.window_size))
            keep_n = max(1, min(self.window_size, keep_n))
            self._keep_indices = set(random.sample(range(self.window_size), keep_n))
            self.stride_method = "random_sampling"
        else:
            if int(self.stride) < 1:
                raise ValueError("Integer stride must be >= 1")
            self.stride = int(self.stride)
            if self.stride_method not in ("periodic_stride", "burst_stride"):
                self.stride_method = "periodic_stride"
            self.window_size = None

        # mode setup
        if grabber_mode not in ("latest", "queue"):
            raise ValueError("grabber_mode must be 'latest' or 'queue'")
        self._grabber_mode = grabber_mode

        if isinstance(self.source, str):
            # video files don't make sense with queue threading here
            self._grabber_mode = "latest"
    
        if self._grabber_mode == "queue":
            self.queue_maxsize = max(0, int(queue_maxsize or 0))
            self.q: queue.Queue = queue.Queue(maxsize=self.queue_maxsize) if self.queue_maxsize > 0 else queue.Queue()
            self._capture_thread: Optional[threading.Thread] = None
            self._stop_event = threading.Event()
            self._is_open = False

        if isinstance(self.source, int) and self._grabber_mode == "latest":
            self.stride_method = None
            self.stride = None

        #choose stride policy    
        self.set_stride_policy()
        
        # choose appropriate get_frame implementation
        self._select_get_frame()

    def open(self) -> bool:
        """
        Open the video source.

        Returns:
            True if successfully opened, False otherwise.
        """
        
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            return False

        fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        self.fps = fps if fps > 0 else self.fallback_fps
        self.width = w if w > 0 else None
        self.height = h if h > 0 else None

        if self._grabber_mode == "queue":
            self._is_open = True

        return True

    def start(self) -> None:
        """
        Start the background capture thread (queue mode only).

        Raises:
            RuntimeError: If called before open().
        """
        
        if not self._is_open or self.cap is None:
            raise RuntimeError("open() must be called successfully before start().")
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        self._stop_event.clear()
        self._capture_thread = threading.Thread(target=self._capture_loop_queue, daemon=True)
        self._capture_thread.start()

    def set_stride_policy(self) -> None:
        """
        Resolve the stride policy into a function (_should_process_fn).
        """
        if self.stride is None:
            self._should_process_fn = None
        elif isinstance(self.stride, int):
            if self.stride_method == "periodic_stride":
                self._should_process_fn = self._should_process_periodic
            elif self.stride_method == "burst_stride":
                self._should_process_fn = self._should_process_burst
            else:
                self._should_process_fn = None
        elif isinstance(self.stride, float):
            self._should_process_fn = self._should_process_fractional
        else:
            raise TypeError("stride must be int, float or None")

    
    def _should_process(self, capture_idx: int) -> bool:
        """
        Decide whether to keep a frame based on stride strategy.

        Args:
            capture_idx: Global index of the frame.

        Returns:
            True if frame should be kept, False otherwise.
        """
        return self._should_process_fn(capture_idx)

    def _should_process_periodic(self, capture_idx: int) -> bool:
        return (capture_idx % self.stride) == 0
    
    def _should_process_burst(self, capture_idx: int) -> bool:
        return (capture_idx % self.stride) != 0
    
    def _should_process_fractional(self, capture_idx: int) -> bool:
        idx_in_window = capture_idx % self.window_size
        if idx_in_window == 0:
            keep_n = int(round(self.stride * self.window_size))
            keep_n = max(1, min(self.window_size, keep_n))
            self._keep_indices = set(random.sample(range(self.window_size), keep_n))
        return idx_in_window in self._keep_indices
    

    def _capture_loop_queue(self) -> None:
        """
        Producer thread loop for queue mode.
        Continuously reads frames, applies stride, enqueues, and drops oldest when full.
        """
        
        while not self._stop_event.is_set():
            ret, img = self.cap.read()
            if not ret:
                break

            idx = self._capture_idx
            self._capture_idx += 1

            if self._should_process(idx):
                timestamp = time.time()
                if self.allow_dynamic_resolution or self.width is None or self.height is None:
                    h, w = img.shape[:2]
                    self.width, self.height = w, h

                frm = Frame(
                    data=img,
                    timestamp=timestamp,
                    read_idx=idx,
                    processed_idx=None,
                    width=self.width,
                    height=self.height,
                    frame_rate=self.fps
                )

                try:
                    self.q.put_nowait(frm)
                except queue.Full:
                    try:
                        _ = self.q.get_nowait()
                        self.q.put_nowait(frm)
                    except queue.Full:
                        pass

        try:
            self.q.put_nowait(None)
        except queue.Full:
            try:
                _ = self.q.get_nowait()
                self.q.put_nowait(None)
            except Exception:
                pass

    def mark_processed(self, frame: Frame) -> None:
        """
        Mark a frame as processed and update counters.

        Args:
            frame: Frame to update.
        """
        
        frame.processed_idx = self._processed_count
        self._processed_count += 1
        

    def stop(self, wait: bool = True) -> None:
        """
        Stop the capture thread.

        Args:
            wait: If True, wait for the thread to join.
        """
        
        self._stop_event.set()
        if self._capture_thread is not None and wait:
            self._capture_thread.join(timeout=2.0)

    
    def _select_get_frame(self) -> None:
        """
        Select the correct get_frame implementation
        depending on mode and source type.
        """
        
        if self._grabber_mode == "queue" and isinstance(self.source, int):
            self.get_frame = self._get_frame_queue
        elif isinstance(self.source, str):
            self.get_frame = self._get_frame_video
        elif self._grabber_mode == "latest" and isinstance(self.source, int):
            self.get_frame = self._get_frame_camera_latest
        else:
            raise ValueError(f"Unsupported grabber_mode/source combination: {self._grabber_mode}, {type(self.source)}")

    def _get_frame_queue(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """
        Get a frame from the producer queue.

        Args:
            timeout: Seconds to wait for a frame, or None for blocking.

        Returns:
            A Frame object, or None if queue is empty or producer ended.
        """
        
        try:
            item = self.q.get(timeout=timeout)
        except queue.Empty:
            return None
        return item

    def _get_frame_video(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """
        Get a frame directly from a video file.

        Args:
            timeout: Ignored (synchronous mode).

        Returns:
            A Frame object, or None if end of video or sampling skipped.
        """
        
        if not self.cap:
            return None
        ret, img = self.cap.read()
        if not ret:
            return None

        idx = self._capture_idx
        self._capture_idx += 1

        if not self._should_process(idx):
            return None

        timestamp = time.time()
        if self.allow_dynamic_resolution or self.width is None or self.height is None:
            h, w = img.shape[:2]
            self.width, self.height = w, h

        return Frame(
            data=img,
            timestamp=timestamp,
            read_idx=idx,
            processed_idx=None,
            width=self.width,
            height=self.height,
            frame_rate=self.fps
        )

    def _get_frame_camera_latest(self, timeout: Optional[float] = None) -> Optional[Frame]:
        """
        Get the latest frame directly from a live camera.

        Args:
            timeout: Ignored (synchronous mode).

        Returns:
            A Frame object, or None if no frame available.
        """
        
        if not self.cap:
            return None
        ret, img = self.cap.read()
        if not ret:
            return None

        idx = self._capture_idx
        self._capture_idx += 1

        timestamp = time.time()
        if self.allow_dynamic_resolution or self.width is None or self.height is None:
            h, w = img.shape[:2]
            self.width, self.height = w, h

        return Frame(
            data=img,
            timestamp=timestamp,
            read_idx=idx,
            processed_idx=None,
            width=self.width,
            height=self.height,
            frame_rate=self.fps
        )

    def release(self) -> None:
        """
        Release the video source and stop capture if running.
        """
        
        if self._grabber_mode == "queue":
            self.stop(wait=True)
            self._is_open = False

        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None










class UADETRACFrameGrabber:
    IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    def __init__(self,
                 base_path: Path = Path(r"C:\Users\hamza\Datasets\TrafficDatasets\UA_DETRAC_Original\DETRAC-Images\DETRAC-Images")):
        self.base_path = Path(base_path)
        self.sequence_dir: Optional[Path] = None
        self.frame_paths: List[Path] = []
        self.frame_count: int = 0
        self.frame_idx: int = 0   # current index (1-based); 0 means not started
        self._last_frame: Optional[np.ndarray] = None

    # -----------------------
    # Helpers
    # -----------------------
    @staticmethod
    def _num_key(p: Path) -> int:
        """Extract integer from filename for natural ordering; fallback to 0."""
        s = ''.join(re.findall(r'\d+', p.stem))
        try:
            return int(s) if s else 0
        except Exception:
            return 0

    def _collect_frames(self, seq_dir: Path) -> List[Path]:
        paths = [p for p in seq_dir.iterdir() if p.suffix.lower() in self.IMAGE_EXTS and p.is_file()]
        # try numeric-aware sort, fallback to lexicographic
        if paths:
            try:
                paths = sorted(paths, key=lambda p: (self._num_key(p), p.name))
            except Exception:
                paths = sorted(paths)
        return paths

    # -----------------------
    # Sequence open / discovery
    # -----------------------
    def list_sequences(self) -> List[Path]:
        """Return sorted sequence directories directly under base_path."""
        if not self.base_path.exists():
            return []
        dirs = sorted([p for p in self.base_path.iterdir() if p.is_dir()])
        return dirs

    def open_sequence(self, sequence: str) -> None:
        """Open sequence by directory name (exact match). Resets internal state."""
        cand = self.base_path / sequence
        if not cand.exists() or not cand.is_dir():
            # fallback: try any dir under base whose stem matches sequence
            matches = [p for p in self.base_path.rglob(sequence) if p.is_dir()]
            cand = matches[0] if matches else cand
        if not cand.exists() or not cand.is_dir():
            raise FileNotFoundError(f"Sequence directory not found: {sequence} (under {self.base_path})")
        self.sequence_dir = cand
        self.frame_paths = self._collect_frames(self.sequence_dir)
        if not self.frame_paths:
            raise FileNotFoundError(f"No image frames found in sequence directory: {self.sequence_dir}")
        self.frame_count = len(self.frame_paths)
        self.reset()

    def open_by_index(self, index: int) -> None:
        """Open the N-th sequence folder under base_path (0-based)."""
        seqs = self.list_sequences()
        if not seqs:
            raise FileNotFoundError(f"No sequence directories under {self.base_path}")
        if index < 0 or index >= len(seqs):
            raise IndexError(f"Index {index} out of range (0..{len(seqs)-1})")
        self.open_sequence(seqs[index].name)

    # -----------------------
    # Frame access
    # -----------------------
    def reset(self) -> None:
        """Reset internal pointer so next_frame() returns frame 1."""
        self.frame_idx = 0
        self._last_frame = None

    def has_next(self) -> bool:
        return self.frame_idx < self.frame_count

    def next_frame(self) -> Optional[Tuple[int, np.ndarray, Path]]:
        """
        Advance pointer and return (idx, bgr_frame, path), or None if exhausted.
        idx is 1-based.
        """
        if self.frame_idx >= self.frame_count:
            return None
        self.frame_idx += 1
        return self.get_frame(self.frame_idx)

    def get_frame(self, idx: int) -> Tuple[int, np.ndarray, Path]:
        """
        Random access (1-based). Returns (idx, bgr_frame, path).
        Raises IndexError for out-of-range.
        """
        if idx < 1 or idx > self.frame_count:
            raise IndexError(f"Frame index {idx} out of range (1..{self.frame_count})")
        p = self.frame_paths[idx - 1]
        bgr = cv2.imread(str(p))
        if bgr is None:
            raise RuntimeError(f"Failed to read image: {p}")
        self.frame_idx = idx
        self._last_frame = bgr
        return idx, bgr, p

    def get_frame_path(self, idx: int) -> Path:
        if idx < 1 or idx > self.frame_count:
            raise IndexError(f"Frame index {idx} out of range (1..{self.frame_count})")
        return self.frame_paths[idx - 1]

    def seek(self, idx: int) -> None:
        """Set internal pointer (1-based). Next next_frame() will return this frame if called after get_frame/seek."""
        if idx < 1 or idx > self.frame_count:
            raise IndexError(f"Seek index {idx} out of range (1..{self.frame_count})")
        self.frame_idx = idx - 1

    def last(self) -> Optional[Tuple[int, np.ndarray, Path]]:
        """Return last loaded frame (idx, frame, path) or None."""
        if self._last_frame is None:
            return None
        return (self.frame_idx, self._last_frame, self.get_frame_path(self.frame_idx))

    # -----------------------
    # Generator helper
    # -----------------------
    def frames(self, start: int = 1, stop: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray, Path]]:
        """Yield frames from start..stop (inclusive). Uses get_frame internally."""
        if self.frame_count == 0:
            return
        if stop is None:
            stop = self.frame_count
        for i in range(start, min(stop, self.frame_count) + 1):
            yield self.get_frame(i)

