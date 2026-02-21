import cv2
import numpy as np
from typing import Optional, Tuple, List, Any
from abc import ABC, abstractmethod

Point = Tuple[int, int]
Polygon = List[Point]
TwoLines = Tuple[Tuple[Point, Point], Tuple[Point, Point]]

class ShapeSelectorBase(ABC):
    """
    Base class for interactive shape selectors.

    Subclasses must implement:
      - required_points (int | None)  : None means unlimited (polygon)
      - _on_click(self, display_pt)   : handle a click (optional; default appends)
      - _draw_preview(self, canvas)   : draw in-progress preview on display canvas
      - _show_preview_on_original(self, orig_img, *shape_args) : final original-size preview

    Common constructor args (kept same as earlier classes):
        window_name, max_display_size, auto_confirm, show_instructions,
        show_preview_after_confirm, preview_wait_secs
    """

    required_points: Optional[int] = None  # subclasses override (2, 4) or None for polygon

    def __init__(self,
                 window_name: str = "ShapeSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        self.window_name = window_name
        self.max_display_size = int(max_display_size)
        self.auto_confirm = bool(auto_confirm)
        self.show_instructions = bool(show_instructions)
        self.show_preview_after_confirm = bool(show_preview_after_confirm)
        self.preview_wait_secs = preview_wait_secs

        # runtime
        self._display_img: Optional[np.ndarray] = None
        self._scale: float = 1.0  # display -> original scale factor (display = original * scale)
        self._pts_display: List[Point] = []
        self._confirmed: bool = False
        self._cancelled: bool = False

    # -----------------------
    # Core selection loop
    # -----------------------
    def _compute_scale(self, w: int, h: int) -> float:
        max_dim = max(w, h)
        if max_dim <= self.max_display_size or self.max_display_size <= 0:
            return 1.0
        return float(self.max_display_size) / float(max_dim)

    def _setup_display_image(self, img: np.ndarray) -> None:
        orig_h, orig_w = img.shape[:2]
        self._scale = self._compute_scale(orig_w, orig_h)
        if self._scale < 1.0:
            disp_w = int(round(orig_w * self._scale))
            disp_h = int(round(orig_h * self._scale))
            self._display_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        else:
            self._display_img = img.copy()

    def _map_to_original(self, pt_display: Point) -> Point:
        if self._scale == 0:
            raise RuntimeError("Invalid scale (0).")
        x_disp, y_disp = pt_display
        x_orig = int(round(x_disp / self._scale))
        y_orig = int(round(y_disp / self._scale))
        return (x_orig, y_orig)

    def _clamp_display_pt(self, x: int, y: int) -> Point:
        if self._display_img is None:
            raise RuntimeError("_display_img is not set")
        x = int(round(max(0, min(self._display_img.shape[1] - 1, x))))
        y = int(round(max(0, min(self._display_img.shape[0] - 1, y))))
        return (x, y)

    def _mouse_cb_wrapper(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pt = self._clamp_display_pt(x, y)
            self._on_click(pt)

    def _on_click(self, display_pt: Point) -> None:
        """
        Default behavior: append point, and if we've already filled required_points, start a new selection.
        Subclasses may override for custom rules (polygon auto-close).
        """
        if self.required_points is None:
            # unlimited / polygon
            self._pts_display.append(display_pt)
        else:
            if len(self._pts_display) < self.required_points:
                self._pts_display.append(display_pt)
            else:
                # restart selection with this click
                self._pts_display = [display_pt]

        # auto-confirm if requested and we reached requirement
        if self.auto_confirm and (self.required_points is not None) and len(self._pts_display) >= self.required_points:
            self._confirmed = True

    def _common_event_handling(self, key: int) -> None:
        # reset
        if key == ord('r'):
            self._pts_display = []
            self._confirmed = False
            self._cancelled = False
        # cancel
        if key == ord('q') or key == 27:
            self._cancelled = True
        # confirm (explicit)
        if (key == ord('c') or key == 13):
            # permit confirmation only when we have enough points
            if (self.required_points is None) or (len(self._pts_display) >= self.required_points):
                self._confirmed = True

    def _run_selection(self, orig_img: np.ndarray):
        if orig_img is None:
            raise ValueError("img must be a numpy array")

        self._setup_display_image(orig_img)
        self._pts_display = []
        self._confirmed = False
        self._cancelled = False

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(self.window_name, self._display_img)
        cv2.setMouseCallback(self.window_name, self._mouse_cb_wrapper)

        try:
            while True:
                canvas = self._display_img.copy()
                self._draw_preview(canvas)
                if self.show_instructions:
                    self._put_instructions(canvas)

                cv2.imshow(self.window_name, canvas)
                key = cv2.waitKey(20) & 0xFF
                self._common_event_handling(key)

                if self._confirmed or self._cancelled:
                    break
        finally:
            cv2.setMouseCallback(self.window_name, lambda *args, **kwargs: None)
            cv2.destroyWindow(self.window_name)

    # -----------------------
    # Hooks for subclasses
    # -----------------------
    @abstractmethod
    def _draw_preview(self, canvas: np.ndarray) -> None:
        """Draw in-progress preview on display canvas (display coords)."""

    @abstractmethod
    def _show_preview_on_original(self, orig_img: np.ndarray, *shape_args: Any) -> None:
        """Show final preview on original-sized image. shape_args are data to draw (points in original coords)."""

    @abstractmethod
    def _final_shape_from_display(self) -> Any:
        """Return the final shape(s) in ORIGINAL coordinates derived from display points.
           For LineSelector -> list of two points (lists)
           For TwoLineSelector -> list [[p1,p2],[p3,p4]]
           For PolygonSelector -> list of points (lists)
        """

    def _put_instructions(self, canvas: np.ndarray) -> None:
        # Generic instructions; subclasses may override if desired.
        h, w = canvas.shape[:2]
        lines = [
            "Left-click: place points",
            "'r': reset  |  'c' or Enter: confirm  |  'q' or ESC: cancel",
        ]
        y0 = 20
        for i, txt in enumerate(lines):
            cv2.putText(canvas, txt, (10, y0 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # public helper for subclasses to access display pts
    def _get_display_points(self) -> List[Point]:
        return list(self._pts_display)


# -----------------------
# Concrete selectors
# -----------------------




class PointSelector(ShapeSelectorBase):
    """
    Interactive selector for a single point.
    Returns the point as a list [x, y] in original coordinates.
    """
    required_points = 1

    def __init__(self,
                 window_name: str = "PointSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)

    def select_point(self, img: np.ndarray) -> Optional[List[int]]:
        """
        Starts the selection process and returns the chosen point in original image coordinates as [x, y].
        """
        self._run_selection(img)
        if self._cancelled:
            return None
        p_disp = self._get_display_points()[0]
        p_orig = self._map_to_original(p_disp)
        p_list = [int(p_orig[0]), int(p_orig[1])]
        if self.show_preview_after_confirm:
            self._show_preview_on_original(img, p_orig)
        return p_list

    def _draw_preview(self, canvas: np.ndarray) -> None:
        """
        Draws the point on the display canvas.
        """
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

    def _final_shape_from_display(self) -> List[int]:
        """
        Returns the final point in original coordinates as [x, y].
        """
        pt_disp = self._get_display_points()[0]
        p = self._map_to_original(pt_disp)
        return [int(p[0]), int(p[1])]

    def _show_preview_on_original(self, orig_img: np.ndarray, pt: Point) -> None:
        """
        Displays a preview of the selected point on the original-sized image.
        """
        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(p):
            x, y = p
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        ptc = clamp(pt)

        cv2.circle(img_copy, ptc, radius=8, color=(0, 255, 0), thickness=-1)
        cv2.putText(img_copy, f"P1 {ptc}", (ptc[0] + 10, ptc[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - selected point preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)

        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))
        cv2.destroyWindow(win)





class LineSelector(ShapeSelectorBase):
    required_points = 2

    def __init__(self,
                 window_name: str = "LineSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)

    def select_line(self, img: np.ndarray) -> Optional[List[List[int]]]:
        self._run_selection(img)
        if self._cancelled:
            return None
        p1_disp, p2_disp = self._get_display_points()[0], self._get_display_points()[1]
        p1 = self._map_to_original(p1_disp)
        p2 = self._map_to_original(p2_disp)
        if self.show_preview_after_confirm:
            self._show_preview_on_original(img, (p1, p2))
        return [[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]]

    def _draw_preview(self, canvas: np.ndarray) -> None:
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 255), thickness=2, lineType=cv2.LINE_AA)

    def _final_shape_from_display(self) -> List[List[int]]:
        p1_disp, p2_disp = self._get_display_points()[0], self._get_display_points()[1]
        p1 = self._map_to_original(p1_disp)
        p2 = self._map_to_original(p2_disp)
        return [[int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])]]

    def _show_preview_on_original(self, orig_img: np.ndarray, shape_args: Any) -> None:
        (p1, p2) = shape_args
        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(pt):
            x, y = pt
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        p1c = clamp(p1)
        p2c = clamp(p2)

        cv2.line(img_copy, p1c, p2c, (0, 180, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.circle(img_copy, p1c, radius=8, color=(0, 255, 0), thickness=-1)
        cv2.circle(img_copy, p2c, radius=8, color=(0, 255, 0), thickness=-1)
        cv2.putText(img_copy, f"P1 {p1c}", (p1c[0] + 10, p1c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img_copy, f"P2 {p2c}", (p2c[0] + 10, p2c[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - selected line preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)

        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))

        cv2.destroyWindow(win)


class TwoLineSelector(ShapeSelectorBase):
    required_points = 4

    def __init__(self,
                 window_name: str = "TwoLineSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)

    def select_two_lines(self, img: np.ndarray) -> Optional[List[List[List[int]]]]:
        self._run_selection(img)
        if self._cancelled:
            return None
        pts = self._get_display_points()
        p1 = self._map_to_original(pts[0]); p2 = self._map_to_original(pts[1])
        p3 = self._map_to_original(pts[2]); p4 = self._map_to_original(pts[3])
        if self.show_preview_after_confirm:
            self._show_preview_on_original(img, ((p1, p2), (p3, p4)))
        return [[ [int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])] ],
                [ [int(p3[0]), int(p3[1])], [int(p4[0]), int(p4[1])] ]]

    def _draw_preview(self, canvas: np.ndarray) -> None:
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)
        if len(pts) >= 2:
            (x1, y1), (x2, y2) = pts[0], pts[1]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 180, 255), thickness=2, lineType=cv2.LINE_AA)
        if len(pts) >= 4:
            (x3, y3), (x4, y4) = pts[2], pts[3]
            cv2.line(canvas, (int(x3), int(y3)), (int(x4), int(y4)), (255, 140, 0), thickness=2, lineType=cv2.LINE_AA)

    def _final_shape_from_display(self) -> List[List[List[int]]]:
        pts = self._get_display_points()
        p1 = self._map_to_original(pts[0]); p2 = self._map_to_original(pts[1])
        p3 = self._map_to_original(pts[2]); p4 = self._map_to_original(pts[3])
        return [[ [int(p1[0]), int(p1[1])], [int(p2[0]), int(p2[1])] ],
                [ [int(p3[0]), int(p3[1])], [int(p4[0]), int(p4[1])] ]]

    def _show_preview_on_original(self, orig_img: np.ndarray, shape_args: Any) -> None:
        (line1, line2) = shape_args
        (p1, p2), (p3, p4) = line1, line2
        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(pt):
            x, y = pt
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        p1c, p2c, p3c, p4c = clamp(p1), clamp(p2), clamp(p3), clamp(p4)

        cv2.line(img_copy, p1c, p2c, (0, 180, 255), thickness=3, lineType=cv2.LINE_AA)
        cv2.line(img_copy, p3c, p4c, (255, 140, 0), thickness=3, lineType=cv2.LINE_AA)

        for idx, pt in enumerate((p1c, p2c, p3c, p4c), start=1):
            cv2.circle(img_copy, pt, radius=8, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_copy, f"P{idx} {pt}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - selected lines preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)

        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))

        cv2.destroyWindow(win)


class PolygonSelector(ShapeSelectorBase):
    """
    Polygon selector with auto-close on click near first point (optional).
    Returns polygon as a list of points [[x,y], ...]
    """

    required_points = None  # unlimited; confirm with 'c' or auto-close
    def __init__(self,
                 window_name: str = "PolygonSelector",
                 max_display_size: int = 1200,
                 min_points: int = 3,
                 auto_close_on_click_near_first: bool = True,
                 close_pixel_radius: int = 10,
                 auto_confirm: bool = False,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        # note: polygon typically uses explicit confirm with 'c' unless auto_close triggers
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)
        self.min_points = max(3, int(min_points))
        self.auto_close = bool(auto_close_on_click_near_first)
        self.close_pixel_radius = int(close_pixel_radius)  # in display pixels

    def select_polygon(self, img: np.ndarray) -> Optional[List[List[int]]]:
        self._run_selection(img)
        if self._cancelled:
            return None
        poly = [[int(p[0]), int(p[1])] for p in self._get_display_points()]
        poly_orig = [[int(p[0]), int(p[1])] for p in [self._map_to_original(pt) for pt in self._get_display_points()]]
        if self.show_preview_after_confirm:
            # pass original-coordinate tuples to preview (preview handles lists as well)
            self._show_preview_on_original(img, poly_orig)
        return poly_orig

    def _on_click(self, display_pt: Point) -> None:
        # override polygon behavior to support auto-close near first point
        if self._display_img is None:
            return
        x, y = display_pt
        if self.auto_close and len(self._pts_display) >= self.min_points:
            first = self._pts_display[0]
            dist2 = (x - first[0])**2 + (y - first[1])**2
            if dist2 <= (self.close_pixel_radius ** 2):
                # close and confirm (do not append a duplicate of first)
                if len(self._pts_display) >= self.min_points:
                    self._confirmed = True
                return

        # normal append
        self._pts_display.append(display_pt)

    def _draw_preview(self, canvas: np.ndarray) -> None:
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        n = len(pts)
        if n >= 2:
            arr = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [arr], isClosed=False, color=(0, 180, 255), thickness=2, lineType=cv2.LINE_AA)
        if self.auto_close and len(pts) >= self.min_points:
            fx, fy = pts[0]
            cv2.circle(canvas, (fx, fy), radius=max(6, self.close_pixel_radius), color=(255, 180, 0), thickness=1)

    def _final_shape_from_display(self) -> List[List[int]]:
        return [[int(p[0]), int(p[1])] for p in [self._map_to_original(pt) for pt in self._get_display_points()]]

    def _show_preview_on_original(self, orig_img: np.ndarray, polygon: List[List[int]]) -> None:
        if len(polygon) < 3:
            return
        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(pt):
            x, y = pt
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        pts = np.array([clamp(p) for p in polygon], dtype=np.int32)

        # filled semi-transparent polygon
        overlay = img_copy.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 180, 255))
        alpha = 0.25
        cv2.addWeighted(overlay, alpha, img_copy, 1 - alpha, 0, img_copy)

        # outline + points + labels
        cv2.polylines(img_copy, [pts], isClosed=True, color=(0, 100, 200), thickness=3, lineType=cv2.LINE_AA)
        for i, (x, y) in enumerate(pts.tolist(), start=1):
            cv2.circle(img_copy, (x, y), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_copy, f"P{i} {x,y}", (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - polygon preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)
        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))
        cv2.destroyWindow(win)


# The classes ShapeSelectorBase, LineSelector, TwoLineSelector, and PolygonSelector
# would be placed here as they were provided in the prompt.


class RectangleSelector(ShapeSelectorBase):
    """
    Interactive selector for a rectangle defined by two opposite corners.
    Returns all four corners as a list of lists: [[tl], [tr], [br], [bl]] with Python ints.
    """
    required_points = 2

    def __init__(self,
                 window_name: str = "RectangleSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)

    def select_rectangle(self, img: np.ndarray) -> Optional[List[List[int]]]:
        """
        Starts the selection process and returns the four corner points of the rectangle
        in original image coordinates as [[tl], [tr], [br], [bl]].
        """
        self._run_selection(img)
        if self._cancelled:
            return None

        pts = self._get_display_points()
        if len(pts) < 2:
            return None

        p1 = self._map_to_original(pts[0])
        p2 = self._map_to_original(pts[1])

        x1, y1 = p1
        x2, y2 = p2

        top_left = (int(min(x1, x2)), int(min(y1, y2)))
        bottom_right = (int(max(x1, x2)), int(max(y1, y2)))
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        rect_corners = [top_left, top_right, bottom_right, bottom_left]
        if self.show_preview_after_confirm:
            self._show_preview_on_original(img, rect_corners)

        # convert to list of lists
        return [[int(x), int(y)] for x, y in rect_corners]

    def _draw_preview(self, canvas: np.ndarray) -> None:
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        if len(pts) == 2:
            pt1_disp, pt2_disp = pts[0], pts[1]
            cv2.rectangle(canvas, pt1_disp, pt2_disp, (0, 180, 255), thickness=2, lineType=cv2.LINE_AA)

    def _final_shape_from_display(self) -> List[List[int]]:
        pts = self._get_display_points()
        p1_orig = self._map_to_original(pts[0])
        p2_orig = self._map_to_original(pts[1])

        x1, y1 = p1_orig
        x2, y2 = p2_orig

        top_left = (int(min(x1, x2)), int(min(y1, y2)))
        bottom_right = (int(max(x1, x2)), int(max(y1, y2)))
        top_right = (bottom_right[0], top_left[1])
        bottom_left = (top_left[0], bottom_right[1])

        corners = [top_left, top_right, bottom_right, bottom_left]
        return [[int(x), int(y)] for x, y in corners]

    def _show_preview_on_original(self, orig_img: np.ndarray, shape_args: Any) -> None:
        rect_corners = shape_args
        if len(rect_corners) != 4:
            return

        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(pt):
            x, y = pt
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        clamped_corners = [clamp(p) for p in rect_corners]
        pts_arr = np.array(clamped_corners, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts_arr], isClosed=True, color=(0, 180, 255), thickness=3, lineType=cv2.LINE_AA)

        for i, pt in enumerate(clamped_corners, start=1):
            cv2.circle(img_copy, pt, radius=8, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_copy, f"P{i} {pt}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - selected rectangle preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)

        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))
        cv2.destroyWindow(win)




# The classes ShapeSelectorBase, LineSelector, TwoLineSelector, and PolygonSelector
# would be placed here as they were provided in the prompt.


class OBBSelector(ShapeSelectorBase):
    """
    Interactive selector for an Oriented Bounding Box (OBB) defined by three points.
    The select_obb method returns four corner points as a list of lists in clockwise order.
    """
    required_points = 3

    def __init__(self,
                 window_name: str = "OBBSelector",
                 max_display_size: int = 1200,
                 auto_confirm: bool = True,
                 show_instructions: bool = True,
                 show_preview_after_confirm: bool = True,
                 preview_wait_secs: Optional[float] = None):
        super().__init__(window_name, max_display_size, auto_confirm,
                         show_instructions, show_preview_after_confirm, preview_wait_secs)

    def select_obb(self, img: np.ndarray) -> Optional[List[List[int]]]:
        """
        Starts the selection process and returns the four corner points of the OBB
        in original image coordinates as [[x,y], ...].
        """
        self._run_selection(img)
        if self._cancelled:
            return None

        pts = self._get_display_points()
        if len(pts) < 3:
            return None

        # Map selected points to original image
        p1 = self._map_to_original(pts[0])
        p2 = self._map_to_original(pts[1])
        p3 = self._map_to_original(pts[2])

        # Compute the 4th point and return all 4 corners
        obb_corners = self._calculate_obb_from_3_points(p1, p2, p3)

        if self.show_preview_after_confirm:
            self._show_preview_on_original(img, obb_corners)

        return obb_corners

    def _calculate_obb_from_3_points(self, p1: Point, p2: Point, p3: Point) -> List[List[int]]:
        p1_arr, p2_arr, p3_arr = np.array(p1), np.array(p2), np.array(p3)
    
        # Determine which two vectors are roughly perpendicular
        v1 = p2_arr - p1_arr
        v2 = p3_arr - p1_arr
        if abs(np.dot(v1, v2)) < 1e-6:
            # Already perpendicular
            p4 = p2_arr + (p3_arr - p1_arr)
            corners = [p1_arr, p2_arr, p4, p3_arr]
        else:
            # Assume p1-p2 and p2-p3 are sides
            v1 = p2_arr - p1_arr
            v2 = p3_arr - p2_arr
            p4 = p1_arr + v2
            corners = [p1_arr, p2_arr, p3_arr, p4]
    
        # Convert to Python ints and lists
        return [[int(x), int(y)] for (x, y) in [(c[0], c[1]) for c in corners]]


    def _draw_preview(self, canvas: np.ndarray) -> None:
        pts = self._get_display_points()
        for i, (x, y) in enumerate(pts):
            cv2.circle(canvas, (int(x), int(y)), radius=6, color=(0, 255, 0), thickness=-1)
            cv2.putText(canvas, f"P{i+1}", (int(x) + 8, int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2, cv2.LINE_AA)

        if len(pts) >= 2:
            cv2.line(canvas, pts[0], pts[1], (0, 180, 255), thickness=2, lineType=cv2.LINE_AA)

        if len(pts) == 3:
            obb_disp = self._calculate_obb_from_3_points(pts[0], pts[1], pts[2])
            pts_arr = np.array(obb_disp, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [pts_arr], isClosed=True, color=(255, 140, 0), thickness=2, lineType=cv2.LINE_AA)

    def _final_shape_from_display(self) -> List[List[int]]:
        pts = self._get_display_points()
        return self._calculate_obb_from_3_points(pts[0], pts[1], pts[2])

    def _show_preview_on_original(self, orig_img: np.ndarray, shape_args: Any) -> None:
        obb_points = shape_args
        if len(obb_points) < 4:
            return

        img_copy = orig_img.copy()
        h, w = img_copy.shape[:2]

        def clamp(pt):
            x, y = pt
            x = int(max(0, min(w - 1, x)))
            y = int(max(0, min(h - 1, y)))
            return (x, y)

        clamped_points = [clamp(p) for p in obb_points]
        pts_arr = np.array(clamped_points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts_arr], isClosed=True, color=(255, 140, 0), thickness=3, lineType=cv2.LINE_AA)

        for i, pt in enumerate(clamped_points, start=1):
            cv2.circle(img_copy, pt, radius=8, color=(0, 255, 0), thickness=-1)
            cv2.putText(img_copy, f"P{i} {pt}", (pt[0] + 10, pt[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        win = f"{self.window_name} - selected OBB preview"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow(win, img_copy)

        if self.preview_wait_secs is None:
            cv2.waitKey(0)
        else:
            cv2.waitKey(int(self.preview_wait_secs * 1000))
        cv2.destroyWindow(win)