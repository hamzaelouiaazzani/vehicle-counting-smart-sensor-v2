# Enhanced Count Visualizer (drop-in replacement for your existing CountVisualizer)
# - Keeps original drawing style and API: .render(frame, *counters)
# - Adds: legend, summary KPI panel (per-area listing, no summed total), per-area stacked class bars, sparklines (history), smooth count animation
# - Backwards-compatible: works with area objects that expose .get_area_info() and have .count_result

import math
import time
from collections import deque, defaultdict
from typing import Optional, Sequence, Any, Dict

import cv2
import numpy as np

# palette used when area objects don't provide a color
DEFAULT_PALETTE = [
    (56, 169, 255),
    (60, 180, 75),
    (240, 128, 128),
    (255, 215, 0),
    (147, 112, 219),
    (255, 105, 180),
]


class CountVisualizer:
    """
    Engaging Count Visualizer.

    Drop-in for the previous CountVisualizer. Call:
        vis = CountVisualizer(class_labels={0: 'car', 1: 'truck'}, history_len=60)
        out = vis.render(frame, *counters)

    Options added:
      - show_legend: draw class legend
      - show_summary: draw summary card (per-area counts; does NOT show a summed total)
      - history_len: number of frames to keep for sparklines/trends
      - class_labels: optional mapping from class index -> name
      - animate_counts: smooth numeric transitions for marketing-friendly animation
    """

    def __init__(
        self,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        text_thickness: int = 2,
        alpha_poly: float = 0.25,
        palette: Optional[Sequence[tuple]] = None,
        show_legend: bool = True,
        show_summary: bool = True,
        history_len: int = 60,
        class_labels: Optional[Dict[int, str]] = None,
        animate_counts: bool = True,
    ):
        self.font = font
        self.font_scale = font_scale
        self.text_thickness = text_thickness
        self.alpha_poly = alpha_poly
        self.palette = list(palette) if palette is not None else DEFAULT_PALETTE
        self.show_legend = show_legend
        self.show_summary = show_summary
        self.history_len = max(4, int(history_len))
        self.class_labels = class_labels or {}
        self.animate_counts = animate_counts

        # history per area name: deque of recent total counts
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        # animated displayed counts (smooth interpolation): area_name -> displayed value
        self._displayed_counts: Dict[str, float] = {}
        # last update timestamp (for interpolation)
        self._last_ts = time.time()

    # ------------------ helpers ------------------

    def _color_for(self, idx: int):
        return self.palette[idx % len(self.palette)]

    def _put_text_box(self, img, text: str, org, bg_color=(0, 0, 0), alpha=0.6, pad=8):
        (w, h), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.text_thickness)
        x, y = int(org[0]), int(org[1])
        x0, y0 = x - pad, y - pad
        x1, y1 = x + w + pad, y + h + pad
        # clamp to image
        h_img, w_img = img.shape[:2]
        x0, x1 = max(0, x0), min(w_img - 1, x1)
        y0, y1 = max(0, y0), min(h_img - 1, y1)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # shadow + text
        cv2.putText(img, text, (x, y + h), self.font, self.font_scale, (0, 0, 0), self.text_thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + h), self.font, self.font_scale, (255, 255, 255), self.text_thickness, cv2.LINE_AA)

    def _draw_polygon(self, img, polygon: np.ndarray, color: tuple, thickness=2, alpha=None):
        if polygon is None:
            return
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] != 2:
            return
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        a = self.alpha_poly if alpha is None else alpha
        cv2.addWeighted(overlay, a, img, 1 - a, 0, img)
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

    def _draw_dashed_line(self, img, p0, p1, color, thickness=2, dash_len=20, gap_len=12):
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        vec = np.array([x1 - x0, y1 - y0], dtype=float)
        dist = np.linalg.norm(vec)
        if dist < 1e-6:
            return
        direction = vec / dist
        step = dash_len + gap_len
        t = 0.0
        while t < dist:
            s = t
            e = min(t + dash_len, dist)
            p_s = (int(x0 + direction[0] * s), int(y0 + direction[1] * s))
            p_e = (int(x0 + direction[0] * e), int(y0 + direction[1] * e))
            cv2.line(img, p_s, p_e, color, thickness, lineType=cv2.LINE_AA)
            t += step
        # arrowhead
        self._draw_arrowhead(img, (int(x1), int(y1)), (-direction[0], -direction[1]), color, thickness=thickness + 1)

    def _draw_arrowhead(self, img, tip, direction, color, length=18, thickness=2):
        dx, dy = float(direction[0]), float(direction[1])
        angle = math.radians(30)
        def rot(vx, vy, a):
            return vx * math.cos(a) - vy * math.sin(a), vx * math.sin(a) + vy * math.cos(a)
        left = rot(dx, dy, angle)
        right = rot(dx, dy, -angle)
        p1 = (int(tip[0] + left[0] * length), int(tip[1] + left[1] * length))
        p2 = (int(tip[0] + right[0] * length), int(tip[1] + right[1] * length))
        cv2.line(img, tip, p1, color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, tip, p2, color, thickness, lineType=cv2.LINE_AA)

    def _draw_badge(self, img, center, text, color, radius=22):
        x, y = int(center[0]), int(center[1])
        # shadow
        cv2.circle(img, (x + 3, y + 3), radius, (0, 0, 0), -1)
        cv2.circle(img, (x, y), radius, color, -1)
        txt = str(text)
        (w, h), baseline = cv2.getTextSize(txt, self.font, self.font_scale * 0.95, 2)
        cv2.putText(img, txt, (x - w // 2, y + h // 2 - 2), self.font, self.font_scale * 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_line_label(self, img, p0, p1, label: str, color, offset_px: int = 18):
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        cx = mx + nx * offset_px
        cy = my + ny * offset_px
        (w, h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.text_thickness)
        x_org = int(cx - w / 2)
        y_org = int(cy - h / 2)
        bg = tuple(max(0, int(c - 40)) for c in color)
        self._put_text_box(img, label, (x_org, y_org), bg_color=bg, alpha=0.85)

    # ------------------ enhanced visuals ------------------
    def _draw_legend(self, img, unique_classes: Sequence[int], origin=(12, 12)):
        x0, y0 = origin
        pad = 8
        box_h = 22
        spacing = 6
        # determine width
        lines = []
        for cls in unique_classes:
            name = self.class_labels.get(cls, f"cls_{cls}")
            lines.append(f"{name}")
        # compute widest
        max_w = 0
        for txt in lines:
            (w, h), _ = cv2.getTextSize(txt, self.font, self.font_scale * 0.9, 1)
            if w > max_w:
                max_w = w
        panel_w = max_w + 3 * pad + box_h
        panel_h = len(lines) * (box_h + spacing) + pad
        # panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        # draw entries
        y = y0 + pad
        for i, cls in enumerate(unique_classes):
            col = self._color_for(i)
            # color box
            cv2.rectangle(img, (x0 + pad, y), (x0 + pad + box_h, y + box_h), col, cv2.FILLED)
            # text
            name = self.class_labels.get(cls, f"cls_{cls}")
            cv2.putText(img, name, (x0 + pad + box_h + pad // 2, y + box_h - 4), self.font, self.font_scale * 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            y += box_h + spacing

    def _draw_summary_panel(self, img, totals: Dict[str, int], area_order: Sequence[str], area_colors: Dict[str, tuple], origin=None):
        """
        Draws a summary panel that lists counts per area (and global if present) WITHOUT summing across them.
        - totals: mapping area_name -> count
        - area_order: sequence of area names in the order they were passed (keeps color mapping stable)
        - area_colors: mapping area_name -> color tuple
        """
        h_img, w_img = img.shape[:2]
        padding = 12
        row_h = 22
        title_h = 28
        n = max(1, len(area_order))
        panel_w = 320
        panel_h = title_h + n * (row_h + 6) + padding
        if origin is None:
            x0, y0 = w_img - panel_w - 12, 12
        else:
            x0, y0 = origin

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

        # title
        title = "Counting Summary (per area)"
        cv2.putText(img, title, (x0 + 14, y0 + 20), self.font, self.font_scale * 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # area rows
        y = y0 + title_h
        box_sz = 14
        for i, name in enumerate(area_order):
            val = totals.get(name, 0)
            col = area_colors.get(name, (120, 120, 120))
            # color pill
            pill_x = x0 + 14
            pill_y = int(y + i * (row_h + 6)) + 6
            cv2.rectangle(img, (pill_x, pill_y), (pill_x + box_sz, pill_y + box_sz), col, cv2.FILLED)
            # area name
            txt_x = pill_x + box_sz + 10
            txt_y = pill_y + box_sz - 2
            cv2.putText(img, name, (txt_x, txt_y), self.font, self.font_scale * 0.9, (240, 240, 240), 1, cv2.LINE_AA)
            # count (right aligned)
            cnt_txt = str(val)
            (w_cnt, _), _ = cv2.getTextSize(cnt_txt, self.font, self.font_scale * 0.95, 2)
            cnt_x = x0 + panel_w - 14 - w_cnt
            cv2.putText(img, cnt_txt, (cnt_x, txt_y), self.font, self.font_scale * 0.95, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_sparkline(self, img, history: Sequence[int], top_left, size=(120, 36), color=(255, 255, 255)):
        w, h = size
        x0, y0 = int(top_left[0]), int(top_left[1])
        if len(history) == 0:
            return
        arr = np.array(history, dtype=float)
        mx = max(arr.max(), 1.0)
        # normalize to [0,1]
        yvals = (arr / mx) * (h - 4)
        # draw background
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + w, y0 + h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        # polyline points
        n = len(yvals)
        if n == 1:
            p = (x0 + 2, int(y0 + h - yvals[0] - 2))
            cv2.circle(img, p, 2, color, -1)
            return
        xs = np.linspace(x0 + 2, x0 + w - 2, n)
        pts = [(int(xs[i]), int(y0 + h - yvals[i] - 2)) for i in range(n)]
        for i in range(len(pts) - 1):
            cv2.line(img, pts[i], pts[i + 1], color, 1, lineType=cv2.LINE_AA)

    def _draw_stacked_bar(self, img, counts: Sequence[int], top_left, size=(100, 12)):
        x0, y0 = int(top_left[0]), int(top_left[1])
        w, h = size
        total = sum(counts)
        if total == 0:
            # empty bar
            cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (80, 80, 80), 1, cv2.LINE_AA)
            return
        cur_x = x0
        for i, c in enumerate(counts):
            frac = c / total
            seg_w = int(round(frac * w))
            col = self._color_for(i)
            cv2.rectangle(img, (cur_x, y0), (cur_x + seg_w, y0 + h), col, cv2.FILLED)
            cur_x += seg_w
        # border
        cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (40, 40, 40), 1, cv2.LINE_AA)

    # ------------------ main render ------------------
    def render(self, frame: np.ndarray, *counters: Any) -> np.ndarray:
        out = frame.copy()
        ts = time.time()
        dt = max(1e-6, ts - self._last_ts)
        self._last_ts = ts

        totals = {}
        unique_classes_set = set()
        area_order = []
        area_colors: Dict[str, tuple] = {}

        for i, ctr in enumerate(counters):
            area_info = ctr.get_area_info()
            count_obj = getattr(ctr, "count_result", None)

            poly = area_info.get("polygon")
            line = area_info.get("line")
            name = area_info.get("name", f"Area_{i}")
            line_vicinity = area_info.get("line_vicinity")

            color = self._color_for(i)

            # store order & color mapping for summary
            area_order.append(name)
            area_colors[name] = color

            # draw polygon
            if poly is not None:
                self._draw_polygon(out, poly, color, thickness=3)
                try:
                    centroid = np.asarray(poly).mean(axis=0)
                    badge_pos = (int(centroid[0]), int(centroid[1]))
                except Exception:
                    badge_pos = (40 + 40 * i, 60 + 40 * i)
            else:
                badge_pos = (40 + 40 * i, 60 + 40 * i)

            # draw line if present
            if line:
                p0, p1 = line
                self._draw_dashed_line(out, p0, p1, color, thickness=2, dash_len=28, gap_len=14)
                self._draw_line_label(out, p0, p1, name, color)

            # read counts
            count_val = getattr(count_obj, "total_count", 0) if count_obj is not None else 0
            counts_by_class = getattr(count_obj, "counts_by_class", None)
            if counts_by_class is None:
                counts_by_class = np.zeros((0,), dtype=int)

            # update history
            self._history[name].append(int(count_val))
            totals[name] = int(count_val)

            # animation interpolation
            disp = float(self._displayed_counts.get(name, float(count_val)))
            if self.animate_counts:
                # approach target smoothly (exponential smoothing)
                alpha = min(0.45, 0.4 + 0.2 * (dt))
                disp = disp + alpha * (float(count_val) - disp)
            else:
                disp = float(count_val)
            self._displayed_counts[name] = disp

            # draw badge with animated count (rounded)
            self._draw_badge(out, badge_pos, int(round(disp)), color, radius=22)

            # draw stacked per-class bar under badge
            top_left = (badge_pos[0] - 50, badge_pos[1] + 28)
            counts_list = counts_by_class.tolist() if hasattr(counts_by_class, "tolist") else list(counts_by_class)
            # limit to first 6 classes for visual clarity
            counts_small = counts_list[:6]
            self._draw_stacked_bar(out, counts_small, top_left, size=(100, 12))

            # draw sparkline next to badge
            hist = list(self._history[name])
            spark_tl = (badge_pos[0] - 60, badge_pos[1] + 44)
            self._draw_sparkline(out, hist, spark_tl, size=(120, 36), color=(230, 230, 230))

            # collect classes for legend
            for idx in range(len(counts_by_class)):
                if counts_by_class[idx] > 0:
                    unique_classes_set.add(idx)

        # draw legend (if requested)
        unique_classes = sorted(unique_classes_set)
        if self.show_legend and unique_classes:
            self._draw_legend(out, unique_classes, origin=(12, 12))

        # draw summary panel (per-area, using area colors provided) - no overall total computed
        if self.show_summary:
            self._draw_summary_panel(out, totals, area_order, area_colors, origin=None)

        # watermark / timestamp
        ts_txt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        self._put_text_box(out, ts_txt, (12, out.shape[0] - 28), bg_color=(20, 20, 20), alpha=0.55, pad=6)

        return out










