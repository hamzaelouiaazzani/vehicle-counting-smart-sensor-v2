
import math
import time
import cv2
import numpy as np
from collections import deque, defaultdict
from typing import Optional, Tuple, Dict, Iterable, Union, List, Any, Sequence


class TrackVisualizer:
    """
    TrackVisualizer with enhanced ROI / global area styling inspired by CountVisualizer.
    Keeps previous features (boxes, labels, confidence bar, trails) and adds:
      - semi-transparent polygon fills with outline
      - dashed counting lines with arrowheads
      - optional badge placement at polygon centroid / line midpoint
      - optional legend and summary panel (rendered when you pass `area_counts`)
    """

    DEFAULT_PALETTE = [
        (255, 127, 14),  # orange
        (44, 160, 44),   # green
        (31, 119, 180),  # blue
        (214, 39, 40),   # red
        (148, 103, 189), # purple
        (140, 86, 75),   # brown
        (227, 119, 194), # pink
        (127, 127, 127), # gray
        (188, 189, 34),  # olive
        (23, 190, 207),  # cyan
    ]

    def __init__(
        self,
        max_trace_len: int = 30,
        trail_thickness: int = 2,
        trail_alpha: float = 0.65,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        font_thickness: int = 1,
        palette: Optional[Iterable[Tuple[int,int,int]]] = None,
        linger_frames: int = 0,
        # area / Count-style visual options:
        areas: Optional[Tuple[List[Dict[str,Any]], Dict[str,Any]]] = None,
        area_palette: Optional[Iterable[Tuple[int,int,int]]] = None,
        area_alpha: float = 0.22,        # subtle fill (not full-screen)
        area_outline_thickness: int = 2,
        show_legend: bool = True,
        show_summary: bool = True,
        class_labels: Optional[Dict[int,str]] = None,
        history_len: int = 60,
    ):
        # visual / text params
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = float(font_scale)
        self.font_thickness = int(font_thickness)
        self.box_thickness = int(box_thickness)

        # traces & trails
        self.max_trace_len = int(max_trace_len)
        self.trail_thickness = int(trail_thickness)
        self.trail_alpha = float(trail_alpha)
        self._traces: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trace_len))
        self._absent_counts: Dict[int, int] = defaultdict(int)
        self.linger_frames = int(linger_frames)

        # palettes & class color mapping
        self.palette = list(palette) if palette is not None else list(self.DEFAULT_PALETTE)
        self._class_color_cache: Dict[int, Tuple[int,int,int]] = {}

        # area visualization state
        self.area_palette = list(area_palette) if area_palette is not None else list(self.DEFAULT_PALETTE)
        self.area_alpha = float(area_alpha)
        self.area_outline_thickness = int(area_outline_thickness)
        self._rois: List[Dict[str,Any]] = []
        self._global_area: Optional[Dict[str,Any]] = None
        if areas is not None:
            self._prepare_areas(areas)

        # enhanced count-style widgets
        self.show_legend = bool(show_legend)
        self.show_summary = bool(show_summary)
        self.class_labels = class_labels or {}
        self.history_len = max(4, int(history_len))
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_len))
        self._displayed_counts: Dict[str, float] = {}
        self._last_ts = time.time()

    # -------------------------
    # Public annotate API (backwards-compatible)
    # -------------------------
    def annotate(
        self,
        frame_data: Union[np.ndarray, dict],
        tracked_array: np.ndarray,
        draw_trails: bool = True,
        area_counts: Optional[Dict[str, int]] = None,   # pass counts per area if you want summary/badge numbers
    ) -> np.ndarray:
        """
        Annotate and return a frame copy.
        tracked_array: np.ndarray (N,8) (x1,y1,x2,y2,id,conf,cls,ind)
        area_counts: optional mapping area_name -> int for summary/badge (visual only)
        """
        if isinstance(frame_data, dict):
            frame = frame_data.get("frame")
            if frame is None:
                raise ValueError("frame_data dict must contain key 'frame'")
        else:
            frame = frame_data
        if frame is None:
            raise ValueError("No frame provided")

        out = frame.copy()
        h, w = out.shape[:2]

        # draw areas first under everything
        if self._rois or self._global_area:
            out = self._draw_areas(out)

        # if no detections, update absent counters & prune traces
        if tracked_array is None or len(tracked_array) == 0:
            self._increment_absent_all_and_prune()
            # draw widgets if user provided counts (empty frame)
            if area_counts is not None and self.show_summary:
                out = self._draw_summary_panel(out, area_counts)
            return out

        arr = np.asarray(tracked_array)
        if arr.ndim != 2 or arr.shape[1] < 8:
            raise ValueError("tracked_array must be shape (N,8) with columns (x1,y1,x2,y2,id,conf,cls,ind)")

        x1s = arr[:, 0].astype(int)
        y1s = arr[:, 1].astype(int)
        x2s = arr[:, 2].astype(int)
        y2s = arr[:, 3].astype(int)
        ids  = arr[:, 4]
        confs= arr[:, 5]
        clss = arr[:, 6].astype(int)

        # update traces (handles absent counters)
        self._increment_absent_all()
        self._update_traces(ids, x1s, y1s, x2s, y2s, w, h)
        self._prune_stale_traces()

        # draw trails under boxes
        if draw_trails:
            out = self._draw_all_trails(out)

        # draw bounding boxes, conf bars, labels
        for (x1,y1,x2,y2, raw_id, raw_conf, cls_idx) in zip(x1s,y1s,x2s,y2s, ids, confs, clss):
            x1 = max(0, min(w-1, int(x1))); y1 = max(0, min(h-1, int(y1)))
            x2 = max(0, min(w-1, int(x2))); y2 = max(0, min(h-1, int(y2)))
            tid = int(raw_id) if not np.isnan(raw_id) else -1

            # normalize confidence to 0..1 if input uses 0..100
            conf = float(raw_conf)
            if conf > 1.0:
                conf = conf / 100.0
            conf = max(0.0, min(1.0, conf))

            color = self._color_for_class(cls_idx)
            self._draw_box(out, (x1,y1,x2,y2), color)
            self._draw_confidence_bar(out, (x1,y1,x2,y2), conf)
            self._draw_label_tag(out, (x1,y1,x2,y2), tid, cls_idx, color, conf)

        # draw optional summary/legend widgets if user passed area_counts
        if area_counts:
            if self.show_legend:
                unique_classes = sorted({int(c) for c in clss.tolist()})
                if unique_classes:
                    out = self._draw_legend(out, unique_classes)
            if self.show_summary:
                out = self._draw_summary_panel(out, area_counts)

        return out

    # -------------------------
    # Area parsing & drawing (CountVisualizer-inspired style)
    # -------------------------
    def _prepare_areas(self, areas_tuple: Tuple[List[Dict[str,Any]], Dict[str,Any]]):
        """Turn user-provided areas tuple (rois_list, global_dict) into internal structure with colors."""
        rois_raw, global_raw = areas_tuple
        palette = self.area_palette
        p_len = len(palette) or 1
        self._rois = []
        for i, r in enumerate(rois_raw or []):
            name = r.get("name", f"ROI_{i}")
            polygon = r.get("polygon")
            polygon = np.asarray(polygon, dtype=np.int32) if polygon is not None else None
            line = r.get("line")
            line_v = r.get("line_vicinity")
            color = palette[i % p_len]
            self._rois.append({"name": name, "polygon": polygon, "line": line, "line_vicinity": line_v, "color": color})

        if global_raw:
            name = global_raw.get("name", "global")
            polygon = global_raw.get("polygon")
            polygon = np.asarray(polygon, dtype=np.int32) if polygon is not None else None
            line = global_raw.get("line")
            line_v = global_raw.get("line_vicinity")
            color = palette[len(self._rois) % p_len]
            self._global_area = {"name": name, "polygon": polygon, "line": line, "line_vicinity": line_v, "color": color}
        else:
            self._global_area = None

    def _draw_areas(self, img: np.ndarray) -> np.ndarray:
        """Draw polygons (filled subtle), dashed lines with arrowhead, and area names/badges."""
        overlay = img.copy()
        h, w = img.shape[:2]

        # draw ROIs
        for a in self._rois:
            color = tuple(int(c) for c in a["color"])
            poly = a.get("polygon")
            if poly is not None:
                self._draw_polygon(overlay, poly, color, thickness=self.area_outline_thickness, alpha=self.area_alpha)
                # badge at centroid (empty by default unless counts provided)
                centroid = None
                try:
                    M = cv2.moments(poly)
                    if M["m00"] != 0:
                        centroid = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                except Exception:
                    centroid = None
                if centroid:
                    # subtle small circle as anchor (not full-screen)
                    self._draw_badge(overlay, centroid, text=a["name"], color=color, radius=18, text_is_label=True)

            if a.get("line") is not None:
                self._draw_dashed_line(overlay, a["line"][0], a["line"][1], color, thickness=2, dash_len=28, gap_len=14)
                self._draw_line_label(overlay, a["line"][0], a["line"][1], a["name"], color)

        # draw global area last (prominence)
        if self._global_area:
            a = self._global_area
            color = tuple(int(c) for c in a["color"])
            poly = a.get("polygon")
            if poly is not None:
                self._draw_polygon(overlay, poly, color, thickness=self.area_outline_thickness, alpha=self.area_alpha)
            if a.get("line") is not None:
                self._draw_dashed_line(overlay, a["line"][0], a["line"][1], color, thickness=3, dash_len=36, gap_len=16)
                self._draw_line_label(overlay, a["line"][0], a["line"][1], a["name"], color)

        # blend overlay onto image — overlay only contains area fills/outlines (no full-screen wash)
        # use alpha blending to let areas be subtle
        cv2.addWeighted(overlay, 1.0, img, 0.0, 0, img)
        return img

    # Count-style helpers (extracted/adapted from your CountVisualizer)
    def _draw_polygon(self, img, polygon: np.ndarray, color: tuple, thickness=2, alpha=None):
        if polygon is None:
            return
        pts = np.asarray(polygon, dtype=np.int32)
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] != 2:
            return
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        a = self.area_alpha if alpha is None else alpha
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
        # arrowhead at end
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

    def _draw_badge(self, img, center, text=None, color=(200,200,200), radius=18, text_is_label: bool = False):
        """Draw a small circular badge with optional text; used to mark ROI centroids."""
        x, y = int(center[0]), int(center[1])
        # shadow
        cv2.circle(img, (x + 2, y + 2), radius, (0, 0, 0), -1)
        cv2.circle(img, (x, y), radius, color, -1)
        if text_is_label and isinstance(text, str):
            # draw a short label (area name) centered next to or inside badge if short
            txt = text if len(text) <= 8 else (text[:7] + "…")
            (w, h), _ = cv2.getTextSize(txt, self.font, self.font_scale * 0.85, 1)
            cv2.putText(img, txt, (x - w // 2, y + h // 2 - 2), self.font, self.font_scale * 0.85, (255,255,255), 1, cv2.LINE_AA)
        elif text is not None and not text_is_label:
            # numeric or count
            txt = str(text)
            (w, h), _ = cv2.getTextSize(txt, self.font, self.font_scale * 0.95, 2)
            cv2.putText(img, txt, (x - w // 2, y + h // 2 - 2), self.font, self.font_scale * 0.95, (255,255,255), 2, cv2.LINE_AA)

    def _draw_line_label(self, img, p0, p1, label: str, color, offset_px: int = 18):
        mx = (p0[0] + p1[0]) / 2.0
        my = (p0[1] + p1[1]) / 2.0
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        norm = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / norm, dx / norm
        cx = mx + nx * offset_px
        cy = my + ny * offset_px
        (w, h), baseline = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        x_org = int(cx - w / 2)
        y_org = int(cy - h / 2)
        bg = tuple(max(0, int(c - 40)) for c in color)
        self._put_text_box(img, label, (x_org, y_org), bg_color=bg, alpha=0.85)

    def _put_text_box(self, img, text: str, org, bg_color=(0, 0, 0), alpha=0.6, pad=8):
        (w, h), baseline = cv2.getTextSize(text, self.font, self.font_scale, self.font_thickness)
        x, y = int(org[0]), int(org[1])
        x0, y0 = x - pad, y - pad
        x1, y1 = x + w + pad, y + h + pad
        # clamp
        h_img, w_img = img.shape[:2]
        x0, x1 = max(0, x0), min(w_img - 1, x1)
        y0, y1 = max(0, y0), min(h_img - 1, y1)
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), bg_color, cv2.FILLED)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        # shadow + text
        cv2.putText(img, text, (x, y + h), self.font, self.font_scale, (0, 0, 0), self.font_thickness + 2, cv2.LINE_AA)
        cv2.putText(img, text, (x, y + h), self.font, self.font_scale, (255, 255, 255), self.font_thickness, cv2.LINE_AA)

    # -------------------------
    # Legend / Summary widgets (draw when user passes counts)
    # -------------------------
    def _draw_legend(self, img, unique_classes: Sequence[int], origin=(12, 12)):
        x0, y0 = origin
        pad = 8
        box_h = 20
        spacing = 6
        lines = [self.class_labels.get(cls, f"cls_{cls}") for cls in unique_classes]
        max_w = 0
        for txt in lines:
            (w, h), _ = cv2.getTextSize(txt, self.font, self.font_scale * 0.9, 1)
            if w > max_w:
                max_w = w
        panel_w = max_w + 3 * pad + box_h
        panel_h = len(lines) * (box_h + spacing) + pad
        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
        y = y0 + pad
        for i, cls in enumerate(unique_classes):
            col = self._color_for(i)
            cv2.rectangle(img, (x0 + pad, y), (x0 + pad + box_h, y + box_h), col, cv2.FILLED)
            name = self.class_labels.get(cls, f"cls_{cls}")
            cv2.putText(img, name, (x0 + pad + box_h + pad // 2, y + box_h - 4), self.font, self.font_scale * 0.9, (255, 255, 255), 1, cv2.LINE_AA)
            y += box_h + spacing

    def _draw_summary_panel(self, img: np.ndarray, totals: Dict[str, int], origin=None):
        """Small per-area summary panel (no summed total) using area colors."""
        h_img, w_img = img.shape[:2]
        padding = 12
        row_h = 20
        title_h = 26
        area_order = list(totals.keys())
        n = len(area_order)
        panel_w = 280
        panel_h = title_h + n * (row_h + 6) + padding
        if origin is None:
            x0, y0 = w_img - panel_w - 12, 12
        else:
            x0, y0 = origin

        overlay = img.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), cv2.FILLED)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

        # title
        title = "Areas"
        cv2.putText(img, title, (x0 + 12, y0 + 18), self.font, self.font_scale * 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # area rows
        y = y0 + title_h
        box_sz = 14
        for i, name in enumerate(area_order):
            val = totals.get(name, 0)
            col = self._color_for(i)
            pill_x = x0 + 14
            pill_y = int(y + i * (row_h + 6)) + 6
            cv2.rectangle(img, (pill_x, pill_y), (pill_x + box_sz, pill_y + box_sz), col, cv2.FILLED)
            # area name
            txt_x = pill_x + box_sz + 10
            txt_y = pill_y + box_sz - 2
            cv2.putText(img, name, (txt_x, txt_y), self.font, self.font_scale * 0.9, (240, 240, 240), 1, cv2.LINE_AA)
            # count right aligned
            cnt_txt = str(val)
            (w_cnt, _), _ = cv2.getTextSize(cnt_txt, self.font, self.font_scale * 0.95, 2)
            cnt_x = x0 + panel_w - 14 - w_cnt
            cv2.putText(img, cnt_txt, (cnt_x, txt_y), self.font, self.font_scale * 0.95, (255, 255, 255), 2, cv2.LINE_AA)

        return img

    # -------------------------
    # Traces & drawing utilities (existing behavior preserved)
    # -------------------------
    def _increment_absent_all(self):
        for tid in list(self._traces.keys()):
            self._absent_counts[tid] = self._absent_counts.get(tid, 0) + 1

    def _increment_absent_all_and_prune(self):
        self._increment_absent_all()
        self._prune_stale_traces()

    def _prune_stale_traces(self):
        if self.linger_frames < 0:
            return
        for tid in list(self._traces.keys()):
            absent = self._absent_counts.get(tid, 0)
            if absent > self.linger_frames:
                self._traces.pop(tid, None)
                self._absent_counts.pop(tid, None)

    def _update_traces(self, ids, x1s, y1s, x2s, y2s, frame_w, frame_h):
        for raw_id, x1, y1, x2, y2 in zip(ids, x1s, y1s, x2s, y2s):
            if np.isnan(raw_id):
                continue
            tid = int(raw_id)
            cx = int((int(x1) + int(x2)) / 2)
            cy = int((int(y1) + int(y2)) / 2)
            cx = max(0, min(frame_w-1, cx))
            cy = max(0, min(frame_h-1, cy))
            self._traces[tid].append((cx, cy))
            self._absent_counts[tid] = 0

    def _draw_all_trails(self, img: np.ndarray) -> np.ndarray:
        overlay = img.copy()
        for tid, dq in self._traces.items():
            if len(dq) < 2:
                continue
            color = self._color_from_id(tid)
            pts = list(dq)
            n = len(pts)
            for i in range(1, n):
                p0 = pts[i-1]; p1 = pts[i]
                age_factor = (i / n)
                thickness = max(1, int(self.trail_thickness * (0.6 + 0.8 * age_factor)))
                cv2.line(overlay, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, self.trail_alpha, img, 1 - self.trail_alpha, 0, img)
        return img

    def _draw_box(self, img: np.ndarray, bbox: Tuple[int,int,int,int], color: Tuple[int,int,int]):
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=self.box_thickness, lineType=cv2.LINE_AA)
        overlay = img.copy()
        glow_thickness = max(1, int(self.box_thickness * 4))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=glow_thickness, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

    def _draw_label_tag(self, img: np.ndarray, bbox: Tuple[int,int,int,int], tid: int, cls_idx: int, color: Tuple[int,int,int], conf: float):
        x1, y1, x2, y2 = bbox
        # use class_labels mapping if available
        cls_name = self.class_labels.get(int(cls_idx), str(cls_idx))
        label = f"{cls_name} [{tid}]"
        conf_pct = int(round(conf * 100))

        ((tw, th), _) = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
        ((ctw, cth), _) = cv2.getTextSize(f"{conf_pct}%", self.font, self.font_scale * 0.9, self.font_thickness)
        pad_x, pad_y = 6, 4

        tag_w = tw + 2 * pad_x + ctw + pad_x
        tag_h = max(th, cth) + 2 * pad_y

        tag_x1 = x1
        tag_y2 = max(0, y1)
        tag_y1 = tag_y2 - tag_h
        tag_x2 = tag_x1 + tag_w

        if tag_y1 < 0:
            tag_y1 = y1
            tag_y2 = y1 + tag_h

        img_h, img_w = img.shape[:2]
        if tag_x2 > img_w:
            tag_x2 = img_w - 1
            tag_x1 = tag_x2 - tag_w
            if tag_x1 < 0:
                tag_x1 = 0

        overlay = img.copy()
        fill_color = tuple(max(0, min(255, int(c * 0.95))) for c in color)
        cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), fill_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)

        text_x = tag_x1 + pad_x
        text_y = tag_y2 - pad_y - 2
        cv2.putText(img, label, (text_x, text_y), self.font, self.font_scale, (255,255,255),
                    thickness=self.font_thickness, lineType=cv2.LINE_AA)

        conf_text = f"{conf_pct}%"
        conf_x = tag_x2 - pad_x - ctw
        conf_y = tag_y2 - pad_y - 2
        cv2.putText(img, conf_text, (conf_x, conf_y), self.font, self.font_scale * 0.9, (255,255,255),
                    thickness=self.font_thickness, lineType=cv2.LINE_AA)

    def _draw_confidence_bar(self, img: np.ndarray, bbox: Tuple[int,int,int,int], conf: float):
        x1, y1, x2, y2 = bbox
        bar_h = max(4, int(6 * self.font_scale))
        bar_w = max(40, int((x2 - x1) * 0.6))
        bx2 = x2
        bx1 = bx2 - bar_w
        by2 = max(0, y1 - 6)
        by1 = by2 - bar_h
        if by1 < 0:
            by1 = y1
            by2 = by1 + bar_h

        cv2.rectangle(img, (bx1, by1), (bx2, by2), (50,50,50), thickness=-1, lineType=cv2.LINE_AA)

        conf = max(0.0, min(1.0, conf))
        fill_w = int(bar_w * conf)
        if conf >= 0.7:
            fill_color = (50, 220, 50)
        elif conf >= 0.4:
            fill_color = (40, 200, 200)
        else:
            fill_color = (30, 120, 220)

        if fill_w > 0:
            cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), fill_color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.rectangle(img, (bx1, by1), (bx2, by2), (200,200,200), thickness=1, lineType=cv2.LINE_AA)

    # -------------------------
    # Color utilities
    # -------------------------
    def _color_for_class(self, cls_idx: int) -> Tuple[int,int,int]:
        if cls_idx in self._class_color_cache:
            return self._class_color_cache[cls_idx]
        color = self.palette[hash(cls_idx) % len(self.palette)]
        self._class_color_cache[cls_idx] = color
        return color

    def _color_for(self, idx: int) -> Tuple[int,int,int]:
        """Generic color pick (used by legend/summary)."""
        return self.area_palette[idx % len(self.area_palette)]

    def _color_from_id(self, tid: int) -> Tuple[int,int,int]:
        r = (tid * 37) % 255
        g = (tid * 59) % 255
        b = (tid * 83) % 255
        return (int(b), int(g), int(r))






















# import cv2
# import numpy as np
# from collections import deque, defaultdict
# from typing import Optional, Tuple, Dict, Iterable, Union


# class TrackVisualizer:
#     DEFAULT_PALETTE = [
#         (255, 127, 14),  # orange
#         (44, 160, 44),   # green
#         (31, 119, 180),  # blue
#         (214, 39, 40),   # red
#         (148, 103, 189), # purple
#         (140, 86, 75),   # brown
#         (227, 119, 194), # pink
#         (127, 127, 127), # gray
#         (188, 189, 34),  # olive
#         (23, 190, 207),  # cyan
#     ]

#     def __init__(
#         self,
#         max_trace_len: int = 30,
#         trail_thickness: int = 2,
#         trail_alpha: float = 0.65,
#         box_thickness: int = 2,
#         font_scale: float = 0.6,
#         font_thickness: int = 1,
#         palette: Optional[Iterable[Tuple[int,int,int]]] = None,
#         linger_frames: int = 0,          # <--- number of frames to keep traces after disappearance
#     ):
#         self.max_trace_len = int(max_trace_len)
#         self.trail_thickness = int(trail_thickness)
#         self.trail_alpha = float(trail_alpha)
#         self.box_thickness = int(box_thickness)
#         self.font_scale = float(font_scale)
#         self.font_thickness = int(font_thickness)
#         self.font = cv2.FONT_HERSHEY_SIMPLEX

#         self.palette = list(palette) if palette is not None else list(self.DEFAULT_PALETTE)

#         # traces: id -> deque of center (x,y)
#         self._traces: Dict[int, deque] = defaultdict(lambda: deque(maxlen=self.max_trace_len))

#         # absent counters: id -> number of consecutive frames the id was NOT observed
#         self._absent_counts: Dict[int, int] = defaultdict(int)

#         # cache mapping from class -> color
#         self._class_color_cache: Dict[int, Tuple[int,int,int]] = {}

#         # how many frames to keep trails after disappearance (0 = remove immediately)
#         self.linger_frames = int(linger_frames)

#     # -------------------------
#     # Public API
#     # -------------------------
#     def annotate(self, frame_data: Union[np.ndarray, dict], tracked_array: np.ndarray, draw_trails: bool = True) -> np.ndarray:
#         """
#         Annotate and return a copy of the frame.

#         frame_data: image array or {'frame': image array}
#         tracked_array: np.ndarray (N,8) -> (x1,y1,x2,y2,id,conf,cls,ind)
#         """
#         # extract frame
#         if isinstance(frame_data, dict):
#             frame = frame_data.get("frame")
#             if frame is None:
#                 raise ValueError("frame_data dict must contain key 'frame'")
#         else:
#             frame = frame_data

#         if frame is None:
#             raise ValueError("No frame provided")

#         # work on a copy
#         out = frame.copy()
#         h, w = out.shape[:2]

#         # safety on tracked_array size / type
#         if tracked_array is None or len(tracked_array) == 0:
#             # If no detections this frame, increment absent counts and prune based on linger_frames
#             self._increment_absent_all_and_prune()
#             return out

#         arr = np.asarray(tracked_array)
#         if arr.ndim != 2 or arr.shape[1] < 8:
#             raise ValueError("tracked_array must be shape (N,8) with columns (x1,y1,x2,y2,id,conf,cls,ind)")

#         x1s = arr[:, 0].astype(int)
#         y1s = arr[:, 1].astype(int)
#         x2s = arr[:, 2].astype(int)
#         y2s = arr[:, 3].astype(int)
#         ids  = arr[:, 4]
#         confs= arr[:, 5]
#         clss = arr[:, 6].astype(int)

#         # increment absent counters for stored ids before updating (assume absent this frame until seen)
#         self._increment_absent_all()

#         # update traces (this also resets absent count for seen ids)
#         self._update_traces(ids, x1s, y1s, x2s, y2s, w, h)

#         # prune any traces that exceeded linger_frames
#         self._prune_stale_traces()

#         # draw trails first (so they appear under boxes)
#         if draw_trails:
#             out = self._draw_all_trails(out)

#         # draw boxes, labels, conf
#         for (x1,y1,x2,y2, raw_id, raw_conf, cls_idx) in zip(x1s,y1s,x2s,y2s, ids, confs, clss):
#             # normalize / clamp
#             x1 = max(0, min(w-1, int(x1))); y1 = max(0, min(h-1, int(y1)))
#             x2 = max(0, min(w-1, int(x2))); y2 = max(0, min(h-1, int(y2)))
#             tid = int(raw_id) if not np.isnan(raw_id) else -1

#             # normalize confidence (accept 0..1 or 0..100)
#             conf = float(raw_conf)
#             if conf > 1.0:               # treat as percentage (0..100)
#                 conf = conf / 100.0
#             conf = max(0.0, min(1.0, conf))

#             color = self._color_for_class(cls_idx)

#             self._draw_box(out, (x1,y1,x2,y2), color)
#             self._draw_confidence_bar(out, (x1,y1,x2,y2), conf)
#             self._draw_label_tag(out, (x1,y1,x2,y2), tid, cls_idx, color, conf)

#         return out

#     # -------------------------
#     # Absent-counter helpers & trace pruning
#     # -------------------------
#     def _increment_absent_all(self):
#         """Mark all currently stored ids as one frame more absent (will be reset if seen this frame)."""
#         for tid in list(self._traces.keys()):
#             self._absent_counts[tid] = self._absent_counts.get(tid, 0) + 1

#     def _increment_absent_all_and_prune(self):
#         """Used when there are no detections this frame: increment and prune."""
#         self._increment_absent_all()
#         self._prune_stale_traces()

#     def _prune_stale_traces(self):
#         """Remove trace entries for ids absent for more than linger_frames."""
#         if self.linger_frames < 0:
#             return
#         for tid in list(self._traces.keys()):
#             absent = self._absent_counts.get(tid, 0)
#             if absent > self.linger_frames:
#                 self._traces.pop(tid, None)
#                 self._absent_counts.pop(tid, None)

#     # -------------------------
#     # Traces update & drawing (unchanged logic, resets absent counts for seen ids)
#     # -------------------------
#     def _update_traces(self, ids, x1s, y1s, x2s, y2s, frame_w, frame_h):
#         for raw_id, x1, y1, x2, y2 in zip(ids, x1s, y1s, x2s, y2s):
#             if np.isnan(raw_id):
#                 continue
#             tid = int(raw_id)
#             cx = int((int(x1) + int(x2)) / 2)
#             cy = int((int(y1) + int(y2)) / 2)
#             cx = max(0, min(frame_w-1, cx))
#             cy = max(0, min(frame_h-1, cy))
#             self._traces[tid].append((cx, cy))
#             # reset absent counter because this id is observed this frame
#             self._absent_counts[tid] = 0

#     def _draw_all_trails(self, img: np.ndarray) -> np.ndarray:
#         overlay = img.copy()
#         for tid, dq in self._traces.items():
#             if len(dq) < 2:
#                 continue
#             color = self._color_from_id(tid)
#             pts = list(dq)
#             n = len(pts)
#             for i in range(1, n):
#                 p0 = pts[i-1]; p1 = pts[i]
#                 age_factor = (i / n)
#                 thickness = max(1, int(self.trail_thickness * (0.6 + 0.8 * age_factor)))
#                 cv2.line(overlay, p0, p1, color, thickness=thickness, lineType=cv2.LINE_AA)
#         cv2.addWeighted(overlay, self.trail_alpha, img, 1 - self.trail_alpha, 0, img)
#         return img

#     def _draw_box(self, img: np.ndarray, bbox: Tuple[int,int,int,int], color: Tuple[int,int,int]):
#         x1, y1, x2, y2 = bbox
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=self.box_thickness, lineType=cv2.LINE_AA)
#         overlay = img.copy()
#         glow_thickness = max(1, int(self.box_thickness * 4))
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=glow_thickness, lineType=cv2.LINE_AA)
#         cv2.addWeighted(overlay, 0.08, img, 0.92, 0, img)

#     def _draw_label_tag(self, img: np.ndarray, bbox: Tuple[int,int,int,int], tid: int, cls_idx: int, color: Tuple[int,int,int], conf: float):
#         x1, y1, x2, y2 = bbox
#         label = f"{cls_idx} [{tid}]"
#         conf_pct = int(round(conf * 100))

#         ((tw, th), _) = cv2.getTextSize(label, self.font, self.font_scale, self.font_thickness)
#         ((ctw, cth), _) = cv2.getTextSize(f"{conf_pct}%", self.font, self.font_scale * 0.9, self.font_thickness)
#         pad_x, pad_y = 6, 4

#         tag_w = tw + 2 * pad_x + ctw + pad_x
#         tag_h = max(th, cth) + 2 * pad_y

#         tag_x1 = x1
#         tag_y2 = max(0, y1)
#         tag_y1 = tag_y2 - tag_h
#         tag_x2 = tag_x1 + tag_w

#         if tag_y1 < 0:
#             tag_y1 = y1
#             tag_y2 = y1 + tag_h

#         img_h, img_w = img.shape[:2]
#         if tag_x2 > img_w:
#             tag_x2 = img_w - 1
#             tag_x1 = tag_x2 - tag_w
#             if tag_x1 < 0:
#                 tag_x1 = 0

#         overlay = img.copy()
#         fill_color = tuple(max(0, min(255, int(c * 0.95))) for c in color)
#         cv2.rectangle(overlay, (tag_x1, tag_y1), (tag_x2, tag_y2), fill_color, thickness=-1, lineType=cv2.LINE_AA)
#         cv2.addWeighted(overlay, 0.90, img, 0.10, 0, img)

#         text_x = tag_x1 + pad_x
#         text_y = tag_y2 - pad_y - 2
#         cv2.putText(img, label, (text_x, text_y), self.font, self.font_scale, (255,255,255),
#                     thickness=self.font_thickness, lineType=cv2.LINE_AA)

#         conf_text = f"{conf_pct}%"
#         conf_x = tag_x2 - pad_x - ctw
#         conf_y = tag_y2 - pad_y - 2
#         cv2.putText(img, conf_text, (conf_x, conf_y), self.font, self.font_scale * 0.9, (255,255,255),
#                     thickness=self.font_thickness, lineType=cv2.LINE_AA)

#     def _draw_confidence_bar(self, img: np.ndarray, bbox: Tuple[int,int,int,int], conf: float):
#         x1, y1, x2, y2 = bbox
#         bar_h = max(4, int(6 * self.font_scale))
#         bar_w = max(40, int((x2 - x1) * 0.6))
#         bx2 = x2
#         bx1 = bx2 - bar_w
#         by2 = max(0, y1 - 6)
#         by1 = by2 - bar_h
#         if by1 < 0:
#             by1 = y1
#             by2 = by1 + bar_h

#         cv2.rectangle(img, (bx1, by1), (bx2, by2), (50,50,50), thickness=-1, lineType=cv2.LINE_AA)

#         conf = max(0.0, min(1.0, conf))
#         fill_w = int(bar_w * conf)
#         if conf >= 0.7:
#             fill_color = (50, 220, 50)  # green
#         elif conf >= 0.4:
#             fill_color = (40, 200, 200)  # medium
#         else:
#             fill_color = (30, 120, 220)  # low

#         if fill_w > 0:
#             cv2.rectangle(img, (bx1, by1), (bx1 + fill_w, by2), fill_color, thickness=-1, lineType=cv2.LINE_AA)

#         cv2.rectangle(img, (bx1, by1), (bx2, by2), (200,200,200), thickness=1, lineType=cv2.LINE_AA)

#     # -------------------------
#     # Color utilities
#     # -------------------------
#     def _color_for_class(self, cls_idx: int) -> Tuple[int,int,int]:
#         if cls_idx in self._class_color_cache:
#             return self._class_color_cache[cls_idx]
#         color = self.palette[hash(cls_idx) % len(self.palette)]
#         self._class_color_cache[cls_idx] = color
#         return color

#     def _color_from_id(self, tid: int) -> Tuple[int,int,int]:
#         r = (tid * 37) % 255
#         g = (tid * 59) % 255
#         b = (tid * 83) % 255
#         return (int(b), int(g), int(r))














