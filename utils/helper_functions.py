from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw
import math

# ALL these functions uses vectorization to reduce time consumtion.
# These functions are used by counting with filtering by polygon
def polygon_to_mask(polygon: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    Rasterize a polygon (Nx2 array of x,y) into a tight mask.

    Returns:
        mask: 2D uint8 numpy array (1 inside polygon, 0 outside)
        x_min: int x-coordinate of mask's top-left in original image
        y_min: int y-coordinate of mask's top-left in original image

    Assumes polygon coordinates are in image pixel space (origin top-left).
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must be shape (N,2)")

    xs = polygon[:, 0]
    ys = polygon[:, 1]

    x_min = int(math.floor(xs.min()))
    y_min = int(math.floor(ys.min()))
    x_max = int(math.ceil(xs.max()))
    y_max = int(math.ceil(ys.max()))

    width = x_max - x_min + 1
    height = y_max - y_min + 1
    if width <= 0 or height <= 0:
        raise ValueError("invalid polygon bbox")

    # Shift polygon into bbox-local coordinates
    shifted = [(float(x - x_min), float(y - y_min)) for x, y in polygon]

    # Draw filled polygon into a single-channel image
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon(shifted, outline=1, fill=1)

    mask = np.asarray(img, dtype=np.uint8)
    return mask, x_min, y_min



def bbox_center_in_polygon(bboxes: np.ndarray , mask: np.ndarray, x_min: int, y_min: int) -> np.ndarray:
    """
    (N,1) bool: True if bbox center is inside polygon mask.
    """
    centers_x = ((bboxes[:,0] + bboxes[:,2]) * 0.5)
    centers_y = ((bboxes[:,1] + bboxes[:,3]) * 0.5)
    x_idx = np.floor(centers_x - x_min).astype(int)
    y_idx = np.floor(centers_y - y_min).astype(int)
    x_idx_clipped = np.clip(x_idx, 0, mask.shape[1]-1)
    y_idx_clipped = np.clip(y_idx, 0, mask.shape[0]-1)
    in_bounds = (x_idx >= 0) & (x_idx < mask.shape[1]) & (y_idx >= 0) & (y_idx < mask.shape[0])
    vals = mask[y_idx_clipped, x_idx_clipped].astype(bool)
    vals[~in_bounds] = False
    return vals.reshape(-1,1)


def bbox_corners_in_polygon(bboxes: np.ndarray, mask: np.ndarray, x_min: int, y_min: int) -> np.ndarray:
    """
    (N,1) bool: True if ANY of the 4 bbox corners is inside polygon mask.
    """
    corners = np.stack([bboxes[:, [0,1]], bboxes[:, [2,1]], bboxes[:, [0,3]], bboxes[:, [2,3]]], axis=1)  # (N,4,2)
    x_idx = np.floor(corners[:,:,0] - x_min).astype(int)
    y_idx = np.floor(corners[:,:,1] - y_min).astype(int)
    x_idx_clipped = np.clip(x_idx, 0, mask.shape[1]-1)
    y_idx_clipped = np.clip(y_idx, 0, mask.shape[0]-1)
    in_bounds = (x_idx >= 0) & (x_idx < mask.shape[1]) & (y_idx >= 0) & (y_idx < mask.shape[0])
    inside_vals = mask[y_idx_clipped, x_idx_clipped].astype(bool)
    inside_vals[~in_bounds] = False
    return np.any(inside_vals, axis=1).reshape(-1,1)


def bbox_any_in_polygon(bboxes: np.ndarray, mask: np.ndarray, x_min: int, y_min: int) -> np.ndarray:
    """
    Fully vectorized (no Python loop over boxes).
    (N,1) bool: True if ANY integer point inside the bbox lies in the polygon mask.
    Implementation:
      - Build per-box integer ranges in mask coords (x_start..x_end, y_start..y_end).
      - Broadcast to a (N, max_y, max_x) grid, ignore padded cells, lookup mask once,
        then reduce per-box with np.any.
    Notes: out-of-mask regions are treated as "outside".
    """
    N = bboxes.shape[0]
    h, w = mask.shape

    x_starts = np.floor(bboxes[:,0] - x_min).astype(int)
    x_ends   = np.ceil( bboxes[:,2] - x_min).astype(int)
    y_starts = np.floor(bboxes[:,1] - y_min).astype(int)
    y_ends   = np.ceil( bboxes[:,3] - y_min).astype(int)

    x_starts_clipped = np.clip(x_starts, 0, w-1)
    x_ends_clipped   = np.clip(x_ends,   -1, w-1)  # allow -1 to indicate empty in-mask range
    y_starts_clipped = np.clip(y_starts, 0, h-1)
    y_ends_clipped   = np.clip(y_ends,   -1, h-1)

    x_lengths = np.maximum(0, x_ends_clipped - x_starts_clipped + 1)
    y_lengths = np.maximum(0, y_ends_clipped - y_starts_clipped + 1)

    if x_lengths.sum() == 0 or y_lengths.sum() == 0:
        return np.zeros((N,1), dtype=bool)

    max_x = int(x_lengths.max())
    max_y = int(y_lengths.max())

    x_offsets = np.arange(max_x)
    y_offsets = np.arange(max_y)

    x_grid = x_starts_clipped[:, None] + x_offsets[None, :]   # (N, max_x)
    y_grid = y_starts_clipped[:, None] + y_offsets[None, :]   # (N, max_y)

    x_grid = np.clip(x_grid, 0, w-1)
    y_grid = np.clip(y_grid, 0, h-1)

    x_coords = np.broadcast_to(x_grid[:, None, :], (N, max_y, max_x))   # (N, max_y, max_x)
    y_coords = np.broadcast_to(y_grid[:, :, None], (N, max_y, max_x))

    x_valid = (x_offsets[None, :] < x_lengths[:, None])   # (N, max_x)
    y_valid = (y_offsets[None, :] < y_lengths[:, None])   # (N, max_y)
    valid = np.logical_and(y_valid[:, :, None], x_valid[:, None, :])  # (N, max_y, max_x)

    mask_vals = mask[y_coords, x_coords].astype(bool)
    mask_vals[~valid] = False

    any_inside = np.any(mask_vals, axis=(1,2)).reshape(-1,1)
    return any_inside

