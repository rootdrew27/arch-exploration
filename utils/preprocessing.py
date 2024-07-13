import torch
import numpy as np

def bbs_corners_to_centers(boxes):
    """Convert from (upper-left, lower-right) to (center, width, height)."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def bbs_centers_to_corners(boxes):
    """Convert from (center, width, height) to (upper-left, lower-right)."""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def normalize_bbs(boxes:torch.Tensor | list | np.ndarray, imgsz:tuple[int]) -> torch.Tensor | list | np.ndarray :
    """Normalize the boxes to be within 0 and 1

    Args:
        boxes : Bounding Boxes  
        imgsz : The image size structured as (Width, Height)

    Returns:
        Normalized boudning boxes
    """
    boxes[:, 0], boxes[:, 2] = boxes[:, 0] / imgsz[0], boxes[:, 2] / imgsz[1]
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] / imgsz[1], boxes[:, 3] / imgsz[1]
    
    return boxes