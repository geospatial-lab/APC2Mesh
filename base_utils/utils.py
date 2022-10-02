import numpy as np

def right_handed_to_left_handed(pts: np.ndarray):
    pts_res = np.zeros_like(pts)
    if pts.shape[0] > 0:
        pts_res[:, 0] = pts[:, 0]
        pts_res[:, 1] = -pts[:, 2]
        pts_res[:, 2] = pts[:, 1]
    return pts_res