import math
import numpy as np
import cv2
from collections import Counter
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json 
from pathlib import Path
import base64
import requests
import re
from matplotlib import pyplot as plt
from collections import defaultdict, Counter

def estimate_global_snap_roll(poses, max_samples=50):
    """
    Estimate a single canonical roll angle for the entire video.
    Returns one of {0, 90, -90, 180, -180} or None.
    """
    snap_rolls = []

    for i, (_, (rotvec, _)) in enumerate(poses):
        if i >= max_samples:
            break

        raw_roll = pixel_roll(rotvec)
        snap = snap_roll_to_canonical(raw_roll)

        if snap is not None:
            snap_rolls.append(int(snap))

    if not snap_rolls:
        return None

    # Majority vote
    return Counter(snap_rolls).most_common(1)[0][0]

def pixel_roll(rotvec):
    """
    Compute camera roll angle (degrees) assuming:
    - world_up = Z
    - rotation matrix is world->cam (R_is_cam2world=0)
    - image_y_down = 1 (OpenCV: origin top-left, +y down)
    """
    R, _ = cv2.Rodrigues(np.array(rotvec, dtype=float))  # from Rodrigues vector
    Rc2w = R.T  # because R_is_cam2world=0 → we want world->cam

    g_w = np.array([0, 0, 1.0], dtype=float)  # world up (Z)
    g_c = Rc2w.T @ g_w                         # gravity in camera frame

    v = np.array([g_c[0], g_c[1]], dtype=float)  # project onto image plane (x right, y down)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None  # undefined when looking straight up/down
    v /= n

    u = np.array([0.0, -1.0], dtype=float)  # image “up” when y points down
    dot = float(np.clip(u @ v, -1.0, 1.0))
    det = float(u[0]*v[1] - u[1]*v[0])
    return np.degrees(np.arctan2(det, dot))  # in (-180, 180]

def snap_roll_to_canonical(roll_deg, tol=20):
    """
    Snap roll angle to nearest canonical orientation.
    Only returns a value if roll is close to ±90 or ±180.
    
    Args:
        roll_deg: float (degrees)
        tol: tolerance in degrees
    
    Returns:
        snapped_angle or None
    """
    if roll_deg is None:
        return None

    # Normalize to [-180, 180]
    roll = ((roll_deg + 180) % 360) - 180

    canonical_angles = [-180, -90, 0, 90, 180]

    best = min(canonical_angles, key=lambda a: abs(roll - a))
    if abs(roll - best) <= tol:
        # Do NOT rotate for near-zero
        if best == 0:
            return None
        return best

    return None