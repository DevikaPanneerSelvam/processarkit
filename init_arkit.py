import math
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import json
from pathlib import Path
import base64
import requests
import re
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
from sklearn.cluster import DBSCAN


DOOR_CLASS_ID = 5
SWITCH_CLASS_ID = 67
MODEL_ENDPOINT = "http://localhost:8000/completion_stream"


# ============================================================
# ORIENTATION (VIDEO-LEVEL)
# ============================================================
def pixel_roll(rotvec):
    """
    Compute camera roll angle (degrees) assuming:
    - world_up = Z
    - rotation matrix is world->cam (R_is_cam2world=0)
    - image_y_down = 1 (OpenCV: origin top-left, +y down)
    """
    Rm, _ = cv2.Rodrigues(np.array(rotvec, dtype=float))
    Rc2w = Rm.T

    g_w = np.array([0, 0, 1.0], dtype=float)      # world up (Z)
    g_c = Rc2w.T @ g_w                            # gravity in camera frame

    v = np.array([g_c[0], g_c[1]], dtype=float)   # project onto image plane (x right, y down)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return None
    v /= n

    u = np.array([0.0, -1.0], dtype=float)        # image “up” when y points down
    dot = float(np.clip(u @ v, -1.0, 1.0))
    det = float(u[0] * v[1] - u[1] * v[0])
    return float(np.degrees(np.arctan2(det, dot)))  # (-180, 180]


def snap_roll_to_canonical(roll_deg, tol=20):
    """
    Snap roll angle to nearest canonical orientation.
    Only returns a value if roll is close to ±90 or ±180 (and NOT near 0).

    Returns: one of {-180, -90, 90, 180} or None
    """
    if roll_deg is None:
        return None

    roll = ((roll_deg + 180) % 360) - 180
    canonical_angles = [-180, -90, 0, 90, 180]

    best = min(canonical_angles, key=lambda a: abs(roll - a))
    if abs(roll - best) <= tol:
        if best == 0:
            return None
        return int(best)

    return None


def estimate_global_snap_roll(poses, max_samples=50, min_consensus=0.7):
    """
    Estimate a single canonical roll angle for the entire video.
    Returns one of {-180, -90, 90, 180} or None.

    Uses majority vote across up to `max_samples` poses. Optionally requires a
    minimum consensus fraction to avoid accidental rotation.
    """
    snap_rolls = []

    for i, (_, (rotvec, _)) in enumerate(poses):
        if i >= max_samples:
            break

        raw = pixel_roll(rotvec)
        snap = snap_roll_to_canonical(raw)
        if snap is not None:
            snap_rolls.append(int(snap))

    if not snap_rolls:
        return None

    counts = Counter(snap_rolls)
    angle, freq = counts.most_common(1)[0]

    if (freq / len(snap_rolls)) < float(min_consensus):
        return None

    return int(angle)


def compute_global_roll_params(global_snap_roll, frame_shape_hw):
    """
    Compute roll affine + output size once, to be reused for:
      1) rotating rgb frames via cv2.warpAffine
      2) rotating projected 2D points via the same affine

    Args:
        global_snap_roll: one of {-180, -90, 90, 180} or None
        frame_shape_hw: (H, W)

    Returns:
        (roll_affine, output_size) where:
          roll_affine: 2x3 float32 or None
          output_size: (W, H) or None
    """
    if global_snap_roll is None or int(global_snap_roll) == 0:
        return None, None

    H, W = frame_shape_hw
    A, nW, nH = compute_roll_affine(H, W, global_snap_roll)
    return A, (nW, nH)


# ============================================================
# TRAJ / POSE HELPERS
# ============================================================
def read_traj(traj_path):
    poses = {}
    with open(traj_path, "r") as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 7:
                continue
            timestamp = f"{round(float(tokens[0]), 3):.3f}"
            rot = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])], dtype=float)
            trans = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])], dtype=float)
            poses[timestamp] = (rot, trans)
    return poses


def extract_timestamp_from_filename(filename: str) -> str:
    """Extract timestamp from image filename like '40966448_496.345.png'."""
    try:
        # keep your original approach; just make it robust to extra suffixes
        base = Path(filename).name
        ts_part = base.split("_", 1)[1].rsplit(".", 1)[0]  # "496.345"
        return f"{round(float(ts_part), 3):.3f}"
    except Exception:
        return None


def get_pose_for_nearest_timestamp(target_ts: str, poses: list[tuple[float, tuple]]) -> tuple[str, tuple]:
    target_value = float(target_ts)
    closest_entry = min(poses, key=lambda item: abs(float(item[0]) - target_value))
    closest_ts = f"{round(float(closest_entry[0]), 3):.3f}"
    return closest_ts, closest_entry[1]


# ============================================================
# CORE PROJECTION (NO ORIENTATION COMPUTATION INSIDE)
# ============================================================
def project_instance(
    mesh_path,
    labels_path,
    pincam_path,
    rotation_vec,
    translation_vec,
    rgb_frame,
    alpha=0.6,
    roll_affine=None,      # passed in (2x3) or None
    output_size=None       # passed in (W, H) or None
):
    # -----------------------------
    # Load data
    # -----------------------------
    vertices, labels = load_mesh_with_labels(mesh_path, labels_path)
    width, height, K = load_intrinsics(pincam_path)
    extrinsic = pose_to_extrinsic(rotation_vec, translation_vec)

    # Transform vertices to camera coordinates
    vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1), dtype=vertices.dtype)))
    cam_coords = (extrinsic @ vertices_h.T).T[:, :3]

    # Keep points in front of camera
    in_front = cam_coords[:, 2] > 0
    cam_coords = cam_coords[in_front]
    labels = labels[in_front]

    # Project 3D → 2D (in ORIGINAL image coordinates)
    pixel_coords = (K @ cam_coords.T).T
    pixel_coords = pixel_coords[:, :2] / cam_coords[:, 2:3]

    overlay = rgb_frame.copy()

    # Determine the image bounds we should use AFTER rotation
    if output_size is not None:
        W, H = output_size
    else:
        H, W = rgb_frame.shape[:2]

    # ------------------------------------
    # APPLY PRECOMPUTED ROLL (IF ANY) TO 2D POINTS
    # ------------------------------------
    if roll_affine is not None:
        pts = np.hstack([pixel_coords, np.ones((pixel_coords.shape[0], 1), dtype=pixel_coords.dtype)])
        pixel_coords = (roll_affine @ pts.T).T[:, :2]

    # -----------------------------
    # PASS 1: FRAME-LEVEL LABEL COUNTING (ALL LABELS)
    # -----------------------------
    label_counter = Counter()
    total_pixels = 0
    contains_target_class = [False]  # door only for now

    for (u, v), label in zip(pixel_coords, labels):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            label_counter[int(label)] += 1
            total_pixels += 1
            if int(label) == DOOR_CLASS_ID:
                contains_target_class[0] = True

    # -----------------------------
    # PASS 2: 3D DOOR INSTANCE CLUSTERING
    # -----------------------------
    door_mask = labels == DOOR_CLASS_ID
    door_cam = cam_coords[door_mask]
    door_instances_3d = {}  # inst_id → indices into cam_coords

    if door_cam.shape[0] > 0:
        clustering = DBSCAN(eps=0.15, min_samples=50).fit(door_cam)
        door_cam_indices = np.where(door_mask)[0]

        for inst_id in set(clustering.labels_):
            if inst_id == -1:
                continue
            door_instances_3d[int(inst_id)] = door_cam_indices[clustering.labels_ == inst_id]

    # -----------------------------
    # PASS 3: 2D INSTANCE ASSIGNMENT (DOORS ONLY)
    # -----------------------------
    door_instances_2d = {}
    instance_colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
    ]

    for inst_id, idxs in door_instances_3d.items():
        xs, ys = [], []

        for i in idxs:
            u, v = pixel_coords[i]
            u, v = int(round(u)), int(round(v))
            if 0 <= u < W and 0 <= v < H:
                xs.append(u)
                ys.append(v)

                color = instance_colors[inst_id % len(instance_colors)]
                cv2.circle(overlay, (u, v), 2, color, -1)

        if xs and ys:
            door_instances_2d[int(inst_id)] = {
                "pixel_count": int(len(xs)),
                "bbox": (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))),
            }

    blended = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)

    return blended, contains_target_class, label_counter, total_pixels, door_instances_2d


# ============================================================
# IO / GEOMETRY
# ============================================================
def load_mesh_with_labels(mesh_path, labels_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    vertices = np.asarray(mesh.vertices)
    labels = np.loadtxt(labels_path, dtype=int)

    if labels.ndim == 0:
        labels = np.array([labels])
    elif labels.ndim > 1:
        labels = labels.squeeze()

    if len(labels) != len(vertices):
        raise ValueError(f"Mismatch: {len(vertices)} vertices vs {len(labels)} labels")

    return vertices, labels


def load_intrinsics(pincam_path: str):
    w, h, fx, fy, cx, cy = np.loadtxt(pincam_path)
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=float)
    return int(w), int(h), K


def pose_to_extrinsic(rotation_vec, translation_vec):
    R_mat = R.from_rotvec(rotation_vec).as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_mat
    T[:3, 3] = np.array(translation_vec, dtype=float)
    return T


# ============================================================
# AFFINE FOR ROLL ROTATION (USED OUTSIDE + INSIDE)
# ============================================================
def compute_roll_affine(H, W, roll_deg):
    """
    Returns affine A (2x3) and output width/height (nW, nH) to rotate an image by roll_deg.
    This is intended to be called ONCE per video, then reused.

    roll_deg in {-180, -90, 90, 180}
    """
    # OpenCV expects center in (x, y)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    A = cv2.getRotationMatrix2D((cx, cy), roll_deg, 1.0)

    # Compute new bounds
    corners = np.array([
        [0, 0, 1],
        [W - 1, 0, 1],
        [0, H - 1, 1],
        [W - 1, H - 1, 1],
    ], dtype=float)

    rc = (A @ corners.T).T
    min_x, min_y = rc[:, 0].min(), rc[:, 1].min()
    max_x, max_y = rc[:, 0].max(), rc[:, 1].max()

    nW = int(math.ceil(max_x - min_x + 1))
    nH = int(math.ceil(max_y - min_y + 1))

    # Shift so top-left is (0,0)
    A[0, 2] -= min_x
    A[1, 2] -= min_y

    return A.astype(np.float32), nW, nH
