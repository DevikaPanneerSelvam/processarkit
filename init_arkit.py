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
from sklearn.cluster import DBSCAN


DOOR_CLASS_ID = 5
SWITCH_CLASS_ID = 67
MODEL_ENDPOINT = "http://localhost:8000/completion_stream"

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

def read_traj(traj_path):
    poses = {}
    with open(traj_path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 7:
                continue
            timestamp = f"{round(float(tokens[0]), 3):.3f}"
            rot = np.array([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            trans = np.array([float(tokens[4]), float(tokens[5]), float(tokens[6])])
            poses[timestamp] = (rot, trans)
    return poses

def extract_timestamp_from_filename(filename: str) -> str:
    """Extracts timestamp from image filename like '40966448_496.345.png'."""
    try:
        return f"{round(float(filename.split('_')[1].split('.')[0] + '.' + filename.split('_')[1].split('.')[1]), 3):.3f}"
    except Exception:
        return None
    
def get_pose_for_nearest_timestamp(target_ts: str, poses: list[tuple[float, tuple]]) -> tuple[str, tuple]:
    target_value = float(target_ts)
    # Find the (timestamp, pose) pair where timestamp is closest to target_value
    closest_entry = min(poses, key=lambda item: abs(float(item[0]) - target_value))
    closest_ts = f"{round(float(closest_entry[0]), 3):.3f}"
    return closest_ts, closest_entry[1]

def project_instance(
    mesh_path,
    labels_path,
    pincam_path,
    rotation_vec,
    translation_vec,
    rgb_frame,
    alpha=0.6
):
    # -----------------------------
    # Load data
    # -----------------------------
    vertices, labels = load_mesh_with_labels(mesh_path, labels_path)
    width, height, K = load_intrinsics(pincam_path)
    extrinsic = pose_to_extrinsic(rotation_vec, translation_vec)

    # Transform vertices to camera coordinates
    vertices_h = np.hstack((vertices, np.ones((vertices.shape[0], 1))))
    cam_coords = (extrinsic @ vertices_h.T).T[:, :3]

    # Keep points in front of camera
    in_front = cam_coords[:, 2] > 0
    cam_coords = cam_coords[in_front]
    labels = labels[in_front]

    # Project 3D → 2D
    pixel_coords = (K @ cam_coords.T).T
    pixel_coords = pixel_coords[:, :2] / cam_coords[:, 2:3]

    overlay = rgb_frame.copy()
    H, W = rgb_frame.shape[:2]

    # ------------------------------------
    # APPLY SAME ROLL AS IMAGE
    # ------------------------------------
    raw_roll = pixel_roll(rotation_vec)
    snap_roll = snap_roll_to_canonical(raw_roll)

    if snap_roll is not None:
        A, _, _ = compute_roll_affine(height, width, snap_roll)

        pts = np.hstack([
            pixel_coords,
            np.ones((pixel_coords.shape[0], 1))
        ])
        pixel_coords = (A @ pts.T).T[:, :2]

    # -----------------------------
    # PASS 1: FRAME-LEVEL LABEL COUNTING (ALL LABELS)
    # -----------------------------
    label_counter = Counter()
    total_pixels = 0
    contains_target_class = [False]  # door only for now

    valid_pixel_indices = []  # indices that project inside image

    for i, ((u, v), label) in enumerate(zip(pixel_coords, labels)):
        u, v = int(round(u)), int(round(v))
        if 0 <= u < W and 0 <= v < H:
            label_counter[label] += 1
            total_pixels += 1
            valid_pixel_indices.append(i)

            if label == DOOR_CLASS_ID:
                contains_target_class[0] = True

    # -----------------------------
    # PASS 2: 3D DOOR INSTANCE CLUSTERING
    # -----------------------------
    door_mask = labels == DOOR_CLASS_ID
    door_cam = cam_coords[door_mask]

    door_instances_3d = {}  # inst_id → indices into cam_coords

    if door_cam.shape[0] > 0:
        clustering = DBSCAN(
            eps=0.15,
            min_samples=50
        ).fit(door_cam)

        door_cam_indices = np.where(door_mask)[0]

        for inst_id in set(clustering.labels_):
            if inst_id == -1:
                continue
            door_instances_3d[int(inst_id)] = door_cam_indices[
                clustering.labels_ == inst_id
            ]

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
            door_instances_2d[inst_id] = {
                "pixel_count": len(xs),
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
            }

    # Alpha blend
    blended = cv2.addWeighted(overlay, alpha, rgb_frame, 1 - alpha, 0)

    return (
        blended,
        contains_target_class,
        label_counter,        # ✅ ALL LABELS
        total_pixels,
        door_instances_2d     # door-specific instance info
    )

def load_mesh_with_labels(mesh_path, labels_path):
    #mesh = o3d.io.read_triangle_mesh(mesh_path)
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
                  [0,  0,  1]])
    return int(w), int(h), K

def pose_to_extrinsic(rotation_vec, translation_vec):
    R_mat = R.from_rotvec(rotation_vec).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = translation_vec
    return T