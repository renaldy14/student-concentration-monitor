"""
features.py — Modul ekstraksi fitur dari landmark wajah.
Menghitung Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR),
dan Head Pose (pitch, yaw, roll) untuk klasifikasi konsentrasi.
"""

import cv2
import numpy as np


# Landmark Index Constants
# MediaPipe FaceMesh 468-point indices.
# Diberi nama eksplisit agar tidak ada magic number.

# Mata kanan (6 titik untuk formula EAR)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Urutan: outer → upper_outer → upper_inner → inner → lower_inner → lower_outer

# Mata kiri (6 titik untuk formula EAR)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Mulut (4 titik untuk formula MAR)
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Head pose (6 titik kunci wajah untuk solvePnP)
HEAD_POSE_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291]
# Urutan: nose_tip, chin, left_eye, right_eye, left_mouth, right_mouth

# Model 3D generik wajah (dalam mm) — dipasangkan dengan HEAD_POSE_LANDMARK_INDICES
FACE_3D_MODEL = np.array([
    [0.0, 0.0, 0.0],            # Nose tip
    [0.0, -330.0, -65.0],       # Chin
    [-225.0, 170.0, -135.0],    # Left eye corner
    [225.0, 170.0, -135.0],     # Right eye corner
    [-150.0, -150.0, -125.0],   # Left mouth corner
    [150.0, -150.0, -125.0],    # Right mouth corner
], dtype=np.float64)


# Private Helpers
def _euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Hitung jarak Euclidean antara dua titik 2D."""
    return float(np.linalg.norm(point_a - point_b))


def _build_camera_matrix(frame_width: int, frame_height: int) -> np.ndarray:
    """
    Bangun camera intrinsic matrix.
    Asumsi: focal length = lebar frame, principal point = tengah frame.
    """
    focal_length = float(frame_width)
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0

    return np.array([
        [focal_length, 0.0, center_x],
        [0.0, focal_length, center_y],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


# Public Feature Functions
def calculate_ear(landmarks: np.ndarray, eye_indices: list) -> float:
    """
    Hitung Eye Aspect Ratio (EAR) untuk satu mata.

    Formula: EAR = (||p2-p6|| + ||p3-p5||) / (2 × ||p1-p4||)

    Interpretasi:
        EAR tinggi (~0.25-0.35) → mata terbuka
        EAR rendah (<0.20)      → mata menutup / mengantuk
    """
    p1 = landmarks[eye_indices[0]][:2]  # outer corner
    p2 = landmarks[eye_indices[1]][:2]  # upper outer
    p3 = landmarks[eye_indices[2]][:2]  # upper inner
    p4 = landmarks[eye_indices[3]][:2]  # inner corner
    p5 = landmarks[eye_indices[4]][:2]  # lower inner
    p6 = landmarks[eye_indices[5]][:2]  # lower outer

    vertical_a = _euclidean_distance(p2, p6)
    vertical_b = _euclidean_distance(p3, p5)
    horizontal = _euclidean_distance(p1, p4)

    if horizontal == 0:
        return 0.0

    return (vertical_a + vertical_b) / (2.0 * horizontal)


def calculate_average_ear(landmarks: np.ndarray) -> float:
    """
    Rata-rata EAR dari kedua mata.
    Menggunakan rata-rata karena satu mata bisa berkedip
    sementara yang lain tetap terbuka.
    """
    right_ear = calculate_ear(landmarks, RIGHT_EYE_INDICES)
    left_ear = calculate_ear(landmarks, LEFT_EYE_INDICES)
    return (right_ear + left_ear) / 2.0


def calculate_mar(landmarks: np.ndarray) -> float:
    """
    Hitung Mouth Aspect Ratio (MAR).

    Formula: MAR = jarak vertikal bibir / jarak horizontal bibir

    Interpretasi:
        MAR tinggi (>0.6) → mulut terbuka lebar (menguap)
        MAR rendah (<0.3) → mulut tertutup normal
    """
    upper = landmarks[UPPER_LIP][:2]
    lower = landmarks[LOWER_LIP][:2]
    left = landmarks[MOUTH_LEFT][:2]
    right = landmarks[MOUTH_RIGHT][:2]

    vertical = _euclidean_distance(upper, lower)
    horizontal = _euclidean_distance(left, right)

    if horizontal == 0:
        return 0.0

    return vertical / horizontal


def calculate_head_pose(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> tuple | None:
    """
    Estimasi orientasi kepala menggunakan cv2.solvePnP.

    Mencocokkan 6 titik wajah 2D (dari landmark) dengan model 3D generik,
    lalu menghitung sudut rotasi kepala relatif terhadap kamera.

    Returns:
        (pitch, yaw, roll) dalam derajat, atau None jika gagal.
        - pitch positif → menunduk
        - yaw positif   → menoleh kanan
        - roll positif  → miring kanan
    """
    image_points = np.array(
        [landmarks[idx][:2] for idx in HEAD_POSE_LANDMARK_INDICES],
        dtype=np.float64,
    )

    camera_matrix = _build_camera_matrix(frame_width, frame_height)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, _ = cv2.solvePnP(
        FACE_3D_MODEL, image_points, camera_matrix, dist_coeffs
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)

    return angles[0], angles[1], angles[2]  # pitch, yaw, roll


def extract_all(
    landmarks: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> dict | None:
    """
    Ekstrak seluruh fitur dari satu set landmark wajah.
    Entry point utama yang menggabungkan EAR, MAR, dan Head Pose.

    Returns:
        dict berisi semua fitur numerik, atau None jika input tidak valid.
    """
    if landmarks is None or len(landmarks) == 0:
        return None

    head_pose = calculate_head_pose(landmarks, frame_width, frame_height)
    if head_pose is None:
        return None

    pitch, yaw, roll = head_pose

    return {
        "ear": calculate_average_ear(landmarks),
        "mar": calculate_mar(landmarks),
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }