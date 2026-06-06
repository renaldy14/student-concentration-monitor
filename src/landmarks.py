"""
landmarks.py — Modul ekstraksi landmark wajah menggunakan MediaPipe FaceLandmarker.
Mengekstrak 468 titik landmark 3D dari setiap crop wajah,
lalu menerjemahkan koordinatnya kembali ke frame asli.
"""
from __future__ import annotations
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# Constants
MODELS_DIR = Path(__file__).parent.parent / "models"
LANDMARKER_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
MAX_FACES_PER_CROP = 1


class FaceLandmarker:
    """
    Wrapper untuk MediaPipe FaceLandmarker Tasks API.
    Satu instance dipakai ulang sepanjang program berjalan (load model sekali saja).
    """

    def __init__(self):
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self) -> vision.FaceLandmarker: # type: ignore
        """Inisialisasi FaceLandmarker dari file .task."""
        if not LANDMARKER_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {LANDMARKER_MODEL_PATH}. "
                f"Jalankan: curl -O https://storage.googleapis.com/mediapipe-models/"
                f"face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = python.BaseOptions(
            model_asset_path=str(LANDMARKER_MODEL_PATH)
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=MAX_FACES_PER_CROP,
        )
        return vision.FaceLandmarker.create_from_options(options)

    def extract(self, face_crop: np.ndarray, offset: tuple = (0, 0)):
        """
        Ekstrak 468 landmark dari satu crop wajah.

        Args:
            face_crop: Gambar crop wajah (BGR, dari detection.crop_face).
            offset: (offset_x, offset_y) untuk translasi koordinat
                    kembali ke ruang koordinat frame asli.

        Returns:
            np.ndarray shape (468, 3) berisi koordinat (x, y, z)
            dalam ruang frame asli, atau None jika tidak terdeteksi.
        """
        if face_crop is None or face_crop.size == 0:
            return None

        crop_h, crop_w = face_crop.shape[:2]
        offset_x, offset_y = offset

        rgb_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_crop)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            return None

        # Konversi normalized landmarks → pixel coordinates + offset
        raw_landmarks = result.face_landmarks[0]
        landmarks = np.array([
            (
                lm.x * crop_w + offset_x,
                lm.y * crop_h + offset_y,
                lm.z * crop_w,  # z relatif terhadap lebar wajah
            )
            for lm in raw_landmarks
        ])

        return landmarks