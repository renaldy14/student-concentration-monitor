"""
preprocessing.py — Modul preprocessing gambar
Menangani peningkatan kontras (CLAHE), resize, dan denoise.
"""

import cv2


def apply_clahe(frame, clip_limit=2.0, grid_size=(8, 8)):
    """
    Meningkatkan kontras menggunakan CLAHE.
    Bekerja di channel L (lightness) pada color space LAB,
    sehingga hanya kontras yang berubah, warna tetap.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def resize_frame(frame, max_width=1280):
    """
    Resize frame jika terlalu besar, menjaga aspect ratio.
    Membantu mempercepat deteksi tanpa kehilangan detail penting.
    """
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return frame


def denoise_frame(frame, kernel_size=5):
    """
    Mengurangi noise menggunakan Gaussian Blur.
    Kernel size rendah agar detail wajah tidak hilang.
    """
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def preprocess(frame, use_clahe=True, max_width=1280, use_denoise=False):
    """
    Pipeline preprocessing lengkap.
    Urutan: resize → CLAHE → denoise (opsional).
    """
    frame = resize_frame(frame, max_width)

    if use_clahe:
        frame = apply_clahe(frame)

    if use_denoise:
        frame = denoise_frame(frame)

    return frame