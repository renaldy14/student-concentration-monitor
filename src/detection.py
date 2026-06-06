"""
detection.py — Modul deteksi wajah menggunakan YOLOv8-face
Menangani download model, loading, dan deteksi wajah.
"""

from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# Path ke folder models di root project
MODELS_DIR = Path(__file__).parent.parent / "models"
YOLO_FACE_PATH = MODELS_DIR / "face_yolov8n.pt"


def download_model():
    """
    Download YOLOv8-face dari Bingsu/adetailer di HuggingFace.
    Hanya download jika file belum ada di folder models/.
    """
    if YOLO_FACE_PATH.exists():
        print("✅ YOLOv8-face model sudah ada.")
        return

    print("⏬ Downloading YOLOv8-face model dari HuggingFace...")
    hf_hub_download(
        repo_id="Bingsu/adetailer",
        filename="face_yolov8n.pt",
        local_dir=str(MODELS_DIR)
    )
    print("✅ YOLOv8-face model berhasil didownload.")


def load_model():
    """Load YOLOv8-face model ke memory."""
    download_model()
    model = YOLO(str(YOLO_FACE_PATH))
    return model


def detect_faces(model, frame, confidence=0.5):
    """
    Deteksi wajah dalam frame menggunakan YOLOv8.

    Returns:
        List of dict, masing-masing berisi:
        - 'bbox': (x1, y1, x2, y2) koordinat bounding box
        - 'confidence': skor kepercayaan deteksi (0.0 - 1.0)
    """
    results = model(frame, conf=confidence, verbose=False)

    faces = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": conf
            })

    return faces


def crop_face(frame, bbox, padding=0.3):
    """
    Crop area wajah dari frame dengan padding 30%.
    Padding diperlukan agar MediaPipe landmark detection lebih akurat
    (butuh area sekitar wajah, bukan hanya wajah saja).

    Returns:
        crop: gambar wajah yang sudah di-crop
        offset: (offset_x, offset_y) untuk translasi koordinat landmark
                kembali ke koordinat frame asli nanti di tahap selanjutnya
    """
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]

    face_w = x2 - x1
    face_h = y2 - y1
    pad_w = int(face_w * padding)
    pad_h = int(face_h * padding)

    # Clamp agar tidak keluar batas frame
    cx1 = max(0, x1 - pad_w)
    cy1 = max(0, y1 - pad_h)
    cx2 = min(w, x2 + pad_w)
    cy2 = min(h, y2 + pad_h)

    crop = frame[cy1:cy2, cx1:cx2]
    offset = (cx1, cy1)

    return crop, offset