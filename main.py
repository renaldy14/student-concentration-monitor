"""
main.py — Student Concentration Monitor
Entry point utama dengan dukungan webcam dan video file.

Usage:
    python main.py                          → webcam mode
    python main.py --video classroom.mp4    → video file mode
    python main.py --report                 → webcam + generate AI report
    python main.py --video vid.mp4 --report → video + AI report
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from src.classifier import (
    STATE_ALERT,
    STATE_DISTRACTED,
    STATE_DROWSY,
    STATE_YAWNING,
    ConcentrationClassifier,
)
from src.detection import crop_face, detect_faces, load_model
from src.features import extract_all
from src.landmarks import FaceLandmarker
from src.preprocessing import preprocess
from src.report import SessionTracker, generate_report, save_report
from src.visualizer import Visualizer


WINDOW_TITLE = "Student Concentration Monitor"


# ─── Argument Parsing ────────────────────────────────────────

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitoring konsentrasi mahasiswa menggunakan Computer Vision",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Path ke file video. Default: webcam.",
    )
    parser.add_argument(
        "--report", action="store_true",
        help="Generate laporan konsentrasi via Gemini AI setelah sesi.",
    )
    return parser.parse_args()


# ─── Video Source ────────────────────────────────────────────

def open_video_source(video_path: str | None) -> tuple[cv2.VideoCapture, int]:
    """
    Buka webcam atau video file.
    Returns (capture, frame_delay_ms).
    """
    if video_path:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        delay = int(1000 / fps) if fps > 0 else 33
        label = f"Video: {video_path}"
    else:
        cap = cv2.VideoCapture(0)
        delay = 1
        label = "Webcam"

    if not cap.isOpened():
        raise RuntimeError(
            f"Tidak bisa membuka {label}. "
            "Pastikan file ada atau kamera diizinkan di System Settings."
        )

    print(f"📷 {label} aktif. Tekan 'q' untuk keluar.\n")
    return cap, delay


# ─── Per-Frame Pipeline ─────────────────────────────────────

def process_faces(
    frame: np.ndarray,
    faces: list[dict],
    landmarker: FaceLandmarker,
    classifier: ConcentrationClassifier,
    frame_w: int,
    frame_h: int,
) -> list[dict]:
    """
    Pipeline per-frame: crop → landmarks → features → classify.
    Mengembalikan list of face_result dicts siap di-render.
    """
    results = []

    for face in faces:
        bbox = face["bbox"]
        face_crop, offset = crop_face(frame, bbox)
        landmarks = landmarker.extract(face_crop, offset)

        if landmarks is None:
            continue

        features = extract_all(landmarks, frame_w, frame_h)

        if features is None:
            continue

        classification = classifier.classify(features)

        results.append({
            "bbox": bbox,
            "features": features,
            "classification": classification,
        })

    return results


# ─── Session Summary ─────────────────────────────────────────

def print_session_summary(summary: dict) -> None:
    """Cetak statistik dasar ke terminal."""
    pct = summary["state_percentages"]
    duration_min = summary["duration_seconds"] / 60

    print("\n" + "=" * 50)
    print("📊 RINGKASAN SESI")
    print("=" * 50)
    print(f"   Durasi       : {duration_min:.1f} menit")
    print(f"   Total frame  : {summary['total_frames']}")
    print(f"   Konsentrasi  : {summary['avg_concentration']:.1f}%")
    print(f"   ├─ Alert     : {pct[STATE_ALERT]:.1f}%")
    print(f"   ├─ Drowsy    : {pct[STATE_DROWSY]:.1f}%")
    print(f"   ├─ Distracted: {pct[STATE_DISTRACTED]:.1f}%")
    print(f"   └─ Yawning   : {pct[STATE_YAWNING]:.1f}%")
    print("=" * 50)


# ─── Main ────────────────────────────────────────────────────

def main() -> None:
    args = parse_arguments()

    # Init
    print("🔄 Loading models...")
    yolo_model = load_model()
    landmarker = FaceLandmarker()
    classifier = ConcentrationClassifier()
    visualizer = Visualizer()
    tracker = SessionTracker()
    print("✅ Semua model loaded.\n")

    try:
        cap, frame_delay = open_video_source(args.video)
    except RuntimeError as err:
        print(f"❌ {err}")
        return

    # ─── Main Loop ───────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            if args.video:
                print("📹 Video selesai diputar.")
            break

        processed = preprocess(frame)
        frame_h, frame_w = processed.shape[:2]
        faces = detect_faces(yolo_model, processed)

        face_results = process_faces(
            processed, faces, landmarker, classifier, frame_w, frame_h,
        )

        tracker.record(face_results)

        visualizer.render(processed, face_results)
        if not classifier.is_calibrated:
            visualizer.render_calibration(processed, classifier.calibration_progress)

        cv2.imshow(WINDOW_TITLE, processed)

        if cv2.waitKey(frame_delay) & 0xFF == ord("q"):
            break

    # ─── Cleanup & Report ────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    summary = tracker.get_summary()
    if summary is None:
        print("⚠️  Tidak ada data untuk diproses.")
        return

    print_session_summary(summary)

    if args.report:
        print("\n🤖 Generating AI report via Gemini...")
        report = generate_report(summary)
        saved_path = save_report(report)
        print(f"\n{'─' * 50}")
        print(report)
        print(f"{'─' * 50}")

    print("\n👋 Program selesai.")


if __name__ == "__main__":
    main()