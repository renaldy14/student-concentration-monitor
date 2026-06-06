"""
classifier.py — Modul klasifikasi status konsentrasi mahasiswa.

Revisi:
- Auto-calibration EAR baseline (adaptif terhadap bentuk mata)
- Weighted concentration (gradual, bukan binary 0/100)
- Threshold drowsiness menggunakan rasio relatif terhadap baseline,
  bukan nilai absolut — menghilangkan bias terhadap bentuk mata
"""

from __future__ import annotations

from collections import deque


# ─── State Labels ────────────────────────────────────────────
STATE_ALERT = "Alert"
STATE_DROWSY = "Drowsy"
STATE_DISTRACTED = "Distracted"
STATE_YAWNING = "Yawning"

# ─── Weighted Concentration per State ────────────────────────
# Dipakai untuk menghitung konsentrasi kelas secara gradual
STATE_CONCENTRATION_WEIGHTS: dict[str, float] = {
    STATE_ALERT: 1.00,       # 100%
    STATE_DISTRACTED: 0.75,  # 75%
    STATE_DROWSY: 0.50,      # 50%
    STATE_YAWNING: 0.15,     # 15%
}

# ─── Thresholds ──────────────────────────────────────────────
# EAR: menggunakan RASIO relatif terhadap baseline (bukan absolut)
# sehingga adaptif terhadap bentuk mata apapun
EAR_CLOSED_RATIO = 0.65           # mata dianggap tertutup jika EAR < 65% baseline
EAR_FALLBACK_THRESHOLD = 0.18     # fallback absolut jika kalibrasi gagal

MAR_YAWN_THRESHOLD = 0.55
YAW_DISTRACTED_DEG = 15.0
PITCH_DISTRACTED_DEG = 15.0
PERCLOS_DROWSY_THRESHOLD = 0.40

# ─── Temporal Settings ───────────────────────────────────────
HISTORY_MAX_FRAMES = 150
CALIBRATION_FRAME_COUNT = 30


class ConcentrationClassifier:
    """
    Klasifikasi konsentrasi dengan auto-calibration untuk EAR dan Head Pose.

    Selama ~1 detik pertama, sistem merekam:
    - Rata-rata pitch & yaw sebagai baseline posisi kepala "netral"
    - 75th percentile EAR sebagai baseline "mata terbuka normal"
      (menggunakan percentile agar robust terhadap kedipan saat kalibrasi)

    Setelah kalibrasi, drowsiness dideteksi dari RASIO EAR terhadap
    baseline — bukan dari angka absolut. Ini menghilangkan bias terhadap
    bentuk mata (sipit, lebar, dll.).
    """

    def __init__(self) -> None:
        self._ear_history: deque = deque(maxlen=HISTORY_MAX_FRAMES)
        self._calibration_buffer: list[dict] = []
        self._pitch_baseline: float = 0.0
        self._yaw_baseline: float = 0.0
        self._ear_baseline: float = 0.0
        self._is_calibrated: bool = False

    # ─── Public Properties ───────────────────────────────────

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def calibration_progress(self) -> float:
        if self._is_calibrated:
            return 1.0
        return len(self._calibration_buffer) / CALIBRATION_FRAME_COUNT

    @property
    def ear_baseline(self) -> float:
        return self._ear_baseline

    # ─── Public Methods ──────────────────────────────────────

    def classify(self, features: dict) -> dict:
        if features is None:
            return {"state": STATE_ALERT, "perclos": 0.0}

        if not self._is_calibrated:
            return self._handle_calibration(features)

        self._ear_history.append(features["ear"])
        perclos = self._calculate_perclos()
        state = self._determine_state(features, perclos)

        return {"state": state, "perclos": perclos}

    def reset(self) -> None:
        self._ear_history.clear()
        self._calibration_buffer.clear()
        self._is_calibrated = False
        self._pitch_baseline = 0.0
        self._yaw_baseline = 0.0
        self._ear_baseline = 0.0

    # ─── Private: Calibration ────────────────────────────────

    def _handle_calibration(self, features: dict) -> dict:
        self._calibration_buffer.append(features)

        if len(self._calibration_buffer) >= CALIBRATION_FRAME_COUNT:
            pitches = [f["pitch"] for f in self._calibration_buffer]
            yaws = [f["yaw"] for f in self._calibration_buffer]
            ears = sorted(f["ear"] for f in self._calibration_buffer)

            self._pitch_baseline = sum(pitches) / len(pitches)
            self._yaw_baseline = sum(yaws) / len(yaws)

            # 75th percentile EAR = baseline "mata terbuka normal"
            # Robust terhadap kedipan saat kalibrasi
            percentile_idx = int(len(ears) * 0.75)
            self._ear_baseline = ears[min(percentile_idx, len(ears) - 1)]

            self._is_calibrated = True
            print(
                f"✅ Kalibrasi selesai. Baseline → "
                f"Pitch: {self._pitch_baseline:.1f}°, "
                f"Yaw: {self._yaw_baseline:.1f}°, "
                f"EAR: {self._ear_baseline:.3f}"
            )

        return {"state": STATE_ALERT, "perclos": 0.0}

    # ─── Private: State Detection ────────────────────────────

    def _determine_state(self, features: dict, perclos: float) -> str:
        if self._is_yawning(features):
            return STATE_YAWNING
        if self._is_drowsy(features, perclos):
            return STATE_DROWSY
        if self._is_distracted(features):
            return STATE_DISTRACTED
        return STATE_ALERT

    def _is_yawning(self, features: dict) -> bool:
        return features["mar"] > MAR_YAWN_THRESHOLD

    def _is_drowsy(self, features: dict, perclos: float) -> bool:
        # Relative threshold: EAR < 65% dari baseline personal
        if self._ear_baseline > 0:
            ear_threshold = self._ear_baseline * EAR_CLOSED_RATIO
        else:
            ear_threshold = EAR_FALLBACK_THRESHOLD

        eyes_closing = features["ear"] < ear_threshold
        sustained_closure = perclos > PERCLOS_DROWSY_THRESHOLD
        return eyes_closing or sustained_closure

    def _is_distracted(self, features: dict) -> bool:
        pitch_dev = abs(features["pitch"] - self._pitch_baseline)
        yaw_dev = abs(features["yaw"] - self._yaw_baseline)
        return pitch_dev > PITCH_DISTRACTED_DEG or yaw_dev > YAW_DISTRACTED_DEG

    def _calculate_perclos(self) -> float:
        if len(self._ear_history) == 0 or self._ear_baseline <= 0:
            return 0.0

        ear_threshold = self._ear_baseline * EAR_CLOSED_RATIO
        closed_count = sum(
            1 for ear in self._ear_history if ear < ear_threshold
        )
        return closed_count / len(self._ear_history)


# ─── Module-Level Utility ────────────────────────────────────

def calculate_weighted_concentration(face_results: list[dict]) -> float:
    """
    Hitung konsentrasi kelas dengan bobot per status.
    Alert=100%, Distracted=75%, Drowsy=50%, Yawning=15%.
    """
    if not face_results:
        return 100.0

    total_weight = sum(
        STATE_CONCENTRATION_WEIGHTS.get(
            r["classification"]["state"], 0.0
        )
        for r in face_results
    )
    return (total_weight / len(face_results)) * 100