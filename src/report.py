"""
report.py — Modul pelacakan sesi dan generasi laporan konsentrasi.

Revisi:
- SessionTracker menggunakan weighted concentration
  (konsisten dengan tampilan real-time di dashboard)
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from google import genai
from dotenv import load_dotenv

from src.classifier import (
    STATE_ALERT,
    STATE_CONCENTRATION_WEIGHTS,
    STATE_DISTRACTED,
    STATE_DROWSY,
    STATE_YAWNING,
    calculate_weighted_concentration,
)


# ─── Constants ───────────────────────────────────────────────
ALL_STATES = [STATE_ALERT, STATE_DROWSY, STATE_DISTRACTED, STATE_YAWNING]
SNAPSHOT_INTERVAL_SEC = 5.0
GEMINI_MODEL = "gemini-1.5-flash"
REPORTS_DIR = Path(__file__).parent.parent / "reports"


class SessionTracker:
    """Mengumpulkan data klasifikasi selama sesi monitoring."""

    def __init__(self) -> None:
        self._start_time: float = time.time()
        self._frame_count: int = 0
        self._state_counts: dict[str, int] = {s: 0 for s in ALL_STATES}
        self._snapshots: list[dict] = []
        self._last_snapshot_time: float = self._start_time

    def record(self, face_results: list[dict]) -> None:
        self._frame_count += 1

        for result in face_results:
            state = result["classification"]["state"]
            if state in self._state_counts:
                self._state_counts[state] += 1

        now = time.time()
        if now - self._last_snapshot_time >= SNAPSHOT_INTERVAL_SEC:
            self._save_snapshot(face_results, now)
            self._last_snapshot_time = now

    def get_summary(self) -> dict | None:
        total = sum(self._state_counts.values())
        if total == 0:
            return None

        duration = time.time() - self._start_time
        pct = {
            state: (count / total) * 100
            for state, count in self._state_counts.items()
        }

        # Weighted average concentration (konsisten dengan dashboard)
        weighted_sum = sum(
            count * STATE_CONCENTRATION_WEIGHTS.get(state, 0.0)
            for state, count in self._state_counts.items()
        )
        avg_concentration = (weighted_sum / total) * 100

        return {
            "duration_seconds": duration,
            "total_frames": self._frame_count,
            "total_classifications": total,
            "state_counts": dict(self._state_counts),
            "state_percentages": pct,
            "avg_concentration": avg_concentration,
            "snapshots": list(self._snapshots),
        }

    def _save_snapshot(self, face_results: list[dict], timestamp: float) -> None:
        total = len(face_results)
        if total == 0:
            return

        elapsed = timestamp - self._start_time
        concentration = calculate_weighted_concentration(face_results)

        self._snapshots.append({
            "elapsed_seconds": round(elapsed, 1),
            "num_faces": total,
            "concentration_pct": round(concentration, 1),
        })


# ─── Report Generation ──────────────────────────────────────

def generate_report(summary: dict) -> str:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        print("⚠️  GEMINI_API_KEY tidak ditemukan di .env → laporan template.")
        return _fallback_report(summary)

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        prompt = _build_prompt(summary)
        response = model.generate_content(prompt)
        return response.text

    except Exception as err:
        print(f"⚠️  Gemini API error: {err} → laporan template.")
        return _fallback_report(summary)


def save_report(report_text: str) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = REPORTS_DIR / f"concentration_report_{timestamp}.md"
    filepath.write_text(report_text, encoding="utf-8")
    print(f"📄 Laporan disimpan: {filepath}")
    return filepath


def _build_prompt(summary: dict) -> str:
    pct = summary["state_percentages"]
    duration_min = summary["duration_seconds"] / 60

    timeline_section = ""
    if summary["snapshots"]:
        timeline_section = "\n\nTimeline konsentrasi kelas:\n"
        for snap in summary["snapshots"]:
            timeline_section += (
                f"- Detik ke-{snap['elapsed_seconds']}: "
                f"{snap['num_faces']} wajah, "
                f"konsentrasi {snap['concentration_pct']}%\n"
            )

    return (
        "Kamu adalah sistem analisis konsentrasi kelas berbasis Computer Vision.\n"
        "Berdasarkan data monitoring berikut, buat laporan analisis dalam Bahasa Indonesia\n"
        "yang profesional dan informatif.\n\n"
        "=== DATA MONITORING ===\n"
        f"Durasi sesi: {duration_min:.1f} menit ({summary['duration_seconds']:.0f} detik)\n"
        f"Total frame diproses: {summary['total_frames']}\n"
        f"Total klasifikasi wajah: {summary['total_classifications']}\n\n"
        "Distribusi status:\n"
        f"- Alert (fokus): {pct[STATE_ALERT]:.1f}%\n"
        f"- Drowsy (mengantuk): {pct[STATE_DROWSY]:.1f}%\n"
        f"- Distracted (teralihkan): {pct[STATE_DISTRACTED]:.1f}%\n"
        f"- Yawning (menguap): {pct[STATE_YAWNING]:.1f}%\n\n"
        f"Rata-rata konsentrasi kelas (weighted): {summary['avg_concentration']:.1f}%\n"
        f"{timeline_section}\n"
        "=== INSTRUKSI ===\n"
        "Buat laporan yang mencakup:\n"
        "1. Ringkasan kondisi konsentrasi kelas secara umum\n"
        "2. Analisis pola yang terdeteksi (kapan konsentrasi turun, dsb.)\n"
        "3. Permasalahan utama yang ditemukan\n"
        "4. Rekomendasi konkret untuk dosen agar meningkatkan engagement\n\n"
        "Format laporan dalam Markdown dengan heading yang jelas.\n"
        "Gunakan bahasa yang profesional tapi mudah dipahami."
    )


def _fallback_report(summary: dict) -> str:
    pct = summary["state_percentages"]
    duration_min = summary["duration_seconds"] / 60

    return (
        "# Laporan Konsentrasi Kelas\n\n"
        "## Ringkasan Sesi\n"
        f"- **Durasi:** {duration_min:.1f} menit\n"
        f"- **Total frame:** {summary['total_frames']}\n"
        f"- **Total klasifikasi:** {summary['total_classifications']}\n\n"
        "## Distribusi Status\n"
        f"| Status | Persentase |\n"
        f"|--------|------------|\n"
        f"| Alert (fokus) | {pct[STATE_ALERT]:.1f}% |\n"
        f"| Drowsy (mengantuk) | {pct[STATE_DROWSY]:.1f}% |\n"
        f"| Distracted (teralihkan) | {pct[STATE_DISTRACTED]:.1f}% |\n"
        f"| Yawning (menguap) | {pct[STATE_YAWNING]:.1f}% |\n\n"
        "## Tingkat Konsentrasi\n"
        f"Rata-rata konsentrasi kelas: **{summary['avg_concentration']:.1f}%**\n\n"
        "---\n"
        "*Laporan ini dihasilkan secara otomatis oleh Student Concentration Monitor.*\n"
    )