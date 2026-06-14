"""
visualizer.py — Modul visualisasi output menggunakan OpenCV.

Revisi:
- Smoothed concentration (transisi gradual, bukan lompatan)
- Auto-hide detail panel jika wajah > 3 (hanya bbox + label)
- Fix drawing order agar state label tidak tertutup info panel
"""

from __future__ import annotations

import cv2
import numpy as np

from src.classifier import (
    STATE_ALERT,
    STATE_DISTRACTED,
    STATE_DROWSY,
    STATE_YAWNING,
    calculate_weighted_concentration,
)


# Visual Constants
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SM = 0.50
FONT_SCALE_MD = 0.60
THICKNESS_THIN = 1
THICKNESS_BOLD = 2
LINE_HEIGHT = 22
PANEL_PADDING = 7

STATE_COLORS: dict[str, tuple] = {
    STATE_ALERT: (0, 200, 0),
    STATE_DROWSY: (0, 140, 255),
    STATE_DISTRACTED: (0, 220, 255),
    STATE_YAWNING: (50, 50, 255),
}
DEFAULT_COLOR = (180, 180, 180)
BG_DARK = (30, 30, 30)
TEXT_WHITE = (255, 255, 255)

DASHBOARD_HEIGHT = 44
STATE_LABEL_HEIGHT = 26

# Behavior Constants
DETAIL_FACE_THRESHOLD = 3          # hide info panel jika wajah > ini
CONCENTRATION_SMOOTHING = 0.05     # EMA alpha (lower = smoother)


class Visualizer:
    """
    Menangani seluruh rendering visual di frame OpenCV.

    Menyimpan satu state: smoothed concentration percentage,
    yang bertransisi secara gradual menggunakan Exponential Moving Average.
    """

    def __init__(self) -> None:
        self._smoothed_concentration: float = 100.0

    # Public API
    def render(self, frame: np.ndarray, face_results: list[dict]) -> None:
        show_detail = len(face_results) <= DETAIL_FACE_THRESHOLD

        for result in face_results:
            self._draw_face_overlay(frame, result, show_detail)

        self._draw_dashboard(frame, face_results)

    def render_calibration(self, frame: np.ndarray, progress: float) -> None:
        frame_h, frame_w = frame.shape[:2]

        bar_w = 300
        bar_h = 25
        bar_x = (frame_w - bar_w) // 2
        bar_y = frame_h // 2

        cv2.rectangle(
            frame, (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h), BG_DARK, -1,
        )
        fill_w = int(bar_w * progress)
        cv2.rectangle(
            frame, (bar_x, bar_y),
            (bar_x + fill_w, bar_y + bar_h), (0, 200, 0), -1,
        )
        cv2.rectangle(
            frame, (bar_x, bar_y),
            (bar_x + bar_w, bar_y + bar_h), TEXT_WHITE, 1,
        )

        label = f"Kalibrasi... {progress:.0%}"
        text_w = cv2.getTextSize(label, FONT, FONT_SCALE_MD, THICKNESS_THIN)[0][0]
        text_x = bar_x + (bar_w - text_w) // 2
        cv2.putText(
            frame, label, (text_x, bar_y - 10),
            FONT, FONT_SCALE_MD, TEXT_WHITE, THICKNESS_THIN, cv2.LINE_AA,
        )

    # Private: Per-Face
    def _draw_face_overlay(
        self, frame: np.ndarray, result: dict, show_detail: bool,
    ) -> None:
        bbox = result["bbox"]
        features = result["features"]
        classification = result["classification"]
        state = classification["state"]
        color = STATE_COLORS.get(state, DEFAULT_COLOR)
        x1, y1, x2, y2 = bbox

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, THICKNESS_BOLD)

        # Panel Information (di layer bawah)
        if show_detail:
            self._draw_info_panel(frame, bbox, features, classification, color)

        # State Label (di layer atas, tidak akan menutup panel)
        self._draw_state_label(frame, state, color, x1, y1)

    def _draw_state_label(
        self, frame: np.ndarray, state: str, color: tuple, x: int, y: int,
    ) -> None:
        text_size = cv2.getTextSize(state, FONT, FONT_SCALE_MD, THICKNESS_BOLD)[0]

        bg_x1 = x
        bg_y1 = y - text_size[1] - 10
        bg_x2 = x + text_size[0] + 12
        bg_y2 = y

        if bg_y1 < DASHBOARD_HEIGHT:
            bg_y1 = y + 2
            bg_y2 = bg_y1 + text_size[1] + 10

        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
        cv2.putText(
            frame, state, (bg_x1 + 6, bg_y2 - 5),
            FONT, FONT_SCALE_MD, TEXT_WHITE, THICKNESS_BOLD, cv2.LINE_AA,
        )

    def _draw_info_panel(
        self,
        frame: np.ndarray,
        bbox: tuple,
        features: dict,
        classification: dict,
        border_color: tuple,
    ) -> None:
        x1, y1, _, y2 = bbox
        frame_h, frame_w = frame.shape[:2]

        info_lines = [
            f"EAR    : {features['ear']:.2f}",
            f"MAR    : {features['mar']:.2f}",
            f"Pitch  : {features['pitch']:.1f}",
            f"Yaw    : {features['yaw']:.1f}",
            f"PERCLOS: {classification['perclos']:.0%}",
        ]

        max_text_w = max(
            cv2.getTextSize(line, FONT, FONT_SCALE_SM, THICKNESS_THIN)[0][0]
            for line in info_lines
        )
        panel_w = max_text_w + PANEL_PADDING * 2
        panel_h = len(info_lines) * LINE_HEIGHT + PANEL_PADDING * 2

        # Posisi di bawah bbox (preferred)
        panel_x = max(0, min(x1, frame_w - panel_w))
        panel_y = y2 + 5

        if panel_y + panel_h > frame_h:
            # Fallback di atas bbox, tapi sisakan ruang untuk state label
            panel_y = y1 - panel_h - STATE_LABEL_HEIGHT - 5

        panel_y = max(DASHBOARD_HEIGHT + 2, panel_y)

        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            BG_DARK, -1,
        )
        cv2.rectangle(
            frame,
            (panel_x, panel_y),
            (panel_x + panel_w, panel_y + panel_h),
            border_color, 1,
        )

        for i, line in enumerate(info_lines):
            text_x = panel_x + PANEL_PADDING
            text_y = panel_y + PANEL_PADDING + (i + 1) * LINE_HEIGHT - 4
            cv2.putText(
                frame, line, (text_x, text_y),
                FONT, FONT_SCALE_SM, TEXT_WHITE, THICKNESS_THIN, cv2.LINE_AA,
            )

    # Private: Dashboard
    def _draw_dashboard(self, frame: np.ndarray, face_results: list[dict]) -> None:
        frame_w = frame.shape[1]
        cv2.rectangle(frame, (0, 0), (frame_w, DASHBOARD_HEIGHT), BG_DARK, -1)

        total = len(face_results)
        if total == 0:
            cv2.putText(
                frame, "Tidak ada wajah terdeteksi", (10, 28),
                FONT, FONT_SCALE_SM, TEXT_WHITE, THICKNESS_THIN, cv2.LINE_AA,
            )
            self._smoothed_concentration = 100.0
            return

        # Weighted concentration + EMA smoothing
        raw_concentration = calculate_weighted_concentration(face_results)
        self._smoothed_concentration += CONCENTRATION_SMOOTHING * (
            raw_concentration - self._smoothed_concentration
        )
        display_pct = self._smoothed_concentration

        # Hitung per status
        counts: dict[str, int] = {s: 0 for s in STATE_COLORS}
        for result in face_results:
            state = result["classification"]["state"]
            counts[state] = counts.get(state, 0) + 1

        # Teks ringkasan
        parts = [f"Faces: {total}"]
        for state, count in counts.items():
            if count > 0:
                parts.append(f"{state}: {count}")
        parts.append(f"Konsentrasi: {display_pct:.0f}%")
        summary = "  |  ".join(parts)

        cv2.putText(
            frame, summary, (10, 25),
            FONT, FONT_SCALE_SM, TEXT_WHITE, THICKNESS_THIN, cv2.LINE_AA,
        )

        # Color bar proporsional
        bar_y = DASHBOARD_HEIGHT - 4
        bar_x = 0
        for state in [STATE_ALERT, STATE_DROWSY, STATE_DISTRACTED, STATE_YAWNING]:
            count = counts[state]
            if count == 0:
                continue
            segment_w = int(frame_w * (count / total))
            cv2.rectangle(
                frame, (bar_x, bar_y),
                (bar_x + segment_w, DASHBOARD_HEIGHT),
                STATE_COLORS[state], -1,
            )
            bar_x += segment_w