# Student Concentration Monitor

Sistem monitoring konsentrasi mahasiswa di kelas menggunakan Computer Vision dan Generative AI.

## Prasyarat

- Python 3.12
- Anaconda atau Miniconda
- Webcam (untuk mode real-time)
- Koneksi internet (untuk download model dan Gemini API)

## Setup

### 1. Buat Conda Environment

```bash
conda create -n student-concentration python=3.12
conda activate student-concentration
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Model MediaPipe

Download file berikut dan simpan ke dalam folder `models/`:

https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

Model YOLOv8-face akan terdownload otomatis saat program pertama kali dijalankan.

### 4. Konfigurasi API Key

Buat file `.env` di root folder project, isi dengan:

```
GEMINI_API_KEY=masukkan_api_key_gemini_kamu_di_sini
```

Dapatkan API key gratis di https://aistudio.google.com/app/apikey (login dengan Google account).

## Cara Menjalankan

```bash
# Mode webcam
python main.py

# Mode webcam + generate AI report setelah sesi
python main.py --report

# Mode video file
python main.py --video path/ke/video.mp4

# Mode video + report
python main.py --video path/ke/video.mp4 --report
```

Tekan **Q** untuk menghentikan program.

## Struktur Project

```
student-concentration-monitor/
├── models/                  # File model (YOLOv8-face, MediaPipe)
├── reports/                 # Output laporan AI (auto-generated)
├── src/
│   ├── preprocessing.py     # CLAHE, resize, denoise
│   ├── detection.py         # YOLOv8 face detection
│   ├── landmarks.py         # MediaPipe 468-point landmark
│   ├── features.py          # EAR, MAR, head pose
│   ├── classifier.py        # Klasifikasi status konsentrasi
│   ├── report.py            # Gemini API report generation
│   └── visualizer.py        # OpenCV rendering
├── main.py                  # Entry point
├── requirements.txt
├── .env.example             # Template API key
└── README.md
```

## Pipeline

```
Input (Webcam/Video)
  → Preprocessing (CLAHE + Resize)
  → Face Detection (YOLOv8-face)
  → Face Crop + Padding 30%
  → Landmark Extraction (MediaPipe 468-point)
  → Feature Extraction (EAR, MAR, Head Pose)
  → Auto-Calibration (~1 detik pertama)
  → Klasifikasi (Alert / Drowsy / Distracted / Yawning)
  → Visualisasi (Bounding Box + Dashboard)
  → Report Generation (Gemini AI)
```

## Catatan

- Pada eksekusi pertama, izinkan akses kamera jika diminta oleh sistem operasi.
- Auto-calibration berjalan selama ~1 detik pertama. Hadapkan wajah ke kamera dalam kondisi normal (mata terbuka, kepala lurus) selama periode ini.
- Jika Gemini API key tidak diisi atau koneksi internet tidak tersedia, sistem otomatis menggunakan laporan template tanpa AI.