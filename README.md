# 🛡️ Cyberbullying Detection System
### XLM-RoBERTa + FastAPI + Discord Moderation Bot + n8n

Sistem deteksi cyberbullying bilingual (Indonesia & Inggris) berbasis Transformer (**XLM-RoBERTa**), dilengkapi dengan API inferensi berbasis **FastAPI**, bot moderasi otomatis 3 tahap di **Discord**, serta integrasi workflow **n8n**.

---

## 📌 Fitur Utama

- **Bilingual Cyberbullying Detection**: Klasifikasi pesan berteks Indonesia & Inggris menggunakan model fine-tuned XLM-RoBERTa.
- **FastAPI Inference Server**: Endpoint HTTP berkinerja tinggi dengan dukungan prediksi teks tunggal maupun batch prediction.
- **Discord Automated Moderation Bot**: Moderasi otomatis 3 tahap (*Warn*, *Mute*, *Kick*) berdasarkan *confidence tier* serta pelacakan riwayat pelanggaran pengguna via SQLite.
- **n8n Workflow Integration**: Otomatisasi pemicu & pipeline moderasi terintegrasi dengan n8n self-hosted.
- **Comprehensive Evaluation & Analytics**: Dilengkapi skrip pengujian kualitas data, uji statistik, evaluasi k-fold, dan pengujian beban (*load testing*).

---

## 📁 Struktur Proyek

```text
Cyberbullying Detection/
├── api/                        ← FastAPI inference server
│   ├── main.py
│   └── requirements.txt
├── configs/                    ← Konfigurasi pelatihan & moderasi
│   ├── training_config.yaml
│   └── moderation_config.yaml
├── data/                       ← Data mentah, eksternal, & olahan
│   ├── raw/
│   ├── external/
│   └── processed/              ← Train/Val/Test splits (auto-generated)
├── dataset/                    ← Dataset CSV bilingual & dataset pendukung
│   ├── dataset_clean.csv
│   └── combined_dataset.csv
├── discord_bot/                ← Discord bot moderasi
│   ├── bot.py
│   ├── database.py
│   ├── metrics.py
│   ├── preprocessor.py
│   └── requirements.txt
├── experiments/                ← Eksperimen pelatihan & tuning
├── models/                     ← Direktori penyimpanan bobot model
├── n8n/                        ← Workflow n8n JSON
│   └── workflow_cyberbully.json
├── outputs/                    ← Laporan evaluasi, grafik, & log pengujian
├── scripts/                    ← Skrip audit, analisis, & pengujian statistik
├── src/                        ← Source code utama (data prep, training, eval)
│   ├── data/
│   │   └── prepare_dataset.py
│   └── models/
│       ├── train_xlmr.py
│       ├── evaluate.py
│       └── predict.py
├── system_quality_tester.py    ← Suite pengujian kualitas sistem
├── load_test_bot.py            ← Load testing untuk Discord bot
├── load_test_n8n.py            ← Load testing untuk n8n
├── docker-compose.yml          ← Orchestration n8n (Docker)
├── Dockerfile                  ← Container build
├── requirements.txt            ← Dependency umum / training
├── .env.example                ← Template variabel lingkungan
└── README.md                   ← Dokumentasi proyek
```

---

## 🚀 Panduan Memulai (Quick Start)

### 1. Clone Repository & Setup Virtual Environment

```bash
# Clone repository
git clone https://github.com/joewin786/bilingual-cyberbullying-moderation-system.git
cd "Cyberbullying Detection"

# Buat virtual environment (opsional tetapi direkomendasikan)
python -m venv .venv

# Aktifkan virtual environment (Windows PowerShell)
.\.venv\Scripts\activate

# Atau di Linux / macOS:
# source .venv/bin/activate
```

### 2. Install Dependencies

```powershell
# Install dependency utama & pelatihan
pip install -r requirements.txt

# Install dependency API
pip install -r api/requirements.txt

# Install dependency Discord Bot
pip install -r discord_bot/requirements.txt
```

### 3. Konfigurasi Environment Variables

Salin file `.env.example` menjadi `.env` dan sesuaikan nilainya:

```powershell
Copy-Item .env.example .env
```

Isi variabel di `.env`:
```env
DISCORD_TOKEN=your_discord_bot_token
MODERATION_CHANNEL_ID=your_channel_id
ADMIN_ROLE_ID=your_role_id
API_URL=http://localhost:8000
```

---

## 💻 Alur Penggunaan

### Step 1 — Persiapan Dataset
```powershell
python src/data/prepare_dataset.py
```
*Output*: `data/processed/train.csv`, `val.csv`, `test.csv`

### Step 2 — Fine-tuning Model XLM-RoBERTa
```powershell
python src/models/train_xlmr.py
```
*Output*: Bobot model tersimpan di `models/xlmr_cyberbully/`

### Step 3 — Evaluasi Model
```powershell
python src/models/evaluate.py
```
*Output*: `outputs/evaluation_report.txt`, `outputs/confusion_matrix.png`

### Step 4 — Jalankan API Server (FastAPI)
```powershell
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Buka dokumentasi Swagger UI interaktif di: `http://localhost:8000/docs`

### Step 5 — Jalankan Discord Moderation Bot
```powershell
python discord_bot/bot.py
```

### Step 6 — Workflow n8n (Opsional)
Jalankan n8n via Docker:
```powershell
docker compose up -d
```
1. Akses dashboard di `http://localhost:5678`
2. Import workflow dari `n8n/workflow_cyberbully.json`
3. Aktifkan workflow.

---

## ⚙️ Skema Moderasi Discord

### 🔴 Tindakan Bertingkat (3-Tier Enforcement)
| Tingkat Pelanggaran | Tindakan Automatis |
|---------------------|--------------------|
| Ke-1 | ⚠️ Peringatan (Warn) + Hapus Pesan |
| Ke-2 | 🔇 Mute Sementara (8 jam) + Hapus Pesan |
| Ke-3+ | 🚫 Kick Pengguna dari Server |

### 🎯 Confidence Threshold
| Skor Keyakinan Model | Tindakan System |
|----------------------|-----------------|
| `< 70%` | ❌ Diabaikan (Non-bully / low confidence) |
| `70% - 85%` | ⚠️ Flag ke channel moderasi untuk review manual |
| `> 85%` | ✅ Eksekusi tindakan otomatis (Warn/Mute/Kick) |

---

## 🔌 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| `GET` | `/health` | Cek kesehatan server & status model |
| `POST` | `/predict` | Klasifikasi 1 teks pesan |
| `POST` | `/predict/batch` | Klasifikasi batch teks pesan (max 32) |

---

## 📝 Lisensi & Catatan
Dibuat untuk penelitian dan pengembangan sistem deteksi cyberbullying bilingual (Indonesia - Inggris).
