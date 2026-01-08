# 🏛️ SubsidiLedger
## Skor Kelayakan Penerima Subsidi Tanpa Tukar Data: Federated Learning dan Audit On-Chain

**SubsidiLedger** adalah sistem Federated Learning yang dikembangkan untuk membantu pemerintah daerah dalam membangun model kecerdasan buatan untuk menilai kelayakan penerima subsidi (bantuan sosial) dengan tetap mematuhi **UU Perlindungan Data Pribadi (UU PDP)**.

---

## 🎯 Problem Statement

Pemerintah daerah menghadapi tantangan dalam membangun model AI untuk menilai kelayakan penerima subsidi karena:

- **📊 Data Tersebar** - Data calon penerima subsidi tersebar di berbagai instansi: **Dinsos** (data sosial ekonomi), **Dukcapil** (data kependudukan), dan **Kemenkes** (data kesehatan)
- **🔒 UU PDP Compliance** - Berdasarkan UU Perlindungan Data Pribadi, data sensitif **tidak boleh diserahkan ke server pusat** atau dibagikan antar instansi tanpa proper consent
- **🚫 Privacy Concerns** - Pengumpulan data ke satu lokasi menimbulkan risiko keamanan dan pelanggaran privasi
- **⚖️ Regulatory Requirements** - Setiap instansi harus menjaga kedaulatan data mereka sesuai regulasi

## 💡 Solusi: Federated Learning

Project ini mengimplementasikan **Federated Learning** sebagai solusi yang memungkinkan kolaborasi antar instansi tanpa perlu berbagi raw data:

### ✅ Cara Kerja:

1. **🏠 Local Training** - Setiap instansi (Dinsos, Dukcapil, Kemenkes) melatih model lokal menggunakan data masing-masing yang **tidak pernah meninggalkan** server lokal mereka
2. **🔐 Encrypted Contribution** - Hasil kontribusi model (model weights) dikumpulkan, dienkripsi, dan dikirim ke server agregasi
3. **🔄 Federated Aggregation** - Server melakukan agregasi menggunakan algoritma **FedAvg** untuk menghasilkan model global tanpa pernah melihat raw data
4. **📥 Model Distribution** - Model global hasil agregasi dapat di-download kembali oleh setiap instansi untuk meningkatkan akurasi prediksi lokal mereka
5. **📋 Audit On-Chain** (Future: Blockchain logging untuk transparansi dan auditability)

### 🛡️ Privacy-Preserving Benefits:

- ✅ **No Raw Data Transfer** - Hanya model weights yang dibagikan, bukan data pribadi
- ✅ **UU PDP Compliant** - Data tetap berada di server lokal masing-masing instansi
- ✅ **Collaborative Learning** - Semua instansi mendapat manfaat dari knowledge sharing
- ✅ **Secure Aggregation** - Model weights dienkripsi saat transfer

---

## 📋 Daftar Isi

- [🌟 Fitur Utama](#-fitur-utama)
- [🏗️ Arsitektur Sistem](#️-arsitektur-sistem)
- [📂 Struktur Project](#-struktur-project)
- [🚀 Quick Start](#-quick-start)
- [📊 Dataset \u0026 Model](#-dataset--model)
- [🌐 Demo Aplikasi](#-demo-aplikasi)
- [📡 Server Infrastruktur](#-server-infrastruktur)
- [📦 Dependencies](#-dependencies)
- [📖 Dokumentasi Detail](#-dokumentasi-detail)

---

## 🌟 Fitur Utama

✅ **Federated Learning** - Training model terdistribusi tanpa centralisasi data  
✅ **Multi-Instansi** - Mendukung 3 instansi pemerintah dengan skema data berbeda  
✅ **Privacy-Preserving** - Data tidak pernah meninggalkan server lokal  
✅ **Model Aggregation** - Agregasi model global menggunakan FedAvg  
✅ **REST API Server** - Server untuk upload, agregasi, dan download model  
✅ **Demo Web Interface** - Aplikasi Flask untuk testing prediksi  
✅ **Dataset Simulation** - Generator dataset sintetis untuk development

---

## 🏗️ Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐            │
│  │   DINSOS    │   │  DUKCAPIL   │   │  KEMENKES   │            │
│  ├─────────────┤   ├─────────────┤   ├─────────────┤            │
│  │ Dataset     │   │ Dataset     │   │ Dataset     │            │
│  │ 100k rows   │   │ 100k rows   │   │ 100k rows   │            │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘            │
│         │                 │                 │                    │
│         ▼                 ▼                 ▼                    │
│  ┌─────────────────────────────────────────────────┐            │
│  │       Federated Learning (TFF)                  │            │
│  │  - 10 Clients per Instansi                      │            │
│  │  - FedAvg Algorithm                             │            │
│  │  - 15 Rounds Training                           │            │
│  └────────────────────┬────────────────────────────┘            │
│                       │                                          │
│                       ▼                                          │
│              ┌────────────────┐                                  │
│              │ Model NPZ      │                                  │
│              │ + Metrics      │                                  │
│              └────────┬───────┘                                  │
└──────────────────────────────────────────────────────────────────┘
                        │
                        │ Upload (POST /upload-model)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                         SERVER LAYER                              │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│              🌍 Federated Aggregation Server                      │
│              (Railway: federatedinstitusi.up.railway.app)         │
│                                                                   │
│  ┌────────────────────────────────────────────────────┐          │
│  │ API Endpoints:                                     │          │
│  │ • POST /upload-model    - Upload model lokal       │          │
│  │ • POST /aggregate       - Agregasi FedAvg          │          │
│  │ • GET  /download-global - Download model global    │          │
│  │ • GET  /logs            - List semua model         │          │
│  │ • GET  /accuracy/{id}   - Metrics client           │          │
│  └────────────────────────────────────────────────────┘          │
│                                                                   │
│  Storage: models/                                                 │
│  ├── dinsos_weights.npz                                           │
│  ├── dukcapil_weights.npz                                         │
│  ├── kemenkes_weights.npz                                         │
│  └── global_model_fedavg_{timestamp}.npz                          │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
                        │
                        │ Download (GET /download-global)
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│                      DEPLOYMENT LAYER                             │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│                    🖥️ Flask Demo App                             │
│                                                                   │
│  Endpoints:                                                       │
│  • POST /predict/dinsos    - Prediksi Dinsos                     │
│  • POST /predict/dukcapil  - Prediksi Dukcapil                   │
│  • POST /predict/kemenkes  - Prediksi Kemenkes                   │
│  • POST /predict/gabungan  - Prediksi Model Global               │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 📂 Struktur Project

```
SubsidiLedger/
├── 📁 Dinsos/                          # Modul Dinas Sosial
│   ├── Dinsos.py                       # Training script (TFF)
│   ├── upload_model.py                 # Upload ke server
│   ├── DATASET/                        # Dataset dinsos
│   ├── Models/                         # Local models
│   └── README.md                       # Dokumentasi Dinsos
│
├── 📁 Dukcapil/                        # Modul Kependudukan
│   ├── Dukcapil.py                     # Training script (TFF)
│   ├── upload_model.py                 # Upload ke server
│   ├── DATASET/                        # Dataset dukcapil
│   ├── Models/                         # Local models
│   └── README.md                       # Dokumentasi Dukcapil
│
├── 📁 Kemenkes/                        # Modul Kesehatan
│   ├── kemenkes.py                     # Training script (TFF)
│   ├── upload_model.py                 # Upload ke server
│   ├── DATASET/                        # Dataset kemenkes
│   ├── Models/                         # Local models
│   └── README.md                       # Dokumentasi Kemenkes
│
├── 📁 Generate Dataset/                # Dataset Generator
│   ├── generate.py                     # Generate dataset simulasi
│   ├── Feature_Cols.py                 # Extract fitur global
│   ├── DATASET/                        # Output datasets
│   ├── Models/                         # fitur_global.pkl
│   └── README.md                       # Dokumentasi Generator
│
├── 📁 Server/                          # Federated Server
│   ├── app.py                          # Flask server API
│   ├── aggregasi.py                    # FedAvg aggregation
│   ├── download.py                     # Download utilities
│   ├── models/                         # Model storage
│   └── API_ENDPOINTS_README.md         # API Documentation
│
├── 📁 Flask/                           # Demo Web Application
│   ├── app.py                          # Flask demo app
│   ├── test.py                         # Testing utilities
│   ├── templates/                      # HTML templates
│   ├── Models/                         # Downloaded models
│   └── .venv/                          # Virtual environment
│
└── README.md                           # ← You are here!
```

---

## 🚀 Quick Start

### 1️⃣ Generate Dataset

Buat dataset simulasi untuk semua instansi:

```bash
cd "Generate Dataset"
python generate.py
python Feature_Cols.py
```

**Output**: 
- `DATASET/dinsos_balanced.csv` (100k rows)
- `DATASET/dukcapil_balanced.csv` (100k rows)
- `DATASET/kemenkes_balanced.csv` (100k rows)
- `Models/fitur_global.pkl` (fitur global)

📖 **Docs**: [Generate Dataset/README.md](Generate%20Dataset/README.md)

---

### 2️⃣ Training Model Lokal

#### Dinsos
```bash
cd Dinsos
python Dinsos.py
```

#### Dukcapil
```bash
cd Dukcapil
python Dukcapil.py
```

#### Kemenkes
```bash
cd Kemenkes
python kemenkes.py
```

**Output**: Model NPZ + preprocessing params di folder `Models/`

📖 **Docs**: 
- [Dinsos/README.md](Dinsos/README.md)
- [Dukcapil/README.md](Dukcapil/README.md)
- [Kemenkes/README.md](Kemenkes/README.md)

---

### 3️⃣ Upload Model ke Server

Setelah training selesai, upload model ke server:

```bash
# Dari masing-masing folder instansi
python upload_model.py
```

**Server**: `https://federatedinstitusi.up.railway.app`

---

### 4️⃣ Agregasi Model Global (Server-Side)

Jalankan agregasi FedAvg di server:

```bash
curl -X POST https://federatedinstitusi.up.railway.app/aggregate
```

Atau jalankan server lokal:

```bash
cd Server
python app.py
```

📖 **Docs**: [Server/API_ENDPOINTS_README.md](Server/API_ENDPOINTS_README.md)

---

### 5️⃣ Deploy Demo Aplikasi

```bash
cd Flask
python app.py
```

Buka browser: `http://localhost:5000`

---

## 📊 Dataset \u0026 Model

### Dataset Overview

| Instansi | Rows | Features | Target | Karakteristik |
|----------|------|----------|--------|---------------|
| **Dinsos** | 100k | 7 | `layak_subsidi` | Data sosial ekonomi |
| **Dukcapil** | 100k | 10 | `layak_subsidi` | Data kependudukan |
| **Kemenkes** | 100k | 7 | `layak_subsidi` | Data kesehatan |

### Model Architecture

```python
Input Layer (n fitur)
├─ Dense(128, relu)
├─ BatchNormalization
├─ Dropout(0.3)
├─ Dense(64, relu)
├─ Dense(32, relu)
└─ Dense(1, sigmoid)
```

### Training Configuration

```python
ALGORITHM      = "FedAvg"
N_CLIENTS      = 10
BATCH_SIZE     = 32
ROUNDS         = 15
CLIENT_LR      = 0.005
SERVER_LR      = 0.01
LOSS           = "Binary Cross-Entropy"
METRICS        = "Binary Accuracy"
```

### Expected Results

- **Training Accuracy**: >90% setelah 15 rounds
- **Model Size**: <100 KB (NPZ format)
- **Training Time**: ~5-10 menit per instansi
- **Upload Time**: <60 detik

---

## 🌐 Demo Aplikasi

Flask demo app menyediakan interface untuk testing prediksi kelayakan subsidi.

### Available Models

| Endpoint | Model | Description |
|----------|-------|-------------|
| `/predict/dinsos` | Dinsos Local | Model Dinas Sosial |
| `/predict/dukcapil` | Dukcapil Local | Model Dukcapil |
| `/predict/kemenkes` | Kemenkes Local | Model Kemenkes |
| `/predict/gabungan` | Global Model | Model hasil agregasi |

### Request Example

```bash
curl -X POST http://localhost:5000/predict/dinsos \
  -H "Content-Type: application/json" \
  -d '{
    "penghasilan": 1500000,
    "jumlah_tanggungan": 4,
    "kondisi_rumah": "sederhana",
    "status_pekerjaan": "buruh harian",
    "pendidikan": "sd",
    "lama_tinggal_tahun": 5
  }'
```

### Response Example

```json
{
  "prediksi": 1,
  "probabilitas": 0.8234,
  "threshold": 0.53
}
```

---

## 📡 Server Infrastruktur

### Production Server
```
https://federatedinstitusi.up.railway.app
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Status server |
| `POST` | `/upload-model` | Upload model client |
| `POST` | `/aggregate` | Agregasi FedAvg |
| `GET` | `/download-global` | Download model global |
| `GET` | `/logs` | List semua model |
| `GET` | `/accuracy/{client}` | Metrics client |
| `DELETE` | `/delete/{filename}` | Hapus model |

📖 **Full API Docs**: [Server/API_ENDPOINTS_README.md](Server/API_ENDPOINTS_README.md)

---

## 📦 Dependencies

### Core Dependencies

```
tensorflow>=2.13.0
tensorflow-federated>=0.53.0
pandas>=1.5.0
numpy>=1.23.0
joblib>=1.2.0
```

### Server Dependencies

```
flask>=2.3.0
flask-cors>=4.0.0
requests>=2.28.0
```

### Install All Dependencies

```bash
pip install tensorflow tensorflow-federated pandas numpy joblib flask flask-cors requests
```

---

## 📖 Dokumentasi Detail

### Per-Module Documentation

| Module | README |
|--------|--------|
| **Generate Dataset** | [Generate Dataset/README.md](Generate%20Dataset/README.md) |
| **Dinsos** | [Dinsos/README.md](Dinsos/README.md) |
| **Dukcapil** | [Dukcapil/README.md](Dukcapil/README.md) |
| **Kemenkes** | [Kemenkes/README.md](Kemenkes/README.md) |
| **Server API** | [Server/API_ENDPOINTS_README.md](Server/API_ENDPOINTS_README.md) |

---

## 🔄 Complete Workflow

```
1. Generate Dataset
   └─→ python generate.py + Feature_Cols.py
        │
        ▼
2. Training Lokal (Parallel untuk 3 instansi)
   ├─→ python Dinsos/Dinsos.py
   ├─→ python Dukcapil/Dukcapil.py
   └─→ python Kemenkes/kemenkes.py
        │
        ▼
3. Upload ke Server
   ├─→ python Dinsos/upload_model.py
   ├─→ python Dukcapil/upload_model.py
   └─→ python Kemenkes/upload_model.py
        │
        ▼
4. Agregasi di Server
   └─→ POST /aggregate
        │
        ▼
5. Download Model Global
   └─→ GET /download-global
        │
        ▼
6. Deploy Demo App
   └─→ python Flask/app.py
```

---

## 🎯 Use Cases

### 1. Privacy-Preserving Collaboration
Tiga instansi pemerintah dapat berkolaborasi melatih model tanpa harus berbagi raw data sensitif.

### 2. Distributed Learning
Setiap instansi melatih model dengan data lokal mereka, kemudian server mengagregasi knowledge dari semua instansi.

### 3. Model Improvement
Model global hasil agregasi dapat di-download kembali oleh setiap instansi untuk meningkatkan akurasi prediksi lokal mereka.

### 4. Scalable Architecture
Mudah menambahkan instansi baru tanpa perlu mengubah arsitektur fundamental.

---

## ⚙️ Configuration

### Training Configuration

Edit di masing-masing file `{instansi}.py`:

```python
INSTANSI   = "dinsos"      # Nama klien
BATCH_SIZE = 32            # Ukuran batch per klien
N_CLIENTS  = 10            # Jumlah klien federated
ROUNDS     = 15            # Jumlah round training
```

### Server Configuration

Edit di `Server/app.py`:

```python
PORT = 8080
HOST = "0.0.0.0"
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
```

### Upload Configuration

Edit di masing-masing file `upload_model.py`:

```python
SERVER_URL   = "https://federatedinstitusi.up.railway.app"
CLIENT_NAME  = "dinsos"
TIMEOUT      = 180         # Timeout upload (detik)
RETRY_LIMIT  = 3           # Maksimal retry upload
```

---

## 🛡️ Security \u0026 Privacy

✅ **No Raw Data Transfer** - Hanya model weights yang dikirim, bukan raw data  
✅ **Local Training** - Data tidak pernah meninggalkan server lokal  
✅ **Secure Upload** - HTTPS encryption untuk upload model  
✅ **Path Traversal Protection** - Server dilindungi dari path traversal attacks  
✅ **CORS Configuration** - Controlled access dari frontend

---

## 📝 Notes

1. **Konsistensi Fitur**: Pastikan `fitur_global.pkl` ada sebelum training
2. **Resource Requirements**: Training membutuhkan ~4GB RAM per instansi
3. **Network**: Upload membutuhkan koneksi internet stabil
4. **Best Practice**: Selalu backup model sebelum upload atau agregasi
5. **Version Control**: Model diberi timestamp untuk tracking versi

---

## 🐛 Troubleshooting

### Training Issues

| Problem | Solution |
|---------|----------|
| Out of Memory | Reduce `BATCH_SIZE` atau `N_CLIENTS` |
| Low Accuracy | Increase `ROUNDS` atau adjust learning rate |
| Feature Mismatch | Regenerate `fitur_global.pkl` |

### Upload Issues

| Problem | Solution |
|---------|----------|
| Timeout Error | Check internet connection, increase `TIMEOUT` |
| Server 500 | Contact server administrator |
| NPZ Not Found | Run training script first |

### Server Issues

| Problem | Solution |
|---------|----------|
| Insufficient Models | Upload minimal 2 models sebelum agregasi |
| Model Not Found | Check filename \u0026 server logs |

---

## 📞 Support

Untuk pertanyaan atau issues, silakan hubungi tim pengembang atau buat issue di repository.

---

## 📄 License

Project ini dikembangkan untuk keperluan riset dan pendidikan dalam implementasi Federated Learning untuk sektor publik.

---

## 🏆 Credits

**Developed by**: Data Engineering Team  
**Framework**: TensorFlow Federated  
**Server**: Railway Platform  
**Last Updated**: 2026-01-08

---

## 🔗 Related Resources

- [TensorFlow Federated Documentation](https://www.tensorflow.org/federated)
- [Federated Learning: Collaborative Machine Learning without Centralized Training Data](https://ai.googleblog.com/2017/04/federated-learning-collaborative.html)
- [Railway Deployment Guide](https://docs.railway.app/)

---

**✨ Happy Federated Learning! ✨**
