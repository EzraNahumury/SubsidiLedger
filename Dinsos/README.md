# README - Dinsos Federated Learning

## 📋 Deskripsi Proyek

Proyek ini mengimplementasikan **Federated Learning** untuk sistem prediksi kelayakan subsidi menggunakan TensorFlow Federated (TFF). Model dilatih secara terdistribusi dengan data dari 10 klien virtual untuk menjaga privasi data sambil meningkatkan akurasi prediksi.

---

## 📊 Struktur Dataset

### Lokasi Dataset
```
DATASET/dinsos_balanced.csv
```

### Detail Dataset
- **Total Data**: 100,000 baris
- **Total Fitur**: 7 kolom
- **Target Variable**: `layak_subsidi` (klasifikasi biner: 0/1)

### Kolom Dataset

| Kolom | Tipe | Deskripsi |
|-------|------|-----------|
| `penghasilan` | Numerik | Penghasilan rumah tangga |
| `jumlah_tanggungan` | Numerik | Jumlah anggota keluarga yang ditanggung |
| `kondisi_rumah` | Kategorikal | Kondisi tempat tinggal (sederhana, layak, dll) |
| `pekerjaan` | Kategorikal | Jenis pekerjaan kepala keluarga |
| `pendidikan_terakhir` | Kategorikal | Tingkat pendidikan terakhir |
| `lama_tinggal_tahun` | Numerik | Lama tinggal di lokasi saat ini (tahun) |
| `layak_subsidi` | Biner (0/1) | **Target** - Kelayakan menerima subsidi |

### Karakteristik Data
- **Balanced Dataset**: Dataset telah diseimbangkan antara kelas positif dan negatif
- **Missing Values**: Data telah dibersihkan dari nilai yang hilang
- **Encoding**: Kolom kategorikal akan di-encode dengan one-hot encoding saat training

---

## 🔧 Fitur Global

File `Models/fitur_global.pkl` berisi daftar fitur global yang digunakan untuk memastikan konsistensi struktur input antara training dan inference.

### Fungsi Fitur Global:
1. **Alignment**: Memastikan semua klien memiliki struktur fitur yang identik
2. **One-Hot Encoding**: Menjaga konsistensi kategori yang di-encode
3. **Dimensionality**: Menentukan ukuran input layer model

---

## 🚀 Training Model

### Menjalankan Training

```bash
python Dinsos.py
```

### Proses Training

#### 1. **Load Dataset**
   - Membaca `DATASET/dinsos_balanced.csv`
   - Memuat fitur global dari `Models/fitur_global.pkl`

#### 2. **Preprocessing**
   - **One-Hot Encoding**: Mengkonversi fitur kategorikal
   - **Feature Alignment**: Menyelaraskan dengan fitur global
   - **Normalisasi Min-Max**: Menormalisasi nilai numerik ke range [0, 1]

#### 3. **Client Setup**
   - **Jumlah Klien**: 10 klien federated
   - **Batch Size**: 32
   - **Data Split**: Data dibagi secara acak ke setiap klien

#### 4. **Model Architecture**
   ```python
   Input Layer (n fitur)
   ├─ Dense(128, relu)
   ├─ BatchNormalization
   ├─ Dropout(0.3)
   ├─ Dense(64, relu)
   ├─ Dense(32, relu)
   └─ Dense(1, sigmoid)
   ```

#### 5. **Federated Training**
   - **Algoritma**: FedAvg (Federated Averaging)
   - **Client Optimizer**: Adam (lr=0.005)
   - **Server Optimizer**: Adam (lr=0.01)
   - **Rounds**: 15 rounds
   - **Loss Function**: Binary Cross-Entropy
   - **Metrics**: Binary Accuracy

#### 6. **Training Output**
   ```
   [DINSOS] Round 01 | acc=0.8542 | loss=0.3456
   [DINSOS] Round 02 | acc=0.8789 | loss=0.2987
   ...
   [DINSOS] Round 15 | acc=0.9234 | loss=0.1876
   ```

---

## 💾 Model Output

Setelah training selesai, model dan artefak terkait akan disimpan di direktori:
```
Models/saved_dinsos_tff/
```

### File yang Dihasilkan

| File | Deskripsi |
|------|-----------|
| **saved_model.pb** | Model TensorFlow SavedModel format |
| **keras_metadata.pb** | Metadata model Keras |
| **variables/** | Folder berisi weight model dalam format TF |
| **dinsos_YYYYMMDD_HHMMSS.npz** | Compressed weights untuk upload ke server |
| **preprocess.pkl** | Parameter preprocessing (feature_cols, mins, rng) |
| **accuracy_history.txt** | Log history akurasi setiap round |
| **best_accuracy.txt** | Akurasi terbaik yang dicapai |
| **fingerprint.pb** | Fingerprint model SavedModel |

### Detail File Penting

#### 📁 `dinsos_YYYYMMDD_HHMMSS.npz`
- Format: NumPy compressed archive
- Isi: Semua weight dari model neural network
- Ukuran: ~65 KB
- Fungsi: Digunakan untuk upload ke server

#### 📁 `preprocess.pkl`
- Format: Joblib pickle
- Isi: 
  - `FEATURE_COLS`: Daftar nama fitur yang digunakan
  - `mins`: Nilai minimum untuk normalisasi
  - `rng`: Range (max-min) untuk normalisasi
- Fungsi: Memastikan preprocessing inference sama dengan training

#### 📁 `accuracy_history.txt`
- Format: Tab-separated values (TSV)
- Kolom: `round`, `accuracy`, `loss`, `timestamp`
- Fungsi: Monitoring performa training

#### 📁 `best_accuracy.txt`
- Format: Single line text
- Isi: Nilai akurasi terbaik (contoh: `0.923400`)
- Fungsi: Quick reference untuk evaluasi model

---

## 📡 Upload Model ke Server

### Server Endpoint
```
https://federatedinstitusi.up.railway.app
```

### Menjalankan Upload

```bash
python upload_model.py
```

### Proses Upload

#### 1. **Load Model NPZ**
   - Mencari file NPZ terbaru di `Models/saved_dinsos_tff/`
   - Jika tidak ada, generate dari SavedModel

#### 2. **Validasi Weight**
   - Memverifikasi jumlah tensor (expected: 12)
   - Struktur: Dense + BN + Dense + Dense layers

#### 3. **Load Metrics**
   - Membaca `best_accuracy.txt`
   - Membaca 10 baris terakhir dari `accuracy_history.txt`

#### 4. **Encode & Prepare Payload**
   ```json
   {
     "client": "dinsos",
     "compressed_weights": "<base64_encoded_npz>",
     "framework": "tensorflow",
     "model_version": "v1.0",
     "metrics": {
       "best_accuracy": 0.9234,
       "history_tail": ["..."]
     }
   }
   ```

#### 5. **POST Request**
   - Endpoint: `/upload-model`
   - Timeout: 180 detik
   - Retry: 3 kali percobaan

#### 6. **Response Server**
   ```
   ✅ Upload sukses (45.32 detik)
   📨 Server response: {"status": "success", "message": "Model uploaded"}
   ```

### Troubleshooting Upload

| Error | Solusi |
|-------|--------|
| `Timeout error` | Periksa koneksi internet, coba lagi |
| `Weight count mismatch` | Pastikan model architecture konsisten |
| `Server 500` | Hubungi administrator server |
| `NPZ not found` | Jalankan `Dinsos.py` terlebih dahulu |

---

## 🔄 Workflow Lengkap

```
┌─────────────────────────────────────────────────────────────┐
│                    1. PERSIAPAN DATA                        │
│  - Dataset: DATASET/dinsos_balanced.csv (100k rows)         │
│  - Fitur Global: Models/fitur_global.pkl                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    2. TRAINING LOKAL                         │
│  Command: python Dinsos.py                                   │
│  - Load dataset & preprocessing                              │
│  - Split ke 10 klien federated                               │
│  - Training 15 rounds dengan FedAvg                          │
│  - Simpan model & artifacts                                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    3. MODEL OUTPUT                           │
│  Lokasi: Models/saved_dinsos_tff/                            │
│  - saved_model.pb (TF SavedModel)                            │
│  - dinsos_YYYYMMDD_HHMMSS.npz (Compressed weights)           │
│  - preprocess.pkl (Preprocessing params)                     │
│  - accuracy_history.txt (Training log)                       │
│  - best_accuracy.txt (Best performance)                      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   4. UPLOAD KE SERVER                        │
│  Command: python upload_model.py                             │
│  - Load NPZ weights                                          │
│  - Validate weight structure                                 │
│  - Encode base64                                             │
│  - POST ke https://federatedinstitusi.up.railway.app         │
│  - Retry hingga 3x jika gagal                                │
└─────────────────────────────────────────────────────────────┘
                            ↓
                    ✅ MODEL DEPLOYED
```

---

## ⚙️ Konfigurasi

### Training Configuration (`Dinsos.py`)
```python
INSTANSI   = "dinsos"      # Nama klien
BATCH_SIZE = 32            # Ukuran batch per klien
N_CLIENTS  = 10            # Jumlah klien federated
ROUNDS     = 15            # Jumlah round training
```

### Upload Configuration (`upload_model.py`)
```python
SERVER_URL   = "https://federatedinstitusi.up.railway.app"
CLIENT_NAME  = "dinsos"
TIMEOUT      = 180         # Timeout upload (detik)
RETRY_LIMIT  = 3           # Maksimal retry upload
```

---

## 📦 Dependencies

```
tensorflow>=2.13.0
tensorflow-federated>=0.53.0
pandas>=1.5.0
numpy>=1.23.0
joblib>=1.2.0
requests>=2.28.0
```

---

## 📝 Catatan

1. **Konsistensi Fitur**: Pastikan `fitur_global.pkl` ada sebelum training
2. **Resource**: Training membutuhkan ~4GB RAM
3. **Waktu Training**: ~5-10 menit tergantung hardware
4. **Upload Size**: Model NPZ ~65 KB, cepat untuk upload
5. **Best Practice**: Selalu backup model sebelum upload

---

## 🎯 Hasil yang Diharapkan

- **Training Accuracy**: >90% setelah 15 rounds
- **Model Size**: <100 KB (NPZ format)
- **Upload Time**: <60 detik
- **Inference Ready**: Model siap untuk deployment di server

---

## 📞 Support

Jika ada masalah atau pertanyaan, hubungi tim pengembang atau cek dokumentasi TensorFlow Federated:
- https://www.tensorflow.org/federated
