# Generate Dataset - Subsidi Ledger

Repository ini berisi skrip untuk menghasilkan dataset simulasi dari tiga instansi pemerintah yang berbeda (Dinsos, Dukcapil, dan Kemenkes) untuk keperluan analisis kelayakan subsidi, serta proses ekstraksi fitur global dari ketiga dataset tersebut.

---

## 📋 Daftar Isi

- [Overview](#overview)
- [Struktur Direktori](#struktur-direktori)
- [Dataset Generation](#dataset-generation)
- [Feature Engineering](#feature-engineering)
- [Cara Menjalankan](#cara-menjalankan)
- [Output Files](#output-files)

---

## Overview

Proyek ini mensimulasikan data dari tiga instansi pemerintah yang berbeda:

1. **DINSOS** - Dinas Sosial: Data sosial ekonomi masyarakat
2. **DUKCAPIL** - Dinas Kependudukan dan Pencatatan Sipil: Data kependudukan
3. **KEMENKES** - Kementerian Kesehatan: Data kesehatan masyarakat

Setiap instansi memiliki skema data yang berbeda, namun semua bertujuan untuk menentukan kelayakan subsidi (`layak_subsidi`) dengan label binary (1 = layak, 0 = tidak layak).

---

## Struktur Direktori

```
Generate Dataset/
├── generate.py           # Script untuk generate dataset dari 3 instansi
├── Feature_Cols.py       # Script untuk ekstraksi fitur global
├── DATASET/              # Output folder untuk dataset
│   ├── dinsos_balanced.csv
│   ├── dukcapil_balanced.csv
│   └── kemenkes_balanced.csv
└── Models/               # Output folder untuk feature engineering
    └── fitur_global.pkl  # List kolom fitur global hasil one-hot encoding
```

---

## Dataset Generation

### Script: `generate.py`

Script ini menghasilkan 3 dataset simulasi dengan karakteristik berbeda untuk setiap instansi.

#### Konfigurasi Default

```python
N_ROWS = 100_000           # Jumlah baris per dataset
RATIO = 0.5                # Balanced: 50% layak : 50% tidak layak
NOISE_PCT = 0.05           # 5% label dibalik untuk simulasi noise
```

#### 1️⃣ DINSOS Dataset

**Fitur:**
- `penghasilan` (int): Penghasilan bulanan
- `jumlah_tanggungan` (int): Jumlah anggota keluarga yang ditanggung
- `kondisi_rumah` (categorical): Kondisi fisik rumah
- `status_pekerjaan` (categorical): Jenis pekerjaan
- `pendidikan` (categorical): Tingkat pendidikan terakhir
- `lama_tinggal_tahun` (int): Lama tinggal di domisili saat ini
- `layak_subsidi` (binary): Target label

**Karakteristik:**
- **Layak subsidi**: Penghasilan rendah (500K-2M), tanggungan banyak (2-8 orang), kondisi rumah tidak layak/sederhana
- **Tidak layak**: Penghasilan tinggi (3.5M-15M), tanggungan sedikit (0-3 orang), kondisi rumah layak/mewah

#### 2️⃣ DUKCAPIL Dataset

**Fitur:**
- `nik_valid` (categorical): Validitas NIK (ya/tidak)
- `memiliki_kk` (categorical): Kepemilikan Kartu Keluarga
- `domisili_tetap` (categorical): Status domisili tetap
- `data_ganda` (categorical): Indikasi duplikasi data
- `masuk_dtks` (categorical): Terdaftar di Data Terpadu Kesejahteraan Sosial
- `status_perkawinan` (categorical): Status perkawinan
- `pekerjaan_kepala_keluarga` (categorical): Pekerjaan kepala keluarga
- `jumlah_anggota_kk` (int): Jumlah anggota dalam KK
- `usia_kepala_keluarga` (int): Usia kepala keluarga
- `layak_subsidi` (binary): Target label

**Karakteristik:**
- **Layak subsidi**: NIK valid 95%, punya KK 95%, domisili tetap 90%, masuk DTKS 90%, data ganda hanya 5%
- **Tidak layak**: NIK valid hanya 40%, punya KK 50%, data ganda 80%, masuk DTKS hanya 20%

#### 3️⃣ KEMENKES Dataset

**Fitur:**
- `penghasilan` (int): Penghasilan bulanan
- `punya_asuransi_lain` (categorical): Kepemilikan asuransi lain
- `penyakit_kronis` (categorical): Jenis penyakit kronis
- `status_pekerjaan` (categorical): Jenis pekerjaan
- `kondisi_rumah` (categorical): Kondisi rumah
- `status_gizi` (categorical): Status gizi keluarga
- `layak_subsidi` (binary): Target label

**Karakteristik:**
- **Layak subsidi**: Penghasilan rendah, tidak punya asuransi lain, punya penyakit kronis, status gizi kurang/stunting
- **Tidak layak**: Penghasilan tinggi, sudah punya asuransi lain, tidak ada penyakit kronis, status gizi baik

#### Noise Simulation

Untuk mensimulasikan kondisi dunia nyata, 5% dari label akan di-flip secara random menggunakan fungsi `flip_labels()`. Ini merefleksikan:
- Kesalahan input data manual
- Inkonsistensi dalam proses labeling
- Edge cases yang sulit dikategorikan

---

## Feature Engineering

### Script: `Feature_Cols.py`

Script ini mengekstraksi **fitur global** dari ketiga dataset untuk memastikan semua model machine learning menggunakan feature space yang konsisten.

#### Proses:

1. **Load semua dataset** dari folder `DATASET/`
   ```python
   dfs = [
       pd.read_csv("DATASET/dinsos_balanced.csv"),
       pd.read_csv("DATASET/dukcapil_balanced.csv"),
       pd.read_csv("DATASET/kemenkes_balanced.csv"),
   ]
   ```

2. **Menggabungkan semua fitur** (tanpa kolom target)
   ```python
   all_features = pd.concat([df.drop(columns=["layak_subsidi"]) for df in dfs], axis=0)
   ```

3. **One-Hot Encoding global** untuk semua fitur kategorikal
   ```python
   global_encoded = pd.get_dummies(all_features, drop_first=False)
   global_cols = list(global_encoded.columns)
   ```

4. **Simpan list kolom global** ke file pickle
   ```python
   joblib.dump(global_cols, "Models/fitur_global.pkl")
   ```

#### Mengapa Diperlukan?

Karena setiap instansi memiliki fitur kategorikal yang berbeda-beda (misalnya `kondisi_rumah` di Dinsos vs Kemenkes memiliki kategori yang berbeda), kita perlu:
- **Feature space yang konsisten** untuk training model
- **Mapping universal** untuk one-hot encoding
- **Compatibility** antara model dari berbagai instansi

Output `fitur_global.pkl` berisi **list nama kolom** setelah one-hot encoding yang akan digunakan oleh semua model downstream.

---

## Cara Menjalankan

### 1. Generate Dataset

Jalankan script untuk membuat dataset simulasi:

```bash
python generate.py
```

**Output:**
```
🚀 Membuat dinsos_balanced.csv ...
dinsos_balanced.csv: 100000 baris | Layak=50,000 | Tidak Layak=50,000

🚀 Membuat dukcapil_balanced.csv ...
dukcapil_balanced.csv: 100000 baris | Layak=50,000 | Tidak Layak=50,000

🚀 Membuat kemenkes_balanced.csv ...
kemenkes_balanced.csv: 100000 baris | Layak=50,000 | Tidak Layak=50,000
```

### 2. Extract Global Features

Setelah dataset terbentuk, jalankan feature engineering:

```bash
python Feature_Cols.py
```

**Output:**
```
📂 Membaca dataset dari ketiga instansi...
🔧 Menggabungkan semua fitur unik dari ketiga instansi...
✅ Fitur global (LIST) tersimpan di 'Models/fitur_global.pkl'
📏 Total fitur unik: [jumlah] kolom
```

---

## Output Files

### DATASET/ Folder

| File | Size | Deskripsi |
|------|------|-----------|
| `dinsos_balanced.csv` | ~3.8 MB | Dataset Dinas Sosial (100k rows) |
| `dukcapil_balanced.csv` | ~4.7 MB | Dataset Dukcapil (100k rows) |
| `kemenkes_balanced.csv` | ~5.1 MB | Dataset Kemenkes (100k rows) |

### Models/ Folder

| File | Size | Deskripsi |
|------|------|-----------|
| `fitur_global.pkl` | ~1.3 KB | List kolom fitur global hasil one-hot encoding |

#### Cara Membaca `fitur_global.pkl`

```python
import joblib

# Load list kolom global
global_cols = joblib.load("Models/fitur_global.pkl")

print(f"Total fitur: {len(global_cols)}")
print(f"Contoh kolom: {global_cols[:10]}")
```

---

## Dependencies

```python
pandas
numpy
joblib
pathlib (built-in)
```

Install dependencies:
```bash
pip install pandas numpy joblib
```

---

## Notes

1. **Random Seed**: Script menggunakan `np.random.seed(42)` untuk reproducibility
2. **Balanced Dataset**: Semua dataset menggunakan ratio 50:50 untuk kelas positif dan negatif
3. **Noise Level**: Default 5% noise dapat diubah melalui parameter `NOISE_PCT`
4. **File Size**: Total ~13.6 MB untuk ketiga dataset (100k rows each)

---

## Workflow Diagram

```
┌─────────────────┐
│  generate.py    │
│  (Dataset Gen)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DATASET/       │
│  ├─ dinsos.csv  │
│  ├─ dukcapil... │
│  └─ kemenkes... │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│Feature_Cols.py  │
│(Feature Eng.)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Models/        │
│  └─fitur_global │
│     .pkl        │
└─────────────────┘
```

---

## Next Steps

Setelah mendapatkan `fitur_global.pkl`, file ini dapat digunakan untuk:
1. **Training model** dengan feature space yang konsisten
2. **Preprocessing data baru** menggunakan kolom yang sama
3. **Federated Learning** antar instansi dengan feature alignment

---

**Created by**: Data Engineering Team  
**Last Updated**: 2026-01-08
