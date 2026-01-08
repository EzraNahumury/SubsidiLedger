# 🌍 API Endpoint Documentation - Federated Learning Server

Dokumentasi lengkap untuk semua endpoint yang tersedia di `app.py`.

## Base URL
```
Production: <Your Railway URL>
Development: http://localhost:8080
```

---

## 📋 Daftar Endpoint

1. [GET /](#1-get--home) - Home/Status Server
2. [POST /upload-model](#2-post-upload-model) - Upload Model dari Client
3. [POST /aggregate](#3-post-aggregate) - Agregasi Model Global (FedAvg)
4. [GET /logs](#4-get-logs) - Daftar File Model
5. [GET /download-global](#5-get-download-global) - Download Model Global Terbaru
6. [GET /download/<filename>](#6-get-downloadfilename) - Download File Spesifik
7. [DELETE /delete/<filename>](#7-delete-deletefilename) - Hapus File Spesifik
8. [POST /delete-model](#8-post-delete-model) - Hapus Model via JSON
9. [GET /accuracy/<client>](#9-get-accuracyclient) - Ambil Best Accuracy Client

---

## 1. GET `/` (Home)

**Deskripsi**: Endpoint untuk mengecek status server dan melihat daftar endpoint yang tersedia.

### Request
```http
GET / HTTP/1.1
```

### Response Success (200 OK)
```json
{
  "message": "🌍 Federated Aggregation Server aktif!",
  "status": "online",
  "endpoints": {
    "/upload-model": "Upload model lokal dari client (POST)",
    "/aggregate": "Lakukan agregasi global (POST)",
    "/logs": "Lihat file di models (GET)",
    "/download/<filename>": "Download file (GET)",
    "/delete/<filename>": "Hapus file (DELETE)",
    "/delete-model": "Hapus file via POST JSON",
    "/accuracy/<client>": "Ambil best accuracy & riwayat (GET)"
  }
}
```

---

## 2. POST `/upload-model`

**Deskripsi**: Upload model bobot (weights) dari client ke server, termasuk metrics akurasi.

### Request Body
```json
{
  "client": "BANK_A",
  "compressed_weights": "<base64-encoded npz file>",
  "metrics": {
    "best_accuracy": 0.9123,
    "history": [
      {
        "round": 1,
        "accuracy": 0.8856,
        "timestamp": "2026-01-08T08:30:00Z"
      },
      {
        "round": 2,
        "accuracy": 0.9123,
        "timestamp": "2026-01-08T08:35:00Z"
      }
    ]
  }
}
```

**Alternatif Format (accuracy di top-level)**:
```json
{
  "client": "BANK_B",
  "compressed_weights": "<base64-encoded npz file>",
  "accuracy": 0.8945
}
```

### Response Success (200 OK)
```json
{
  "status": 200,
  "client": "BANK_A",
  "saved_weights": "models/BANK_A_weights.npz",
  "message": "model uploaded",
  "metrics": {
    "best_path": "models/logs/BANK_A_best_accuracy.txt",
    "reported_accuracy": 0.9123,
    "written_best": true,
    "history_written": 2,
    "history_path": "models/logs/BANK_A_accuracy_history.txt"
  }
}
```

### Response Error (400 Bad Request)
```json
{
  "status": "error",
  "message": "client missing"
}
```

---

## 3. POST `/aggregate`

**Deskripsi**: Melakukan agregasi Federated Averaging (FedAvg) dari semua model yang telah diupload.

### Request Body (Optional)
```json
{
  "data_sizes": {
    "BANK_A_weights.npz": 50000,
    "BANK_B_weights.npz": 30000,
    "BANK_C_weights.npz": 20000
  }
}
```

### Response Success (200 OK)
```json
{
  "status": "success",
  "method": "FedAvg",
  "num_clients": 3,
  "num_layers": 8,
  "total_parameters": 125432,
  "avg_global_weight": 0.00245,
  "avg_global_weight_change_percent": 1.234567,
  "saved": "models/global_model_fedavg_20260108_153045.npz",
  "client_mean_weight": {
    "BANK_A_weights.npz": 0.00251,
    "BANK_B_weights.npz": 0.00239,
    "BANK_C_weights.npz": 0.00246
  },
  "client_mean_weight_percentage": {
    "BANK_A_weights.npz": 34.2456,
    "BANK_B_weights.npz": 32.5123,
    "BANK_C_weights.npz": 33.2421
  },
  "fedavg_data_contribution_percentage": {
    "BANK_A_weights.npz": 50.0,
    "BANK_B_weights.npz": 30.0,
    "BANK_C_weights.npz": 20.0
  }
}
```

### Response Error - Insufficient Models (400 Bad Request)
```json
{
  "status": "error",
  "message": "Hanya ditemukan 1 model lokal (BANK_A_weights.npz). Minimal 2 model diperlukan untuk melakukan Federated Averaging.",
  "found_models": ["BANK_A_weights.npz"],
  "required": 2,
  "current": 1
}
```

---

## 4. GET `/logs`

**Deskripsi**: Mendapatkan daftar semua file model (.npz) yang ada di server, terurut berdasarkan waktu terbaru.

### Request
```http
GET /logs HTTP/1.1
```

### Response Success (200 OK)
```json
{
  "status": 200,
  "files": [
    {
      "client": "GLOBAL",
      "name": "global_model_fedavg_20260108_153045.npz",
      "message": "Model global hasil agregasi FedAvg",
      "timestamp": "2026-01-08T08:30:45Z"
    },
    {
      "client": "BANK_A",
      "name": "BANK_A_weights.npz",
      "message": "Model dari client BANK_A",
      "timestamp": "2026-01-08T08:28:12Z"
    },
    {
      "client": "BANK_B",
      "name": "BANK_B_weights.npz",
      "message": "Model dari client BANK_B",
      "timestamp": "2026-01-08T08:27:34Z"
    }
  ],
  "message": "Daftar model berhasil diambil dari server"
}
```

---

## 5. GET `/download-global`

**Deskripsi**: Download model global terbaru hasil agregasi FedAvg.

### Request
```http
GET /download-global HTTP/1.1
```

### Response Success (200 OK)
**Content-Type**: `application/octet-stream`  
**Headers**:
```
X-File-Name: global_model_fedavg_20260108_153045.npz
X-File-Size: 2048576
X-Last-Modified: 1704712845.123
X-Description: Model global terbaru hasil Federated Averaging
```
**Body**: Binary file (NPZ format)

### Response Error - No Global Model (404 Not Found)
```json
{
  "status": "error",
  "message": "Model global belum tersedia. Silakan jalankan endpoint /aggregate setelah minimal 2 model lokal terkirim.",
  "hint": "Pastikan minimal dua client telah mengunggah model mereka."
}
```

---

## 6. GET `/download/<filename>`

**Deskripsi**: Download file model spesifik berdasarkan nama file.

### Request
```http
GET /download/BANK_A_weights.npz HTTP/1.1
```

### Response Success (200 OK)
**Content-Type**: `application/octet-stream`  
**Body**: Binary file (NPZ format)

### Response Error (404 Not Found)
```json
{
  "status": "error",
  "message": "BANK_A_weights.npz tidak ditemukan"
}
```

---

## 7. DELETE `/delete/<filename>`

**Deskripsi**: Hapus file model spesifik dan log akurasi terkait.

### Request
```http
DELETE /delete/BANK_A_weights.npz HTTP/1.1
```

### Response Success (200 OK)
```json
{
  "status": "success",
  "deleted": "BANK_A_weights.npz",
  "deleted_logs": {
    "best": true,
    "history": true,
    "folder_best": false,
    "folder_history": false
  }
}
```

### Response Error (404 Not Found)
```json
{
  "status": "error",
  "message": "File BANK_A_weights.npz tidak ditemukan"
}
```

---

## 8. POST `/delete-model`

**Deskripsi**: Hapus model menggunakan client name atau filename melalui JSON request.

### Request Body (Via Client Name)
```json
{
  "client": "BANK_A"
}
```

### Request Body (Via Filename)
```json
{
  "filename": "BANK_A_weights.npz"
}
```

### Response Success (200 OK)
```json
{
  "status": "success",
  "deleted": "BANK_A_weights.npz",
  "deleted_logs": {
    "best": true,
    "history": true,
    "folder_best": false,
    "folder_history": false
  }
}
```

### Response Error (404 Not Found)
```json
{
  "status": "error",
  "message": "BANK_A_weights.npz tidak ditemukan"
}
```

---

## 9. GET `/accuracy/<client>`

**Deskripsi**: Mendapatkan best accuracy dan riwayat akurasi untuk client tertentu.

### Request
```http
GET /accuracy/BANK_A HTTP/1.1
```

### Response Success (200 OK) - With Data
```json
{
  "client": "BANK_A",
  "best_accuracy": 0.9123,
  "history_tail": [
    "2026-01-08T08:25:12Z\t0.885600",
    "2026-01-08T08:28:34Z\t0.901200",
    "2026-01-08T08:32:45Z\t0.912300"
  ],
  "source": "models/logs/BANK_A_best_accuracy.txt"
}
```

### Response Success (200 OK) - No Data
```json
{
  "client": "BANK_C",
  "best_accuracy": null,
  "history_tail": [],
  "source": null
}
```

---

## 🔒 CORS Configuration

Server dikonfigurasi dengan CORS untuk mendukung:
- **Allowed Origins**: 
  - Frontend URL dari environment variable `FRONTEND_URL`
  - `http://localhost:3000` (untuk development)
- **Allowed Methods**: GET, POST, PUT, DELETE, OPTIONS
- **Allowed Headers**: Content-Type, Authorization, X-Requested-With
- **Credentials**: Supported

---

## 📁 File Structure

```
models/
├── logs/
│   ├── BANK_A_best_accuracy.txt
│   ├── BANK_A_accuracy_history.txt
│   ├── BANK_B_best_accuracy.txt
│   └── BANK_B_accuracy_history.txt
├── BANK_A_weights.npz
├── BANK_B_weights.npz
├── global_model_fedavg_20260108_153045.npz
└── last_avg_weight.json
```

---

## ⚡ Quick Start Examples

### Example 1: Upload Model dari Client
```bash
curl -X POST http://localhost:8080/upload-model \
  -H "Content-Type: application/json" \
  -d '{
    "client": "BANK_A",
    "compressed_weights": "<base64_encoded_npz>",
    "accuracy": 0.9123
  }'
```

### Example 2: Agregasi Model
```bash
curl -X POST http://localhost:8080/aggregate \
  -H "Content-Type: application/json"
```

### Example 3: Download Model Global
```bash
curl -X GET http://localhost:8080/download-global \
  -o global_model.npz
```

### Example 4: Cek Status Server
```bash
curl -X GET http://localhost:8080/
```

---

## 🛡️ Error Handling

Semua endpoint mengembalikan error dalam format JSON:

```json
{
  "status": "error",
  "message": "Deskripsi error yang detail"
}
```

HTTP Status Codes yang digunakan:
- `200` - Success
- `400` - Bad Request (invalid input)
- `404` - Not Found (file tidak ditemukan)
- `500` - Internal Server Error

---

## 📝 Notes

1. **Model Upload**: Compressed weights harus dalam format base64-encoded NPZ file
2. **Agregasi**: Minimal 2 model client diperlukan sebelum agregasi dapat dilakukan
3. **Timestamps**: Semua timestamp dalam format ISO 8601 dengan timezone UTC
4. **Accuracy Range**: Akurasi secara otomatis di-clamp ke range [0.0, 1.0]
5. **File Safety**: Path traversal attacks dicegah dengan `safe_model_path()` function
