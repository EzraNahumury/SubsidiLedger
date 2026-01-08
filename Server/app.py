#!/usr/bin/env python3
import os
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
import io
import json
import shutil
import zipfile
import tempfile
from werkzeug.utils import secure_filename

# ==========================================================
# 🚀 INISIALISASI FLASK + CORS
# ==========================================================
app = Flask(__name__)

# Ambil allowed frontend dari env (set di Railway)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

CORS(
    app,
    resources={r"/*": {"origins": [FRONTEND_URL, "http://localhost:3000"]}},
    supports_credentials=True,
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"]
)


# fallback CORS headers (pastikan send_file juga mendapat header)
@app.after_request
def apply_cors(response):
    origin = request.headers.get("Origin")
    if origin and origin in [FRONTEND_URL, "http://localhost:3000"]:
        response.headers["Access-Control-Allow-Origin"] = origin
    else:
        response.headers["Access-Control-Allow-Origin"] = FRONTEND_URL
    response.headers["Access-Control-Allow-Credentials"] = "true"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    return response

# ==========================================================
# 🗂️ FOLDERS
# ==========================================================
MODELS_DIR = Path("models")            # lowercase "models" dipakai konsisten
MODELS_DIR.mkdir(parents=True, exist_ok=True)

LOGS_DIR = MODELS_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# UTIL: path safety
# ==========================================================
def safe_model_path(filename: str):
    """
    Kembalikan path absolut aman untuk file di folder models.
    Mencegah path traversal (../../).
    """
    model_folder = MODELS_DIR.resolve()
    candidate = model_folder / filename
    try:
        candidate_resolved = candidate.resolve(strict=False)
        if not str(candidate_resolved).startswith(str(model_folder)):
            # Path traversal detected
            return None
        return candidate_resolved
    except Exception:
        return None

def remove_logs_for_client(client: str) -> dict:
    """
    Hapus best_accuracy & history untuk client dari LOGS_DIR dan juga
    dari folder client di MODELS_DIR jika ada.
    Mengembalikan dict berisi info berkas yg dihapus.
    """
    deleted = {"best": False, "history": False, "folder_best": False, "folder_history": False}
    try:
        best_path = LOGS_DIR / f"{client}_best_accuracy.txt"
        history_path = LOGS_DIR / f"{client}_accuracy_history.txt"

        if best_path.exists():
            try:
                best_path.unlink()
                deleted["best"] = True
            except Exception as e:
                print(f"⚠️ Gagal menghapus {best_path}: {e}")

        if history_path.exists():
            try:
                history_path.unlink()
                deleted["history"] = True
            except Exception as e:
                print(f"⚠️ Gagal menghapus {history_path}: {e}")

        # Cek juga jika ada folder models/<client>/best_accuracy.txt dll.
        client_folder = MODELS_DIR / client
        if client_folder.exists() and client_folder.is_dir():
            folder_best = client_folder / "best_accuracy.txt"
            folder_history = client_folder / "accuracy_history.txt"
            if folder_best.exists():
                try:
                    folder_best.unlink()
                    deleted["folder_best"] = True
                except Exception as e:
                    print(f"⚠️ Gagal menghapus {folder_best}: {e}")
            if folder_history.exists():
                try:
                    folder_history.unlink()
                    deleted["folder_history"] = True
                except Exception as e:
                    print(f"⚠️ Gagal menghapus {folder_history}: {e}")

    except Exception as e:
        print(f"⚠️ Error saat remove_logs_for_client({client}): {e}")
    return deleted

# ==========================================================
# 1️⃣ ENDPOINT: UPLOAD MODEL DARI CLIENT (dengan logging akurasi)
# ==========================================================
@app.route('/upload-model', methods=['POST'])
def upload_model():
    """
    Ekspektasi JSON body:
    {
      "client": "BANK_A",
      "compressed_weights": "<base64 npz>",
      "metrics": { "best_accuracy": 0.9123, "history": [...] }   # optional
      // atau "accuracy": 0.9123
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "invalid json body"}), 400

        client = data.get("client")
        compressed_weights = data.get("compressed_weights")

        if not client:
            return jsonify({"status": "error", "message": "client missing"}), 400
        if not compressed_weights:
            return jsonify({"status": "error", "message": "compressed_weights missing"}), 400

        # Decode base64 -> load npz from bytes
        try:
            binary_data = base64.b64decode(compressed_weights)
            buffer = io.BytesIO(binary_data)
            npzfile = np.load(buffer, allow_pickle=True)
            weights = [npzfile[key] for key in npzfile]
        except Exception as e:
            return jsonify({"status": "error", "message": f"failed to decode/load npz: {e}"}), 400

        # Simpan bobot model
        save_path = MODELS_DIR / f"{client}_weights.npz"
        np.savez_compressed(save_path, *weights)
        print(f"✅ Model dari {client} disimpan di {save_path}")

        # -------------------------
        # handle metrics / accuracy logging (accept various formats)
        # -------------------------
        metrics = data.get("metrics") or {}
        if isinstance(metrics, str):
            try:
                metrics = json.loads(metrics)
            except Exception:
                metrics = {}

        # possible scalar fields at top-level or inside metrics
        accuracy_value = None
        if isinstance(metrics, dict):
            accuracy_value = metrics.get("accuracy") or metrics.get("best_accuracy")
        if accuracy_value is None:
            accuracy_value = data.get("accuracy") or data.get("best_accuracy")

        # possible history provided inside metrics
        history_items = None
        if isinstance(metrics, dict):
            history_items = metrics.get("history") or metrics.get("accuracy_history")

        best_path = LOGS_DIR / f"{client}_best_accuracy.txt"
        history_path = LOGS_DIR / f"{client}_accuracy_history.txt"

        metrics_log = {}

        # 1) append history entries if provided (history_items)
        if history_items:
            try:
                lines = []
                if isinstance(history_items, str):
                    for ln in history_items.splitlines():
                        if ln.strip():
                            lines.append(ln.strip())
                elif isinstance(history_items, list):
                    for item in history_items:
                        if isinstance(item, dict):
                            r = item.get("round", "")
                            accv = item.get("acc") or item.get("accuracy") or item.get("value", "")
                            ts = item.get("timestamp", "") or item.get("time", "")
                            if r != "":
                                try:
                                    accf = float(accv) if accv != "" else ""
                                    lines.append(f"{r}\t{accf:.6f}\t{ts}")
                                except Exception:
                                    lines.append(json.dumps(item, ensure_ascii=False))
                            else:
                                lines.append(json.dumps(item, ensure_ascii=False))
                        else:
                            lines.append(str(item))
                else:
                    lines = [str(history_items)]

                with open(history_path, "a", encoding="utf-8") as hf:
                    for ln in lines:
                        ln = ln.strip()
                        if not ln:
                            continue
                        hf.write(ln + "\n")

                metrics_log["history_written"] = len(lines)
                metrics_log["history_path"] = str(history_path)
                print(f"📈 History untuk {client} ditambahkan ({len(lines)} baris) -> {history_path}")
            except Exception as e:
                metrics_log["history_error"] = str(e)
                print(f"⚠️ Gagal menulis history untuk {client}: {e}")

        # 2) handle scalar/best accuracy update (update best jika meningkat)
        if accuracy_value is not None:
            try:
                acc = float(accuracy_value)
                acc = max(0.0, min(1.0, acc))  # clamp to [0,1]

                timestamp = datetime.utcnow().isoformat() + "Z"
                # append to history as timestamped entry
                try:
                    with open(history_path, "a", encoding="utf-8") as hf:
                        hf.write(f"{timestamp}\t{acc:.6f}\n")
                except Exception:
                    pass

                prev_val = -1.0
                if best_path.exists():
                    try:
                        prev_txt = best_path.read_text(encoding="utf-8").strip()
                        prev_val = float(prev_txt) if prev_txt else -1.0
                    except Exception:
                        prev_val = -1.0

                written_best = False
                if acc > prev_val:
                    best_path.write_text(f"{acc:.6f}\n", encoding="utf-8")
                    written_best = True

                metrics_log.update({
                    "best_path": str(best_path),
                    "reported_accuracy": acc,
                    "written_best": written_best
                })
                print(f"📈 Metrics diterima dari {client}: acc={acc:.6f} -> log tersimpan")
            except Exception as e:
                metrics_log["accuracy_error"] = str(e)
                print(f"⚠️ Gagal memproses accuracy untuk {client}: {e}")

        # build response
        resp = {
            "status": 200,
            "client": client,
            "saved_weights": str(save_path),
            "message": "model uploaded"
        }
        if metrics_log:
            resp["metrics"] = metrics_log

        return jsonify(resp), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================================
# 2️⃣ ENDPOINT: AGREGASI SEMUA MODEL (FedAvg sederhana)
# ==========================================================
LAST_WEIGHT_FILE = MODELS_DIR / "last_avg_weight.json"

@app.route('/aggregate', methods=['POST'])
def aggregate_models():
    try:
        model_dir = MODELS_DIR
        client_files = [f for f in os.listdir(model_dir) if f.endswith("_weights.npz")]

        # Data size (optional) → untuk perhitungan kontribusi FedAvg
        req_json = request.get_json(silent=True) or {}
        data_sizes = req_json.get("data_sizes", {})

        # Jika model kurang dari 2 → beri pesan lebih informatif
        if len(client_files) < 2:
            if len(client_files) == 0:
                msg = (
                    "Tidak ada model lokal yang ditemukan. "
                    "Setidaknya dua client harus mengirimkan model sebelum agregasi dapat dilakukan."
                )
            else:  # hanya 1 model
                msg = (
                    f"Hanya ditemukan 1 model lokal ({client_files[0]}). "
                    "Minimal 2 model diperlukan untuk melakukan Federated Averaging."
                )

            return jsonify({
                "status": "error",
                "message": msg,
                "found_models": client_files,
                "required": 2,
                "current": len(client_files)
            }), 400

        print(f"🧮 Memulai Federated Averaging untuk {len(client_files)} client...")

        # =======================================
        # LOAD semua bobot client
        # =======================================
        all_weights = []
        client_weights_dict = {}   # untuk dimasukkan ke response
        client_mean_dict = {}      # untuk perhitungan kontribusi mean weight

        for fname in client_files:
            npz = np.load(model_dir / fname, allow_pickle=True)
            weights = [npz[key] for key in npz]
            all_weights.append(weights)

            # convert ke JSON-friendly
            client_weights_dict[fname] = [w.tolist() for w in weights]

            # Hitung rata-rata bobot client
            flat = np.concatenate([w.flatten() for w in weights])
            client_mean_dict[fname] = float(np.mean(flat))

            print(f"✅ {fname} dimuat ({len(weights)} layer)")

        num_layers = len(all_weights[0])
        avg_weights = []

        # =======================================
        # RATA-RATA FEDAVG
        # =======================================
        for layer_idx in range(num_layers):
            layer_values = [client[layer_idx] for client in all_weights]
            try:
                stacked = np.stack(layer_values, axis=0)
                averaged = np.mean(stacked, axis=0)
                avg_weights.append(averaged)
            except Exception:
                avg_weights.append(layer_values[-1])
                print(f"⚠️ Layer {layer_idx} BatchNorm moving stats, tidak di-average")

        # =======================================
        # SIMPAN MODEL GLOBAL
        # =======================================
        from datetime import datetime

        # ============================
        # BUAT NAMA FILE PAKAI TIMESTAMP
        # ============================
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"global_model_fedavg_{timestamp}.npz"
        save_path = model_dir / filename

        np.savez_compressed(save_path, *avg_weights)

        print(f"🎯 FedAvg selesai → disimpan di {save_path}")


        # =======================================
        # HITUNG TOTAL PARAMETER
        # =======================================
        total_params = sum(w.size for w in avg_weights)

        # =======================================
        # HITUNG RATA-RATA BOBOT GLOBAL
        # =======================================
        flat_weights = np.concatenate([w.flatten() for w in avg_weights])
        avg_global_weight = float(np.mean(flat_weights))

        # =======================================
        # BACA NILAI SEBELUMNYA
        # =======================================
        last_weight_path = model_dir / "last_avg_weight.json"
        last_weight = None

        if last_weight_path.exists():
            with open(last_weight_path, "r") as f:
                last_weight = json.load(f).get("avg_global_weight", None)

        # =======================================
        # HITUNG PERSENTASE PERUBAHAN GLOBAL
        # =======================================
        if last_weight is not None and last_weight != 0:
            change_percent = ((avg_global_weight - last_weight) / abs(last_weight)) * 100
        else:
            change_percent = 0.0  # Agregasi pertama

        # =======================================
        # SIMPAN NILAI TERBARU
        # =======================================
        with open(last_weight_path, "w") as f:
            json.dump({"avg_global_weight": avg_global_weight}, f)

        # =======================================
        # KONTRIBUSI 1 — Persentase berdasarkan Mean Weight
        # =======================================
        abs_means = {c: abs(v) for c, v in client_mean_dict.items()}
        total_abs_mean = sum(abs_means.values())

        mean_weight_percentage = {
            c: round((abs_means[c] / total_abs_mean) * 100, 4) if total_abs_mean != 0 else 0
            for c in client_files
        }

        # =======================================
        # KONTRIBUSI 2 — FedAvg contribution (berdasarkan jumlah data)
        # =======================================
        if data_sizes:
            total_data = sum(data_sizes.values())
            fedavg_contrib = {
                c: round((data_sizes.get(c, 0) / total_data) * 100, 4) if total_data != 0 else 0
                for c in client_files
            }
        else:
            fedavg_contrib = None  # Jika tidak ada data size

        # =======================================
        # RESPONSE SUCCESS
        # =======================================
        # =======================================
        # RESPONSE SUCCESS
        # =======================================

        response_json = {
            "status": "success",
            "method": "FedAvg",
            "num_clients": len(client_files),
            "num_layers": num_layers,
            "total_parameters": int(total_params),
            "avg_global_weight": avg_global_weight,
            "avg_global_weight_change_percent": round(change_percent, 6),
            "saved": str(save_path),

            # "client_weights": client_weights_dict,
            "client_mean_weight": client_mean_dict,
            "client_mean_weight_percentage": mean_weight_percentage,
            "fedavg_data_contribution_percentage": fedavg_contrib
        }

        # 🔥 INI YANG AKAN MUNCUL DI RAILWAY LOG
        print("\n========== FEDAVG JSON RESULT ==========")
        print(json.dumps(response_json, indent=2))
        print("========================================\n")

        # ⬇️ Baru return JSON ke client
        return jsonify(response_json)


    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ==========================================================
# 3️⃣ LIST FILES DI FOLDER models
# ==========================================================
@app.route('/logs', methods=['GET'])
def list_files():
    try:
        files = os.listdir(MODELS_DIR)
        result = []

        for fname in files:

            # Abaikan folder
            if os.path.isdir(MODELS_DIR / fname):
                continue

            # Hanya file .npz
            if not fname.endswith(".npz"):
                continue

            file_path = MODELS_DIR / fname
            file_mtime = file_path.stat().st_mtime   # UNIX timestamp

            # ============================
            # DETEKSI GLOBAL MODEL BARU
            # ============================
            if fname.startswith("global_model_fedavg_"):
                client_label = "GLOBAL"
                message = "Model global hasil agregasi FedAvg"
            else:
                # File client → contoh: bankA_weights.npz
                client_label = fname.replace("_weights.npz", "").upper()
                message = f"Model dari client {client_label}"

            # Format timestamp ISO
            timestamp = datetime.utcfromtimestamp(file_mtime).isoformat() + "Z"

            result.append({
                "client": client_label,
                "name": fname,
                "message": message,
                "timestamp": timestamp,
                "mtime": file_mtime   # Untuk sorting
            })

        # ======================
        # SORT BY NEWEST FIRST
        # ======================
        result.sort(key=lambda x: x["mtime"], reverse=True)

        # hapus field mtime sebelum dikirim ke client
        for item in result:
            item.pop("mtime", None)

        return jsonify({
            "status": 200,
            "files": result,
            "message": "Daftar model berhasil diambil dari server"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ==========================================================
# 4️⃣ DOWNLOAD GLOBAL MODELS
# ==========================================================
@app.route('/download-global', methods=['GET'])
def download_global():
    try:
        # Cari semua file global FedAvg dengan timestamp
        global_files = list(MODELS_DIR.glob("global_model_fedavg_*.npz"))

        if not global_files:
            return jsonify({
                "status": "error",
                "message": (
                    "Model global belum tersedia. "
                    "Silakan jalankan endpoint /aggregate setelah minimal 2 model lokal terkirim."
                ),
                "hint": "Pastikan minimal dua client telah mengunggah model mereka."
            }), 404

        # Ambil file terbaru berdasarkan waktu modifikasi
        latest_file = max(global_files, key=lambda f: f.stat().st_mtime)

        file_size = latest_file.stat().st_size
        last_modified = latest_file.stat().st_mtime

        print(f"📤 Global model diunduh: {latest_file.name} ({file_size} bytes)")

        # ========== BUAT NAMA FILE BARU UNTUK DOWNLOAD ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        download_name = f"global_model_fedavg_{timestamp}.npz"

        response = send_file(
            str(latest_file),
            as_attachment=True,
            download_name=download_name  # nama file saat di-download
        )

        # Tambahan header info
        response.headers["X-File-Name"] = download_name
        response.headers["X-File-Size"] = file_size
        response.headers["X-Last-Modified"] = last_modified
        response.headers["X-Description"] = "Model global terbaru hasil Federated Averaging"

        return response

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ==========================================================
# 5️⃣ DOWNLOAD FILE SPESIFIK
# ==========================================================

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    safe_path = safe_model_path(filename)
    if safe_path is None or not safe_path.exists():
        return jsonify({"status": "error", "message": f"{filename} tidak ditemukan"}), 404
    return send_file(str(safe_path), as_attachment=True)

# ==========================================================
# 🗑️ HAPUS MODEL (aman) — sekarang juga menghapus history/metrics terkait
# ==========================================================
from urllib.parse import unquote

@app.route('/delete/<path:filename>', methods=['DELETE'])
def delete_file(filename):
    # decode spasi dan karakter URL lain
    filename = unquote(filename)

    safe_path = safe_model_path(filename)
    if safe_path is None:
        return jsonify({"status": "error", "message": "Invalid filename"}), 400
    if not safe_path.exists():
        return jsonify({"status": "error", "message": f"File {filename} tidak ditemukan"}), 404

    try:
        safe_path.unlink()
        print(f"🗑️ File dihapus: {safe_path}")

        # Extract client name before "_weights"
        client = None
        if filename.endswith("_weights.npz"):
            client = filename[:-len("_weights.npz")]

        deleted_logs_info = None
        if client:
            deleted_logs_info = remove_logs_for_client(client)
            print(f"🗑️ Logs dihapus untuk client={client}: {deleted_logs_info}")

        result = {"status": "success", "deleted": filename}
        if deleted_logs_info is not None:
            result["deleted_logs"] = deleted_logs_info

        return jsonify(result)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/delete-model', methods=['POST'])
def delete_model_by_client():
    try:
        data = request.get_json() or {}
        client = data.get("client")
        filename = data.get("filename")

        # Build filename correctly even with spaces
        if client:
            target = f"{client}_weights.npz"
        elif filename:
            target = filename
        else:
            return jsonify({"status": "error", "message": "client atau filename required"}), 400

        safe_path = safe_model_path(target)
        if safe_path is None:
            return jsonify({"status": "error", "message": "Invalid filename"}), 400
        if not safe_path.exists():
            return jsonify({"status": "error", "message": f"{target} tidak ditemukan"}), 404

        safe_path.unlink()
        print(f"🗑️ Model dihapus: {safe_path}")

        # Determine client_name
        if client:
            client_name = client
        elif target.endswith("_weights.npz"):
            client_name = target[:-len("_weights.npz")]
        else:
            client_name = None

        deleted_logs_info = None
        if client_name:
            deleted_logs_info = remove_logs_for_client(client_name)
            print(f"🗑️ Logs dihapus untuk client={client_name}: {deleted_logs_info}")

        resp = {"status": "success", "deleted": target}
        if deleted_logs_info is not None:
            resp["deleted_logs"] = deleted_logs_info

        return jsonify(resp)

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



# ==========================================================
# Endpoint: ambil best accuracy & tail history untuk client
# - Cek models/logs/<client>_best_accuracy.txt terlebih dahulu
# - Jika tidak ada, cari folder yang cocok di models/* dan baca best_accuracy.txt di sana
# ==========================================================
@app.route('/accuracy/<client>', methods=['GET'])
def get_accuracy(client):
    try:
        # 1) Cek logs folder dulu (preferred)
        best_path = LOGS_DIR / f"{client}_best_accuracy.txt"
        history_path = LOGS_DIR / f"{client}_accuracy_history.txt"

        best = None
        history_tail = []
        source = None

        if best_path.exists():
            # baca best dari logs
            try:
                with open(best_path, "r", encoding="utf-8") as bf:
                    best = float(bf.read().strip())
                source = str(best_path)
            except Exception:
                best = None

            # baca beberapa baris terakhir dari history jika ada
            if history_path.exists():
                try:
                    with open(history_path, "r", encoding="utf-8") as hf:
                        lines = hf.read().strip().splitlines()
                        history_tail = lines[-20:] if len(lines) > 20 else lines
                except Exception:
                    history_tail = []

            return jsonify({"client": client, "best_accuracy": best, "history_tail": history_tail, "source": source})

        # 2) Jika tidak ada, cari folder model yang cocok di MODELS_DIR
        found_folder = None
        for p in MODELS_DIR.iterdir():
            if p.is_dir() and client.lower() in p.name.lower():
                found_folder = p
                break

        # Jika ditemukan folder, coba baca best_accuracy.txt di sana
        if found_folder:
            candidate_best = found_folder / "best_accuracy.txt"
            candidate_history = found_folder / "accuracy_history.txt"  # mungkin tidak ada
            if candidate_best.exists():
                try:
                    with open(candidate_best, "r", encoding="utf-8") as bf:
                        best = float(bf.read().strip())
                    source = str(candidate_best)
                except Exception:
                    best = None

            if candidate_history.exists():
                try:
                    with open(candidate_history, "r", encoding="utf-8") as hf:
                        lines = hf.read().strip().splitlines()
                        history_tail = lines[-20:] if len(lines) > 20 else lines
                except Exception:
                    history_tail = []

            return jsonify({"client": client, "best_accuracy": best, "history_tail": history_tail, "source": source})

        # 3) Tidak ditemukan
        return jsonify({"client": client, "best_accuracy": None, "history_tail": [], "source": None})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# ==========================================================
# 7️⃣ HOME
# ==========================================================
@app.route("/")
def home():
    return {
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

# ==========================================================
# 8️⃣ RUN SERVER (untuk Railway)
# ==========================================================
if __name__ == "__main__":
    # Gunakan port 8080 karena Railway expects that
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
