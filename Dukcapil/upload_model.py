import os
import time
import base64
import numpy as np
import requests
import tensorflow as tf
from pathlib import Path

# ======================================================
# ⚙️ KONFIGURASI
# ======================================================
SERVER_URL   = "https://federatedinstitusi.up.railway.app"
CLIENT_NAME  = "dukcapil"   # WAJIB lowercase & konsisten

MODEL_PATH   = Path("models/saved_dukcapil_tff")

TIMEOUT      = 180  # detik
RETRY_LIMIT  = 3

# SESUAIKAN DENGAN MODEL TRAINING
EXPECTED_WEIGHTS = 12   # Dense + BN + Dense + Dense

# ======================================================
# 🔍 LOAD MODEL LOKAL
# ======================================================
def load_local_model(model_path: Path) -> tf.keras.Model:
    print(f"📂 Memuat model dari: {model_path}")

    try:
        model = tf.keras.models.load_model(model_path)
        print("✅ Model Keras berhasil dimuat")
        return model
    except Exception as e:
        print(f"⚠️ Gagal load Keras: {e}")
        print("➡️ Mencoba TFSMLayer...")

    from tensorflow import keras
    model = keras.Sequential([
        keras.layers.TFSMLayer(
            str(model_path),
            call_endpoint="serving_default"
        )
    ])

    print("✅ Model SavedModel dimuat via TFSMLayer")
    return model

# ======================================================
# 💾 SIMPAN BOBOT → NPZ
# ======================================================
def save_weights_npz(model: tf.keras.Model, save_path: Path):
    weights = [w.numpy() for w in model.weights]
    np.savez_compressed(save_path, *weights)

    size_mb = save_path.stat().st_size / 1024 / 1024
    print(f"💾 Bobot disimpan: {save_path.name} ({size_mb:.2f} MB)")

# ======================================================
# 📊 LOAD METRICS
# ======================================================
def load_metrics(model_dir: Path):
    metrics = {}

    best_acc = model_dir / "best_accuracy.txt"
    hist_acc = model_dir / "accuracy_history.txt"

    if best_acc.exists():
        try:
            metrics["best_accuracy"] = float(best_acc.read_text().strip())
        except Exception:
            pass

    if hist_acc.exists():
        try:
            lines = hist_acc.read_text().splitlines()
            metrics["history_tail"] = lines[-10:]  # diperkecil
        except Exception:
            pass

    return metrics

# ======================================================
# 📦 CARI FILE NPZ TERBARU
# ======================================================
def find_existing_npz(model_dir: Path) -> Path | None:
    npz_files = list(model_dir.glob(f"{CLIENT_NAME}_*.npz"))
    if not npz_files:
        return None

    npz_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return npz_files[0]

# ======================================================
# 🔎 VALIDASI WEIGHT
# ======================================================
def validate_npz(npz_path: Path):
    data = np.load(npz_path)
    n_weights = len(data.files)

    print(f"🔎 Validasi weight: {n_weights} tensor")

    if n_weights != EXPECTED_WEIGHTS:
        raise ValueError(
            f"❌ Jumlah weight tidak sesuai! "
            f"Expected {EXPECTED_WEIGHTS}, got {n_weights}"
        )

# ======================================================
# 📡 UPLOAD KE SERVER
# ======================================================
def upload_model_to_server(npz_path: Path, model_dir: Path):
    print(f"📦 Menggunakan bobot: {npz_path.name}")

    validate_npz(npz_path)

    with open(npz_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "client": CLIENT_NAME,
        "compressed_weights": encoded,
        "framework": "tensorflow",
        "model_version": "v1.0",
    }

    metrics = load_metrics(model_dir)
    if metrics:
        payload["metrics"] = metrics

    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            print(f"📡 Upload model ({CLIENT_NAME}) percobaan {attempt}...")
            start = time.time()

            res = requests.post(
                f"{SERVER_URL}/upload-model",
                json=payload,
                timeout=TIMEOUT
            )

            dur = time.time() - start

            if res.status_code == 200:
                print(f"✅ Upload sukses ({dur:.2f} detik)")
                print("📨 Server response:", res.json())
                return True
            else:
                print(f"⚠️ Server menolak ({res.status_code}): {res.text}")

        except requests.RequestException as e:
            print(f"❌ Gagal upload: {e}")
            time.sleep(3)

    return False

# ======================================================
# 🧠 MAIN
# ======================================================
if __name__ == "__main__":
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    npz_path = find_existing_npz(MODEL_PATH)

    if npz_path is None:
        print("📦 NPZ belum ada, membuat dari model...")
        model = load_local_model(MODEL_PATH)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        npz_path = MODEL_PATH / f"{CLIENT_NAME}_{timestamp}.npz"

        save_weights_npz(model, npz_path)
    else:
        size_mb = npz_path.stat().st_size / 1024 / 1024
        print(f"✅ Bobot ditemukan: {npz_path.name} ({size_mb:.2f} MB)")

    success = upload_model_to_server(npz_path, MODEL_PATH)

    if success:
        print(f"\n🎉 Model {CLIENT_NAME.upper()} berhasil dikirim ke server!")
    else:
        print(f"\n❌ Model {CLIENT_NAME.upper()} gagal dikirim.")
