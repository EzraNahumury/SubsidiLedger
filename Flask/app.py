import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify
from keras.layers import TFSMLayer

# ============================================================
# ⚙️ Konfigurasi Flask
# ============================================================
app = Flask(__name__, template_folder="templates")

# ============================================================
# 📦 Konfigurasi Model
# ============================================================
MODELS = {
    "dinsos": {
        "path": "Models/saved_dinsos_tff",
        "preproc": "Models/saved_dinsos_tff/preprocess_dinsos.pkl"
    },
    "dukcapil": {
        "path": "Models/saved_dukcapil_tff",
        "preproc": "Models/saved_dukcapil_tff/preprocess_dukcapil.pkl"
    },
    "kemenkes": {
        "path": "Models/saved_kemenkes_tff",
        "preproc": "Models/saved_kemenkes_tff/preprocess_kemenkes.pkl"
    },
    "gabungan": {
        "path": "Models/saved_global2_tff",
        "preproc": "Models/fitur_global.pkl"
    },
}


# ============================================================
# 🧩 Fungsi Preprocessing Input
# ============================================================
def preprocess_input(data, preproc):
    """Melakukan preprocessing input dari form sesuai fitur model"""
    feature_cols = list(preproc["FEATURE_COLS"])
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    def norm(x):
        return str(x).strip().lower()

    CANON = {
        "status_pekerjaan": {
            "karyawan tetap": "pegawai tetap",
            "pegawai tetap": "pegawai tetap",
            "pns": "PNS",
            "buruh": "buruh harian",
        },
        "kondisi_rumah": {
            "tdk layak": "tidak layak",
            "sangat sederhana": "sangat sederhana",
            "semi permanen": "semi permanen",
            "sederhana": "sederhana",
            "layak": "layak",
            "mewah": "mewah",
        }
    }

    # DataFrame kosong berisi semua kolom fitur
    df = pd.DataFrame([0]*len(feature_cols), index=feature_cols).T

    # Isi kolom numerik
    for num_col in ["penghasilan", "jumlah_tanggungan", "lama_tinggal_tahun",
                    "jumlah_anggota_kk", "usia_kepala_keluarga"]:
        if num_col in data and str(data[num_col]).strip() != "":
            try:
                df[num_col] = float(data[num_col])
            except:
                df[num_col] = 0.0

    # One-hot encoding (exact match)
    for col in feature_cols:
        if "_" not in col:
            continue
        base, cat = col.rsplit("_", 1)
        if base in data:
            raw = data[base]
            v = norm(raw)
            if base in CANON:
                v = norm(CANON[base].get(v, raw))
            if norm(cat) == v:
                df[col] = 1

    # Normalisasi sesuai skala training
    df = ((df - mins) / rng).fillna(0.0)
    return df.astype("float32").to_numpy()


# ============================================================
# 🧠 Prediksi Model dengan Threshold Otomatis Adaptif
# ============================================================
def predict_with_threshold(model_name, data):
    model_path = MODELS[model_name]["path"]
    preproc_path = MODELS[model_name]["preproc"]

    if not os.path.exists(model_path):
        return {"error": f"Model {model_name} tidak ditemukan."}

    # Load model & preprocessor
    model = TFSMLayer(model_path, call_endpoint="serving_default")
    preproc = joblib.load(preproc_path)

    # Preprocess input
    X = preprocess_input(data, preproc)
    outputs = model(X, training=False)
    y_prob = float(list(outputs.values())[0].numpy().flatten()[0])

    # ========================================================
    # 🔍 Threshold Otomatis (Dynamic)
    # ========================================================
    # logika adaptif berdasarkan probabilitas
    if y_prob < 0.48:
        best_t = 0.42
    elif y_prob < 0.55:
        best_t = 0.5
    elif y_prob < 0.65:
        best_t = 0.53
    else:
        best_t = 0.55

    # Hasil prediksi akhir
    y_pred = int(y_prob >= best_t)

    return {
        "prediksi": y_pred,
        "probabilitas": round(y_prob, 4),
        "threshold": round(best_t, 3)
    }


# ============================================================
# 🌐 ROUTES
# ============================================================
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict/<model>", methods=["POST"])
def predict(model):
    if model not in MODELS:
        return jsonify({"error": "Model tidak dikenali"}), 400

    data = request.get_json(force=True)
    result = predict_with_threshold(model, data)
    return jsonify(result)


# ============================================================
# 🚀 MAIN ENTRY
# ============================================================
if __name__ == "__main__":
    app.run(debug=True)
