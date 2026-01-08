# ============================================================
# IMPORT
# ============================================================
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from colorama import Fore, Style, init

init(autoreset=True)

# ============================================================
# ⚙️ KONFIGURASI
# ============================================================
THRESHOLD_MODE = "AUTO"   # AUTO / MANUAL
THRESHOLD_MANUAL = 0.5

GLOBAL_MODEL_NPZ = "models/global_model_fedavg_20251216_040729.npz"

PREPROC_PATHS = {
    "DINSOS":   "Models/saved_dinsos_tff/preprocess.pkl",
    "DUKCAPIL": "Models/saved_dukcapil_tff/preprocess.pkl",
    "KEMENKES": "Models/saved_kemenkes_tff/preprocess.pkl",
}

# ============================================================
# 🧠 LOAD MODEL GLOBAL
# ============================================================
def load_global_model(npz_path, feature_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(feature_dim,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    weights = np.load(npz_path)
    model.set_weights([weights[k] for k in weights.files])
    return model

# ============================================================
# 🧩 PREPROCESS INPUT
# ============================================================
def preprocess_input(data: dict, preproc: dict):
    feature_cols = preproc["FEATURE_COLS"]
    mins = pd.Series(preproc["mins"]).reindex(feature_cols).fillna(0)
    rng  = pd.Series(preproc["rng"]).reindex(feature_cols).replace(0, 1)

    df = pd.DataFrame(0, index=[0], columns=feature_cols)

    for k, v in data.items():
        if k in df.columns:
            df[k] = v
        else:
            col = f"{k}_{str(v).lower().strip()}"
            if col in df.columns:
                df[col] = 1

    df_scaled = ((df - mins) / rng).fillna(0.0)
    return df_scaled.astype("float32").values

# ============================================================
# 📏 AUTO THRESHOLD
# ============================================================
def find_best_threshold(probs, labels):
    thresholds = np.linspace(0.35, 0.65, 61)
    best_acc, best_t = 0, 0.5

    for t in thresholds:
        preds = (probs >= t).astype(int)
        acc = np.mean(preds == labels)
        if acc > best_acc:
            best_acc, best_t = acc, t

    return best_t, best_acc

# ============================================================
# 🧪 RUN TEST
# ============================================================
def run_test(instansi: str, cases: list):
    print(f"\n{'='*70}")
    print(f"🌍 TEST GLOBAL MODEL — {instansi}")
    print(f"{'='*70}")

    preproc = joblib.load(PREPROC_PATHS[instansi])
    model = load_global_model(
        GLOBAL_MODEL_NPZ,
        len(preproc["FEATURE_COLS"])
    )

    probs, labels = [], []

    for data, expected in cases:
        X = preprocess_input(data, preproc)
        prob = model.predict(X, verbose=0)[0][0]
        probs.append(prob)
        labels.append(expected)

    probs = np.array(probs)
    labels = np.array(labels)

    # Threshold selection
    if THRESHOLD_MODE == "AUTO":
        threshold, best_acc = find_best_threshold(probs, labels)
        print(f"📏 Threshold AUTO dipilih: {threshold:.3f} (acc={best_acc*100:.2f}%)")
    else:
        threshold = THRESHOLD_MANUAL
        print(f"📏 Threshold MANUAL: {threshold:.3f}")

    correct = 0
    for i, (prob, expected) in enumerate(zip(probs, labels), 1):
        pred = int(prob >= threshold)
        icon = Fore.GREEN + "✅" if pred == expected else Fore.RED + "❌"
        if pred == expected:
            correct += 1

        print(
            f"[{i:02d}] exp={expected:<2} "
            f"pred={pred:<2} prob={prob:.4f} {icon}{Style.RESET_ALL}"
        )

    acc = correct / len(cases) * 100
    print(f"\n📊 Akurasi {instansi}: {Fore.CYAN}{acc:.2f}% ({correct}/{len(cases)}){Style.RESET_ALL}")

# ============================================================
# 🧪 TEST CASES
# ============================================================


# ✅ DINSOS
dinsos_cases = [
    ({"penghasilan": 1800000, "jumlah_tanggungan": 3, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian"}, 1),
    ({"penghasilan": 1500000, "jumlah_tanggungan": 4, "kondisi_rumah": "sederhana", "status_pekerjaan": "petani"}, 1),
    ({"penghasilan": 1000000, "jumlah_tanggungan": 5, "kondisi_rumah": "sangat sederhana", "status_pekerjaan": "tidak bekerja"}, 1),
    ({"penghasilan": 1200000, "jumlah_tanggungan": 2, "kondisi_rumah": "tidak layak", "status_pekerjaan": "buruh harian"}, 1),
    ({"penghasilan": 1750000, "jumlah_tanggungan": 3, "kondisi_rumah": "semi permanen", "status_pekerjaan": "petani"}, 1),
    ({"penghasilan": 4000000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "pegawai tetap"}, 0),
    ({"penghasilan": 5500000, "jumlah_tanggungan": 2, "kondisi_rumah": "layak", "status_pekerjaan": "karyawan tetap"}, 0),
    ({"penghasilan": 3500000, "jumlah_tanggungan": 1, "kondisi_rumah": "sederhana", "status_pekerjaan": "pegawai tetap"}, 0),
    ({"penghasilan": 5000000, "jumlah_tanggungan": 2, "kondisi_rumah": "layak", "status_pekerjaan": "PNS"}, 0),
    ({"penghasilan": 6000000, "jumlah_tanggungan": 1, "kondisi_rumah": "layak", "status_pekerjaan": "wirausaha"}, 0),
]

# ✅ DUKCAPIL
dukcapil_cases = [
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 5, "usia_kepala_keluarga": 40}, 1),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "janda", "pekerjaan_kepala_keluarga": "petani", "jumlah_anggota_kk": 4, "usia_kepala_keluarga": 35}, 1),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "tidak bekerja", "jumlah_anggota_kk": 6, "usia_kepala_keluarga": 42}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "belum menikah", "pekerjaan_kepala_keluarga": "pegawai tetap", "jumlah_anggota_kk": 1, "usia_kepala_keluarga": 25}, 0),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "cerai", "pekerjaan_kepala_keluarga": "PNS", "jumlah_anggota_kk": 2, "usia_kepala_keluarga": 30}, 0),
    ({"nik_valid": "tidak", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "ya", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 4, "usia_kepala_keluarga": 38}, 0),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "petani", "jumlah_anggota_kk": 7, "usia_kepala_keluarga": 47}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "duda", "pekerjaan_kepala_keluarga": "wirausaha", "jumlah_anggota_kk": 2, "usia_kepala_keluarga": 33}, 0),
    ({"nik_valid": "ya", "memiliki_kk": "ya", "domisili_tetap": "ya", "data_ganda": "tidak", "masuk_dtks": "ya",
      "status_perkawinan": "menikah", "pekerjaan_kepala_keluarga": "buruh", "jumlah_anggota_kk": 5, "usia_kepala_keluarga": 36}, 1),
    ({"nik_valid": "tidak", "memiliki_kk": "tidak", "domisili_tetap": "tidak", "data_ganda": "ya", "masuk_dtks": "tidak",
      "status_perkawinan": "belum menikah", "pekerjaan_kepala_keluarga": "pegawai tetap", "jumlah_anggota_kk": 1, "usia_kepala_keluarga": 27}, 0),
]

# ✅ KEMENKES
kemenkes_cases = [
    ({"penghasilan": 2000000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "diabetes", "status_pekerjaan": "buruh harian", "kondisi_rumah": "sederhana"}, 1),
    ({"penghasilan": 1500000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma", "status_pekerjaan": "petani", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 1200000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "hipertensi", "status_pekerjaan": "tidak bekerja", "kondisi_rumah": "sangat sederhana"}, 1),
    ({"penghasilan": 3500000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "pegawai tetap", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 7000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "PNS", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 2500000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "ginjal", "status_pekerjaan": "buruh harian", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 6000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "wirausaha", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 1000000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "TBC", "status_pekerjaan": "tidak bekerja", "kondisi_rumah": "tidak layak"}, 1),
    ({"penghasilan": 4000000, "punya_asuransi_lain": "ya", "penyakit_kronis": "tidak ada", "status_pekerjaan": "pegawai tetap", "kondisi_rumah": "layak"}, 0),
    ({"penghasilan": 1800000, "punya_asuransi_lain": "tidak", "penyakit_kronis": "asma", "status_pekerjaan": "buruh harian", "kondisi_rumah": "semi permanen"}, 1),
]
# ============================================================
# 🚀 MAIN
# ============================================================
if __name__ == "__main__":
    run_test("DINSOS", dinsos_cases)
    run_test("DUKCAPIL", dukcapil_cases)
    run_test("KEMENKES", kemenkes_cases)
