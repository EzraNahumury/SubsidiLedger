import pandas as pd, joblib

print("📂 Membaca dataset dari ketiga instansi...")
dfs = [
    pd.read_csv("DATASET/dinsos_balanced.csv"),
    pd.read_csv("DATASET/dukcapil_balanced.csv"),
    pd.read_csv("DATASET/kemenkes_balanced.csv"),
]

# Gabungkan semua kolom (tanpa kolom target)
print("🔧 Menggabungkan semua fitur unik dari ketiga instansi...")
all_features = pd.concat([df.drop(columns=["layak_subsidi"]) for df in dfs], axis=0)

# One-hot encode seluruh kolom gabungan
global_encoded = pd.get_dummies(all_features, drop_first=False)
global_cols = list(global_encoded.columns)

# Simpan ke file (berisi LIST, bukan dictionary)
joblib.dump(global_cols, "Models/fitur_global.pkl")

print("✅ Fitur global (LIST) tersimpan di 'Models/fitur_global.pkl'")
print(f"📏 Total fitur unik: {len(global_cols)} kolom")
