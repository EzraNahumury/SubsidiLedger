# ============================================================
# IMPORT
# ============================================================
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import joblib
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# KONFIGURASI
# ============================================================
INSTANSI   = "dukcapil"  # ganti sesuai instansi
BATCH_SIZE = 32
N_CLIENTS  = 10
ROUNDS     = 15

BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "DATASET/dukcapil_balanced.csv"
FEATURE_PATH = BASE_DIR / "Models/fitur_global.pkl"
SAVE_DIR   = BASE_DIR / f"Models/saved_{INSTANSI}_tff"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n=== TRAINING FEDERATED {INSTANSI.upper()} ===")

# ============================================================
# LOAD DATASET
# ============================================================
print(f"📂 Load dataset: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"✅ {len(df):,} baris | {len(df.columns)} kolom")

label_col = "layak_subsidi"
if label_col not in df.columns:
    raise ValueError("Kolom 'layak_subsidi' tidak ditemukan!")

y_all = df[label_col].astype("float32").values
X_raw = df.drop(columns=[label_col])

# drop kolom ID / timestamp
drop_cols = [c for c in X_raw.columns if "id" in c.lower() or "timestamp" in c.lower()]
if drop_cols:
    X_raw = X_raw.drop(columns=drop_cols)

# ============================================================
# LOAD FITUR GLOBAL
# ============================================================
if not FEATURE_PATH.exists():
    raise FileNotFoundError(
        "Models/fitur_global.pkl tidak ditemukan.\n"
        "Jalankan dulu pembuat fitur global."
    )

FEATURE_LIST = joblib.load(FEATURE_PATH)
FEATURE_DIM  = len(FEATURE_LIST)
print(f"🔑 Total fitur global: {FEATURE_DIM}")

# ============================================================
# ONE-HOT + ALIGN
# ============================================================
print("🔧 One-hot encoding & align fitur...")
X_oh = pd.get_dummies(X_raw, drop_first=False).astype(float)

for col in FEATURE_LIST:
    if col not in X_oh.columns:
        X_oh[col] = 0.0

X_oh = X_oh[FEATURE_LIST]  # URUTAN WAJIB IDENTIK

# ============================================================
# NORMALISASI MIN-MAX
# ============================================================
print("🔧 Normalisasi Min-Max...")
mins = X_oh.min()
rng  = (X_oh.max() - mins).replace(0, 1.0)
X_scaled = ((X_oh - mins) / rng).fillna(0.0).astype("float32")

print("✅ Preprocessing selesai")

# ============================================================
# TF.DATASET & SPLIT CLIENT
# ============================================================
def to_tf_dataset(X, y):
    return tf.data.Dataset.from_tensor_slices(
        (X.values.astype("float32"),
         y.reshape(-1, 1).astype("float32"))
    ).shuffle(len(X)).batch(BATCH_SIZE)

def split_clients(X, y, n_clients=N_CLIENTS):
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    size = len(X) // n_clients
    clients = []
    for i in range(n_clients):
        s, e = i * size, (i + 1) * size if i < n_clients - 1 else len(X)
        clients.append(to_tf_dataset(X.iloc[idx[s:e]], y[idx[s:e]]))
    return clients

clients = split_clients(X_scaled, y_all, N_CLIENTS)
print(f"👥 {len(clients)} klien federated siap")

# ============================================================
# MODEL FN (IDENTIK DENGAN TEST)
# ============================================================
def model_fn():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(FEATURE_DIM,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    return tff.learning.models.from_keras_model(
        keras_model=model,
        input_spec=clients[0].element_spec,
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")]
    )

# ============================================================
# FEDERATED PROCESS
# ============================================================
process = tff.learning.algorithms.build_unweighted_fed_avg(
    model_fn,
    client_optimizer_fn=tff.learning.optimizers.build_adam(0.005),
    server_optimizer_fn=tff.learning.optimizers.build_adam(0.01),
)

state = process.initialize()

# ============================================================
# TRAINING + HISTORY
# ============================================================
history = []

print("\n🚀 TRAINING START")
for r in range(1, ROUNDS + 1):
    state, metrics = process.next(state, clients)

    m    = metrics["client_work"]["train"]
    acc  = float(m["binary_accuracy"])
    loss = float(m["loss"])

    history.append((r, acc, loss))
    print(f"[{INSTANSI.upper()}] Round {r:02d} | acc={acc:.4f} | loss={loss:.4f}")

# ============================================================
# SIMPAN MODEL KERAS
# ============================================================
keras_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(FEATURE_DIM,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

process.get_model_weights(state).assign_weights_to(keras_model)
keras_model.save(SAVE_DIR, include_optimizer=False)

# ============================================================
# SIMPAN MODEL .NPZ
# ============================================================
timestamp = time.strftime("%Y%m%d_%H%M%S")
np.savez_compressed(
    SAVE_DIR / f"{INSTANSI}_{timestamp}.npz",
    *keras_model.get_weights()
)

# ============================================================
# SIMPAN PREPROCESS
# ============================================================
joblib.dump(
    {
        "FEATURE_COLS": FEATURE_LIST,
        "mins": mins,
        "rng": rng,
    },
    SAVE_DIR / "preprocess.pkl"
)

# ============================================================
# SIMPAN HISTORY LOG
# ============================================================
history_path = SAVE_DIR / "accuracy_history.txt"
if not history_path.exists():
    history_path.write_text("round\taccuracy\tloss\ttimestamp\n")

with open(history_path, "a", encoding="utf-8") as f:
    for r, acc, loss in history:
        f.write(
            f"{r}\t{acc:.6f}\t{loss:.6f}\t"
            f"{datetime.utcnow().isoformat()}Z\n"
        )

# ============================================================
# BEST ACCURACY
# ============================================================
best_acc = max(h[1] for h in history)
(SAVE_DIR / "best_accuracy.txt").write_text(f"{best_acc:.6f}\n")

# ============================================================
# SELESAI
# ============================================================
print("\n✅ TRAINING SELESAI")
print(f"📂 Model        : {SAVE_DIR}")
print(f"🏆 Best Acc     : {best_acc:.4f}")
print(f"📈 History file : accuracy_history.txt")
print(f"💾 Weights NPZ  : {INSTANSI}_{timestamp}.npz")
