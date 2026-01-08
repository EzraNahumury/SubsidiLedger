import requests
import numpy as np
from pathlib import Path

SERVER_URL = "https://federatedinstitusi.up.railway.app"
DOWNLOAD_DIR = Path("models/global")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

url = f"{SERVER_URL}/download-global"

print("🌍 Mengunduh model global terbaru...")
print(url)

try:
    response = requests.get(url, timeout=120)
except requests.exceptions.RequestException as e:
    print(f"❌ Gagal koneksi: {e}")
    raise SystemExit(1)

if response.status_code != 200:
    print(f"❌ Gagal download ({response.status_code})")
    try:
        print("📨 Pesan server:", response.json())
    except Exception:
        print(response.text)
    raise SystemExit(1)

filename = response.headers.get("X-File-Name", "global_model_fedavg_latest.npz")
save_path = DOWNLOAD_DIR / filename

with open(save_path, "wb") as f:
    f.write(response.content)

# --- Validasi dasar ---
if save_path.stat().st_size < 1024:
    print("❌ File terlalu kecil, kemungkinan invalid.")
    raise SystemExit(1)

# --- Validasi NPZ ---
try:
    data = np.load(save_path)
    print(f"🔎 NPZ valid: {len(data.files)} tensor")
except Exception as e:
    print(f"❌ File NPZ rusak: {e}")
    raise SystemExit(1)

size_mb = save_path.stat().st_size / 1024 / 1024

print("\n✅ BERHASIL")
print(f"📁 File disimpan : {save_path}")
print(f"📦 Ukuran        : {size_mb:.2f} MB")

print("\nℹ️ Metadata:")
print(" - X-File-Name     :", response.headers.get("X-File-Name"))
print(" - X-File-Size     :", response.headers.get("X-File-Size"))
print(" - X-Last-Modified :", response.headers.get("X-Last-Modified"))
print(" - X-Description   :", response.headers.get("X-Description"))
