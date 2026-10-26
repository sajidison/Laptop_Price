"""Generate a large synthetic laptop dataset with realistic price structure.

Usage: python generate_laptop_data.py [n_rows]   (default 100000)
"""
import sys
import numpy as np
import pandas as pd

N = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
rng = np.random.default_rng(42)

# ---- Brands: (name, brand price multiplier, sampling weight) ----
brands = {
    "Apple": (1.35, 0.08), "Dell": (1.05, 0.15), "HP": (1.0, 0.16),
    "Lenovo": (1.0, 0.16), "Asus": (1.0, 0.13), "Acer": (0.88, 0.10),
    "MSI": (1.10, 0.06), "Razer": (1.40, 0.02), "Microsoft": (1.15, 0.03),
    "Samsung": (1.05, 0.03), "LG": (1.05, 0.02), "Huawei": (0.95, 0.02),
    "Xiaomi": (0.85, 0.02), "Gigabyte": (1.0, 0.02),
}
b_names = list(brands)
b_probs = np.array([v[1] for v in brands.values()]); b_probs /= b_probs.sum()

# ---- CPUs: (name, score, weight) ----
cpus = {
    "Intel Celeron": (1, 0.05), "Intel Pentium": (1.5, 0.03),
    "Intel Core i3": (3, 0.10), "Intel Core i5": (5, 0.25),
    "Intel Core i7": (7, 0.17), "Intel Core i9": (9.5, 0.05),
    "Intel Core Ultra 5": (6, 0.04), "Intel Core Ultra 7": (8, 0.03),
    "AMD Ryzen 3": (3, 0.04), "AMD Ryzen 5": (5.5, 0.10),
    "AMD Ryzen 7": (7.5, 0.07), "AMD Ryzen 9": (9.5, 0.02),
    "Apple M1": (6.5, 0.015), "Apple M2": (7.5, 0.015),
    "Apple M3": (8.5, 0.015), "Apple M3 Pro": (9.5, 0.01),
    "Apple M4": (9, 0.01),
}
c_names = list(cpus)
c_probs = np.array([v[1] for v in cpus.values()]); c_probs /= c_probs.sum()

gpus = {
    "Integrated": 0, "NVIDIA GTX 1650": 3, "NVIDIA RTX 3050": 4,
    "NVIDIA RTX 3060": 5.5, "NVIDIA RTX 4050": 5, "NVIDIA RTX 4060": 6.5,
    "NVIDIA RTX 4070": 8, "NVIDIA RTX 4080": 10, "NVIDIA RTX 4090": 13,
    "NVIDIA RTX 5070": 9, "AMD Radeon RX 6600M": 4.5,
}
g_names = list(gpus)

ram_opts = np.array([4, 8, 16, 32, 64])
ram_probs = np.array([0.04, 0.30, 0.42, 0.20, 0.04])
storage_opts = np.array([128, 256, 512, 1024, 2048, 4096])
storage_probs = np.array([0.03, 0.18, 0.40, 0.28, 0.09, 0.02])
screen_opts = np.array([11.6, 13.3, 14.0, 15.6, 16.0, 17.3])
screen_probs = np.array([0.04, 0.16, 0.25, 0.35, 0.12, 0.08])
res_opts = ["1366x768", "1920x1080", "1920x1200", "2560x1440", "2560x1600", "2880x1800", "3840x2160"]
res_mult = {"1366x768": 0.92, "1920x1080": 1.0, "1920x1200": 1.04, "2560x1440": 1.10,
            "2560x1600": 1.14, "2880x1800": 1.18, "3840x2160": 1.30}
res_probs = np.array([0.06, 0.45, 0.10, 0.12, 0.10, 0.07, 0.10])
panel_opts = ["TN", "IPS", "VA", "OLED", "Mini-LED"]
panel_mult = {"TN": 0.95, "IPS": 1.0, "VA": 1.0, "OLED": 1.15, "Mini-LED": 1.20}
panel_probs = np.array([0.05, 0.70, 0.08, 0.14, 0.03])
os_opts = ["Windows 11 Home", "Windows 11 Pro", "macOS", "Linux", "ChromeOS", "No OS"]

n = N
brand = rng.choice(b_names, n, p=b_probs)
cpu = rng.choice(c_names, n, p=c_probs)

# Apple brand <-> Apple chips; non-Apple can't have Apple chips
is_apple_brand = brand == "Apple"
is_apple_cpu = np.char.startswith(cpu.astype(str), "Apple")
apple_chips = [c for c in c_names if c.startswith("Apple")]
other_chips = [c for c in c_names if not c.startswith("Apple")]
o_p = np.array([cpus[c][1] for c in other_chips]); o_p /= o_p.sum()
a_p = np.array([cpus[c][1] for c in apple_chips]); a_p /= a_p.sum()
cpu = np.where(is_apple_brand, rng.choice(apple_chips, n, p=a_p), cpu)
cpu = np.where(~is_apple_brand & is_apple_cpu, rng.choice(other_chips, n, p=o_p), cpu)
cpu_score = np.array([cpus[c][0] for c in cpu])
is_apple = np.char.startswith(cpu.astype(str), "Apple")

# GPU: high-end CPUs / gaming brands more likely to have dedicated GPU
gaming_prob = np.clip(0.05 + 0.04 * (cpu_score - 3), 0.02, 0.6)
gaming_prob = np.where(np.isin(brand, ["MSI", "Razer", "Gigabyte"]), 0.92, gaming_prob)
gaming_prob = np.where(is_apple, 0.0, gaming_prob)
has_dgpu = rng.random(n) < gaming_prob
gpu_pool = np.array(g_names[1:])
# better CPU -> better GPU
gpu_idx = np.clip(((cpu_score - 3) / 7 * (len(gpu_pool) - 1) + rng.normal(0, 1.8, n)).round(),
                  0, len(gpu_pool) - 1).astype(int)
gpu = np.where(has_dgpu, gpu_pool[gpu_idx], "Integrated")
gpu_score = np.array([gpus[g] for g in gpu])

# RAM / storage scale with tier
tier = cpu_score + gpu_score * 0.5
ram_p_matrix = np.tile(ram_probs, (n, 1))
shift = np.clip((tier - 6) / 10, -0.5, 0.8)[:, None] * np.array([-0.6, -0.4, 0.0, 0.5, 0.5])
ram_p_matrix = np.clip(ram_p_matrix + shift * 0.3, 0.005, None)
ram_p_matrix /= ram_p_matrix.sum(1, keepdims=True)
cum = ram_p_matrix.cumsum(1)
ram = ram_opts[(rng.random(n)[:, None] > cum).sum(1).clip(0, 4)]

st_p = np.tile(storage_probs, (n, 1))
shift = np.clip((tier - 6) / 10, -0.5, 0.8)[:, None] * np.array([-0.5, -0.4, -0.1, 0.3, 0.4, 0.3])
st_p = np.clip(st_p + shift * 0.3, 0.005, None); st_p /= st_p.sum(1, keepdims=True)
cum = st_p.cumsum(1)
storage = storage_opts[(rng.random(n)[:, None] > cum).sum(1).clip(0, 5)]

storage_type = np.where(rng.random(n) < 0.94, "SSD", "HDD+SSD")
storage_type = np.where(storage <= 256, np.where(rng.random(n) < 0.08, "eMMC", storage_type), storage_type)

screen = rng.choice(screen_opts, n, p=screen_probs)
res = rng.choice(res_opts, n, p=res_probs)
panel = rng.choice(panel_opts, n, p=panel_probs)
# sanity: high-end screens only on decent machines
low_end = tier < 4
res = np.where(low_end & np.isin(res, ["3840x2160", "2880x1800", "2560x1600", "2560x1440"]), "1920x1080", res)
res = np.where(low_end & (rng.random(n) < 0.25), "1366x768", res)
panel = np.where(low_end & np.isin(panel, ["OLED", "Mini-LED"]), "IPS", panel)
refresh = np.where(has_dgpu, rng.choice([60, 120, 144, 165, 240], n, p=[0.1, 0.2, 0.3, 0.25, 0.15]), 60)
refresh = np.where(~has_dgpu & (rng.random(n) < 0.1), 90, refresh)
touch = rng.random(n) < np.where(np.isin(brand, ["Microsoft", "HP", "Lenovo", "Dell", "Samsung"]), 0.25, 0.06)
touch = np.where(is_apple, False, touch)

os_ = np.where(is_apple, "macOS",
      rng.choice(["Windows 11 Home", "Windows 11 Pro", "Linux", "ChromeOS", "No OS"], n,
                 p=[0.62, 0.20, 0.05, 0.08, 0.05]))
os_ = np.where((os_ == "ChromeOS") & (tier > 5), "Windows 11 Home", os_)

# Weight and battery
weight = 0.9 + 0.055 * screen + 0.08 * gpu_score + rng.normal(0, 0.12, n)
weight = np.where(is_apple, 1.0 + 0.04 * (screen - 13) + rng.normal(0, 0.05, n), weight)
weight = np.clip(weight, 0.85, 3.5).round(2)
battery = np.clip(45 + 3.5 * screen - 1.5 * gpu_score + rng.normal(0, 10, n), 30, 100)
battery = np.where(is_apple, 52 + 3 * (screen - 13) + rng.normal(0, 6, n) + 8, battery).round(0)

year = rng.choice([2020, 2021, 2022, 2023, 2024, 2025, 2026], n, p=[0.05, 0.08, 0.15, 0.22, 0.25, 0.18, 0.07])
condition = rng.choice(["New", "Refurbished", "Used"], n, p=[0.78, 0.12, 0.10])
has_backlit = rng.random(n) < np.where(tier > 4, 0.9, 0.5)
has_fingerprint = rng.random(n) < np.where(tier > 5, 0.6, 0.2)
wifi = np.where(year >= 2023, rng.choice(["WiFi 6", "WiFi 6E", "WiFi 7"], n, p=[0.5, 0.35, 0.15]),
                rng.choice(["WiFi 5", "WiFi 6"], n, p=[0.5, 0.5]))
ports = rng.integers(2, 8, n)
warranty = rng.choice([1, 2, 3], n, p=[0.7, 0.2, 0.1])
build = rng.choice(["Plastic", "Aluminum", "Magnesium", "Carbon fiber"], n, p=[0.45, 0.42, 0.10, 0.03])
build = np.where(is_apple, "Aluminum", build)

# ---------------- Price model (USD) ----------------
price = 180.0
price += 62 * cpu_score ** 1.15
price += 70 * gpu_score ** 1.20
price += np.where(ram <= 8, ram * 7, 56 + (ram - 8) * 11)
price += np.where(storage <= 512, storage * 0.14, 72 + (storage - 512) * 0.10)
price += np.where(storage_type == "HDD+SSD", -25, 0) + np.where(storage_type == "eMMC", -35, 0)
price += (screen - 13) * 22
price *= np.array([res_mult[r] for r in res])
price *= np.array([panel_mult[p] for p in panel])
price += np.where(refresh >= 144, 55, np.where(refresh >= 90, 25, 0))
price += np.where(touch, 70, 0) + np.where(has_fingerprint, 20, 0) + np.where(has_backlit, 12, 0)
price += (battery - 50) * 1.8
price -= (weight - 1.6) * 40  # lighter = pricier (premium materials)
price += np.select([build == "Aluminum", build == "Magnesium", build == "Carbon fiber"], [45, 90, 150], 0)
price += np.select([os_ == "Windows 11 Pro", os_ == "No OS"], [45, -80], 0)
price += np.select([os_ == "ChromeOS"], [-60], 0)
price += (warranty - 1) * 35 + (ports - 4) * 6
price += np.select([wifi == "WiFi 7", wifi == "WiFi 6E"], [40, 18], 0)
price *= np.array([brands[b][0] for b in brand])

# Depreciation by age & condition
age = 2026 - year
price *= np.clip(1 - 0.10 * age, 0.45, 1.0)
price *= np.select([condition == "Refurbished", condition == "Used"], [0.80, 0.62], 1.0)

# Market noise (multiplicative, heteroscedastic)
price *= rng.lognormal(0, 0.06, n)
price = np.clip(price, 150, None).round(2)

df = pd.DataFrame({
    "brand": brand, "cpu": cpu, "gpu": gpu, "ram_gb": ram, "storage_gb": storage,
    "storage_type": storage_type, "screen_inch": screen, "resolution": res,
    "panel_type": panel, "refresh_hz": refresh, "touchscreen": touch.astype(int),
    "os": os_, "weight_kg": weight, "battery_wh": battery.astype(int), "release_year": year,
    "condition": condition, "backlit_keyboard": has_backlit.astype(int),
    "fingerprint_reader": has_fingerprint.astype(int), "wifi": wifi, "usb_ports": ports,
    "warranty_years": warranty, "build_material": build, "price_usd": price,
})

# Sprinkle realistic missing values (~1%) in a few columns
for col in ["weight_kg", "battery_wh", "panel_type"]:
    mask = rng.random(n) < 0.01
    df.loc[mask, col] = np.nan

df.to_csv("/mnt/user-data/outputs/laptop_dataset.csv", index=False)
print(f"Saved {len(df):,} rows x {df.shape[1]} cols")
print(df.price_usd.describe().round(2))
