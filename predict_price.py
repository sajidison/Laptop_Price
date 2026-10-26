"""Predict a laptop's price with the trained model."""
import joblib, pandas as pd

model = joblib.load("laptop_price_model.joblib")

laptop = pd.DataFrame([{
    "brand": "Lenovo", "cpu": "Intel Core i7", "gpu": "NVIDIA RTX 4060",
    "ram_gb": 16, "storage_gb": 1024, "storage_type": "SSD", "screen_inch": 15.6,
    "resolution": "2560x1440", "panel_type": "IPS", "refresh_hz": 165, "touchscreen": 0,
    "os": "Windows 11 Home", "weight_kg": 2.3, "battery_wh": 80, "release_year": 2025,
    "condition": "New", "backlit_keyboard": 1, "fingerprint_reader": 0, "wifi": "WiFi 6E",
    "usb_ports": 4, "warranty_years": 2, "build_material": "Aluminum",
}])
print(f"Estimated price: ${model.predict(laptop)[0]:,.2f}")
