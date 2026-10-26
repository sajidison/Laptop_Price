import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==========================================
# 2. MODEL LOADING (WITH CACHING)
# ==========================================
@st.cache_resource
def load_model(model_path: str = "model/laptop_price_model.joblib"):
    """Loads and caches the trained ML pipeline/model."""
    try:
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        st.error(
            f"❌ Model file `{model_path}` not found! Please check if the path is correct."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()


model = load_model("model/laptop_price_model.joblib")

# ==========================================
# 3. UI LAYOUT & USER INPUTS
# ==========================================
st.title("💻 Laptop Price Estimation Tool")
st.markdown(
    "Configure the laptop specifications below to predict the estimated price."
)
st.markdown("---")

# Organized using tabs/columns for better UX
tab1, tab2, tab3 = st.tabs(
    ["⚙️ Core Hardware", "🖥️ Display & Build", "🔌 Features & Extras"]
)

with tab1:
    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox(
            "Brand",
            [
                "Apple",
                "Asus",
                "Dell",
                "HP",
                "Lenovo",
                "MSI",
                "Acer",
                "Razer",
                "Samsung",
                "Other",
            ],
        )
        ram_gb = st.select_slider(
            "RAM (GB)", options=[4, 8, 12, 16, 24, 32, 64, 128], value=16
        )

    with col2:
        cpu = st.selectbox(
            "CPU Brand / Series",
            [
                "Intel Core i3",
                "Intel Core i5",
                "Intel Core i7",
                "Intel Core i9",
                "AMD Ryzen 3",
                "AMD Ryzen 5",
                "AMD Ryzen 7",
                "AMD Ryzen 9",
                "Apple M1/M2/M3",
                "Other",
            ],
        )
        storage_gb = st.select_slider(
            "Storage Capacity (GB)",
            options=[128, 256, 512, 1024, 2048, 4096],
            value=512,
        )

    with col3:
        gpu = st.selectbox(
            "GPU Brand / Type",
            [
                "Integrated Intel",
                "Integrated AMD",
                "NVIDIA GTX",
                "NVIDIA RTX 30-Series",
                "NVIDIA RTX 40-Series",
                "AMD Radeon Dedicated",
                "Apple Integrated",
            ],
        )
        storage_type = st.selectbox(
            "Storage Type", ["SSD", "HDD", "NVMe SSD", "Hybrid", "eMMC"]
        )

with tab2:
    col1, col2, col3 = st.columns(3)

    with col1:
        screen_inch = st.number_input(
            "Screen Size (Inches)",
            min_value=10.0,
            max_value=18.0,
            value=15.6,
            step=0.1,
        )
        resolution = st.selectbox(
            "Resolution",
            [
                "1366x768",
                "1920x1080 (FHD)",
                "2560x1440 (QHD)",
                "3840x2160 (4K)",
                "2880x1800",
            ],
        )

    with col2:
        panel_type = st.selectbox(
            "Panel Type", ["IPS", "OLED", "TN", "VA", "Mini-LED"]
        )
        refresh_hz = st.selectbox(
            "Refresh Rate (Hz)", [60, 90, 120, 144, 165, 240, 360], index=0
        )

    with col3:
        weight_kg = st.number_input(
            "Weight (kg)",
            min_value=0.5,
            max_value=5.0,
            value=1.8,
            step=0.1,
            format="%.2f",
        )
        build_material = st.selectbox(
            "Build Material",
            ["Aluminum", "Plastic", "Magnesium Alloy", "Carbon Fiber", "Metal"],
        )

with tab3:
    col1, col2, col3 = st.columns(3)

    with col1:
        os = st.selectbox(
            "Operating System",
            ["Windows 11", "Windows 10", "macOS", "Linux", "ChromeOS", "No OS"],
        )
        battery_wh = st.number_input(
            "Battery (Wh)", min_value=20, max_value=100, value=56
        )
        release_year = st.number_input(
            "Release Year",
            min_value=2018,
            max_value=2026,
            value=2024,
            step=1,
        )

    with col2:
        condition = st.selectbox("Condition", ["New", "Refurbished", "Used"])
        warranty_years = st.number_input(
            "Warranty (Years)", min_value=0, max_value=5, value=1
        )
        usb_ports = st.number_input(
            "USB Ports Count", min_value=1, max_value=8, value=3
        )

    with col3:
        touchscreen = st.checkbox("Touchscreen", value=False)
        backlit_keyboard = st.checkbox("Backlit Keyboard", value=True)
        fingerprint_reader = st.checkbox("Fingerprint Reader", value=False)
        wifi = st.selectbox("Wi-Fi Standard", ["Wi-Fi 5", "Wi-Fi 6 / 6E", "Wi-Fi 7"])

# ==========================================
# 4. PREDICTION LOGIC
# ==========================================
st.markdown("---")

if st.button("🚀 Predict Laptop Price", use_container_width=True, type="primary"):
    input_data = pd.DataFrame(
        [
            {
                "brand": brand,
                "cpu": cpu,
                "gpu": gpu,
                "ram_gb": ram_gb,
                "storage_gb": storage_gb,
                "storage_type": storage_type,
                "screen_inch": screen_inch,
                "resolution": resolution,
                "panel_type": panel_type,
                "refresh_hz": refresh_hz,
                "touchscreen": int(touchscreen),
                "os": os,
                "weight_kg": weight_kg,
                "battery_wh": battery_wh,
                "release_year": release_year,
                "condition": condition,
                "backlit_keyboard": int(backlit_keyboard),
                "fingerprint_reader": int(fingerprint_reader),
                "wifi": wifi,
                "usb_ports": usb_ports,
                "warranty_years": warranty_years,
                "build_material": build_material,
            }
        ]
    )

    try:
        prediction = model.predict(input_data)[0]

        # Uncomment if your model was trained on log-transformed prices:
        # prediction = np.expm1(prediction)

        st.balloons()
        st.success(f"### Estimated Price: **${prediction:,.2f} USD**")

    except Exception as e:
        st.error(f"Prediction failed. Error: {e}")
        st.info(
            "💡 **Tip:** Ensure your input feature categories match the exact values used during your model training."
        )