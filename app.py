import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.write("✅ App started")

# ----------------------------
# Load model and dataframe
# ----------------------------
try:
    pipe = joblib.load("pipe.pkl")
    df = joblib.load("df.pkl")
except Exception as e:
    st.error(f"Error loading artifacts: {e}")
    st.stop()

# ----------------------------
# User interface
# ----------------------------
if df is not None:
    st.title("💻 Laptop Price Predictor (PKR)")

    # Company
    company = st.selectbox("Company", sorted(df["Company"].unique()))

    # Type of laptop
    laptop_type = st.selectbox("Type", sorted(df["TypeName"].unique()))

    # RAM (match training dtype; your X_train shows float64, but values are discrete GBs)
    ram = st.selectbox("RAM (GB)", sorted(df["Ram"].unique()))

    # Operating System
    opsys = st.selectbox("Operating System", sorted(df["OpSys"].unique()))

    # Weight
    weight = st.number_input(
        "Weight (kg)", min_value=0.5, max_value=5.0, value=1.5, step=0.1
    )

    # Touchscreen / IPS as Yes/No -> 0/1 (X_train uses int64)
    touchscreen_opt = st.selectbox("Touchscreen", ["No", "Yes"])
    ips_opt = st.selectbox("IPS Panel", ["No", "Yes"])

    # Screen Size
    screen_size = st.number_input(
        "Screen Size (inches)", min_value=10.0, max_value=20.0, value=15.6, step=0.1
    )

    # Resolution (used only to compute ppi; Resolution itself is NOT sent to the model)
    resolution = st.selectbox(
        "Resolution",
        [
            "1920x1080", "1366x768", "3840x2160", "2560x1440", "1600x900",
            "1280x800", "2560x1600", "3840x2400", "1920x1200", "1440x900", "1280x720",
        ],
    )
    res_x, res_y = map(int, resolution.split("x"))
    ppi = ((res_x**2 + res_y**2) ** 0.5) / screen_size

    # Auto-derive 4K flag from resolution (X_train has '4k' int column)
    four_k_flag = 1 if max(res_x, res_y) >= 3840 or min(res_x, res_y) >= 2160 else 0
    st.caption(f"Auto-detected 4K: {'Yes' if four_k_flag == 1 else 'No'}")

    # CPU Brand / Type
    cpu_brand = st.selectbox("CPU Brand", sorted(df["Cpu_Brand"].unique()))
    cpu_type = st.selectbox("CPU Type", sorted(df["Cpu_Type"].unique()))

    # CPU Speed
    cpu_speed = st.number_input(
        "CPU Speed (GHz)", min_value=0.5, max_value=6.0, value=2.0, step=0.1
    )

    # SSD / HDD (GB)
    ssd = st.number_input("SSD (GB)", min_value=0, max_value=4096, value=256, step=128)
    hdd = st.number_input("HDD (GB)", min_value=0, max_value=4096, value=0, step=128)

    # GPU Brand
    gpu_brand = st.selectbox("GPU Brand", sorted(df["Gpu_Brand"].unique()))

    # ----------------------------
    # Prediction Button
    # ----------------------------
    if st.button("Predict Price"):
        # Build input to match X_train exactly (15 columns)
        user_selections = {
            "Company": company,
            "TypeName": laptop_type,
            "Ram": float(ram),  # keep numeric; matches training dtype
            "OpSys": opsys,
            "Weight": float(weight),
            "Touchscreen": 1 if touchscreen_opt == "Yes" else 0,
            "IPS": 1 if ips_opt == "Yes" else 0,
            "4k": int(four_k_flag),
            "ppi": float(ppi),
            "Cpu_Brand": cpu_brand,
            "Cpu_Type": cpu_type,
            "Cpu_Speed": float(cpu_speed),
            "SSD_GB": int(ssd),
            "HDD_GB": int(hdd),
            "Gpu_Brand": gpu_brand,
        }

        input_df = pd.DataFrame([user_selections])

        # Optional debug helpers
        with st.expander("🔍 Debug: Input DataFrame & Expected Columns"):
            st.dataframe(input_df)
            try:
                st.write("Model expects columns:", list(pipe.feature_names_in_))
            except Exception:
                st.write("`pipe.feature_names_in_` not available on this pipeline.")

        try:
            # Model outputs log(price); convert back to INR
            pred_log = pipe.predict(input_df)[0]
            pred_inr = float(np.exp(pred_log)) * 3

            st.success(f"💰 Predicted Price: ₹{pred_inr:,.0f}")
        except Exception as e:
            st.error(f"Prediction error: {e}")
else:
    st.error("`df.pkl` failed to load.")
    st.write("Please check the input values and try again.")