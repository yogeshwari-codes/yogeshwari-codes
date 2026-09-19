import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================== ROM CORE LOGIC ======================
INPUT_BITS = 8
ROM_SIZE = 2 ** INPUT_BITS
INPUT_MIN = -8.0
INPUT_MAX = 8.0
OUTPUT_SCALE = 255

def mathematical_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@st.cache_data
def build_rom_table():
    addresses = np.arange(ROM_SIZE)
    real_inputs = INPUT_MIN + (addresses / (ROM_SIZE - 1)) * (INPUT_MAX - INPUT_MIN)
    exact = mathematical_sigmoid(real_inputs)
    return np.round(exact * OUTPUT_SCALE).astype(np.uint8)

ROM_TABLE = build_rom_table()

def rom_sigmoid(x):
    x_clamped = np.clip(x, INPUT_MIN, INPUT_MAX)
    addr = int(round((x_clamped - INPUT_MIN) / (INPUT_MAX - INPUT_MIN) * (ROM_SIZE - 1)))
    return ROM_TABLE[addr] / OUTPUT_SCALE

# ====================== WEBSITE UI ======================
st.set_page_config(
    page_title="ROM Sigmoid Lookup Table",
    page_icon="🧠",
    layout="wide"
)

st.title("8-bit ROM Lookup Table for Approximate Sigmoid Activation")
st.markdown("**EC2201 Mini Project** | Digital Systems Concept Demonstration")

st.markdown("---")

# Sidebar
st.sidebar.header("Project Info")
st.sidebar.write("""
**Topic:** ROM-Based Lookup Table for Feature Transformation  
**Concept:** Pure ROM Lookup (No runtime calculation)  
**Application:** TinyML / Edge AI Activation Function
""")

# Main Interactive Section
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Interactive ROM Lookup")
    user_input = st.slider("Enter Input Value (x)", -10.0, 10.0, 0.0, 0.1)
    
    exact = mathematical_sigmoid(np.clip(user_input, -8, 8))
    approx = rom_sigmoid(user_input)
    error = abs(exact - approx)

    st.metric("Exact Sigmoid", f"{exact:.6f}")
    st.metric("ROM Approximation", f"{approx:.6f}")
    st.metric("Absolute Error", f"{error:.6f}")

with col2:
    st.subheader("How it Works")
    st.markdown("""
    1. Input is mapped to an **8-bit address** (0–255)
    2. The address is used to **lookup** a pre-computed value from ROM
    3. No mathematical calculation happens at runtime
    4. Result is scaled back to [0, 1]
    """)

# ROM Table Preview
st.markdown("---")
st.subheader("ROM Table Preview (First & Last 10 Entries)")

rom_df = pd.DataFrame({
    "Address": list(range(10)) + list(range(246, 256)),
    "Real Input": [INPUT_MIN + (a/(ROM_SIZE-1))*(INPUT_MAX-INPUT_MIN) for a in list(range(10)) + list(range(246, 256))],
    "ROM Value (0-255)": list(ROM_TABLE[:10]) + list(ROM_TABLE[246:]),
    "Approx Sigmoid": [v/255 for v in list(ROM_TABLE[:10]) + list(ROM_TABLE[246:])]
})
st.dataframe(rom_df, use_container_width=True)

# Comparison Plot
st.markdown("---")
st.subheader("Exact Sigmoid vs ROM Approximation")

x = np.linspace(-8, 8, 500)
exact_curve = mathematical_sigmoid(x)
approx_curve = np.array([rom_sigmoid(xi) for xi in x])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, exact_curve, label="Mathematical Sigmoid", linewidth=2)
ax.plot(x, approx_curve, label="ROM Lookup", linestyle="--", linewidth=2)
ax.set_xlabel("Input (x)")
ax.set_ylabel("Sigmoid(x)")
ax.set_title("Comparison of Exact vs ROM Approximation")
ax.legend()
ax.grid(True, alpha=0.3)
st.pyplot(fig)

# Test Cases Section
st.markdown("---")
st.subheader("Test Cases Results")

normal_cases = [-6, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 6]
edge_cases = [-8, 8, -10, 12, 999]

results = []
for x in normal_cases + edge_cases:
    exact = mathematical_sigmoid(np.clip(x, -8, 8))
    approx = rom_sigmoid(x)
    results.append({
        "Type": "Normal" if x in normal_cases else "Edge/Fault",
        "Input": x,
        "Exact": round(exact, 5),
        "ROM": round(approx, 5),
        "Error": round(abs(exact - approx), 5)
    })

st.dataframe(pd.DataFrame(results), use_container_width=True)

st.success("All core requirements of the mini-project are demonstrated above.")
