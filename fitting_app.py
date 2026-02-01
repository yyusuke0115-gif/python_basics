import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.title("Probability Distribution Fitting Tool")

# --- 1. Data Generation (Creating an "Unknown" dataset) ---
st.sidebar.header("1. Generate Data")
true_dist = st.sidebar.selectbox("True Distribution (The Answer)", ["Normal", "Exponential", "Uniform"])
sample_size = st.sidebar.slider("Sample Size", 50, 1000, 200)

if st.sidebar.button("Generate New Data"):
    if true_dist == "Normal":
        st.session_state.data = np.random.normal(loc=5.0, scale=2.0, size=sample_size)
    elif true_dist == "Exponential":
        st.session_state.data = np.random.exponential(scale=2.0, size=sample_size)
    else:
        st.session_state.data = np.random.uniform(low=0, high=10, size=sample_size)

if 'data' not in st.session_state:
    st.warning("Please click 'Generate New Data' in the sidebar.")
    st.stop()

data = st.session_state.data

# --- 2. Distribution Fitting ---
st.header("2. Fitting Results")

# Fit to Normal Distribution
mu_fit, sigma_fit = stats.norm.fit(data)

# Fit to Exponential Distribution
loc_fit, scale_fit = stats.expon.fit(data)

# --- 3. Visualization ---
fig, ax = plt.subplots()
ax.hist(data, bins=30, density=True, alpha=0.5, color='gray', label='Original Data')

# Plot Fitted Normal Curve
x = np.linspace(min(data), max(data), 100)
pdf_norm = stats.norm.pdf(x, mu_fit, sigma_fit)
ax.plot(x, pdf_norm, 'r-', lw=2, label=f'Fitted Normal\n(mu={mu_fit:.2f}, sigma={sigma_fit:.2f})')

# Plot Fitted Exponential Curve
pdf_expon = stats.expon.pdf(x, loc_fit, scale_fit)
ax.plot(x, pdf_expon, 'g--', lw=2, label=f'Fitted Expon\n(scale={scale_fit:.2f})')

ax.set_title("Fitting Different Distributions to the Data")
ax.legend()
st.pyplot(fig)

st.write("The red line shows the best-fit Normal distribution, and the green dashed line shows the best-fit Exponential distribution.")