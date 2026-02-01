import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Config ---
st.title("Central Limit Theorem Simulator")

st.sidebar.header("Settings")

dist_type = st.sidebar.selectbox(
    "Select Original Distribution",
    ("Uniform", "Exponential", "Binomial")
)

n = st.sidebar.slider("Sample Size (n)", 1, 100, 10)
m = st.sidebar.slider("Number of Iterations (m)", 100, 10000, 1000)

# --- Data Generation ---
# Generate parent distribution for visualization
if dist_type == "Uniform":
    parent_data = np.random.rand(10000)
elif dist_type == "Exponential":
    parent_data = np.random.exponential(scale=1.0, size=10000)
else:
    parent_data = np.random.binomial(n=1, p=0.5, size=10000)

# Generate sample means
sample_means = []
for _ in range(m):
    if dist_type == "Uniform":
        data = np.random.rand(n)
    elif dist_type == "Exponential":
        data = np.random.exponential(scale=1.0, size=n)
    else:
        data = np.random.binomial(n=1, p=0.5, size=n)
    sample_means.append(np.mean(data))

# --- Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 1. Parent Distribution
ax1.hist(parent_data, bins=50, color='lightgray', edgecolor='black')
ax1.set_title(f"Original {dist_type} Distribution")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")

# 2. Sample Mean Distribution (CLT)
ax2.hist(sample_means, bins=50, color='skyblue', edgecolor='black', density=True)

# Add Normal Distribution Curve
mu = np.mean(sample_means)
sigma = np.std(sample_means)
x = np.linspace(min(sample_means), max(sample_means), 100)
p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
ax2.plot(x, p, 'r', linewidth=2, label='Normal Dist')

ax2.set_title(f"Sample Mean Distribution (n={n})")
ax2.set_xlabel("Mean Value")
ax2.set_ylabel("Density")
ax2.legend()

st.pyplot(fig)

# --- Statistics Output ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Sample Mean (μ)", f"{mu:.4f}")
with col2:
    st.metric("Std Deviation (σ)", f"{sigma:.4f}")