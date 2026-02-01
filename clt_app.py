import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Page Configuration ---
st.set_page_config(page_title="CLT Simulator", layout="wide")
st.title("Central Limit Theorem (CLT) Simulator")
st.markdown("""
This app visualizes how the **distribution of sample means** approaches a **Normal Distribution** as the sample size ($n$) increases, regardless of the original distribution's shape.
""")

# --- 2. Sidebar Settings ---
st.sidebar.header("Simulation Parameters")

dist_type = st.sidebar.selectbox(
    "1. Select Parent Distribution",
    ("Uniform", "Exponential", "Binomial", "Bimodal (Double Peak)")
)

n = st.sidebar.slider("2. Sample Size (n)", min_value=1, max_value=100, value=10, 
                     help="Number of items in each single sample.")
m = st.sidebar.slider("3. Iterations (m)", min_value=100, max_value=10000, value=1000, 
                     help="How many times we calculate the sample mean.")

# --- 3. Data Generation & Theoretical Values ---
# Initialize theoretical variables
t_mu, t_sigma = 0.0, 0.0

if dist_type == "Uniform":
    parent_data = np.random.rand(10000)
    t_mu, t_sigma = 0.5, np.sqrt(1/12)
    def get_sample(size): return np.random.rand(size)

elif dist_type == "Exponential":
    parent_data = np.random.exponential(scale=1.0, size=10000)
    t_mu, t_sigma = 1.0, 1.0
    def get_sample(size): return np.random.exponential(scale=1.0, size=size)

elif dist_type == "Binomial":
    parent_data = np.random.binomial(n=1, p=0.5, size=10000)
    t_mu, t_sigma = 0.5, 0.5
    def get_sample(size): return np.random.binomial(n=1, p=0.5, size=size)

else: # Bimodal
    d1 = np.random.normal(loc=-2, scale=0.8, size=5000)
    d2 = np.random.normal(loc=2, scale=0.8, size=5000)
    parent_data = np.concatenate([d1, d2])
    t_mu, t_sigma = 0.0, np.sqrt(0.8**2 + 2**2)
    def get_sample(size):
        indices = np.random.choice([0, 1], size=size)
        return np.array([np.random.normal(-2, 0.8) if i == 0 else np.random.normal(2, 0.8) for i in indices])

# Generate m sample means
sample_means = [np.mean(get_sample(n)) for _ in range(m)]

# --- 4. Visualization ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Parent Distribution
ax1.hist(parent_data, bins=50, color='lightgray', edgecolor='black', alpha=0.7)
ax1.set_title(f"Original {dist_type} Distribution")
ax1.set_xlabel("Value")
ax1.set_ylabel("Frequency")

# Plot 2: Sample Mean Distribution
ax2.hist(sample_means, bins=50, color='skyblue', edgecolor='black', density=True, alpha=0.7)

# Add Normal Curve based on simulation results
mu, sigma = np.mean(sample_means), np.std(sample_means)
x_axis = np.linspace(min(sample_means), max(sample_means), 100)
y_axis = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x_axis - mu) / sigma)**2)
ax2.plot(x_axis, y_axis, 'r', linewidth=2, label='Normal Curve')

ax2.set_title(f"Sample Mean Distribution (n={n})")
ax2.set_xlabel("Mean Value")
ax2.set_ylabel("Density")
ax2.legend()

st.pyplot(fig)

# --- 5. Theoretical Verification ---
st.divider()
st.header("Theoretical Verification")

expected_se = t_sigma / np.sqrt(n)

col1, col2 = st.columns(2)
with col1:
    st.latex(r"E[\bar{X}] = \mu")
    st.metric("Expected Mean", f"{t_mu:.4f}")
    st.metric("Actual Mean", f"{mu:.4f}", delta=f"{mu-t_mu:.4f}", delta_color="off")

with col2:
    st.latex(r"SE = \frac{\sigma}{\sqrt{n}}")
    st.metric("Expected SE", f"{expected_se:.4f}")
    st.metric("Actual SE", f"{sigma:.4f}", delta=f"{sigma-expected_se:.4f}", delta_color="off")

st.info(f"Observation: In {dist_type}, as n increases from 1 to {n}, notice how the right graph transforms into a bell curve.")