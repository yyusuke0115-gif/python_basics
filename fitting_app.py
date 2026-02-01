import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Multi-Dist Fitting", layout="wide")
st.title("Probability Distribution Fitting Tool (4 Models)")

# --- 1. Data Generation ---
st.sidebar.header("1. Settings")
# 選択肢を増やす
dist_options = ["Normal", "Exponential", "Uniform", "Gamma", "Laplace"]
true_dist = st.sidebar.selectbox("True Distribution (The Answer)", dist_options)
sample_size = st.sidebar.slider("Sample Size", 50, 1000, 300)

if st.sidebar.button("Generate New Data"):
    if true_dist == "Normal":
        st.session_state.data = np.random.normal(5.0, 2.0, sample_size)
    elif true_dist == "Exponential":
        st.session_state.data = np.random.exponential(2.0, sample_size)
    elif true_dist == "Uniform":
        st.session_state.data = np.random.uniform(0, 10, sample_size)
    elif true_dist == "Gamma":
        st.session_state.data = np.random.gamma(shape=2.0, scale=2.0, size=sample_size)
    elif true_dist == "Laplace":
        st.session_state.data = np.random.laplace(5.0, 1.0, sample_size)

if 'data' not in st.session_state:
    st.info("Please generate data from the sidebar.")
    st.stop()

data = st.session_state.data

# --- 2. Multi-Fitting & Scoring ---
st.header("Fitting Results & Log-Likelihood")

# 各分布のフィッティングと対数尤度の計算
results = {}

# 1. Normal
params_norm = stats.norm.fit(data)
results["Normal"] = (params_norm, stats.norm.logpdf(data, *params_norm).sum())

# 2. Exponential
params_expon = stats.expon.fit(data)
results["Exponential"] = (params_expon, stats.expon.logpdf(data, *params_expon).sum())

# 3. Gamma
params_gamma = stats.gamma.fit(data)
results["Gamma"] = (params_gamma, stats.gamma.logpdf(data, *params_gamma).sum())

# 4. Laplace
params_laplace = stats.laplace.fit(data)
results["Laplace"] = (params_laplace, stats.laplace.logpdf(data, *params_laplace).sum())

# --- 3. Visualization ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data, bins=40, density=True, alpha=0.3, color='gray', label='Generated Data')

x = np.linspace(min(data), max(data), 200)
colors = {'Normal': 'red', 'Exponential': 'green', 'Gamma': 'blue', 'Laplace': 'orange'}

for name, (params, log_lh) in results.items():
    if name == "Normal": pdf = stats.norm.pdf(x, *params)
    elif name == "Exponential": pdf = stats.expon.pdf(x, *params)
    elif name == "Gamma": pdf = stats.gamma.pdf(x, *params)
    elif name == "Laplace": pdf = stats.laplace.pdf(x, *params)
    
    ax.plot(x, pdf, label=f"{name} (LL: {log_lh:.1f})", color=colors[name], lw=2)

ax.legend()
st.pyplot(fig)

# --- 4. Rank Table ---
st.subheader("Leaderboard: Which fits best?")
# スコア順に並び替え
ranked_results = sorted(results.items(), key=lambda x: x[1][1], reverse=True)

cols = st.columns(len(ranked_results))
for i, (name, (params, log_lh)) in enumerate(ranked_results):
    with cols[i]:
        st.metric(f"Rank {i+1}: {name}", f"{log_lh:.1f}")

winner = ranked_results[0][0]
if winner == true_dist:
    st.success(f"Perfect! The mathematical winner is **{winner}**, which matches the ground truth.")
else:
    st.warning(f"Surprise! **{winner}** fits better than the original {true_dist}. This happens with small samples.")