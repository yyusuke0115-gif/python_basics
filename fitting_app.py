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
st.header("Model Selection (AIC vs Log-Likelihood)")

# 各分布のフィッティングと対数尤度の計算
results = {}

# 各分布のパラメータ数 (k)
# Normal: mu, sigma (2)
# Exponential: loc, scale (2)
# Gamma: shape, loc, scale (3)
# Laplace: loc, scale (2)

def calculate_aic(log_lh, k):
    return -2 * log_lh + 2 * k

# --- Fitting Calculations ---
# Normal (k=2)
p_norm = stats.norm.fit(data)
ll_norm = stats.norm.logpdf(data, *p_norm).sum()
results["Normal"] = {"params": p_norm, "ll": ll_norm, "aic": calculate_aic(ll_norm, 2)}

# Exponential (k=2)
p_expon = stats.expon.fit(data)
ll_expon = stats.expon.logpdf(data, *p_expon).sum()
results["Exponential"] = {"params": p_expon, "ll": ll_expon, "aic": calculate_aic(ll_expon, 2)}

# Gamma (k=3) ← パラメータが多い！
p_gamma = stats.gamma.fit(data)
ll_gamma = stats.gamma.logpdf(data, *p_gamma).sum()
results["Gamma"] = {"params": p_gamma, "ll": ll_gamma, "aic": calculate_aic(ll_gamma, 3)}

# --- 3. Visualization ---
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(data, bins=40, density=True, alpha=0.3, color='gray', label='Generated Data')

x = np.linspace(min(data), max(data), 200)
colors = {'Normal': 'red', 'Exponential': 'green', 'Gamma': 'blue', 'Laplace': 'orange'}

# ここで results.items() から辞書の中身を取り出すよう修正
for name, info in results.items():
    params = info["params"]
    log_lh = info["ll"]
    
    if name == "Normal": pdf = stats.norm.pdf(x, *params)
    elif name == "Exponential": pdf = stats.expon.pdf(x, *params)
    elif name == "Gamma": pdf = stats.gamma.pdf(x, *params)
    elif name == "Laplace": pdf = stats.laplace.pdf(x, *params)
    
    ax.plot(x, pdf, label=f"{name} (AIC: {info['aic']:.1f})", color=colors[name], lw=2)

ax.legend()
st.pyplot(fig)

# --- 4. Rank Table (AIC順: 小さいほど良い) ---
st.subheader("Leaderboard: Best Model by AIC")

# AICの値で昇順（小さい順）にソート
ranked_results = sorted(results.items(), key=lambda x: x[1]['aic'], reverse=False)

cols = st.columns(len(ranked_results))
for i, (name, info) in enumerate(ranked_results):
    with cols[i]:
        # AICを表示。前回との差分としてLog-LHを表示するのも面白いです
        st.metric(f"Rank {i+1}: {name}", f"AIC: {info['aic']:.1f}", delta=f"LL: {info['ll']:.1f}", delta_color="off")

winner = ranked_results[0][0]
if winner == true_dist:
    st.success(f"Perfect! AIC chose **{winner}**, which is the true model.")
else:
    st.warning(f"Note: AIC preferred **{winner}** over the true {true_dist}. (Sample size: {sample_size})")