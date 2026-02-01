import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("中心極限定理シミュレーター")

st.sidebar.header("設定")

# 1. 元の分布を選択
dist_type = st.sidebar.selectbox(
    "元の分布を選択してください",
    ("一様分布 (Uniform)", "指数分布 (Exponential)", "二項分布 (Binomial)")
)

# 2. サンプルサイズ (n) と試行回数 (m)
n = st.sidebar.slider("サンプルサイズ (一度に引く数 n)", 1, 100, 10)
m = st.sidebar.slider("試行回数 (平均をとる回数 m)", 100, 10000, 1000)

# データの生成
samples = []

for _ in range(m):
    if dist_type == "一様分布 (Uniform)":
        data = np.random.rand(n)
    elif dist_type == "指数分布 (Exponential)":
        data = np.random.exponential(scale=1.0, size=n)
    else: # 二項分布
        data = np.random.binomial(n=1, p=0.5, size=n)
    
    samples.append(np.mean(data))

# 可視化
fig, ax = plt.subplots()
ax.hist(samples, bins=50, color='skyblue', edgecolor='black', density=True)

# 理論上の正規分布（ベルカーブ）の線を重ねるための計算
mu = np.mean(samples)
sigma = np.std(samples)
x = np.linspace(min(samples), max(samples), 100)
p = (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ((x - mu) / sigma)**2)
ax.plot(x, p, 'r', linewidth=2, label='Normal Distribution')

ax.set_title(f"{dist_type} の標本平均の分布 (n={n}, m={m})")
ax.set_xlabel("平均値")
ax.set_ylabel("密度")
ax.legend()

st.pyplot(fig)

st.write(f"**平均:** {mu:.4f}")
st.write(f"**標準偏差:** {sigma:.4f}")