import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.title("🤖 AI学習シミュレーター")
st.write("ボタンを押すと、AIがデータの法則を見つけ出します！")

# --- 1. サイドバーで設定を変えられるようにする ---
st.sidebar.header("ハイパーパラメータ設定")
learning_rate = st.sidebar.slider("学習率 (Learning Rate)", 0.01, 0.5, 0.1)
epochs = st.sidebar.slider("学習回数 (Epochs)", 10, 100, 50)

# --- 2. データ生成 ---
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
true_y = 4 + 3 * X + np.random.randn(100, 1) # 正解: y = 3x + 4

# --- 3. 「学習開始」ボタンが押されたら動く ---
if st.button("学習スタート"):
    # パラメータ初期化
    w = np.random.randn(1)
    b = np.random.randn(1)
    
    loss_history = []
    
    # プログレスバー（進行状況）を表示
    progress_bar = st.progress(0)
    
    for i in range(epochs):
        # 予測と学習 (Gradient Descent)
        y_pred = w * X + b
        loss = np.mean((y_pred - true_y) ** 2)
        loss_history.append(loss)
        
        dw = 2 * np.mean((y_pred - true_y) * X)
        db = 2 * np.mean(y_pred - true_y)
        
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # バーを進める
        progress_bar.progress((i + 1) / epochs)

    st.success("学習完了！")
    
    # --- 4. 結果をブラウザに表示 (st.pyplot) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # 左：グラフ
    ax1.scatter(X, true_y, color='blue', alpha=0.5)
    ax1.plot(X, w * X + b, color='red', linewidth=3)
    ax1.set_title(f"Result: y = {w[0]:.2f}x + {b[0]:.2f}")
    ax1.grid(True)
    
    # 右：誤差の推移
    ax2.plot(loss_history)
    ax2.set_title("Loss History")
    ax2.grid(True)
    
    st.pyplot(fig) # これでグラフがブラウザに出ます