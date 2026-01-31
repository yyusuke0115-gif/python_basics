import numpy as np
import matplotlib.pyplot as plt

# 1. 偽の「正解データ」を作る
# 真の法則: y = 3x + 4 に、少しノイズ(雑音)を混ぜたもの
np.random.seed(42)
X = 2 * np.random.rand(100, 1) # 0〜2の範囲のランダムなx
y = 4 + 3 * X + np.random.randn(100, 1) # y = 3x + 4 + ノイズ

# 2. AIのパラメータ（wとb）をテキトーに初期化
# 最初は全然違う場所(直線の角度)を向いています
w = np.random.randn(1) # 重み (Weight)
b = np.random.randn(1) # バイアス (Bias)

# ハイパーパラメータ
learning_rate = 0.1
iterations = 100

print(f"学習開始時のパラメータ: w={w[0]:.2f}, b={b[0]:.2f}")

# 3. 学習ループ
loss_history = []

for i in range(iterations):
    # 予測する (Forward Propagation)
    y_pred = w * X + b
    
    # 誤差を計算する (Mean Squared Error)
    loss = np.mean((y_pred - y) ** 2)
    loss_history.append(loss)
    
    # 【ここが数学の核心】
    # 誤差関数の「偏微分」を計算して、修正方向を決める
    # L = (y_pred - y)^2 なので、wで偏微分すると...
    dw = 2 * np.mean((y_pred - y) * X) # wに関する傾き
    db = 2 * np.mean(y_pred - y)       # bに関する傾き
    
    # パラメータ更新 (Gradient Descent)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    
    if i % 10 == 0:
        print(f"Step {i}: Loss={loss:.4f}, w={w[0]:.2f}, b={b[0]:.2f}")

print(f"学習完了！ AIが見つけた法則: y = {w[0]:.2f}x + {b[0]:.2f}")
print(f"本来の正解 (真の法則):     y = 3.00x + 4.00")

# 4. 結果を可視化
plt.figure(figsize=(10, 5))

# 左側: データと学習した直線のグラフ
plt.subplot(1, 2, 1)
plt.scatter(X, y, color='blue', alpha=0.5, label='Data (Noisy)')
plt.plot(X, w * X + b, color='red', linewidth=3, label='AI Prediction')
plt.title("Linear Regression Result")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)

# 右側: 誤差が減っていく様子のグラフ
plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.title("Loss History")
plt.xlabel("Iterations")
plt.ylabel("Loss (MSE)")
plt.grid(True)

plt.tight_layout()
plt.show()