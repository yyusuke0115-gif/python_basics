import numpy as np
import matplotlib.pyplot as plt

# 1. 関数とその微分（導関数）を定義
def function(x):
    return x ** 2  # y = x^2 (放物線)

def derivative(x):
    return 2 * x   # f'(x) = 2x (これを使って「坂の傾き」を知る)

# 2. パラメータ設定
x = -8.0       # スタート地点（左の高いところから開始）
learning_rate = 0.05  # 学習率（一歩の大きさ。大きいと大股、小さいと小股で進む）
steps = []     # 軌跡を記録するリスト

# 3. 学習ループ（谷底に向かって進む処理）
print("学習を開始します...")
for i in range(20):
    y = function(x)
    steps.append((x, y)) # 現在地を記録
    
    # 【最重要】傾き(grad)を使って、谷底の方向へxを更新する
    # 傾きがプラスなら左へ、マイナスなら右へ進む（傾きの逆方向へ進む）
    grad = derivative(x)
    x = x - learning_rate * grad 
    
    print(f"Step {i+1}: x = {x:.4f}, 傾き = {grad:.4f}")

# 4. 結果をグラフにする
steps = np.array(steps)

# 背景の放物線を描く
x_base = np.arange(-10, 10, 0.1)
y_base = function(x_base)
plt.plot(x_base, y_base, label='Loss Function (y=x^2)')

# ボールの軌跡を描く（赤丸でプロット）
plt.scatter(steps[:, 0], steps[:, 1], color='red', s=50, zorder=5)
plt.plot(steps[:, 0], steps[:, 1], color='red', linestyle='--', label='AI Learning Path')

plt.title("Gradient Descent Visualization")
plt.xlabel("Parameter X")
plt.ylabel("Loss (Error)")
plt.legend()
plt.grid(True)
plt.show()