import numpy as np
import matplotlib.pyplot as plt

# 1. データを作る（-10 から 10 までを 0.1 刻みで）
x = np.arange(-10, 10, 0.1)

# 2. 数式を定義する（ここではAIの活性化関数でも使われるシグモイド関数っぽく、あるいは単純な二次関数）
# せっかくなので、綺麗なカーブを描く「サイン波」と「二次関数」を重ねてみましょう
y1 = np.sin(x)           # sin(x)
y2 = x ** 2 / 20         # x^2 を 20で割って広げたもの

# 3. グラフを描画する
plt.figure(figsize=(10, 6)) # 画面のサイズ設定

plt.plot(x, y1, label='y = sin(x)', color='blue')       # 青線
plt.plot(x, y2, label='y = x^2 / 20', color='red', linestyle='--') # 赤点線

plt.title("Welcome to Data Science!") # タイトル
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.legend() # 凡例を表示
plt.grid(True) # グリッド線を表示

# 4. 画面に表示する
print("グラフウィンドウを表示します...")
plt.show()