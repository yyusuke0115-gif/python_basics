import sys
import random
from datetime import datetime

# 実行環境の確認
print("--- Python環境セットアップ確認 ---")
print(f"Pythonバージョン: {sys.version.split()[0]}")
print(f"現在時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 運試し機能
print("\n[システム] 運試しプログラムを起動中...")
lucky_number = random.randint(0, 100)

print(f"今日のあなたのラッキーナンバーは... {lucky_number} です！")

if lucky_number >= 80:
    print(">> 結果: 素晴らしいスタートです！Devフォルダでの開発が捗りそうですね。")
elif lucky_number >= 50:
    print(">> 結果: 順調です！まずはコードを書いて慣れていきましょう。")
else:
    print(">> 結果: エラーが出ずにこれが表示されていれば、実質100点満点です！")