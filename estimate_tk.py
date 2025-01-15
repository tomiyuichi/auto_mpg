import sys
class NullWriter:
    def write(self, message): pass
    def flush(self): pass

sys.stdout = NullWriter()  # 標準出力を無効化
sys.stderr = NullWriter()  # 標準エラー出力を無効化

import joblib
import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import filedialog, messagebox

# 保存したスケーラーを読み込む
scaler = joblib.load("scaler.pkl")

# モデルを読み込む
model = tf.keras.models.load_model("auto_mpg_model.keras")

# 予測関数
def predict():
    try:
        # 入力データを取得
        input_data = entry_input.get()
        
        # 入力データを処理 (例: カンマ区切りの数値を配列に変換)
        # X_new = np.array([[6, 225, 100, 3233, 15.4, 76, 1]])  # 新しい特徴量データ
        X_new = np.array([float(x) for x in input_data.split(",")]).reshape(1, -1)

        # 新しい特徴量データをスケーリング
        X_new_scaled = scaler.transform(X_new)
        
        # モデルで予測
        prediction = model.predict(X_new_scaled)
        
        # 結果を表示
        # print("予測結果:", prediction)
        # print(f"Predicted MPG: {prediction[0][0]}")
        result_label.config(text=f"予測結果: {prediction}")        

    except Exception as e:
        import traceback
        error_message = traceback.format_exc()
        messagebox.showerror("エラー", f"詳細エラー: {error_message}")
        # messagebox.showerror("エラー", f"入力エラーまたはモデルエラー: {e}")


# Tkinter ウィンドウの作成
root = tk.Tk()
root.title("TensorFlowモデル予測アプリ")

# 入力ラベルとエントリボックス
tk.Label(root, text="入力データ (カンマ区切り):").pack(pady=5)
entry_input = tk.Entry(root, width=50)
entry_input.pack(pady=5)

# 予測ボタン
btn_predict = tk.Button(root, text="予測", command=predict)
btn_predict.pack(pady=10)

# 結果表示ラベル
result_label = tk.Label(root, text="予測結果: 未実行", fg="blue")
result_label.pack(pady=5)

# アプリの実行
root.mainloop()