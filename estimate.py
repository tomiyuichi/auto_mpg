import joblib
import numpy as np
import tensorflow as tf

# 保存したスケーラーを読み込む
scaler = joblib.load("scaler.pkl")

# 新しい特徴量データをスケーリング
X_new = np.array([[6, 225, 100, 3233, 15.4, 76, 1]])  # 新しい特徴量データ
X_new_scaled = scaler.transform(X_new)

# モデルを読み込む
model = tf.keras.models.load_model("auto_mpg_model.keras")

# 予測
prediction = model.predict(X_new_scaled)
print("予測結果:", prediction)
# print(f"Predicted MPG: {prediction[0][0]}")
