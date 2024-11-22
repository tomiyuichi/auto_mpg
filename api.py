from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # リクエストbodyを定義するために必要
import tensorflow as tf
import numpy as np
import uvicorn
import joblib

# FastAPI アプリケーションの初期化
app = FastAPI()

# モデルのロード
model = tf.keras.models.load_model('auto_mpg_model.keras')

# 保存したスケーラーを読み込む
scaler = joblib.load("scaler.pkl")

# 入力データのスキーマ
class InputData(BaseModel):
    cylinders:    float
    displacement: float
    horsepower:   float
    weight:       int
    acceleration: float
    model_year:   int
    origin:       int

# ルートエンドポイント（動作確認用）
@app.get("/")
async def root():
    return {"message": "Hello World"}

# 推論用エンドポイント
@app.post("/predict/")
def predict(data: InputData):
    try:
        # 特徴量の取得と変換
        x0 = data.cylinders
        x1 = data.displacement
        x2 = data.horsepower
        x3 = data.weight
        x4 = data.acceleration
        x5 = data.model_year
        x6 = data.origin
        X_new = np.array([[x0, x1, x2, x3, x4, x5, x6]])  # 新しい特徴量データ
        X_new_scaled = scaler.transform(X_new)
        # 予測の実行
        prediction = model.predict(X_new_scaled)
        # 結果を返す
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8008, log_level="debug")

# # 新しい特徴量データをスケーリング
# X_new = np.array([[6, 225, 100, 3233, 15.4, 76, 1]])  # 新しい特徴量データ
# X_new_scaled = scaler.transform(X_new)

# # モデルを読み込む
# model = tf.keras.models.load_model("auto_mpg_model.keras")

# # 予測
# prediction = model.predict(X_new_scaled)
# print("予測結果:", prediction)