import datetime
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split


# log dir for tensorboard
log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# データ読み込み
# CSVファイルの読み込み
dataset = pd.read_csv("auto_mpg.csv")

# 欠損値の処理
dataset = dataset.dropna()

# 各カラムで "?" が含まれている行を削除
for col in dataset.columns.values:
    dataset = dataset[dataset[col] != "?"]
print(dataset.head())

# 特徴量とターゲットの分割
X = dataset.drop("mpg", axis=1)
y = dataset["mpg"]

# car_nameはdrop
X = X.drop("car_name", axis=1)

# データの標準化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# スケーラーの保存
joblib.dump(scaler, "scaler.pkl")


# データ分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# モデル構築
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(1)
])

# モデルコンパイル
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# モデル訓練
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tensorboard_callback])
          # model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

# 評価
loss, mae = model.evaluate(X_test, y_test, verbose=2)
print(f"Mean Absolute Error on Test Data: {mae}")

model.summary()
model.save('auto_mpg_model.keras')
# model.save('auto_mpg_model.h5')
