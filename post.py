import requests
import json
# cylinders:    float
# displacement: float
# horsepower:   float
# weight:       int
# acceleration: float
# model_year:   int
# origin:       int

# POST先のURL
url = "http://localhost:8008/predict/"

# 送信するデータ (JSON形式)
data = {
    "cylinders":       6,
    "displacement":  225,
    "horsepower":    100,
    "weight":        3233,
    "acceleration":  15.4,
    "model_year":     76, 
    "origin":        1
}

# リクエストヘッダー (必要に応じて設定)
headers = {
    "Content-Type": "application/json",  # データがJSON形式であることを指定
    "Authorization": "Bearer your_token_here"  # 認証が必要な場合
}

# POSTリクエストの送信
response = requests.post(url, data=json.dumps(data), headers=headers)

# レスポンスの確認
if response.status_code == 200:
    print("成功:", response.json())  # JSON形式のレスポンスを取得
else:
    print("エラー:", response.status_code, response.text)

# print(response.status_code)
# print(response.text)