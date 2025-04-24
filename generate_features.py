import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np

# GitHub 上の画像一覧取得API
GITHUB_API_URL = "https://api.github.com/repos/openai/openai-cookbook/contents/examples/data/sample_clothes/sample_images"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

# 仮の性別割り当て（奇数: 男性、偶数: 女性）
def assign_gender(filename):
    file_id = int(filename.split(".")[0])
    return "男性" if file_id % 2 == 1 else "女性"

# 平均色ベクトル抽出
def extract_color_vector(image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).resize((32, 32))
        arr = np.array(image).reshape(-1, 3)
        return np.mean(arr, axis=0).tolist()
    except Exception as e:
        print(f"⚠️ Error processing {image_url}: {e}")
        return [0, 0, 0]

# GitHub APIで画像ファイル名を取得
def fetch_image_filenames():
    response = requests.get(GITHUB_API_URL)
    if response.status_code != 200:
        raise Exception("GitHub APIの取得に失敗しました")
    return [item['name'] for item in response.json() if item['name'].endswith('.jpg')]

# メイン処理：features.json を生成
def generate_features():
    filenames = fetch_image_filenames()
    features = {}

    for filename in filenames:
        image_url = SAMPLE_IMAGES_URL + filename
        color_vector = extract_color_vector(image_url)
        gender = assign_gender(filename)

        features[filename] = {
            "color": color_vector,
            "gender": gender
        }
        print(f"✅ Processed: {filename} | Gender: {gender}")

    # ✅ 出力ファイル名を features.json に変更
    with open("features.json", "w") as f:
        json.dump(features, f, indent=2)
    print("\n🎉 features.json が生成されました！")

# 実行
if __name__ == "__main__":
    generate_features()
