import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np
import os

# --- GitHub画像情報 ---
GITHUB_API_URL = "https://api.github.com/repos/openai/openai-cookbook/contents/examples/data/sample_clothes/sample_images"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

# --- One-hot 性別変換 ---
def assign_gender(filename):
    file_id = int(filename.split(".")[0])
    return "Male" if file_id % 2 == 1 else "Female"

gender_map = {
    "Male": [1, 0, 0],
    "Female": [0, 1, 0],
    "Other": [0, 0, 1]
}

# --- 色ベクトル抽出（中央領域の平均RGB） ---
def extract_feature_vector(image_url, gender):
    try:
        # 性別ベクトル
        gender_vec = gender_map.get(gender, [0, 0, 0])

        # 画像取得・中央32x32の平均色を抽出
        image = Image.open(BytesIO(requests.get(image_url).content)).convert("RGB").resize((64, 64))
        arr = np.array(image)
        center = arr[16:48, 16:48].reshape(-1, 3)
        rgb_mean = np.mean(center, axis=0).tolist()

        return gender_vec + rgb_mean, rgb_mean
    except Exception as e:
        print(f"⚠️ Error processing {image_url}: {e}")
        return None, None

# --- GitHub画像一覧取得 ---
def fetch_image_filenames():
    response = requests.get(GITHUB_API_URL)
    if response.status_code != 200:
        raise Exception("❌ Failed to fetch GitHub image list")
    return [item['name'] for item in response.json() if item['name'].endswith('.jpg')]

# --- features.json 出力 ---
def generate_features():
    filenames = fetch_image_filenames()
    features = []

    for filename in filenames:
        gender = assign_gender(filename)
        image_url = SAMPLE_IMAGES_URL + filename

        feature_vector, rgb_vector = extract_feature_vector(image_url, gender)
        if feature_vector is None:
            continue

        features.append({
            "filename": filename,
            "gender": gender,
            "rgb": rgb_vector,
            "feature_vector": feature_vector
        })

        print(f"✅ {filename} → {gender} / RGB: {np.round(rgb_vector, 1).tolist()}")

    with open("features.json", "w") as f:
        json.dump(features, f, indent=2)

    print(f"\n🎉 {len(features)} items saved to features.json")

# --- 実行 ---
if __name__ == "__main__":
    generate_features()
