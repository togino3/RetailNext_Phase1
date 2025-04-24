import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np

SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"
GITHUB_API_URL = "https://api.github.com/repos/openai/openai-cookbook/contents/examples/data/sample_clothes/sample_images"

def extract_color_vector(image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).resize((32, 32))
        arr = np.array(image).reshape(-1, 3)
        return np.mean(arr, axis=0).tolist()
    except:
        return [0, 0, 0]

def generate_json():
    print("▶ GitHubから画像リストを取得中...")
    response = requests.get(GITHUB_API_URL)
    image_files = [item['name'] for item in response.json() if item['name'].endswith('.jpg')]

    features = {}
    for filename in image_files:
        url = SAMPLE_IMAGES_URL + filename
        print(f"  - 処理中: {filename}")
        features[filename] = extract_color_vector(url)

    with open("color_features.json", "w") as f:
        json.dump(features, f, indent=2)

    print("✅ color_features.json が作成されました！")

if __name__ == "__main__":
    generate_json()
