import requests
import json
from PIL import Image
from io import BytesIO
import numpy as np

GITHUB_API_URL = "https://api.github.com/repos/openai/openai-cookbook/contents/examples/data/sample_clothes/sample_images"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

def assign_gender(filename):
    file_id = int(filename.split(".")[0])
    return "Male" if file_id % 2 == 1 else "Female"

def assign_category(filename):
    file_id = int(filename.split(".")[0])
    return "Top" if file_id % 3 == 0 else "Bottom"

def extract_color_vector(image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).resize((32, 32))
        arr = np.array(image).reshape(-1, 3)
        return np.mean(arr, axis=0).tolist()
    except Exception as e:
        print(f"⚠️ Error processing {image_url}: {e}")
        return None  # ← 修正ポイント

def fetch_image_filenames():
    response = requests.get(GITHUB_API_URL)
    if response.status_code != 200:
        raise Exception("Failed to fetch GitHub image list")
    return [item['name'] for item in response.json() if item['name'].endswith('.jpg')]

def generate_features():
    filenames = fetch_image_filenames()
    features = []

    for filename in filenames:
        image_url = SAMPLE_IMAGES_URL + filename
        vector = extract_color_vector(image_url)
        if vector is None:
            continue  # skip faulty image

        features.append({
            "filename": filename,
            "vector": vector,
            "gender": assign_gender(filename),
            "category": assign_category(filename)
        })
        print(f"✅ Processed: {filename}")

    with open("features.json", "w") as f:
        json.dump(features, f, indent=2)
    print("🎉 features.json has been successfully generated!")

if __name__ == "__main__":
    generate_features()
