import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import base64
import os
import json
from datetime import datetime
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import csv
import random

# OpenAI APIキーの設定
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 定数の設定
POSTS_FILE = "data/posts.json"
PRODUCT_INFO_CSV = "data/product_info.csv"
GENERATED_IMAGE_PATH = "data/generated.png"
BASE_IMAGE_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images"

# ディレクトリの作成
os.makedirs("data", exist_ok=True)
if not os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "w") as f:
        json.dump([], f)

# ダミー商品データの生成
def generate_dummy_product_info():
    filenames = [f"100{n:02}.jpg" for n in range(20, 40)]
    with open(PRODUCT_INFO_CSV, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "name", "price", "stock"])
        for fname in filenames:
            name = f"ファッションアイテム {fname.split('.')[0]}"
            price = f"¥{random.randint(3000, 8000)}"
            stock = random.choice(["在庫あり", "在庫少", "売切れ"])
            writer.writerow([fname, name, price, stock])

if not os.path.exists(PRODUCT_INFO_CSV):
    generate_dummy_product_info()

# CLIPモデルの初期化
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# 画像のベクトル取得
def get_image_embedding_from_url(img_url):
    image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm()

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
    return features / features.norm()

# 類似商品TOP3の取得
def find_top_similar_products_from_url(coord_path, top_k=3):
    coord_vec = get_image_embedding(coord_path)
    df = pd.read_csv(PRODUCT_INFO_CSV)
    sims = []
    for _, row in df.iterrows():
        img_url = f"{BASE_IMAGE_URL}/{row['filename']}"
        try:
            product_vec = get_image_embedding_from_url(img_url)
            sim = cosine_similarity(coord_vec, product_vec)[0][0]
            sims.append((row, sim))
        except Exception as e:
            continue
    top = sorted(sims, key=lambda x: x[1], reverse=True)[:top_k]
    return [(f"{BASE_IMAGE_URL}/{r[0]['filename']}", r[0]) for r in top]

# 投稿の読み込みと保存
def load_posts():
    with open(POSTS_FILE, "r") as f:
        return json.load(f)

def save_post(post):
    posts = load_posts()
    posts.append(post)
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f)

# Streamlitの設定
st.set_page_config(page_title="RetailNext Portal", layout="wide")
st.title("🌟 RetailNext Coordinator")

tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])

# タブ1：コーデ生成
with tab1:
    with st.form("form"):
        uploaded_image = st.file_uploader("📸 自分の写真をアップロード", type=["jpg", "png", "jpeg"])
        country = st.text_input("🌍 国")
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body = st.selectbox("体型", ["スリム", "標準", "ぽっちゃり"])
        color = st.color_picker("🎨 好きな色")
        style = st.selectbox("アニメスタイル", ["日本", "ディズニー", "アメコミ", "CG"])
        theme = st.text_input("ファッションテーマ（例：春っぽく）")
        submitted = st.form_submit_button("✨ AIコーディネート生成")

    if submitted:
        if uploaded_image is None:
            st.warning("📷 まずは自分の写真をアップロードしてください")
        else:
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_API_KEY)
            prompt = f"""
以下のお客様の要望に合うファッションコーディネート画像を作成してください。
・国: {country} / 性別: {gender} / 年齢: {age} / 体型: {body}
・好きな色: {color} / ファッションテーマ: {theme} / アニメスタイル: {style}
背景は白、全身アニメスタイルの人物1人だけにしてください。
"""
            response = client.images.generate(
                model="dall-e-3",
                prompt=prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            st.image(image_url, caption="🧠 AIによるコーディネート", use_column_width=True)

            img = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
            img.save(GENERATED_IMAGE_PATH)

            st.subheader("🛍 類似商品（AI推薦）")
            top_items = find_top_similar_products_from_url(GENERATED_IMAGE_PATH)
            cols = st.columns(3)
            for col, (img_url, item) in zip(cols, top_items):
                with col:
                    st.image(img_url, use_column_width=True)
                    st.caption(f"{item['name']} / {item['price']} / {item['stock']}")
                    st.markdown("[🛒 カートに追加](#)")

            if st.button("🌍 コーデを共有する"):
                save_post({
                    "timestamp": str(datetime.now()),
                    "image_url": image_url,
                    "country": country,
                    "gender": gender,
                    "age": age,
                    "body":
::contentReference[oaicite:2]{index=2}
 
