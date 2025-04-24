import streamlit as st
import pandas as pd
import json
import requests
from PIL import Image
import os
from datetime import datetime
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import random
import csv

# OpenAI APIキー
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# 定数
POSTS_FILE = "data/posts.json"
PRODUCT_INFO_CSV = "data/product_info.csv"
GENERATED_IMAGE_PATH = "data/generated.png"
BASE_IMAGE_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images"

# 初期化
os.makedirs("data", exist_ok=True)
if not os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "w") as f:
        json.dump([], f)

# ダミー商品データを自動生成
def generate_dummy_product_info():
    filenames = [f"100{n:02}.jpg" for n in range(20, 40)]  # 仮ファイル名
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

# CLIPモデル初期化
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ベクトル取得
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

# 類似商品TOP3取得
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

# 投稿ロード/保存
def load_posts():
    with open(POSTS_FILE, "r") as f:
        return json.load(f)

def save_post(post):
    posts = load_posts()
    posts.append(post)
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f)

# UI
st.set_page_config(page_title="RetailNext Portal", layout="wide")
st.title("🌟 RetailNext Coordinator")

tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])

# -------------------------------
# タブ1：コーデ生成
# -------------------------------
with tab1:
    with st.form("form"):
        country = st.text_input("🌍 国")
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body = st.selectbox("体型", ["スリム", "標準", "ぽっちゃり"])
        color = st.color_picker("🎨 好きな色")
        style = st.selectbox("アニメスタイル", ["日本レトロ", "ディズニー", "アメコミ", "CG"])
        theme = st.text_input("ファッションテーマ（例：春っぽく）")
        submitted = st.form_submit_button("✨ AIコーディネート生成")

    if submitted:
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
                "body": body,
                "style": style,
                "theme": theme,
                "like_count": 0
            })
            st.success("✅ 投稿しました！")

# -------------------------------
# タブ2：みんなのコーデ
# -------------------------------
with tab2:
    posts = load_posts()
    st.subheader("🌐 投稿一覧")
    for i, post in enumerate(reversed(posts)):
        cols = st.columns([1, 3, 1])
        with cols[0]:
            st.image(post["image_url"], width=150)
        with cols[1]:
            st.markdown(f"**{post['theme']} ({post['style']})**")
            st.caption(f"{post['country']} / {post['age']}歳 / {post['gender']} / {post['body']}")
        with cols[2]:
            if st.button("👍 いいね", key=f"like_{i}"):
                post["like_count"] += 1
                with open(POSTS_FILE, "w") as f:
                    json.dump(posts, f)
            st.write(f"❤️ {post['like_count']}")

# -------------------------------
# タブ3：人気ランキング
# -------------------------------
with tab3:
    st.subheader("🔥 いいねランキング")
    posts = sorted(load_posts(), key=lambda x: x["like_count"], reverse=True)
    for post in posts[:5]:
        st.image(post["image_url"], width=300)
        st.caption(f"❤️ {post['like_count']}｜{post['country']} / {post['age']}歳 / {post['style']}")

    st.subheader("🌎 国別トレンド")
    df = pd.DataFrame(posts)
    if not df.empty:
        for country in df["country"].value_counts().index[:3]:
            st.markdown(f"#### {country}")
            country_posts = df[df["country"] == country].sort_values("like_count", ascending=False).head(3)
            for _, row in country_posts.iterrows():
                st.image(row["image_url"], width=200)
                st.caption(f"❤️ {row['like_count']}｜{row['theme']} / {row['age']}歳 / {row['gender']}")
