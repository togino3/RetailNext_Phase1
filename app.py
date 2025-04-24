import streamlit as st
import openai
import requests
from PIL import Image
import os
import json
import uuid
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- 設定 ---
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")

# --- APIキー ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- データフォルダ ---
POSTS_FILE = "posts.json"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

# --- 初期化 ---
if not os.path.exists(POSTS_FILE):
    with open(POSTS_FILE, "w") as f:
        json.dump([], f)

def load_posts():
    with open(POSTS_FILE, "r") as f:
        return json.load(f)

def save_post(post):
    posts = load_posts()
    posts.append(post)
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)

# --- UIタブ構成 ---
tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])

with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("coord_form"):
        user_photo = st.file_uploader("📷 顔写真をアップロード", type=["jpg", "png"])
        country = st.selectbox("🌍 国", ["Japan", "USA", "France", "Brazil", "India"])
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body_shape = st.selectbox("体型", ["スリム", "普通", "がっしり"])
        favorite_color = st.color_picker("🎨 好きな色", "#000000")
        anime_style = st.selectbox("アニメスタイル", ["日本レトロ", "ディズニー", "アメリカンコミック", "CG"])
        concept = st.text_input("ファッションテーマ（例：春っぽく）", "")

        submitted = st.form_submit_button("✨AIコーディネート生成")

    if submitted and user_photo:
        img_bytes = user_photo.read()
        base64_img = openai.Image.create_variation(
            image=img_bytes,
            n=1,
            size="512x512"
        ).data[0].url

        st.image(base64_img, caption="👕 AIコーデ提案", use_container_width=True)

        # 類似商品表示（ダミーURLからランダム選出）
        st.subheader("🛍 類似アイテム")
        for i in range(3):
            image_url = f"{SAMPLE_IMAGES_URL}{np.random.randint(10000, 10500)}.jpg"
            st.image(image_url, width=200, caption=f"商品{i+1} - ¥{np.random.randint(2900, 8900)}")
            st.markdown(f"[🛒 カートに追加（ダミー）](#)", unsafe_allow_html=True)

        # 投稿保存
        save_post({
            "id": str(uuid.uuid4()),
            "image_url": base64_img,
            "country": country,
            "gender": gender,
            "age": age,
            "style": anime_style,
            "color": favorite_color,
            "theme": concept,
            "likes": 0
        })

with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("coord_form"):
        user_photo = st.file_uploader("📷 顔写真をアップロード", type=["jpg", "png"])
        country = st.selectbox("🌍 国", ["Japan", "USA", "France", "Brazil", "India"])
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body_shape = st.selectbox("体型", ["スリム", "普通", "がっしり"])
        favorite_color = st.color_picker("🎨 好きな色", "#000000")
        anime_style = st.selectbox("アニメスタイル", ["日本レトロ", "ディズニー", "アメリカンコミック", "CG"])
        concept = st.text_input("ファッションテーマ（例：春っぽく）", "")

        submitted = st.form_submit_button("✨AIコーディネート生成")

    if submitted and user_photo:
        img_bytes = user_photo.read()
        base64_img = openai.Image.create_variation(
            image=img_bytes,
            n=1,
            size="512x512"
        ).data[0].url

        st.image(base64_img, caption="👕 AIコーデ提案", use_container_width=True)

        # 類似商品表示（ダミーURLからランダム選出）
        st.subheader("🛍 類似アイテム")
        for i in range(3):
            image_url = f"{SAMPLE_IMAGES_URL}{np.random.randint(10000, 10500)}.jpg"
            st.image(image_url, width=200, caption=f"商品{i+1} - ¥{np.random.randint(2900, 8900)}")
            st.markdown(f"[🛒 カートに追加（ダミー）](#)", unsafe_allow_html=True)

        # 投稿保存
        save_post({
            "id": str(uuid.uuid4()),
            "image_url": base64_img,
            "country": country,
            "gender": gender,
            "age": age,
            "style": anime_style,
            "color": favorite_color,
            "theme": concept,
            "likes": 0
        })

with tab3:
    st.header("🔥 人気ランキング")

    posts = sorted(load_posts(), key=lambda x: x["likes"], reverse=True)
    if not posts:
        st.info("まだランキングがありません。")
    else:
        for i, post in enumerate(posts[:10]):
            with st.container():
                st.subheader(f"#{i+1}　❤️ {post['likes']} Likes")
                st.image(post["image_url"], caption=f"{post['country']} / {post['gender']} / {post['age']}歳", use_container_width=True)
                st.markdown(f"🧵 テーマ: `{post['theme']}`　🎨 色: `{post['color']}`　🧍‍♀️ スタイル: `{post['style']}`")
