import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
import os
import json
import uuid
from io import BytesIO
import numpy as np

# --- 設定 ---
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- 定数 ---
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

def update_likes(post_id):
    posts = load_posts()
    for p in posts:
        if p["id"] == post_id:
            p["likes"] += 1
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)

# --- UIタブ ---
tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])



# -----------------------------
# 🧠 コーデ診断
# -----------------------------
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
        # 顔写真を一時保存
        image = Image.open(user_photo)
        image.save("temp.png")

        # プロンプトの作成
        user_prompt = f"""
以下のお客様の要望に合うファッションコーディネート画像を作成してください：

・住んでいる国: {country}
・性別: {gender}
・年齢: {age}
・体型: {body_shape}
・好きな色: {favorite_color}
・ファッションテーマ: {concept}
・アニメスタイル: {anime_style}

出力形式は、1人の人物がそのファッションに身を包んでいるアニメスタイルの全身イラスト。
背景は白、余計な要素を含めず、人物と服装に焦点を当ててください。
"""

        # DALL·E 3による画像生成
        with open("temp.png", "rb") as image_file:
            response = client.Image.create(
                model="dall-e-3",
                prompt=user_prompt,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        os.remove("temp.png")

        ai_img_url = response["data"][0]["url"]
        st.image(ai_img_url, caption="👕 AIコーデ提案", use_container_width=True)

        # 類似アイテム（ランダムに3件）
        st.subheader("🛍 類似アイテム")
        for i in range(3):
            rand_id = np.random.randint(10000, 10500)
            item_url = f"{SAMPLE_IMAGES_URL}{rand_id}.jpg"
            st.image(item_url, width=200, caption=f"商品{i+1} - ¥{np.random.randint(2900, 8900)}")
            st.markdown(f"[🛒 カートに追加（ダミー）](#)", unsafe_allow_html=True)

        # 保存
        save_post({
            "id": str(uuid.uuid4()),
            "image_url": ai_img_url,
            "country": country,
            "gender": gender,
            "age": age,
            "style": anime_style,
            "color": favorite_color,
            "theme": concept,
            "likes": 0
        })



# -----------------------------
# 🌐 みんなのコーデ
# -----------------------------
with tab2:
    st.header("🌐 みんなのコーディネート")
    posts = load_posts()[::-1]

    if not posts:
        st.info("まだ投稿がありません。")
    else:
        for post in posts:
            with st.container():
                st.image(post["image_url"], caption=f"{post['country']} / {post['gender']} / {post['age']}歳", use_container_width=True)
                st.markdown(f"🧵 テーマ: `{post['theme']}`　🎨 色: `{post['color']}`　🧍‍♀️ スタイル: `{post['style']}`")
                if st.button(f"❤️ いいね ({post['likes']})", key=post["id"]):
                    update_likes(post["id"])
                    st.experimental_rerun()



# -----------------------------
# 🔥 人気ランキング
# -----------------------------
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
                st.markdown(f"""
                - 🧵 テーマ: `{post['theme']}`  
                - 🎨 色: `{post['color']}`  
                - 🧍‍♀️ スタイル: `{post['style']}`  
                """)
