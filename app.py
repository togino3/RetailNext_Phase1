import streamlit as st
import openai
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
import json
import os
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ✅ APIキー読み込み
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ✅ 設定
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")
POSTS_FILE = "posts.json"
SAMPLE_IMAGES = [f"https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/{i}.jpg" for i in range(10022, 10050)]

# ✅ 初期化
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

def like_post(post_id):
    posts = load_posts()
    for post in posts:
        if post["id"] == post_id:
            post["likes"] += 1
    with open(POSTS_FILE, "w") as f:
        json.dump(posts, f, indent=2)

def extract_color_vector(image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).resize((32, 32))
        arr = np.array(image).reshape(-1, 3)
        return np.mean(arr, axis=0)
    except:
        return np.array([0, 0, 0])

def find_similar_images(generated_url, top_k=3):
    base_vec = extract_color_vector(generated_url)
    similarities = []
    for img_url in SAMPLE_IMAGES:
        vec = extract_color_vector(img_url)
        sim = cosine_similarity([base_vec], [vec])[0][0]
        similarities.append((sim, img_url))
    return [url for _, url in sorted(similarities, reverse=True)[:top_k]]


# --- タブ構成 ---
tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])

with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("fashion_form"):
        uploaded_image = st.file_uploader("👕 顔写真をアップロード", type=["jpg", "jpeg", "png"])
        country = st.selectbox("🌍 国", ["Japan", "USA", "France", "Brazil", "India"])
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body_shape = st.selectbox("体型", ["スリム", "標準", "ぽっちゃり"])
        favorite_color = st.color_picker("🎨 好きな色")
        anime_style = st.selectbox("アニメスタイル", ["日本レトロ", "ディズニー", "アメリカンコミック", "CG"])
        fashion_theme = st.text_input("🧵 ファッションテーマ（例：春っぽく、明るく）")
        submitted = st.form_submit_button("✨ AIコーディネート生成")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        user_prompt = f"""
以下のお客様の要望に合うファッションコーディネート画像を作成してください：

・住んでいる国: {country}
・性別: {gender}
・年齢: {age}
・体型: {body_shape}
・好きな色: {favorite_color}
・ファッションテーマ: {fashion_theme}
・アニメスタイル: {anime_style}

出力形式は、1人の人物がそのファッションに身を包んでいるアニメスタイルの全身イラスト。
背景は白、余計な要素を含めず、人物と服装に焦点を当ててください。
"""

        response = client.images.generate(
            model="dall-e-3",
            prompt=user_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        st.image(image_url, caption="👕 AIコーデ提案", use_container_width=True)

        # 類似商品表示
        st.subheader("🛍 類似商品")
        for url in find_similar_images(image_url):
            st.image(url, width=200)
            st.markdown(f"[🛒 カートに追加（ダミー）](#)", unsafe_allow_html=True)

        save_post({
            "id": str(uuid.uuid4()),
            "image_url": image_url,
            "country": country,
            "gender": gender,
            "age": age,
            "style": anime_style,
            "color": favorite_color,
            "theme": fashion_theme,
            "likes": 0
        })

with tab2:
    st.subheader("🌐 みんなのコーデ")
    for post in load_posts()[::-1]:
        st.image(post["image_url"], caption=f"{post['country']} / {post['gender']} / {post['age']}歳")
        st.markdown(f"🧵 テーマ: `{post['theme']}`　🎨 色: `{post['color']}`　🧍‍♀️ スタイル: `{post['style']}`")
        if st.button(f"❤️ いいね {post['likes']}", key=post["id"]):
            like_post(post["id"])
            st.experimental_rerun()

with tab3:
    st.subheader("🔥 人気ランキング")
    posts = sorted(load_posts(), key=lambda x: x["likes"], reverse=True)
    for i, post in enumerate(posts[:10]):
        st.image(post["image_url"], caption=f"#{i+1} ❤️ {post['likes']} Likes", use_container_width=True)
        st.markdown(f"{post['country']} / {post['gender']} / {post['age']}歳 - 🧵 {post['theme']}")
