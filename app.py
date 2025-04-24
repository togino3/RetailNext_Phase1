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

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")
POSTS_FILE = "posts.json"
FEATURES_FILE = "features.json"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

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

@st.cache_data
def load_feature_vectors():
    with open(FEATURES_FILE, "r") as f:
        data = json.load(f)
    for item in data:
        item["vector"] = np.array(item["vector"])
    return data

def extract_color_vector(image_url):
    try:
        image = Image.open(BytesIO(requests.get(image_url).content)).resize((32, 32))
        arr = np.array(image).reshape(-1, 3)
        return np.mean(arr, axis=0)
    except:
        return np.array([0, 0, 0])

def find_similar_images(image_url, target_gender, target_category, top_k=3):
    base_vec = extract_color_vector(image_url)
    features = load_feature_vectors()
    similarities = []

    for item in features:
        if item["gender"] != target_gender or item["category"] != target_category:
            continue
        vec = item["vector"]
        sim = cosine_similarity([base_vec], [vec])[0][0]
        image_url = SAMPLE_IMAGES_URL + item["filename"]
        similarities.append((sim, image_url))

    return [url for _, url in sorted(similarities, reverse=True)[:top_k]]

# --- タブ構成 ---
tab1, tab2 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーディネート"])

# ------------------------
# 🧠 コーデ診断
# ------------------------
with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("fashion_form"):
        uploaded_image = st.file_uploader("👕 顔写真をアップロード", type=["jpg", "jpeg", "png"])
        country = st.text_input("🌍 国（例：Japan, USA など）")
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body_shape = st.selectbox("体型", ["スリム", "標準", "ぽっちゃり"])
        favorite_color = st.text_input("🎨 好きな色（例：black, pink など）")
        draw_style = st.selectbox("作画スタイル", ["ディズニー", "アメリカンコミック", "日本", "CG"])
        fashion_theme = st.text_input("🧵 ファッションテーマ（例：春っぽく、明るく）")
        submitted = st.form_submit_button("✨ AIコーディネート生成")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        user_prompt = f"""
以下の条件に基づいて、人物が全身で1人で写っている作画スタイルのファッションコーディネート画像を生成してください：

・国: {country}
・性別: {gender}
・年齢: {age}歳
・体型: {body_shape}
・好きな色: {favorite_color}
・ファッションテーマ: {fashion_theme}
・作画スタイル: {draw_style}

【出力画像の条件】
- 背景は白
- 人物とファッションが中心
- 顔は作画スタイルで自然、目立ちすぎない
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

        # 類似商品（色＋性別＋カテゴリベース）
        st.subheader("🛍 類似商品")
        category = "トップス" if "シャツ" in fashion_theme or "トップス" in fashion_theme else "ボトムス"  # 仮判定ロジック
        similar_images = find_similar_images(image_url, gender, category)
        for url in similar_images:
            st.image(url, width=200)
            st.markdown(f"[🛒 カートに追加（ダミー）](#)", unsafe_allow_html=True)

        save_post({
            "id": str(uuid.uuid4()),
            "image_url": image_url,
            "country": country,
            "gender": gender,
            "age": age,
            "style": draw_style,
            "color": favorite_color,
            "theme": fashion_theme,
            "likes": 0
        })
        st.success("👚 コーデ画像をコミュニティに投稿しました！")



# ------------------------
# 🌐 みんなのコーディネート
# ------------------------
with tab2:
    st.header("🌐 みんなのコーディネート")

    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]

    if top_posts:
        st.subheader("🔥 人気ランキング TOP 5")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1}　❤️ {post['likes']} Likes")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(post["image_url"], width=150)
                with col2:
                    st.markdown(f"**🧵 テーマ:** {post['theme']}")
                    st.markdown(f"**🌍 国:** {post['country']}")
                    st.markdown(f"**👤 性別:** {post['gender']} / **🎂 年齢:** {post['age']}歳")
                    st.markdown(f"**💪 体型:** {post.get('body_shape', 'N/A')} / **🎨 色:** {post['color']}")
                    st.markdown(f"**🎞️ スタイル:** {post['style']}")
        st.markdown("---")

    st.subheader("🧑‍🤝‍🧑 みんなの投稿一覧")

    if not posts:
        st.info("まだ投稿がありません。")
    else:
        for post in reversed(posts[:20]):  # 最新の20件を表示
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(post["image_url"], width=150)
                with col2:
                    st.markdown(f"**🧵 テーマ:** {post['theme']}")
                    st.markdown(f"**🌍 国:** {post['country']}")
                    st.markdown(f"**👤 性別:** {post['gender']} / **🎂 年齢:** {post['age']}歳")
                    st.markdown(f"**💪 体型:** {post.get('body_shape', 'N/A')} / **🎨 色:** {post['color']}")
                    st.markdown(f"**🎞️ スタイル:** {post['style']}")
                    st.markdown(f"❤️ {post['likes']} likes")
                    if st.button("👍 いいねする", key=post["id"]):
                        like_post(post["id"])
                        st.experimental_rerun()
