import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import json
import os
import base64
from datetime import datetime
import random
import pandas as pd

st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")

POSTS_FILE = "posts.json"

# -------------------------------------
# 初期化関数
# -------------------------------------
def load_posts():
    if os.path.exists(POSTS_FILE):
        with open(POSTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_post(post):
    posts = load_posts()
    posts.append(post)
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)

# -------------------------------------
# 類似画像のダミーデータ作成
# -------------------------------------
def get_similar_items():
    base_url = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"
    image_ids = random.sample(range(10000, 10500), 3)
    return [f"{base_url}{img_id}.jpg" for img_id in image_ids]

# -------------------------------------
# 投稿UIタブ
# -------------------------------------
tab1, tab2, tab3 = st.tabs(["🧠 コーデ診断", "🌐 みんなのコーデ", "🔥 人気ランキング"])

# -------------------------------------
# タブ①：コーデ診断
# -------------------------------------
with tab1:
    st.title("🌟 RetailNext Coordinator")
    st.markdown("ユーザーの特徴に基づいてAIがコーディネートを提案します")

    with st.form("fashion_form"):
        uploaded_image = st.file_uploader("🖼 自分の写真をアップロード", type=["jpg", "jpeg", "png"])
        country = st.text_input("🌍 国", placeholder="例: Japan")
        gender = st.selectbox("性別", ["男性", "女性", "その他"])
        age = st.slider("年齢", 1, 100, 25)
        body = st.selectbox("体型", ["スリム", "標準", "がっしり"])
        color = st.color_picker("🎨 好きな色")
        style = st.selectbox("アニメスタイル", ["日本レトロ", "ディズニー", "アメリカンコミック", "CG"])
        theme = st.text_input("ファッションテーマ（例：春っぽく）")
        submitted = st.form_submit_button("✨ AIコーディネート生成")

    if submitted and uploaded_image:
        st.image(uploaded_image, caption="アップロード画像", use_column_width=True)

        # ✅ 画像生成プロンプト（OpenAI連携用）
        prompt = (
            f"以下のお客様の要望に合うファッションコーディネート画像を作成してください：\n"
            f"国: {country}, 性別: {gender}, 年齢: {age}, 体型: {body}, 好きな色: {color}, "
            f"アニメスタイル: {style}, ファッションテーマ: {theme}"
        )

        st.markdown("🔧 使用プロンプト:")
        st.code(prompt)

        # ✅ ダミー生成画像
        st.subheader("🧥 AIコーディネート画像（イメージ）")
        sample_result = "https://source.unsplash.com/512x512/?fashion,clothes"
        st.image(sample_result, use_column_width=True)

        # ✅ 類似商品（ダミー画像3つ）
        st.subheader("🛍 類似商品の提案")
        cols = st.columns(3)
        similar_items = get_similar_items()
        for i, col in enumerate(cols):
            with col:
                st.image(similar_items[i], caption=f"商品 {i+1}")
                st.markdown("🛒 [カートに追加（ダミー）](#)")

        # ✅ コーデを保存（共有）
        if st.button("🌍 コーデを共有する"):
            save_post({
                "timestamp": str(datetime.now()),
                "image_url": sample_result,
                "country": country,
                "gender": gender,
                "age": age,
                "body": body,
                "color": color,
                "style": style,
                "theme": theme,
                "like_count": 0
            })
            st.success("✅ コーディネートを共有しました！")

# -------------------------------------
# タブ②：みんなのコーデ
# -------------------------------------
with tab2:
    st.header("🌐 みんなのコーデ")
    posts = load_posts()
    if not posts:
        st.info("まだ共有されたコーデがありません。")
    else:
        for i, post in enumerate(reversed(posts)):
            st.image(post["image_url"], width=300, caption=f"{post['theme']} | {post['country']} | {post['age']}歳")
            st.markdown(f"👍 {post['like_count']} いいね")
            if st.button(f"❤️ いいねする", key=f"like_{i}"):
                post["like_count"] += 1
                with open(POSTS_FILE, "w", encoding="utf-8") as f:
                    json.dump(posts, f, ensure_ascii=False, indent=2)
                st.rerun()

# -------------------------------------
# タブ③：人気ランキング
# -------------------------------------
with tab3:
    st.header("🔥 人気ランキング（いいね数順）")
    posts = sorted(load_posts(), key=lambda x: x["like_count"], reverse=True)
    for post in posts[:10]:
        st.image(post["image_url"], width=300)
        st.markdown(f"🏳️ 国: {post['country']} | 年齢: {post['age']} | 性別: {post['gender']}")
        st.markdown(f"🎨 テーマ: {post['theme']} | ❤️ いいね: {post['like_count']}")
