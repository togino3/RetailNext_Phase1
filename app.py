import streamlit as st
import json
import numpy as np
import pandas as pd
import ast
from openai import OpenAI
from typing import List, Dict
from PIL import Image
import requests
from io import BytesIO
import uuid
import os

# --- Setup ---
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

POSTS_FILE = "posts.json"
EMBEDDED_JSON_FILE = "embedded_products_with_id.json"

# --- Post Management ---
if "posts" not in st.session_state:
    if os.path.exists(POSTS_FILE):
        with open(POSTS_FILE, "r") as f:
            st.session_state["posts"] = json.load(f)
    else:
        st.session_state["posts"] = []

def load_posts():
    return st.session_state["posts"]

def save_post(post):
    st.session_state["posts"].append(post)
    with open(POSTS_FILE, "w") as f:
        json.dump(st.session_state["posts"], f, indent=2)

def like_post(post_id):
    for post in st.session_state["posts"]:
        if post["id"] == post_id:
            post["likes"] += 1
    with open(POSTS_FILE, "w") as f:
        json.dump(st.session_state["posts"], f, indent=2)

# --- 類似検索 ---
def get_embedding_3small(text: str, api_key: str):
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)

def recommend_from_embedded_json(user_profile: Dict, top_k: int = 5):
    with open(EMBEDDED_JSON_FILE, "r") as f:
        items = json.load(f)

    query_text = (
        f"{user_profile['theme']} fashion for {user_profile['gender']}, "
        f"color: {user_profile['color']}, suitable for ceremony or special event."
    )
    embedding = get_embedding_3small(query_text, st.secrets["OPENAI_API_KEY"])

    all_vectors = np.array([item["embedding"] for item in items], dtype=np.float32)
    scores = np.dot(all_vectors, embedding) / (
        np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(embedding) + 1e-5
    )
    for i, score in enumerate(scores):
        items[i]["score"] = score

    top_items = sorted(items, key=lambda x: x["score"], reverse=True)[:top_k]
    return top_items

# --- UI Layout ---
tab1, tab2 = st.tabs(["🧐 AI Coordinator", "🌐 Community Gallery"])

with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("fashion_form"):
        uploaded_image = st.file_uploader("😊 Upload your face photo", type=["jpg", "jpeg", "png"])
        country = st.text_input("🌍 Country (e.g., USA, Japan, etc.)")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 25)
        body_shape = st.selectbox("Body Shape", ["Slim", "Regular", "Curvy"])
        favorite_color = st.text_input("🎨 Favorite Color (e.g., black, pink)")
        draw_style = st.selectbox("Drawing Style", ["Disney", "American Comic", "Japanese Anime", "3D CG"])
        fashion_theme = st.text_input("🎭 Fashion Theme (e.g., spring, bright)")
        submitted = st.form_submit_button("✨ Generate AI Coordination")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_url = None
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=f"Full-body anime-style fashion image of a {gender}, age {age}, body shape {body_shape}, color {favorite_color}, theme {fashion_theme}, style {draw_style}. White background.",
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url
            st.image(image_url, caption="👕 AI Coordination Suggestion", width=300)
        except Exception as e:
            st.error("Image generation failed")
            st.exception(e)

        save_post({
            "id": str(uuid.uuid4()),
            "image_url": image_url if image_url else "N/A",
            "country": country,
            "gender": gender,
            "age": age,
            "body_shape": body_shape,
            "color": favorite_color,
            "theme": fashion_theme,
            "style": draw_style,
            "likes": 0
        })

        st.subheader("🧠 Similar Items Recommendation")
        user_profile = {"gender": gender, "theme": fashion_theme, "color": favorite_color}
        try:
            similar = recommend_from_embedded_json(user_profile)
            for item in similar:
                image_url = f"https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/{item['id']}.jpg"
                st.image(image_url, width=200)
                st.markdown(f"**{item['productDisplayName']}** - {item['gender']}, {item['baseColour']}, {item['season']}, {item['usage']}")
        except Exception as e:
            st.error("Recommendation failed")
            st.exception(e)

with tab2:
    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]
    if top_posts:
        st.subheader("🍭 Recommended Products")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1} ❤️ {post['likes']} Likes")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(post["image_url"], width=150)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body Shape:** {post['body_shape']} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")
    st.markdown("---")

    st.subheader("🧑‍🤝‍🧑 All Community Posts")
    for post in reversed(posts[:20]):
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(post["image_url"], width=150)
            with col2:
                st.markdown(f"**🧵 Theme:** {post['theme']}")
                st.markdown(f"**🌍 Country:** {post['country']}")
                st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                st.markdown(f"**💪 Body Shape:** {post['body_shape']} / 🎨 Color: {post['color']}")
                st.markdown(f"**🎞️ Style:** {post['style']}")
                st.markdown(f"❤️ {post['likes']} likes")
                if st.button("👍 Like", key=post["id"]):
                    like_post(post["id"])
                    st.experimental_rerun()
