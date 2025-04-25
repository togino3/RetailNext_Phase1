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
EMBEDDINGS_FILE = "sample_styles_with_embeddings.csv"

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

# --- 類似検索 + GPT推薦 ---
def recommend_from_precomputed(user_profile: Dict, top_k: int = 3):
    df = pd.read_csv(EMBEDDINGS_FILE)
    df["embedding"] = df["embeddings"].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float32))

    # --- Step 1: filter by gender and color ---
    df_filtered = df[
        (df["gender"].str.lower() == user_profile["gender"].lower()) &
        (df["baseColour"].str.lower().str.contains(user_profile["color"].lower(), na=False))
    ]

    if df_filtered.empty:
        df_filtered = df  # fallback to all if no match

    all_vectors = np.stack(df_filtered["embedding"].values)

    query_text = (
        f"{user_profile['theme']} fashion for {user_profile['gender']}, "
        f"color: {user_profile['color']}, suitable for ceremony or special event."
    )

    embedding = client.embeddings.create(model="text-embedding-ada-002", input=query_text).data[0].embedding
    embedding = np.array(embedding, dtype=np.float32)

    scores = np.dot(all_vectors, embedding) / (
        np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(embedding) + 1e-5
    )
    df_filtered["score"] = scores
    top_items = df_filtered.sort_values("score", ascending=False).head(top_k)

    items_description = "\n".join(
        [f"{row['productDisplayName']} - {row['gender']}, {row['baseColour']}, {row['articleType']}" for _, row in top_items.iterrows()]
    )
    user_msg = f"""
User Profile:
- Gender: {user_profile['gender']}
- Theme: {user_profile['theme']}
- Favorite Color: {user_profile['color']}

Top Matching Items:
{items_description}
    """

    system_msg = "You are a fashion AI assistant. Please recommend items based on user's profile and the matching product list. Prioritize items that match gender and color exactly."
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )

    return response.choices[0].message.content, top_items.to_dict(orient="records")

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

        st.subheader("🧠 GPT's Recommendation")
        user_profile = {"gender": gender, "theme": fashion_theme, "color": favorite_color}
        try:
            rec_text, matched = recommend_from_precomputed(user_profile)
            st.markdown(rec_text)
            st.markdown("### 🛍 Recommend Item")
            cols = st.columns(3)
            for i, item in enumerate(matched):
                with cols[i % 3]:
                    st.image(item.get("imageUrl", ""), caption=item["productDisplayName"], use_container_width=True)
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
