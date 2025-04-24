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

# セッション内に投稿データ保持
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

def like_post(post_id):
    for post in st.session_state["posts"]:
        if post["id"] == post_id:
            post["likes"] += 1

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
        full_url = SAMPLE_IMAGES_URL + item["filename"]
        similarities.append((sim, full_url))
    return [url for _, url in sorted(similarities, reverse=True)[:top_k]]

def show_image(image_url, width=150):
    if os.path.exists(image_url):
        st.image(image_url, width=width)
    elif image_url.startswith("http"):
        st.image(image_url, width=width)
    else:
        st.warning("❗ Image not found: " + image_url)




# --- Tab Layout ---
tab1, tab2 = st.tabs(["🧠 AI Coordinator", "🌐 Community Gallery"])

# ==========================
# 🧠 AI Coordinator タブ
# ==========================
with tab1:
    st.title("🌟 RetailNext Coordinator")

    with st.form("fashion_form"):
        uploaded_image = st.file_uploader("😊 Upload your face photo", type=["jpg", "jpeg", "png"])
        country = st.text_input("🌍 Country (e.g., USA, Japan)")
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.slider("Age", 1, 100, 25)
        body_shape = st.selectbox("Body Shape", ["Slim", "Regular", "Curvy"])
        favorite_color = st.text_input("🎨 Favorite Color (e.g., Blue, Pink)")
        draw_style = st.selectbox("Drawing Style", ["Disney", "American Comic", "Japanese", "CG"])
        fashion_theme = st.text_input("🧵 Fashion Theme (e.g., Spring, Business)")
        submitted = st.form_submit_button("✨ Generate AI Coordination")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        prompt = f"""
Generate a full-body anime-style fashion coordination image for one person with the following details:
Country: {country}, Gender: {gender}, Age: {age}, Body: {body_shape}, Color: {favorite_color}, Theme: {fashion_theme}, Style: {draw_style}
Requirements: white background, clearly clothed, no nudity or excessive exposure.
"""

        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        st.image(image_url, caption="👕 AI Coordination Suggestion", width=300)

        st.subheader("🛍 Similar Items")
        category = "Top" if "shirt" in fashion_theme.lower() or "top" in fashion_theme.lower() else "Bottom"
        similar_images = find_similar_images(image_url, gender, category)
        for url in similar_images:
            st.image(url, width=120)
            st.markdown(f"[🛒 Add to Cart (Dummy)](#)", unsafe_allow_html=True)

        new_post = {
            "id": str(uuid.uuid4()),
            "image_url": image_url,
            "country": country,
            "gender": gender,
            "age": age,
            "body_shape": body_shape,
            "color": favorite_color,
            "theme": fashion_theme,
            "style": draw_style,
            "likes": 0
        }
        save_post(new_post)
        st.success("✅ Your coordination has been posted!")

# ==========================
# 🌐 Community Gallery タブ
# ==========================
with tab2:
    st.header("🌐 Community Gallery")
    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]

    if top_posts:
        st.subheader("🔥 Top 5 Popular Coordinations")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1} ❤️ {post['likes']} Likes")
                col1, col2 = st.columns([1, 2])
                with col1:
                    show_image(post["image_url"], width=150)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body:** {post['body_shape']} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")

    st.markdown("---")
    st.subheader("👥 All Community Posts")

    if not posts:
        st.info("No posts yet.")
    else:
        for post in reversed(posts[-20:]):
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    show_image(post["image_url"], width=150)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body:** {post['body_shape']} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")
                    st.markdown(f"❤️ {post['likes']} likes")
                    if st.button("👍 Like", key=post["id"]):
                        like_post(post["id"])
                        st.experimental_rerun()
