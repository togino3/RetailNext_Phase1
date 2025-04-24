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

# --- Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")

POSTS_FILE = "posts.json"
FEATURES_FILE = "features.json"
SAMPLE_IMAGES_URL = "https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/"

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

@st.cache_data
def load_feature_vectors():
    with open(FEATURES_FILE, "r") as f:
        data = json.load(f)
    for item in data:
        item["feature_vector"] = np.array(item["feature_vector"])  # ✅ 修正ポイント
    return data

# --- 類似画像推薦用：PILベースのベクトル抽出 ---
def extract_color_vector_from_PIL(pil_image):
    image = pil_image.resize((32, 32)).convert("RGB")
    arr = np.array(image).reshape(-1, 3)
    return np.mean(arr, axis=0)

def find_similar_images_from_PIL(pil_image, target_gender, top_k=5):
    base_rgb = np.array(pil_image.resize((64, 64)).convert("RGB"))
    center_rgb = base_rgb[16:48, 16:48].reshape(-1, 3)
    mean_rgb = np.mean(center_rgb, axis=0)
    
    features = load_feature_vectors()
    similarities = []

    for item in features:
        if item["gender"] != target_gender:
            continue
        vec = np.array(item["feature_vector"])
        sim = cosine_similarity([mean_rgb], [vec[3:]])[0][0]
        full_url = SAMPLE_IMAGES_URL + item["filename"]
        similarities.append((sim, full_url))

    return [url for _, url in sorted(similarities, reverse=True)[:top_k]]

# --- Tab Layout ---
tab1, tab2 = st.tabs(["🧠 AI Coordinator", "🌐 Community Gallery"])

# -----------------------
# 🧠 AI Coordinator
# -----------------------
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
        fashion_theme = st.text_input("🧵 Fashion Theme (e.g., spring, bright)")
        submitted = st.form_submit_button("✨ Generate AI Coordination")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()

        user_prompt = f"""
Generate a full-body anime-style fashion coordination image for one person, based on the following conditions:

- Country: {country}
- Gender: {gender}
- Age: {age}
- Body Shape: {body_shape}
- Favorite Color: {favorite_color}
- Fashion Theme: {fashion_theme}
- Drawing Style: {draw_style}

[Output Requirements]
- Background should be white
- Focus on the outfit and the person
- Face should be natural and not overly emphasized
- The person must be clearly clothed; avoid nudity or excessive exposure
"""

        response = client.images.generate(
            model="dall-e-3",
            prompt=user_prompt,
            size="1024x1024",
            quality="standard",
            n=1
        )
        image_url = response.data[0].url
        st.image(image_url, caption="👕 AI Coordination Suggestion", width=300)

        # --- ローカルに一時保存してPILで開く ---
        dalle_img_response = requests.get(image_url)
        local_path = f"generated/{str(uuid.uuid4())}.png"
        os.makedirs("generated", exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(dalle_img_response.content)
        dalle_image = Image.open(local_path)

        # --- 類似アイテム推薦 ---
        category = "Top" if "shirt" in fashion_theme.lower() or "top" in fashion_theme.lower() else "Bottom"
        st.subheader("🛍 Similar Items")
        similar_images = find_similar_images_from_PIL(dalle_image, gender)
        cols = st.columns(5)
        for i, url in enumerate(similar_images):
            with cols[i % 5]:
                st.image(url, use_column_width=True)
                st.markdown(f"[🛒 Add to Cart](#)", unsafe_allow_html=True)

        save_post({
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
        })

        st.success("👚 Your coordination has been posted to the community!")

# -----------------------
# 🌐 Community Gallery
# -----------------------
with tab2:

    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]

    if top_posts:
        st.subheader("🔥 Top 5 Popular Coordinations")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1}　❤️ {post['likes']} Likes")
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_path = post["image_url"]
                    if not image_path.startswith("http"):
                        image_path = os.path.join(".", image_path)
                    st.image(image_path, width=150)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body Shape:** {post.get('body_shape', 'N/A')} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")
        st.markdown("---")

    st.subheader("🧑‍🤝‍🧑 All Community Posts")
    if not posts:
        st.info("No posts yet.")
    else:
        for post in reversed(posts[:20]):
            with st.container():
                col1, col2 = st.columns([1, 2])
                with col1:
                    image_path = post["image_url"]
                    if not image_path.startswith("http"):
                        image_path = os.path.join(".", image_path)
                    st.image(image_path, width=150)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body Shape:** {post.get('body_shape', 'N/A')} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")
                    st.markdown(f"❤️ {post['likes']} likes")
                    if st.button("👍 Like", key=post["id"]):
                        like_post(post["id"])
                        st.experimental_rerun()