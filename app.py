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
import faiss
from typing import List, Dict

# --- OpenAI Client Setup ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
st.set_page_config(page_title="🌟 RetailNext Coordinator", layout="wide")

POSTS_FILE = "posts.json"
PRODUCTS_FILE = "products.json"

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

# --- GPT Recommendation (Embedding + FAISS + GPT-4o) ---
def embed_product_text(product: Dict) -> List[float]:
    text = f"{product['name']}. {product['description']}. Color: {product['color']}. Style: {product['style']}."
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def build_index(products: List[Dict]):
    embeddings = [embed_product_text(p) for p in products]
    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, embeddings

def generate_recommendation(user_profile: Dict, matched_products: List[Dict]) -> str:
    system_msg = """
You are a fashion assistant. Based on the user's description and the matching items, recommend the best one in natural language.
"""
    user_msg = f"""
User Profile:
- Gender: {user_profile['gender']}
- Theme: {user_profile['theme']}
- Favorite Color: {user_profile['color']}

Matching Items:
"""
    for i, item in enumerate(matched_products):
        user_msg += f"{i+1}. {item['name']} - {item['description']}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
    )
    return response.choices[0].message.content

def recommend_with_gpt_streamlit(user_profile: Dict, products_file: str = "products.json", top_k: int = 5):
    with open(products_file) as f:
        products = json.load(f)

    st.info("🔍 Generating smart recommendation with GPT-4o...")

    index, _ = build_index(products)
    query_text = f"{user_profile['theme']} fashion for {user_profile['gender']}, favorite color: {user_profile['color']}"
    query_emb = client.embeddings.create(model="text-embedding-3-small", input=query_text).data[0].embedding

    D, I = index.search(np.array([query_emb]).astype("float32"), k=top_k)
    matched = [products[i] for i in I[0]]
    return generate_recommendation(user_profile, matched), matched

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

        st.subheader("🧠 GPT's Recommendation")
        user_profile = {"gender": gender, "theme": fashion_theme, "color": favorite_color}
        try:
            rec_text, matched = recommend_with_gpt_streamlit(user_profile)
            st.markdown(rec_text)

            st.markdown("### 🖼️ Top 3 Matching Items")
            cols = st.columns(3)
            for i, item in enumerate(matched):
                with cols[i % 3]:
                    st.image(item["image_url"], caption=item["name"], use_container_width=True)
                    st.markdown(f"[🛒 View Product]({item['product_url']})", unsafe_allow_html=True)
        except Exception as e:
            st.error("GPT recommendation failed")
            st.exception(e)

with tab2:
    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]

    if top_posts:
        st.subheader("🔥 Top 5 Popular Coordinations")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1} ❤️ {post['likes']} Likes")
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
                    st.markdown(f"**🏋️ Body Shape:** {post.get('body_shape', 'N/A')} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎮 Style:** {post['style']}")
        st.markdown("---")

    st.subheader("🧑‍📽️ All Community Posts")
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
                    st.markdown(f"**🏋️ Body Shape:** {post.get('body_shape', 'N/A')} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎮 Style:** {post['style']}")
                    st.markdown(f"❤️ {post['likes']} likes")
                    if st.button("👍 Like", key=post["id"]):
                        like_post(post["id"])
                        st.experimental_rerun()
