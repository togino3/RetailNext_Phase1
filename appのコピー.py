import streamlit as st
import json
import numpy as np
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
EMBEDDED_JSON_FILE = "embedded_products.json"

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




# --- Embedding and Recommendation Functions ---

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

def recommend_from_embedded_json(user_profile: Dict, top_k: int = 3):
    with open(EMBEDDED_JSON_FILE, "r") as f:
        items = json.load(f)

    color_map = {
        "navy": ["blue", "black"],
        "orange": ["red", "yellow"],
        "grey": ["gray"]
    }

    normalized_color = user_profile["color"].lower()
    expanded_colors = [normalized_color]
    for k, v in color_map.items():
        if normalized_color == k:
            expanded_colors += v

    # --- まず厳しくフィルタリング ---
    filtered_items = [
        item for item in items
        if item["gender"].lower() == user_profile["gender"].lower()
        and any(c in item["baseColour"].lower() for c in expanded_colors)
    ]

    # --- ヒットしなければgenderだけに緩める ---
    if not filtered_items:
        filtered_items = [
            item for item in items
            if item["gender"].lower() == user_profile["gender"].lower()
        ]

    # --- それでもヒットしなければ全商品対象にする ---
    if not filtered_items:
        filtered_items = items

    query_text = (
        f"{user_profile['theme']} fashion for {user_profile['gender']}, "
        f"color: {user_profile['color']}"
    )

    embedding = get_embedding_3small(query_text, st.secrets["OPENAI_API_KEY"])

    all_vectors = np.array([item["embedding"] for item in filtered_items], dtype=np.float32)
    scores = np.dot(all_vectors, embedding) / (
        np.linalg.norm(all_vectors, axis=1) * np.linalg.norm(embedding) + 1e-5
    )

    for i, score in enumerate(scores):
        filtered_items[i]["score"] = score

    top_items = sorted(filtered_items, key=lambda x: x["score"], reverse=True)[:top_k]
    return top_items




# --- GPT Short Recommendation ---

def generate_simple_recommendation(items: List[Dict]):
    item_descriptions = "\n".join([
        f"{item['productDisplayName']} ({item['baseColour']}, {item['season']})"
        for item in items
    ])
    prompt = f"Recommend outfits based on the following items briefly (within 2 lines):\n{item_descriptions}"

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a fashion assistant. Respond very briefly within 2 lines."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

# --- Refinement Profile Update ---

def update_profile_from_feedback(user_profile: Dict, user_feedback: str) -> Dict:
    color_keywords = ["red", "blue", "black", "white", "pink", "yellow", "green", "purple", "grey", "gray", "orange", "navy"]
    body_shape_keywords = ["slim", "curvy", "regular"]
    draw_style_keywords = ["disney", "american comic", "japanese anime", "3d cg"]
    theme_keywords = ["casual", "formal", "street", "vintage", "business", "sporty", "luxury"]

    feedback_lower = user_feedback.lower()

    for color in color_keywords:
        if color in feedback_lower:
            user_profile["color"] = color
            break

    for shape in body_shape_keywords:
        if shape in feedback_lower:
            user_profile["body_shape"] = shape.capitalize()
            break

    for style in draw_style_keywords:
        if style in feedback_lower:
            user_profile["draw_style"] = style.title()
            break

    for theme in theme_keywords:
        if theme in feedback_lower:
            user_profile["theme"] = theme
            break

    return user_profile



# --- UI Layout ---
tab1, tab2 = st.tabs(["🛍️ RetailNext Coordinator", "🌐 Community Gallery"])

with tab1:
    st.subheader("🛍️ RetailNext Coordinator")

    with st.form("✨ Personalize Your Look"):
        uploaded_image = st.file_uploader("Upload your face photo", type=["jpg", "jpeg", "png"])
        country = st.text_input("Country (e.g., USA, Japan, etc.)")
        gender = st.selectbox("Gender", ["Men", "Women", "Other"])
        age = st.slider("Age", 1, 100, 25)
        body_shape = st.selectbox("Body Shape", ["Slim", "Regular", "Curvy"])
        favorite_color = st.text_input("Favorite Color (e.g., black, pink)")
        draw_style = st.selectbox("Drawing Style", ["Disney", "American Comic", "Japanese Anime", "3D CG"])
        fashion_theme = st.text_input("Fashion Theme (e.g., spring, bright)")
        submitted = st.form_submit_button("✨ Generate AI Coordination")

    if submitted and uploaded_image:
        image = Image.open(uploaded_image)
        buffered = BytesIO()
        image.save(buffered, format="PNG")

        # --- DALL-E用プロンプト（露出防止ガード入り） ---
        original_prompt = (
            f"Full-body fashion illustration of a {gender}, age {age}, body shape {body_shape}, "
            f"wearing seasonally appropriate, elegant, modest clothing in {favorite_color} color, themed around {fashion_theme}. "
            "The outfit should cover chest, abdomen, and knees, avoiding revealing skin, and should reflect elegance. "
            f"Style: {draw_style}. White background."
        )

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=original_prompt,
                size="1024x1024",
                quality="standard",
                n=1
            )
            image_url = response.data[0].url

            # セッション保存
            st.session_state["original_prompt"] = original_prompt
            st.session_state["original_image_url"] = image_url
            st.session_state["user_profile"] = {
                "gender": gender,
                "theme": fashion_theme,
                "color": favorite_color,
                "body_shape": body_shape,
                "draw_style": draw_style
            }

            # 画像表示
            st.image(image_url, caption="👕 AI Coordination Suggestion", width=600)
            st.markdown("🔗 **Share on SNS：** [Share on SNS](https://dummy-share-url.com)")
            st.info(
                "📢 The generated coordination will also be shared in the Community Gallery.\n\n"
                "Click on your favorite image to view related recommended items!"
            )

            # 投稿保存
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

            # --- 初回レコメンド表示 ---
            st.subheader("🛒 Your Recommended Items")
            try:
                similar = recommend_from_embedded_json(st.session_state["user_profile"], top_k=3)
                gpt_recommendation = generate_simple_recommendation(similar)
                st.info(gpt_recommendation)

                cols = st.columns(3)
                for i, item in enumerate(similar):
                    with cols[i % 3]:
                        item_image_url = f"https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/{item['id']}.jpg"
                        st.image(item_image_url, width=200)
                        st.markdown(f"**{item['productDisplayName']}**\n{item['gender']}, {item['baseColour']}\n{item['season']} / {item['usage']}")
                        if st.button("🛒 Go to EC Site", key=f"ec_button_{item['id']}"):
                            st.info("This would navigate to the EC site.")
            except Exception as e:
                st.error("Failed to recommend similar items.")
                st.exception(e)

        except Exception as e:
            st.error("Image generation failed")
            st.exception(e)


# --- Refinement Section ---
if "original_prompt" in st.session_state:
    st.markdown("### 🎨 Want to adjust the coordination?")
    user_feedback = st.text_input("💬 Tell us your preference (e.g., 'Make it more casual', 'Use brighter colors')")

    if st.button("🔄 Update Coordination with Your Feedback"):
        if not user_feedback.strip():
            st.warning("⚠️ Please enter your feedback before updating the coordination.")
        else:
            with st.spinner("Updating your coordination..."):
                try:
                    # GPT-4oでリファインプロンプト作成
                    refinement_prompt = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a prompt engineer specializing in improving fashion illustration prompts for DALL-E 3 while maintaining elegance and modesty."},
                            {"role": "user", "content": f"The original prompt was:\n{st.session_state['original_prompt']}\n\nUser feedback is:\n{user_feedback}\n\nPlease refine the prompt while still keeping it modest, elegant, and non-revealing."}
                        ],
                        temperature=0.2
                    ).choices[0].message.content.strip()

                    # DALL-Eでリファイン後画像生成
                    refined_response = client.images.generate(
                        model="dall-e-3",
                        prompt=refinement_prompt,
                        size="1024x1024",
                        quality="standard",
                        n=1
                    )
                    refined_image_url = refined_response.data[0].url

                    # 新しい画像表示
                    st.image(refined_image_url, caption="🎨 Refined AI Coordination Suggestion", width=600)
                    st.success("Here’s your updated coordination based on your feedback!")

                    # --- 🛒 Refinedレコメンド ---
                    st.subheader("🛒 Updated Recommended Items")

                    updated_profile = update_profile_from_feedback(st.session_state["user_profile"].copy(), user_feedback)
                    st.session_state["user_profile"] = updated_profile
                    refined_items = recommend_from_embedded_json(updated_profile, top_k=3)
                    refined_recommendation = generate_simple_recommendation(refined_items)
                    st.info(refined_recommendation)

                    cols = st.columns(3)
                    for i, item in enumerate(refined_items):
                        with cols[i % 3]:
                            item_image_url = f"https://raw.githubusercontent.com/openai/openai-cookbook/main/examples/data/sample_clothes/sample_images/{item['id']}.jpg"
                            st.image(item_image_url, width=200)
                            st.markdown(f"**{item['productDisplayName']}**\n{item['gender']}, {item['baseColour']}\n{item['season']} / {item['usage']}")
                            if st.button("🛒 Go to EC Site", key=f"refined_ec_button_{item['id']}"):
                                st.info("This would navigate to the EC site.")

                except Exception as e:
                    st.error("Failed to update coordination.")
                    st.exception(e)




with tab2:
    posts = load_posts()
    top_posts = sorted(posts, key=lambda x: x["likes"], reverse=True)[:5]

    if top_posts:
        st.subheader("🌟 Popular Coordinations")
        for i, post in enumerate(top_posts):
            with st.container():
                st.markdown(f"### #{i+1} ❤️ {post['likes']} Likes")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(post["image_url"], width=300)
                with col2:
                    st.markdown(f"**🧵 Theme:** {post['theme']}")
                    st.markdown(f"**🌍 Country:** {post['country']}")
                    st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                    st.markdown(f"**💪 Body Shape:** {post['body_shape']} / 🎨 Color: {post['color']}")
                    st.markdown(f"**🎞️ Style:** {post['style']}")

    st.markdown("---")

    st.subheader("🖼️ All Community Looks")
    for post in reversed(posts[:20]):
        with st.container():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(post["image_url"], width=300)
            with col2:
                st.markdown(f"**🧵 Theme:** {post['theme']}")
                st.markdown(f"**🌍 Country:** {post['country']}")
                st.markdown(f"**👤 Gender:** {post['gender']} / 🎂 Age: {post['age']}")
                st.markdown(f"**💪 Body Shape:** {post['body_shape']} / 🎨 Color: {post['color']}")
                st.markdown(f"**🎞️ Style:** {post['style']}")
                st.markdown(f"❤️ {post['likes']} likes")
                if st.button("👍 Like", key=post["id"]):
                    like_post(post["id"])
                    st.rerun()

