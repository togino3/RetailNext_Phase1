
import json
import numpy as np
import faiss
from openai import OpenAI
from typing import List, Dict
import streamlit as st

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def embed_product_text(product: Dict) -> List[float]:
    text = f"{product['name']}. {product['description']}. Color: {product['color']}. Style: {product['style']}."
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
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
        user_msg += f"{i+1}. {item['name']} - {item['description']}
"

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
    return generate_recommendation(user_profile, matched)
