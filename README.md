# 🌟 RetailNext Coordinator

**RetailNext Coordinator** is a demo fashion coordination assistant powered by OpenAI.  
It helps users create anime-style outfit images, recommends similar items, and suggests matching products using GPT-4o and RAG.

This app is built using **Streamlit**, **OpenAI API**, and **FAISS**.

---

## 📸 Features

### 1. Upload & Generate AI Fashion
- Upload your face photo
- Input gender, body shape, color, theme, style
- Generates a full-body anime-style outfit using `dall-e-3`

### 2. Similar Items Recommendation
- Extracts image features (RGB vector from center of image)
- Recommends similar clothing samples using cosine similarity

### 3. GPT-Based Product Matching
- Vector search (`text-embedding-3-small`) + FAISS on `products.json`
- GPT-4o selects best items with natural-language explanation

### 4. Community Gallery
- Shows user posts with 💖 likes and rankings

---

## 🚀 How to Deploy on Streamlit Cloud

1. Upload this repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect GitHub and choose this repo
4. Set `app.py` as the main file
5. Add your OpenAI key in `.streamlit/secrets.toml` like this:

```toml
OPENAI_API_KEY = "sk-..."
```

---

## 📁 Project Structure

```bash
retailnext-phase1/
├── app.py                       # Main Streamlit UI
├── gpt_product_recommender.py  # GPT product match logic (optional)
├── products.json               # Product info for GPT matching
├── posts.json                  # Sample posts for gallery
├── image/                      # Uploaded/generated images
├── sample_styles.csv           # Original data reference (optional)
├── requirements.txt            # All required libraries
└── .streamlit/secrets.toml     # API Key (Streamlit Cloud secret)
```

---

## 📦 Requirements

Add these to `requirements.txt`:

```txt
streamlit
openai
faiss-cpu
numpy
pillow
scikit-learn
requests
```

---

## 📊 Tech Stack Overview

| Component         | Tool / Model          |
|------------------|------------------------|
| UI Framework      | Streamlit              |
| Image Generation  | OpenAI `dall-e-3`      |
| Embedding Model   | `text-embedding-3-small` |
| GPT Model         | `gpt-4o`               |
| Vector Search     | FAISS                  |
| Vector Comparison | `cosine_similarity` from sklearn |

---

## 👩‍💼 Use Case (for business proposal)

This can be used as:
- AI fashion concierge prototype
- Retail demo for customer engagement
- Community style trend visualizer

**Built for technical PoC presentation or OpenAI Solution Engineer demo.**

---

## 📅 Last Updated

2025-04-25
