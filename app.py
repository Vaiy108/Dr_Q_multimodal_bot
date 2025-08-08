
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from PIL import Image
import requests
from io import BytesIO
import openai
import os

# Load models
text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load 3MDBench dataset
dataset = load_dataset("univanxx/3mdbench", split="train")

# Precompute multimodal embeddings
@st.cache_resource(show_spinner=True)
def compute_embeddings():
    embeddings = []
    for entry in dataset.select(range(50)):  # limit for performance
        text = entry['description']
        text_emb = text_encoder.encode(text, convert_to_tensor=True)

        try:
            response = requests.get(entry['image'], timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            inputs = clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_emb = clip_model.get_image_features(**inputs).squeeze()
        except:
            image_emb = torch.zeros(512)

        combined = torch.cat((text_emb, image_emb), dim=0)
        embeddings.append((combined, entry))
    return embeddings

multimodal_data = compute_embeddings()

# OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI
st.title("🩺 Multimodal Medical Chatbot")
user_query = st.text_input("Describe your symptoms or ask a medical question:")

if user_query:
    query_emb = text_encoder.encode(user_query, convert_to_tensor=True)
    fake_image_emb = torch.zeros(512)
    query_combined = torch.cat((query_emb, fake_image_emb), dim=0)

    # Find best match
    similarities = [util.pytorch_cos_sim(query_combined, item[0]) for item in multimodal_data]
    best_idx = torch.argmax(torch.tensor(similarities)).item()
    best_match = multimodal_data[best_idx][1]

    # Show image + text
    st.subheader("Matched Case")
    st.image(best_match['image'], caption="Medical Image")
    st.write("**Description:**", best_match['description'])
    st.write("**Diagnosis:**", best_match['diagnosis'])

    # Translate or enrich with GPT
    with st.spinner("Generating medical explanation..."):
        prompt = f"The patient case involves: {best_match['description']}. Diagnosis: {best_match['diagnosis']}. Explain this in simple terms."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful medical assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        reply = response['choices'][0]['message']['content']
        st.success("AI Explanation:")
        st.write(reply)

st.caption("This chatbot is for educational purposes only and does not provide medical advice.")
