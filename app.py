# ================================
#   Cache-Safe Multimodal App
# ================================

import os

# ====== Force all cache dirs to /tmp (writable in most environments) ======
CACHE_BASE = "/tmp/cache"
os.environ["HF_HOME"] = f"{CACHE_BASE}/hf_home"
os.environ["TRANSFORMERS_CACHE"] = f"{CACHE_BASE}/transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = f"{CACHE_BASE}/sentence_transformers"
os.environ["HF_DATASETS_CACHE"] = f"{CACHE_BASE}/hf_datasets"
os.environ["TORCH_HOME"] = f"{CACHE_BASE}/torch"
os.environ["STREAMLIT_CACHE_DIR"] = f"{CACHE_BASE}/streamlit_cache"
os.environ["STREAMLIT_STATIC_DIR"] = f"{CACHE_BASE}/streamlit_static"

# Create the directories before imports
for path in os.environ.values():
    if path.startswith(CACHE_BASE):
        os.makedirs(path, exist_ok=True)

# ====== Imports ======
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset, get_dataset_split_names
from PIL import Image
import openai
import comet_llm
from opik import track

os.environ["STREAMLIT_CONFIG_DIR"] = "/tmp/.streamlit"
os.environ["STREAMLIT_CACHE_DIR"] = f"{CACHE_BASE}/streamlit_cache"
os.environ["STREAMLIT_STATIC_DIR"] = f"{CACHE_BASE}/streamlit_static"

os.makedirs("/tmp/.streamlit", exist_ok=True)


# ========== API Key ==========
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPIK_API_KEY"] = os.getenv("OPIK_API_KEY")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK_WORKSPACE")
# ========== Load Models ==========
@st.cache_resource(show_spinner=False)
def load_models():
    clip_model = CLIPModel.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    clip_processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32",
        cache_dir=os.environ["TRANSFORMERS_CACHE"]
    )
    text_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder=os.environ["SENTENCE_TRANSFORMERS_HOME"]
    )
    return clip_model, clip_processor, text_model

clip_model, clip_processor, text_model = load_models()

# ========== Load Dataset ==========
@st.cache_resource(show_spinner=False)
def load_medical_data():
    available_splits = get_dataset_split_names("univanxx/3mdbench")
    split_to_use = "train" if "train" in available_splits else available_splits[0]
    dataset = load_dataset(
        "univanxx/3mdbench",
        split=split_to_use,
        cache_dir=os.environ["HF_DATASETS_CACHE"]
    )
    return dataset

data = load_medical_data()

from openai import OpenAI
client = OpenAI(api_key=openai.api_key)
# Temporary debug display
#st.write("Dataset columns:", data.features.keys())

# After seeing the real column name, let's say it's "text" instead of "description":
text_field = "text" if "text" in data.features else list(data.features.keys())[0]


@st.cache_data(show_spinner=False)
def prepare_combined_texts(_dataset):
    combined = []
    for gc, c in zip(_dataset["general_complaint"], _dataset["complaints"]):
        gc_str = gc if gc else ""
        c_str = c if c else ""
        combined.append(f"General complaint: {gc_str}. Additional details: {c_str}")
    return combined

combined_texts = prepare_combined_texts(data)

# Then use dynamic access:
#text_embeddings = embed_texts(data[text_field])

# ========== 🧠 Embedding Function ==========
@st.cache_data(show_spinner=False)
def embed_dataset_texts(_texts):
    return text_model.encode(_texts, convert_to_tensor=True)

def embed_query_text(query):
    return text_model.encode([query], convert_to_tensor=True)[0]

# Pick which text column to use
TEXT_COLUMN = "complaints"  # or "general_complaint", depending on your needs

# ========== 🧑‍⚕️ App UI ==========
st.title("🩺 Dr_Q_bot - Multimodal Medical Chatbot")

query = st.text_input("Enter your medical question or symptom description:")
uploaded_file = st.file_uploader("Upload an image to find similar medical cases:", type=["png", "jpg", "jpeg"])

# Add author info in the sidebar
with st.sidebar:
    st.markdown("## 👤👤Authors")
    st.markdown("**Vasan Iyer**")
    st.markdown("**Eric J Giacomucci**")
    st.markdown("[GitHub](https://github.com/Vaiy108)")
    st.markdown("[LinkedIn](https://linkedin.com/in/vasan-iyer)") 

@track
def get_chat_completion_openai(client, prompt: str):
    return client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4" if you need the older GPT-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=150
    )

@track
def get_similar_prompt(query):
    text_embeddings = embed_dataset_texts(combined_texts)  # cached
    query_embedding = embed_query_text(query)  # recalculated each time

    cos_scores = util.pytorch_cos_sim(query_embedding, text_embeddings)[0]
    top_result = torch.topk(cos_scores, k=1)
    idx = top_result.indices[0].item()
    return data[idx]

# Cache dataset image embeddings (takes time, so cached)
@st.cache_data(show_spinner=True)
def embed_dataset_images(_dataset):
    features = []
    for item in _dataset:
        # Load image from URL/path or raw bytes - adapt this if needed
        img = item["image"]
        inputs = clip_processor(images=img, return_tensors="pt")
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs)
        feat /= feat.norm(p=2, dim=-1, keepdim=True)
        features.append(feat.cpu())
    return torch.cat(features, dim=0)

dataset_image_features = embed_dataset_images(data)

#if query:
if st.button("Submit") and query:
    with st.spinner("Searching medical cases..."):


        # Compute similarity
        selected = get_similar_prompt(query)

        # Show Image
        st.image(selected['image'], caption="Most relevant medical image", use_container_width=True)

        # Show Text
        st.markdown(f"**Case Description:** {selected[TEXT_COLUMN]}")

        # GPT Explanation
        if openai.api_key:
            prompt = f"Explain this case in plain English: {selected[TEXT_COLUMN]}"

            explanation = get_chat_completion_openai(client, prompt)
            explanation = explanation.choices[0].message.content

            st.markdown(f"### 🤖 Explanation by GPT:\n{explanation}")
        else:
            st.warning("OpenAI API key not found. Please set OPENAI_API_KEY as a secret environment variable.")

if uploaded_file is not None:
    print('uploading file')
    print(uploaded_file)
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Your uploaded image", use_container_width=True)

    # Embed uploaded image
    inputs = clip_processor(images=query_image, return_tensors="pt")
    with torch.no_grad():
        query_feat = clip_model.get_image_features(**inputs)
    query_feat /= query_feat.norm(p=2, dim=-1, keepdim=True)

    # Compute cosine similarity
    similarities = (dataset_image_features @ query_feat.T).squeeze(1)  # [num_dataset_images]

    top_k = 3
    top_results = torch.topk(similarities, k=top_k)

    st.write(f"Top {top_k} similar medical cases:")

    for rank, idx in enumerate(top_results.indices):
        score = top_results.values[rank].item()
        similar_img = data[int(idx)]['image']
        st.image(similar_img, caption=f"Similarity: {score:.3f}", use_container_width=True)
        st.markdown(f"**Case description:** {data[int(idx)]['complaints']}")
else:
    print("no image")

st.caption("This chatbot is for educational purposes only and does not provide medical advice.")
