import io
import os
import json
import glob
from typing import List, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow / Keras
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as _e:
    TF_AVAILABLE = False
    TF_IMPORT_ERR = str(_e)

st.set_page_config(page_title="CIFAR-100 Keras Demo", page_icon="🧪", layout="centered")
st.title("🧪 CIFAR-100 – Keras Model Canlı Demo")

# CIFAR-100 label names
tags = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle",
    "chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard",
    "lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple","motorcycle","mountain",
    "mouse","mushroom","oak","orange","orchid","otter","palm","pear","pickup_truck","pine",
    "plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel",
    "streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train",
    "trout","tulip","turtle","wardrobe","whale","willow","wolf","woman","worm"
]

MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)

from huggingface_hub import hf_hub_download, list_repo_files

HF_REPO = "misterpy-web/erty2323"
HF_TOKEN = os.environ.get("HF_TOKEN")

@st.cache_data(show_spinner=False, ttl=60)
def list_models() -> List[str]:
    local = sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5")) +
                   glob.glob(os.path.join(MODELS_DIR, "*.keras")))
    local = [os.path.basename(x) for x in local]
    try:
        remote_files = list_repo_files(HF_REPO, repo_type="model", token=HF_TOKEN)
        remote = [f for f in remote_files if f.endswith((".h5", ".keras"))]
    except Exception:
        remote = []
    return sorted(set(local + remote))

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    path = os.path.join(MODELS_DIR, name)
    if not os.path.exists(path):
        path = hf_hub_download(HF_REPO, name, repo_type="model", token=HF_TOKEN, local_dir=MODELS_DIR)
    return tf.keras.models.load_model(path, compile=False)

st.sidebar.header("📦 Model seçimi")
models = list_models()
if not models:
    st.sidebar.error("Hiç model bulunamadı")
    st.stop()
selected = st.sidebar.selectbox("Model dosyası", models)

if not TF_AVAILABLE:
    st.error("TensorFlow yüklü değil. requirements.txt dosyasına `tensorflow==2.20.0` ekleyin.")
    st.stop()

try:
    model = load_model(selected)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

shape = model.input_shape
if isinstance(shape, list):
    shape = shape[0]
try:
    size = int(shape[1])
except Exception:
    size = 32

uploaded = st.file_uploader("Bir görüntü yükleyin", type=["png","jpg","jpeg","bmp","webp"])
if uploaded is None:
    st.info("👆 Bir görüntü yükleyin")
    st.stop()

img = Image.open(io.BytesIO(uploaded.read()))
st.image(img, caption="Yüklenen Görsel", use_container_width=True)

x = img.convert("RGB").resize((size, size))
x = tf.keras.preprocessing.image.img_to_array(x) / 255.0
x = tf.expand_dims(x, 0)

with st.spinner("Tahmin ediliyor..."):
    preds = model.predict(x, verbose=0)
probs = tf.nn.softmax(preds, axis=1).numpy()[0]

idxs = np.argsort(-probs)[:5]
st.subheader("🔮 Tahminler (Top-5)")
for i, idx in enumerate(idxs, 1):
    st.write(f"**{i}. {tags[idx]}** — {probs[idx]:.3f}")

st.success("Tamamlandı ✅")
