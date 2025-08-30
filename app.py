import io
import os
import json
import glob
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# TensorFlow / Keras (kurulu değilse uygulama çökmemesi için korumalı import)
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_IMPORT_ERR = None
except Exception as _e:
    TF_AVAILABLE = False
    TF_IMPORT_ERR = str(_e)

st.set_page_config(page_title="CIFAR-100 Keras (H5) Demo", page_icon="🧪", layout="centered")
st.title("🧪 CIFAR-100 – Keras .h5 Model Canlı Demo")

# -------------------------------------------------------------
# CIFAR-100 fine label names (index order 0..99) – yedek olarak kullanılır
# -------------------------------------------------------------
CIFAR100_FINE = [
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

# -------------------------------------------------------------
# Otomatik model bulma: hem Hugging Face hem de ./models klasöründen
# -------------------------------------------------------------
MODELS_DIR = os.environ.get("MODELS_DIR", "models")
os.makedirs(MODELS_DIR, exist_ok=True)
HF_REPO = "misterpy-web/erty2323"

@st.cache_data(show_spinner=False, ttl=60)
def list_hf_files() -> List[str]:
    """Hugging Face repo içindeki .h5/.keras dosyalarını listeler (private/public)."""
    try:
        from huggingface_hub import HfApi
        token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=HF_REPO)
        return [f for f in files if f.lower().endswith((".h5", ".keras"))]
    except Exception as e:
        st.warning(f"HF listelenemedi: {e}")
        return []

@st.cache_resource(show_spinner=False)
def download_hf_model(filename: str) -> str:
    """HF'den dosyayı models/ altına indir ve yerel yolu döndür (private/public)."""
    from huggingface_hub import hf_hub_download
    token = st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    path = hf_hub_download(repo_id=HF_REPO, filename=filename, local_dir=MODELS_DIR, token=token)
    return path

def available_models() -> List[str]:
    local_files = sorted(glob.glob(os.path.join(MODELS_DIR, "*.h5")) +
                         glob.glob(os.path.join(MODELS_DIR, "*.keras")))
    local_names = {os.path.basename(f): f for f in local_files}
    hf_files = list_hf_files()
    for f in hf_files:
        if f not in local_names:
            try:
                local_path = download_hf_model(f)
                local_names[f] = local_path
            except Exception as e:
                st.error(f"{f} indirilemedi: {e}")
    return sorted(local_names.values())

with st.sidebar:
    st.header("📦 Model seçimi")
    FOUND_MODELS = available_models()
    if not FOUND_MODELS:
        st.info("models/ klasörüne .h5 veya .keras dosyaları koyun veya HF reposuna yükleyin.")
        selected_model_path = ""
    else:
        selected_name = st.selectbox("Model (.h5/.keras)", [os.path.basename(p) for p in FOUND_MODELS])
        selected_model_path = [p for p in FOUND_MODELS if os.path.basename(p) == selected_name][0]

    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

# -------------------------------------------------------------
# Yardımcılar
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    if not path:
        raise FileNotFoundError("Model yolu boş.")
    model = tf.keras.models.load_model(path, compile=False)
    return model

def infer_input_size(model) -> int:
    shape = model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    try:
        h, w = int(shape[1]), int(shape[2])
        if h > 0 and w > 0:
            return h if h == w else max(h, w)
    except Exception:
        pass
    return 32

def find_labels_for_model(model_path: str) -> List[str]:
    stem = os.path.splitext(model_path)[0]
    for candidate in [stem + ".labels.json", stem + "_labels.json", stem + ".json", os.path.join(os.path.dirname(model_path), "labels.json")]:
        if os.path.exists(candidate):
            try:
                with open(candidate, "r", encoding="utf-8") as f:
                    labels = json.load(f)
                if isinstance(labels, dict):
                    labels = [labels[str(i)] for i in range(len(labels))]
                if isinstance(labels, list) and len(labels) > 0:
                    return labels
            except Exception:
                pass
    return CIFAR100_FINE

def preprocess(pil: Image.Image, size: int) -> tf.Tensor:
    pil = pil.convert("RGB").resize((size, size))
    arr = tf.keras.preprocessing.image.img_to_array(pil)
    arr = arr / 255.0
    return tf.expand_dims(arr, 0)

def model_has_softmax(model) -> bool:
    try:
        last = model.layers[-1]
        act = getattr(last, "activation", None)
        if act is None:
            return False
        return act.__name__ == "softmax"
    except Exception:
        return False

def to_probabilities(model, preds: np.ndarray) -> np.ndarray:
    if preds.ndim == 1:
        preds = preds[None, :]
    if not model_has_softmax(model):
        preds = tf.nn.softmax(preds, axis=1).numpy()
    return preds

def topk_from_probs(probs: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    k = int(min(k, probs.shape[1]))
    idxs = np.argsort(-probs, axis=1)[0][:k].tolist()
    vals = probs[0, idxs].astype(float).tolist()
    return idxs, vals

# -------------------------------------------------------------
# Ana akış – resim yükleme
# -------------------------------------------------------------
uploaded = st.file_uploader("Bir görüntü yükleyin", type=["png","jpg","jpeg","bmp","webp"], accept_multiple_files=False)

if not selected_model_path:
    st.warning("Sol taraftan bir model seçin veya HF'ye yükleyin.")
    st.stop()

if not TF_AVAILABLE:
    st.error("TensorFlow yüklü değil. requirements.txt ile ekleyin.")
    st.stop()

try:
    model = load_keras_model(selected_model_path)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

labels = find_labels_for_model(selected_model_path)

auto_size = infer_input_size(model)
INPUT_SIZE = auto_size
st.caption(f"Giriş boyutu: {INPUT_SIZE}px (auto)")

if uploaded is None:
    st.info("👆 Bir görüntü seçtiğinizde tahmin yapılacaktır.")
    st.stop()

img = Image.open(io.BytesIO(uploaded.read()))
st.image(img, caption="Yüklenen Görsel", use_container_width=True)

x = preprocess(img, INPUT_SIZE)
with st.spinner("Tahmin ediliyor..."):
    preds = model.predict(x, verbose=0)
probs = to_probabilities(model, preds)

idxs, vals = topk_from_probs(probs, topk)

st.subheader(f"🔮 Tahminler (Top-{topk})")
for r, (i, p) in enumerate(zip(idxs, vals), start=1):
    name = labels[i] if 0 <= i < len(labels) else f"class_{i}"
    st.write(f"**{r}. {name}** — {p:.3f}")

st.success("Tamamlandı ✅")
#st.markdown("---")


