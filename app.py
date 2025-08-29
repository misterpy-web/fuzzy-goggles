import io
import pickle
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="CIFAR-100 Pickle Demo", page_icon="🧪", layout="centered")
st.title("🧪 CIFAR-100 Pickle Model – Canlı Demo")

# -------------------------------------------------------------
# CIFAR-100 fine label names (index order 0..99)
# Source: CIFAR-100 dataset class order (fine labels)
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

# -------------------- Sidebar Controls ----------------------
with st.sidebar:
    st.header("⚙️ Ayarlar")
    model_path = st.text_input("Pickle model dosyası", value="model.pkl")
    input_size = st.number_input("Giriş boyutu (kare)", min_value=16, max_value=256, value=32, step=2,
                                 help="CIFAR-100 32x32'dir; farklı eğittiyseniz değiştirin.")
    normalize_01 = st.checkbox("0-1 aralığına ölçekle (/255)", value=True)
    use_cifar_stats = st.checkbox("CIFAR-100 mean/std ile standartlaştır", value=False,
                                  help="mean=[0.5071,0.4867,0.4408], std=[0.2675,0.2565,0.2761]")
    channel_order = st.selectbox("Kanal sırası", options=["RGB", "BGR"], index=0)
    flatten = st.checkbox("Düzle (1xH*W*C)", value=True, help="Çoğu sklearn modeli 2D input ister.")
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
    label_source = st.selectbox("Sınıf etiketleri", ["CIFAR-100 (fine)", "Özel JSON yolundan oku"], index=0)
    custom_labels_path = st.text_input("Özel labels.json (opsiyonel)", value="", help="[\"class0\",...]")

# -------------------- Helpers -------------------------------
@st.cache_resource(show_spinner=False)
def load_pickle_model(path: str):
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

@st.cache_data(show_spinner=False)
def load_custom_labels(path: str) -> Optional[List[str]]:
    if not path:
        return None
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, dict):
            labels = [labels[str(i)] for i in range(len(labels))]
        return labels
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def get_labels() -> List[str]:
    if label_source == "CIFAR-100 (fine)":
        return CIFAR100_FINE
    lbls = load_custom_labels(custom_labels_path)
    return lbls if lbls else CIFAR100_FINE


def preprocess_image(pil_img: Image.Image, size: int, normalize: bool, standardize: bool, order: str, flatten_: bool) -> np.ndarray:
    img = pil_img.convert("RGB").resize((size, size))
    arr = np.asarray(img, dtype=np.float32)
    if normalize:
        arr = arr / 255.0
    if standardize:
        mean = np.array([0.5071, 0.4867, 0.4408], dtype=np.float32)
        std = np.array([0.2675, 0.2565, 0.2761], dtype=np.float32)
        arr = (arr - mean) / std
    if order == "BGR":
        arr = arr[..., ::-1]
    if flatten_:
        arr = arr.reshape(1, -1)
    else:
        arr = np.expand_dims(arr, 0)  # (1,H,W,C)
    return arr


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return e / np.s
