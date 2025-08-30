import io
import os
import json
import pathlib
import re
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
from PIL import Image

# -------------------------------------------------------------
# TensorFlow / Keras (korumalı import)
# -------------------------------------------------------------
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as _e:
    TF_AVAILABLE = False
    TF_IMPORT_ERR = str(_e)

st.set_page_config(page_title="CIFAR-100 Keras (H5) Demo", page_icon="🧪", layout="centered")
st.title("🧪 CIFAR-100 – Keras Model Canlı Demo")

# -------------------------------------------------------------
# CIFAR-100 fine label names (yedek)
# -------------------------------------------------------------
CIFAR100_FINE = [
    "apple","aquarium_fish","baby","bear","beaver","bed","bee","beetle","bicycle","bottle",
    "bowl","boy","bridge","bus","butterfly","camel","can","castle","caterpillar","cattle","chair","chimpanzee","clock","cloud","cockroach","couch","crab","crocodile","cup","dinosaur",
    "dolphin","elephant","flatfish","forest","fox","girl","hamster","house","kangaroo","keyboard","lamp","lawn_mower","leopard","lion","lizard","lobster","man","maple","motorcycle","mountain",
    "mouse","mushroom","oak","orange","orchid","otter","palm","pear","pickup_truck","pine","plain","plate","poppy","porcupine","possum","rabbit","raccoon","ray","road","rocket",
    "rose","sea","seal","shark","shrew","skunk","skyscraper","snail","snake","spider","squirrel","streetcar","sunflower","sweet_pepper","table","tank","telephone","television","tiger","tractor","train",
    "trout","tulip","turtle","wardrobe","whale","willow","wolf","woman","worm"
]

# -------------------------------------------------------------
# Ayarlar: SADECE BU HUGGING FACE REPO İÇİN ÇEK (sabitle)
# -------------------------------------------------------------
HF_REPO = "misterpy-web/erty2323"  # 🔒 sadece bu repo
MODELS_DIR = pathlib.Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Yardımcılar
# -------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def list_hf_files(repo_id: str) -> List[str]:
    from huggingface_hub import HfApi
    api = HfApi()
    files = api.list_repo_files(repo_id=repo_id)
    return [f for f in files if f.lower().endswith((".h5", ".keras"))]


@st.cache_resource(show_spinner=False)
def hf_download(repo_id: str, filename: str, revision: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=str(MODELS_DIR))
    return path


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
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
    candidates = [
        stem + ".labels.json",
        stem + "_labels.json",
        stem + ".json",
        os.path.join(os.path.dirname(model_path), "labels.json"),
        "labels.json",
    ]
    for candidate in candidates:
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


def letterbox(img: Image.Image, size: int) -> Image.Image:
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(pil: Image.Image, size: int, keep_ratio: bool) -> tf.Tensor:
    pil = pil.convert("RGB")
    pil = letterbox(pil, size) if keep_ratio else pil.resize((size, size))
    arr = tf.keras.preprocessing.image.img_to_array(pil)
    arr = arr / 255.0
    return tf.expand_dims(arr, 0)


def model_has_softmax(model) -> bool:
    try:
        last = model.layers[-1]
        act = getattr(last, "activation", None)
        return (act is not None) and (act.__name__ == "softmax")
    except Exception:
        return False


def to_probabilities(model, preds: np.ndarray, force_softmax: bool) -> np.ndarray:
    if preds.ndim == 1:
        preds = preds[None, :]
    if force_softmax or not model_has_softmax(model):
        preds = tf.nn.softmax(preds, axis=1).numpy()
    return preds


def topk_from_probs(probs: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    k = int(min(k, probs.shape[1]))
    idxs = np.argsort(-probs, axis=1)[0][:k].tolist()
    vals = probs[0, idxs].astype(float).tolist()
    return idxs, vals

# -------------------------------------------------------------
# Sidebar – Tek menü: HF (sabit repo) + Yerel (GitHub/models) birleştirilmiş
# -------------------------------------------------------------
with st.sidebar:
    st.header("📦 Model seçimi")

    if not TF_AVAILABLE:
        st.error(
            """TensorFlow yüklü değil veya bu Python sürümüyle uyumlu değil.


requirements.txt örneği:

```
streamlit==1.49.1
pillow
tensorflow==2.20.0
huggingface_hub
requests
```
"""
        )
        st.stop()

    # 1) Yerel modelleri tara
    local_files = [p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")]

    # 2) HF repo dosyalarını al (tam pathlerle)
    try:
        hf_paths = list_hf_files(HF_REPO)  # örn: ["weights/model.h5", "model.keras", ...]
    except Exception as e:
        hf_paths = []
        st.warning(f"HF listelenemedi: {e}")

    # 3) Tek bir dropdown için görünür isimleri oluştur
    #    Çakışmaları çözmek için aynı isim iki kaynaktan geliyorsa sonuna kaynak etiketi ekle
    from collections import Counter
    local_basenames = local_files
    hf_basenames = [os.path.basename(p) for p in hf_paths]
    counts = Counter(local_basenames + hf_basenames)

    options = []  # (display, source, identifier)
    for name in sorted(local_basenames):
        display = name if counts[name] == 1 else f"{name} (local)"
        options.append((display, "local", name))
    for full in sorted(hf_paths):
        base = os.path.basename(full)
        display = base if counts[base] == 1 else f"{base} (hf:{full})"
        options.append((display, "hf", full))

    if not options:
        st.info("Henüz model yok. HF repodan indirilecek dosyalar otomatik menüye eklenecek.")
        st.stop()

    display_labels = [d for (d, _, _) in options]
    choice = st.selectbox("Model dosyası", display_labels)
    chosen = next((tpl for tpl in options if tpl[0] == choice), None)

    # Seçime göre yol belirle: yerelde varsa direkt kullan; HF ise yoksa indir, varsa cache'ten kullan
    selected_model_path = ""
    if chosen is not None:
        _disp, src, ident = chosen
        if src == "local":
            selected_model_path = str(MODELS_DIR / ident)
        else:
            # ident: HF içindeki repo-relatif yol
            local_target = MODELS_DIR / os.path.basename(ident)
            if local_target.exists():
                selected_model_path = str(local_target)
            else:
                try:
                    selected_model_path = hf_download(HF_REPO, ident, None)
                except Exception as e:
                    st.error(f"Model indirilemedi: {e}")
                    st.stop()

    # Ortak ayarlar
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
    with st.expander("Gelişmiş (opsiyonel)"):
        manual_size = st.number_input("Zorla giriş boyutu (0 = otomatik)", min_value=0, max_value=1024, value=0, step=8)
        force_softmax = st.checkbox("Çıkışa softmax uygula (zorla)", value=False)
        keep_aspect = st.checkbox("En-boy oranını koru (pad)", value=False)

# -------------------------------------------------------------
# Ana akış
# -------------------------------------------------------------
# Modeli indir
try:
    selected_model_path = hf_download(HF_REPO, chosen_file, rev or None) if auto_download else str(MODELS_DIR / pathlib.Path(chosen_file).name)
except Exception as e:
    st.error(f"Model indirilemedi: {e}")
    st.stop()

# Yükle
try:
    model = load_keras_model(selected_model_path)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

labels = find_labels_for_model(selected_model_path)
INPUT_SIZE = manual_size if manual_size > 0 else infer_input_size(model)
st.caption(f"Giriş boyutu: {INPUT_SIZE}px")

uploaded = st.file_uploader("Bir görüntü yükleyin", type=["png","jpg","jpeg","bmp","webp"])

if uploaded is None:
    st.info("👆 Bir görsel yükleyin.")
    st.stop()

img = Image.open(io.BytesIO(uploaded.read()))
st.image(img, caption="Yüklenen Görsel", use_container_width=True)

x = preprocess(img, INPUT_SIZE, keep_aspect)
with st.spinner("Tahmin ediliyor..."):
    preds = model.predict(x, verbose=0)

probs = to_probabilities(model, preds, force_softmax)
idxs, vals = topk_from_probs(probs, topk)

st.subheader(f"🔮 Tahminler (Top-{topk})")
for r, (i, p) in enumerate(zip(idxs, vals), start=1):
    name = labels[i] if 0 <= i < len(labels) else f"class_{i}"
    st.write(f"**{r}. {name}** — {p:.3f}")

st.success("Tamamlandı ✅")

st.markdown("---")
st.markdown(
    """
**requirements.txt** önerisi:
```
streamlit==1.49.1
pillow
tensorflow==2.20.0
huggingface_hub
requests
```
> Streamlit Cloud Python 3.13 ile uyum için TF 2.20.0 kullanın.
    """
)
