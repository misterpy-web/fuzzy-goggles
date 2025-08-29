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
# Kaynak seçenekleri: Local / Hugging Face / Direct URL
# -------------------------------------------------------------
import pathlib
MODELS_DIR = pathlib.Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FOUND_MODELS = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])

with st.sidebar:
    st.header("📦 Model Kaynağı")
    source = st.radio("Model kaynağı", ["Local (.h5)", "Hugging Face Hub", "Direct URL"], index=0)

    selected_model_path = ""
    if source == "Local (.h5)":
        if not FOUND_MODELS:
            st.info("models/ klasörüne .h5/.keras dosyaları koyun.")
        else:
            selected_name = st.selectbox("Yerel model", FOUND_MODELS)
            selected_model_path = str(MODELS_DIR / selected_name)

    elif source == "Hugging Face Hub":
        st.caption("Örn. repo: your-username/cifar100-model, filename: model.h5")
        hf_repo = st.text_input("HF repo id", value="")
        hf_filename = st.text_input("Dosya adı", value="model.h5")
        hf_revision = st.text_input("Revizyon/branch (opsiyonel)", value="")
        if st.button("📥 Hugging Face'ten indir"):
            try:
                from huggingface_hub import hf_hub_download
                path = hf_hub_download(repo_id=hf_repo, filename=hf_filename, revision=(hf_revision or None), local_dir=str(MODELS_DIR))
                st.success(f"İndirildi: {path}")
                selected_model_path = path
            except Exception as e:
                st.error(f"HF indirme hatası: {e}")

    elif source == "Direct URL":
        url = st.text_input("Model URL (.h5/.keras)", value="")
        url_name = st.text_input("Kaydetme adı", value="model_from_url.h5")
        if st.button("📥 URL'den indir"):
            try:
                import requests
                dest = MODELS_DIR / url_name
                r = requests.get(url, timeout=60)
                r.raise_for_status()
                dest.write_bytes(r.content)
                st.success(f"İndirildi: {dest}")
                selected_model_path = str(dest)
            except Exception as e:
                st.error(f"URL indirme hatası: {e}")

    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)

    with st.expander("Gelişmiş (opsiyonel)"):
        manual_size = st.number_input("Zorla giriş boyutu (0 = otomatik)", min_value=0, max_value=1024, value=0, step=8)
        force_softmax = st.checkbox("Çıkışa softmax uygula (zorla)", value=False,
                                    help="Model logit döndürüyorsa aktif edin.")
        keep_aspect = st.checkbox("En-boy oranını koru (pad)", value=False)

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
    """Modelin giriş boyutunu (kare) otomatik çıkar."""
    shape = model.input_shape
    # Çoklu input ise ilkini al
    if isinstance(shape, list):
        shape = shape[0]
    # (None, H, W, C) bekleriz (channels_last)
    try:
        h, w = int(shape[1]), int(shape[2])
        if h > 0 and w > 0:
            return h if h == w else max(h, w)
    except Exception:
        pass
    # Varsayılan: 32 (CIFAR)
    return 32


def find_labels_for_model(model_path: str) -> List[str]:
    """Aynı isimli labels.json varsa onu kullan, yoksa CIFAR-100 fine."""
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


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """En-boy oranını koruyarak pad'li kare resim oluşturur."""
    w, h = img.size
    scale = min(size / w, size / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = img.resize((nw, nh))
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(resized, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def preprocess(pil: Image.Image, size: int, keep_ratio: bool) -> tf.Tensor:
    if keep_ratio:
        pil = letterbox(pil.convert("RGB"), size)
    else:
        pil = pil.convert("RGB").resize((size, size))
    arr = tf.keras.preprocessing.image.img_to_array(pil)
    # Otomatik ölçekleme: Eğer modelde Rescaling katmanı varsa ona bırakırız.
    # Yoksa 0-1 aralığına ölçekleriz.
    arr = arr / 255.0
    return tf.expand_dims(arr, 0)  # (1, H, W, C)


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
    if force_softmax or not model_has_softmax(model):
        preds = tf.nn.softmax(preds, axis=1).numpy()
    return preds


def topk_from_probs(probs: np.ndarray, k: int) -> Tuple[List[int], List[float]]:
    k = int(min(k, probs.shape[1]))
    idxs = np.argsort(-probs, axis=1)[0][:k].tolist()
    vals = probs[0, idxs].astype(float).tolist()
    return idxs, vals

# -------------------------------------------------------------
# Ana akış – resim yükleme ALANI ANA SAYFADA!
# -------------------------------------------------------------
uploaded = st.file_uploader("Bir görüntü yükleyin", type=["png","jpg","jpeg","bmp","webp"], accept_multiple_files=False)

if not selected_model_path:
    st.warning("Sol taraftan bir model seçin veya tam yol girin.")
    st.stop()

if not TF_AVAILABLE:
    st.error("TensorFlow yüklü değil. Aşağıdaki talimatlarla kurun ve tekrar deneyin.

`requirements.txt` içeriği örneği:

```
streamlit
pillow
tensorflow==2.15.0.post1
```

`runtime.txt` (Streamlit Cloud için Python sürümü sabitleme):

````
3.10
````

> Alternatif: `tensorflow-cpu==2.15.0.post1` de kullanılabilir.

Hata: " + (TF_IMPORT_ERR or "unknown"))
    st.stop()

try:
    model = load_keras_model(selected_model_path)
except Exception as e:
    st.error(f"Model yüklenemedi: {e}")
    st.stop()

labels = find_labels_for_model(selected_model_path)

# Giriş boyutunu otomatik belirle (isteğe bağlı manuel override)
auto_size = infer_input_size(model)
INPUT_SIZE = manual_size if manual_size > 0 else auto_size
st.caption(f"Giriş boyutu: {INPUT_SIZE}px (auto)")

if uploaded is None:
    st.info("👆 Bir görüntü seçtiğinizde tahmin yapılacaktır.")
    st.stop()

# Görseli göster
img = Image.open(io.BytesIO(uploaded.read()))
st.image(img, caption="Yüklenen Görsel", use_column_width=True)

# Ön-işleme ve tahmin
x = preprocess(img, INPUT_SIZE, keep_aspect)
with st.spinner("Tahmin ediliyor..."):
    preds = model.predict(x, verbose=0)
probs = to_probabilities(model, preds)

# Top-K sonuçlar
idxs, vals = topk_from_probs(probs, topk)

st.subheader(f"🔮 Tahminler (Top-{topk})")
for r, (i, p) in enumerate(zip(idxs, vals), start=1):
    name = labels[i] if 0 <= i < len(labels) else f"class_{i}"
    st.write(f"**{r}. {name}** — {p:.3f}")

st.success("Tamamlandı ✅")

st.markdown("---")
st.markdown(
    """
**Kullanım:**
1. `models/` klasörüne `.h5` / `.keras` dosyalarınızı koyun.
2. Gerekirse aynı klasöre `labels.json` (veya `model.labels.json`) ekleyin; yoksa **CIFAR-100 fine** etiketleri kullanılır.
3. Uygulama giriş boyutunu **otomatik** algılar (model.input_shape). İsterseniz Gelişmiş bölümünden override edebilirsiniz.
    """
)
