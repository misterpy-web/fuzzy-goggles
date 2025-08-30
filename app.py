import io
import os
import json
import glob
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
    TF_IMPORT_ERR = None
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
# Model depolama klasörü (mevcut klasör altında ./models)
# -------------------------------------------------------------
MODELS_DIR = pathlib.Path(os.environ.get("MODELS_DIR", "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------
# Yardımcılar: etiket, boyut, ön-işleme ve indirme rutinleri
# -------------------------------------------------------------
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

# ---------------------- İNDİRME YARDIMCILARI ----------------------
@st.cache_resource(show_spinner=False)
def hf_download(repo_id: str, filename: str, revision: Optional[str] = None) -> str:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision, local_dir=str(MODELS_DIR))
    return path


def parse_hf_resolve_or_blob(url: str) -> Optional[Tuple[str, str, str]]:
    """HF linklerini ayrıştır: return (repo_id, revision, filepath) or None.
    Destek: .../resolve/<rev>/<filepath> veya .../blob/<rev>/<filepath>
    """
    m = re.search(r"huggingface\.co/([^/]+/[^/]+)/(?:(resolve|blob))/([^/]+)/(.+)$", url)
    if not m:
        return None
    repo_id, _rtype, revision, filepath = m.groups()
    return repo_id, revision, filepath


def download_many_hf_links(links_text: str) -> Tuple[List[str], List[Tuple[str, str]]]:
    downloaded, failed = [], []
    try:
        from huggingface_hub import hf_hub_download  # noqa: F401
    except Exception as e:
        return downloaded, [("huggingface_hub import", "huggingface_hub kurulu değil")] 
    for raw in links_text.splitlines():
        url = raw.strip()
        if not url:
            continue
        if "huggingface.co" not in url:
            failed.append((url, "HF linki değil"))
            continue
        parsed = parse_hf_resolve_or_blob(url)
        if not parsed:
            failed.append((url, "Desen çözümlenemedi (resolve/blob)"))
            continue
        repo_id, revision, filepath = parsed
        try:
            path = hf_download(repo_id, filepath, revision)
            downloaded.append(path)
        except Exception as e:
            failed.append((url, str(e)))
    return downloaded, failed


def direct_url_download(url: str, save_as: str) -> str:
    import requests
    dest = MODELS_DIR / save_as
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return str(dest)

# -------------------------------------------------------------
# Sidebar – model kaynağı ve indirme seçenekleri
# -------------------------------------------------------------
FOUND_LOCAL = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])

with st.sidebar:
    st.header("📦 Model Kaynağı")
    source = st.radio("Model kaynağı", ["Local (.h5)", "Hugging Face Hub", "Direct URL", "HF Linkleri (çoklu)"], index=0)

    selected_model_path = ""

    if source == "Local (.h5)":
        if not FOUND_LOCAL:
            st.info("models/ klasörüne .h5/.keras dosyaları koyun veya diğer sekmelerden indirin.")
        else:
            selected_name = st.selectbox("Yerel model", FOUND_LOCAL)
            selected_model_path = str(MODELS_DIR / selected_name)

    elif source == "Hugging Face Hub":
        st.caption("Örn. repo: your-user/cifar100-model · filename: model.h5 · rev: main")
        hf_repo = st.text_input("HF repo id", value="")
        hf_filename = st.text_input("Dosya adı", value="model.h5")
        hf_revision = st.text_input("Revizyon/branch (opsiyonel)", value="")
        if st.button("📥 HF'ten indir"):
            try:
                path = hf_download(hf_repo, hf_filename, hf_revision or None)
                st.success(f"İndirildi: {path}")
                selected_model_path = path
                FOUND_LOCAL = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])
            except Exception as e:
                st.error(f"HF indirme hatası: {e}")

    elif source == "Direct URL":
        url = st.text_input("Model URL (.h5/.keras)", value="")
        url_name = st.text_input("Kaydetme adı", value="model_from_url.h5")
        if st.button("📥 URL'den indir"):
            try:
                path = direct_url_download(url, url_name)
                st.success(f"İndirildi: {path}")
                selected_model_path = path
                FOUND_LOCAL = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])
            except Exception as e:
                st.error(f"URL indirme hatası: {e}")

    elif source == "HF Linkleri (çoklu)":
        st.caption("Her satıra bir HF dosya linki yapıştırın (resolve/blob). Örn: https://huggingface.co/user/repo/resolve/main/model.h5")
        links_text = st.text_area("HF linkleri", height=140, placeholder=(
            "https://huggingface.co/username/repo/resolve/main/model.h5
"
            "https://huggingface.co/username/another-repo/blob/main/weights/model.keras"
        ))
        auto_select_last = st.checkbox("İndirilen son modeli otomatik seç", value=True)
        if st.button("📥 Linkleri indir ve ekle"):
            downloaded, failed = download_many_hf_links(links_text or "")
            if downloaded:
                st.success("İndirilenler:
" + "
".join(downloaded))
                FOUND_LOCAL = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])
                if auto_select_last:
                    selected_model_path = downloaded[-1]
            if failed:
                st.warning("İndirilemeyenler:
" + "
".join(f"- {u} → {err}" for u, err in failed))

    # Ortak ayarlar
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
    with st.expander("Gelişmiş (opsiyonel)"):
        manual_size = st.number_input("Zorla giriş boyutu (0 = otomatik)", min_value=0, max_value=1024, value=0, step=8)
        force_softmax = st.checkbox("Çıkışa softmax uygula (zorla)", value=False, help="Model logit döndürüyorsa kullanın.")
        keep_aspect = st.checkbox("En-boy oranını koru (pad)", value=False)

# -------------------------------------------------------------
# Ana akış
# -------------------------------------------------------------
if not TF_AVAILABLE:
    st.error("TensorFlow yüklü değil. requirements.txt içine `tensorflow==2.15.0.post1` ekleyin.
Ayrıca: huggingface_hub ve requests de gereklidir.")
    st.stop()

if not any([p for p in MODELS_DIR.glob("*.h5")] + [p for p in MODELS_DIR.glob("*.keras")]) and not selected_model_path:
    st.warning("Model bulunamadı. Soldan indirin veya yerel bir model seçin.")
    st.stop()

if not selected_model_path:
    # Local listeden hızlı seçim
    FOUND_LOCAL = sorted([p.name for p in MODELS_DIR.glob("*.h5")] + [p.name for p in MODELS_DIR.glob("*.keras")])
    if FOUND_LOCAL:
        selected_model_path = str(MODELS_DIR / FOUND_LOCAL[0])

if not selected_model_path:
    st.warning("Lütfen bir model seçin.")
    st.stop()

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
st.image(img, caption="Yüklenen Görsel", use_column_width=True)

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
streamlit
pillow
tensorflow==2.15.0.post1
huggingface_hub
requests
```

> Streamlit Cloud kullanıyorsanız Python sürümü için `runtime.txt` içine `3.10` koymanız tavsiye edilir.
    """
)
