# app.py
import io
import json
from typing import List, Tuple, Optional

import streamlit as st
from PIL import Image

# ====== KERAS / TENSORFLOW ======
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False
# -----------------------------------------------------------------------------

st.set_page_config(page_title="Görüntü Sınıflandırma Demo", page_icon="🖼️", layout="centered")
st.title("🖼️ Görüntü Sınıflandırma – Canlı Demo")

with st.sidebar:
    st.header("⚙️ Ayarlar")
    framework = st.selectbox(
        "Framework",
        [opt for opt in ["PyTorch", "TensorFlow/Keras"] if (opt == "PyTorch" and TORCH_AVAILABLE) or (opt == "TensorFlow/Keras" and TF_AVAILABLE)],
        help="Eğittiğiniz modele göre seçin."
    )
    model_path = st.text_input("Model dosyası yolu", value="model.pt" if framework=="PyTorch" else "model.h5",
                               help="Örn. PyTorch için .pt/.pth, Keras için .h5/.keras")
    labels_path = st.text_input("Sınıf etiketleri (JSON)", value="labels.json",
                                help='JSON içinde ["cat","dog",...] gibi bir liste.')
    input_size = st.number_input("Giriş boyutu (kare)", min_value=64, max_value=1024, value=224, step=16)
    topk = st.number_input("Top-K", min_value=1, max_value=10, value=5, step=1)
    st.caption("Not: Modelinize özel normalize/ön-işleme gerekiyorsa aşağıdaki fonksiyonları düzenleyin.")

uploaded = st.file_uploader("Bir görüntü yükleyin", type=["jpg", "jpeg", "png", "bmp", "webp"])

# ---------- Yardımcılar ----------
@st.cache_data(show_spinner=False)
def load_labels(labels_path: str) -> Optional[List[str]]:
    try:
        with open(labels_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        if isinstance(labels, dict):
            # { "0": "cat", "1": "dog" } -> sıraya göre listele
            labels = [labels[str(i)] for i in range(len(labels))]
        return labels
    except Exception:
        return None

# ====== PyTorch varyantı ======
@st.cache_resource(show_spinner=False)
def load_torch_model(model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.jit.load(model_path, map_location=device) if model_path.endswith(".pt") and not model_path.endswith(".pth") else torch.load(model_path, map_location=device)
    model.eval()
    return model, device

def torch_preprocess(pil: Image.Image, input_size: int):
    # Modelinizin eğitimde kullandığı normalize değerlerine göre düzenleyin
    tfm = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    return tfm(pil).unsqueeze(0)

@torch.no_grad()
def torch_predict_topk(model, device, img_tensor, k: int) -> Tuple[List[int], List[float]]:
    img_tensor = img_tensor.to(device)
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    probs_k, idxs = torch.topk(probs, k=min(k, probs.size(1)), dim=1)
    return idxs[0].cpu().tolist(), probs_k[0].cpu().tolist()

# ====== Keras varyantı ======
@st.cache_resource(show_spinner=False)
def load_keras_model(model_path: str):
    model = tf.keras.models.load_model(model_path, compile=False)
    return model

def keras_preprocess(pil: Image.Image, input_size: int):
    pil = pil.resize((input_size, input_size))
    arr = tf.keras.preprocessing.image.img_to_array(pil)
    # Modelinizin eğitimde kullandığı ölçekleme/normalize adımlarını düzenleyin
    arr = arr / 255.0
    arr = tf.expand_dims(arr, 0)  # (1, H, W, C)
    return arr

def keras_predict_topk(model, arr, k: int) -> Tuple[List[int], List[float]]:
    preds = model.predict(arr, verbose=0)
    if preds.ndim == 1:
        preds = preds[None, :]
    probs = tf.nn.softmax(preds, axis=1).numpy()
    idxs = probs.argsort(axis=1)[:, ::-1][:, :k][0].tolist()
    vals = probs[0, idxs].tolist()
    return idxs, vals

# ---------- Uygulama akışı ----------
labels = load_labels(labels_path)
if labels is None:
    st.info("📄 İsteğe bağlı: `labels.json` dosyası ekleyerek sınıf isimlerini gösterebilirsiniz.")

if uploaded:
    img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    st.image(img, caption="Yüklenen Görsel", use_column_width=True)

    with st.spinner("Model çalışıyor..."):
        if framework == "PyTorch":
            if not TORCH_AVAILABLE:
                st.error("PyTorch bulunamadı. `pip install torch torchvision` deneyin.")
            else:
                try:
                    model, device = load_torch_model(model_path)
                except Exception as e:
                    st.error(f"Model yüklenemedi: {e}")
                else:
                    tensor = torch_preprocess(img, input_size)
                    idxs, probs = torch_predict_topk(model, device, tensor, topk)

        elif framework == "TensorFlow/Keras":
            if not TF_AVAILABLE:
                st.error("TensorFlow/Keras bulunamadı. `pip install tensorflow` deneyin.")
            else:
                try:
                    model = load_keras_model(model_path)
                except Exception as e:
                    st.error(f"Model yüklenemedi: {e}")
                else:
                    arr = keras_preprocess(img, input_size)
                    idxs, probs = keras_predict_topk(model, arr, topk)

    if uploaded and 'idxs' in locals():
        st.subheader("🔮 Tahminler")
        for rank, (i, p) in enumerate(zip(idxs, probs), start=1):
            name = labels[i] if (labels and i < len(labels)) else f"class_{i}"
            st.write(f"**{rank}. {name}** — {p:.3f} olasılık")
        st.success("Tamam!")
else:
    st.write("👈 Soldan model/ayarları girip bir görüntü yükleyin.")
