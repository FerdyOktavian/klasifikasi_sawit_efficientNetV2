import os
import tempfile

import streamlit as st
from PIL import Image

from predict import load_sawit_model, predict_image


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Klasifikasi Kematangan Buah Sawit",
    page_icon="🌴",
    layout="centered"
)


# =========================
# CSS
# =========================
st.markdown("""
<style>
/* Lebarin area utama di HP */
.block-container {
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

/* Semua tombol Streamlit dibuat lebar */
div[data-testid="stButton"] button {
    width: 100% !important;
    min-height: 48px !important;
    font-size: 16px !important;
}

/* Area kamera */
[data-testid="stCameraInput"] {
    width: 100% !important;
}

/* Wrapper utama kamera */
[data-testid="stCameraInput"] > div {
    width: 100% !important;
}

/* Wrapper preview kamera */
[data-testid="stCameraInput"] div:has(video) {
    width: 100% !important;
    min-height: 620px !important;
}

/* Preview kamera */
[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 620px !important;
    min-height: 620px !important;
    max-height: none !important;
    object-fit: cover !important;
    border-radius: 14px 14px 0 0 !important;
}

/* Canvas preview kamera, kalau browser pakai canvas */
[data-testid="stCameraInput"] canvas {
    width: 100% !important;
    height: 620px !important;
    min-height: 620px !important;
    max-height: none !important;
    object-fit: cover !important;
    border-radius: 14px 14px 0 0 !important;
}

/* Setelah foto diambil */
[data-testid="stCameraInput"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 14px 14px 0 0 !important;
}

/* Tombol Take Photo bawaan kamera */
[data-testid="stCameraInput"] button {
    width: 100% !important;
    min-height: 58px !important;
    font-size: 20px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    width: 100% !important;
}

/* Gambar hasil upload/foto */
[data-testid="stImage"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 12px !important;
}

/* Khusus layar HP */
@media (max-width: 768px) {
    [data-testid="stCameraInput"] div:has(video),
    [data-testid="stCameraInput"] video,
    [data-testid="stCameraInput"] canvas {
        height: 70vh !important;
        min-height: 620px !important;
        max-height: none !important;
    }
}
</style>
""", unsafe_allow_html=True)


# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model_cached():
    return load_sawit_model()


model = load_model_cached()


# =========================
# SAFE RERUN
# =========================
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


# =========================
# SESSION STATE
# =========================
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False


# =========================
# HEADER
# =========================
st.title("🌴 Klasifikasi Kematangan Buah Kelapa Sawit")

st.write(
    "Aplikasi ini menggunakan model EfficientNetV2S untuk mengklasifikasikan "
    "tingkat kematangan buah kelapa sawit berdasarkan citra gambar."
)


# =========================
# INPUT MODE
# =========================
input_mode = st.radio(
    "Pilih metode input gambar:",
    ["Upload Gambar", "Ambil Foto"],
    horizontal=True
)

uploaded_file = None
camera_file = None


# =========================
# UPLOAD / CAMERA
# =========================
if input_mode == "Upload Gambar":
    st.session_state.camera_open = False

    uploaded_file = st.file_uploader(
        "Upload gambar buah kelapa sawit",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False
    )

else:
    if st.session_state.camera_open:
        if st.button("❌ Tutup Kamera"):
            st.session_state.camera_open = False
            safe_rerun()

        st.caption(
            "📱 Jika kamera depan terbuka, tekan ikon tukar kamera pada preview "
            "untuk memakai kamera belakang."
        )

        camera_file = st.camera_input("Ambil foto buah kelapa sawit")

    else:
        if st.button("📷 Buka Kamera"):
            st.session_state.camera_open = True
            safe_rerun()

        st.info("Tekan tombol **Buka Kamera** untuk mengambil foto.")


image_source = uploaded_file if uploaded_file is not None else camera_file


# =========================
# PREDICTION
# =========================
if image_source is not None:
    temp_path = None

    try:
        image_pil = Image.open(image_source).convert("RGB")

        if input_mode == "Upload Gambar":
            st.image(image_pil, caption="Gambar yang diupload")
        else:
            st.image(image_pil, caption="Foto yang diambil")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            image_pil.save(temp_file.name)
            temp_path = temp_file.name

        with st.spinner("Model sedang memproses gambar..."):
            predicted_class, confidence, probabilities = predict_image(model, temp_path)

        st.subheader("Hasil Prediksi")

        label_map = {
            "belum_masak": "Belum Masak",
            "masak": "Masak",
            "terlalu_masak": "Terlalu Masak"
        }

        display_class = label_map.get(predicted_class, predicted_class)

        if confidence < 50:
            st.warning(
                "⚠️ Confidence rendah. Hasil prediksi belum terlalu yakin. "
                "Coba ambil foto ulang dengan pencahayaan lebih jelas dan buah terlihat lebih dekat."
            )
            st.info(f"Prediksi sementara: **{display_class}**")
        else:
            st.success(f"Prediksi: **{display_class}**")

        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Probabilitas Tiap Kelas")

        for class_name, prob in probabilities.items():
            display_name = label_map.get(class_name, class_name)
            safe_prob = max(0, min(100, int(round(prob))))

            st.write(f"{display_name}: {prob:.2f}%")
            st.progress(safe_prob)

    except Exception as e:
        st.error("Terjadi error saat memproses gambar.")
        st.exception(e)

    finally:
        if temp_path is not None and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

else:
    if input_mode == "Upload Gambar":
        st.warning("Silakan upload gambar terlebih dahulu.")
    else:
        if st.session_state.camera_open:
            st.warning("Silakan ambil foto terlebih dahulu.")
