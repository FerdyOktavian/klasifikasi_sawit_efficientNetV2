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

/* Besarin area kamera */
[data-testid="stCameraInput"] {
    width: 100% !important;
}

/* Besarin preview kamera */
[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 430px !important;
    object-fit: cover !important;
    border-radius: 14px 14px 0 0 !important;
}

/* Setelah foto diambil */
[data-testid="stCameraInput"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 14px 14px 0 0 !important;
}

/* Tombol kamera bawaan Streamlit */
[data-testid="stCameraInput"] button {
    width: 100% !important;
    min-height: 52px !important;
    font-size: 18px !important;
}

/* File uploader agar rapi */
[data-testid="stFileUploader"] {
    width: 100% !important;
}

/* Khusus layar HP */
@media (max-width: 768px) {
    [data-testid="stCameraInput"] video {
        height: 460px !important;
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
        accept_multiple_files=False,
        width="stretch"
    )

else:
    if st.session_state.camera_open:
        if st.button("❌ Tutup Kamera", width="stretch"):
            st.session_state.camera_open = False
            st.rerun()

        st.caption(
            "📱 Jika kamera depan terbuka, tekan ikon tukar kamera pada preview "
            "untuk memakai kamera belakang."
        )

        camera_file = st.camera_input(
            "Ambil foto buah kelapa sawit",
            width="stretch"
        )

    else:
        if st.button("📷 Buka Kamera", width="stretch"):
            st.session_state.camera_open = True
            st.rerun()

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
            st.image(
                image_pil,
                caption="Gambar yang diupload",
                width="stretch"
            )
        else:
            st.image(
                image_pil,
                caption="Foto yang diambil",
                width="stretch"
            )

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
