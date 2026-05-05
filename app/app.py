import os
import tempfile
import streamlit as st
from PIL import Image

from predict import load_sawit_model, predict_image


st.set_page_config(
    page_title="Klasifikasi Kematangan Buah Sawit",
    page_icon="🌴",
    layout="centered"
)
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

/* Tombol Take Photo */
[data-testid="stCameraInput"] button {
    width: 100% !important;
    height: 52px !important;
    font-size: 18px !important;
}

/* Khusus layar HP */
@media (max-width: 768px) {
    [data-testid="stCameraInput"] video {
        height: 460px !important;
    }
}
</style>
""", unsafe_allow_html=True)

st.title("🌴 Klasifikasi Kematangan Buah Kelapa Sawit")
st.write(
    "Aplikasi ini menggunakan model EfficientNetV2S untuk mengklasifikasikan "
    "tingkat kematangan buah kelapa sawit berdasarkan citra gambar."
)

@st.cache_resource
def load_model_cached():
    return load_sawit_model()

model = load_model_cached()
input_mode = st.radio(
    "Pilih metode input gambar:",
    ["Upload Gambar", "Ambil Foto"],
    horizontal=True
)

uploaded_file = None
camera_file = None

if input_mode == "Upload Gambar":
    uploaded_file = st.file_uploader(
        "Upload gambar buah kelapa sawit",
        type=["jpg", "jpeg", "png", "webp"]
    )
else:
    camera_file = st.camera_input("Ambil foto buah kelapa sawit")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    image_pil = Image.open(image_source).convert("RGB")

    st.image(image_pil, caption="Gambar yang diupload", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image_pil.save(temp_file.name)
        temp_path = temp_file.name

    if st.button("Prediksi"):
        with st.spinner("Model sedang memproses gambar..."):
            predicted_class, confidence, probabilities = predict_image(model, temp_path)

        st.subheader("Hasil Prediksi")

        label_map = {
            "belum_masak": "Belum Masak",
            "masak": "Masak",
            "terlalu_masak": "Terlalu Masak"
        }

        display_class = label_map.get(predicted_class, predicted_class)

        st.success(f"Prediksi: **{display_class}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        st.subheader("Probabilitas Tiap Kelas")

        for class_name, prob in probabilities.items():
            display_name = label_map.get(class_name, class_name)
            st.write(f"{display_name}: {prob:.2f}%")
            st.progress(int(prob))

    try:
        os.remove(temp_path)
    except:
        pass
else:
    if input_mode == "Upload Gambar":
        st.warning("Silakan upload gambar terlebih dahulu.")
    else:
        st.warning("Silakan ambil foto terlebih dahulu.")
