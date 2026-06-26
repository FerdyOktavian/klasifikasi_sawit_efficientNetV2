import os
import tempfile

import streamlit as st
from PIL import Image

from predict import load_sawit_model, load_class_names, predict_image


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Klasifikasi Kematangan Buah Sawit",
    page_icon="🌴",
    layout="centered"
)


# =========================
# CSS MODERN UI
# =========================
st.markdown("""
<style>
/* =========================
   GLOBAL STYLE
========================= */
.stApp {
    background: linear-gradient(135deg, #fff8e7 0%, #fffaf0 45%, #f6ffe9 100%) !important;
    color: #1f2937 !important;
}

.block-container {
    padding-top: 1.3rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 920px !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* Semua teks default */
p, span, label, div {
    color: #1f2937;
}

/* =========================
   HERO HEADER
========================= */
.hero-card {
    background: linear-gradient(135deg, #16a34a 0%, #22c55e 55%, #d9b51f 115%);
    padding: 30px 24px;
    border-radius: 28px;
    color: #ffffff !important;
    box-shadow: 0 18px 45px rgba(22, 163, 74, 0.25);
    margin-bottom: 22px;
    position: relative;
    overflow: hidden;
}

.hero-card * {
    color: #ffffff !important;
}

.hero-card::after {
    content: "";
    position: absolute;
    width: 175px;
    height: 175px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.18);
    top: -55px;
    right: -45px;
}

.hero-badge {
    display: inline-block;
    padding: 8px 15px;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.23);
    font-size: 14px;
    font-weight: 800;
    margin-bottom: 18px;
}

.hero-title {
    font-size: 38px;
    line-height: 1.12;
    font-weight: 950;
    margin: 0 0 16px 0;
    letter-spacing: -0.8px;
}

.hero-subtitle {
    font-size: 16px;
    line-height: 1.75;
    margin: 0;
    max-width: 720px;
    font-weight: 500;
}

/* =========================
   CARD STYLE
========================= */
.glass-card {
    background: #fffdf6;
    border: 1px solid #e7d9b0;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 12px 30px rgba(92, 64, 20, 0.08);
    margin-bottom: 18px;
}

.section-title {
    font-size: 20px;
    font-weight: 900;
    color: #064e3b !important;
    margin-bottom: 6px;
}

.section-desc {
    font-size: 14px;
    color: #475569 !important;
    margin-bottom: 14px;
}

/* =========================
   RADIO BUTTON
========================= */
div[role="radiogroup"] {
    background: #fffdf6 !important;
    padding: 12px 16px !important;
    border-radius: 18px !important;
    border: 1px solid #d8c99c !important;
    box-shadow: 0 8px 20px rgba(92, 64, 20, 0.06);
}

div[role="radiogroup"] label {
    color: #1f2937 !important;
    font-weight: 700 !important;
}

div[role="radiogroup"] label span {
    color: #1f2937 !important;
}

div[role="radiogroup"] div {
    color: #1f2937 !important;
}

/* =========================
   BUTTON
========================= */
div[data-testid="stButton"] button {
    width: 100% !important;
    min-height: 52px !important;
    border-radius: 16px !important;
    border: none !important;
    background: linear-gradient(135deg, #15803d, #22c55e) !important;
    color: #ffffff !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    box-shadow: 0 10px 24px rgba(22, 163, 74, 0.25) !important;
}

div[data-testid="stButton"] button p,
div[data-testid="stButton"] button span {
    color: #ffffff !important;
}

div[data-testid="stButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(22, 163, 74, 0.32) !important;
}

/* =========================
   FILE UPLOADER
========================= */
[data-testid="stFileUploader"] {
    background: #fffdf6 !important;
    border: 2px dashed #22c55e !important;
    border-radius: 22px !important;
    padding: 12px !important;
}

[data-testid="stFileUploader"] section {
    background: #fef3c7 !important;
    border: 1px solid #facc15 !important;
    border-radius: 18px !important;
}

[data-testid="stFileUploader"] section div {
    color: #1f2937 !important;
}

[data-testid="stFileUploader"] section span {
    color: #1f2937 !important;
    font-weight: 700 !important;
}

[data-testid="stFileUploader"] button {
    background: #15803d !important;
    color: #ffffff !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 800 !important;
}

[data-testid="stFileUploader"] button p,
[data-testid="stFileUploader"] button span {
    color: #ffffff !important;
}

/* =========================
   ALERT / INFO / WARNING
========================= */
div[data-testid="stAlert"] {
    border-radius: 18px !important;
    color: #1f2937 !important;
}

div[data-testid="stAlert"] * {
    color: #1f2937 !important;
    font-weight: 600 !important;
}

/* =========================
   IMAGE
========================= */
[data-testid="stImage"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 22px !important;
    box-shadow: 0 14px 35px rgba(15, 23, 42, 0.12);
}

/* =========================
   CAMERA
========================= */
[data-testid="stCameraInput"] {
    width: 100% !important;
}

[data-testid="stCameraInput"] video {
    width: 100% !important;
    height: 620px !important;
    object-fit: contain !important;
    object-position: center center !important;
    background: #111827 !important;
    border-radius: 24px !important;
    display: block !important;
    margin: 0 auto !important;
    box-shadow: 0 18px 38px rgba(15, 23, 42, 0.18);
}

[data-testid="stCameraInput"] img {
    width: 100% !important;
    height: auto !important;
    border-radius: 24px !important;
    display: block !important;
    margin: 0 auto !important;
}

[data-testid="stCameraInput"] button {
    width: 100% !important;
    min-height: 58px !important;
    font-size: 18px !important;
    font-weight: 900 !important;
    margin-top: 10px !important;
    border-radius: 16px !important;
    background: #15803d !important;
    color: #ffffff !important;
}

/* =========================
   RESULT CARD
========================= */
.result-card {
    background: #fffdf6;
    border: 1px solid #86efac;
    border-left: 8px solid #16a34a;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 35px rgba(22, 163, 74, 0.13);
    margin-top: 12px;
    margin-bottom: 18px;
}

.result-label {
    font-size: 14px;
    color: #475569 !important;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.result-main {
    font-size: 30px;
    color: #064e3b !important;
    font-weight: 950;
    margin-bottom: 8px;
}

.result-confidence {
    font-size: 17px;
    color: #166534 !important;
    font-weight: 900;
}

.low-card {
    background: #fff7ed;
    border: 1px solid #fdba74;
    border-left: 8px solid #f97316;
    border-radius: 24px;
    padding: 22px;
    box-shadow: 0 14px 35px rgba(249, 115, 22, 0.12);
    margin-top: 12px;
    margin-bottom: 18px;
}

.low-card * {
    color: #9a3412 !important;
}

.low-main {
    font-size: 26px;
    font-weight: 950;
    margin-bottom: 8px;
}

/* =========================
   PROBABILITY
========================= */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #16a34a, #facc15) !important;
}

.prob-row {
    background: #fffdf6;
    border: 1px solid #eadfbf;
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 10px;
    box-shadow: 0 8px 20px rgba(92, 64, 20, 0.05);
}

.prob-name {
    font-weight: 900;
    color: #064e3b !important;
}

.prob-value {
    float: right;
    font-weight: 950;
    color: #15803d !important;
}

/* =========================
   FOOTER
========================= */
.footer-note {
    text-align: center;
    color: #475569 !important;
    font-size: 13px;
    font-weight: 700;
    padding: 18px 0 6px 0;
}

.footer-note strong {
    color: #064e3b !important;
}

/* =========================
   MOBILE
========================= */
@media (max-width: 768px) {
    .block-container {
        padding-left: 0.75rem !important;
        padding-right: 0.75rem !important;
    }

    .hero-card {
        padding: 24px 18px;
        border-radius: 24px;
    }

    .hero-title {
        font-size: 29px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .glass-card {
        padding: 17px;
        border-radius: 22px;
    }

    [data-testid="stCameraInput"] video {
        height: 700px !important;
        object-fit: contain !important;
    }

    .result-main {
        font-size: 26px;
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


@st.cache_resource
def load_class_names_cached():
    return load_class_names()


model = load_model_cached()
class_names = load_class_names_cached()


# =========================
# SAFE RERUN
# =========================
def safe_rerun():
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()


def get_uploaded_file_suffix(uploaded_file):
    filename = getattr(uploaded_file, "name", "") or ""
    ext = os.path.splitext(filename)[1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".webp"]:
        return ext

    mime_type = getattr(uploaded_file, "type", "") or ""

    if "png" in mime_type:
        return ".png"
    if "webp" in mime_type:
        return ".webp"

    return ".jpg"


# =========================
# SESSION STATE
# =========================
if "camera_open" not in st.session_state:
    st.session_state.camera_open = False


# =========================
# HEADER MODERN
# =========================
st.markdown("""
<div class="hero-card">
    <div class="hero-badge">🌴 EfficientNetV2S Classification System</div>
    <h1 class="hero-title">Klasifikasi Kematangan Buah Kelapa Sawit</h1>
    <p class="hero-subtitle">
        Aplikasi berbasis AI untuk membantu mengenali tingkat kematangan buah kelapa sawit
        melalui gambar. Silakan upload gambar atau ambil foto langsung dari kamera.
    </p>
</div>
""", unsafe_allow_html=True)


# =========================
# INPUT MODE
# =========================
st.markdown("""
<div class="glass-card">
    <div class="section-title">📥 Pilih Metode Input</div>
    <div class="section-desc">
        Gunakan upload gambar dari galeri atau ambil foto langsung melalui kamera perangkat.
    </div>
</div>
""", unsafe_allow_html=True)

input_mode = st.radio(
    "Pilih metode input gambar:",
    ["Upload Gambar", "Ambil Foto"],
    horizontal=True,
    label_visibility="collapsed"
)

uploaded_file = None
camera_file = None


# =========================
# UPLOAD / CAMERA
# =========================
if input_mode == "Upload Gambar":
    st.session_state.camera_open = False

    st.markdown("""
    <div class="glass-card">
        <div class="section-title">🖼️ Upload Gambar</div>
        <div class="section-desc">
            Format yang didukung: JPG, JPEG, PNG, dan WEBP.
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload gambar buah kelapa sawit",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=False,
        label_visibility="collapsed"
    )

else:
    st.markdown("""
    <div class="glass-card">
        <div class="section-title">📷 Ambil Foto</div>
        <div class="section-desc">
            Arahkan kamera ke buah kelapa sawit dengan pencahayaan yang jelas.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.camera_open:
        if st.button("❌ Tutup Kamera"):
            st.session_state.camera_open = False
            safe_rerun()

        st.info(
            "📱 Jika kamera depan terbuka, tekan ikon tukar kamera pada preview "
            "untuk memakai kamera belakang."
        )

        camera_file = st.camera_input("Ambil foto buah kelapa sawit", label_visibility="collapsed")

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

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">🖼️ Preview Gambar</div>
            <div class="section-desc">
                Berikut gambar yang akan diproses oleh model.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if input_mode == "Upload Gambar":
            st.image(image_pil, caption="Gambar yang diupload")
        else:
            st.image(image_pil, caption="Foto yang diambil")

        # Simpan file asli ke temporary file agar proses prediksi tidak mengubah piksel gambar.
        # Ini penting supaya hasil upload Streamlit lebih konsisten dengan React/FastAPI.
        suffix = get_uploaded_file_suffix(image_source)
        image_source.seek(0)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(image_source.getvalue())
            temp_path = temp_file.name

        with st.spinner("🔍 Model sedang memproses gambar..."):
            predicted_class, confidence, probabilities = predict_image(
                model,
                temp_path,
                class_names
            )

        label_map = {
            "belum_masak": "Belum Masak",
            "masak": "Masak",
            "terlalu_masak": "Terlalu Masak"
        }

        display_class = label_map.get(predicted_class, predicted_class)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">📊 Hasil Prediksi</div>
            <div class="section-desc">
                Hasil klasifikasi berdasarkan gambar yang dimasukkan.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < 50:
            st.markdown(f"""
            <div class="low-card">
                <div class="result-label">Confidence Rendah</div>
                <div class="low-main">⚠️ Prediksi Sementara: {display_class}</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
                <p style="color:#9a3412; margin-top:12px; line-height:1.6;">
                    Model belum terlalu yakin dengan hasil ini. Coba ambil foto ulang dengan
                    pencahayaan lebih jelas dan posisi buah terlihat lebih dekat.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Prediksi Model</div>
                <div class="result-main">✅ {display_class}</div>
                <div class="result-confidence">Confidence: {confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="glass-card">
            <div class="section-title">📈 Probabilitas Tiap Kelas</div>
            <div class="section-desc">
                Nilai berikut menunjukkan tingkat keyakinan model pada setiap kelas.
            </div>
        </div>
        """, unsafe_allow_html=True)

        for class_name, prob in probabilities.items():
            display_name = label_map.get(class_name, class_name)
            safe_prob = max(0, min(100, int(round(prob))))

            st.markdown(f"""
            <div class="prob-row">
                <span class="prob-name">{display_name}</span>
                <span class="prob-value">{prob:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)

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


# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer-note">
    🌴 Sistem Klasifikasi Kematangan Buah Kelapa Sawit berbasis EfficientNetV2S<br>
    Dibuat oleh <strong>Muhammad Ferdy Oktavian</strong>
</div>
""", unsafe_allow_html=True)
