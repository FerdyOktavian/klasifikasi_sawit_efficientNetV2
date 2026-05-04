import os
import tempfile
import streamlit as st
from PIL import Image

from predict import load_sawit_model, predict_image


st.set_page_config(
    page_title="SawitVision — Klasifikasi Kematangan Sawit",
    page_icon="🌴",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Lato:wght@300;400;700&display=swap');

:root {
    --cream:   #F7F4EE;
    --cream-2: #EDE9DF;
    --cream-3: #E2DDD2;
    --green:   #2D5A3D;
    --green-light: #4A7C5F;
    --green-pale:  #EBF2ED;
    --gold:    #B8860B;
    --gold-pale: #FBF5E6;
    --text:    #1A1A1A;
    --text-2:  #4A4740;
    --text-3:  #7A766E;
    --border:  #D8D3C8;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

html { scroll-behavior: smooth; }

.stApp {
    background: var(--cream);
    color: var(--text);
    font-family: 'Lato', sans-serif;
}

/* ─── Hide default streamlit chrome ─── */
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ─────────────────────────────────────
   NAVBAR  (position:fixed via JS trick)
───────────────────────────────────── */
.navbar {
    position: fixed;
    top: 0; left: 0; right: 0;
    z-index: 9999;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 64px;
    height: 72px;
    background: rgba(247, 244, 238, 0.96);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid var(--border);
}

.nav-brand {
    font-family: 'Playfair Display', serif;
    font-size: 50px;
    font-weight: 700;
    color: var(--green);
    letter-spacing: -0.3px;
}

.nav-brand span {
    color: var(--gold);
}

.nav-links {
    display: flex;
    align-items: center;
    gap: 40px;
}

.nav-links a {
    font-family: 'Lato', sans-serif;
    font-size: 20px;
    font-weight: 400;
    letter-spacing: 0.5px;
    color: var(--text-2) !important;
    text-decoration: none;
    transition: color 0.2s;
}

.nav-links a:hover {
    color: var(--green-light) !important;
}

.nav-cta {
    background: var(--green-light) !important;
    color: #fff !important;
    padding: 10px 24px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    letter-spacing: 0.4px !important;
}

.nav-cta:hover {
    background: var(--green-light) !important;
    color: #fff !important;
}

/* Spacer so content doesn't hide behind fixed navbar */
.navbar-spacer {
    height: 50px;
}

/* ─────────────────────────────────────
   HERO
───────────────────────────────────── */
.hero {
    background: var(--cream);
    padding: 100px 64px 100px;
    border-bottom: 1px solid var(--border);
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 80px;
    align-items: center;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 16px;
    font-size: 18px;
    font-weight: 700;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--green);
    background: var(--green-pale);
    border: 1px solid #C3D9CA;
    border-radius: 4px;
    padding: 7px 16px;
    margin-bottom: 32px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 66px;
    font-weight: 900;
    line-height: 1.1;
    color: var(--text);
    letter-spacing: 3.5px;
    margin-bottom: 28px;
}

.hero-title em {
    font-style: italic;
    color: var(--green);
}

.hero-body {
    font-size: 22px;
    font-weight: 300;
    line-height: 1.85;
    color: var(--text-2);
    margin-bottom: 48px;
    max-width: 480px;
}

.hero-stats {
    display: flex;
    gap: 40px;
    padding-top: 32px;
    border-top: 1px solid var(--border);
}

.stat-item {}

.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 50px;
    font-weight: 700;
    color: var(--green);
    line-height: 1;
    margin-bottom: 6px;
}

.stat-lbl {
    font-size: 20px;
    font-weight: 400;
    color: var(--text-3);
    letter-spacing: 0.5px;
}

.hero-right {
    display: flex;
    flex-direction: column;
    gap: 20px;
}

.hero-card {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px 32px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.04);
}

.hero-card-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-top: 4px;
    flex-shrink: 0;
}

.dot-unripe   { background: #D4552A; }
.dot-ripe     { background: var(--green); }
.dot-overripe { background: var(--gold); }

.hero-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 8px;
}

.hero-card-text {
    font-size: 18px;
    font-weight: 300;
    line-height: 1.7;
    color: var(--text-2);
}

/* ─────────────────────────────────────
   SECTION WRAPPER
───────────────────────────────────── */
.section {
    padding: 100px 64px;
    border-bottom: 1px solid var(--border);
}

.section-alt {
    background: #fff;
}

.section-eyebrow {
    font-size: 20px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 16px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 52px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 20px;
}

.section-desc {
    font-size: 22px;
    font-weight: 450;
    line-height: 1.8;
    color: var(--text-2);
    max-width: 600px;
    margin-bottom: 1px;
}

/* ─────────────────────────────────────
   UPLOAD & RESULT
───────────────────────────────────── */
.upload-box {
    background: #fff;
    border: 1.5px dashed var(--green-light);
    border-radius: 16px;
    padding: 32px;
    margin-bottom: 0;
}

/* Streamlit file uploader overrides */
div[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stFileUploader"] label {
    font-family: 'Playfair Display', serif !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] p {
    font-size: 14px !important;
    color: var(--text-3) !important;
    font-family: 'Lato', sans-serif !important;
}

div[data-testid="stFileUploader"] button {
    background: var(--green) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 10px 22px !important;
    letter-spacing: 0.3px !important;
}

.result-panel {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.05);
}

.result-badge {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--green);
    background: var(--green-pale);
    border: 1px solid #C3D9CA;
    border-radius: 4px;
    padding: 5px 12px;
    display: inline-block;
    margin-bottom: 20px;
}

.result-class-name {
    font-family: 'Playfair Display', serif;
    font-size: 56px;
    font-weight: 900;
    color: var(--text);
    letter-spacing: -1.5px;
    margin-bottom: 8px;
    line-height: 1;
}

.result-conf-text {
    font-size: 20px;
    font-weight: 300;
    color: var(--text-2);
    margin-bottom: 36px;
}

.result-conf-text strong {
    font-weight: 700;
    color: var(--green);
    font-size: 26px;
}

.divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
}

.prob-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 20px;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
}

.prob-name {
    font-size: 18px;
    font-weight: 400;
    color: var(--text-2);
}

.prob-pct {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--green);
}

/* Progress bar override */
.stProgress > div > div > div {
    background: var(--cream-3) !important;
    height: 8px !important;
    border-radius: 999px !important;
}

.stProgress > div > div > div > div {
    background: var(--green) !important;
    border-radius: 999px !important;
}

.result-note {
    font-size: 16px;
    color: var(--text-3);
    line-height: 1.7;
    margin-top: 24px;
    padding-top: 20px;
    border-top: 1px solid var(--border);
}

/* ─────────────────────────────────────
   STEPS
───────────────────────────────────── */
.steps-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 2px;
    background: var(--border);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
}

.step-cell {
    background: #fff;
    padding: 40px 32px;
}

.step-num {
    font-family: 'Playfair Display', serif;
    font-size: 56px;
    font-weight: 900;
    color: var(--cream-3);
    line-height: 1;
    margin-bottom: 20px;
}

.step-title {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
}

.step-text {
    font-size: 18px;
    font-weight: 300;
    line-height: 1.75;
    color: var(--text-2);
}

/* ─────────────────────────────────────
   SISTEM
───────────────────────────────────── */
.sistem-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 24px;
}

.sistem-card {
    background: var(--cream);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 32px;
}

.sistem-icon {
    font-size: 34px;
    margin-bottom: 20px;
    display: block;
}

.sistem-title {
    font-family: 'Playfair Display', serif;
    font-size: 26px;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 12px;
}

.sistem-text {
    font-size: 18px;
    font-weight: 300;
    line-height: 1.8;
    color: var(--text-2);
}

/* ─────────────────────────────────────
   BIO
───────────────────────────────────── */
.bio-layout {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 80px;
    align-items: start;
}

.bio-body {
    font-size: 20px;
    font-weight: 300;
    line-height: 1.85;
    color: var(--text-2);
    margin-bottom: 20px;
}

.bio-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 32px;
}

.bio-tag {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green);
    border: 1px solid #C3D9CA;
    border-radius: 4px;
    padding: 7px 16px;
    background: var(--green-pale);
}

.bio-data {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
}

.bio-item {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px 24px;
}

.bio-item-label {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 8px;
}

.bio-item-val {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: var(--text);
}

/* ─────────────────────────────────────
   KONTAK
───────────────────────────────────── */
.kontak-layout {
    display: grid;
    grid-template-columns: 0.6fr 1fr;
    gap: 80px;
    align-items: start;
}

.kontak-list {
    display: flex;
    flex-direction: column;
    gap: 12px;
}

.kontak-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px 24px;
}

.kontak-key {
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
}

.kontak-val {
    font-size: 18px;
    font-weight: 400;
    color: var(--text);
}

/* ─────────────────────────────────────
   FOOTER
───────────────────────────────────── */
.footer {
    padding: 40px 64px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: var(--green);
}

.footer-brand {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 700;
    color: #fff;
}

.footer-brand span { color: rgba(255,255,255,0.55); }

.footer-copy {
    font-size: 16px;
    color: rgba(255,255,255,0.55);
    font-weight: 300;
}

/* ─────────────────────────────────────
   IMAGE
───────────────────────────────────── */
img {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* ─────────────────────────────────────
   STREAMLIT GENERAL
───────────────────────────────────── */
.stAlert {
    background: var(--green-pale);
    border: 1px solid #C3D9CA;
    border-radius: 10px;
    color: var(--text);
}

/* Spinner */
.stSpinner > div {
    border-top-color: var(--green) !important;
}

/* ─────────────────────────────────────
   RESPONSIVE
───────────────────────────────────── */
@media (max-width: 900px) {
    .navbar { padding: 0 24px; }
    .nav-links { display: none; }
    .hero { grid-template-columns: 1fr; padding: 80px 24px 60px; }
    .hero-title { font-size: 40px; }
    .section { padding: 72px 24px; }
    .steps-grid { grid-template-columns: 1fr 1fr; }
    .sistem-grid { grid-template-columns: 1fr; }
    .bio-layout { grid-template-columns: 1fr; gap: 40px; }
    .kontak-layout { grid-template-columns: 1fr; gap: 32px; }
    .footer { flex-direction: column; gap: 12px; text-align: center; padding: 32px 24px; }
}
</style>
""", unsafe_allow_html=True)


# ── LOAD MODEL ──
@st.cache_resource
def load_model_cached():
    return load_sawit_model()

model = load_model_cached()

label_map = {
    "belum_masak": "Belum Masak",
    "masak": "Masak",
    "terlalu_masak": "Terlalu Masak"
}


# ══════════════════════════════════════
# NAVBAR
# ══════════════════════════════════════
st.markdown("""
<div class="navbar">
    <div class="nav-brand">Sawit<span>Vision</span></div>
    <div class="nav-links">
        <a href="#prediksi">Prediksi</a>
        <a href="#cara-pakai">Cara Pakai</a>
        <a href="#sistem">Sistem</a>
        <a href="#biografi">Biografi</a>
        <a href="#kontak" class="nav-cta">Kontak</a>
    </div>
</div>
<div class="navbar-spacer"></div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# HERO
# ══════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-left">
        <div class="hero-tag">🌴 Sistem AI Berbasis Deep Learning</div>
        <div class="hero-title">
            Identifikasi Kematangan<br>
            Buah <em>Kelapa Sawit</em>
        </div>
        <div class="hero-body">
            Sistem klasifikasi berbasis EfficientNetV2S yang mampu mengidentifikasi
            tingkat kematangan buah kelapa sawit secara akurat hanya dari sebuah
            foto. Cepat, tepat, dan mudah digunakan.
        </div>
        <div class="hero-stats">
            <div class="stat-item">
                <div class="stat-num">3</div>
                <div class="stat-lbl">Kelas Kematangan</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">CNN</div>
                <div class="stat-lbl">Metode Klasifikasi</div>
            </div>
            <div class="stat-item">
                <div class="stat-num">AI</div>
                <div class="stat-lbl">Deep Learning</div>
            </div>
        </div>
    </div>
    <div class="hero-right">
        <div class="hero-card">
            <div class="hero-card-dot dot-unripe"></div>
            <div>
                <div class="hero-card-title">Belum Masak</div>
                <div class="hero-card-text">Warna cenderung gelap atau kehijauan. Kandungan minyak masih rendah dan belum siap untuk dipanen.</div>
            </div>
        </div>
        <div class="hero-card">
            <div class="hero-card-dot dot-ripe"></div>
            <div>
                <div class="hero-card-title">Masak</div>
                <div class="hero-card-text">Warna matang optimal dan kandungan minyak tertinggi. Kondisi ideal untuk proses pemanenan.</div>
            </div>
        </div>
        <div class="hero-card">
            <div class="hero-card-dot dot-overripe"></div>
            <div>
                <div class="hero-card-title">Terlalu Masak</div>
                <div class="hero-card-text">Kematangan berlebih dengan tanda brondolan lepas. Kualitas minyak menurun dan segera perlu penanganan.</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# PREDIKSI
# ══════════════════════════════════════
st.markdown('<div id="prediksi"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
    <div class="section-eyebrow">Upload & Analisis</div>
    <div class="section-title">Mulai Prediksi</div>
    <div class="section-desc">
        Upload gambar buah kelapa sawit Anda. Sistem akan secara otomatis menganalisis
        dan menampilkan hasil klasifikasi beserta tingkat kepercayaan model.
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div style="background:#fff; padding: 0 64px 80px;">', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Pilih Gambar Buah Sawit",
            type=["jpg", "jpeg", "png", "webp"],
            help="Format yang didukung: JPG, JPEG, PNG, WEBP"
        )
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
            st.image(image_pil, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image_pil.save(tmp.name)
                tmp_path = tmp.name

            try:
                with st.spinner("Model sedang menganalisis gambar..."):
                    predicted_class, confidence, probabilities = predict_image(model, tmp_path)

                display_class = label_map.get(predicted_class, predicted_class)

                st.markdown(f"""
                <div class="result-panel">
                    <div class="result-badge">Hasil Klasifikasi</div>
                    <div class="result-class-name">{display_class}</div>
                    <div class="result-conf-text">
                        Tingkat keyakinan model: <strong>{confidence:.1f}%</strong>
                    </div>
                    <hr class="divider">
                    <div class="prob-section-title">Distribusi Probabilitas</div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="result-panel" style="margin-top:-1px; border-top: none; border-radius: 0 0 16px 16px; padding-top: 0;">', unsafe_allow_html=True)
                for class_name, prob in probabilities.items():
                    dname = label_map.get(class_name, class_name)
                    st.markdown(f"""
                    <div class="prob-row">
                        <span class="prob-name">{dname}</span>
                        <span class="prob-pct">{prob:.1f}%</span>
                    </div>
                    """, unsafe_allow_html=True)
                    st.progress(int(prob))
                    st.markdown('<div style="height:10px;"></div>', unsafe_allow_html=True)

                st.markdown("""
                <div class="result-note">
                    Catatan: Hasil prediksi dipengaruhi oleh kualitas gambar, pencahayaan,
                    dan jarak pengambilan foto. Gunakan gambar yang jelas untuk hasil terbaik.
                </div>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            except Exception as e:
                st.error("Terjadi kesalahan saat melakukan prediksi.")
                st.write(e)
            finally:
                try:
                    os.remove(tmp_path)
                except:
                    pass
        else:
            st.markdown("""
            <div class="result-panel" style="text-align:center; padding: 60px 40px;">
                <div style="font-size: 48px; margin-bottom: 20px;">🌿</div>
                <div style="font-family: 'Playfair Display', serif; font-size: 22px; font-weight: 700; color: var(--text); margin-bottom: 12px;">
                    Belum Ada Gambar
                </div>
                <div style="font-size: 15px; font-weight: 300; color: var(--text-3); line-height: 1.7;">
                    Upload gambar buah kelapa sawit di sebelah kiri untuk memulai analisis klasifikasi.
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ══════════════════════════════════════
# CARA PAKAI
# ══════════════════════════════════════
st.markdown('<div id="cara-pakai"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section">
    <div class="section-eyebrow">Panduan Penggunaan</div>
    <div class="section-title">Cara Menggunakan</div>
    <div class="section-desc">
        Empat langkah sederhana untuk mendapatkan hasil klasifikasi kematangan buah sawit.
    </div>
    <div class="steps-grid">
        <div class="step-cell">
            <div class="step-num">01</div>
            <div class="step-title">Siapkan Gambar</div>
            <div class="step-text">Gunakan foto buah kelapa sawit dengan pencahayaan yang merata dan objek terlihat jelas. Pastikan gambar tidak buram.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">02</div>
            <div class="step-title">Upload File</div>
            <div class="step-text">Klik tombol upload dan pilih gambar dari perangkat Anda. Format yang didukung adalah JPG, JPEG, PNG, dan WEBP.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">03</div>
            <div class="step-title">Analisis Otomatis</div>
            <div class="step-text">Sistem secara otomatis memproses gambar menggunakan model deep learning EfficientNetV2S tanpa perlu klik tambahan.</div>
        </div>
        <div class="step-cell">
            <div class="step-num">04</div>
            <div class="step-title">Baca Hasilnya</div>
            <div class="step-text">Dapatkan kelas prediksi, nilai confidence, dan distribusi probabilitas lengkap untuk setiap kategori kematangan.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# SISTEM
# ══════════════════════════════════════
st.markdown('<div id="sistem"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
    <div class="section-eyebrow">Tentang Teknologi</div>
    <div class="section-title">Tentang Sistem</div>
    <div class="section-desc">
        Sistem ini dibangun menggunakan pendekatan deep learning mutakhir untuk
        klasifikasi citra buah kelapa sawit secara akurat dan efisien.
    </div>
    <div class="sistem-grid">
        <div class="sistem-card">
            <span class="sistem-icon">🧠</span>
            <div class="sistem-title">Arsitektur Model</div>
            <div class="sistem-text">
                Menggunakan EfficientNetV2S, arsitektur Convolutional Neural Network generasi
                terbaru yang menggabungkan efisiensi komputasi dengan akurasi tinggi dalam
                klasifikasi citra digital.
            </div>
        </div>
        <div class="sistem-card">
            <span class="sistem-icon">📷</span>
            <div class="sistem-title">Input Sistem</div>
            <div class="sistem-text">
                Menerima citra digital buah kelapa sawit yang diupload pengguna melalui
                antarmuka web. Mendukung format JPG, JPEG, PNG, dan WEBP dengan berbagai
                ukuran resolusi.
            </div>
        </div>
        <div class="sistem-card">
            <span class="sistem-icon">📊</span>
            <div class="sistem-title">Output Sistem</div>
            <div class="sistem-text">
                Menghasilkan prediksi kelas kematangan (Belum Masak, Masak, atau Terlalu Masak),
                nilai confidence model, serta distribusi probabilitas lengkap secara real-time.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# BIOGRAFI
# ══════════════════════════════════════
st.markdown('<div id="biografi"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section">
    <div class="section-eyebrow">Pengembang</div>
    <div class="section-title">Biografi Pembuat</div>
    <div class="bio-layout">
        <div>
            <div class="bio-body">
                Aplikasi ini dikembangkan sebagai bagian dari penelitian skripsi di bidang
                kecerdasan buatan dan pengolahan citra digital. Tujuan utama penelitian adalah
                membantu proses identifikasi tingkat kematangan buah kelapa sawit secara lebih
                objektif menggunakan teknologi deep learning.
            </div>
            <div class="bio-body">
                Sistem dirancang agar mudah digunakan oleh siapa saja melalui antarmuka web
                yang intuitif, tanpa memerlukan keahlian teknis khusus di bidang kecerdasan buatan.
            </div>
            <div class="bio-tags">
                <span class="bio-tag">Artificial Intelligence</span>
                <span class="bio-tag">Computer Vision</span>
                <span class="bio-tag">Deep Learning</span>
                <span class="bio-tag">Skripsi</span>
            </div>
        </div>
        <div class="bio-data">
            <div class="bio-item">
                <div class="bio-item-label">Nama</div>
                <div class="bio-item-val">Muhammad Ferdy</div>
            </div>
            <div class="bio-item">
                <div class="bio-item-label">Bidang</div>
                <div class="bio-item-val">Computer Vision</div>
            </div>
            <div class="bio-item">
                <div class="bio-item-label">Topik</div>
                <div class="bio-item-val">Klasifikasi Sawit</div>
            </div>
            <div class="bio-item">
                <div class="bio-item-label">Tahun</div>
                <div class="bio-item-val">2026</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# KONTAK
# ══════════════════════════════════════
st.markdown('<div id="kontak"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="section section-alt">
    <div class="kontak-layout">
        <div>
            <div class="section-eyebrow">Hubungi Kami</div>
            <div class="section-title">Kontak<br>Pembuat</div>
        </div>
        <div class="kontak-list">
            <div class="kontak-row">
                <span class="kontak-key">Nama</span>
                <span class="kontak-val">Muhammad Ferdy</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Email</span>
                <span class="kontak-val">isi_email_kamu@gmail.com</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Instagram</span>
                <span class="kontak-val">@username_kamu</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Universitas</span>
                <span class="kontak-val">isi_nama_kampus_kamu</span>
            </div>
            <div class="kontak-row">
                <span class="kontak-key">Program Studi</span>
                <span class="kontak-val">isi_program_studi_kamu</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════
# FOOTER
# ══════════════════════════════════════
st.markdown("""
<div class="footer">
    <div class="footer-brand">Sawit<span>Vision</span> AI</div>
    <div class="footer-copy">© 2026 — Sistem Klasifikasi Kematangan Buah Kelapa Sawit</div>
</div>
""", unsafe_allow_html=True)