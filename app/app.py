import os
import tempfile
import streamlit as st
from PIL import Image
import io
import html

from predict import load_sawit_model, predict_image


st.set_page_config(
    page_title="SawitVision — Klasifikasi Kematangan Sawit",
    page_icon="🌴",
    layout="wide"
)


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700;900&family=Lato:wght@300;400;700;900&display=swap');

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

* { 
    box-sizing: border-box; 
    margin: 0; 
    padding: 0; 
}

html { 
    scroll-behavior: smooth; 
}

.stApp {
    background: var(--cream);
    color: var(--text);
    font-family: 'Lato', sans-serif;
}

#MainMenu, footer, header { 
    visibility: hidden; 
}

section[data-testid="stSidebar"] { 
    display: none; 
}

.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ─────────────────────────────────────
   NAVBAR
───────────────────────────────────── */
.navbar {
    position: fixed;
    top: 0; 
    left: 0; 
    right: 0;
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
    font-size: 46px;
    font-weight: 900;
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
    font-size: 18px;
    font-weight: 700;
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
    border-radius: 8px !important;
    font-weight: 900 !important;
    font-size: 16px !important;
    letter-spacing: 0.4px !important;
}

.nav-cta:hover {
    background: var(--green) !important;
    color: #fff !important;
}

.navbar-spacer {
    height: 72px;
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
    font-weight: 900;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--green);
    background: var(--green-pale);
    border: 1px solid #C3D9CA;
    border-radius: 6px;
    padding: 8px 18px;
    margin-bottom: 32px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 66px;
    font-weight: 900;
    line-height: 1.1;
    color: var(--text);
    letter-spacing: 1px;
    margin-bottom: 28px;
}

.hero-title em {
    font-style: italic;
    color: var(--green);
}

.hero-body {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.85;
    color: var(--text-2);
    margin-bottom: 48px;
    max-width: 540px;
}

.hero-stats {
    display: flex;
    gap: 40px;
    padding-top: 32px;
    border-top: 1px solid var(--border);
}

.stat-num {
    font-family: 'Playfair Display', serif;
    font-size: 50px;
    font-weight: 900;
    color: var(--green);
    line-height: 1;
    margin-bottom: 6px;
}

.stat-lbl {
    font-size: 18px;
    font-weight: 700;
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
    border-radius: 18px;
    padding: 30px 34px;
    display: flex;
    align-items: flex-start;
    gap: 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.05);
}

.hero-card-dot {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    margin-top: 6px;
    flex-shrink: 0;
}

.dot-unripe   { background: #D4552A; }
.dot-ripe     { background: var(--green); }
.dot-overripe { background: var(--gold); }

.hero-card-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 8px;
}

.hero-card-text {
    font-size: 18px;
    font-weight: 700;
    line-height: 1.7;
    color: var(--text-2);
}

/* ─────────────────────────────────────
   SECTION
───────────────────────────────────── */
.section {
    padding: 100px 64px;
    border-bottom: 1px solid var(--border);
}

.section-alt {
    background: #fff;
}

.section-eyebrow {
    font-size: 19px;
    font-weight: 900;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--green);
    margin-bottom: 16px;
}

.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 54px;
    font-weight: 900;
    color: var(--text);
    letter-spacing: -1px;
    line-height: 1.1;
    margin-bottom: 20px;
}

.section-desc {
    font-size: 22px;
    font-weight: 700;
    line-height: 1.8;
    color: var(--text-2);
    max-width: 760px;
    margin-bottom: 1px;
}

/* ─────────────────────────────────────
   PREDICTION WRAPPER
───────────────────────────────────── */
.prediction-wrapper {
    background: #fff;
    padding: 0 64px 80px;
    overflow: hidden;
}

[data-testid="stHorizontalBlock"],
[data-testid="column"] {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ─────────────────────────────────────
   FILE UPLOADER FINAL CLEAN
───────────────────────────────────── */
div[data-testid="stFileUploader"] {
    background: #ffffff !important;
    border: 1.5px dashed var(--green-light) !important;
    border-radius: 18px !important;
    padding: 24px !important;
    width: 100% !important;
    max-width: 100% !important;
    overflow: hidden !important;
    margin: 0 !important;
}

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label p,
div[data-testid="stFileUploader"] label span {
    font-family: 'Playfair Display', serif !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    color: var(--text) !important;
    margin-bottom: 14px !important;
}

div[data-testid="stFileUploader"] section {
    background: #20222c !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px !important;
}

div[data-testid="stFileUploader"] section > div {
    border: none !important;
    outline: none !important;
    min-height: 0 !important;
}

div[data-testid="stFileUploader"] section p,
div[data-testid="stFileUploader"] section span,
div[data-testid="stFileUploader"] small {
    color: #F7F4EE !important;
    font-size: 15px !important;
    font-weight: 700 !important;
}

/* Tombol utama upload */
# div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"]) {
#     position: relative !important;
#     background: var(--green) !important;
#     border: none !important;
#     border-radius: 10px !important;
#     padding: 12px 24px !important;
#     min-width: 130px !important;
#     height: 48px !important;
#     opacity: 1 !important;
#     color: transparent !important;
#     font-size: 0 !important;
#     overflow: hidden !important;
# }

/* Sembunyikan isi bawaan agar tidak double */
div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"]) * {
    display: none !important;
}

# /* Tulis ulang tombol upload yang rapi */
# div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"])::after {
#     content: "＋ Upload";
#     display: inline-flex !important;
#     align-items: center !important;
#     justify-content: center !important;
#     color: #ffffff !important;
#     font-family: 'Lato', sans-serif !important;
#     font-size: 15px !important;
#     font-weight: 900 !important;
#     letter-spacing: 0.2px !important;
# }

/* File yang sudah dipilih */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    background: #20222c !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 10px !important;
    max-width: 100% !important;
    overflow: hidden !important;
    padding: 8px 10px !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {
    color: #ffffff !important;
    font-weight: 800 !important;
}

/* Tombol X hapus file jangan kena style upload */
div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] {
    background: var(--green) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    min-width: 36px !important;
    width: 36px !important;
    height: 36px !important;
    padding: 0 !important;
}

div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] * {
    display: initial !important;
    color: #ffffff !important;
}

/* ─────────────────────────────────────
   CAMERA BUTTON
───────────────────────────────────── */
.input-divider {
    text-align: center;
    margin: 28px 0 24px;
    color: var(--text-3);
    font-size: 16px;
    font-weight: 900;
    letter-spacing: 2px;
}

div.stButton > button {
    width: auto !important;
    min-width: 230px !important;
    background: var(--green) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    padding: 18px 26px !important;
    letter-spacing: 0.3px !important;
}

div.stButton > button:hover {
    background: var(--green-light) !important;
    color: #fff !important;
}

div[data-testid="stCameraInput"] {
    background: #ffffff !important;
    border: 1.5px dashed var(--green-light) !important;
    border-radius: 18px !important;
    padding: 24px !important;
    margin-top: 22px !important;
    overflow: hidden !important;
}

div[data-testid="stCameraInput"] label {
    font-family: 'Playfair Display', serif !important;
    font-size: 22px !important;
    font-weight: 900 !important;
    color: var(--text) !important;
}

div[data-testid="stCameraInput"] button {
    background: var(--green) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 15px !important;
    font-weight: 900 !important;
    padding: 10px 22px !important;
}

/* ─────────────────────────────────────
   IMAGE PREVIEW
───────────────────────────────────── */
.image-preview-wrap {
    margin-top: 26px;
    width: 100%;
    max-width: 100%;
    overflow: hidden;
}

img {
    max-width: 100% !important;
    height: auto !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
}

/* ─────────────────────────────────────
   RESULT PANEL
───────────────────────────────────── */
.result-panel {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 42px !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.05) !important;
    overflow: hidden !important;
}

.result-badge {
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--green);
    background: var(--green-pale);
    border: 1px solid #C3D9CA;
    border-radius: 6px;
    padding: 6px 14px;
    display: inline-block;
    margin-bottom: 22px;
}

.result-class-name {
    font-family: 'Playfair Display', serif;
    font-size: 58px;
    font-weight: 900;
    color: var(--text);
    letter-spacing: -1.5px;
    margin-bottom: 10px;
    line-height: 1;
}

.result-conf-text {
    font-size: 21px;
    font-weight: 700;
    color: var(--text-2);
    margin-bottom: 32px;
}

.result-conf-text strong {
    font-weight: 900;
    color: var(--green);
    font-size: 28px;
}

.divider,
.result-divider {
    border: none !important;
    height: 1px !important;
    background: var(--border) !important;
    margin: 30px 0 !important;
}

.prob-section-title {
    font-family: 'Playfair Display', serif;
    font-size: 25px;
    font-weight: 900;
    color: var(--text);
    margin-top: 0 !important;
    padding-top: 0 !important;
    margin-bottom: 24px;
}

.prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.prob-name {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-2);
}

.prob-pct {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 900;
    color: var(--green);
}

.stProgress {
    max-width: 100% !important;
    overflow: hidden !important;
}

.stProgress > div {
    max-width: 100% !important;
}

.stProgress > div > div > div {
    background: var(--cream-3) !important;
    height: 9px !important;
    border-radius: 999px !important;
}

.stProgress > div > div > div > div {
    background: var(--green) !important;
    border-radius: 999px !important;
}

.result-note {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-3);
    line-height: 1.8;
    margin-top: 26px;
    padding-top: 22px;
    border-top: 1px solid var(--border);
}

/* ─────────────────────────────────────
   EMPTY STATE
───────────────────────────────────── */
.empty-state {
    background: #fff !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    padding: 72px 40px !important;
    text-align: center !important;
    box-shadow: 0 8px 28px rgba(0,0,0,0.04) !important;
}

.empty-icon { 
    font-size: 54px; 
    margin-bottom: 20px; 
}

.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 12px;
}

.empty-text {
    font-size: 17px;
    font-weight: 700;
    color: var(--text-2);
    line-height: 1.7;
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
    padding: 42px 34px;
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
    font-size: 25px;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 12px;
}

.step-text {
    font-size: 18px;
    font-weight: 700;
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
    padding: 42px 34px;
}

.sistem-icon {
    font-size: 36px;
    margin-bottom: 20px;
    display: block;
}

.sistem-title {
    font-family: 'Playfair Display', serif;
    font-size: 28px;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 12px;
}

.sistem-text {
    font-size: 18px;
    font-weight: 700;
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
    font-weight: 700;
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
    font-size: 13px;
    font-weight: 900;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--green);
    border: 1px solid #C3D9CA;
    border-radius: 6px;
    padding: 8px 17px;
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
    font-size: 14px;
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 8px;
}

.bio-item-val {
    font-family: 'Playfair Display', serif;
    font-size: 24px;
    font-weight: 900;
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
    font-weight: 900;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--text-3);
}

.kontak-val {
    font-size: 18px;
    font-weight: 700;
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
    font-size: 24px;
    font-weight: 900;
    color: #fff;
}

.footer-brand span { 
    color: rgba(255,255,255,0.60); 
}

.footer-copy {
    font-size: 16px;
    color: rgba(255,255,255,0.70);
    font-weight: 700;
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

.stSpinner > div {
    border-top-color: var(--green) !important;
}

/* ─────────────────────────────────────
   RESPONSIVE
───────────────────────────────────── */
@media (max-width: 900px) {
    .navbar { 
        padding: 0 24px; 
        height: 72px;
    }

    .nav-brand {
        font-size: 34px;
    }

    .nav-links { 
        display: none; 
    }

    .navbar-spacer {
        height: 72px;
    }

    .hero { 
        grid-template-columns: 1fr; 
        padding: 82px 24px 60px; 
        gap: 42px;
    }

    .hero-title { 
        font-size: 44px; 
        letter-spacing: 0;
    }

    .hero-body {
        font-size: 19px;
    }

    .hero-stats {
        flex-direction: column;
        gap: 20px;
    }

    .section { 
        padding: 72px 24px; 
    }

    .section-title {
        font-size: 42px;
    }

    .section-desc {
        font-size: 19px;
    }

    .prediction-wrapper {
        padding: 0 24px 64px;
    }

    div.stButton > button {
        width: 100% !important;
        min-width: 100% !important;
    }

    div[data-testid="stCameraInput"] video {
        width: 100% !important;
        min-height: 70vh !important;
        object-fit: cover !important;
        border-radius: 12px !important;
    }

    .result-panel {
        padding: 30px !important;
    }

    .result-class-name {
        font-size: 42px;
    }

    .steps-grid { 
        grid-template-columns: 1fr; 
    }

    .sistem-grid { 
        grid-template-columns: 1fr; 
    }

    .bio-layout { 
        grid-template-columns: 1fr; 
        gap: 40px; 
    }

    .bio-data {
        grid-template-columns: 1fr;
    }

    .kontak-layout { 
        grid-template-columns: 1fr; 
        gap: 32px; 
    }

    .kontak-row {
        flex-direction: column;
        align-items: flex-start;
        gap: 8px;
    }

    .footer { 
        flex-direction: column; 
        gap: 12px; 
        text-align: center; 
        padding: 32px 24px; 
    }
}
/* =====================================================
   FINAL UPLOAD BUTTON + SELECTED FILE CARD
===================================================== */



/* Sembunyikan isi tombol upload bawaan supaya tidak dobel */
div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"]) * {
    display: none !important;
}

/* Tulis ulang tombol upload */
div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"])::after {
    content: "↥ Upload";
    color: #1A1A1A !important;
    font-family: 'Lato', sans-serif !important;
    font-size: 17px !important;
    font-weight: 900 !important;
    letter-spacing: 0.2px !important;
}

/* Card file yang sudah dipilih */
.selected-file-card {
    background: #20222c;
    border-radius: 14px;
    padding: 18px 20px;
    margin-top: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 14px;
}

.selected-file-left {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.selected-file-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: #D9B85C;
    color: #1A1A1A;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    flex-shrink: 0;
}

.selected-file-name {
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 360px;
}

.selected-file-size {
    color: #d7d7d7;
    font-size: 13px;
    font-weight: 700;
    margin-top: 3px;
}

/* Tombol X */
.clear-file-btn {
    background: var(--green);
    color: white;
    border-radius: 25px;
    padding: 20px 24px;
    font-size: 18px;
    font-weight: 900;
    text-align: center;
}

/* Supaya tombol X Streamlit tidak terlalu lebar */
div[data-testid="stButton"] button {
    width: auto !important;
}

/* Tombol kamera tetap rapi */
div.stButton > button {
    background: var(--green) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 17px !important;
    font-weight: 900 !important;
    padding: 14px 26px !important;
}
/* =====================================================
   FIX FINAL UPLOAD SELECTED CARD + RESULT BAR
===================================================== */

/* Card file setelah gambar dipilih */
.selected-file-card {
    background: #20222c;
    border-radius: 14px;
    padding: 18px 20px;
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.selected-file-left {
    display: flex;
    align-items: center;
    gap: 14px;
    min-width: 0;
}

.selected-file-icon {
    width: 44px;
    height: 44px;
    border-radius: 10px;
    background: #D9B85C;
    color: #1A1A1A;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    font-weight: 900;
    flex-shrink: 0;
}

.selected-file-name {
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 360px;
}

.selected-file-size {
    color: #d7d7d7;
    font-size: 13px;
    font-weight: 700;
    margin-top: 4px;
}

/* Tombol X untuk hapus/ganti gambar */
button[kind="secondary"] {
    border: none !important;
}

/* Progress bar custom di dalam card hasil */
.custom-prob-wrap {
    margin-top: 8px;
    margin-bottom: 22px;
}

.custom-prob-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}

.custom-prob-name {
    font-size: 18px;
    font-weight: 700;
    color: var(--text-2);
}

.custom-prob-pct {
    font-family: 'Playfair Display', serif;
    font-size: 22px;
    font-weight: 900;
    color: var(--green);
}

.custom-progress-bg {
    width: 100%;
    height: 10px;
    background: var(--cream-3);
    border-radius: 999px;
    overflow: hidden;
}

.custom-progress-fill {
    height: 100%;
    background: var(--green);
    border-radius: 999px;
}

/* Biar tombol X tidak melebar */
div[data-testid="stButton"] button {
    color: #ffffff !important;
    width: auto !important;
    min-width: auto !important;
}
/* Card file setelah gambar dipilih */
.selected-file-area {
    display: grid;
    grid-template-columns: 1fr 58px;
    gap: 12px;
    align-items: stretch;
}
.selected-file-row {
    background: #20222c;
    border-radius: 14px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 14px;
    min-height: 66px;
    width: 100%;
}

.selected-file-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: #D9B85C;
    color: #1A1A1A;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 19px;
    flex-shrink: 0;
}

.selected-file-info {
    min-width: 0;
}

.selected-file-name {
    color: #ffffff;
    font-size: 16px;
    font-weight: 900;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.selected-file-size {
    color: #d7d7d7;
    font-size: 13px;
    font-weight: 700;
    margin-top: 4px;
}
.clear-upload-btn {
    height: 66px;
    width: 58px;
    background: #2D5A3D;
    color: #ffffff;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    font-weight: 900;
}
/* Bikin tombol Upload bawaan Streamlit kelihatan */
div[data-testid="stFileUploader"] button {
    background: #D9B85C !important;
    color: #1A1A1A !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 16px !important;
    font-weight: 900 !important;
    padding: 12px 24px !important;
}

/* Teks di dalam tombol Upload */
div[data-testid="stFileUploader"] button p,
div[data-testid="stFileUploader"] button span {
    color: #1A1A1A !important;
    font-size: 16px !important;
    font-weight: 900 !important;
}

/* Icon di tombol Upload */
div[data-testid="stFileUploader"] button svg {
    color: #1A1A1A !important;
    fill: #1A1A1A !important;
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
    <div class="section-eyebrow">Upload &amp; Analisis</div>
    <div class="section-title">Mulai Prediksi</div>
    <div class="section-desc">
        Upload gambar buah kelapa sawit Anda. Sistem akan secara otomatis menganalisis
        dan menampilkan hasil klasifikasi beserta tingkat kepercayaan model.
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown('<div style="background:#fff; padding: 0 64px 80px;">', unsafe_allow_html=True)
    if "show_camera" not in st.session_state:
        st.session_state.show_camera = False

    uploaded_file = None
    camera_file = None
    image_source = None
    image_pil = None

    if "input_bytes" not in st.session_state:
        st.session_state.input_bytes = None

    if "input_name" not in st.session_state:
        st.session_state.input_name = None

    if "input_source" not in st.session_state:
        st.session_state.input_source = None

    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0

    left_col, right_col = st.columns([1, 1], gap="large")

    with left_col:
        if st.session_state.input_bytes is None:
            uploaded_file = st.file_uploader(
                "Pilih Gambar Buah Sawit",
                type=["jpg", "jpeg", "png", "webp"],
                help="Format yang didukung: JPG, JPEG, PNG, WEBP",
                key=f"file_uploader_{st.session_state.upload_key}"
            )

            if uploaded_file is not None:
                st.session_state.input_bytes = uploaded_file.getvalue()
                st.session_state.input_name = uploaded_file.name
                st.session_state.input_source = "upload"
                st.session_state.upload_key += 1
                st.rerun()

            st.markdown('<div class="input-divider">ATAU</div>', unsafe_allow_html=True)

            if not st.session_state.show_camera:
                if st.button("📷 Ambil Foto Langsung"):
                    st.session_state.show_camera = True
                    st.rerun()
            else:
                camera_file = st.camera_input("Kamera aktif - ambil foto buah sawit")

                if camera_file is not None:
                    st.session_state.input_bytes = camera_file.getvalue()
                    st.session_state.input_name = "foto_kamera.jpg"
                    st.session_state.input_source = "camera"
                    st.session_state.show_camera = False
                    st.rerun()

                if st.button("Tutup Kamera"):
                    st.session_state.show_camera = False
                    st.rerun()

        else:
            file_size_kb = len(st.session_state.input_bytes) / 1024
            safe_name = html.escape(st.session_state.input_name or "gambar_sawit.jpg")
            source_label = "Foto Kamera" if st.session_state.input_source == "camera" else "File Upload"

            file_col, clear_col = st.columns([0.85, 0.15], gap="small")

            with file_col:
                st.markdown(f"""
                <div class="selected-file-row">
                    <div class="selected-file-icon">🖼️</div>
                    <div class="selected-file-info">
                        <div class="selected-file-name">{safe_name}</div>
                        <div class="selected-file-size">{source_label} • {file_size_kb:.1f} KB</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with clear_col:
                if st.button("Hapus", key="clear_uploaded_image", help="Hapus gambar"):
                    st.session_state.input_bytes = None
                    st.session_state.input_name = None
                    st.session_state.input_source = None
                    st.session_state.show_camera = False
                    st.session_state.upload_key += 1
                    st.rerun()

            image_pil = Image.open(io.BytesIO(st.session_state.input_bytes)).convert("RGB")

            st.markdown('<div style="margin-top: 24px;">', unsafe_allow_html=True)
            st.image(image_pil, caption="Gambar yang dianalisis")
            st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        if st.session_state.input_bytes is not None and image_pil is not None:
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
                    Upload gambar atau ambil foto langsung di sebelah kiri untuk memulai analisis klasifikasi.
                </div>
            </div>
            """, unsafe_allow_html=True)


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
