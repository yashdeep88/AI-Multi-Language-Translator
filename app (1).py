# app.py
# Main Streamlit UI and application workflow.
import streamlit as st
print(st.__version__)
import streamlit as st
import os
import tempfile
import numpy as np


# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Multi-Language Translator",
    page_icon="🌐",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        color: #e0e0f0;
    }

    /* Hero header */
    .hero-header {
        text-align: center;
        padding: 2rem 1rem 1rem;
    }
    .hero-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero-header p {
        color: #a0aec0;
        font-size: 1.1rem;
    }

    /* Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(10px);
    }

    /* Section labels */
    .section-label {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #a78bfa;
        margin-bottom: 0.5rem;
    }

    /* Language badge */
    .lang-badge {
        display: inline-block;
        background: rgba(96, 165, 250, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(96, 165, 250, 0.3);
        border-radius: 999px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }

    /* Translated text box */
    .translated-box {
        background: rgba(52, 211, 153, 0.08);
        border: 1px solid rgba(52, 211, 153, 0.25);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        font-size: 1.1rem;
        color: #d1fae5;
        min-height: 80px;
        white-space: pre-wrap;
    }

    /* History entry */
    .history-entry {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #a78bfa;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
    }
    .history-entry .meta {
        color: #718096;
        font-size: 0.75rem;
        margin-bottom: 0.25rem;
    }

    /* Streamlit default button override */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed, #2563eb);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.55rem 1.5rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        opacity: 0.85;
        color: white;
    }

    /* Selectbox */
    div[data-baseweb="select"] > div {
        background: rgba(255,255,255,0.07) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #e0e0f0 !important;
        border-radius: 10px !important;
    }

    /* Text area */
    .stTextArea textarea {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.15) !important;
        color: #e0e0f0 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Lazy Module Loading (cached so models load once) ──────────────────────────
@st.cache_resource(show_spinner="🔄 Loading translation model (first run may take a few minutes)...")
def load_translator():
    from translator import translate_text
    return translate_text

@st.cache_resource(show_spinner="🎙️ Loading speech recognition model...")
def load_stt():
    import speech_to_text as stt_module
    return stt_module

@st.cache_resource
def load_tts():
    from text_to_speech import text_to_speech, cleanup_audio_file
    return text_to_speech, cleanup_audio_file

# Import lightweight modules
from language_detection import detect_language
from utils import SUPPORTED_LANGUAGES

# ── Session State ─────────────────────────────────────────────────────────────
if "translation_history" not in st.session_state:
    st.session_state.translation_history = []
if "last_audio_path" not in st.session_state:
    st.session_state.last_audio_path = None

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🌐 AI Multi-Language Translator</h1>
    <p>Voice-to-Voice · Text Translation · Auto Language Detection · 14 Languages</p>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_output = st.columns([1, 1], gap="large")

# ───────────────────── LEFT COLUMN – Input ─────────────────────────────
with col_input:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # === Translation engine selector ===
    st.markdown('<p class="section-label">Translation Engine</p>', unsafe_allow_html=True)
    engine = st.selectbox(
        label="Translation Engine",
        options=["Fast (Online)", "Offline (Small AI Model)"],
        index=0,
        label_visibility="collapsed",
        key="translation_engine_select",
        help="Fast uses online translate (no large downloads). Offline uses a smaller multilingual AI model (first download, then works offline).",
    )
    os.environ["TRANSLATION_BACKEND"] = "online" if engine.startswith("Fast") else "offline"

    # === Target language selector ===
    st.markdown('<p class="section-label">Target Language</p>', unsafe_allow_html=True)
    target_language = st.selectbox(
        label="Target Language",
        options=SUPPORTED_LANGUAGES,
        index=1,          # default: Hindi
        label_visibility="collapsed",
        key="target_lang_select"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # === Input Mode Tabs ===
    tab_text, tab_voice = st.tabs(["✏️ Text Input", "🎙️ Voice Input"])

    # ── TEXT INPUT TAB ──
    with tab_text:
        st.markdown('<p class="section-label">Enter text to translate</p>', unsafe_allow_html=True)
        user_text = st.text_area(
            label="Input Text",
            placeholder="Type your text here...",
            height=160,
            label_visibility="collapsed",
            key="text_input_area"
        )

        if st.button("🌐 Translate Text", use_container_width=True, key="translate_text_btn"):
            if user_text.strip():
                with st.spinner("Detecting language & translating..."):
                    translate_fn = load_translator()
                    detected_lang = detect_language(user_text)
                    translated = translate_fn(user_text, detected_lang, target_language)

                st.session_state.detected_lang = detected_lang
                st.session_state.translated_text = translated
                st.session_state.input_text_display = user_text

                # Add to history
                st.session_state.translation_history.insert(0, {
                    "input": user_text,
                    "output": translated,
                    "src_lang": detected_lang,
                    "tgt_lang": target_language
                })
            else:
                st.warning("Please enter some text to translate.")

    # ── VOICE INPUT TAB ──
    with tab_voice:
        st.info("Upload audio file (wav/mp3)")

        audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

        if audio_file is not None:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")

            with open(tmp.name, "wb") as f:
                f.write(audio_file.read())

            stt = load_stt()
            text, err = stt.transcribe_audio(tmp.name)

            os.remove(tmp.name)

            if text:
                translate_fn = load_translator()
                detected = detect_language(text)
                result = translate_fn(text, detected, target_language)

                st.session_state.result = result
                st.session_state.input = text
                st.session_state.detected = detected
            else:
                st.error("Transcription failed")


# ────────────────────── RIGHT COLUMN – Output ─────────────────────────────
with col_output:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    st.markdown('<p class="section-label">Translation Output</p>', unsafe_allow_html=True)

    detected_lang = st.session_state.get("detected_lang", None)
    translated_text = st.session_state.get("translated_text", None)
    input_text_display = st.session_state.get("input_text_display", None)

    if detected_lang:
        st.markdown(
            f'<span class="lang-badge">🔍 Detected: {detected_lang} → {target_language}</span>',
            unsafe_allow_html=True
        )

    if input_text_display:
        st.markdown('<p class="section-label" style="margin-top:0.8rem">Original Text</p>', unsafe_allow_html=True)
        st.markdown(f'<div class="translated-box" style="color:#c7d2fe">{input_text_display}</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-label" style="margin-top:0.8rem">Translated Text</p>', unsafe_allow_html=True)
    if translated_text:
        st.markdown(f'<div class="translated-box">{translated_text}</div>', unsafe_allow_html=True)

        # ── Audio Playback ──
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-label">🔊 Audio Playback</p>', unsafe_allow_html=True)

        if st.button("🔊 Generate & Play Audio", use_container_width=True, key="tts_btn"):
            with st.spinner("Generating speech..."):
                tts_fn, cleanup_fn = load_tts()
                audio_path = tts_fn(translated_text, target_language)

            if audio_path:
                # Cleanup previous temp file
                if st.session_state.last_audio_path:
                    cleanup_fn(st.session_state.last_audio_path)
                st.session_state.last_audio_path = audio_path

                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                # Autoplay audio
                import base64
                audio_b64 = base64.b64encode(audio_bytes).decode()
                st.markdown(
                    f'<audio autoplay controls style="width:100%;border-radius:10px;margin-top:0.5rem">'
                    f'<source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">'
                    f'</audio>',
                    unsafe_allow_html=True
                )
            else:
                st.error("Failed to generate audio. Please check your internet connection (gTTS requires internet).")
    else:
        st.markdown('<div class="translated-box" style="color:#4a5568;font-style:italic">Translation will appear here...</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ── Translation History ────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">📜 Translation History (This Session)</p>', unsafe_allow_html=True)

history = st.session_state.translation_history
if not history:
    st.markdown('<p style="color:#4a5568;font-style:italic;font-size:0.9rem">No translations yet. Start translating above!</p>', unsafe_allow_html=True)
else:
    cols = st.columns([1, 5])
    with cols[1]:
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.translation_history = []
            st.rerun()

    for i, entry in enumerate(history[:10]):  # Show latest 10
        st.markdown(f"""
        <div class="history-entry">
            <div class="meta">#{i+1} · {entry['src_lang']} → {entry['tgt_lang']}</div>
            <div><strong>In:</strong> {entry['input'][:120]}{'...' if len(entry['input']) > 120 else ''}</div>
            <div style="color:#a0fabe"><strong>Out:</strong> {entry['output'][:120]}{'...' if len(entry['output']) > 120 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
