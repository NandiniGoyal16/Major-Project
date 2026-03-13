import streamlit as st
import cv2
import numpy as np
from PIL import Image
from emotion_engine import EmotionEngine
from music_engine import MusicEngine
from audio_engine import AudioEngine
from composition_engine import CompositionEngine
import os
import tempfile
from pathlib import Path

# Set page config
st.set_page_config(page_title="Emotion Music Generator", layout="wide")

# Sidebar for options
st.sidebar.header("Settings")
emotion_variant = st.sidebar.selectbox("Emotion Recognition Variant", [8, 5], index=0)
# Rebranded as Cortex Transformer (Democratizing AI Music)
st.sidebar.info("Generation Engine: Cortex Transformer")

# Initialize engines
@st.cache_resource
def load_engines(variant):
    emotion = EmotionEngine(model_variant=variant)
    music = MusicEngine()
    audio = AudioEngine()
    comp = CompositionEngine()
    return emotion, music, audio, comp

# Centralized Emotion to Valence/Arousal Mapping
# Renamed Neutral to Calm, and filtered to 5 core emotions as requested
EMOTION_TO_VA = {
    "Happy": (0.8, 0.6),
    "Angry": (-0.7, 0.9),
    "Sad": (-0.9, -0.7),
    "Calm": (0.2, -0.6), # Formerly Neutral
    "Romantic": (0.6, 0.2), # Happy + Calm quadrant
}

try:
    with st.spinner("Initializing models..."):
        emotion_engine, music_engine, audio_engine, comp_engine = load_engines(emotion_variant)
except Exception as e:
    st.error(f"Error loading engines: {e}")
    st.stop()

# Get genres from discovery
available_genres = sorted(list(comp_engine.genre_map.keys()))

st.title("🎵 Cortex Transformer")
st.markdown("### Democratizing AI Music Composition")

# 1. UI Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("1. Choose Input")
    input_mode = st.radio("Input Source", ["Image Upload", "Live Camera", "Manual Selection"])
    
    st.header("2. Composition Style")
    # Genre Selection First
    selected_genre = st.radio("Choose Musical Genre", available_genres)
    
    # Filter instruments by selected genre
    genre_instruments = sorted(list(comp_engine.genre_map.get(selected_genre, {}).keys()))
    
    inst_mode = st.radio("Selection Mode", ["Single Instrument", "Multiple Instruments"])

    if inst_mode == "Single Instrument":
        instrument = st.selectbox(f"Choose an {selected_genre} Instrument", genre_instruments)
        selected_instruments = [instrument]
    else:
        selected_instruments = st.multiselect(f"Choose Multiple {selected_genre} Instruments", genre_instruments)
        if not selected_instruments:
            st.warning(f"Please select at least one {selected_genre} instrument.")
            selected_instruments = [genre_instruments[0]] if genre_instruments else []
    
    st.header("3. Music Settings")
    duration = st.slider("Duration (seconds)", 5, 30, 20)
    input_image = None
    manual_emotion = None

    if input_mode == "Manual Selection":
        manual_emotion = st.selectbox("Choose an Emotion", ["Happy", "Angry", "Sad", "Calm", "Romantic"])
    elif input_mode == "Image Upload":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_image = Image.open(uploaded_file)
            st.image(input_image, caption="Uploaded Image", use_container_width=True)
    elif input_mode == "Live Camera":
        camera_file = st.camera_input("Take a photo")
        if camera_file is not None:
            input_image = Image.open(camera_file)
            st.image(input_image, caption="Captured Image", use_container_width=True)

# 2. Processing and Results
with col2:
    st.header("2. Process & Results")
    
    emotion_res = None
    if input_mode == "Manual Selection" and manual_emotion:
        v, a = EMOTION_TO_VA.get(manual_emotion, (0.0, 0.0))
        emotion_res = {'emotion': manual_emotion, 'valence': v, 'arousal': a}
    elif input_image is not None:
        with st.spinner("Analyzing emotion..."):
            # Convert PIL Image to numpy array
            img_np = np.array(input_image)
            # Ensure BGR for OpenCV
            if len(img_np.shape) == 2: # Gray
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            elif img_np.shape[2] == 4: # RGBA
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
            else: # RGB
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Use predict (assuming it takes np array and returns dict with 'emotion', 'valence', 'arousal')
            res = emotion_engine.predict(img_np)
            if res:
                # Rename 'Neutral' Result to 'Calm' if returned by engine
                if res['emotion'] == 'Neutral' or res['emotion'].lower() == 'neutral':
                    res['emotion'] = 'Calm'
                emotion_res = res
                st.success(f"Detected Emotion: **{res['emotion']}** (V:{res['valence']:.2f}, A:{res['arousal']:.2f})")
            else:
                st.error("No face detected or emotion recognition failed.")

    if emotion_res:
        st.subheader("Composition Details")
        st.write(f"Vibe: **{emotion_res['emotion']}**")
        
        # Musical quadrant mapping for display
        q_map = {1: "Q1: Joyful/Excited", 2: "Q2: Tense/Aggressive", 3: "Q3: Melancholic/Sad", 4: "Q4: Calm/Peaceful"}
        q = music_engine._map_valence_arousal_to_tag(emotion_res['valence'], emotion_res['arousal'])
        st.info(f"Musical Mapping: {q_map.get(q, 'Mixed')}")

        # Build Instrument Label for display
        inst_label = ", ".join(selected_instruments) if selected_instruments else "None"
        
        # 3. Generate Music
        st.header("3. Generate Music")
        if st.button("Generate Music Piece"):
            if not selected_instruments:
                st.error("Please select at least one instrument.")
            else:
                tmp_dir = Path(tempfile.mkdtemp())
                wav_path = tmp_dir / "generated.wav"
                
                with st.spinner(f"Cortex Transformer composing for {inst_label} ({duration}s)..."):
                    try:
                        # Pass list of selected_instruments to compose
                        audio_clip = comp_engine.compose(
                            emotion_res['emotion'], 
                            selected_instruments, 
                            duration_sec=duration
                        )
                        if audio_clip:
                            audio_clip.export(str(wav_path), format="wav")
                            st.audio(str(wav_path))
                            st.success(f"Cortex Generation Complete: {emotion_res['emotion']} mood ({duration}s)")
                        else:
                            st.error("Generation failed: No audio clips found for selected instruments.")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
    else:
        st.info("Provide an input above to see the emotion result.")

st.markdown("---")
st.caption("Backend powered by Original Pretrained EmoNet & Cortex Composition Engine. High-fidelity acoustic merging (up to 30s).")
