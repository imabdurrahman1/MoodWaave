# DEPLOYMENT: Models hosted on Hugging Face Hub
# Update REPO_ID after uploading models to HF Hub
import streamlit as st
import os
from huggingface_hub import hf_hub_download
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.patches import Circle

# Add src to path to import pipeline
sys.path.append(os.path.join(os.getcwd(), 'src'))
from pipeline import predict_song, load_models

#  Model Download (For HF Spaces) 
def download_models():
    REPO_ID = "imabdurrahman1/MoodWaave-models"
    files = [
        "genre_classifier.pkl", "genre_scaler.pkl", "genre_encoder.pkl",
        "mood_regressor.pkl", "mood_scaler.pkl"
    ]
    
    os.makedirs("models", exist_ok=True)
    
    try:
        for file in files:
            local_path = os.path.join("models", file)
            if not os.path.exists(local_path):
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=file,
                    repo_type="model",
                    local_dir="models/"
                )
    except Exception as e:
        st.error(f"Error downloading models from HF Hub: {e}")

# Download models if not present
with st.spinner("Loading models for the first time..."):
    download_models()

# 1. Page Configuration
st.set_page_config(
    page_title="MOODWAVE | Genre & Mood Intelligence",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Design System - Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Syne:wght@700;800&display=swap');

    /* Global Styles */
    .stApp {
        background-color: #0A0A0A;
        color: #FFFFFF;
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stAppDeployButton {display:none;}

    /* Typography */
    h1, h2, h3 {
        font-family: 'Syne', sans-serif;
        text-transform: uppercase;
        letter-spacing: -1px;
    }

    .main-title {
        font-size: 4rem;
        background: linear-gradient(90deg, #F5A623, #FFD17E);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1;
    }

    .subtitle {
        color: #888888;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 500;
    }

    /* Cards */
    .mood-card {
        background-color: #141414;
        border: 1px solid #2A2A2A;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 24px rgba(0,0,0,0.4);
        height: 100%;
        transition: transform 0.3s ease;
    }

    .mood-card:hover {
        transform: translateY(-5px);
        border-color: #F5A623;
    }

    .card-label {
        color: #888888;
        text-transform: uppercase;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 1rem;
    }

    .card-value {
        font-family: 'Syne', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.1;
    }

    .card-subvalue {
        color: #888888;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* File Uploader Customization */
    .stFileUploader section {
        background-color: #141414 !important;
        border: 2px dashed #2A2A2A !important;
        border-radius: 16px !important;
        padding: 2rem !important;
        transition: all 0.3s ease;
    }

    .stFileUploader section:hover {
        border-color: #F5A623 !important;
    }

    /* Bars */
    .stat-bar-container {
        width: 100%;
        background-color: #2A2A2A;
        border-radius: 8px;
        height: 12px;
        margin-bottom: 0.5rem;
        overflow: hidden;
    }

    .stat-bar-fill {
        height: 100%;
        border-radius: 8px;
    }

    .valence-fill { background: linear-gradient(90deg, #F5A623, #FFD17E); }
    .arousal-fill { background: linear-gradient(90deg, #E05C5C, #FF8A8A); }

    .stat-label-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.2rem;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Sidebar Customization */
    [data-testid="stSidebar"] {
        background-color: #0A0A0A;
        border-right: 1px solid #2A2A2A;
    }

    .sidebar-text {
        color: #888888;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .stat-pill {
        background-color: #141414;
        border: 1px solid #2A2A2A;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.85rem;
        margin-bottom: 0.5rem;
        display: inline-block;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# 3. Model Loading (Cached)
@st.cache_resource
def get_models():
    return load_models()

models = get_models()

# 4. Helper Functions
def get_quadrant_color(quadrant):
    colors = {
        'Happy': '#F5A623',
        'Angry': '#E05C5C',
        'Sad': '#4A90E2',
        'Calm': '#50C878'
    }
    return colors.get(quadrant, '#F5A623')

def get_genre_icon(genre):
    icons = {
        'blues': '🎷', 'classical': '🎻', 'country': '🤠', 'disco': '💃',
        'hiphop': '🎤', 'jazz': '🎺', 'metal': '🤘', 'pop': '🍭',
        'reggae': '🏝️', 'rock': '🎸'
    }
    return icons.get(genre.lower(), '🎵')

def plot_circumplex(results):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='#141414')
    ax.set_facecolor('#0A0A0A')
    
    # Grid and dividers
    ax.axhline(0.5, color='#2A2A2A', linewidth=2, zorder=1)
    ax.axvline(0.5, color='#2A2A2A', linewidth=2, zorder=1)
    ax.grid(color='#2A2A2A', linestyle=':', alpha=0.5, zorder=0)
    
    # Quadrant labels (Ghost text)
    ax.text(0.75, 0.75, 'HAPPY', color='#444444', fontsize=24, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(0.25, 0.75, 'ANGRY', color='#444444', fontsize=24, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(0.25, 0.25, 'SAD', color='#444444', fontsize=24, fontweight='bold', ha='center', va='center', alpha=0.3)
    ax.text(0.75, 0.25, 'CALM', color='#444444', fontsize=24, fontweight='bold', ha='center', va='center', alpha=0.3)
    
    for res in results:
        v, a = res['valence'], res['arousal']
        q_color = get_quadrant_color(res['mood_quadrant'])
        
        # Main Point
        ax.scatter(v, a, s=150, color=q_color, edgecolors='white', linewidth=1.5, zorder=3,
                   path_effects=[path_effects.withSimplePatchShadow(offset=(2, -2), shadow_rgbFace='black', alpha=0.5)])
        
        # Annotation (for small batches)
        if len(results) <= 5:
            ax.annotate(res['filename'], (v, a), xytext=(0, 10), 
                        textcoords='offset points', color='white', weight='bold', 
                        ha='center', fontsize=8, bbox=dict(boxstyle="round,pad=0.2", fc="#141414", ec="#2A2A2A", alpha=0.8))

    # Axis styling
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('VALENCE', color='#888888', fontweight='bold', labelpad=10)
    ax.set_ylabel('AROUSAL', color='#888888', fontweight='bold', labelpad=10)
    
    ax.tick_params(colors='#888888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2A2A2A')
    
    return fig

# 5. Sidebar Information
with st.sidebar:
    st.markdown("<h2 style='color: #F5A623;'>SYSTEM SPECS</h2>", unsafe_allow_html=True)
    st.markdown("""
        <div class='stat-pill'>Genre Accuracy: 72.5%</div>
        <div class='stat-pill'>Mood Accuracy: 74.0%</div>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3 style='margin-top:2rem;'>GLOSSARY</h3>", unsafe_allow_html=True)
    st.markdown("""
        <p class='sidebar-text'>
        <b>Valence:</b> Measures the emotional positivity. Higher values indicate 'Happy' or 'Calm' (positive), lower values indicate 'Sad' or 'Angry' (negative).
        <br><br>
        <b>Arousal:</b> Measures intensity or energy level. Higher values indicate high energy ('Angry' or 'Happy'), lower values indicate low energy ('Sad' or 'Calm').
        </p>
    """, unsafe_allow_html=True)

# 6. Main Layout
st.markdown("<h1 class='main-title'>MOODWAVE</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>GENRE & MOOD INTELLIGENCE</p>", unsafe_allow_html=True)

uploaded_files = st.file_uploader("Drop audio files (.mp3, .wav) to analyze", type=['mp3', 'wav'], accept_multiple_files=True)

if uploaded_files:
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    os.makedirs("outputs", exist_ok=True)
    
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Analyzing {uploaded_file.name} ({i+1}/{len(uploaded_files)})")
        
        # Save temp file
        temp_path = os.path.join("outputs", f"temp_{int(time.time())}_{uploaded_file.name}")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            res = predict_song(temp_path, models=models)
            if "error" not in res:
                results.append(res)
            else:
                st.error(f"Failed to analyze {uploaded_file.name}: {res['error']}")
        except Exception as e:
            st.error(f"Error analyzing {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text("Analysis Complete.")
    time.sleep(1)
    status_text.empty()
    progress_bar.empty()

    if results:
        #  Batch View 
        if len(results) == 1:
            result = results[0]
            # 3-Column Results for Single File
            col1, col2, col3 = st.columns(3)
            
            col1.markdown(f"""
                <div class='mood-card'>
                    <div class='card-label'>Predicted Genre</div>
                    <div style='font-size: 3rem;'>{get_genre_icon(result['predicted_genre'])}</div>
                    <div class='card-value' style='color: #F5A623;'>{result['predicted_genre'].upper()}</div>
                </div>
            """, unsafe_allow_html=True)
            
            q_color = get_quadrant_color(result['mood_quadrant'])
            col2.markdown(f"""
                <div class='mood-card'>
                    <div class='card-label'>Mood Profile</div>
                    <div class='card-value' style='color: {q_color};'>{result['mood_nuanced'].upper()}</div>
                    <div class='card-subvalue'>
                        Classification: {result['mood_quadrant']}
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
            v_perc = result['valence'] * 100
            a_perc = result['arousal'] * 100
            col3.markdown(f"""
                <div class='mood-card'>
                    <div class='card-label'>Spectral Metrics</div>
                    <div class='stat-label-row'>
                        <span>VALENCE</span>
                        <span style='color: #F5A623;'>{result['valence']:.3f}</span>
                    </div>
                    <div class='stat-bar-container'>
                        <div class='stat-bar-fill valence-fill' style='width: {v_perc}%;'></div>
                    </div>
                    <div class='stat-label-row' style='margin-top: 1.5rem;'>
                        <span>AROUSAL</span>
                        <span style='color: #E05C5C;'>{result['arousal']:.3f}</span>
                    </div>
                    <div class='stat-bar-container'>
                        <div class='stat-bar-fill arousal-fill' style='width: {a_perc}%;'></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Table View for Multiple Files
            df_results = pd.DataFrame(results)
            st.markdown("### BATCH RESULTS")
            st.dataframe(
                df_results[['filename', 'predicted_genre', 'mood_nuanced', 'mood_quadrant', 'valence', 'arousal']],
                use_container_width=True,
                hide_index=True
            )
            
            # Export CSV
            csv = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 DOWNLOAD RESULTS AS CSV",
                data=csv,
                file_name="moodwave_batch_results.csv",
                mime="text/csv",
            )

        # Unified Circumplex Plot
        st.markdown("<h3 style='margin-top: 3rem; color: #888888;'>EMOTIONAL CIRCUMPLEX</h3>", unsafe_allow_html=True)
        fig = plot_circumplex(results)
        st.pyplot(fig)

else:
    # Empty state / Welcome
    st.markdown("""
        <div style='text-align: center; margin-top: 5rem; padding: 4rem; background-color: #141414; border-radius: 16px; border: 1px solid #2A2A2A;'>
            <div style='font-size: 5rem; margin-bottom: 1rem;'>🌑</div>
            <h2 style='color: #888888;'>READY TO SYNC</h2>
            <p style='color: #444444;'>Awaiting audio input to begin frequency decomposition.</p>
        </div>
    """, unsafe_allow_html=True)
