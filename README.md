# MOODWAAVE — Music Genre & Mood Intelligence System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?logo=python)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Platform Streamlit](https://img.shields.io/badge/platform-Streamlit-FF4B4B?logo=streamlit)

---

###  [Live Demo on Hugging Face](https://huggingface.co/spaces/imabdurrahman1/moodwaave)
*(Placeholder for Demo GIF)*

---

## Project Overview
**MOODWAAVE** is a sophisticated dual-output machine learning system designed to decode the DNA of music. By analyzing raw audio signals (.mp3/.wav), the system simultaneously classifies the track into one of ten musical genres and maps its emotional state onto the **Russell Circumplex Model**. Utilizing 113 handcrafted acoustic features, MOODWAAVE provides a language-agnostic analysis, meaning it can accurately interpret the "vibe" and genre of a song regardless of the language spoken in the lyrics, focusing purely on the underlying harmonic and rhythmic architecture.

## Features
- **Dual-Output Architecture:** Parallel prediction of Genre (Classification) and Mood (Regression).
- **Emotional Intelligence:** Predicts continuous Valence and Arousal scores to determine the specific mood quadrant.
- **Russell Circumplex Mapping:** Categorizes songs into four emotional states: **Happy**, **Angry**, **Sad**, and **Calm**.
- **113-Feature Extraction:** Deep signal analysis including MFCCs, Chroma, Spectral Contrast, Tonnetz, and Tempo.
- **Language Agnostic:** Works flawlessly across diverse cultures and languages by analyzing acoustic properties rather than linguistic content.
- **Interactive Dashboard:** A sleek Streamlit UI for real-time song analysis and visualization.

## System Architecture
```text
      ┌───────────────┐
      │  Raw Audio    │ (.mp3 / .wav)
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │ Feature Engine│ (librosa: 113 Acoustic Features)
      └───────┬───────┘
              ▼
      ┌───────┴───────┐
      │               │
      ▼               ▼
┌────────────┐  ┌────────────┐
│ Genre Model│  │ Mood Model │ (LightGBM / XGBoost)
│ (LightGBM) │  │ (Regressor)│
└─────┬──────┘  └─────┬──────┘
      │               │
      └───────┬───────┘
              ▼
      ┌───────────────┐
      │    RESULTS    │ (Genre Label + Valence/Arousal Score)
      └───────────────┘
```

## Performance Results
The system has been rigorously trained and validated on the **GTZAN** and **PMEmo** datasets.

| Metric | Score |
| :--- | :--- |
| **Genre Accuracy** | **72.50%** |
| **Genre Macro F1** | **0.7166** |
| **Mood Quadrant Accuracy** | **74.03%** |
| **Arousal R² Score** | **0.6365** |
| **Valence R² Score** | **0.3945** |

## Installation
To get MOODWAAVE running locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/imabdurrahman1/MOODWAAVE.git
   cd MOODWAAVE
   ```

2. **Set Up Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Launch the interactive Streamlit dashboard:
```bash
streamlit run src/app.py
```
Upload your favorite track and let MOODWAAVE analyze its soul!

## Dataset Credits
- **GTZAN Dataset:** Used for Genre Classification (Tzanetakis & Cook).
- **PMEmo Dataset:** Used for Mood/Emotional Regression (HuiZhangDB).

## Project Structure
```text
MOODWAAVE/
├── data/               # Dataset storage (CSV & Metadata)
├── models/             # Saved .pkl models & scalers
├── notebooks/          # EDA & Model Training experiments
├── outputs/            # Generated plots and performance charts
├── src/                # Core Application Logic
│   ├── app.py          # Streamlit Interface
│   ├── feature_extractor.py
│   └── pipeline.py     # Inference Pipeline
├── requirements.txt    # Project dependencies
└── README.md
```

## License
This project is licensed under the **MIT License** - see the LICENSE file for details.

## Hugging Face Configuration Block
---
title: MoodWaave
emoji: 🎧
colorFrom: yellow
colorTo: red
sdk: streamlit
app_file: src/app.py
pinned: false
---

## Author
**Abdur Rahman M**
*   **Institution:** VIT Vellore
*   **Email:** m.abdurrahman4040@gmail.com
*   **GitHub:** [github.com/imabdurrahman1](https://github.com/imabdurrahman1)
