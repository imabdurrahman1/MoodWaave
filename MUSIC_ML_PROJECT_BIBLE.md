# 🎵 MUSIC ML PROJECT BIBLE
## Dual-Output Prediction of Genre and Emotional Mood
### Using Optimized Ensemble Learning

---

> **READ THIS FIRST**
> This is your single source of truth. Work phase by phase. Never skip a phase gate check. Always run Gemini CLI from inside your `music_ml_project/` folder. Never stack unrun code blocks.

---

## PROJECT OVERVIEW

**Goal:** Build a system that takes any audio file and predicts:
1. Its **Genre** (e.g., Jazz, Metal, Classical) — Classification
2. Its **Mood** as Valence + Arousal scores — Regression → mapped to quadrant (Happy/Angry/Sad/Calm)

**Datasets:**
- GTZAN → Genre labels (1000 .wav files, 10 genres, 100 each)
- PMEmo → Mood labels (794 songs, continuous Valence + Arousal scores)

**Stack:** Python, librosa, scikit-learn, XGBoost, LightGBM — CPU only

**Architecture:**
```
Raw Audio (.wav/.mp3)
        ↓
Feature Extraction (librosa)
[MFCCs, Chroma, Spectral, Tempo, ZCR, RMS]
        ↓
        ├──→ Genre Classifier (XGBoost/LightGBM)
        │           ↓ Genre Label
        │
        └──→ Mood Regressor (MultiOutputRegressor)
                    ↓
            Valence Score + Arousal Score
                    ↓
            Quadrant Label (post-hoc)
```

---

## KNOWN DATA QUIRKS (read before starting)

| Issue | Detail |
|---|---|
| PMEmo column names | Are `Arousal(mean)` and `Valence(mean)` — NOT `Arousal_mean` / `Valence_mean` |
| GTZAN artist bleed | Some artists appear across splits — always use StratifiedShuffleSplit seed=42 |
| GTZAN corrupt file | `jazz.00054.wav` is known to be corrupted — wrap all loading in try/except |
| PMEmo features | Pre-extracted features already available — no need to run librosa on PMEmo audio |

---

## FOLDER STRUCTURE

```
music_ml_project/
│
├── data/
│   ├── gtzan/
│   │   └── genres_original/
│   │       ├── blues/        (100 .wav)
│   │       ├── classical/    (100 .wav)
│   │       ├── country/      (100 .wav)
│   │       ├── disco/        (100 .wav)
│   │       ├── hiphop/       (100 .wav)
│   │       ├── jazz/         (100 .wav)
│   │       ├── metal/        (100 .wav)
│   │       ├── pop/          (100 .wav)
│   │       ├── reggae/       (100 .wav)
│   │       └── rock/         (100 .wav)
│   ├── pmemo/
│   │   ├── features/         (pre-extracted CSVs)
│   │   └── annotations/
│   │       └── static_annotations.csv
│   ├── test_songs/           (your own .mp3/.wav for testing)
│   ├── gtzan_features.csv    (generated in Phase 2)
│   └── pmemo_ready.csv       (generated in Phase 4)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_genre_model.ipynb
│   ├── 04_mood_model.ipynb
│   └── 05_pipeline.ipynb
│
├── src/
│   ├── feature_extractor.py
│   ├── genre_classifier.py
│   ├── mood_regressor.py
│   └── pipeline.py
│
├── models/
│   ├── genre_classifier.pkl
│   ├── genre_scaler.pkl
│   ├── genre_encoder.pkl
│   ├── mood_regressor.pkl
│   └── mood_scaler.pkl
│
├── outputs/                  (all plots and charts saved here)
└── requirements.txt
```

---

## DATASET DOWNLOAD LINKS

| Dataset | URL |
|---|---|
| GTZAN | https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification |
| PMEmo | https://github.com/HuiZhangDB/PMEmo |

---

## GEMINI CLI — GLOBAL RULES

> Follow these every single session without exception.

1. **Always run Gemini CLI from inside `music_ml_project/`**
2. **Paste the session context header once at the start of every new session**
3. **Run code before asking the next question — never stack unrun blocks**
4. **When something breaks:** paste exact error + say "don't rewrite everything, just fix the broken part"
5. **End every prompt with an action word:** "Run it", "Save it", "Show output"
6. **Never let Gemini use `os.system()` for audio — librosa only**
7. **After Phase 3 and 4, tell Gemini your exact accuracy numbers before tuning**

---

## SESSION CONTEXT HEADER
### Paste this ONCE at the start of every new Gemini CLI session:

```
Project context: dual-output music ML system on Windows, CPU only.
Genre classifier trained on GTZAN dataset at data/gtzan/genres_original/ (1000 .wav, 10 genres).
Mood regressor trained on PMEmo at data/pmemo/ — PMEmo column names are Arousal(mean) and Valence(mean).
Stack: Python, librosa, scikit-learn, xgboost, lightgbm, joblib.
Outputs → outputs/ folder. Trained models → models/ folder.
All audio loaded with librosa (sr=22050, mono=True, duration=30).
Feature extraction in src/feature_extractor.py saves to data/gtzan_features.csv.
Don't explain what you're doing. Just execute and show output.
```

---

## DEBUGGING PROMPTS
### Use these whenever something breaks:

**Import error:**
```
I got this import error: [PASTE ERROR].
I am on Windows, CPU only. Check my requirements.txt and fix the import.
Don't rewrite the whole file, just fix the broken line.
```

**File not found error:**
```
I got FileNotFoundError: [PASTE ERROR].
Look at my actual folder structure and find the correct path.
Fix only the path, nothing else.
```

**Shape/dimension error:**
```
I got this error: [PASTE ERROR].
This is likely a feature aggregation issue.
Check how MFCCs are being collapsed from 2D to 1D and fix it.
```

**Model not converging / bad accuracy:**
```
My [MODEL NAME] is giving [X]% accuracy which seems wrong.
Check: 1) Was scaler fit on test data accidentally?
2) Is label encoding consistent between train and test?
3) Is there data leakage in the split?
Print shapes of X_train, X_test, y_train, y_test to verify.
```

---

---

# PHASE 0 — ENVIRONMENT & DATA ACQUISITION
## Status: ✅ COMPLETE

**Confirmed:**
- 100 files × 10 genres = 1000 GTZAN .wav files ✅
- PMEmo static_annotations.csv loads with Arousal(mean) and Valence(mean) columns ✅

**Phase 0 Gate:** ✅ Already passed — proceed to Phase 1.

---

---

# PHASE 1 — EXPLORATORY DATA ANALYSIS
## Duration: 3–4 days
## Goal: Understand your data visually before touching models

---

### What You're Looking For:
- Are all 10 genres represented equally?
- Can you visually see differences between genre spectrograms?
- Is the PMEmo mood data spread across all 4 quadrants or skewed?
- Are there obvious outliers or corrupted files?

---

### CLI PROMPT 1A — Genre Distribution
```
Look at data/gtzan/genres_original/. Count .wav files per genre subfolder.
Plot a bar chart of file count per genre with genre names on x-axis.
Save to outputs/genre_distribution.png. Show output.
```

---

### CLI PROMPT 1B — Waveform Visualization
```
Load one .wav file from each genre folder in data/gtzan/genres_original/
using librosa (sr=22050, mono=True, duration=30).
Plot all 10 waveforms in a 2x5 grid. Title each subplot with genre name.
Save to outputs/waveforms.png
```

---

### CLI PROMPT 1C — Mel Spectrograms
```
Load one .wav from each of the 10 genre folders using librosa.
Compute Mel Spectrogram for each using librosa.feature.melspectrogram.
Plot all 10 in a 2x5 grid using librosa.display.specshow with colorbar.
Title each with genre name. Save to outputs/spectrograms.png
```

---

### CLI PROMPT 1D — MFCC Heatmap
```
Load data/gtzan/genres_original/blues/blues.00000.wav using librosa
(sr=22050, mono=True, duration=30).
Extract 13 MFCCs. Plot as heatmap over time using seaborn.
X-axis = time frames, Y-axis = MFCC coefficient number.
Add colorbar. Save to outputs/mfcc_heatmap.png
```

---

### CLI PROMPT 1E — PMEmo Circumplex Plot
```
Load data/pmemo/annotations/static_annotations.csv.
Columns are named Arousal(mean) and Valence(mean).
Plot scatter of Valence(mean) on x-axis vs Arousal(mean) on y-axis.
Add KDE density overlay.
Draw vertical line at x=0.5 and horizontal line at y=0.5.
Label 4 quadrants: Happy (top-right), Angry (top-left),
Sad (bottom-left), Calm (bottom-right).
Count songs per quadrant and display counts in plot title.
Save to outputs/circumplex.png
```

---

### CLI PROMPT 1F — Tempo Distribution
```
Iterate through all .wav files in data/gtzan/genres_original/.
For each file extract tempo using librosa.beat.beat_track
(sr=22050, mono=True, duration=30).
Wrap each file load in try/except — skip and log failures.
Build a DataFrame with columns: genre, tempo.
Plot a boxplot of tempo grouped by genre using seaborn.
Save to outputs/tempo_by_genre.png
Note: this will take several minutes on CPU — add tqdm progress bar.
```

---

### PHASE 1 GATE — Before proceeding to Phase 2, answer these:
- [ ] All 6 plots generated without errors?
- [ ] Metal and Classical spectrograms look visually different?
- [ ] PMEmo data — which quadrant has the most songs?
- [ ] Any genres with fewer than 100 files? (indicates missing files)
- [ ] Did any files fail to load in Prompt 1F?

---

---

# PHASE 2 — FEATURE ENGINEERING
## Duration: 4–5 days
## Goal: Convert raw audio into a structured numerical dataset

---

### Features You Will Extract Per Song:

| Feature | What It Captures | Output Dimensions |
|---|---|---|
| MFCC (13 coefficients) | Timbral texture | 13 mean + 13 std = 26 |
| MFCC Delta | Rate of change of timbre | 13 mean + 13 std = 26 |
| Chroma STFT | Harmonic/pitch content | 12 mean + 12 std = 24 |
| Spectral Centroid | Brightness of sound | 1 mean + 1 std = 2 |
| Spectral Rolloff | High frequency energy | 1 mean + 1 std = 2 |
| Spectral Bandwidth | Frequency spread | 1 mean + 1 std = 2 |
| Zero Crossing Rate | Noisiness / percussion | 1 mean + 1 std = 2 |
| RMS Energy | Loudness / intensity | 1 mean + 1 std = 2 |
| Tempo | Speed / rhythmic energy | 1 scalar |
| **TOTAL** | | **~87 features per song** |

---

### CLI PROMPT 2A — Build Feature Extractor Script
```
Build src/feature_extractor.py that defines a function extract_features(file_path)
which loads audio with librosa (sr=22050, mono=True, duration=30) and returns
a flat dictionary with these exact keys:

mfcc_mean_1 to mfcc_mean_13 (mean of each MFCC over time)
mfcc_std_1 to mfcc_std_13 (std of each MFCC over time)
delta_mfcc_mean_1 to delta_mfcc_mean_13
delta_mfcc_std_1 to delta_mfcc_std_13
chroma_mean_1 to chroma_mean_12
chroma_std_1 to chroma_std_12
spectral_centroid_mean, spectral_centroid_std
spectral_rolloff_mean, spectral_rolloff_std
spectral_bandwidth_mean, spectral_bandwidth_std
zcr_mean, zcr_std
rms_mean, rms_std
tempo (single float)

All time-series features must be collapsed to scalars using np.mean and np.std over axis=1.
Wrap everything in try/except — return None if file fails.

Add a main block that:
1. Iterates all .wav files in data/gtzan/genres_original/
2. Extracts features for each file
3. Adds a label column = genre folder name
4. Skips None results and logs failed files
5. Saves final DataFrame to data/gtzan_features.csv
6. Uses tqdm for progress bar
7. Prints total rows saved and any failed files at the end

Run it now.
```

---

### CLI PROMPT 2B — Validate Feature Quality
```
Load data/gtzan_features.csv.
Run these quality checks and print results:

1. Shape of DataFrame
2. Number of null values per column — flag any column with nulls
3. Number of duplicate rows
4. Class distribution — count of rows per label
5. Run PCA to 2 components on all feature columns (exclude label).
   Scale with StandardScaler first.
   Plot scatter colored by label with legend.
   Save to outputs/pca_genres.png
6. Compute correlation matrix. List all feature pairs with correlation > 0.95
7. Print final verdict: READY or NEEDS FIXING with reason

Do not proceed to modeling if there are nulls or fewer than 900 rows total.
```

---

### PHASE 2 GATE — Before proceeding to Phase 3:
- [ ] data/gtzan_features.csv exists with ~1000 rows and ~88 columns?
- [ ] Zero null values?
- [ ] PCA plot shows some genre clustering (even rough)?
- [ ] No genre has fewer than 90 rows?

---

---

# PHASE 3 — GENRE CLASSIFICATION MODEL
## Duration: 5–6 days
## Goal: Train and tune a 10-class genre classifier

---

### CLI PROMPT 3A — Train and Compare 4 Models
```
Load data/gtzan_features.csv.
Separate features X (all columns except label) and target y (label column).

Preprocessing:
- Encode y with LabelEncoder — save mapping to a variable
- Split with StratifiedShuffleSplit: 80/20, random_state=42
- Scale X with StandardScaler fit ONLY on X_train, then transform both

Train these 4 models:
1. SVM: kernel=rbf, C=10, gamma=scale
2. RandomForest: n_estimators=200, max_depth=None, min_samples_split=5, random_state=42
3. XGBoost: n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8, random_state=42
4. LightGBM: n_estimators=200, learning_rate=0.1, num_leaves=31, random_state=42

For each model print:
- Test accuracy
- Macro F1 score
- Training time in seconds
- Full classification report

Plot confusion matrix heatmap for each model with genre names on axes.
Save each to outputs/confusion_matrix_{model_name}.png

Print a final summary DataFrame comparing all 4 models side by side.
```

---

### CLI PROMPT 3B — Hyperparameter Tuning
#### (Run after 3A — replace [BEST MODEL] with your actual result)
```
My best genre model from comparison was [BEST MODEL NAME].
Baseline accuracy was [X]% and macro F1 was [Y].

Run RandomizedSearchCV on this model with:
- cv=5 stratified
- n_iter=30
- scoring=f1_macro
- n_jobs=-1
- random_state=42

Define an appropriate parameter grid for this model type.

After fitting:
1. Print best parameters
2. Print best CV F1 score
3. Retrain with best params on full training set
4. Evaluate on test set — print accuracy and macro F1
5. Compare to baseline — did it improve?

Then plot a learning curve: training score vs validation score
as training sample size increases.
Save to outputs/learning_curve_genre.png

Save final model to models/genre_classifier.pkl
Save scaler to models/genre_scaler.pkl
Save encoder to models/genre_encoder.pkl
Use joblib for all saves.
```

---

### CLI PROMPT 3C — Feature Importance Analysis
```
Load the saved genre model from models/genre_classifier.pkl.
Load data/gtzan_features.csv for feature names.

If the model has feature_importances_ attribute:
- Plot top 20 most important features as horizontal bar chart
- Save to outputs/genre_feature_importance.png

Also: look at the confusion matrix from 3A.
Which 3 genre pairs are most confused with each other?
Print those pairs and their confusion counts.
```

---

### PHASE 3 GATE — Before proceeding to Phase 4:
- [ ] All 3 model files saved in models/ folder?
- [ ] Best model accuracy above 70%? (below this means feature extraction went wrong)
- [ ] Can you name the 2 genres your model confuses most?
- [ ] Learning curve shows convergence (not still rising at max samples)?

---

---

# PHASE 4 — MOOD REGRESSION MODEL
## Duration: 5–6 days
## Goal: Predict continuous Valence and Arousal scores from audio features

---

> ⚠️ CRITICAL: PMEmo column names are `Arousal(mean)` and `Valence(mean)` — use these exactly in all prompts.

---

### CLI PROMPT 4A — Prepare PMEmo Data
```
Load data/pmemo/annotations/static_annotations.csv.
The column names are exactly: musicId, Arousal(mean), Valence(mean)

Also load the pre-extracted features from data/pmemo/features/.
Check what files are in that folder and load the appropriate CSV.
Merge features with annotations on musicId.

After merging:
1. Print shape before and after merge — how many rows were lost?
2. Check for nulls in all columns
3. Create target arrays: y_valence = Valence(mean), y_arousal = Arousal(mean)
4. Plot distribution of both targets as histograms with KDE
5. Create quadrant_label column:
   - Valence(mean) >= 0.5 AND Arousal(mean) >= 0.5 → Happy
   - Valence(mean) < 0.5 AND Arousal(mean) >= 0.5 → Angry
   - Valence(mean) < 0.5 AND Arousal(mean) < 0.5 → Sad
   - Valence(mean) >= 0.5 AND Arousal(mean) < 0.5 → Calm
6. Print count per quadrant
7. Save cleaned merged data to data/pmemo_ready.csv
8. Print final shape and head(5)
```

---

### CLI PROMPT 4B — Train Mood Regression Models
```
Load data/pmemo_ready.csv.
Identify feature columns (exclude musicId, Arousal(mean), Valence(mean), quadrant_label).
Set X = feature columns, y = [Valence(mean), Arousal(mean)] stacked as 2-column array.

Preprocessing:
- Split 80/20 random_state=42
- Scale X with StandardScaler fit on X_train only
- Do NOT scale y — targets are already between 0 and 1

Train these using MultiOutputRegressor wrapper:
1. XGBoost Regressor: n_estimators=200, learning_rate=0.1, max_depth=6
2. LightGBM Regressor: n_estimators=200, learning_rate=0.1
3. Random Forest Regressor: n_estimators=200, random_state=42

For each model print:
- MAE for Valence separately
- MAE for Arousal separately
- R² for Valence separately
- R² for Arousal separately

For the best model also:
- Plot predicted vs actual Valence as scatter
- Plot predicted vs actual Arousal as scatter
- Plot predicted (V,A) points vs actual (V,A) points on circumplex
  with quadrant lines — actual in grey, predicted in color
- Save all plots to outputs/

Then derive quadrant labels from predicted V and A.
Compare to actual quadrant labels using classification_report.
This gives an interpretable accuracy number.

Save best model to models/mood_regressor.pkl
Save its scaler to models/mood_scaler.pkl
```

---

### CLI PROMPT 4C — Mood Error Analysis
```
Load models/mood_regressor.pkl and models/mood_scaler.pkl.
Load data/pmemo_ready.csv.

For the test set predictions:
1. Which quadrant has the highest average prediction error?
2. Are errors symmetric or does the model skew toward one region of the circumplex?
3. Is Valence or Arousal harder to predict? Print R² for both.
4. Plot error distribution (predicted - actual) for both Valence and Arousal
   as histograms. Save to outputs/mood_error_distribution.png

Print a written summary of findings.
```

---

### PHASE 4 GATE — Before proceeding to Phase 5:
- [ ] models/mood_regressor.pkl and models/mood_scaler.pkl saved?
- [ ] Valence MAE below 0.15? (above this means features are not predictive)
- [ ] R² above 0.3 for at least one target?
- [ ] Quadrant accuracy above 50%? (random baseline is 25%)

---

---

# PHASE 5 — UNIFIED INFERENCE PIPELINE
## Duration: 3 days
## Goal: Connect both models into a single prediction function

---

### CLI PROMPT 5A — Build Pipeline Script
```
Build src/pipeline.py using all saved models in models/ folder.

Import extract_features from src/feature_extractor.py

Define function predict_song(file_path) that:
1. Loads audio with librosa (sr=22050, mono=True, duration=30)
2. Calls extract_features(file_path) to get feature dict
3. Converts to DataFrame — must match exact column order used during training
4. Loads models/genre_classifier.pkl, models/genre_scaler.pkl, models/genre_encoder.pkl
5. Scales features with genre_scaler, predicts genre, decodes with genre_encoder
6. Loads models/mood_regressor.pkl, models/mood_scaler.pkl
7. Scales features with mood_scaler, predicts [valence, arousal]
8. Derives quadrant label from V/A values using same 0.5 threshold logic
9. Returns dict: {filename, predicted_genre, valence, arousal, mood_quadrant}
10. Prints result in clean formatted output

Add CLI support: if script run directly with a file path as argument,
call predict_song(sys.argv[1]) and print result.

Test it on one file from data/test_songs/ and show output.
```

---

### CLI PROMPT 5B — Pipeline Stress Test
```
Run predict_song() on 5 different songs from data/test_songs/.
For each song plot a result card showing:
- Song filename
- Predicted genre
- Valence and Arousal as a point plotted on the circumplex diagram
- Mood quadrant label highlighted

Save result cards to outputs/prediction_cards.png

Also test these edge cases and print what happens:
1. A very short file (under 5 seconds)
2. A silent or near-silent file
3. A non-audio file passed as input
The pipeline should not crash — it should return a clear error message.
```

---

### PHASE 5 GATE — Before proceeding to Phase 6:
- [ ] predict_song() runs without errors on any valid audio file?
- [ ] Output contains all 5 keys: filename, genre, valence, arousal, mood_quadrant?
- [ ] Edge cases handled gracefully without crashing?

---

---

# PHASE 6 — FINAL EVALUATION & DOCUMENTATION
## Duration: 3–4 days
## Goal: Consolidate results, generate final report

---

### CLI PROMPT 6A — Genre Final Report
```
Load models/genre_classifier.pkl, genre_scaler.pkl, genre_encoder.pkl.
Load data/gtzan_features.csv.
Re-run predictions on the test set.

Generate:
1. Confusion matrix with percentage labels (not raw counts)
   Save to outputs/final_confusion_matrix.png
2. Per-class F1 bar chart — sorted from best to worst performing genre
   Save to outputs/genre_f1_by_class.png
3. Feature importance: top 20 features as horizontal bar chart
   Save to outputs/final_feature_importance.png
4. Print interpretation: which 2 genre pairs are most confused
   and give an acoustic reason why
```

---

### CLI PROMPT 6B — Mood Final Report
```
Load models/mood_regressor.pkl and mood_scaler.pkl.
Load data/pmemo_ready.csv.
Re-run predictions on test set.

Generate:
1. Circumplex plot: actual points in grey, predicted in blue,
   thin lines connecting each pair to visualize error magnitude
   Save to outputs/final_circumplex_error.png
2. Which quadrant has lowest prediction error?
3. Bar chart: MAE per quadrant
   Save to outputs/mae_by_quadrant.png
4. Print: is Valence or Arousal harder to predict and why
```

---

### CLI PROMPT 6C — Master Summary Table
```
Generate a single clean summary DataFrame with these rows and values
based on all results so far:

Metric                  | Value
------------------------|-------
Genre Model             | [best model name]
Genre Accuracy          | [X]%
Genre Macro F1          | [X]
Valence MAE             | [X]
Arousal MAE             | [X]
Valence R²              | [X]
Arousal R²              | [X]
Mood Quadrant Accuracy  | [X]%
Total Songs (Genre)     | 1000
Total Songs (Mood)      | ~794

Print and save as outputs/final_summary.csv
Also save as a styled HTML table to outputs/final_summary.html
```

---

### CLI PROMPT 6D — Export Notebooks to PDF
```
Convert all notebooks in notebooks/ folder to PDF format.
Use jupyter nbconvert --to pdf for each.
Save converted PDFs to outputs/ folder.
If PDF conversion fails due to latex, use --to html instead.
```

---

### PHASE 6 GATE — Project Complete When:
- [ ] outputs/final_confusion_matrix.png exists
- [ ] outputs/final_circumplex_error.png exists
- [ ] outputs/final_summary.csv exists
- [ ] All 5 model files exist in models/ folder
- [ ] predict_song() works on any new audio file

---

---

# QUICK REFERENCE — ALL OUTPUTS

| File | Generated In |
|---|---|
| outputs/genre_distribution.png | Phase 1 |
| outputs/waveforms.png | Phase 1 |
| outputs/spectrograms.png | Phase 1 |
| outputs/mfcc_heatmap.png | Phase 1 |
| outputs/circumplex.png | Phase 1 |
| outputs/tempo_by_genre.png | Phase 1 |
| outputs/pca_genres.png | Phase 2 |
| outputs/confusion_matrix_*.png | Phase 3 |
| outputs/learning_curve_genre.png | Phase 3 |
| outputs/genre_feature_importance.png | Phase 3 |
| outputs/mood_error_distribution.png | Phase 4 |
| outputs/prediction_cards.png | Phase 5 |
| outputs/final_confusion_matrix.png | Phase 6 |
| outputs/final_circumplex_error.png | Phase 6 |
| outputs/mae_by_quadrant.png | Phase 6 |
| outputs/final_summary.csv | Phase 6 |
| data/gtzan_features.csv | Phase 2 |
| data/pmemo_ready.csv | Phase 4 |
| models/genre_classifier.pkl | Phase 3 |
| models/genre_scaler.pkl | Phase 3 |
| models/genre_encoder.pkl | Phase 3 |
| models/mood_regressor.pkl | Phase 4 |
| models/mood_scaler.pkl | Phase 4 |

---

# WHEN TO COME BACK TO CLAUDE

You don't need Claude for every step. Come back when:

1. **A phase gate metric is far below target** (e.g., genre accuracy below 60%)
2. **Gemini keeps giving you the same broken code** after 2 fix attempts
3. **You need to make an architectural decision** not covered here
4. **After Phase 3** — tell Claude your actual accuracy numbers and ask if tuning is worth it
5. **After Phase 4** — share your Valence/Arousal R² scores for interpretation advice

When starting a new Claude conversation, paste this at the top:

> *"I am building a dual-output music ML project (genre + mood). Genre classifier on GTZAN 1000 tracks, mood regressor on PMEmo with continuous Valence/Arousal targets. CPU only, Windows, using librosa/sklearn/xgboost/lightgbm. I am currently on Phase [X]. My current results are: [paste metrics]. My issue is: [describe problem]."*

---

*Project Bible v1.0 — Generated as part of guided ML project mentorship*
*Last Updated: Phase 0 Complete*
