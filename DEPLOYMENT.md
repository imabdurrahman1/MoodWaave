#  MOODWAAVE Deployment Guide

This guide will walk you through deploying your **Music Genre & Mood Intelligence System** to the web using **GitHub** and **Hugging Face Spaces**.

---

##  Phase 1: Push to GitHub

GitHub will host your source code and trigger the deployment.

1.  **Initialize Git (if not already done):**
    ```powershell
    git init
    git add .
    git commit -m "Initial commit: Ready for deployment"
    ```

2.  **Connect to your GitHub Repository:**
    *(Replace the URL with your actual GitHub repo link)*
    ```powershell
    git remote add origin https://github.com/USERNAME/MoodWaave.git
    git branch -M main
    ```

3.  **Push your code:**
    ```powershell
    git push -u origin main
    ```

---

##  Phase 2: Host Models on Hugging Face Hub

Since model files (`.pkl`) are too large for GitHub, we host them on the Hugging Face Model Hub.

1.  **Get your Access Token:**
    *   Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
    *   Create a **New Token** with `Write` access. Copy it.

2.  **Set the Environment Variable (Windows):**
    ```powershell
    $env:HF_TOKEN = "your_actual_token_here"
    ```

3.  **Run the Upload Script:**
    *   Open `upload_models_to_hf.py` and replace `YOUR_HF_USERNAME` with your actual username.
    *   Run: `python upload_models_to_hf.py`

---

##  Phase 3: Create Hugging Face Space

This is where your Streamlit app will live.

1.  **Create a New Space:**
    *   Go to [huggingface.co/new-space](https://huggingface.co/new-space).
    *   **Space Name:** `moodwaave` (or your choice).
    *   **SDK:** Select **Streamlit**.
    *   **Repository:** Choose **Connect to GitHub** and select your `MoodWaave` repository.
    *   **Main File:** Set this to `src/app.py`.

2.  **Add your Secret Token (CRITICAL):**
    *   Once the Space is created, go to the **Settings** tab of your Space.
    *   Scroll to **Variables and Secrets**.
    *   Click **New Secret**.
    *   **Name:** `HF_TOKEN`
    *   **Value:** Paste your Hugging Face Access Token here.

---

##  Phase 4: Access and Update

1.  **Get your Public URL:**
    *   Your app will be live at: `https://huggingface.co/spaces/YOUR_USERNAME/moodwaave`.
    *   It may take 2–3 minutes to build for the first time.

2.  **Updating the App:**
    *   Any time you run `git push`, Hugging Face will detect the change and **automatically redeploy** your app.

---

##  Common Errors & Fixes

| Error | Solution |
| :--- | :--- |
| **ModuleNotFoundError** | Add the missing library to `requirements.txt` and push to GitHub. |
| **Model Download Fails** | Check `src/app.py`. Ensure `REPO_ID` matches your model repo name exactly. |
| **App Crashes on Audio Load** | Ensure `soundfile` and `audioread` are listed in `requirements.txt`. |
| **Authentication Error** | Ensure your `HF_TOKEN` secret in the Space settings is correct and has `Read` or `Write` access. |

---
*Created by Abdur Rahman M*
